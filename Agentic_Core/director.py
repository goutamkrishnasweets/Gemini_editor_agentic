"""
Agentic_Core / director.py
==========================
Autonomous Director & ReAct Orchestration Loop for Gemini Editor.
Directs ingestion, perception rendering, audit, and publishing using Gemini 2.5/1.5 Flash.
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger("AgenticCore.Director")

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from google import genai
from google.genai import types

from .tool_adapters import get_agent_tools, TOOL_DISPATCH_MAP

try:
    from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router, GeminiGovernor
    _HAS_GOVERNOR = True
except ImportError:
    gemini_router = None
    GeminiGovernor = None
    _HAS_GOVERNOR = False

# RPM guard: Minimum delay between director turns to prevent hitting 15 RPM burst limit
_MIN_TURN_INTERVAL_SEC = 1.0

# ── Hard limits that make a "frozen" Director impossible ─────────────────────
# Previously the API-error handler was `while True: ... continue` with no cap, so
# any deterministic 400/404 (e.g. thought_signature) retried forever (15s sleeps).
_MAX_API_RETRIES = int(os.getenv("DIRECTOR_MAX_API_RETRIES", "3"))        # transient errors per turn
_DEADLINE_SEC = float(os.getenv("DIRECTOR_DEADLINE_SEC", "900"))           # wall-clock budget for whole goal
_HTTP_TIMEOUT_SEC = float(os.getenv("DIRECTOR_HTTP_TIMEOUT_SEC", "90"))    # per Gemini HTTP request
_MAX_IDENTICAL_CALLS = 2                                                   # same tool+args max executions per goal
_PUBLISH_WORDS = ("publish", "post ", "upload", "broadcast", "share to", "go live")


def classify_api_error(err: Exception) -> str:
    """
    Classify a Gemini SDK error. Returns one of:
      auth | signature | quota | server | model_gone | model_invalid | fatal
    Only 'quota' and 'server' are retried (bounded). Everything else fails fast
    (or recovers once) instead of looping forever.
    """
    s = str(err).lower()
    is_quota = any(k in s for k in ("429", "quota", "resource_exhausted", "rate_limit", "rate limit"))
    if any(k in s for k in ("api key", "api_key_invalid", "unauthorized", "permission_denied")) and not is_quota:
        return "auth"
    if "thought_signature" in s or "thought signature" in s:
        return "signature"
    if is_quota:
        return "quota"
    if any(k in s for k in (
        "503", "500", "504", "overloaded", "service unavailable", "unavailable", "server error",
        "deadline_exceeded", "timed out", "timeout", "connection", "remoteprotocol", "readerror", "reset by peer",
    )):
        return "server"
    if any(k in s for k in ("404", "not_found", "not found", "no longer available", "unsupported", "bidigeneratecontent")):
        return "model_gone"
    if any(k in s for k in ("400", "invalid_argument", "invalid argument", "bad request")):
        return "model_invalid"
    return "fatal"


DIRECTOR_SYSTEM_INSTRUCTION = """
You are the Autonomous Creative Director for Gemini Editor — a professional automated video production system.
Your job is to achieve the user's video editing and publishing goal by autonomously executing the available tools in sequence:

Available Tools:
1. `tool_ingest_source(source, limit, platform)`: Ingests target URL or creator handle, downloads streams, and prepares clip directories.
2. `tool_render_edit(clip_dir, edit_directive, skip_existing)`: Renders AI perception, watermark inpainting, BGM selection, and FFmpeg synthesis.
3. `tool_audit_clip(video_path, clip_dir, niche)`: Audits rendered video duration, quality, dopamine retention, and generates viral SEO.
4. `tool_publish_clip(video_path, clip_dir, platforms)`: Ingests to monetization gate and publishes to YouTube Shorts, Meta, and TikTok.
5. `tool_get_active_sessions(limit)`: Lists recent active video editing sessions from Telegram session memory.
6. `tool_reedit_session(session_id, edit_directive)`: Re-edits an existing video session based on creative directives (e.g. faster cuts, change music, adjust pacing).

Execution Rules:
1. Brand New Creation: Always start by ingesting the requested source using `tool_ingest_source`, render with `tool_render_edit`, and audit with `tool_audit_clip`.
2. Re-Editing / Revision / Retries:
   - If the user goal is to re-edit, retry, or adjust an existing session (or user clicked a retry button or entered a custom prompt):
     * If session_id is given (e.g. 'sess_...'), call `tool_reedit_session(session_id=..., edit_directive=...)`.
     * If session_id is not given (e.g. 're-edit the last reel'), call `tool_get_active_sessions` first to get the most recent session, then call `tool_reedit_session`.
   - Always follow up `tool_reedit_session` with `tool_audit_clip`.
3. Quality Gate: If audit passes (`audit_passed: true`), you may publish with `tool_publish_clip` if publication was requested.
4. Conclude with a clear, concise summary of all actions taken, files generated, and publishing status.
"""


def _resolve_gemini_api_key() -> str:
    """Finds Gemini API key across environment and standard project .env files."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    candidate_files = [
        os.path.join(_REPO_ROOT, ".env"),
        os.path.join(_REPO_ROOT, "Credentials", ".env"),
        os.path.join(os.path.dirname(_REPO_ROOT), ".env"),
    ]

    for env_path in candidate_files:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_clean = line.strip()
                        if line_clean.startswith("GEMINI_API_KEY=") or line_clean.startswith("GOOGLE_API_KEY="):
                            val = line_clean.split("=", 1)[1].strip().strip("'").strip('"')
                            if val:
                                return val
            except Exception:
                pass
    return ""


class AutonomousDirector:
    """
    Autonomous ReAct Loop Controller for Gemini Editor.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or _resolve_gemini_api_key()
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in Credentials/.env or system environment."
            )

        if not model_name:
            env_model = os.getenv("GEMINI_DIRECTOR_MODEL") or os.getenv("GEMINI_MODEL")
            if env_model:
                model_name = env_model
            elif _HAS_GOVERNOR and gemini_router is not None:
                # 100% dynamic: Governor provides the real-time highest ranked unbanned model
                model_name = gemini_router.get_available_model(task_type="reasoning")

            # Fallback to dynamic discovered roster from API cache if Governor has no state yet
            if not model_name:
                try:
                    from Gemini_Modules.gemini_router_module.list_models import get_active_models_and_ratings
                    active_roster, _ = get_active_models_and_ratings()
                    if active_roster:
                        model_name = active_roster[0]
                except Exception:
                    pass

        if not model_name:
            raise RuntimeError("Gemini Governor found zero active, unbanned generative models.")

        self.model_name = model_name
        try:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(_HTTP_TIMEOUT_SEC * 1000)),
            )
        except Exception:
            # Older SDKs without HttpOptions(timeout=...) — fall back (deadline still protects us)
            self.client = genai.Client(api_key=self.api_key)
        self.tools = get_agent_tools()

        # Activate video session budget tracking in governor
        if _HAS_GOVERNOR and gemini_router is not None:
            try:
                gemini_router.begin_video_session(
                    video_id=f"agent_run_{int(time.time())}",
                    video_duration=60.0
                )
            except Exception:
                pass

    @staticmethod
    def _inject_thought_signatures(chat) -> None:
        """Gemini 3.x requires thought_signature on function_call parts in history."""
        for attr in ("_curated_history", "_comprehensive_history"):
            hist = getattr(chat, attr, None)
            if hist and isinstance(hist, list):
                for content in hist:
                    for part in getattr(content, "parts", []) or []:
                        if getattr(part, "function_call", None) is not None and not getattr(part, "thought_signature", None):
                            try:
                                part.thought_signature = b"skip_thought_signature_validator"
                            except Exception:
                                pass

    def _report_success(self) -> None:
        if _HAS_GOVERNOR and gemini_router is not None:
            try:
                with gemini_router.state_lock:
                    st = gemini_router.model_states.get(self.model_name)
                    if st:
                        st["success_count"] += 1
                        st["total_calls"] += 1
                        st["last_used_at"] = time.monotonic()
            except Exception:
                pass

    def _make_config(self) -> "types.GenerateContentConfig":
        """Builds generation config with dynamic model tier awareness and thinking token suppression."""
        name_lower = (self.model_name or "").lower()
        if "pro" in name_lower or "ultra" in name_lower:
            tier_guidance = "Operating Tier: DEEP REASONING (Thorough multi-step validation and rigorous quality audit)."
        elif any(w in name_lower for w in ("lite", "nano", "small", "mini")):
            tier_guidance = "Operating Tier: WORKHORSE (High-throughput, ultra-concise JSON tool calling, zero unnecessary prose)."
        else:
            tier_guidance = "Operating Tier: BALANCED FLASH (High-speed multi-tool execution, fast tool chaining, aggressive pacing)."

        dynamic_instruction = (
            f"You are the Autonomous Creative Director for Gemini Editor (Active Model: {self.model_name}).\n"
            f"{tier_guidance}\n\n"
            f"{DIRECTOR_SYSTEM_INSTRUCTION}"
        )

        is_lite = any(w in name_lower for w in ("lite", "nano", "small", "mini"))

        kwargs: Dict[str, Any] = dict(
            system_instruction=dynamic_instruction,
            tools=self.tools,
            temperature=0.2,
        )
        # Thinking config is only valid for full reasoning models (Flash/Pro), Lite/Nano endpoints reject it with 400
        if not is_lite and os.getenv("DIRECTOR_THINKING", "off").lower() != "on":
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _summary_input(goal: str, execution_log: List[Dict[str, Any]]) -> str:
        if not execution_log:
            return goal
        lines = [f"- {e.get('tool')}: status={e.get('result_status')}" for e in execution_log]
        return (
            f"Original Goal: {goal}\nCompleted workflow steps so far:\n" + "\n".join(lines) +
            "\nContinue directly with the remaining workflow without repeating completed steps."
        )

    def run_goal(
        self,
        goal: str,
        max_turns: int = 10,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes an autonomous goal from start to finish with bounded retries,
        quota exhaustion detection, and rate-limit guardrails.
        """
        start_time = time.time()
        logger.info(f"\n{'='*70}\n🤖 [AUTONOMOUS DIRECTOR] Commencing Goal: '{goal}'\n{'='*70}")

        def _notify(event: str, data: Dict[str, Any]):
            if progress_callback:
                try:
                    progress_callback(event, data)
                except Exception as _cb_err:
                    logger.debug(f"Progress callback error: {_cb_err}")

        _notify("director_started", {"goal": goal, "model": self.model_name})

        # Configure Chat with Function Calling
        config = self._make_config()

        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )

        turn = 0
        execution_log: List[Dict[str, Any]] = []
        last_call_signature: Optional[str] = None
        call_counts: Dict[str, int] = {}
        deadline = start_time + _DEADLINE_SEC
        signature_recovered = False
        publish_allowed = any(w in (goal.lower() + " ") for w in _PUBLISH_WORDS)

        def _fail(kind: str, message: str, turn_no: int) -> Dict[str, Any]:
            logger.error(f"❌ [DIRECTOR {kind.upper()}] {message}")
            _notify("director_failed", {"kind": kind, "message": message})
            return {
                "status": "error" if kind != "timeout" else "timeout",
                "error_kind": kind,
                "message": message,
                "turns_executed": turn_no,
                "execution_time_sec": round(time.time() - start_time, 2),
                "execution_log": execution_log,
            }

        # First prompt
        current_input: Any = goal

        while turn < max_turns:
            turn += 1
            turn_start = time.time()
            logger.info(f"🔄 [DIRECTOR TURN {turn}/{max_turns}] Model={self.model_name} | Waiting for decision...")
            _notify("turn_start", {"turn": turn, "max_turns": max_turns, "model": self.model_name})

            if time.time() > deadline:
                return _fail("timeout", f"Director exceeded its {_DEADLINE_SEC:.0f}s deadline before turn {turn}.", turn)

            turn_response = None
            api_attempts = 0

            # Bounded API call: every failure path either retries a LIMITED number of times,
            # recovers once, or returns a clean error.
            while True:
                try:
                    self._inject_thought_signatures(chat)
                    turn_response = chat.send_message(current_input)
                    self._report_success()
                    break

                except Exception as api_err:
                    kind = classify_api_error(api_err)
                    logger.warning(f"⚠️ [DIRECTOR API {kind.upper()}] model={self.model_name}: {api_err}")

                    if kind in ("auth", "fatal"):
                        return _fail(kind, f"Gemini API {kind} error on turn {turn}: {api_err}", turn)

                    if kind == "signature":
                        if signature_recovered:
                            return _fail("signature", f"thought_signature error persisted after recovery: {api_err}", turn)
                        signature_recovered = True
                        logger.info("🩹 [DIRECTOR] Recovering from thought_signature error with a fresh chat (once).")
                        chat = self.client.chats.create(model=self.model_name, config=self._make_config())
                        current_input = self._summary_input(goal, execution_log)
                        continue

                    # Route model failure through Governor (the Director NEVER rotates models on its own)
                    if _HAS_GOVERNOR and gemini_router is not None and kind in ("model_gone", "model_invalid", "quota", "server"):
                        if kind == "model_invalid":
                            ban_reason = "400"
                        elif kind == "model_gone":
                            ban_reason = "404"
                        elif kind == "quota":
                            ban_reason = "429"
                        else:
                            ban_reason = "5xx"
                        gemini_router.mark_model_banned(self.model_name, error_type=ban_reason)

                        # Request the next best unbanned model dynamically from Governor
                        next_model = gemini_router.get_available_model(task_type="reasoning")
                        if next_model and next_model != self.model_name:
                            logger.warning(
                                f"🔄 [GOVERNOR ROTATION] Model '{self.model_name}' banned ({kind}). "
                                f"Governor dynamically assigned active model '{next_model}'. Rebinding session..."
                            )
                            self.model_name = next_model
                            chat = self.client.chats.create(model=self.model_name, config=self._make_config())
                            current_input = self._summary_input(goal, execution_log)
                            api_attempts = 0
                            continue

                    # If no alternative model could be provided by Governor, fail cleanly
                    if kind in ("model_gone", "model_invalid"):
                        return _fail(kind, f"Model '{self.model_name}' is unavailable or rejected parameters (all model fallbacks exhausted): {api_err}", turn)

                    # quota / server retry tracking
                    api_attempts += 1

                    # Fast-fail for daily/long-term quota exhaustion if no other models available
                    err_lower = str(api_err).lower()
                    if kind == "quota" and any(k in err_lower for k in ("free_tier_requests", "freetier", "limit: 20", "retry in 12h", "retry in 11h", "generativelanguage.googleapis.com")):
                        return _fail("quota_exhausted", f"Daily quota exhausted across all available models: {api_err}", turn)

                    if api_attempts > _MAX_API_RETRIES:
                        return _fail(kind, f"Gemini API still failing on {self.model_name} after {_MAX_API_RETRIES} retries ({kind}): {api_err}", turn)

                    base = 5.0 if kind == "quota" else 2.0
                    wait = min(base * (2 ** (api_attempts - 1)), 30.0)
                    if time.time() + wait > deadline:
                        return _fail("timeout", f"Not enough time left to retry ({kind}) within the {_DEADLINE_SEC:.0f}s deadline.", turn)
                    logger.warning(f"⏳ [DIRECTOR] Retry {api_attempts}/{_MAX_API_RETRIES} on {self.model_name} in {wait:.0f}s ({kind}).")
                    _notify("api_retry", {"attempt": api_attempts, "max": _MAX_API_RETRIES, "wait": wait, "reason": kind})
                    time.sleep(wait)

            if turn_response is None:
                return {
                    "status": "error",
                    "message": "Turn loop completed without valid response.",
                    "turns_executed": turn,
                    "execution_log": execution_log
                }

            response = turn_response

            # Check if model wants to call tools
            function_calls = response.function_calls
            if not function_calls:
                try:
                    final_text = response.text or "Goal execution completed."
                except Exception:
                    final_text = "Goal execution completed."
                logger.info(f"🏁 [DIRECTOR FINISHED] Final response reached in {turn} turn(s).")
                _notify("director_completed", {"final_text": final_text, "turns": turn})

                # Extract any rendered video paths and session info generated during execution
                latest_video = None
                latest_session = None
                latest_retry_count = None
                audit_passed = None

                for item in execution_log:
                    res_data = item.get("result") or {}
                    if isinstance(res_data, dict):
                        if res_data.get("new_master_video") or res_data.get("rendered_video_path"):
                            latest_video = res_data.get("new_master_video") or res_data.get("rendered_video_path")
                        if res_data.get("session_id"):
                            latest_session = res_data.get("session_id")
                        if res_data.get("retry_count") is not None:
                            latest_retry_count = res_data.get("retry_count")
                        if res_data.get("audit_passed") is not None:
                            audit_passed = res_data.get("audit_passed")

                return {
                    "status": "success",
                    "final_summary": final_text,
                    "turns_executed": turn,
                    "execution_time_sec": round(time.time() - start_time, 2),
                    "execution_log": execution_log,
                    "rendered_video_path": latest_video,
                    "session_id": latest_session,
                    "retry_count": latest_retry_count,
                    "audit_passed": audit_passed
                }

            # Extract any thought_signatures emitted by the model on its function_call parts
            tool_signatures = {}
            if getattr(response, "candidates", None) and response.candidates:
                c_content = getattr(response.candidates[0], "content", None)
                if c_content and getattr(c_content, "parts", None):
                    for p in c_content.parts:
                        fc = getattr(p, "function_call", None)
                        if fc and getattr(p, "thought_signature", None):
                            tool_signatures[fc.name] = p.thought_signature

            # Dispatch tool calls
            tool_response_parts = []
            for call in function_calls:
                tool_name = call.name
                tool_args = dict(call.args or {})

                # Duplicate call guardrail (prevent infinite loops)
                call_sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                call_counts[call_sig] = call_counts.get(call_sig, 0) + 1
                if tool_name == "tool_publish_clip" and not publish_allowed:
                    logger.warning("🛑 [GUARDRAIL] Blocked tool_publish_clip: the user's goal never asked to publish.")
                    tool_result = {
                        "status": "aborted",
                        "error": "Publishing was not requested in the user's goal. Do NOT publish. Finish with a summary."
                    }
                elif call_sig == last_call_signature or call_counts[call_sig] > _MAX_IDENTICAL_CALLS:
                    logger.warning(f"⚠️ [GUARDRAIL HIT] Model repeated an identical call: {call_sig}")
                    tool_result = {
                        "status": "aborted",
                        "error": "Duplicate tool call detected. You already called this tool with identical arguments. Change your approach or finish."
                    }
                else:
                    last_call_signature = call_sig
                    handler = TOOL_DISPATCH_MAP.get(tool_name)
                    if not handler:
                        tool_result = {"status": "error", "message": f"Unknown tool: '{tool_name}'"}
                    else:
                        logger.info(f"⚙️ [DISPATCHING TOOL] {tool_name}({tool_args})")
                        _notify("tool_executing", {"tool": tool_name, "args": tool_args})
                        try:
                            tool_result = handler(**tool_args)
                        except Exception as tool_ex:
                            tool_result = {
                                "status": "error",
                                "message": f"Tool execution crashed: {str(tool_ex)}"
                            }

                execution_log.append({
                    "turn": turn,
                    "tool": tool_name,
                    "args": tool_args,
                    "result_status": tool_result.get("status"),
                    "result": tool_result
                })

                _notify("tool_completed", {"tool": tool_name, "status": tool_result.get("status")})

                # Format part response for Gemini
                resp_part = types.Part.from_function_response(
                    name=tool_name,
                    response={"result": tool_result}
                )
                sig = tool_signatures.get(tool_name) or b"skip_thought_signature_validator"
                try:
                    resp_part.thought_signature = sig
                except Exception:
                    pass
                tool_response_parts.append(resp_part)

            # Feed tool results back into the chat
            current_input = tool_response_parts

            # RPM Guard: Pace turns to stay within 15 RPM rate limit
            elapsed_turn = time.time() - turn_start
            if elapsed_turn < _MIN_TURN_INTERVAL_SEC:
                time.sleep(_MIN_TURN_INTERVAL_SEC - elapsed_turn)

        # Exhausted turns
        logger.warning(f"⚠️ [DIRECTOR TIMEOUT] Reached max turns ({max_turns}) without completion.")
        return {
            "status": "turn_limit_reached",
            "message": f"Director reached maximum turn limit ({max_turns}).",
            "turns_executed": turn,
            "execution_log": execution_log,
            "execution_time_sec": round(time.time() - start_time, 2)
        }
