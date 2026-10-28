import json

import logging

import logging.handlers

import os

import random

import re

import signal

import threading

import time

import hashlib

from collections import deque

from datetime import datetime, timedelta

from typing import Any, Optional, Dict, List

try:
    from .list_models import get_active_models_and_ratings, refresh_gemini_models_cache
except ImportError:
    try:
        from Gemini_Modules.gemini_router_module.list_models import get_active_models_and_ratings, refresh_gemini_models_cache
    except ImportError:
        try:
            from Tools.Gemini_Modules.gemini_router_module.list_models import get_active_models_and_ratings, refresh_gemini_models_cache
        except ImportError:
            try:
                from list_models import get_active_models_and_ratings, refresh_gemini_models_cache
            except ImportError:
                get_active_models_and_ratings = None
                refresh_gemini_models_cache = None



# ── LOAD ENV EARLY ────────────────────────────────────────────────────────────

# GeminiGovernor is a singleton instantiated at import time.

# dotenv MUST be loaded before that so GEMINI_API_KEY and tuning params are set.

try:

    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)

    _governor_dir = os.path.dirname(os.path.abspath(__file__))
    _env_candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "Credentials", ".env"),
        os.path.join(_governor_dir, ".env"),
        os.path.join(os.path.dirname(_governor_dir), ".env"),
        os.path.join(os.path.dirname(_governor_dir), "Credentials", ".env"),
        os.path.join(os.path.dirname(os.path.dirname(_governor_dir)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(_governor_dir)), "Credentials", ".env"),
    ]

    for _cand_path in _env_candidates:
        if os.path.exists(_cand_path):
            _load_dotenv(_cand_path, override=False)

except Exception:

    pass

# ─────────────────────────────────────────────────────────────────────────────



# Suppress redundant warning from google.genai._api_client
if os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
    os.environ.pop("GOOGLE_API_KEY", None)

try:
    from google import genai
    from google.genai import types
except ImportError:
    try:
        import google.generativeai as genai
        types = None
    except ImportError:
        genai = None
        types = None

try:
    import requests
except ImportError:
    requests = None

try:
    from Diagnostics_Modules.gemini_trace import GeminiTrace
except ImportError:
    GeminiTrace = None

try:
    from langfuse import Langfuse
    _HAS_LANGFUSE = True
except ImportError:
    _HAS_LANGFUSE = False





logger = logging.getLogger("gemini_governor")



# ══════════════════════════════════════════════════════════════════════════════

# GLOBAL CIRCUIT BREAKER SINGLETON (Operation Beast Control v2)

# Shared across ALL calls in the entire process lifetime.

# When Gemini API goes down completely (5xx errors), this singleton

# remembers that state so no call wastes time retrying.

# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_GEMINI_STATE = {

    "gemini_down_until": 0.0,        # timestamp — skip all Gemini calls until this
    "consecutive_5xx": 0,            # capped at 50
    "last_fail_latency": 0.0,        # to distinguish quota (slow) vs random (fast)
}
_GLOBAL_STATE_LOCK = threading.Lock()

try:
    CIRCUIT_BREAKER_DEFAULT_THRESHOLD = int(os.getenv("GEMINI_CIRCUIT_BREAKER_THRESHOLD", "10"))
except (ValueError, TypeError):
    CIRCUIT_BREAKER_DEFAULT_THRESHOLD = 10


def is_gemini_globally_down() -> bool:
    """Check if the global circuit breaker is active."""
    with _GLOBAL_STATE_LOCK:
        return time.time() < _GLOBAL_GEMINI_STATE.get("gemini_down_until", 0)


def _record_gemini_5xx(latency: float, threshold: Optional[int] = None):
    """Record a 5xx failure. Trip breaker if threshold is met."""
    eff_threshold = threshold if threshold is not None else CIRCUIT_BREAKER_DEFAULT_THRESHOLD
    with _GLOBAL_STATE_LOCK:
        _GLOBAL_GEMINI_STATE["consecutive_5xx"] = min(
            _GLOBAL_GEMINI_STATE["consecutive_5xx"] + 1, 50
        )
        _GLOBAL_GEMINI_STATE["last_fail_latency"] = latency
       
        # Trip condition: threshold+ consecutive slow failures (5xx server deaths)
        if (_GLOBAL_GEMINI_STATE["consecutive_5xx"] >= eff_threshold
                and _GLOBAL_GEMINI_STATE["last_fail_latency"] > 3.0):
            cooldown = 60 + random.uniform(0, 5)  # jitter prevents burst alignment
            _GLOBAL_GEMINI_STATE["gemini_down_until"] = time.time() + cooldown
            logger.warning(
                f"🛑 [CIRCUIT BREAKER] Gemini globally DOWN for {cooldown:.0f}s. "
                f"({_GLOBAL_GEMINI_STATE['consecutive_5xx']} consecutive 5xx >= {eff_threshold}, "
                f"avg latency {latency:.1f}s)"
            )



def _reset_gemini_circuit():
    """Reset the circuit breaker on a successful call."""
    with _GLOBAL_STATE_LOCK:
        _GLOBAL_GEMINI_STATE["consecutive_5xx"] = 0
        _GLOBAL_GEMINI_STATE["last_fail_latency"] = 0.0
        _GLOBAL_GEMINI_STATE["gemini_down_until"] = 0.0


def get_seconds_until_midnight() -> int:
    """
    Calculates exact real-time seconds remaining from current moment until 12:00 AM (Midnight) of the new day.
    For example: if RPD hits at 3:00 PM, calculates the exact time difference to 12:00 AM (9 hours = 32,400s).
    """
    now = datetime.now()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_remaining = int((midnight - now).total_seconds())
    return max(seconds_remaining, 60)

get_seconds_until_midnight_pt = get_seconds_until_midnight  # backward compatibility alias

# ══════════════════════════════════════════════════════════════════════════════





class GeminiGovernor:

    """

    V3.5 World-Class Intelligent Router.

    Handles multi-tier scoring, adaptive cooldowns, rate limiting, and persistence.

    """



    _instance = None

    _lock = threading.Lock()



    def __new__(cls):

        with cls._lock:

            if cls._instance is None:

                cls._instance = super(GeminiGovernor, cls).__new__(cls)

                cls._instance._init_governor()

            return cls._instance



    def _init_governor(self):

        # Granular Locks for high-performance concurrency

        self.state_lock = threading.RLock()

        self.rate_limit_lock = threading.Lock()
        self.memory_lock = threading.Lock()

        # Langfuse Observability
        self.langfuse = None
        if _HAS_LANGFUSE:
            try:
                pk = os.getenv("LANGFUSE_PUBLIC_KEY")
                sk = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
                if pk and sk:
                    self.langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
                    logger.info("🔭 [LANGFUSE] Observability Engine ACTIVE.")
            except Exception as _lf_err:
                logger.warning(f"⚠️ [LANGFUSE] Initialization failed: {_lf_err}")



        # Config from Environment

        self.MAX_RPM = int(os.getenv("GEMINI_MAX_REQUESTS_PER_MIN", 15))

        self.MAX_FPM = int(os.getenv("GEMINI_MAX_FALLBACKS_PER_MIN", 2))

        self.JITTER = float(os.getenv("GEMINI_JITTER_RANGE", 0.03))

        self.DECAY = float(os.getenv("GEMINI_DECAY_RATE", 0.99))

        self.MEMORY_TTL = int(os.getenv("GEMINI_MEMORY_TTL", 120))

        self.STATE_FILE = os.getenv("GEMINI_STATE_FILE", "Credentials/gemini_states.json")

        self.LOG_FILE = os.getenv("GEMINI_LOG_FILE", "logs/gemini_governor.log")
        self.ROUTING_JSON = os.getenv("GEMINI_ROUTING_JSON", "logs/gemini_routing.json")

        self.REQUEST_DEADLINE = 300  # seconds — raised from 90s; heavy vision payloads take ~40-60s per model

        self.global_cooldown_until = 0

        self.last_successful_model = None



        # Validation

        assert 0 <= self.JITTER < 0.1, "JITTER_RANGE must be between 0 and 0.1"

        assert self.MAX_RPM > 0, "MAX_REQUESTS_PER_MIN must be positive"



        # Model State & Metrics

        self.model_states = {}

        self.cache = {}

        self.stats = {

            "logical_requests": 0,

            "api_calls": 0,

            "cache_hits": 0,

            "blocked_calls": 0,

            "multi_task_calls": 0,

            "failures": 0,

            "calls_per_module": {},

            "payload_sizes": [],

        }



        # Sliding Window Queues (Monotonic)

        self.request_timestamps = deque()

        # Vanguard Elite Tracking

        self.session_locks = {} # {session_id: {task_type: model_name}}

        self.session_costs = {} # {session_id: total_calls} - simple cost tracking

        self.MAX_BUDGET = 20    # Threshold calls before force-lite

        self.fallback_timestamps = deque()

        self.recent_task_failures = {}  # { (model, task): timestamp }



        # ── VIDEO BUDGET SYSTEM ─────────────────────────────────────────────

        # Adaptive per-video call cap based on clip duration

        self._video_session_id: Optional[str] = None

        self._video_call_count: int = 0

        self._video_call_budget: int = 5       # default; overridden in begin_video_session

        self._video_budget_log: list = []



        # Task priority: lower number = higher priority (always runs first)

        self.TASK_PRIORITY = {

            "watermark":  1,   # pixel-precise — must run

            "caption":    2,   # creative output — must run

            "narrative":  3,   # story quality — important

            "price":      4,   # monetisation — medium

            "reasoning":  4,   # orchestration — medium

            "master":     4,   # vision verify — medium

            "vision":     5,   # semantic scene — low (OpenCV can handle)

            "analysis":   5,   # content analysis — lowest

        }

        # ────────────────────────────────────────────────────────────────────



        # Setup Logging

        self._setup_structured_logging()



        # Initialize/Load Models

        self.current_key_hash = self._get_key_hash()

        self._initialize_models()

        self._load_states()



    def _get_key_hash(self):

        """Compute a non-reversible hash of the API key for change detection."""

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not key:

            governor_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(os.getcwd(), ".env"),
                os.path.join(os.getcwd(), "Credentials", ".env"),
                os.path.join(governor_dir, ".env"),
                os.path.join(os.path.dirname(governor_dir), ".env"),
                os.path.join(os.path.dirname(governor_dir), "Credentials", ".env"),
                os.path.join(os.path.dirname(os.path.dirname(governor_dir)), ".env"),
                os.path.join(os.path.dirname(os.path.dirname(governor_dir)), "Credentials", ".env"),
            ]

            for env_path in candidates:
                if os.path.exists(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line_clean = line.strip()
                                if line_clean.startswith("GEMINI_API_KEY=") or line_clean.startswith("GOOGLE_API_KEY="):
                                    key = line_clean.split("=", 1)[1].strip().strip("'").strip('"')
                                    break
                        if key:
                            break
                    except Exception:
                        pass

        key = key or ""

        return hashlib.sha256(key.encode()).hexdigest()



    def _tick_ban_timers_unlocked(self):

        now = time.monotonic()

        elapsed = now - getattr(self, "last_time_check", now)

        self.last_time_check = now

       

        for m, state in self.model_states.items():

            if state.get("status") == "BANNED":

                rem = state.get("ban_remaining_seconds", 0) - elapsed

                if rem <= 0:

                    state.update({

                        "status": "ACTIVE",

                        "ban_remaining_seconds": 0,

                        "warmup_calls": 5

                    })

                else:

                    state["ban_remaining_seconds"] = rem



    def _setup_structured_logging(self):

        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

        self.routing_logger = logging.getLogger("gemini_routing")

        self.routing_logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if re-initialized

        if not self.routing_logger.handlers:

            handler = logging.handlers.RotatingFileHandler(

                self.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5

            )

            handler.setFormatter(logging.Formatter("%(message)s"))

            self.routing_logger.addHandler(handler)



    def _get_models_cache_path(self) -> str:
        """Locates storage/gemini_models_cache.json."""
        if os.getenv("GEMINI_CACHE_FILE"):
            return os.path.abspath(os.getenv("GEMINI_CACHE_FILE"))
        cands = [
            os.path.join(os.getcwd(), "storage", "gemini_models_cache.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "gemini_models_cache.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "gemini_models_cache.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "storage", "gemini_models_cache.json"),
        ]
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0]

    def _initialize_models(self):
        """Initializes supported models directly from storage/gemini_models_cache.json with dynamic discovery fallback."""
        models = []
        cache_path = self._get_models_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    if isinstance(cache_data, dict):
                        models = cache_data.get("models", [])
                        ratings = cache_data.get("task_ratings", {})
                        if ratings:
                            self.TASK_MODEL_RATINGS = ratings
                        logger.info(f"📋 [GOVERNOR] Successfully loaded {len(models)} models directly from JSON cache: {cache_path}")
            except Exception as exc:
                logger.warning(f"⚠️ [GOVERNOR] Error reading JSON models cache {cache_path}: {exc}")

        # If cache file wasn't found or empty, delegate to list_models module dynamically
        if not models and get_active_models_and_ratings:
            try:
                disc_models, ratings = get_active_models_and_ratings()
                if disc_models:
                    models = disc_models
                if ratings:
                    self.TASK_MODEL_RATINGS = ratings
            except Exception as exc:
                logger.warning(f"Dynamic model lookup failed: {exc}")

        with self.state_lock:
            for m in models:
                if m not in self.model_states:
                    self.model_states[m] = {
                        "status": "ACTIVE",
                        "ban_remaining_seconds": 0,
                        "success_count": 5.0, # optimistic start
                        "total_calls": 5.0,
                        "429_count": 0,
                        "fail_count": 0,
                        "avg_latency": 0.5,
                        "last_used_at": 0,
                        "last_success_at": 0,
                        "warmup_calls": 0
                    }



    def begin_video_session(self, video_id: str, video_duration: float = 0.0):

        """

        Call at the start of every video to reset the per-video call budget.

        Budget adapts to clip length:

          < 15s  → 3 calls  (short-form, minimal AI)

          < 60s  → 5 calls  (standard short)

          >= 60s → 7 calls  (long-form, more complexity)

        """

        self._video_session_id = video_id

        self._video_call_count = 0

        self._video_budget_log = []



        if video_duration < 15:

            self._video_call_budget = 12

        elif video_duration < 60:

            self._video_call_budget = 18

        else:

            self._video_call_budget = 25



        logger.info(

            f"🎬 [BUDGET] New video session: {video_id} | "

            f"duration={video_duration:.1f}s | budget={self._video_call_budget} calls"

        )

        # Ensure models and task ratings are kept current if TTL expired
        if get_active_models_and_ratings:
            try:
                self._initialize_models()
            except Exception:
                pass



    def _load_states(self):

        self.last_time_check = time.monotonic()  # initialize timer correctly

        if not os.path.exists(self.STATE_FILE):

            return

        try:

            with open(self.STATE_FILE, "r") as f:

                data = json.load(f)

               

                # Check for API Key change

                stored_hash = data.get("_key_hash")

                if stored_hash and stored_hash != self.current_key_hash:

                    logger.info("🔑 New API Key detected! Wiping penalty states and refreshing live model roster.")

                    with self.state_lock:

                        for m in self.model_states:

                            self.model_states[m].update({

                                "status": "ACTIVE",

                                "ban_remaining_seconds": 0,

                                "fail_count": 0,

                                "429_count": 0

                            })

                    if refresh_gemini_models_cache:
                        try:
                            refresh_gemini_models_cache(force=True)
                            self._initialize_models()
                        except Exception:
                            pass

                    # Do not load old states from json

                    return

               

                with self.state_lock:

                    for m, state in data.items():

                        if m.startswith("_"): continue # Skip metadata

                        if m in self.model_states:

                            # Handle legacy banned_until smoothly

                            if "banned_until" in state:

                                state.pop("banned_until", None)

                                state["ban_remaining_seconds"] = 0

                            self.model_states[m].update(state)

        except Exception as e:

            logger.warning(f"Failed to load states: {e}")



    def _save_states(self):

        """Atomic write to prevent corruption."""

        try:

            with self.state_lock:

                self._tick_ban_timers_unlocked()

                serializable = {"_key_hash": self.current_key_hash}

                for m, state in self.model_states.items():

                    s = state.copy()

                    serializable[m] = s

           

            temp_file = self.STATE_FILE + ".tmp"

            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)

            with open(temp_file, "w") as f:

                json.dump(serializable, f, indent=2)

            os.replace(temp_file, self.STATE_FILE)

        except Exception as e:

            logger.warning(f"Failed to save states: {e}")



    def is_model_available(self, model_name):

        with self.state_lock:

            self._tick_ban_timers_unlocked()

            state = self.model_states.get(model_name)

            if not state:

                return True



            if state["status"] == "BANNED":

                return False

            return True



    def _clean_window(self, queue, window=60):

        now = time.monotonic()

        while queue and now - queue[0] > window:

            queue.popleft()



    def can_make_request(self):

        with self.rate_limit_lock:

            self._clean_window(self.request_timestamps)

            return len(self.request_timestamps) < self.MAX_RPM



    def can_make_fallback(self):

        with self.rate_limit_lock:

            self._clean_window(self.fallback_timestamps)

            return len(self.fallback_timestamps) < self.MAX_FPM



    def _add_request_event(self, is_fallback=False):

        now = time.monotonic()

        with self.rate_limit_lock:

            self.request_timestamps.append(now)

            if is_fallback:

                self.fallback_timestamps.append(now)



    def mark_model_banned(self, model_name, error_type="429", seconds: Optional[int] = None, error_message: str = ""):

        with self.state_lock:

            self._tick_ban_timers_unlocked()

            state = self.model_states.get(model_name)
            if not state:
                self.model_states[model_name] = {
                    "status": "ACTIVE",
                    "ban_remaining_seconds": 0,
                    "success_count": 0,
                    "total_calls": 0,
                    "429_count": 0,
                    "fail_count": 0,
                    "avg_latency": 1.0,
                    "last_used_at": time.time(),
                    "last_success_at": 0,
                    "warmup_calls": 0
                }
                state = self.model_states[model_name]

            if seconds is not None:
                state["status"] = "BANNED"
                state["ban_remaining_seconds"] = seconds
                logger.warning(f"🚫 Model {model_name} BANNED for {seconds}s (manual override).")
                return

            if error_type == "429":
                state["429_count"] += 1
                duration_sec = 45
                state["status"] = "BANNED"
                state["ban_remaining_seconds"] = duration_sec
                logger.warning(f"🚫 Model {model_name} BANNED for 45s due to 429 quota (RPM/TPM).")

            elif error_type == "timeout":

                state["status"] = "BANNED"

                state["ban_remaining_seconds"] = 30

                logger.warning(f"⏳ Model {model_name} isolated for 30s due to Timeout.")

            elif error_type == "5xx":

                state["status"] = "BANNED"

                state["ban_remaining_seconds"] = 90

                logger.warning(f"🔥 Model {model_name} isolated for 90s due to Server Error.")

            elif error_type in ("404", "403", "400", "not_found", "model_deprecated", "permission_denied", "client_error"):
                state["status"] = "BANNED"
                state["ban_remaining_seconds"] = 86400  # 24-hour ban for deprecated/inaccessible model
                logger.warning(f"🚫 Model {model_name} PERMANENTLY BANNED due to {error_type} error.")
                if refresh_gemini_models_cache:
                    try:
                        refresh_gemini_models_cache(force=True)
                        self._initialize_models()
                    except Exception as ref_err:
                        logger.warning(f"Failed to auto-refresh model cache: {ref_err}")
            elif error_type == "safety":

                state["status"] = "BANNED"

                state["ban_remaining_seconds"] = 300

                logger.warning(f"🛡️ Model {model_name} BANNED for 300s due to Safety Block.")



    def _get_cache_key(self, prompt: Any, metadata: Dict[str, Any]) -> str:

        """Standardized cache key for requests (Vanguard Pattern)."""

        # Ensure prompt is stringifiable for hashing

        p_str = str(prompt)

        m_str = json.dumps(metadata, sort_keys=True)

        raw = f"{p_str}_{m_str}"

        return hashlib.sha256(raw.encode()).hexdigest()



    def get_available_model(self, task_type, session_id=None, exclude_models=None):

        """

        World-Class Weighted Scoring Engine (V4.2 'Elite' Edition).

        Factors: Stability Lock, Success Rate, Failure Decay, Latency, Penalty,

                 Cost Guard, Cold Start Bias.

        exclude_models: optional set/list of model names to skip (used by retry loop).

        """

        exclude_set = set(exclude_models) if exclude_models else set()



        # 1. Stability Lock Check

        if session_id and session_id in self.session_locks:

            locked_model = self.session_locks[session_id].get(task_type)

            if locked_model and locked_model not in exclude_set and self.model_states.get(locked_model, {}).get("status") != "BANNED":

                return locked_model



        # 2. Max Cost Guard Check

        force_lite = False

        if session_id and self.session_costs.get(session_id, 0) > self.MAX_BUDGET:

            # Don't force lite if the task inherently forbids lite models (which would cause a dead end)

            if task_type not in ["watermark", "master", "vision"]:

                logger.info(f"💰 [MAX_COST_GUARD] Session {session_id} over budget ({self.MAX_BUDGET} calls). Forcing Lite model.")

                force_lite = True



        with self.state_lock:

            self._tick_ban_timers_unlocked()

            # Task Boosting (V4.0 — Dynamic JSON & Full Roster Matrix)
            boosts = getattr(self, "TASK_MODEL_RATINGS", {})
            task_boost = boosts.get(task_type, {})



            # Cost Weights (Inverse)
            # For complex multi-turn reasoning, tools, and heavy vision, prioritize robust production Flash models
            if task_type in ("reasoning", "watermark", "vision", "master"):
                cost_weights = {
                    "flash": 0.6,
                    "lite": 0.9,
                    "pro": 1.2
                }
            else:
                cost_weights = {
                    "pro": 1.0,
                    "flash": 0.7,
                    "lite": 0.3
                }



            best_model = None

            max_score = -float('inf')

            now = time.monotonic()



            for name, state in self.model_states.items():

                if state["status"] == "BANNED":

                    continue

               

                # Skip models excluded by the retry loop
                if name in exclude_set:
                    continue

                # 🛡️ 1.5 Filter (Blocked by default unless GEMINI_ALLOW_1_5 is explicitly set)
                block_1_5 = os.getenv("GEMINI_BLOCK_1_5", "1").lower() in ("1", "true", "yes") or os.getenv("GEMINI_ALLOW_1_5", "0").lower() not in ("1", "true", "yes")
                if block_1_5 and "1.5" in name:
                    continue

                name_lower = name.lower()
                # Skip specialized non-generative, canary, or experimental preview endpoints
                if any(kw in name_lower for kw in (
                    "embedding", "embed", "imagen", "image", "preview-image", "bison", "aqa", "gecko",
                    "text-001", "tts", "preview-tts", "customtools", "transcribe",
                    "robotics", "computer-use", "live-translate", "thinking",
                    "audio", "native-audio", "bidi", "realtime", "omni",
                    "banana", "canary", "test"
                )):
                    continue

                # For vision tasks, skip audio-only endpoints
                if task_type in ("watermark", "vision") and "audio" in name_lower:
                    continue

                # 🛡️ FATAL FLAW FIX (REMOVED LITE RESTRICTION FOR MULTI-MODEL ROTATION)
                # The Free Tier flash-lite endpoints consistently throw 5xx Server Errors
                # when fed multi-image logic (watermark/vision). Force skip them.
                # [OVERRIDE: Allowed as fallback for multi-model rotation if user requests]
                # if task_type in ["watermark", "master", "vision"] and "lite" in name:
                #    continue

               

                # Filter out recently failed models for this specific task (Request Memory)

                with self.memory_lock:

                    fail_time = self.recent_task_failures.get((name, task_type))

                    if fail_time and now - fail_time < self.MEMORY_TTL:

                        continue



                # 1. Base Score (Success Rate)

                total = state["total_calls"]

                success_rate = (state["success_count"] / total) if total > 0 else 1.0

               

                # [ELITE] Failure Decay & Adaptive Weighting

                state["total_calls"] *= self.DECAY

                state["success_count"] *= self.DECAY

                state.setdefault("fail_count", 0)

                state["fail_count"] *= 0.9 # Decay failures over time

               

                # 2. Latency & Reliability Penalties

                # Penalty = (FailCount * 10) + (AvgLatency * 0.4)

                reliability_penalty = (state["fail_count"] * 10.0)

                latency_penalty = (state["avg_latency"] * 0.4)



                # 3. Freshness & Exploration

                freshness = min(1.0, (now - state["last_used_at"]) / 300) if state["last_used_at"] > 0 else 1.0

               

                # 4. Cold Start Bias (Vanguard Elite Safeguard)

                # New models receive a small penalty until proven stable (>= 5 calls)

                cold_start_penalty = 5.0 if total < 5 else 0.0



                # 5. [LITE_ONLY] For Cost Guard

                if force_lite and "lite" not in name:

                    continue



                # 6. Elite Scoring Formula

                base_score = (

                    (success_rate * 50) -

                    reliability_penalty -

                    latency_penalty -

                    cold_start_penalty +

                    (freshness * 5)

                )



                # 7. Apply Task Boost

                boost_val = task_boost.get(name, 1.0)

               

                # 8. Apply Cost Weighting

                c_type = "pro" if "pro" in name else ("lite" if "lite" in name else "flash")

                cost_inv = 1.0 / cost_weights.get(c_type, 0.7)



                # 9. Jitter (Clamp +/- 0.03)

                jitter = random.uniform(-self.JITTER, self.JITTER)



                final_score = (base_score * boost_val * cost_inv) + jitter



                # Soft Recovery (Warmup)

                if state.get("warmup_calls", 0) > 0:

                    final_score *= 0.5



                if final_score > max_score:

                    max_score = final_score

                    best_model = name



            # Update Logic (Vanguard Elite)

            if best_model:

                if session_id:

                    # Set Stability Lock

                    if session_id not in self.session_locks:

                        self.session_locks[session_id] = {}

                    self.session_locks[session_id][task_type] = best_model

                    # Increment Session Cost

                    self.session_costs[session_id] = self.session_costs.get(session_id, 0) + 1



                return best_model

            # If all candidate models are currently banned or cooling down
            if not best_model and not exclude_set:
                if refresh_gemini_models_cache:
                    try:
                        refresh_gemini_models_cache(force=True)
                        self._initialize_models()
                    except Exception:
                        pass
            return None

    def get_valid_models_for_task(self, task_type: str) -> List[str]:
        """
        Returns all candidate models eligible for this task_type,
        ordered by task affinity / ratings descending.
        Excludes non-generative endpoints and blocked versions.
        """
        with self.state_lock:
            boosts = getattr(self, "TASK_MODEL_RATINGS", {})
            task_boost = boosts.get(task_type, {})
            block_1_5 = os.getenv("GEMINI_BLOCK_1_5", "1").lower() in ("1", "true", "yes") or os.getenv("GEMINI_ALLOW_1_5", "0").lower() not in ("1", "true", "yes")
            non_gen_kws = (
                "embedding", "embed", "imagen", "bison", "aqa", "gecko",
                "text-001", "tts", "preview-tts", "customtools", "transcribe",
                "robotics", "computer-use", "live-translate", "thinking",
                "audio", "native-audio", "bidi", "realtime",
                "banana", "canary", "test"
            )

            valid_models = []
            for name in self.model_states.keys():
                name_lower = name.lower()
                if any(kw in name_lower for kw in non_gen_kws):
                    continue
                if block_1_5 and "1.5" in name_lower:
                    continue
                if task_type in ("watermark", "vision") and "audio" in name_lower:
                    continue
                if task_boost and task_boost.get(name, 1.0) <= 0:
                    continue
                valid_models.append(name)

            # Sort by task boost rating descending, then Pro > Flash > Lite
            def _rank_key(m):
                boost = task_boost.get(m, 1.0)
                m_low = m.lower()
                tier_val = 0 if "pro" in m_low else (1 if ("flash" in m_low and "lite" not in m_low) else 2)
                return (boost, -tier_val)

            valid_models.sort(key=_rank_key, reverse=True)
            return valid_models

    def get_min_cooldown_remaining(self, task_type: str = "reasoning") -> float:
        """
        Returns the minimum remaining cooldown/ban time in seconds across models eligible for task_type.
        Returns 0.0 if any model is available immediately.
        """
        with self.state_lock:
            self._tick_ban_timers_unlocked()
            eligible_bans = []
            for name, state in self.model_states.items():
                name_lower = name.lower()
                if any(kw in name_lower for kw in (
                    "embedding", "embed", "imagen", "bison", "aqa", "gecko",
                    "text-001", "tts", "preview-tts", "customtools", "transcribe",
                    "robotics", "computer-use", "live-translate", "3.", "3-", "thinking",
                    "audio", "native-audio", "bidi", "realtime"
                )):
                    continue
                if state["status"] == "BANNED":
                    rem = state.get("ban_remaining_seconds", 0)
                    if 0 < rem < 86400:  # Ignore permanent deprecation bans
                        eligible_bans.append(rem)
                else:
                    return 0.0
            return float(min(eligible_bans)) if eligible_bans else 15.0


    def simplify_prompt(self, prompt: Any, tier: str = "high") -> Any:

        """

        Public wrapper for prompt simplification used by the VANGUARD retry loop.

        tier: 'high' (full prompt), 'mid' (truncated), 'low' (aggressive simplification).

        """

        if tier == "high":

            return prompt

           

        # 🛡️ FATAL BUG FIX: Do NOT destroy the image dictionary schema

        if isinstance(prompt, list):

            new_prompt = []

            for item in prompt:

                if isinstance(item, str):

                    # Vision/forensic prompts are longer; use 8000-char limit to preserve JSON schema
                    mid_limit = 8000

                    if tier == "mid" and len(item) > mid_limit:

                        new_prompt.append(item[:mid_limit] + "\n[TRUNCATED]")

                    elif tier == "low":

                        new_prompt.append(f"Simplify decision:\n{item[:500]}")

                    else:

                        new_prompt.append(item)

                else:

                    new_prompt.append(item) # Keep the dictionary (b64 image) intact

            return new_prompt



        p_str = str(prompt)

        if tier == "mid":

            if len(p_str) > 2000:

                return p_str[:2000] + "\n[TRUNCATED — respond with best judgement]"

            return prompt

        else:  # low

            return f"Simplify decision:\n{p_str[:500]}"

       



    def _call_ollama(self, prompt: Any) -> Optional[str]:

        """

        [VANGUARD] Local Fallback to Ollama (Phi-3).

        """

        # Ensure prompt is safely converted to text, discarding Vision PIL images

        if isinstance(prompt, list):

            prompt = "\n".join([str(item) for item in prompt if isinstance(item, str)])

        elif not isinstance(prompt, str):

            try:

                prompt = str(prompt)

            except Exception:

                prompt = "Analyze image safely."



        import requests

        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        model = os.getenv("OLLAMA_MODEL", "phi3")

       

        try:

            payload = {

                "model": model,

                "prompt": prompt,

                "stream": False

            }

            response = requests.post(url, json=payload, timeout=15)

            if response.status_code == 200:

                return response.json().get("response")

            return None

        except Exception as e:

            if "refused" in str(e).lower() or "connection" in str(e).lower():

                logger.error(f"⚠️ [VANGUARD] Ollama Service UNAVAILABLE at {url}. Ensure Ollama is running for local fallback.")

            else:

                logger.error(f"❌ [VANGUARD] Ollama Error: {e}")

            return None



    def _timeout_handler(self, signum, frame):

        raise TimeoutError("Gemini API call timed out after 10s")



    def generate(

        self,

        task_type,

        prompt,

        metadata=None,

        module_name="unknown",

        gen_config=None,

        safety_settings=None,

        existing_confidence=None,

        model_name: Optional[str] = None,

        session_id: Optional[str] = None,

        api_key: Optional[str] = None,

    ):

        """

        VANGUARD BULLETPROOF GENERATOR: Loop-based Retry + Global Deadline + Jitter.

        """

        self.stats["logical_requests"] += 1
        self.stats["calls_per_module"][module_name] = self.stats["calls_per_module"].get(module_name, 0) + 1
        logger.info(f"🤖 [GEMINI CALL] Module: '{module_name}' | Task: '{task_type}'")



        # ── EARLY API KEY GATE — fail fast, no wasteful retries ──────────────

        _api_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "").strip()

        if not _api_key:

            logger.error(
                "🔑 [GOVERNOR] GEMINI_API_KEY is not set in your .env file!\n"
                "   → Open .env and set: GEMINI_API_KEY=AIzaSy...\n"
                "   → Get a free key at: https://aistudio.google.com/app/apikey"
            )

            return None

        # ──────────────────────────────────────────────────────────────────




        # ── INITIAL CHECKS ──────────────────────────────────────────────

        metadata = metadata or {}

        cache_key = self._get_cache_key(prompt, metadata)

        if cache_key in self.cache:

            self.stats["cache_hits"] += 1

            return self.cache[cache_key]



        # ── ORCHESTRA FAST LANE (Groq + Mistral primary) ─────────────────────

        # Text tasks → RouterOrchestra (Groq/Mistral) with cache + global budget

        # Vision tasks → Orchestra returns None → falls through to Gemini loop below

        TEXT_TASKS = {"reasoning", "narrative", "price", "analysis", "caption", "master", "creative", "seo_generation", "seo", "viral_metadata"}

        is_escalated = False

        if task_type in TEXT_TASKS and isinstance(prompt, str):

            try:

                from Intelligence_Modules.router_orchestra import orchestra

                orch_result = orchestra.route(

                    prompt=prompt,

                    task_type=task_type,

                    visual_context=metadata.get("visual_context")

                )

                if orch_result:

                    self.cache[cache_key] = orch_result

                    return orch_result

               

                # Orchestra failed or budget exhausted → mark for escalation fallback

                logger.warning(f"⚠️ [ORCHESTRA] No result for '{task_type}'. Falling back to Gemini.")

                is_escalated = True

                metadata["escalated_fallback"] = True

            except ImportError:
                # Optional module not present; proceed directly to Gemini
                pass
            except Exception as _orch_err:
                logger.warning(f"⚠️ [ORCHESTRA] Error: {_orch_err}. Falling back to Gemini.")



        # ── PRIORITY-AWARE BUDGET GATE (Gemini specific) ──────────────────────

        # This gate now ONLY applies to Gemini calls, preserving quota for vision.

        if self._video_session_id is not None:

            task_priority = self.TASK_PRIORITY.get(task_type, 5)

            remaining = self._video_call_budget - self._video_call_count



            # Escalation bypass: If third-party models failed, Gemini MUST "got their back"

            if is_escalated:

                logger.info(f"🛡️ [VANGUARD] 'God-Mode' Fallback activated for '{task_type}'. Bypassing tight budget.")

            else:

                # Standard budget logic for non-escalated Gemini calls (e.g. Vision)

                if task_priority >= 5 and remaining <= 1:

                    logger.warning(

                        f"🚫 [BUDGET] Skipping low-priority task '{task_type}' "

                        f"({module_name}) — only {remaining} calls left."

                    )

                    self._video_budget_log.append({"task": task_type, "module": module_name, "action": "skipped_low_priority"})

                    return None



                if self._video_call_count >= self._video_call_budget:

                    logger.warning(

                        f"🚫 [BUDGET] Hard limit reached ({self._video_call_budget} calls). "

                        f"Skipping '{task_type}' ({module_name})."

                    )

                    self._video_budget_log.append({"task": task_type, "module": module_name, "action": "skipped_budget_exhausted"})

                    return None



            self._video_call_count += 1

            logger.info(

                f"📊 [BUDGET] Gemini Call {self._video_call_count}/{self._video_call_budget} "

                f"| task='{task_type}' | module='{module_name}' | escalated={is_escalated}"

            )

        # ────────────────────────────────────────────────────────────────────



        # ── LANGFUSE TRACE INITIALIZATION ──────────────────────────────────

        trace = None

        if self.langfuse:

            try:

                trace = self.langfuse.trace(

                    name=f"amtce_{task_type}",

                    tags=[module_name, "production"],

                    metadata={

                        "module": module_name,

                        "session_id": session_id,

                        "video_session": self._video_session_id

                    }

                )

            except: pass

        # ──────────────────────────────────────────────────────────────────



        start_time = time.time()
        attempts = 0
        cooldown_cycles = 0
        tried_models = set()
        last_error = "Unknown"

        # Dynamic model exhaustion calculation based on task type
        valid_models = self.get_valid_models_for_task(task_type)
        total_candidates = len(valid_models) if valid_models else len(self.model_states)
        MAX_ATTEMPTS = total_candidates if total_candidates > 0 else 5
        circuit_threshold = max(total_candidates, 10)

        # --- THE RELIABILITY LOOP ---
        while True:
            # 1. Check Global Deadline
            elapsed = time.time() - start_time
            if elapsed > self.REQUEST_DEADLINE:
                logger.warning(f"⌛ [VANGUARD] Global Deadline ({self.REQUEST_DEADLINE}s) hit for {task_type}. Exiting loop.")
                break

            # 2. Routing (Pick best model, avoiding tried ones)
            current_model = None
            if model_name and attempts == 0:
                # Use requested model if it's not banned
                if self.is_model_available(model_name):
                    current_model = model_name

            if not current_model and attempts < MAX_ATTEMPTS:
                current_model = self.get_available_model(task_type, session_id=session_id, exclude_models=tried_models)

            if not current_model or attempts >= MAX_ATTEMPTS:
                with self.state_lock:
                    self._tick_ban_timers_unlocked()
                    recoverable_models = [
                        m for m, s in self.model_states.items()
                        if s.get("status") == "ACTIVE" or s.get("ban_remaining_seconds", 0) <= 95
                    ]

                if not recoverable_models:
                    logger.error(
                        f"🚨 [FATAL QUOTA] All candidate models are permanently inaccessible (403/404)!\n"
                        f"🛑 Halting retry loop ({attempts} attempts tried). Please check your API key and permissions."
                    )
                    break

                if cooldown_cycles < 2:
                    cooldown_cycles += 1
                    logger.warning(
                        f"⏳ [ALL MODELS QUOTA HIT] All candidate models in quarantine for '{task_type}'.\n"
                        f"   😴 Entering 90s master cooldown to guarantee full rolling RPM/TPM reset across models... (Pass {cooldown_cycles}/2)\n"
                        f"   💡 TIP: If you want to continue immediately, update GEMINI_API_KEY in Credentials/.env"
                    )
                    # Interruptible sleep in 1s slices to detect live API key updates in .env
                    initial_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
                    key_updated = False
                    for _ in range(90):
                        time.sleep(1.0)
                        try:
                            from dotenv import dotenv_values
                            cands = [
                                os.path.join(os.getcwd(), "Credentials", ".env"),
                                os.path.join(os.getcwd(), ".env"),
                            ]
                            for c in cands:
                                if os.path.exists(c):
                                    vals = dotenv_values(c)
                                    new_k = vals.get("GEMINI_API_KEY") or vals.get("GOOGLE_API_KEY")
                                    if new_k and new_k != initial_key:
                                        os.environ["GEMINI_API_KEY"] = new_k
                                        key_updated = True
                                        logger.info("🔑 [HOT RELOAD] New GEMINI_API_KEY detected during cooldown! Waking up immediately.")
                                        break
                            if key_updated:
                                break
                        except Exception:
                            pass

                    with self.state_lock:
                        if key_updated:
                            for m, s in self.model_states.items():
                                s["status"] = "ACTIVE"
                                s["ban_remaining_seconds"] = 0
                                s["429_count"] = 0
                        else:
                            self._tick_ban_timers_unlocked()
                            for m in recoverable_models:
                                if self.model_states.get(m, {}).get("ban_remaining_seconds", 0) <= 0:
                                    self.model_states[m]["status"] = "ACTIVE"
                                    self.model_states[m]["ban_remaining_seconds"] = 0

                    tried_models.clear()
                    attempts = 0  # Reset attempt counter so the new rotation pass executes fully
                    continue
                else:
                    logger.error(
                        f"🛑 [VANGUARD] Maximum 90s cooldown passes (2) exceeded for '{task_type}'. Exiting to local fallback.\n"
                        f"💡 Please update GEMINI_API_KEY in Credentials/.env"
                    )
                    break

            # 3. Prompt Degradation based on attempt intensity
            # Tier: high (0-3 attempts), mid (4-5), low (6+)
            # VISION TASKS (watermark/vision) preserve JSON schema on high tier across attempts.
            prompt_tier = "high"
            if task_type in ("watermark", "vision"):
                prompt_tier = "high"
            elif attempts >= 8:
                prompt_tier = "low"
            elif attempts >= 4:
                prompt_tier = "mid"

           

            active_prompt = self.simplify_prompt(prompt, prompt_tier)



            # 4. API Call with Internal Timeout (15s)

            self.stats["api_calls"] += 1

            call_start = time.monotonic()

           

            logger.info(f"🚀 [VANGUARD] [Module: '{module_name}'] Attempt {attempts+1}: Trying {current_model} ({prompt_tier} tier)...")
           
            # Langfuse Generation Start
            generation = None
            if trace:
                try:
                    generation = trace.generation(
                        name=f"attempt_{attempts+1}",
                        model=current_model,
                        input=active_prompt,
                        metadata={"prompt_tier": prompt_tier, "attempt": attempts+1, "module": module_name}
                    )
                except: pass

            try:

                # Configure API — supports modern google-genai with fallback to google-generativeai
                effective_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

                if not effective_key:
                    raise ValueError("GEMINI_API_KEY not set — add it to your .env file or user profile")

                # Dual SDK support
                if hasattr(genai, "Client"):
                    # Modern google.genai SDK
                    client = genai.Client(
                        api_key=effective_key,
                        http_options=types.HttpOptions(timeout=300_000) if (types and hasattr(types, "HttpOptions")) else None
                    )
                    config_params = {}
                    if gen_config:
                        config_params.update(gen_config)
                    if safety_settings:
                        config_params["safety_settings"] = safety_settings

                    response = client.models.generate_content(
                        model=current_model,
                        contents=active_prompt,
                        config=types.GenerateContentConfig(**config_params) if (types and hasattr(types, "GenerateContentConfig") and config_params) else (config_params or None)
                    )
                elif hasattr(genai, "GenerativeModel") or hasattr(genai, "configure"):
                    # Legacy google.generativeai SDK fallback
                    genai.configure(api_key=effective_key)
                    legacy_model = genai.GenerativeModel(
                        model_name=current_model,
                        generation_config=gen_config or None,
                        safety_settings=safety_settings or None
                    )
                    response = legacy_model.generate_content(active_prompt)
                else:
                    # Direct Google REST API fallback (Zero SDK dependencies needed)
                    import urllib.request
                    target_model = current_model.replace("models/", "")
                    rest_url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
                    
                    if isinstance(active_prompt, str):
                        parts = [{"text": active_prompt}]
                    elif isinstance(active_prompt, list):
                        parts = []
                        for item in active_prompt:
                            if isinstance(item, str):
                                parts.append({"text": item})
                            elif isinstance(item, dict):
                                parts.append(item)
                            else:
                                parts.append({"text": str(item)})
                    else:
                        parts = [{"text": str(active_prompt)}]
                        
                    payload = {"contents": [{"parts": parts}]}
                    if gen_config:
                        payload["generationConfig"] = gen_config
                    if safety_settings:
                        payload["safetySettings"] = safety_settings
                        
                    req_data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        rest_url,
                        data=req_data,
                        headers={"Content-Type": "application/json", "User-Agent": "AMTCE-Governor/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        
                    class _RESTCandidate:
                        def __init__(self, c_data):
                            self.raw = c_data
                            
                    class _RESTResponse:
                        def __init__(self, raw):
                            self.candidates = [_RESTCandidate(c) for c in raw.get("candidates", [])]
                            self._text = ""
                            if self.candidates:
                                raw_parts = raw["candidates"][0].get("content", {}).get("parts", [])
                                if raw_parts:
                                    self._text = "".join(p.get("text", "") for p in raw_parts)
                        @property
                        def text(self):
                            return self._text
                            
                    response = _RESTResponse(res_json)

               

                # 5. RESPONSE PARSING (Safety Aware)

                duration_ms = int((time.monotonic() - call_start) * 1000)

               

                try:

                    # Check for valid candidates (detects Safety Blocks)

                    if not response.candidates:

                         # No candidates found — usually a safety block or internal error

                         logger.warning(f"🛡️ [VANGUARD] Gemini blocked response for {current_model}. Reason: Possible Safety Filter.")

                         raise Exception("GEMINI_SAFETY_BLOCK")

                   

                    # Accessing .text is safer after candidate check

                    result = response.text

                except Exception as e_text:

                    err_t = str(e_text).lower()

                    if "safety" in err_t or "blocked" in err_t or "candidate" in err_t:

                        logger.warning(f"🛡️ [VANGUARD] Response censored by Gemini Safety Engine.")

                        raise Exception("GEMINI_SAFETY_BLOCK")

                    raise e_text # Re-raise if it's something else

               

                # SUCCESS Logic

                _reset_gemini_circuit()  # ← Beast Control: clear the breaker

               

                # Update Latency State

                with self.state_lock:

                    state = self.model_states[current_model]

                    state["success_count"] += 1

                    state["total_calls"] += 1

                    state["avg_latency"] = (state["avg_latency"] * 0.7) + ((duration_ms/1000.0) * 0.3)

                    state["last_used_at"] = time.monotonic()



                # Structured Success Log

                logger.info(json.dumps({

                    "vanguard_event": "success",

                    "caller_module": module_name,

                    "attempt": attempts + 1,

                    "model": current_model,

                    "latency_ms": duration_ms,

                    "prompt_tier": prompt_tier,

                    "total_request_time_ms": int((time.time() - start_time) * 1000)

                }))



                self.cache[cache_key] = result
                
                if generation:
                    try:
                        generation.end(output=result, metadata={"latency_ms": duration_ms})
                    except: pass

                return result



            except Exception as e:

                err_msg = str(e).lower()

                call_latency = time.monotonic() - call_start

               

                # ── SAFETY BLOCK ROTATION ────────────────────────────────────

                if "gemini_safety_block" in err_msg:

                    logger.warning(f"🛡️ [VANGUARD] Safety block encountered on {current_model}. Rotating...")

                    self.mark_model_banned(current_model, "safety")

                    tried_models.add(current_model)

                    attempts += 1

                    continue



                # ── 429 CIRCUIT BREAKER (TRUE ROTATION) ───────────────────────

                if "429" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg:

                    logger.warning(f"⚠️ [VANGUARD] Quota hit on {current_model}. Banning and rotating to next model...")

                    self.mark_model_banned(current_model, "429", error_message=err_msg)

                    tried_models.add(current_model)

                    attempts += 1

                    # Exact 2.6s live rotation gap (Danger 3 guard: triggers only on actual HTTP failure)
                    time.sleep(2.6)

                    continue # ← try next model

                # ────────────────────────────────────────────────────────────



                # ── FATAL AUTHENTICATION ERROR ─────────────────────────────

                if "api key expired" in err_msg or "api_key_invalid" in err_msg or "api key not valid" in err_msg:

                    logger.error(f"🛑 [VANGUARD] FATAL: API Key is invalid or expired: {e}")

                    break # A bad API key means NO model will work.

                # ────────────────────────────────────────────────────────────



                err_lower = err_msg.lower()
                if "404" in err_lower or "not_found" in err_lower or "not found" in err_lower or "deprecated" in err_lower:
                    error_type = "404"
                elif "403" in err_lower or "permission_denied" in err_lower or "denied access" in err_lower or "permission denied" in err_lower:
                    error_type = "403"
                elif any(code in err_lower for code in ["503", "500", "502", "504", "server error", "internal error", "service unavailable", "bad gateway"]):
                    error_type = "5xx"
                elif "timeout" in err_lower or "deadline" in err_lower or "timed out" in err_lower:
                    error_type = "timeout"
                elif "safety" in err_lower or "blocked" in err_lower:
                    error_type = "safety"
                elif "400" in err_lower or "invalid_argument" in err_lower or "invalid argument" in err_lower:
                    error_type = "404"
                else:
                    error_type = "client_error"

                logger.error(f"❌ [VANGUARD] Call failed on model '{current_model}': {e} (Classified as: {error_type})")

                self.mark_model_banned(current_model, error_type=error_type, error_message=err_msg)

                tried_models.add(current_model)

                attempts += 1

                self.stats["failures"] += 1

                if generation:
                    try:
                        generation.end(status_message=err_msg, level="ERROR", metadata={"error_type": error_type})
                    except: pass



                # ── BEAST CONTROL: Record 5xx to global circuit breaker ────
                if error_type == "5xx":
                    _record_gemini_5xx(call_latency, threshold=circuit_threshold)
                    if is_gemini_globally_down():
                        logger.warning(
                            f"🛑 [CIRCUIT] Breaker tripped mid-loop after {attempts} attempts "
                            f"({_GLOBAL_GEMINI_STATE['consecutive_5xx']} 5xx errors >= {circuit_threshold}). Exiting retry loop."
                        )
                        break
                # ────────────────────────────────────────────────────────────

                # ── MODEL ROTATION COOLDOWN ───────────────────
                wait = 2.6
                logger.info(f"⏳ [ROTATION GAP] Waiting {wait:.1f}s before next model attempt (attempt {attempts})...")
                time.sleep(wait)



                logger.warning(json.dumps({

                    "vanguard_event": "failure",

                    "attempt": attempts,

                    "model": current_model,

                    "reason": error_type,

                    "prompt_tier": prompt_tier

                }))



        # --- FALLBACK LAYER (If all Gemini fail or Deadline hit) ---

        logger.warning(f"⚠️ [VANGUARD] All attempts failed or deadline hit. Falling back to Ollama.")

       

        # Use Ultra-Simple prompt for local fallback

        final_fallback_prompt = self.simplify_prompt(prompt, "low")

        result = self._call_ollama(final_fallback_prompt)

       

        if result:

            self.cache[cache_key] = result

            return result

       

        # [mkpv-fix] Final Silent Fallback: Prevent error leaks in public captions/scripts
        # Note: 'creative', 'watermark', 'vision' expect JSON and must return None so callers trigger structured defaults cleanly
        if task_type in ("caption", "narrative"):
             logger.info(f"🛡️ [VANGUARD] ABSOLUTE FALLBACK triggered for '{task_type}'. Returning clean default.")
             return "Trending Spotlight"

       

        return None



    def _log_decision(self, model, score, task, reason):

        log_entry = {

            "timestamp": datetime.now().isoformat(),

            "selected_model": model,

            "score": round(score, 3) if score else 0,

            "task_type": task,

            "reason": reason,

            "rate_limited": not self.can_make_request()

        }

        self.routing_logger.info(json.dumps(log_entry))



    def embed(self, text, model_name="models/gemini-embedding-001", module_name="unknown"):
        # Re-enabled: Routing embedding calls through the governor for API key abstraction
        try:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("🔑 [GOVERNOR] GEMINI_API_KEY not set — skipping embed()")
                return []

            if hasattr(genai, "Client"):
                client = genai.Client(api_key=api_key)
                # Try the standard gemini-embedding-001 first
                try:
                    result = client.models.embed_content(
                        model=model_name,
                        contents=text
                    )
                    if hasattr(result, 'embeddings') and result.embeddings:
                        # Depending on SDK version, it might be result.embeddings[0].values or similar
                        emb = result.embeddings[0]
                        return emb.values if hasattr(emb, 'values') else emb
                    return result.get('embedding', [])
                except Exception as inner_e:
                    # Fallback to gemini-embedding-001 if requested model fails (v1beta SDK differences)
                    result = client.models.embed_content(
                        model="models/gemini-embedding-001",
                        contents=text
                    )
                    if hasattr(result, 'embeddings') and result.embeddings:
                        emb = result.embeddings[0]
                        return emb.values if hasattr(emb, 'values') else emb
                    return result.get('embedding', [])
            elif hasattr(genai, "embed_content") or hasattr(genai, "configure"):
                genai.configure(api_key=api_key)
                target_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
                try:
                    res = genai.embed_content(
                        model=target_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    return res.get('embedding', [])
                except Exception:
                    res = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    return res.get('embedding', [])
            else:
                # Direct REST API fallback for embed (Zero SDK dependency)
                import urllib.request
                target_model = model_name.replace("models/", "")
                rest_url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:embedContent?key={api_key}"
                payload = {"content": {"parts": [{"text": str(text)}]}}
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(rest_url, data=req_data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                return res_json.get("embedding", {}).get("values", [])
                
        except Exception as e:
            logger.warning(f"⚠️ [VANGUARD] Embedding generation failed: {e}")
            return []



    def print_usage_report(self):

        print("\n" + "═" * 40)

        print("📊 [GEMINI_ELITE_ROUTER_V3.5]")

        print(f"  total_api_calls={self.stats['api_calls']}")

        print(f"  logic_requests={self.stats['logical_requests']}")

        print(f"  cache_hits={self.stats['cache_hits']}")

        print(f"  failures={self.stats['failures']}")

       

        print("\nModel Health:")

        with self.state_lock:

            for m, state in self.model_states.items():

                status = state["status"]

                if status == "BANNED":

                    rem = state.get("ban_remaining_seconds", 0)

                    status += f" ({int(rem)}s remaining)"

                print(f"  - {m:25}: {status:10} | Success: {int(state['success_count']):3} | Latency: {state['avg_latency']:.2f}s")

        print("═" * 40 + "\n")





# Global Singleton

gemini_router = GeminiGovernor()