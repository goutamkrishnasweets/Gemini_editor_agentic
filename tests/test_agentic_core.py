"""
tests/test_agentic_core.py
==========================
Unit & Integration tests for the Unified Agentic Core and Re-edit Tooling.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Agentic_Core.tool_adapters import (
    get_agent_tools,
    TOOL_DISPATCH_MAP,
    tool_get_active_sessions,
    tool_reedit_session,
)
from Agentic_Core.director import AutonomousDirector, DIRECTOR_SYSTEM_INSTRUCTION


def test_agent_tools_registry():
    """Verify all 6 tools are registered in TOOL_DISPATCH_MAP and Google GenAI schemas."""
    tools = get_agent_tools()
    assert len(tools) == 1
    decls = tools[0].function_declarations
    tool_names = [d.name for d in decls]

    expected = [
        "tool_ingest_source",
        "tool_render_edit",
        "tool_audit_clip",
        "tool_publish_clip",
        "tool_get_active_sessions",
        "tool_reedit_session",
    ]
    for exp in expected:
        assert exp in tool_names, f"Missing declaration for {exp}"
        assert exp in TOOL_DISPATCH_MAP, f"Missing dispatch handler for {exp}"


def test_tool_get_active_sessions():
    """Verify tool_get_active_sessions returns list of sessions cleanly."""
    res = tool_get_active_sessions(limit=3)
    assert res.get("status") == "success"
    assert "sessions" in res
    assert isinstance(res["sessions"], list)


def test_tool_reedit_session_missing_session():
    """Verify tool_reedit_session cleanly handles non-existent sessions."""
    res = tool_reedit_session("sess_non_existent_12345", edit_directive="faster cuts")
    assert res.get("status") == "error"
    assert "not found" in res.get("message", "").lower()


def test_director_system_instruction():
    """Verify director system instruction covers session re-editing."""
    assert "tool_reedit_session" in DIRECTOR_SYSTEM_INSTRUCTION
    assert "tool_get_active_sessions" in DIRECTOR_SYSTEM_INSTRUCTION
    assert "Re-Editing / Revision / Retries" in DIRECTOR_SYSTEM_INSTRUCTION


def test_dispatch_agent_goal_routing():
    """Verify _dispatch_agent_goal accepts callback query updates without crashing."""
    import asyncio
    from main import _dispatch_agent_goal

    mock_msg = MagicMock()
    mock_msg.chat_id = 123456789
    mock_msg.reply_text = AsyncMock(return_value=MagicMock(edit_text=AsyncMock()))

    mock_query = MagicMock()
    mock_query.message = mock_msg
    mock_query.from_user.id = 123456789
    mock_query.answer = AsyncMock()

    mock_update = MagicMock()
    mock_update.message = None
    mock_update.callback_query = mock_query

    mock_context = MagicMock()
    mock_context.bot.send_message = AsyncMock()

    with patch("Agentic_Core.run_agent.run_agentic_goal_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "status": "success",
            "final_summary": "Test re-edit complete",
            "turns_executed": 2,
            "execution_time_sec": 1.5,
            "rendered_video_path": None,
            "session_id": "sess_test_1"
        }

        async def _test_coro():
            await _dispatch_agent_goal(mock_update, "Re-edit session 'sess_test_1' with faster cuts", context=mock_context)
            mock_query.answer.assert_called_once()

        asyncio.run(_test_coro())


def test_director_resolves_model_from_governor():
    """Verify director queries Governor dynamically for available reasoning model."""
    with patch("Agentic_Core.director.gemini_router") as mock_router:
        mock_router.get_available_model.return_value = "gemini-test-dynamic-model"
        director = AutonomousDirector(api_key="test_key_abc")
        assert director.model_name == "gemini-test-dynamic-model"
        mock_router.get_available_model.assert_called_once_with(task_type="reasoning")


def test_director_rotates_on_model_gone_404():
    """Verify that on 404 NOT_FOUND, director delegates ban to Governor and rotates models."""
    with patch("Agentic_Core.director.gemini_router") as mock_router:
        mock_router.get_available_model.side_effect = ["gemini-failing-model", "gemini-healthy-model"]
        director = AutonomousDirector(api_key="test_key_abc")
        director.client = MagicMock()

        mock_chat1 = MagicMock()
        mock_chat1.send_message.side_effect = Exception("404 NOT_FOUND. Model models/gemini-failing-model is no longer available.")

        mock_chat2 = MagicMock()
        mock_response = MagicMock()
        mock_response.function_calls = None
        mock_response.text = "Goal achieved with healthy model."
        mock_chat2.send_message.return_value = mock_response

        director.client.chats.create.side_effect = [mock_chat1, mock_chat2]

        result = director.run_goal(goal="Process clip", max_turns=2)

        assert result["status"] == "success"
        assert director.model_name == "gemini-healthy-model"
        mock_router.mark_model_banned.assert_called_with("gemini-failing-model", error_type="404")


def test_director_rotates_on_quota_429():
    """Verify that on 429 quota exhaustion, director delegates ban to Governor and rotates models."""
    with patch("Agentic_Core.director.gemini_router") as mock_router:
        mock_router.get_available_model.side_effect = ["gemini-quota-model", "gemini-alternate-model"]
        director = AutonomousDirector(api_key="test_key_abc")
        director.client = MagicMock()

        mock_chat1 = MagicMock()
        mock_chat1.send_message.side_effect = Exception("429 Resource has been exhausted (quota exceeded).")

        mock_chat2 = MagicMock()
        mock_response = MagicMock()
        mock_response.function_calls = None
        mock_response.text = "Goal achieved on alternate model."
        mock_chat2.send_message.return_value = mock_response

        director.client.chats.create.side_effect = [mock_chat1, mock_chat2]

        result = director.run_goal(goal="Process clip", max_turns=2)

        assert result["status"] == "success"
        assert director.model_name == "gemini-alternate-model"
        mock_router.mark_model_banned.assert_called_with("gemini-quota-model", error_type="429")




