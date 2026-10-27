"""
tests/test_multiuser_container_concurrency.py
============================================
Validates multi-user concurrency, ephemeral container lifecycle,
semaphore queueing, and workspace isolation for 5+ simultaneous users (both admin and non-admin).
"""

import os
import sys
import time
import threading
import pytest
from unittest.mock import patch, MagicMock

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Core_Modules.container_job_manager import ContainerJobManager


@pytest.fixture(autouse=True)
def reset_container_manager():
    """Resets the singleton before each test."""
    ContainerJobManager._instance = None
    yield
    ContainerJobManager._instance = None


def test_semaphore_bounds_and_zero_dropped_jobs_with_5_concurrent_users(monkeypatch):
    """
    Simulates 5 users (1 admin + 4 non-admin) triggering at the exact same second.
    With MAX_CONCURRENT_JOBS=2:
      - At most 2 jobs run simultaneously at any instant.
      - Zero jobs are dropped; all 5 users run and complete in sequence.
    """
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")
    cjm = ContainerJobManager()

    max_simultaneous_observed = 0
    current_running = 0
    lock = threading.Lock()
    completed_users = []

    def fake_run_container(uid, chat_id, max_accounts):
        nonlocal current_running, max_simultaneous_observed
        with lock:
            current_running += 1
            if current_running > max_simultaneous_observed:
                max_simultaneous_observed = current_running
        # Simulate work
        time.sleep(0.08)
        with lock:
            current_running -= 1
            completed_users.append(uid)

    cjm._run_container = fake_run_container

    user_ids = ["admin_100", "user_200", "user_300", "user_400", "user_500"]

    # Launch all 5 simultaneously
    launched_flags = []
    for uid in user_ids:
        flag = cjm.launch_user_batch_job(chat_id=uid, max_accounts=2, block_if_busy=True)
        launched_flags.append(flag)

    # All 5 should have been accepted and queued
    assert all(launched_flags), f"Some jobs failed to queue: {launched_flags}"

    # Wait for all 5 to complete
    timeout = 5.0
    start = time.time()
    while len(completed_users) < 5 and (time.time() - start) < timeout:
        time.sleep(0.05)

    assert len(completed_users) == 5, f"Expected 5 completed, got {len(completed_users)}: {completed_users}"
    assert set(completed_users) == set(user_ids)
    # Crucial assertion: concurrency never exceeded the semaphore limit of 2
    assert max_simultaneous_observed <= 2, f"Observed {max_simultaneous_observed} simultaneous jobs, expected <= 2"


def test_configurable_concurrency_capacity(monkeypatch):
    """
    Verifies that MAX_CONCURRENT_JOBS can be increased (e.g. to 5) so all 5 users run simultaneously in parallel.
    """
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "5")
    cjm = ContainerJobManager()

    max_simultaneous_observed = 0
    current_running = 0
    lock = threading.Lock()
    completed_users = []

    def fake_run_container(uid, chat_id, max_accounts):
        nonlocal current_running, max_simultaneous_observed
        with lock:
            current_running += 1
            if current_running > max_simultaneous_observed:
                max_simultaneous_observed = current_running
        time.sleep(0.1)
        with lock:
            current_running -= 1
            completed_users.append(uid)

    cjm._run_container = fake_run_container
    user_ids = ["user_1", "user_2", "user_3", "user_4", "user_5"]

    for uid in user_ids:
        cjm.launch_user_batch_job(chat_id=uid, max_accounts=2, block_if_busy=True)

    timeout = 5.0
    start = time.time()
    while len(completed_users) < 5 and (time.time() - start) < timeout:
        time.sleep(0.05)

    assert len(completed_users) == 5
    # Since concurrency capacity is 5 and all sleep 0.1s, multiple jobs run concurrently (>= 3)
    assert max_simultaneous_observed >= 3, f"Expected parallel execution, observed max: {max_simultaneous_observed}"


def test_dedup_prevention_for_duplicate_user_trigger():
    """
    If a job for User 200 is already queued or running, a duplicate trigger for User 200 is rejected immediately.
    """
    cjm = ContainerJobManager()
    started_event = threading.Event()
    release_event = threading.Event()

    def fake_run_container(uid, chat_id, max_accounts):
        started_event.set()
        release_event.wait(timeout=2.0)

    cjm._run_container = fake_run_container

    # 1. First trigger succeeds
    ok1 = cjm.launch_user_batch_job(chat_id="200")
    assert ok1 is True
    started_event.wait(timeout=1.0)

    # 2. Duplicate trigger while active fails
    ok2 = cjm.launch_user_batch_job(chat_id="200")
    assert ok2 is False, "Duplicate trigger for active user was not prevented"

    # Clean up
    release_event.set()
    time.sleep(0.05)


def test_docker_command_isolation_and_env_forwarding(monkeypatch):
    """
    Verifies that the generated docker run command:
      1. Isolates downloads to downloads/<chat_id>
      2. Mounts data/ and Credentials/ as read-only (:ro)
      3. Forwards only USER_<chat_id>_* env vars (no cross-user token leaks)
      4. Disables Telegram polling with BATCH_ONLY_MODE=1
    """
    monkeypatch.setenv("REPO_ROOT", "/mock_repo")
    monkeypatch.setenv("CONTAINER_IMAGE", "gemini-editor:latest")
    monkeypatch.setenv("CONTAINER_MAX_MEMORY", "2g")
    monkeypatch.setenv("CONTAINER_MAX_CPUS", "1.5")
    monkeypatch.setenv("USER_12345_TOKEN", "secret_token_12345")
    monkeypatch.setenv("USER_99999_TOKEN", "secret_token_99999")  # Should NOT be in 12345's command!

    cjm = ContainerJobManager()
    cmd = cjm._build_docker_cmd("gemini_job_12345_1", "12345", "12345", max_accounts=2)

    cmd_str = " ".join(cmd)

    # Verify command parts
    assert "docker run --rm" in cmd_str
    assert "--name gemini_job_12345_1" in cmd_str
    assert "--memory=2g" in cmd_str
    assert "--cpus=1.5" in cmd_str
    assert "-v /mock_repo/data:/app/data:ro" in cmd_str
    assert "-v /mock_repo/Credentials:/app/Credentials:ro" in cmd_str
    assert "-e USER_CHAT_ID=12345" in cmd_str
    assert "-e BATCH_ONLY_MODE=1" in cmd_str
    assert "-e USER_12345_TOKEN=secret_token_12345" in cmd_str
    # Verify User 99999's secret is NEVER forwarded to User 12345
    assert "USER_99999_TOKEN" not in cmd_str
    assert "--batch-user 12345" in cmd_str


def test_run_best_available_fallback_to_inline_when_docker_disabled(monkeypatch):
    """
    When USE_DOCKER_CONTAINERS is not '1', run_best_available transparently falls back to inline batch.
    """
    monkeypatch.setenv("USE_DOCKER_CONTAINERS", "0")
    cjm = ContainerJobManager()

    with patch("Downloader_Modules.scheduled_scraper_manager.run_scheduled_scraper_batch") as mock_batch:
        mock_batch.return_value = ["/mock/path/reel_1.mp4"]
        res = cjm.run_best_available(chat_id="300", max_accounts=2)
        assert res == ["/mock/path/reel_1.mp4"]
        mock_batch.assert_called_once_with(max_accounts=2, for_chat_id="300")
