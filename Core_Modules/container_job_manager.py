"""
Core_Modules / container_job_manager.py
========================================
Ephemeral per-user Docker job launcher.

Architecture:
  - Host bot (main.py) runs 24/7 as Telegram listener.
  - When a scheduled slot fires for a user, the scheduler calls
    ContainerJobManager.launch_user_batch_job(chat_id) instead of
    running the batch inline.
  - A Docker container is spawned: isolated workspace, read-only
    shared data/, no Telegram polling conflict, auto-destroyed on exit.
  - A threading.Semaphore caps concurrent containers so a burst of
    simultaneous slots cannot OOM-kill the VPS.

Environment variables consumed:
  CONTAINER_IMAGE        Docker image name (default: gemini-editor:latest)
  CONTAINER_MAX_MEMORY   Hard memory cap per container (default: 2g)
  CONTAINER_MAX_CPUS     CPU quota per container (default: 1.5)
  MAX_CONCURRENT_JOBS    How many containers can run simultaneously (default: 2)
  REPO_ROOT              Absolute repo path (auto-detected if unset)
  USE_DOCKER_CONTAINERS  Set to "1" to enable Docker mode.
                         Any other value (or unset) → run batch inline
                         (safe fallback for GitHub Actions or local dev).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger("container_job_manager")

_REPO_ROOT = os.getenv("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_IMAGE = "gemini-editor:latest"
_DEFAULT_MEMORY = "2g"
_DEFAULT_CPUS = "1.5"
_DEFAULT_MAX_CONCURRENT = 2


class ContainerJobManager:
    """
    Manages per-user ephemeral Docker batch jobs.
    Thread-safe. Singleton pattern — use ContainerJobManager.instance().
    """

    _instance: Optional["ContainerJobManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        max_c = int(os.getenv("MAX_CONCURRENT_JOBS", str(_DEFAULT_MAX_CONCURRENT)))
        self._semaphore = threading.BoundedSemaphore(max_c)
        self._max_concurrent = max_c
        self._active_jobs: Dict[str, str] = {}  # {chat_id: container_name}
        self._queued_jobs: set[str] = set()    # {chat_id}
        self._jobs_lock = threading.RLock()

    @classmethod
    def instance(cls) -> "ContainerJobManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ───────────────────────────────────────────────────────────

    def is_docker_mode(self) -> bool:
        """Returns True only when Docker mode is explicitly enabled via env."""
        return os.getenv("USE_DOCKER_CONTAINERS", "0").strip() == "1"

    def is_docker_available(self) -> bool:
        """Checks if the Docker daemon is reachable."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def is_job_running(self, chat_id: str) -> bool:
        """Returns True if an active container is already running for this user."""
        with self._jobs_lock:
            name = self._active_jobs.get(str(chat_id))
            if not name:
                return False
        # Verify container actually still alive via docker inspect
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", name],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def launch_user_batch_job(
        self,
        chat_id: Optional[str],
        max_accounts: int = 2,
        block_if_busy: bool = True
    ) -> bool:
        """
        Launches an isolated Docker container to run the batch pipeline for a
        specific user (identified by chat_id).

        When block_if_busy=True (default), requests are queued in dedicated worker
        threads that wait on the semaphore slot, guaranteeing NO user is dropped even
        if 5+ users trigger at the exact same second.

        Args:
            chat_id: Telegram chat_id string. None = system/admin pool run.
            max_accounts: Max accounts to scrape in this batch (default 2).
            block_if_busy: If True, queues the job until a concurrency slot opens.
                           If False, skips immediately if semaphore is full.
        """
        uid = str(chat_id) if chat_id else "system"

        with self._jobs_lock:
            # Dedup check: don't double-queue or double-run the exact same user
            if uid in self._queued_jobs or uid in self._active_jobs or self.is_job_running(uid):
                logger.warning(
                    "⏭️ [CONTAINER JOB] Job for user %s already queued or running — skipping duplicate launch.", uid
                )
                return False

            self._queued_jobs.add(uid)

        def _worker():
            try:
                if block_if_busy:
                    logger.info("⏳ [CONTAINER JOB] User %s queued. Waiting for semaphore slot...", uid)
                    self._semaphore.acquire(blocking=True)
                else:
                    acquired = self._semaphore.acquire(blocking=False)
                    if not acquired:
                        logger.warning(
                            "⚠️ [CONTAINER JOB] Max concurrent jobs (%d) reached. Skipping batch for user %s.",
                            self._max_concurrent, uid
                        )
                        return

                try:
                    with self._jobs_lock:
                        self._queued_jobs.discard(uid)
                        self._active_jobs[uid] = self._container_name(uid)
                    self._run_container(uid, chat_id, max_accounts)
                finally:
                    self._semaphore.release()
                    logger.info("🔓 [CONTAINER JOB] Semaphore released for user %s.", uid)
            except Exception as _we:
                logger.error("❌ [CONTAINER JOB] Error in worker for user %s: %s", uid, _we)
            finally:
                with self._jobs_lock:
                    self._queued_jobs.discard(uid)
                    self._active_jobs.pop(uid, None)

        t = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"ContainerJobWorker-{uid}"
        )
        t.start()
        return True

    def launch_inline_batch(
        self,
        chat_id: Optional[str],
        max_accounts: int = 2
    ) -> list:
        """
        Fallback: runs the batch inline (no Docker) using the existing
        run_scheduled_scraper_batch function. Used when Docker is unavailable
        or USE_DOCKER_CONTAINERS != "1".
        """
        from Downloader_Modules.scheduled_scraper_manager import run_scheduled_scraper_batch
        logger.info(
            "🔄 [INLINE BATCH] Running inline batch for user %s (max_accounts=%d)...",
            chat_id or "system", max_accounts
        )
        return run_scheduled_scraper_batch(max_accounts=max_accounts, for_chat_id=chat_id)

    def run_best_available(
        self,
        chat_id: Optional[str],
        max_accounts: int = 2,
    ) -> list:
        """
        Smart dispatcher:
          1. If USE_DOCKER_CONTAINERS=1 AND Docker daemon reachable → container mode.
          2. Otherwise → inline batch (safe for GitHub Actions, local dev, VPS without Docker).
        Returns the rendered reels list (only populated in inline mode;
        Docker mode is async, returns [] immediately).
        """
        if self.is_docker_mode() and self.is_docker_available():
            launched = self.launch_user_batch_job(chat_id=chat_id, max_accounts=max_accounts)
            if launched:
                logger.info(
                    "🐳 [CONTAINER JOB] Docker batch dispatched for user %s. Review card will arrive after container exits.",
                    chat_id or "system"
                )
            return []  # async — results delivered via Telegram when container finishes
        else:
            return self.launch_inline_batch(chat_id=chat_id, max_accounts=max_accounts)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _container_name(self, uid: str) -> str:
        return f"gemini_job_{uid}_{int(time.time())}"

    def _build_docker_cmd(
        self, container_name: str, uid: str, chat_id: Optional[str], max_accounts: int
    ) -> list:
        image = os.getenv("CONTAINER_IMAGE", _DEFAULT_IMAGE)
        memory = os.getenv("CONTAINER_MAX_MEMORY", _DEFAULT_MEMORY)
        cpus = os.getenv("CONTAINER_MAX_CPUS", _DEFAULT_CPUS)
        repo_root = (os.getenv("REPO_ROOT") or _REPO_ROOT).replace("\\", "/")

        downloads_dir = f"{repo_root}/downloads/{uid}"
        try:
            os.makedirs(downloads_dir, exist_ok=True)
        except Exception:
            pass

        # Collect all USER_<id>_* env vars to forward into the container
        user_env_args = []
        if chat_id:
            for k, v in os.environ.items():
                if k.startswith(f"USER_{chat_id}_") and v:
                    user_env_args += ["-e", f"{k}={v}"]

        # Check if user has personal API keys in telegram_users.json
        user_gemini = None
        user_apify = None
        if chat_id:
            try:
                from Telegram_Storage_Modules.telegram_user_manager import load_all_users
                all_u = load_all_users()
                u_info = all_u.get(str(chat_id), {})
                user_gemini = u_info.get("gemini_api_key")
                user_apify = u_info.get("apify_api_token")
            except Exception:
                pass

        gemini_key_to_pass = user_gemini or os.getenv("GEMINI_API_KEY", "")
        apify_token_to_pass = user_apify or os.getenv("APIFY_API_TOKEN", os.getenv("APIFY_TOKEN", ""))

        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            f"--memory={memory}",
            f"--cpus={cpus}",
            # Shared data — read-only to prevent concurrent JSON corruption
            "-v", f"{repo_root}/data:/app/data:ro",
            # Shared credentials — read-only
            "-v", f"{repo_root}/Credentials:/app/Credentials:ro",
            # User's personal downloads workspace — read-write, isolated by chat_id
            "-v", f"{downloads_dir}:/app/downloads/{uid}:rw",
            # Pass core env vars with user-specific quota overrides
            "-e", f"USER_CHAT_ID={chat_id or ''}",
            "-e", f"TELEGRAM_BOT_TOKEN={os.getenv('TELEGRAM_BOT_TOKEN', '')}",
            "-e", f"TELEGRAM_ADMIN_ID={os.getenv('TELEGRAM_ADMIN_ID', '')}",
            "-e", f"GEMINI_API_KEY={gemini_key_to_pass}",
            "-e", f"APIFY_API_TOKEN={apify_token_to_pass}",
            "-e", f"MAX_ACCOUNTS={max_accounts}",
            # Prevent the container from starting Telegram polling
            "-e", "BATCH_ONLY_MODE=1",
        ] + user_env_args + [
            image,
            "python", "-u", "main.py", "--batch-user", str(chat_id or "")
        ]
        return cmd

    def _run_container(self, uid: str, chat_id: Optional[str], max_accounts: int):
        """Internal: actually runs the docker container (blocking call in a thread)."""
        container_name = self._container_name(uid)
        with self._jobs_lock:
            self._active_jobs[uid] = container_name

        cmd = self._build_docker_cmd(container_name, uid, chat_id, max_accounts)
        logger.info(
            "🐳 [CONTAINER JOB] Starting container '%s' for user %s (max_accounts=%d)",
            container_name, uid, max_accounts
        )
        logger.debug("🐳 [CONTAINER JOB] Docker cmd: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=int(os.getenv("CONTAINER_TIMEOUT_SECONDS", "3600"))
            )
            if result.returncode == 0:
                logger.info(
                    "✅ [CONTAINER JOB] Container '%s' for user %s finished successfully.",
                    container_name, uid
                )
            else:
                logger.error(
                    "❌ [CONTAINER JOB] Container '%s' for user %s exited with code %d.",
                    container_name, uid, result.returncode
                )
        except subprocess.TimeoutExpired:
            logger.error(
                "⏱️ [CONTAINER JOB] Container '%s' timed out. Killing...", container_name
            )
            try:
                subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10)
            except Exception:
                pass
        except Exception as e:
            logger.error("❌ [CONTAINER JOB] Unexpected error running container '%s': %s", container_name, e)
        finally:
            with self._jobs_lock:
                self._active_jobs.pop(uid, None)

    def cleanup_orphaned_containers(self):
        """
        Kills and removes any dangling gemini_job_* containers that are no longer
        tracked in _active_jobs (e.g., survived a restart). Safe to call periodically.
        """
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=gemini_job_", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            live_names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
            with self._jobs_lock:
                tracked = set(self._active_jobs.values())
            orphans = [n for n in live_names if n not in tracked]
            for name in orphans:
                logger.warning("🧹 [CONTAINER JOB] Orphaned container found: %s — removing.", name)
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        except Exception as e:
            logger.debug("Notice during orphan cleanup: %s", e)
