import datetime
import pytest


def check_ist_execution_window(total_mins: int) -> tuple[bool, str]:
    """
    Simulates the CI window guard logic:
    - Window 1: Morning 04:20 AM (260m) - 08:20 AM (500m) IST
    - Window 2: Evening 04:20 PM (980m) - 08:20 PM (1220m) IST
    - Night quiet zone: 08:22 PM (1222m) to 04:15 AM (255m)
    - Midday quiet zone: 08:22 AM (502m) to 04:15 PM (975m)
    """
    is_night = (total_mins >= 1222 or total_mins < 255)
    is_midday = (502 <= total_mins < 975)
    
    if is_night:
        return False, "NIGHT_QUIET_ZONE"
    elif is_midday:
        return False, "MIDDAY_QUIET_ZONE"
    else:
        return True, "ACTIVE_ALLOWED_WINDOW"


def test_night_hours_blocked():
    # 01:00 AM IST (60 mins)
    allowed, reason = check_ist_execution_window(60)
    assert not allowed
    assert reason == "NIGHT_QUIET_ZONE"

    # 02:30 AM IST (150 mins)
    allowed, reason = check_ist_execution_window(150)
    assert not allowed
    assert reason == "NIGHT_QUIET_ZONE"

    # 11:30 PM IST (23 * 60 + 30 = 1410 mins)
    allowed, reason = check_ist_execution_window(1410)
    assert not allowed
    assert reason == "NIGHT_QUIET_ZONE"


def test_midday_gap_blocked():
    # 10:00 AM IST (600 mins)
    allowed, reason = check_ist_execution_window(600)
    assert not allowed
    assert reason == "MIDDAY_QUIET_ZONE"

    # 01:00 PM IST (780 mins)
    allowed, reason = check_ist_execution_window(780)
    assert not allowed
    assert reason == "MIDDAY_QUIET_ZONE"


def test_active_scheduled_windows_allowed():
    # Morning trigger: 04:20 AM IST (260 mins)
    allowed, reason = check_ist_execution_window(260)
    assert allowed
    assert reason == "ACTIVE_ALLOWED_WINDOW"

    # Morning mid-run: 06:30 AM IST (390 mins)
    allowed, reason = check_ist_execution_window(390)
    assert allowed
    assert reason == "ACTIVE_ALLOWED_WINDOW"

    # Evening trigger: 04:20 PM IST (980 mins)
    allowed, reason = check_ist_execution_window(980)
    assert allowed
    assert reason == "ACTIVE_ALLOWED_WINDOW"

    # Evening mid-run: 07:00 PM IST (1140 mins)
    allowed, reason = check_ist_execution_window(1140)
    assert allowed
    assert reason == "ACTIVE_ALLOWED_WINDOW"
