from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.event_time import event_has_started, event_local_datetime, event_matches_local_date
from core.game_status import is_available_for_pregame, is_live_status


class EventTimeTests(unittest.TestCase):
    def test_utc_event_is_assigned_to_mexico_local_date(self) -> None:
        event = {"date": "2026-07-17T02:00:00Z"}
        local = event_local_datetime(event)
        self.assertEqual(local.strftime("%Y-%m-%d %H:%M"), "2026-07-16 20:00")
        self.assertTrue(event_matches_local_date(event, date(2026, 7, 16)))
        self.assertFalse(event_matches_local_date(event, date(2026, 7, 17)))

    def test_started_event_is_not_available_for_pregame(self) -> None:
        event = {"date": "2026-07-16T18:00:00-06:00"}
        now = datetime(2026, 7, 16, 18, 1, tzinfo=ZoneInfo("America/Mexico_City"))
        self.assertTrue(event_has_started(event, now))
        self.assertTrue(is_live_status({"short": "1H", "long": "First Half"}))
        self.assertFalse(is_available_for_pregame({"short": "1H", "long": "First Half"}))

    def test_future_event_remains_available(self) -> None:
        event = {"date": "2026-07-16", "time": "20:00"}
        now = datetime(2026, 7, 16, 19, 59, tzinfo=ZoneInfo("America/Mexico_City"))
        self.assertFalse(event_has_started(event, now))


if __name__ == "__main__":
    unittest.main()
