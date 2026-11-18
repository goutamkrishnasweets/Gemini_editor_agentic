"""Intelligence_Modules — News/Trend Intel & Zero-Download Metadata Scouting."""
from Intelligence_Modules.news_trend_fetcher import (
    NewsTrendFetcher,
    fetch_news_for_target,
    scout_candidate_videos_metadata,
)

__all__ = [
    "NewsTrendFetcher",
    "fetch_news_for_target",
    "scout_candidate_videos_metadata",
]
