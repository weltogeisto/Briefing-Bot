import json
from pathlib import Path

from src import source_collection as sc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def load_config():
    with (SRC / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_rss_candidates_normalize_to_stable_records():
    cfg = load_config()
    source = {
        "id": "sachsen-rss",
        "label": "Sachsen News",
        "type": "rss",
        "tier": "official",
        "url": "https://www.sachsen.de/rss.xml",
    }
    xml = """<?xml version="1.0"?>
<rss><channel><item>
  <title>Neue CIO Leitung fuer Sachsen digital ernannt</title>
  <link>https://www.sachsen.de/news/cio-leitung</link>
  <description>Sachsen.digital baut digitale Verwaltung weiter aus.</description>
  <pubDate>2026-04-28</pubDate>
</item></channel></rss>"""

    candidates = sc.parse_rss_candidates(xml, source, cfg, "2026-04-28T06:30:00Z")

    assert candidates == [
        {
            "title": "Neue CIO Leitung fuer Sachsen digital ernannt",
            "url": "https://www.sachsen.de/news/cio-leitung",
            "domain": "sachsen.de",
            "source_id": "sachsen-rss",
            "source_label": "Sachsen News",
            "source_type": "rss",
            "source_tier": "official",
            "published_at": "2026-04-28",
            "snippet": "Sachsen.digital baut digitale Verwaltung weiter aus.",
            "account_matches": ["Freistaat Sachsen"],
            "theme_matches": [],
            "leadership_hits": ["cio", "leitung", "ernannt"],
            "procurement_hits": [],
            "regulatory_hits": [],
            "collected_at": "2026-04-28T06:30:00Z",
        }
    ]


def test_ranking_prefers_leadership_account_trigger_over_generic_regulation():
    cfg = load_config()
    leadership = {
        "title": "Neue CIO Leitung fuer Sachsen digital ernannt",
        "url": "https://www.sachsen.de/news/cio-leitung",
        "source_tier": "official",
        "account_matches": ["Freistaat Sachsen"],
        "theme_matches": [],
        "leadership_hits": ["cio", "leitung"],
        "procurement_hits": [],
        "regulatory_hits": [],
    }
    regulation = {
        "title": "EU AI Act Frist naehert sich",
        "url": "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
        "source_tier": "official",
        "account_matches": [],
        "theme_matches": ["AI Governance & Compliance"],
        "leadership_hits": [],
        "procurement_hits": [],
        "regulatory_hits": ["ai act"],
    }

    ranked = sc.rank_candidates([regulation, leadership], cfg)

    assert ranked[0]["title"] == leadership["title"]
    assert ranked[0]["score_reasons"][:3] == ["leadership", "account_match", "official_source"]
    assert ranked[1]["is_regulatory_only"] is True


def test_collect_source_candidates_deduplicates_and_tracks_rejections():
    cfg = load_config()
    cfg["sources"] = [
        {
            "id": "feed",
            "label": "Feed",
            "type": "rss",
            "tier": "official",
            "url": "https://example.gov/feed.xml",
            "enabled": True,
        }
    ]
    xml = """<rss><channel>
<item><title>Neue CIO Leitung fuer Sachsen</title><link>https://example.gov/a</link></item>
<item><title>Neue CIO Leitung fuer Sachsen</title><link>https://example.gov/a</link></item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(
        cfg,
        fetcher=lambda url: xml,
        collected_at="2026-04-28T06:30:00Z",
    )

    assert len(candidates) == 1
    assert candidates[0]["url"] == "https://example.gov/a"
    assert rejected[0]["rejection_reason"] == "duplicate_or_missing_url"
