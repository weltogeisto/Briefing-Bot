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
            "event_types": ["role_change"],
            "admin_it_action_hits": ["digitale verwaltung"],
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
<item><title>Neue CIO Leitung fuer Freistaat Sachsen ernannt</title><link>https://example.gov/a</link></item>
<item><title>Neue CIO Leitung fuer Freistaat Sachsen ernannt</title><link>https://example.gov/a</link></item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(
        cfg,
        fetcher=lambda url: xml,
        collected_at="2026-04-28T06:30:00Z",
    )

    assert len(candidates) == 1
    assert candidates[0]["url"] == "https://example.gov/a"
    assert rejected[0]["rejection_reason"] == "duplicate_or_missing_url"


def test_rss_pubdate_outside_lookback_is_rejected_before_ranking():
    cfg = load_config()
    cfg["lookback_hours"] = 72
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
<item>
  <title>Neue Cloud Plattform fuer Verwaltung startet</title>
  <link>https://example.gov/old-cloud</link>
  <description>Verwaltungsdigitalisierung und Plattformbetrieb fuer Behoerden.</description>
  <pubDate>Fri, 01 May 2026 10:00:48 +0200</pubDate>
</item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(
        cfg,
        fetcher=lambda url: xml,
        collected_at="2026-05-17T08:00:00Z",
    )

    assert candidates == []
    assert any(item.get("rejection_reason") == "stale_outside_lookback" for item in rejected)


def test_account_matching_does_not_map_sachsen_anhalt_to_freistaat_sachsen():
    cfg = load_config()
    source = {
        "id": "sachsen-anhalt-rss",
        "label": "Sachsen-Anhalt RSS",
        "type": "rss",
        "tier": "official",
        "url": "https://www.sachsen-anhalt.de/rss.xml",
    }
    xml = """<rss><channel><item>
  <title>Sachsen-Anhalt startet neue Cloud Plattform</title>
  <link>https://www.sachsen-anhalt.de/cloud</link>
  <description>Digitalisierung der Verwaltung in Sachsen-Anhalt.</description>
  <pubDate>Sun, 17 May 2026 07:00:00 +0200</pubDate>
</item></channel></rss>"""

    candidates = sc.parse_rss_candidates(xml, source, cfg, "2026-05-17T08:00:00Z")

    assert candidates[0]["account_matches"] == ["Sachsen-Anhalt"]


def test_generic_minister_mention_is_not_a_leadership_trigger():
    cfg = load_config()
    candidate = sc.make_candidate(
        source={"id": "feed", "type": "rss", "tier": "official", "url": "https://example.gov/rss.xml"},
        title="Minister besucht Digitalmesse in Sachsen-Anhalt",
        url="https://example.gov/minister-messe",
        snippet="Der Minister spricht ueber Digitalisierung, aber es gibt keine neue Rolle oder Entscheidung.",
        published_at="Sun, 17 May 2026 07:00:00 +0200",
        cfg=cfg,
        collected_at="2026-05-17T08:00:00Z",
    )
    scored = sc.score_candidate(candidate, cfg)

    assert "minister" not in [hit.casefold() for hit in scored["leadership_hits"]]
    assert "leadership" not in scored["score_reasons"]


def test_theme_only_digitalization_without_material_event_is_rejected():
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
<item>
  <title>Sachsen-Anhalt informiert ueber Digitalisierung der Verwaltung</title>
  <link>https://example.gov/digitalisierung</link>
  <description>Allgemeiner Rueckblick ohne konkrete neue Entscheidung.</description>
  <pubDate>Sun, 17 May 2026 07:00:00 +0200</pubDate>
</item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(
        cfg,
        fetcher=lambda url: xml,
        collected_at="2026-05-17T08:00:00Z",
    )

    assert candidates == []
    assert any(item.get("rejection_reason") == "below_relevance_threshold" for item in rejected)
