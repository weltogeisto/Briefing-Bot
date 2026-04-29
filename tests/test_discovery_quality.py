import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def load_run_briefing():
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_types_mod = types.ModuleType("google.genai.types")

    class DummyGenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyGoogleSearch:
        pass

    class DummyTool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    genai_types_mod.GenerateContentConfig = DummyGenerateContentConfig
    genai_types_mod.GoogleSearch = DummyGoogleSearch
    genai_types_mod.Tool = DummyTool
    genai_mod.Client = object
    genai_mod.types = genai_types_mod
    google_mod.genai = genai_mod

    markdown_mod = types.ModuleType("markdown")
    markdown_mod.markdown = lambda text, **kwargs: text

    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = genai_types_mod
    sys.modules["markdown"] = markdown_mod

    spec = importlib.util.spec_from_file_location("run_briefing", SRC / "run_briefing.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(name):
    with (SRC / name).open(encoding="utf-8") as f:
        return json.load(f)


def load_source_collection():
    spec = importlib.util.spec_from_file_location("source_collection", SRC / "source_collection.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_candidate():
    return {
        "title": "Neue CIO Leitung fuer Sachsen digital ernannt",
        "url": "https://www.sachsen.de/news/cio-leitung",
        "domain": "sachsen.de",
        "source_tier": "official",
        "account_matches": ["Freistaat Sachsen"],
        "theme_matches": [],
        "leadership_hits": ["cio", "leitung"],
        "procurement_hits": [],
        "regulatory_hits": [],
        "score": 29,
        "score_reasons": ["leadership", "account_match", "official_source"],
    }


def test_ci_runs_compile_and_tests_on_main_and_codex_branches():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in ci
    assert "push:" in ci
    assert "- main" in ci
    assert '- "codex/**"' in ci
    assert "python -m compileall src" in ci
    assert "python -m pytest -q" in ci


def test_json_config_files_are_valid_and_source_catalog_has_balanced_groups():
    config = load_json("config.json")
    universe = load_json("entity_universe.json")
    catalog = load_json("source_catalog.json")

    assert config["accounts"]
    assert universe["states"]

    group_ids = {group["id"] for group in catalog["groups"]}
    assert {
        "procurement",
        "official_accounts",
        "policy_regulatory",
        "cyber_ai_cloud_trade",
    }.issubset(group_ids)

    for group in catalog["groups"]:
        assert group["label"]
        assert group["search_guidance"]
        assert group["domains"]

    source_ids = {source["id"] for source in config["sources"]}
    rss_sources = [source for source in config["sources"] if source["type"] == "rss"]
    assert len(config["sources"]) >= 20
    assert len(rss_sources) >= 5
    assert {
        "sachsen-anhalt-ministerien-rss",
        "sachsen-anhalt-mid-rss",
        "kommune21-rss",
        "egovernment-rss",
        "behoerden-spiegel-rss",
        "public-manager-rss",
        "vitako-rss",
        "govdigital-rss",
    }.issubset(source_ids)
    assert any(source["tier"] == "trade" and source["type"] == "rss" for source in config["sources"])


def test_collector_rejects_official_items_without_material_relevance():
    sc = load_source_collection()
    config = load_json("config.json")
    source = {
        "id": "official-test",
        "label": "Official Test",
        "type": "rss",
        "tier": "official",
        "url": "https://example.gov/rss.xml",
        "enabled": True,
    }
    config["sources"] = [source]
    rss = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>Sommerfest im Ministerium</title>
    <link>https://example.gov/sommerfest</link>
    <description>Ein allgemeiner Termin ohne Digitalbezug.</description>
  </item>
  <item>
    <title>Neue Cloud Plattform fuer Verwaltung startet</title>
    <link>https://example.gov/cloud-plattform</link>
    <description>Verwaltungsdigitalisierung und Plattformbetrieb fuer Behoerden.</description>
  </item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(config, fetcher=lambda url: rss)

    assert [candidate["title"] for candidate in candidates] == ["Neue Cloud Plattform fuer Verwaltung startet"]
    assert any(item.get("rejection_reason") == "below_relevance_threshold" for item in rejected)


def test_collector_does_not_treat_beauftragte_as_procurement_auftrag():
    sc = load_source_collection()
    config = load_json("config.json")
    source = {
        "id": "official-test",
        "label": "Official Test",
        "type": "rss",
        "tier": "official",
        "url": "https://example.gov/rss.xml",
        "enabled": True,
    }
    config["sources"] = [source]
    rss = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>Sachsens Datenschutzbeauftragte stellt Jahresbericht vor</title>
    <link>https://example.gov/datenschutzbericht</link>
    <description>Bericht der Datenschutzbeauftragten.</description>
  </item>
</channel></rss>"""

    candidates, rejected = sc.collect_source_candidates(config, fetcher=lambda url: rss)

    assert candidates == []
    assert any(item.get("rejection_reason") == "below_relevance_threshold" for item in rejected)


def test_discovery_plan_uses_broader_spotlight_and_deduplicates_priority_accounts():
    rb = load_run_briefing()
    config = load_json("config.json")
    universe = load_json("entity_universe.json")
    catalog = load_json("source_catalog.json")

    config["entity_coverage"]["always_include"] = [
        "FITKO (IT-Planungsrat)",
        "BVA (Bundesverwaltungsamt)",
        "IT.Niedersachsen",
    ]
    config["entity_coverage"]["spotlight_entities_per_day"] = 8

    plan = rb.build_discovery_plan(config, universe, catalog, "2026-04-26")

    assert [account["name"] for account in config["accounts"]] == plan["priority_entities"]
    assert "FITKO (IT-Planungsrat)" not in plan["spotlight_entities"]
    assert "BVA (Bundesverwaltungsamt)" not in plan["spotlight_entities"]
    assert "IT.Niedersachsen" in plan["spotlight_entities"]
    assert len(plan["spotlight_entities"]) >= 8
    assert {group["id"] for group in plan["source_groups"]} >= {
        "procurement",
        "official_accounts",
        "policy_regulatory",
        "cyber_ai_cloud_trade",
    }


def test_prompt_includes_balanced_source_discovery_before_entity_searches(monkeypatch):
    rb = load_run_briefing()
    config = load_json("config.json")
    brevity = config["model"]["brevity"]

    monkeypatch.setattr(rb, "today_iso_utc", lambda: "2026-04-26")

    prompt = rb.build_prompt(config, brevity)

    source_index = prompt.index("BALANCED SOURCE DISCOVERY")
    entity_index = prompt.index("PRIORITY ENTITIES TO RESEARCH")
    assert source_index < entity_index
    assert "procurement" in prompt.lower()
    assert "official sources as factual authority" in prompt
    assert "trade sources as leads" in prompt
    assert "evergabe-online.de" in prompt
    assert "bund.de" in prompt
    system_instruction = rb.build_system_instruction(brevity)
    combined = f"{system_instruction}\n{prompt}"
    assert "citations are handled automatically via grounding metadata" not in combined
    assert "markdown link" in combined.lower()
    assert "No verified signals in this period" in combined


def test_quality_diagnostics_block_on_weak_evidence():
    rb = load_run_briefing()
    markdown = """PUBLIC SECTOR DAILY

⚡ TODAY'S TOP PRIORITY

## 1. VERIFIED HARD SIGNALS

| Element | Details |
|---------|---------|
| **Signal** | Placeholder-like [Entity] should be flagged. |
| **Source** | One weak source. |

## 2. REGULATORY COUNTDOWN: EU AI Act
## 3. STRATEGIC THEME: AI Governance
## 4. PRIORITIZED ACTION ITEMS
## 5. CALENDAR: KEY DATES AHEAD
## 6. GARTNER CAPABILITY QUICK REFERENCE
"""

    result = rb.evaluate_briefing_quality(markdown, {"min_links": 3, "min_domains": 2})

    assert result["should_warn"] is True
    assert result["should_block"] is True
    assert "unresolved_placeholders" in result["warning_codes"]
    assert "citation_threshold" in result["warning_codes"]
    assert "source_domain_count" in result


def test_pasted_bad_briefing_shape_blocks_delivery():
    rb = load_run_briefing()
    markdown = """PUBLIC SECTOR DAILY

TODAY'S TOP PRIORITY

## 1. VERIFIED HARD SIGNALS

| Element | Details |
|---------|---------|
| **Signal** | FITKO is actively coordinating efforts for OZG 2.0. |
| **Source** | FITKO Official Communications (Ongoing initiatives, no single recent press release found for this specific update). |

| Element | Details |
|---------|---------|
| **Signal** | BVA is undertaking significant projects to modernise register systems and portals. |
| **Source** | BVA Procurement Notices and Public Statements (General information, specific recent tender details not yet public). |

## 2. REGULATORY COUNTDOWN: EU AI Act
## 3. STRATEGIC THEME: AI Governance
## 4. PRIORITIZED ACTION ITEMS
## 5. CALENDAR: KEY DATES AHEAD
## 6. GARTNER CAPABILITY QUICK REFERENCE
"""

    result = rb.evaluate_briefing_quality(markdown, {"min_links": 3, "min_domains": 2})

    assert result["should_block"] is True
    assert "citation_threshold" in result["blocking_codes"]
    assert "weak_source_language" in result["blocking_codes"]


def test_empty_grounding_metadata_allows_explicit_markdown_source_links():
    rb = load_run_briefing()
    markdown = """PUBLIC SECTOR DAILY

TODAY'S TOP PRIORITY

## 1. VERIFIED HARD SIGNALS

| Element | Details |
|---------|---------|
| **Signal** | FITKO opened a consultation. |
| **Source** | [FITKO consultation](https://www.fitko.de/aktuelles/details/oeffentliche-konsultation-zur-foederalen-api-autorisierungsinfrastruktur-gestartet) |

## 2. REGULATORY COUNTDOWN: EU AI Act
Source: [European Commission](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)

## 3. STRATEGIC THEME: AI Governance
Source: [Sachsen Digitalagentur](https://www.digitalagentur.sachsen.de/cop-ki-und-verwaltung.html)

## 4. PRIORITIZED ACTION ITEMS
## 5. CALENDAR: KEY DATES AHEAD
## 6. GARTNER CAPABILITY QUICK REFERENCE
"""

    response = types.SimpleNamespace(
        text=markdown,
        candidates=[types.SimpleNamespace(grounding_metadata=types.SimpleNamespace())],
    )

    with_links = rb.add_citations_markdown(response)
    result = rb.evaluate_briefing_quality(with_links, {"min_links": 3, "min_domains": 2})

    assert with_links == markdown
    assert result["should_block"] is False
    assert result["source_link_count"] == 3
    assert result["source_domain_count"] == 3


def test_newsroom_source_colon_rows_count_as_source_rows():
    rb = load_run_briefing()
    markdown = """TODAY'S TOP PRIORITY

## 1. NEWSROOM DIGEST

### Top Story
- **What changed:** Sachsen named a new digital leadership contact.
- **Why it matters:** Named-account stakeholder change creates an outreach trigger.
- **BD move:** Review the source and qualify the account angle.
- **Source:** [Sachsen official notice](https://www.sachsen.de/news/cio-leitung)

## 2. BD ACTIONS

| Priority | Account | Trigger | Next move | Evidence |
|----------|---------|---------|-----------|----------|
| **P1** | Freistaat Sachsen | leadership | Review source and qualify outreach. | [Sachsen official notice](https://www.sachsen.de/news/cio-leitung) |

## 3. NO-SIGNAL / SUPPRESSED LEADS

- No rejected candidates recorded.

## 4. QUALITY FOOTER

- Candidate count: 1
- Source domains: sachsen.de
- Rejected count: 0
- Fallback mode: gemini
"""

    result = rb.evaluate_briefing_quality(markdown, {"min_links": 1, "min_domains": 1})

    assert result["should_block"] is False
    assert result["source_row_count"] == 1
    assert "missing_source_rows" not in result["blocking_codes"]


def test_source_only_digest_counts_rendered_source_colon_rows():
    rb = load_run_briefing()
    cfg = load_json("config.json")
    markdown = rb.render_source_only_digest(cfg, [sample_candidate()], [])

    result = rb.evaluate_briefing_quality(markdown, {"min_links": 1, "min_domains": 1})

    assert result["should_block"] is False
    assert result["source_row_count"] == 1
    assert result["source_link_count"] == 1
    assert result["source_domain_count"] == 1


def test_main_sends_quality_alert_not_briefing_when_generated_text_has_no_links(monkeypatch):
    rb = load_run_briefing()
    bad_markdown = """PUBLIC SECTOR DAILY

TODAY'S TOP PRIORITY

## 1. VERIFIED HARD SIGNALS

| Element | Details |
|---------|---------|
| **Signal** | FITKO continues OZG 2.0 coordination. |
| **Source** | General information on ongoing initiatives, no single recent press release found. |

## 2. REGULATORY COUNTDOWN: EU AI Act
## 3. STRATEGIC THEME: AI Governance
## 4. PRIORITIZED ACTION ITEMS
## 5. CALENDAR: KEY DATES AHEAD
## 6. GARTNER CAPABILITY QUICK REFERENCE
"""

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return types.SimpleNamespace(text="OK", candidates=[])
            return types.SimpleNamespace(text=bad_markdown, candidates=[])

        def list(self):
            return []

    fake_client = types.SimpleNamespace(models=FakeModels())
    sent = []

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("RECIPIENT_EMAIL", "bd@example.com")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    monkeypatch.setenv("BRIEFING_ARTIFACT_DIR", str(ROOT / ".pytest-artifacts"))
    monkeypatch.setattr(rb, "collect_source_candidates", lambda cfg: ([sample_candidate()], []))
    monkeypatch.setattr(rb.genai, "Client", lambda api_key: fake_client)
    monkeypatch.setattr(rb, "throttle_gemini_calls", lambda: None)
    monkeypatch.setattr(rb, "today_iso_utc", lambda: "2026-04-28")
    monkeypatch.setattr(
        rb,
        "send_email",
        lambda subject, html, sender, recipients, smtp_password: sent.append(
            {"subject": subject, "html": html, "recipients": recipients}
        ),
    )

    rb.main()

    assert len(sent) == 1
    assert "quality alert" in sent[0]["subject"].lower()
    assert "weak_source_language" in sent[0]["html"]
    assert "FITKO continues OZG 2.0 coordination" not in sent[0]["html"]


def test_main_sends_source_only_digest_after_rate_limit_when_candidates_exist(monkeypatch):
    rb = load_run_briefing()

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return types.SimpleNamespace(text="OK", candidates=[])
            raise RuntimeError("HTTP 429: Provider returned error")

        def list(self):
            return []

    fake_models = FakeModels()
    fake_client = types.SimpleNamespace(models=fake_models)
    sent = []

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("RECIPIENT_EMAIL", "bd@example.com")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    monkeypatch.setenv("BRIEFING_ARTIFACT_DIR", str(ROOT / ".pytest-artifacts"))
    monkeypatch.setattr(rb, "collect_source_candidates", lambda cfg: ([sample_candidate()], []))
    monkeypatch.setattr(rb.genai, "Client", lambda api_key: fake_client)
    monkeypatch.setattr(rb, "throttle_gemini_calls", lambda: None)
    monkeypatch.setattr(rb, "today_iso_utc", lambda: "2026-04-28")
    monkeypatch.setattr(
        rb,
        "send_email",
        lambda subject, html, sender, recipients, smtp_password: sent.append(
            {"subject": subject, "html": html, "recipients": recipients}
        ),
    )

    rb.main()

    assert len(sent) == 1
    assert "source-only digest" in sent[0]["subject"].lower()
    assert "Neue CIO Leitung fuer Sachsen digital ernannt" in sent[0]["html"]
    assert len(fake_models.calls) == 4


def test_main_sends_source_only_digest_after_preflight_503_when_candidates_exist(monkeypatch):
    rb = load_run_briefing()

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

        def list(self):
            return []

    fake_models = FakeModels()
    fake_client = types.SimpleNamespace(models=fake_models)
    sent = []

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("RECIPIENT_EMAIL", "bd@example.com")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    monkeypatch.setenv("BRIEFING_ARTIFACT_DIR", str(ROOT / ".pytest-artifacts"))
    monkeypatch.setattr(rb, "collect_source_candidates", lambda cfg: ([sample_candidate()], []))
    monkeypatch.setattr(rb.genai, "Client", lambda api_key: fake_client)
    monkeypatch.setattr(rb, "throttle_gemini_calls", lambda: None)
    monkeypatch.setattr(rb, "today_iso_utc", lambda: "2026-04-28")
    monkeypatch.setattr(
        rb,
        "send_email",
        lambda subject, html, sender, recipients, smtp_password: sent.append(
            {"subject": subject, "html": html, "recipients": recipients}
        ),
    )

    rb.main()

    assert len(sent) == 1
    assert "source-only digest" in sent[0]["subject"].lower()
    assert "Neue CIO Leitung fuer Sachsen digital ernannt" in sent[0]["html"]
    assert len(fake_models.calls) == 1


def test_newsroom_prompt_contains_only_structured_candidates_with_urls():
    rb = load_run_briefing()
    cfg = load_json("config.json")
    prompt = rb.build_newsroom_prompt(cfg, [sample_candidate()], [])

    assert "Candidate source records" in prompt
    assert "https://www.sachsen.de/news/cio-leitung" in prompt
    assert "Top story must be a leadership/stakeholder" in prompt
    assert "EU AI Act or regulatory items are compact side notes" in prompt


def test_main_sends_quality_alert_when_collector_finds_no_candidates(monkeypatch):
    rb = load_run_briefing()
    sent = []

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("RECIPIENT_EMAIL", "bd@example.com")
    monkeypatch.setenv("SMTP_USER", "sender@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    monkeypatch.setenv("BRIEFING_ARTIFACT_DIR", str(ROOT / ".pytest-artifacts"))
    monkeypatch.setattr(rb, "collect_source_candidates", lambda cfg: ([], [{"source_id": "x", "rejection_reason": "fetch"}]))
    monkeypatch.setattr(rb, "today_iso_utc", lambda: "2026-04-28")
    monkeypatch.setattr(
        rb,
        "send_email",
        lambda subject, html, sender, recipients, smtp_password: sent.append(
            {"subject": subject, "html": html, "recipients": recipients}
        ),
    )

    rb.main()

    assert len(sent) == 1
    assert "quality alert" in sent[0]["subject"].lower()
    assert "no_collected_candidates" in sent[0]["html"]
