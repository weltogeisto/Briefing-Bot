#!/usr/bin/env python3
"""
Generates a daily briefing and emails it via Gmail SMTP.

Required env vars (GitHub Secrets):
  GEMINI_API_KEY, RECIPIENT_EMAIL, SMTP_USER, SMTP_PASSWORD
"""

from __future__ import annotations

import os
import json
import datetime as dt
import smtplib
import ssl
import re
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
import markdown as md


# -----------------------------
# Helpers
# -----------------------------

def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def today_iso_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def load_json(rel_path: str) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict:
    return load_json("config.json")


def load_entity_universe(cfg: dict) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    rel = (cfg.get("entity_coverage") or {}).get("entity_universe_path", "entity_universe.json")
    path = os.path.join(here, rel)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"states": {}}


def dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def estimate_word_count(text: str) -> int:
    """
    Estimates word count from model text for output-size guardrails.
    """
    return len(re.findall(r"\S+", text or ""))


def extract_markdown_links(markdown_text: str) -> List[str]:
    links = re.findall(r"\[[^\]]+\]\((https?://[^)\s]+)\)", markdown_text or "")
    return dedup_keep_order(links)


def build_compression_prompt(markdown_text: str, max_words: int) -> str:
    return f"""You are compressing a markdown briefing to satisfy a strict maximum length.

Compression requirements:
1. Preserve ALL section headers and overall section order.
2. Preserve factual claims (entities, dates, numbers, deadlines, commitments).
3. Remove repetition, verbose phrasing, and non-essential filler.
4. Keep markdown citations/links intact where they exist.
5. If any citation cannot be kept inline, include it in a final 'Sources' list.
6. Output must remain valid markdown.
7. Final output must be <= {max_words} words.

Return only the compressed markdown.

--- BEGIN BRIEFING ---
{markdown_text}
--- END BRIEFING ---"""


def is_obviously_retired_model(model_id: str) -> bool:
    """
    Avoid known-retired families like Gemini 1.0/1.5 which commonly return 404.
    """
    m = (model_id or "").strip().lower()
    return ("gemini-1.0" in m) or ("gemini-1.5" in m)


def sanitize_config_models(cfg_models: List[str]) -> List[str]:
    """
    Drops obviously retired model ids, trims whitespace, de-dups.
    """
    cleaned = []
    for m in cfg_models or []:
        m = (m or "").strip()
        if not m:
            continue
        if is_obviously_retired_model(m):
            # skip old/retired ids that cause 404
            continue
        cleaned.append(m)
    return dedup_keep_order(cleaned)


def pick_spotlight_entities(cfg: dict, universe: dict, iso_date: str) -> List[str]:
    cov = cfg.get("entity_coverage", {}) or {}
    k = int(cov.get("spotlight_entities_per_day", 25))
    always = list(dict.fromkeys([e for e in cov.get("always_include", []) if isinstance(e, str) and e.strip()]))

    # Flatten universe into stable list
    flat: List[str] = []
    states = (universe.get("states") or {})
    for state_name in sorted(states.keys()):
        cats = (states[state_name].get("categories") or {})
        for cat_name in sorted(cats.keys()):
            for ent in cats[cat_name]:
                if isinstance(ent, str) and ent.strip():
                    flat.append(ent.strip())

    flat_unique = dedup_keep_order(flat)
    if not flat_unique:
        return always

    seed = sum(ord(ch) for ch in iso_date)
    start = seed % len(flat_unique)

    spotlight: List[str] = []
    i = start
    while len(spotlight) < k and len(spotlight) < len(flat_unique):
        spotlight.append(flat_unique[i])
        i = (i + 1) % len(flat_unique)

    out: List[str] = []
    for e in always + spotlight:
        if e not in out:
            out.append(e)
    return out


# -----------------------------
# System Instruction & Prompt
# -----------------------------

def build_system_instruction(brevity: Dict[str, Any]) -> str:
    """
    System instruction that enforces grounding, format rules, and output structure.
    This is processed with higher priority than the user prompt.
    """
    target_words = brevity.get("target_words", [900, 1200])
    if not isinstance(target_words, list) or len(target_words) != 2:
        target_words = [900, 1200]
    target_min, target_max = int(target_words[0]), int(target_words[1])
    max_words = int(brevity.get("max_words", 1300))
    section_limits = brevity.get("section_word_limits", {}) or {}
    top_priority_limit = int(section_limits.get("top_priority", 80))
    signal_limit = int(section_limits.get("signal", 120))
    regulatory_limit = int(section_limits.get("regulatory_countdown", 220))

    return f"""You are a German public sector business intelligence analyst creating a daily briefing for business development purposes.

CRITICAL RULES (MUST FOLLOW):
1. USE GOOGLE SEARCH for EVERY factual claim. You have access to Google Search - USE IT for every piece of information.
2. ONLY include information you can verify through search. If you cannot find current sources, OMIT the item entirely.
3. DO NOT invent or hallucinate: tenders, budgets, deadlines, leadership changes, projects, or initiatives.
4. DO NOT use your training data for facts - ALWAYS search for current information.
5. Focus on information from the LAST 72 HOURS. Older information should be clearly marked with its date.
6. DO NOT paste raw URLs in the text - citations are handled automatically via grounding metadata.
7. Keep personal data minimal: job titles/roles are OK, but no private emails or phone numbers.
8. Brevity is mandatory: target {target_min}-{target_max} words total, never exceed {max_words} words.
9. Enforce section limits: Top Priority <= {top_priority_limit} words; each Hard Signal block (including table text) <= {signal_limit} words; Regulatory Countdown section <= {regulatory_limit} words.

ITEM SCORING RUBRIC (APPLY BEFORE INCLUDING ANY CANDIDATE ITEM):
- Score each candidate item on a 0-5 scale for:
  1) Impact on near-term BD opportunity
  2) Urgency/deadline proximity
  3) Verifiability from official sources
- Compute total score out of 15.
- Minimum inclusion threshold: include only items scoring >=10/15.
- If an item scores below threshold, DROP it (do not summarize it in the output).

OUTPUT FORMAT (Markdown - follow this structure EXACTLY):

COMPACT OUTPUT RULES:
- If fewer than 3 high-confidence developments are available for a section, provide 2 verified signals instead of forcing 3.
- Limit bullets per subsection to a maximum of 3.
- Use concise phrasing and single-sentence bullets where possible.

---

⚡ TODAY'S TOP PRIORITY

**[Key deadline or development with T-X Days countdown if applicable].** [1-2 sentences explaining why this matters and what action to take.]

**Gartner Play:** [Specific positioning recommendation - what to pitch and how.]

---

## 1. VERIFIED HARD SIGNALS

*Confirmed developments requiring immediate BD attention*

### 1.1 [Signal Title with Key Metric if applicable]

| Element | Details |
|---------|---------|
| **Signal** | [What happened - be specific with numbers, dates, entities] |
| **Source** | [Official source and date] |
| **Advisory Opening** | [How to position - specific service/capability to offer, pain point to address] |
| **Gartner Asset** | [Relevant Gartner research, frameworks, or tools to reference] |
| **Decision-Oriented Outcome** | [Concrete decision this enables now, e.g., who to contact this week and why] |

### 1.2 [Next Signal Title]
[Same table structure...]

### 1.3 [Next Signal Title]
[Same table structure...]

---

## 2. REGULATORY COUNTDOWN: [PRIMARY REGULATION]

🚨 **CRITICAL DEADLINE: [Date] (T-X Days)**

[1-2 sentences on what takes effect and implications]

**What Public Sector Agencies Are Missing:**
- [Gap 1]
- [Gap 2]
- [Gap 3]

**Gartner Advisory Positioning:**
- **[Capability Area 1]** — [How to position]
- **[Capability Area 2]** — [How to position]
- **[Capability Area 3]** — [How to position]

*[Any relevant note about upcoming platforms, registries, or requirements]*

---

## 3. STRATEGIC THEME: [THEME NAME]

[2-3 sentence overview of current state and why it matters]

**Verified Developments:**
- [Development 1 with entity and date]
- [Development 2 with entity and date]
- [Development 3 with entity and date]
- **Decision-Oriented Outcome:** [One concrete decision to take this week (owner, target contact, and purpose)]

**Gartner Play:** Position **[specific capabilities]** as core to [objective]. Talk track: [specific talking points].

---

## 4. PRIORITIZED ACTION ITEMS

| Priority | Target | Action | Timing |
|----------|--------|--------|--------|
| **P1** | [Entity] | [Specific action with context + decision-oriented outcome (who to contact this week)] | This Week |
| **P1** | [Entity] | [Specific action with context + decision-oriented outcome (who to contact this week)] | This Week |
| **P2** | [Entity] | [Specific action with context + decision-oriented outcome] | Next 2 Weeks |
| **P2** | [Entity] | [Specific action with context + decision-oriented outcome] | [Quarter] |
| **P3** | [Entity] | [Specific action with context + decision-oriented outcome] | [Quarter] |

---

## 5. CALENDAR: KEY DATES AHEAD

| Date | Event |
|------|-------|
| [Month Year] | [Event description] |
| [Specific Date] | [Event with significance] |
| [Specific Date] | **[MAJOR DEADLINE]** — [Description] |

---

## 6. GARTNER CAPABILITY QUICK REFERENCE

*Match these capabilities to prospect pain points for effective positioning*

| Capability Theme | Key Deliverables |
|------------------|------------------|
| **AI Governance & Compliance** | [Specific deliverables] |
| **Cloud & Infrastructure** | [Specific deliverables] |
| **Procurement & Sourcing** | [Specific deliverables] |
| **Sovereignty & Exit** | [Specific deliverables] |
| **Workforce & Literacy** | [Specific deliverables] |

---

*This briefing is based on verified market signals as of [DATE].*

*Sources: [List key source domains used]*

**For Gartner internal BD use only. Do not distribute externally.**"""


def build_prompt(cfg: dict, brevity: Dict[str, Any]) -> str:
    """
    User prompt with the specific briefing parameters for today.
    """
    today = today_iso_utc()
    lookback = int(cfg.get("lookback_hours", 72))
    briefing_mode = (cfg.get("briefing_mode", "standard") or "standard").strip().lower()
    if briefing_mode not in {"standard", "compact"}:
        briefing_mode = "standard"

    territory = ", ".join(cfg.get("territory", [])) or "[Territory]"
    prepared_for = cfg.get("prepared_for", "[Your Name]")

    universe = load_entity_universe(cfg)
    spotlight = pick_spotlight_entities(cfg, universe, today)
    spotlight_str = ", ".join(spotlight) if spotlight else "(none)"

    brand_title = (cfg.get("brand") or {}).get("title", "GARTNER PUBLIC SECTOR DAILY")
    brand_subtitle = (cfg.get("brand") or {}).get("subtitle", "New Business Intelligence Briefing")

    target_words = brevity.get("target_words", [900, 1200])
    if not isinstance(target_words, list) or len(target_words) != 2:
        target_words = [900, 1200]
    target_min, target_max = int(target_words[0]), int(target_words[1])
    max_words = int(brevity.get("max_words", 1300))
    section_limits = brevity.get("section_word_limits", {}) or {}
    top_priority_limit = int(section_limits.get("top_priority", 80))
    signal_limit = int(section_limits.get("signal", 120))
    regulatory_limit = int(section_limits.get("regulatory_countdown", 220))

    # Build account details
    acct_lines: List[str] = []
    for a in cfg.get("accounts", []):
        name = a.get("name", "Unknown")
        projects = ", ".join(a.get("active_projects_seed", []))
        acct_lines.append(f"- {name}: Active projects/areas: {projects}")
    acct_section = "\n".join(acct_lines) if acct_lines else "- German public sector entities"

    # Build theme details
    theme_lines: List[str] = []
    for t in cfg.get("themes", []):
        name = t.get("name", "Theme")
        seeds = ", ".join(t.get("seed", []))
        theme_lines.append(f"- {name}: {seeds}")
    theme_section = "\n".join(theme_lines) if theme_lines else "- Digital transformation themes"

    # Build regulatory items
    reg_lines: List[str] = []
    for r in cfg.get("regulatory_items", []):
        reg_lines.append(f"- {r.get('name', 'Item')}: {r.get('date', 'TBD')} - {r.get('notes', '')}")
    reg_section = "\n".join(reg_lines) if reg_lines else "- EU AI Act and related regulations"

    # Build capability themes for quick reference
    capability_themes = [
        "AI Governance & Compliance: AI strategy roadmaps, risk frameworks, AI Act alignment, AI model inventory templates, risk assessment playbooks",
        "Cloud & Infrastructure: Landing zone design, FinOps/cost governance, private/hybrid cloud architectures, migration-factory patterns",
        "Procurement & Sourcing: AI contracting guidance, market intelligence, vendor landscape analysis (MQ SCPS), lifecycle SLAs",
        "Sovereignty & Exit: Data residency assessments, sovereign cloud options, exit-strategy planning with trigger documentation",
        "Workforce & Literacy: AI literacy programs, cohort training, CoE setup, model science and AI security skill development"
    ]

    return f"""Generate today's {brand_title}
{brand_subtitle}

**Date:** {today}
**Prepared for:** {prepared_for}
**Territory:** {territory}
**Briefing mode:** {briefing_mode}

SEARCH GOOGLE NOW for current developments from the last {lookback} hours.

PRIORITY ENTITIES TO RESEARCH (search for recent news, press releases, procurement announcements):
{acct_section}

ADDITIONAL ENTITIES IN TODAY'S SPOTLIGHT:
{spotlight_str}

KEY REGULATORY ITEMS (verify current status and calculate T-minus days from today {today}):
{reg_section}

STRATEGIC THEMES TO COVER (pick ONE for section 3, choose based on most newsworthy current developments):
{theme_section}

GARTNER CAPABILITY THEMES FOR SECTION 6:
{chr(10).join('- ' + c for c in capability_themes)}

INSTRUCTIONS:
1. Search Google for CURRENT news about each entity and theme
2. Find 2-3 VERIFIED hard signals with real sources (budgets, MoUs, procurements, policy announcements)
3. Calculate exact T-minus days for regulatory deadlines from today's date ({today})
4. Create actionable Gartner Plays and Advisory Openings for each signal
5. Generate the briefing following the EXACT format from your system instructions
6. Include real source domains in the footer (e.g., BMDS.bund.de, Bundestag.de, eGovernment.de)
7. Apply briefing_mode behavior:
   - compact: prioritize high-signal items, keep wording concise, and follow compact output rules.
   - standard: provide fuller context while still following compact output constraints when evidence is limited.

START THE OUTPUT WITH:
🇩🇪 {brand_title}
{brand_subtitle}

Date: {today} | Prepared for: {prepared_for} | Territory: {territory}"""


# -----------------------------
# Grounding citations handling
# -----------------------------

def add_citations_markdown(response) -> str:
    text = getattr(response, "text", "") or ""
    if not text:
        return text

    try:
        cand = response.candidates[0]
        gm = cand.grounding_metadata
        supports = gm.grounding_supports
        chunks = gm.grounding_chunks
    except Exception:
        return text + "\n\n_Note: No grounding metadata returned (no citations available)._"

    supports_sorted = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)
    for support in supports_sorted:
        end_index = support.segment.end_index
        idxs = list(getattr(support, "grounding_chunk_indices", []) or [])
        if not idxs:
            continue

        links = []
        for i in idxs:
            if i < len(chunks) and getattr(chunks[i], "web", None):
                uri = chunks[i].web.uri
                links.append(f"[{i+1}]({uri})")
        if not links:
            continue

        cite = " " + ", ".join(links)
        if 0 <= end_index <= len(text):
            text = text[:end_index] + cite + text[end_index:]

    return text


def markdown_to_html(markdown_text: str) -> str:
    html = md.markdown(markdown_text, extensions=["extra", "sane_lists", "smarty"], output_format="html5")
    return f"""\
<html>
  <body style="font-family: Arial, Helvetica, sans-serif; line-height: 1.45;">
    {html}
    <hr/>
    <p style="font-size: 12px; color: #666;">
      Generated with Gemini + Google Search grounding. If citations are missing, the model may not have returned grounding metadata for this run.
    </p>
  </body>
</html>
"""


def ensure_citations_present(markdown_text: str, fallback_citations: List[str]) -> str:
    if not fallback_citations:
        return markdown_text

    has_inline_links = bool(re.search(r"\[[^\]]+\]\((https?://[^)\s]+)\)", markdown_text or ""))
    if has_inline_links:
        return markdown_text

    source_lines = "\n".join(f"- {u}" for u in fallback_citations)
    return f"{markdown_text.rstrip()}\n\n## Sources\n{source_lines}\n"


def normalize_markdown(markdown_text: str) -> str:
    return (markdown_text or "").strip()


def briefing_validation_requirements(cfg: dict) -> Dict[str, Any]:
    briefing_mode = (cfg.get("briefing_mode", "standard") or "standard").strip().lower()
    if briefing_mode not in {"standard", "compact"}:
        briefing_mode = "standard"

    compact_min = int(os.getenv("BRIEFING_MIN_WORDS_COMPACT", "250"))
    standard_min = int(os.getenv("BRIEFING_MIN_WORDS_STANDARD", "400"))
    min_words = compact_min if briefing_mode == "compact" else standard_min

    required_markers = [
        "⚡ TODAY'S TOP PRIORITY",
        "## 1. VERIFIED HARD SIGNALS",
    ]
    return {
        "mode": briefing_mode,
        "min_words": min_words,
        "required_markers": required_markers,
    }


def validate_briefing_markdown(markdown_text: str, cfg: dict) -> Dict[str, Any]:
    normalized = normalize_markdown(markdown_text)
    final_word_count = estimate_word_count(normalized)
    requirements = briefing_validation_requirements(cfg)
    missing_markers = [m for m in requirements["required_markers"] if m not in normalized]
    validation_passed = bool(normalized) and final_word_count >= requirements["min_words"] and not missing_markers

    reason_parts: List[str] = []
    if not normalized:
        reason_parts.append("empty_output")
    if final_word_count < requirements["min_words"]:
        reason_parts.append(f"word_count_too_low:{final_word_count}<{requirements['min_words']}")
    if missing_markers:
        reason_parts.append(f"missing_markers:{', '.join(missing_markers)}")

    return {
        "normalized_markdown": normalized,
        "final_word_count": final_word_count,
        "validation_passed": validation_passed,
        "reason": "; ".join(reason_parts) if reason_parts else "ok",
        "requirements": requirements,
    }


# -----------------------------
# Email
# -----------------------------

def send_email(subject: str, html_body: str, sender: str, recipients: List[str], smtp_password: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, smtp_password)
        server.sendmail(sender, recipients, msg.as_string())


# -----------------------------
# Model selection + generation
# -----------------------------

class QuotaExceededError(RuntimeError):
    """Raised when Gemini requests fail due to quota / rate-limit exhaustion."""


def summarize_error(exc: Exception) -> str:
    msg = " ".join(str(exc).strip().split())
    return msg[:280] + ("..." if len(msg) > 280 else "")


def is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = ["resource_exhausted", "429", "too many requests", "quota", "rate limit", "ratelimit"]
    return any(m in msg for m in markers)


def list_available_models(client: genai.Client, limit: int = 40) -> List[str]:
    """
    Best-effort: list available model IDs for debugging in Actions logs.
    """
    out: List[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", None) or getattr(m, "model", None) or ""
            name = str(name)
            # Often "models/..." — strip that for readability
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                out.append(name)
            if len(out) >= limit:
                break
    except Exception:
        return []
    return out


def throttle_gemini_calls() -> None:
    min_interval_sec = float(os.getenv("GEMINI_MIN_INTERVAL_SEC", "12"))
    if min_interval_sec > 0:
        print(f"INFO: Throttling Gemini call for {min_interval_sec:.1f}s")
        time.sleep(min_interval_sec)


def run_preflight_check(client: genai.Client, model_ids: List[str]) -> None:
    """
    Send one tiny request so we can fail fast with a clear reason in logs.
    """
    if not model_ids:
        raise RuntimeError("No model IDs configured for preflight check.")

    model = model_ids[0]
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=20,
    )
    try:
        throttle_gemini_calls()
        client.models.generate_content(model=model, contents="Respond exactly with: OK", config=cfg)
        print(f"INFO: Preflight check succeeded with model '{model}'.")
    except Exception as e:
        msg = summarize_error(e)
        print(f"WARN: Preflight check failed on model '{model}': {msg}")
        if is_quota_error(e):
            raise QuotaExceededError(f"Preflight quota/rate-limit failure on '{model}': {msg}") from e
        raise RuntimeError(f"Preflight failed on '{model}': {msg}") from e


def generate_with_fallback(
    client: genai.Client,
    model_ids: List[str],
    prompt: str,
    gen_cfg: types.GenerateContentConfig,
):
    last_err: Optional[Exception] = None
    tried: List[str] = []
    quota_seen = False

    for mid in model_ids:
        try:
            tried.append(mid)
            print(f"INFO: Trying model='{mid}' max_output_tokens={gen_cfg.max_output_tokens}")
            throttle_gemini_calls()
            return client.models.generate_content(model=mid, contents=prompt, config=gen_cfg)
        except Exception as e:
            last_err = e
            quota_seen = quota_seen or is_quota_error(e)
            print(f"WARN: Model '{mid}' failed: {summarize_error(e)}")

    # If everything failed, print available models (helps you fix config without guesswork)
    avail = list_available_models(client, limit=40)
    if avail:
        print("DEBUG: Available models (first 40):")
        for x in avail:
            print(f" - {x}")

    err_msg = (
        "All models failed.\n"
        f"Tried: {tried}\n"
        f"Last error: {summarize_error(last_err) if last_err else 'unknown'}"
    )
    if quota_seen:
        raise QuotaExceededError(err_msg)
    raise RuntimeError(err_msg)


def build_quota_alert_html(subject_prefix: str, error_message: str) -> str:
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<html>
  <body style="font-family: Arial, Helvetica, sans-serif; line-height: 1.45;">
    <h2>{subject_prefix} — quota/rate-limit alert</h2>
    <p>The daily briefing run could not complete due to Gemini API quota/rate limits.</p>
    <ul>
      <li><strong>Time:</strong> {now_utc}</li>
      <li><strong>Reason:</strong> {error_message}</li>
      <li><strong>Action needed:</strong> Enable billing or increase Gemini API quota/rate limits for this project.</li>
    </ul>
    <p>This is an operational fallback email sent by the workflow when content generation is blocked.</p>
  </body>
</html>
"""


def build_operational_alert_html(subject_prefix: str, reason: str) -> str:
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<html>
  <body style="font-family: Arial, Helvetica, sans-serif; line-height: 1.45;">
    <h2>{subject_prefix} — operational alert</h2>
    <p>The daily briefing run could not produce valid briefing content.</p>
    <ul>
      <li><strong>Time:</strong> {now_utc}</li>
      <li><strong>Reason:</strong> empty/invalid briefing output</li>
      <li><strong>Details:</strong> {reason}</li>
      <li><strong>Action needed:</strong> Review generation prompt/instructions and recent model behavior in workflow logs.</li>
    </ul>
    <p>This is an operational fallback email sent by the workflow when briefing output validation fails.</p>
  </body>
</html>
"""


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = load_config()

    recipient_env = require_env("RECIPIENT_EMAIL")
    smtp_user = require_env("SMTP_USER")
    smtp_password = require_env("SMTP_PASSWORD")

    recipients = [r.strip() for r in recipient_env.split(",") if r.strip()]
    if not recipients:
        raise RuntimeError("RECIPIENT_EMAIL did not contain any valid addresses.")

    model_cfg = cfg.get("model") or {}
    brevity = model_cfg.get("brevity", {}) or {}

    prompt = build_prompt(cfg, brevity)

    # Create client with explicit API key (google-genai SDK looks for GOOGLE_API_KEY by default)
    api_key = require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # System instruction for strict format and grounding enforcement
    system_instruction = build_system_instruction(brevity)
    print("INFO: Omitted due to low confidence/impact: items below rubric threshold are intentionally excluded.")

    # Google Search grounding tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    base_max_tokens = int((cfg.get("model") or {}).get("max_output_tokens", 3800))
    base_temperature = float((cfg.get("model") or {}).get("temperature", 0.2))
    max_words = int(brevity.get("max_words", 1300))
    print(f"INFO: Effective max_words cap={max_words}")

    # Use only explicitly configured models.
    cfg_models_raw = (cfg.get("model") or {}).get("preferred_models", []) or []
    cfg_models = sanitize_config_models(cfg_models_raw)
    model_ids = dedup_keep_order(cfg_models)
    if not model_ids:
        raise RuntimeError("No non-retired model IDs are configured.")

    primary_cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[grounding_tool],
        temperature=base_temperature,
        top_p=0.95,
        max_output_tokens=base_max_tokens,
    )

    reduced_max_tokens = max(1200, int(base_max_tokens * 0.65))
    reduced_cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[grounding_tool],
        temperature=base_temperature,
        top_p=0.95,
        max_output_tokens=min(reduced_max_tokens, base_max_tokens),
    )

    compression_cfg = types.GenerateContentConfig(
        system_instruction="You are an expert editor. Follow compression instructions exactly.",
        temperature=0.0,
        top_p=0.9,
        max_output_tokens=base_max_tokens,
    )

    subject_prefix = (cfg.get("email") or {}).get("subject_prefix", "Public Sector Intelligence Briefing")
    skip_preflight = os.getenv("SKIP_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}

    try:
        if skip_preflight:
            print("INFO: SKIP_PREFLIGHT is enabled; skipping preflight Gemini call.")
        else:
            # Preflight to classify common operational failures early.
            run_preflight_check(client, model_ids)

        try:
            response = generate_with_fallback(client, model_ids, prompt, primary_cfg)
        except QuotaExceededError:
            print("WARN: Quota pressure detected, retrying with reduced token budget.")
            response = generate_with_fallback(client, model_ids, prompt, reduced_cfg)

        initial_text = getattr(response, "text", "") or ""
        before_words = estimate_word_count(initial_text)
        print(f"INFO: Initial briefing word count={before_words} (max_words={max_words})")

        md_with_cites = add_citations_markdown(response)
        final_markdown = md_with_cites

        if before_words > max_words:
            print("WARN: Briefing exceeds max_words; requesting strict compression pass.")
            fallback_links = extract_markdown_links(md_with_cites)
            compression_prompt = build_compression_prompt(md_with_cites, max_words)
            compressed_response = generate_with_fallback(client, model_ids, compression_prompt, compression_cfg)
            compressed_text = getattr(compressed_response, "text", "") or ""
            compressed_text = ensure_citations_present(compressed_text, fallback_links)
            after_words = estimate_word_count(compressed_text)
            print(f"INFO: Compressed briefing word count={after_words} (max_words={max_words})")
            final_markdown = compressed_text

            if after_words > max_words:
                raise RuntimeError(
                    f"Compressed briefing still exceeds max_words (before={before_words}, after={after_words}, max={max_words}). Email suppressed."
                )

        validation = validate_briefing_markdown(final_markdown, cfg)
        final_markdown = validation["normalized_markdown"]
        print(f"INFO: initial_word_count={before_words}")
        print(f"INFO: final_word_count={validation['final_word_count']}")
        print(f"INFO: validation_passed={validation['validation_passed']}")

        if not validation["validation_passed"]:
            print(f"WARN: Briefing validation failed (attempt=1): {validation['reason']}")
            retry_prompt = (
                f"{prompt}\n\n"
                "RETRY INSTRUCTIONS (MANDATORY):\n"
                "- Output the FULL markdown template exactly as specified in the instructions.\n"
                "- Do NOT omit sections, headers, or tables.\n"
                "- Ensure the output is complete, non-empty, and includes all required section markers.\n"
                "- No placeholders like 'same structure' or abbreviated omissions.\n"
            )
            retry_response = generate_with_fallback(client, model_ids, retry_prompt, primary_cfg)
            retry_md_with_cites = add_citations_markdown(retry_response)
            retry_before_words = estimate_word_count(getattr(retry_response, "text", "") or "")
            retry_validation = validate_briefing_markdown(retry_md_with_cites, cfg)

            print(f"INFO: initial_word_count={retry_before_words}")
            print(f"INFO: final_word_count={retry_validation['final_word_count']}")
            print(f"INFO: validation_passed={retry_validation['validation_passed']}")

            if not retry_validation["validation_passed"]:
                print(f"WARN: Briefing validation failed (attempt=2): {retry_validation['reason']}")
                alert_subject = f"{subject_prefix} — operational alert — {today_iso_utc()}"
                alert_html = build_operational_alert_html(subject_prefix, retry_validation["reason"])
                send_email(alert_subject, alert_html, smtp_user, recipients, smtp_password)
                print("OK: operational alert sent due to invalid briefing output.")
                return

            final_markdown = retry_validation["normalized_markdown"]

        html = markdown_to_html(final_markdown)

        subject = f"{subject_prefix} — {today_iso_utc()}"
        send_email(subject, html, smtp_user, recipients, smtp_password)
        print("OK: briefing sent.")
    except QuotaExceededError as e:
        # Graceful operational fallback: notify recipients but do not fail the workflow.
        alert_subject = f"{subject_prefix} — quota alert — {today_iso_utc()}"
        alert_html = build_quota_alert_html(subject_prefix, summarize_error(e))
        send_email(alert_subject, alert_html, smtp_user, recipients, smtp_password)
        print("OK: quota alert sent to recipients.")



if __name__ == "__main__":
    main()
