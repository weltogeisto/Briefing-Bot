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

def build_system_instruction() -> str:
    """
    System instruction that enforces grounding, format rules, and output structure.
    This is processed with higher priority than the user prompt.
    """
    return """You are a German public sector business intelligence analyst creating a daily briefing for business development purposes.

CRITICAL RULES (MUST FOLLOW):
1. USE GOOGLE SEARCH for EVERY factual claim. You have access to Google Search - USE IT for every piece of information.
2. ONLY include information you can verify through search. If you cannot find current sources, OMIT the item entirely.
3. DO NOT invent or hallucinate: tenders, budgets, deadlines, leadership changes, projects, or initiatives.
4. DO NOT use your training data for facts - ALWAYS search for current information.
5. Focus on information from the LAST 72 HOURS. Older information should be clearly marked with its date.
6. DO NOT paste raw URLs in the text - citations are handled automatically via grounding metadata.
7. Keep personal data minimal: job titles/roles are OK, but no private emails or phone numbers.

OUTPUT FORMAT (Markdown - follow this structure EXACTLY):

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

**Gartner Play:** Position **[specific capabilities]** as core to [objective]. Talk track: [specific talking points].

---

## 4. PRIORITIZED ACTION ITEMS

| Priority | Target | Action | Timing |
|----------|--------|--------|--------|
| **P1** | [Entity] | [Specific action with context] | This Week |
| **P1** | [Entity] | [Specific action with context] | This Week |
| **P2** | [Entity] | [Specific action with context] | Next 2 Weeks |
| **P2** | [Entity] | [Specific action with context] | [Quarter] |
| **P3** | [Entity] | [Specific action with context] | [Quarter] |

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


def build_prompt(cfg: dict) -> str:
    """
    User prompt with the specific briefing parameters for today.
    """
    today = today_iso_utc()
    lookback = int(cfg.get("lookback_hours", 72))

    territory = ", ".join(cfg.get("territory", [])) or "[Territory]"
    prepared_for = cfg.get("prepared_for", "[Your Name]")

    universe = load_entity_universe(cfg)
    spotlight = pick_spotlight_entities(cfg, universe, today)
    spotlight_str = ", ".join(spotlight) if spotlight else "(none)"

    brand_title = (cfg.get("brand") or {}).get("title", "GARTNER PUBLIC SECTOR DAILY")
    brand_subtitle = (cfg.get("brand") or {}).get("subtitle", "New Business Intelligence Briefing")

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


def generate_with_fallback(
    client: genai.Client,
    model_ids: List[str],
    prompt: str,
    gen_cfg: types.GenerateContentConfig,
):
    last_err: Optional[Exception] = None
    tried: List[str] = []

    for mid in model_ids:
        try:
            tried.append(mid)
            return client.models.generate_content(model=mid, contents=prompt, config=gen_cfg)
        except Exception as e:
            last_err = e

    # If everything failed, print available models (helps you fix config without guesswork)
    avail = list_available_models(client, limit=40)
    if avail:
        print("DEBUG: Available models (first 40):")
        for x in avail:
            print(f" - {x}")

    raise RuntimeError(
        "All models failed.\n"
        f"Tried: {tried}\n"
        f"Last error: {last_err}"
    )


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

    prompt = build_prompt(cfg)

    # Create client with explicit API key (google-genai SDK looks for GOOGLE_API_KEY by default)
    api_key = require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # System instruction for strict format and grounding enforcement
    system_instruction = build_system_instruction()

    # Google Search grounding tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    gen_cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[grounding_tool],
        temperature=float((cfg.get("model") or {}).get("temperature", 0.2)),
        top_p=0.95,
        max_output_tokens=int((cfg.get("model") or {}).get("max_output_tokens", 3800)),
    )

    # Model fallback chain - prioritize stable models with good Search grounding support
    safe_models = [
        "gemini-2.0-flash",           # Stable, excellent Search grounding support
        "gemini-2.0-flash-001",       # Specific stable version
        "gemini-2.5-flash-preview-05-20",  # Preview with grounding
        "gemini-2.5-pro-preview-05-06",    # Pro preview
        "gemini-flash-latest",        # Latest flash alias
    ]

    cfg_models_raw = (cfg.get("model") or {}).get("preferred_models", []) or []
    cfg_models = sanitize_config_models(cfg_models_raw)

    # Always try safe models first, then any user config models (minus retired ones)
    model_ids = dedup_keep_order(safe_models + cfg_models)

    response = generate_with_fallback(client, model_ids, prompt, gen_cfg)

    md_with_cites = add_citations_markdown(response)
    html = markdown_to_html(md_with_cites)

    subject_prefix = (cfg.get("email") or {}).get("subject_prefix", "Public Sector Intelligence Briefing")
    subject = f"{subject_prefix} — {today_iso_utc()}"

    send_email(subject, html, smtp_user, recipients, smtp_password)
    print("OK: briefing sent.")


if __name__ == "__main__":
    main()
