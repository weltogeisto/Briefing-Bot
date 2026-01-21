#!/usr/bin/env python3
"""
Generates a daily briefing in the structure of the "Public Sector Intelligence Briefing" template
and emails it via Gmail SMTP.

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
from typing import List, Dict, Any

from google import genai
from google.genai import types
import markdown as md


def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def load_config() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def today_iso_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def load_entity_universe(cfg: dict) -> dict:
    """
    Loads an expandable universe of public-sector entities for spotlight rotation.
    Path is relative to this script's folder (src/).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    rel = (cfg.get("entity_coverage") or {}).get("entity_universe_path", "entity_universe.json")
    path = os.path.join(here, rel)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"states": {}}


def pick_spotlight_entities(cfg: dict, universe: dict, iso_date: str) -> List[str]:
    """
    Deterministically pick a rotating subset of entities for 'spotlight' coverage.
    No persistent state required (reproducible per day).
    """
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

    # De-duplicate preserving order
    seen = set()
    flat_unique: List[str] = []
    for e in flat:
        if e not in seen:
            seen.add(e)
            flat_unique.append(e)

    if not flat_unique:
        return always

    # Rotate based on simple stable "seed" from date
    seed = sum(ord(ch) for ch in iso_date)
    start = seed % len(flat_unique)

    spotlight: List[str] = []
    i = start
    while len(spotlight) < k and len(spotlight) < len(flat_unique):
        spotlight.append(flat_unique[i])
        i = (i + 1) % len(flat_unique)

    # Merge always + spotlight, de-dup
    out: List[str] = []
    for e in always + spotlight:
        if e not in out:
            out.append(e)
    return out


def build_prompt(cfg: dict) -> str:
    today = today_iso_utc()
    lookback = int(cfg.get("lookback_hours", 72))

    territory = ", ".join(cfg.get("territory", [])) or "[Territory]"
    prepared_for = cfg.get("prepared_for", "[Your Name]")

    # Entity coverage (spotlight rotation)
    universe = load_entity_universe(cfg)
    spotlight = pick_spotlight_entities(cfg, universe, today)
    spotlight_str = ", ".join(spotlight) if spotlight else "(none)"

    # Accounts section blocks
    acct_blocks: List[str] = []
    for i, a in enumerate(cfg.get("accounts", []), start=1):
        acct_blocks.append(
            f"""
1.{i} {a.get("name", "Unknown Account")}

Current Signals:
- (Find verified recent signals within the lookback window; omit if not verifiable.)

Key Contacts:
- (List roles/titles only; do not include private personal data.)
- Seeds: {", ".join(a.get("contact_roles_seed", []))}

Active Projects:
- (If verifiable, list active initiatives/programmes/procurements; omit if not verifiable.)
- Seeds: {", ".join(a.get("active_projects_seed", []))}

Advisory Opening:
- (Translate verified signals into a consultative opening; include 1–2 outreach angles.)
""".strip()
        )

    # Themes blocks
    theme_blocks: List[str] = []
    for j, t in enumerate(cfg.get("themes", []), start=1):
        theme_blocks.append(
            f"""
3.{j} {t.get("name", "Theme")}
- (2–4 bullets; tie to public-sector realities and BD positioning.)
- Seeds: {", ".join(t.get("seed", []))}
""".strip()
        )

    # IMPORTANT: join outside the f-string to avoid backslash issues in f-string expressions
    acct_section = "\n\n".join(acct_blocks) if acct_blocks else "(No target accounts configured.)"
    theme_section = "\n\n".join(theme_blocks) if theme_blocks else "(No strategic themes configured.)"

    # Regulatory seeds
    reg_items = cfg.get("regulatory_items", []) or []
    if reg_items:
        reg_seed_lines = []
        for r in reg_items:
            reg_seed_lines.append(f"- {r.get('name','(item)')}: {r.get('date','(date)')} ({r.get('notes','')})")
        reg_seed = "\n".join(reg_seed_lines)
    else:
        reg_seed = "- (Find the most relevant regulatory countdown items for the territory.)"

    brand_title = (cfg.get("brand") or {}).get("title", "PUBLIC SECTOR INTELLIGENCE BRIEFING")
    brand_subtitle = (cfg.get("brand") or {}).get("subtitle", "Daily Strategic Analysis (grounded with Google Search)")

    return f"""
{brand_title}
{brand_subtitle}

Date: {today}
Prepared for: {prepared_for}
Territory: {territory}
Lookback: last ~{lookback} hours

ENTITY SPOTLIGHT (rotating): {spotlight_str}

NON-NEGOTIABLE RULES:
- Use Google Search grounding for EVERY factual claim.
- Do NOT invent tenders, budgets, deadlines, leadership moves, or initiatives.
- If you cannot find credible sources for an item, omit it.
- Do NOT paste raw URLs in the text. Citations will be attached via grounding metadata.
- Keep personal data minimal: roles/titles OK; no private emails/phone numbers.
- COVERAGE GOAL: Provide broad territory coverage using sources that span many entities (procurement portals, pressrooms, etc.).
- Additionally, focus deeper on the ENTITY SPOTLIGHT list above.

OUTPUT MUST MATCH THIS STRUCTURE (Markdown):

⚡ TODAY'S TOP PRIORITY
- One crisp primary action item or deadline alert. If a countdown is relevant, include "T-<days>".
- Must be supported by sources.

1. TARGET ACCOUNT INTELLIGENCE
{acct_section}

2. REGULATORY COUNTDOWN
- Include 1–3 items that matter NOW for the territory. Prefer EU/national regs impacting AI/cloud/cyber/procurement.
- Seeds you may consider (verify everything):
{reg_seed}

3. STRATEGIC THEMES
{theme_section}

4. PRIORITIZED ACTION ITEMS
Provide a Markdown table with columns: P | Target | Action | Timing
- P must be P1/P2/P3.
- Actions must be concrete outreach/analysis steps.

5. KEY DATES AHEAD
Provide a Markdown table with columns: Date | Event | Why it matters
- Next ~30–90 days; only verifiable items.

6. SOURCES & METHODOLOGY
- List the source types you used (e.g., official portals, procurement portals, ministry press releases).
- Briefly describe how you filtered signals (lookback window, territory focus).

Finish with:
Confidential — Do not distribute externally.
""".strip()


def add_citations_markdown(response) -> str:
    """
    Inserts Markdown links like [1](url) based on grounding metadata.
    If grounding metadata is missing, add a note.
    """
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

    # Insert citations at end of each supported segment.
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


def generate_with_fallback(
    client: genai.Client,
    model_ids: List[str],
    prompt: str,
    gen_cfg: types.GenerateContentConfig,
):
    last_err = None
    for mid in model_ids:
        try:
            return client.models.generate_content(model=mid, contents=prompt, config=gen_cfg)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All models failed. Last error: {last_err}")


def main() -> None:
    cfg = load_config()

    # Required env vars
    require_env("GEMINI_API_KEY")  # picked up by genai.Client() internally
    recipient_env = require_env("RECIPIENT_EMAIL")
    smtp_user = require_env("SMTP_USER")
    smtp_password = require_env("SMTP_PASSWORD")

    recipients = [r.strip() for r in recipient_env.split(",") if r.strip()]
    if not recipients:
        raise RuntimeError("RECIPIENT_EMAIL did not contain any valid addresses.")

    prompt = build_prompt(cfg)

    client = genai.Client()

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    gen_cfg = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=float((cfg.get("model") or {}).get("temperature", 0.2)),
        top_p=0.95,
        max_output_tokens=int((cfg.get("model") or {}).get("max_output_tokens", 3800)),
    )

    model_ids = (cfg.get("model") or {}).get(
        "preferred_models",
        ["gemini-3-flash-preview", "gemini-2.0-flash", "gemini-1.5-flash"],
    )

    response = generate_with_fallback(client, model_ids, prompt, gen_cfg)

    md_with_cites = add_citations_markdown(response)
    html = markdown_to_html(md_with_cites)

    subject_prefix = (cfg.get("email") or {}).get("subject_prefix", "Public Sector Intelligence Briefing")
    subject = f"{subject_prefix} — {today_iso_utc()}"

    send_email(subject, html, smtp_user, recipients, smtp_password)
    print("OK: briefing sent.")


if __name__ == "__main__":
    main()
