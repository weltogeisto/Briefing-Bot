# Daily Briefing — Claude Routine Operating Instructions

This file is the operating manual for the **Claude Routine** that produces the
Public Sector Daily briefing. Instead of a hand-rolled scraper plus a small LLM
that only formats pre-collected text, the routine uses Claude as the engine:
Claude discovers stories with live web search, judges BD relevance with reasoning,
writes the digest, and sends it via Gmail SMTP.

## How to create the routine (one-time setup)

1. Go to <https://claude.ai/code/routines> (or run `/schedule` in the CLI).
2. **New routine** → name it `Public Sector Daily Briefing`.
3. **Prompt:** paste the block under "Routine prompt" below.
4. **Repository:** add `weltogeisto/Briefing-Bot`.
5. **Environment:** create/select an environment with:
   - **Network access:** `Trusted` is enough — `WebSearch`/`WebFetch` and MCP
     connectors route through Anthropic. (Use `Full` only if you later want direct
     `curl` to specific sites.)
   - **Environment variables:** `SMTP_USER`, `SMTP_PASSWORD` (Gmail App Password,
     2FA required), `RECIPIENT_EMAIL` (comma-separated for multiple), and
     optionally `SUBJECT_PREFIX`.
   - **Setup script:** `pip install -r requirements.txt` (only `markdown` is needed
     for sending).
6. **Trigger:** Schedule → daily at your preferred local time (e.g. 07:30). The
   minimum interval is one hour; times are converted from your local zone.
7. **Create**, then **Run now** once to verify the first email arrives.

### Routine prompt

> Produce today's Public Sector Daily briefing for the cloned `Briefing-Bot`
> repository by following `ROUTINE.md` exactly. Read `src/config.json` for the
> accounts, themes, sources, regulatory items, and lookback window. Research with
> live web search, apply the BD-first ranking and evidence rules, write the
> newsroom digest to `briefing.md`, then send it by running
> `python src/send_briefing.py --input briefing.md`. Do not fabricate: if there
> are no verified signals for the period, send the honest "no verified signals"
> digest rather than filler.

## What Claude does each run

### 1. Load configuration
Read `src/config.json`. Use:
- `accounts` — named target accounts, their `active_projects_seed`, `aliases`, and `contact_roles_seed`.
- `themes` — strategic topics and `keywords`.
- `regulatory_items` — static regulatory dates to reference (verify against official sources).
- `lookback_hours` — the recency window (default 72h). Only include developments within this window.
- `territory`, `prepared_for`, `email_v2.max_stories`, `email_v2.suppressed_leads`.
Also read `src/entity_universe.json` and rotate in a few spotlight entities (see `entity_coverage`).

### 2. Discover (live web search — this replaces the old scraper)
For each priority account and the rotating spotlight entities, search for developments in the last `lookback_hours`:
- Official press releases, procurement/tender notices, contract awards.
- Budget decisions, policy/strategy publications, project milestones, consultation deadlines.
- Leadership / stakeholder changes (CIO/CDO/Staatssekretär appointments).
Prefer the official and primary sources for these German public-sector bodies (federal: FITKO, IT-Planungsrat, BMDS, DigitalService, BVA, BSI, BDBOS; states: Sachsen, Sachsen-Anhalt, Thüringen, Niedersachsen; sovereignty ecosystem: openCode, ZenDiS, govdigital, Vitako; procurement: service.bund.de, e-Vergabe). Use trade media (eGovernment Computing, Kommune21, Behörden Spiegel, Public Manager) as **leads only** and confirm against an official/primary source before treating a claim as verified. Use `WebFetch` to open the actual article and confirm the date and substance — do not rely on a search snippet alone.

### 3. Rank (BD-first)
Order candidates by:
1. Leadership / stakeholder-access signals.
2. Named-account project, procurement, or operating-model triggers.
3. Official-source confirmations.
4. Theme matches.
5. Broad market or regulatory shifts.
Generic regulation must not outrank a named-account opportunity. EU AI Act content is a compact side note unless tied to a concrete named-account action. Cluster the same story across multiple outlets into one entry and cite the strongest source.

### 4. Evidence rules (do not fabricate)
- Every story must include a markdown link to the source you actually verified.
- Use only facts you confirmed via web fetch this run — no memory, no generic background.
- If an account has no usable evidence this period, list it under suppressed / no-signal with the reason; do not invent activity.
- If there are no verified hard signals at all, say so plainly. A short honest briefing beats a padded one.

### 5. Write `briefing.md`
Output markdown in exactly this structure (fast-scan newsroom digest, under ~1300 words):

```
TODAY'S TOP PRIORITY

## 1. NEWSROOM DIGEST

### Top Story
- **What changed:** one concrete, verified change.
- **Why it matters:** why this creates a BD opening now.
- **BD move:** specific next action tied to account, trigger, and timing.
- **Source:** [source title](https://verified-url)

### Other Verified Stories
Up to (max_stories - 1) short bullets, each with **Source:** markdown links.

## 2. BD ACTIONS

| Priority | Account | Trigger | Next move | Evidence |
|----------|---------|---------|-----------|----------|

## 3. NO-SIGNAL / SUPPRESSED LEADS

Important accounts/leads with no verified signal this period, with the reason. No speculation.

## 4. REGULATORY COUNTDOWN

Compact T-minus notes for configured regulatory_items, dates verified against official sources.

## 5. QUALITY FOOTER

Story count, source domains used, suppressed count, and date/lookback window.
```

Start the output with:
```
🇩🇪 PUBLIC SECTOR DAILY — Business Intelligence Briefing
Date: <today> | Prepared for: <prepared_for> | Territory: <territory>
```

### 6. Send
Write the digest to `briefing.md`, then run:

```bash
python src/send_briefing.py --input briefing.md
```

The script converts the markdown to HTML and emails it via Gmail SMTP using the
environment variables above. To preview without sending, add `--dry-run`.

## Notes
- The legacy Gemini pipeline (`src/run_briefing.py`, GitHub Actions `daily.yml`)
  still works and can stay as a fallback, but the routine is the primary engine.
- Keep curation in `src/config.json`; this file describes *behavior*, the config
  holds *what to cover*.
