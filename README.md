# 🇩🇪 Public Sector Intelligence Briefing — Gemini (Grounded) + GitHub Actions

This repo sends you a **daily email** that matches the structure of your **Public Sector Intelligence Briefing** template:
- ⚡ Today’s Top Priority
- Target-account intelligence (FITKO, BVA, Saxony, Sachsen‑Anhalt, Thüringen, Niedersachsen — configurable)
- Regulatory countdown (e.g., EU AI Act dates — grounded & cited)
- Strategic themes
- Prioritized action items (P1/P2/P3 table)
- Key dates ahead
- Sources & methodology

It uses the **Gemini API** with **Google Search grounding** to generate **cited** insights, then emails the briefing via **Gmail SMTP**.

---

## 1) Add GitHub Actions secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | What it is |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `RECIPIENT_EMAIL` | Your inbox (comma-separated OK) |
| `SMTP_USER` | Your Gmail address (sender) |
| `SMTP_PASSWORD` | Gmail **App Password** (requires 2FA) |

---

## 2) Run it
- Manual: **Actions → “Send Public Sector Briefing” → Run workflow**
- Scheduled: daily cron in `.github/workflows/daily.yml`

Cron is UTC. Adjust if you want a fixed Berlin send time year-round.

---

## 3) Customize territory + accounts
Edit: `src/config.json`

- `prepared_for`: your name
- `territory`: list of regions / orgs
- `accounts`: target accounts with seed “active projects to watch”
- `themes`: your 3–5 strategic themes
- `regulatory_items`: optional list of key regs/dates to always track

---

## Notes on citations
The script requests grounded output and injects citations from Gemini’s `groundingMetadata`.
If the model returns no grounding metadata on a run, the email will include a note.

---


## Operational behavior (quota / billing)
If Gemini API quota or rate limits are exhausted, the workflow now sends a short **quota alert email** instead of failing silently.

What to check if this happens repeatedly:
- Enable billing for your Google AI Studio project.
- Increase Gemini API limits (RPM/TPM).
- Reduce `model.max_output_tokens` in `src/config.json`.

## Included reference
Your original DOCX template is included in `assets/` for reference.


---

## Scaling to “all public sector entities” in your states (recommended approach)

Trying to enumerate *every* entity name inside the LLM prompt is brittle and incomplete.
Instead, this template supports a **two-layer coverage model**:

### Layer A — Portal-first coverage (broad, entity-agnostic)
Each run prioritizes signals from sources that naturally cover **many public-sector entities at once**, e.g.:
- procurement / tender portals
- ministry & agency pressrooms
- budget / programme announcements

This gives you “whole-territory” coverage without maintaining a massive list.

### Layer B — Rotating entity spotlight (deep, specific)
A configurable number of entities are “spotlighted” each day (e.g., 25) so that over a week you cycle through the universe.
You can also set an “always-on” list (state CIO office, central IT provider, key ministries, etc.).

### Where to configure it
Edit `src/config.json`:
- `entity_coverage.mode`
- `entity_coverage.spotlight_entities_per_day`
- `entity_coverage.always_include`
- `entity_coverage.entity_universe_path`

And edit the universe itself:
- `src/entity_universe.json`

Tip: Start with top-level entities (state ministries, central IT providers, key agencies) and add subordinate bodies + major Kommunen over time.
