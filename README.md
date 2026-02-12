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

Reliability features in the current implementation:
- optional preflight model check (can be skipped to save one API call)
- single-model execution based on your configured `preferred_models` list
- automatic retry with a reduced token budget when quota pressure is detected
- quota/rate-limit alert email fallback so runs do not fail silently
- configurable throttling between Gemini requests (`GEMINI_MIN_INTERVAL_SEC`)

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
- `briefing_mode`: `compact` (default) for tight, signal-first daily updates, or `standard` when you want more narrative context
- `lookback_hours`: recency window used in prompt instructions (default `72`)
- `accounts`: target accounts with seed “active projects to watch”
- `themes`: your 3–5 strategic themes
- `regulatory_items`: optional list of key regs/dates to always track
- `email.subject_prefix`: subject prefix for both briefing and operational fallback emails

### Model + output control

Edit the `model` section in `src/config.json`:

- `preferred_models`: your preferred model IDs in execution order (recommended: only `gemini-2.5-flash`; retired 1.0/1.5 IDs are automatically ignored)
- `max_output_tokens`: generation token budget for the main pass
- `temperature`: model creativity level
- `brevity.target_words`: target range used in instructions
- `brevity.max_words`: hard target used in instructions for concise output
- `brevity.section_word_limits`: per-section word ceilings injected into the system instruction

If output still runs long, the script performs a compression pass that preserves section order and citations.

### Choosing a briefing mode

- `compact` (recommended for daily runs): Keeps the briefing short, allows 2 verified signals when fewer than 3 high-confidence developments exist, and limits subsection bullet density.
- `standard`: Better for stakeholder readouts where extra background is useful. It still remains evidence-based, but allows richer explanatory context.

---

## Notes on citations
The script requests grounded output and injects citations from Gemini’s `groundingMetadata`.
If the model returns no grounding metadata on a run, the email will include a note.

---


## Operational behavior (quota / billing)
If Gemini API quota or rate limits are exhausted, the workflow sends a short **quota alert email** instead of failing silently.

Operational sequence:
1. optional preflight check on the first configured model (`SKIP_PREFLIGHT=1` disables it)
2. generation with the configured model list (recommended single model: `gemini-2.5-flash`)
3. retry with reduced token budget if quota pressure appears
4. fallback alert email if generation remains blocked

What to check if this happens repeatedly:
- Enable billing for your Google AI Studio project.
- Increase Gemini API limits (RPM/TPM).
- Reduce `model.max_output_tokens` in `src/config.json`.
- Optionally tighten `model.brevity.target_words` and `model.brevity.section_word_limits`.

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
