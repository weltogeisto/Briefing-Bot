# Public Sector Daily Briefing Bot

This repo sends a daily email briefing for German public sector business development. The current v2 flow is collector-first: it gathers public source candidates before Gemini writes anything, ranks them for BD relevance, and only sends a briefing when the evidence is strong enough.

The email is designed as a fast-scan newsroom digest:

- Top story: what changed, why it matters, BD move, evidence
- 2-4 verified stories with source links
- No-signal / suppressed leads when important accounts did not have enough evidence
- Small quality footer with links, domains, rejected candidates, fallback mode, and cost posture

The bot is intentionally low-cost and automation-only: GitHub Actions schedule, Gmail SMTP, public web/RSS sources, GitHub artifacts for audit, and Gemini used sparingly for synthesis.

## Reliability Model

The bot must not send unsupported "verified" claims. Weak evidence is a suppression condition, not a cosmetic warning.

Current safeguards:

- Pre-Gemini source collection from configured public sources
- Structured candidate records before LLM synthesis
- Ranking that favors leadership/stakeholder and named-account triggers over generic regulation
- Hard quality gates for missing links, missing domains, unresolved placeholders, missing source rows, weak source language, or unsupported generated claims
- Source-only digest fallback when Gemini quota/rate limits fail after usable candidates were collected
- Operational alert when collection fails, no usable candidates exist, or the generated briefing fails evidence gates
- GitHub Actions artifacts for candidate sources, rejected candidates, final markdown, and quality diagnostics
- Monthly keepalive workflow to prevent GitHub from auto-disabling scheduled workflows after repo inactivity

EU AI Act and other regulatory items are allowed, but they should normally appear as compact side notes. They should not become the top priority unless tied to a concrete named-account action.

## Setup

Add these GitHub Actions secrets in:

`Settings -> Secrets and variables -> Actions -> New repository secret`

| Secret | What it is |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `RECIPIENT_EMAIL` | Recipient inboxes, comma-separated if needed |
| `SMTP_USER` | Gmail sender address |
| `SMTP_PASSWORD` | Gmail app password, requires 2FA |

The delivery path uses Gmail SMTP. Do not rename these secrets unless you also update the workflow and runtime code.

## Running

Manual run:

`Actions -> Send Public Sector Briefing -> Run workflow`

Scheduled run:

`.github/workflows/daily.yml`

The cron is UTC. The default schedule is `30 6 * * *`, which is 07:30 Berlin in winter and 08:30 Berlin in summer.

The repo also includes `.github/workflows/keepalive.yml`. GitHub can disable scheduled workflows after 60 days of repository inactivity; keepalive commits a small timestamp file monthly so the schedule keeps running.

## Configuration

Edit `src/config.json`.

Core fields:

- `prepared_for`: audience label in the email
- `territory`: territory covered by the digest
- `lookback_hours`: recency window used for candidate collection and prompt context
- `accounts`: target accounts, project seeds, and contact-role seeds
- `themes`: strategic topics used for matching and ranking
- `regulatory_items`: static regulatory dates that can be referenced with official evidence
- `email.subject_prefix`: subject prefix for briefing and operational emails

Collector and v2 fields:

- `sources`: curated public sources collected before Gemini runs
- `ranking`: weights for leadership, account match, official source, procurement/project trigger, theme match, and regulatory content
- `email_v2.enabled`: enables the newsroom digest behavior
- `email_v2.max_stories`: maximum number of stories in the generated digest
- `email_v2.suppressed_leads`: number of rejected/no-signal items to show
- `email_v2.source_only_fallback`: sends a basic source-only digest if Gemini fails after collection succeeds
- `email_v2.top_priority`: current policy for top-story selection

Entity coverage fields:

- `entity_coverage.entity_universe_path`: optional entity universe file
- `entity_coverage.spotlight_entities_per_day`: number of entities to rotate into focus
- `entity_coverage.always_include`: accounts/entities that should stay in daily focus

## Source Collection

Sources are configured in `src/config.json` under `sources`.

Supported source types:

- `rss`
- `html_page`
- `procurement_search`

The live catalog intentionally mixes official and trade sources. Trade media is useful for surfacing account-relevant leads that official sites may not frame commercially, but the email should treat trade items as leads and prefer official/primary confirmation where available.

Current source groups include:

- Federal / shared-service sources: FITKO, IT-Planungsrat, BMDS, DigitalService, BVA, BSI, BDBOS
- State sources: Sachsen, Sachsen-Anhalt, Thueringen, Niedersachsen
- Procurement sources: service.bund.de and e-Vergabe Bund
- Digital sovereignty ecosystem: openCode, ZenDiS, govdigital, Vitako
- Trade media: eGovernment Computing, Kommune21, Behoerden Spiegel, Public Manager

Each collected candidate is normalized into a structured record with fields such as title, date, source URL, domain, source tier, account/theme matches, snippet, score, and collection timestamp.

The collector deduplicates by URL/title and separates usable candidates from rejected candidates. It also rejects low-relevance official-page noise, for example generic ministry navigation or press items that only match because a source is official. Rejections are retained in artifacts so the run can be audited without cluttering the email.

## Ranking Policy

Ranking is BD-first:

1. Leadership or stakeholder access signals
2. Named-account project, procurement, or operating-model triggers
3. Official-source confirmations
4. Theme matches
5. Broad market or regulatory shifts

Generic regulation should not outrank a named-account opportunity trigger. EU AI Act content is treated as a side note unless the source creates a specific action for a named account.

Trade media can rank when it concerns configured accounts or themes. It should not be presented as final proof of an official decision unless the story links to or can be paired with an official/primary source.

## Gemini Use

Gemini receives structured source candidates, not a blank instruction to discover stories. The prompt requires every hard signal source row to include a markdown link to the official or primary source used.

The prompt must not tell the model to omit URLs or rely on automatic citation metadata. Grounding metadata can be useful, but explicit markdown links from collected candidates are the evidence path the quality gate can verify.

If no verified hard signals are found, the briefing should say so. It must not fill the email with generic FITKO/BVA/Sachsen claims.

## Quality Gates

Generated output is blocked when diagnostics include evidence failures such as:

- Empty output
- Missing required sections
- Unresolved placeholders
- Zero links or zero domains
- Missing source rows
- Source rows that use weak phrases such as "no single recent press release", "not yet public", "general information", or "ongoing initiatives" as if they were verified evidence

Blocked generated output is not sent as a briefing. The bot sends an operational alert with the failed diagnostics instead, unless a source-only fallback is available and appropriate.

Warning-only diagnostics are reserved for minor issues that do not undermine factual grounding.

## Fallback Behavior

The normal sequence is:

1. Optional Gemini preflight check, unless `SKIP_PREFLIGHT=1`
2. Collect source candidates from configured public sources
3. Rank and select candidates
4. Ask Gemini to write the newsroom digest from those candidates
5. Evaluate the generated briefing with hard evidence gates
6. Send the generated briefing, source-only fallback, or operational alert

Fallback rules:

- If Gemini quota/rate limits fail and usable candidates exist, send a source-only digest.
- If Gemini output fails quality gates, suppress it and send an operational alert.
- If collection fails or produces no usable candidates, send an operational alert or an explicit no-verified-signals message rather than fabricated stories.

## GitHub Actions Artifacts

Each run uploads quiet audit/debug artifacts from `artifacts/`:

- `candidate_signals.json`
- `rejected_candidates.json`
- `final_briefing.md`
- `quality_report.json`
- `source_only_fallback.md`, when fallback mode is used

The email should remain clean. The artifacts are the place to debug source coverage, ranking, rejected leads, and quality decisions.

## Local Checks

Run:

```powershell
python -m compileall src
python -m pytest -q
```

The CI workflow should run the same checks on `main`, pull requests, and `codex/**` branches.

## Repository Settings

This project is automation-only and does not require GitHub Pages.

Recommended settings:

- In `Settings -> Pages`, disable Pages unless intentionally used in a fork.
- In Actions, normal operation only requires `Send Public Sector Briefing` and `Keepalive`.
- Keep SMTP and Gemini credentials in GitHub Actions secrets only.

## Cost Posture

The setup is designed to stay free or near-free:

- GitHub Actions scheduler
- Gmail SMTP
- Public source collection
- No database
- No hosted web app
- Gemini used only after candidates have been collected
- Source-only fallback when Gemini is unavailable

To reduce quota pressure, lower `model.max_output_tokens`, reduce `email_v2.max_stories`, tighten source lists, or keep `preferred_models` focused on a low-cost Gemini Flash model.

## Included Reference

The original DOCX briefing template is kept in `assets/` for reference, but the production email is now the v2 newsroom digest rather than the old long-form template.
