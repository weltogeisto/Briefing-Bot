from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
import html
import http.client
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Fetcher = Callable[[str], str]


LEADERSHIP_TERMS = [
    "cio",
    "cdo",
    "chief digital",
    "leitung",
    "leiter",
    "leiterin",
    "präsident",
    "präsidentin",
    "staatssekretär",
    "staatssekretärin",
    "ernannt",
    "berufen",
    "übernimmt",
    "personalie",
]

ROLE_CHANGE_TERMS = [
    "ernannt",
    "berufen",
    "übernimmt",
    "uebernimmt",
    "wechselt",
    "neuer",
    "neue",
    "neues",
    "personalie",
]

DIGITAL_LEADERSHIP_ROLES = {
    "cio",
    "cdo",
    "chief digital",
    "leitung",
    "leiter",
    "leiterin",
    "präsident",
    "präsidentin",
    "staatssekretär",
    "staatssekretärin",
}

PROJECT_MILESTONE_TERMS = [
    "startet",
    "gestartet",
    "launch",
    "rollout",
    "pilot",
    "plattform",
    "projekt",
    "meilenstein",
    "beschluss",
    "strategie",
    "veröffentlicht",
    "veroeffentlicht",
]

BUDGET_TERMS = [
    "budget",
    "haushalt",
    "förderung",
    "foerderung",
    "investiert",
    "million",
    "mio.",
    "euro",
]

DEADLINE_TERMS = [
    "frist",
    "deadline",
    "konsultation",
    "bewerbungsfrist",
    "inkraft",
]

PROCUREMENT_TERMS = [
    "ausschreibung",
    "vergabe",
    "auftrag",
    "rahmenvertrag",
    "beschaffung",
    "tender",
    "award",
    "procurement",
]

REGULATORY_TERMS = [
    "ai act",
    "ki-verordnung",
    "regulierung",
    "verordnung",
    "compliance",
]

BOUNDARY_TERMS = {
    "ki",
    "auftrag",
}


def default_fetcher(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BriefingBot/2.0 (+https://github.com/weltogeisto/Briefing-Bot)",
            "Accept": "text/html,application/rss+xml,application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def domain_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url or "")
    host = (parsed.netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def normalize_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_source_datetime(value: str) -> Optional[dt.datetime]:
    text = normalize_text(value)
    if not text:
        return None

    # RSS commonly emits RFC-822 dates, e.g. "Fri, 01 May 2026 10:00:48 +0200".
    try:
        parsed = parsedate_to_datetime(text)
        if parsed:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        pass

    match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)

    match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b", text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)

    return None


def extract_date(value: str) -> str:
    text = normalize_text(value)
    parsed = parse_source_datetime(text)
    if parsed:
        return parsed.date().isoformat()
    return text[:40]


def source_terms(cfg: dict, key: str) -> List[str]:
    terms: List[str] = []
    for item in cfg.get(key, []) or []:
        if isinstance(item, str) and item.strip():
            terms.append(item.strip())
    return terms


def account_terms(cfg: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for account in cfg.get("accounts", []) or []:
        if not isinstance(account, dict):
            continue
        name = str(account.get("name", "")).strip()
        if not name:
            continue
        terms = [name]
        terms.extend(str(alias) for alias in account.get("aliases", []) or [])
        terms.extend(str(p) for p in account.get("active_projects_seed", []) or [])
        short = re.sub(r"\s*\([^)]*\)", "", name).strip()
        if short and short != name:
            terms.append(short)
        # Do not auto-add broad one-word state aliases such as "Sachsen" from
        # "Freistaat Sachsen": they create false account matches against
        # "Sachsen-Anhalt". Use explicit account aliases in config instead.
        out[name] = [t for t in terms if t]
    return out


def theme_terms(cfg: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for theme in cfg.get("themes", []) or []:
        if not isinstance(theme, dict):
            continue
        name = str(theme.get("name", "")).strip()
        if not name:
            continue
        terms = [name]
        terms.extend(str(s) for s in theme.get("seed", []) or [])
        terms.extend(str(s) for s in theme.get("keywords", []) or [])
        out[name] = [t for t in terms if t]
    return out


def matching_names(text: str, term_map: Dict[str, List[str]]) -> List[str]:
    haystack = text.casefold()
    matches: List[str] = []
    for name, terms in term_map.items():
        term_matched = False
        for term in terms:
            folded = term.strip().casefold()
            if len(folded) < 3:
                continue
            if len(folded) <= 3:
                if re.search(rf"\b{re.escape(folded)}\b", haystack):
                    term_matched = True
                    break
            elif folded in haystack:
                term_matched = True
                break
        if term_matched:
            matches.append(name)
    return matches


def term_hits(text: str, terms: Iterable[str]) -> List[str]:
    haystack = text.casefold()
    hits: List[str] = []
    for term in terms:
        folded = term.casefold()
        if folded in BOUNDARY_TERMS:
            if re.search(rf"\b{re.escape(folded)}\b", haystack):
                hits.append(term)
            continue
        if folded in haystack:
            hits.append(term)
    return hits


def event_type_hits(text: str, leadership_hits: List[str], procurement_hits: List[str]) -> List[str]:
    haystack = text.casefold()
    event_types: List[str] = []

    roles = {hit.casefold() for hit in leadership_hits if hit.casefold() in DIGITAL_LEADERSHIP_ROLES}
    role_change = bool(roles) and bool(term_hits(text, ROLE_CHANGE_TERMS))
    if role_change:
        event_types.append("role_change")

    if procurement_hits:
        event_types.append("procurement")

    if term_hits(text, PROJECT_MILESTONE_TERMS):
        event_types.append("project_milestone")

    if term_hits(text, BUDGET_TERMS) or re.search(r"\b\d+(?:[,.]\d+)?\s*(?:mio\.?|millionen?|€|eur)\b", haystack):
        event_types.append("budget")

    if term_hits(text, DEADLINE_TERMS):
        event_types.append("deadline")

    # Preserve order while removing duplicates.
    seen = set()
    unique: List[str] = []
    for event_type in event_types:
        if event_type not in seen:
            unique.append(event_type)
            seen.add(event_type)
    return unique


def make_candidate(
    *,
    source: dict,
    title: str,
    url: str,
    snippet: str,
    published_at: str,
    cfg: dict,
    collected_at: str,
) -> Dict[str, Any]:
    title = normalize_text(title)
    snippet = normalize_text(snippet)
    url = urllib.parse.urljoin(str(source.get("url", "")), url or str(source.get("url", "")))
    text = f"{title} {snippet}"
    accounts = matching_names(text, account_terms(cfg))
    themes = matching_names(text, theme_terms(cfg))
    leadership_hits = term_hits(text, LEADERSHIP_TERMS)
    procurement_hits = term_hits(text, PROCUREMENT_TERMS)
    regulatory_hits = term_hits(text, REGULATORY_TERMS)
    event_types = event_type_hits(text, leadership_hits, procurement_hits)
    return {
        "title": title,
        "url": url,
        "domain": domain_from_url(url),
        "source_id": source.get("id", ""),
        "source_label": source.get("label", source.get("id", "")),
        "source_type": source.get("type", ""),
        "source_tier": source.get("tier", "trade"),
        "published_at": extract_date(published_at),
        "snippet": snippet,
        "account_matches": accounts,
        "theme_matches": themes,
        "leadership_hits": leadership_hits,
        "procurement_hits": procurement_hits,
        "regulatory_hits": regulatory_hits,
        "event_types": event_types,
        "collected_at": collected_at,
    }


def parse_rss_candidates(xml_text: str, source: dict, cfg: dict, collected_at: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"xml_parse_error: {exc}") from exc
    candidates: List[Dict[str, Any]] = []
    channel_items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for item in channel_items:
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        description = item.findtext("description") or item.findtext("summary") or ""
        published = item.findtext("pubDate") or item.findtext("date") or ""
        if title and link:
            candidates.append(
                make_candidate(
                    source=source,
                    title=title,
                    url=link,
                    snippet=description,
                    published_at=published,
                    cfg=cfg,
                    collected_at=collected_at,
                )
            )
    for entry in atom_entries:
        title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        summary = entry.findtext("{http://www.w3.org/2005/Atom}summary") or ""
        published = entry.findtext("{http://www.w3.org/2005/Atom}updated") or ""
        if title and link:
            candidates.append(
                make_candidate(
                    source=source,
                    title=title,
                    url=link,
                    snippet=summary,
                    published_at=published,
                    cfg=cfg,
                    collected_at=collected_at,
                )
            )
    return candidates


def parse_html_candidates(html_text: str, source: dict, cfg: dict, collected_at: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for match in re.finditer(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html_text, flags=re.I | re.S):
        href, label = match.groups()
        title = normalize_text(label)
        if len(title) < 12:
            continue
        candidates.append(
            make_candidate(
                source=source,
                title=title,
                url=href,
                snippet="",
                published_at="",
                cfg=cfg,
                collected_at=collected_at,
            )
        )
    page_title = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
    if page_title:
        title = normalize_text(page_title.group(1))
        if title:
            candidates.append(
                make_candidate(
                    source=source,
                    title=title,
                    url=str(source.get("url", "")),
                    snippet="",
                    published_at="",
                    cfg=cfg,
                    collected_at=collected_at,
                )
            )
    return candidates


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen = set()
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate.get("url") or "").strip().lower() or (candidate.get("title") or "").strip().casefold()
        if not key or key in seen:
            rejected.append({**candidate, "rejection_reason": "duplicate_or_missing_url"})
            continue
        seen.add(key)
        kept.append(candidate)
    return kept, rejected


def score_candidate(candidate: Dict[str, Any], cfg: dict) -> Dict[str, Any]:
    weights = cfg.get("ranking", {}) or {}
    score = 0
    reasons: List[str] = []
    event_types = candidate.get("event_types")
    event_types_known = isinstance(event_types, list)
    event_types = event_types or []
    if candidate.get("leadership_hits") and (not event_types_known or "role_change" in event_types):
        score += int(weights.get("leadership", 12))
        reasons.append("leadership")
    if candidate.get("account_matches"):
        score += int(weights.get("account_match", 10))
        reasons.append("account_match")
    if candidate.get("source_tier") == "official":
        score += int(weights.get("official_source", 7))
        reasons.append("official_source")
    if candidate.get("procurement_hits"):
        score += int(weights.get("procurement_project", 6))
        reasons.append("procurement_project")
    if event_types:
        score += int(weights.get("material_event", 5))
        reasons.append("material_event")
    if candidate.get("theme_matches"):
        score += int(weights.get("theme_match", 4))
        reasons.append("theme_match")
    if candidate.get("regulatory_hits"):
        score += int(weights.get("regulatory", 1))
        reasons.append("regulatory")
    candidate = dict(candidate)
    candidate["score"] = score
    candidate["score_reasons"] = reasons
    candidate["is_regulatory_only"] = bool(candidate.get("regulatory_hits")) and not (
        candidate.get("account_matches") or candidate.get("leadership_hits") or candidate.get("procurement_hits")
    )
    return candidate


def candidate_timestamp(candidate: Dict[str, Any]) -> float:
    parsed = parse_source_datetime(str(candidate.get("published_at", "") or ""))
    if not parsed:
        return 0.0
    return parsed.timestamp()


def rank_candidates(candidates: List[Dict[str, Any]], cfg: dict) -> List[Dict[str, Any]]:
    scored = [score_candidate(candidate, cfg) for candidate in candidates]
    return sorted(
        scored,
        key=lambda c: (
            c.get("is_regulatory_only", False),
            -int(c.get("score", 0)),
            -candidate_timestamp(c),
            c.get("title") or "",
        ),
    )


def split_relevance(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        reasons = set(candidate.get("score_reasons", []) or [])
        event_types = set(candidate.get("event_types", []) or [])
        has_material_event = bool(event_types)
        procurement_portal_hit = (
            "procurement_project" in reasons
            and candidate.get("source_type") == "procurement_search"
        )
        trade_account_material = (
            candidate.get("source_tier") == "trade"
            and "account_match" in reasons
            and has_material_event
        )
        leadership_hits = {str(hit).casefold() for hit in candidate.get("leadership_hits", []) or []}
        digital_leadership = bool(DIGITAL_LEADERSHIP_ROLES & leadership_hits) and "role_change" in event_types and "account_match" in reasons
        theme_material = "theme_match" in reasons and has_material_event
        regulatory_material = "regulatory" in reasons and ("deadline" in event_types or "account_match" in reasons)
        named_account_material = "account_match" in reasons and has_material_event
        if procurement_portal_hit or trade_account_material or digital_leadership or theme_material or regulatory_material or named_account_material:
            kept.append(candidate)
        else:
            rejected.append({**candidate, "rejection_reason": "below_relevance_threshold"})
    return kept, rejected


def split_freshness(
    candidates: List[Dict[str, Any]],
    cfg: dict,
    collected_at: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookback_hours = int(cfg.get("lookback_hours", 0) or 0)
    if lookback_hours <= 0:
        return candidates, []

    now = parse_source_datetime(collected_at) or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=lookback_hours)
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        published = parse_source_datetime(str(candidate.get("published_at", "") or ""))
        # HTML pages often have no article-level timestamp. Keep unknown-date
        # candidates for relevance scoring, but strictly reject dated RSS items
        # outside the configured lookback.
        if published and published < cutoff:
            rejected.append({**candidate, "rejection_reason": "stale_outside_lookback"})
        else:
            kept.append(candidate)
    return kept, rejected


def collect_source_candidates(
    cfg: dict,
    *,
    fetcher: Optional[Fetcher] = None,
    collected_at: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fetch = fetcher or default_fetcher
    collected_at = collected_at or dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    all_candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for source in cfg.get("sources", []) or []:
        if not isinstance(source, dict) or not source.get("enabled", True):
            continue
        url = str(source.get("url", "")).strip()
        if not url:
            continue
        try:
            body = fetch(url)
            source_type = source.get("type", "rss")
            if source_type == "rss":
                all_candidates.extend(parse_rss_candidates(body, source, cfg, collected_at))
            elif source_type in {"html_page", "procurement_search"}:
                all_candidates.extend(parse_html_candidates(body, source, cfg, collected_at))
            else:
                rejected.append({"source_id": source.get("id", ""), "url": url, "rejection_reason": "unknown_source_type"})
        except Exception as exc:
            rejected.append({"source_id": source.get("id", ""), "url": url, "rejection_reason": f"fetch_or_parse_error:{exc}"})
    deduped, duplicate_rejections = deduplicate_candidates(all_candidates)
    fresh, stale_rejections = split_freshness(deduped, cfg, collected_at)
    ranked = rank_candidates(fresh, cfg)
    relevant, relevance_rejections = split_relevance(ranked)
    return relevant, rejected + duplicate_rejections + stale_rejections + relevance_rejections
