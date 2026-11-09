from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.context.models import ProspectContext
from src.researcher.models import EntityType
from src.researcher.search.url_utils import extract_domain

logger = logging.getLogger(__name__)

# Common titles and honorifics to strip during name normalization
HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "lady", "dame", "rev", "reverend", "hon",
}

CREDENTIALS_AND_SUFFIXES = {
    "phd", "md", "mba", "esq", "esquire", "jr", "sr",
    "ii", "iii", "iv", "cpa", "jd", "pmp", "cfa",
}

# Corporate legal forms and organizational designators
CORP_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c.", "ltd", "limited",
    "corp", "corporation", "co", "company", "gmbh", "ag",
    "pty ltd", "pty", "sa", "bv", "plc", "technologies",
    "technology", "systems", "labs", "software", "solutions",
    "group", "holdings", "enterprises", "services",
}

# Common boilerplate, navigation, and noise words to filter out
NOISE_WORDS = {
    "home", "about", "about us", "contact", "contact us", "privacy policy",
    "terms", "terms of service", "terms and conditions", "sign in", "log in",
    "login", "register", "signup", "sign up", "search", "menu", "navigation",
    "cookies", "cookie policy", "read more", "subscribe", "newsletter",
    "blog", "news", "press", "press release", "press releases", "careers",
    "jobs", "overview", "product", "products", "pricing", "resources",
    "all rights reserved", "copyright", "united states", "san francisco",
    "new york", "london", "customer stories", "case studies", "security",
    "white paper", "white papers", "documentation", "api", "developer",
    "developers", "support", "help", "faq", "faqs", "share", "tweet",
    "facebook", "twitter", "linkedin", "youtube", "instagram", "author",
    "admin", "administrator", "editor", "staff", "anonymous",
}

# Regex patterns for executive role mentions and bylines
ROLE_PATTERNS = [
    re.compile(
        r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*(?:,\s*|\s+is\s+|\s+was\s+appointed\s+as\s+|\s+serves\s+as\s+)(?:the\s+)?(?P<role>CEO|CTO|CFO|COO|CIO|CMO|CRO|Founder|Co-Founder|President|Vice President|VP(?:\s+of\s+[A-Za-z\s]+)?|Director(?:\s+of\s+[A-Za-z\s]+)?|Head\s+of\s+[A-Za-z\s]+|Chief\s+[A-Za-z\s]+Officer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<role>CEO|CTO|CFO|COO|President|Founder|Co-Founder|VP(?:\s+of\s+[A-Za-z\s]+)?|Director(?:\s+of\s+[A-Za-z\s]+)?)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
        re.IGNORECASE,
    ),
]

# LinkedIn profile URL pattern
LINKEDIN_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


def normalize_person_name(name: str) -> str:
    """Normalize a person's name for deterministic matching.
    
    Strips honorifics, credentials, middle initials/punctuation, and folds case.
    """
    if not name:
        return ""
    # Strip non-alphanumeric except spaces and hyphens
    cleaned = re.sub(r"[^\w\s\-]", " ", name)
    tokens = [t.lower() for t in cleaned.split() if t]
    if not tokens:
        return ""

    # Strip leading honorifics
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)

    # Strip trailing credentials
    while tokens and tokens[-1] in CREDENTIALS_AND_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_org_name(name: str) -> str:
    """Normalize an organization name for comparison.
    
    Strips corporate legal designators, punctuation, and folds case.
    """
    if not name:
        return ""
    cleaned = re.sub(r"[^\w\s\-&]", " ", name)
    tokens = [t.lower() for t in cleaned.split() if t]
    if not tokens:
        return ""

    # Strip trailing corporate designators
    while tokens and tokens[-1] in CORP_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def clean_domain(domain_or_url: str) -> str:
    """Extract and normalize a clean domain host."""
    if not domain_or_url:
        return ""
    dom = extract_domain(domain_or_url).lower().strip()
    if dom.startswith("www."):
        dom = dom[4:]
    return dom


class CandidateObservation(BaseModel):
    """An individual observation of an entity candidate in a source."""

    model_config = ConfigDict(frozen=True)

    source_id: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    extraction_method: str = "heuristic"
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DiscoveredEntityCandidate(BaseModel):
    """Normalized, deduplicated entity candidate discovered across research results."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    raw_name: str
    normalized_name: str
    domains: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    linkedin_urls: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    observations: List[CandidateObservation] = Field(default_factory=list)
    confidence_score: float = 0.5
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_source_id(self) -> Optional[str]:
        return self.source_ids[0] if self.source_ids else None

    def to_attributes_json(self) -> str:
        data = {
            "normalized_name": self.normalized_name,
            "domains": self.domains,
            "roles": self.roles,
            "organizations": self.organizations,
            "linkedin_urls": self.linkedin_urls,
            "locations": self.locations,
            "source_ids": self.source_ids,
            "observations": [obs.model_dump() for obs in self.observations],
            "attributes": self.attributes,
        }
        return json.dumps(data)


class _MutableCandidate:
    """Internal mutable accumulator used during rule-based candidate discovery."""

    def __init__(self, entity_type: EntityType, raw_name: str, normalized_name: str) -> None:
        self.candidate_id = str(uuid.uuid4())
        self.entity_type = entity_type
        self.raw_name = raw_name
        self.normalized_name = normalized_name
        self.domains: Set[str] = set()
        self.roles: Set[str] = set()
        self.organizations: Set[str] = set()
        self.linkedin_urls: Set[str] = set()
        self.locations: Set[str] = set()
        self.source_ids: List[str] = []
        self.observations: List[CandidateObservation] = []
        self.attributes: Dict[str, Any] = {}

    def add_observation(
        self,
        observation: CandidateObservation,
        domains: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        organizations: Optional[List[str]] = None,
        linkedin_urls: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
    ) -> None:
        self.observations.append(observation)
        if observation.source_id and observation.source_id not in self.source_ids:
            self.source_ids.append(observation.source_id)

        if domains:
            for d in domains:
                cd = clean_domain(d)
                if cd:
                    self.domains.add(cd)
        if roles:
            for r in roles:
                clean_r = r.strip()
                if clean_r:
                    self.roles.add(clean_r)
        if organizations:
            for o in organizations:
                clean_o = o.strip()
                if clean_o:
                    self.organizations.add(clean_o)
        if linkedin_urls:
            for lu in linkedin_urls:
                clean_lu = lu.strip()
                if clean_lu:
                    self.linkedin_urls.add(clean_lu)
        if locations:
            for loc in locations:
                clean_loc = loc.strip()
                if clean_loc:
                    self.locations.add(clean_loc)

    def freeze(self) -> DiscoveredEntityCandidate:
        return DiscoveredEntityCandidate(
            candidate_id=self.candidate_id,
            entity_type=self.entity_type,
            raw_name=self.raw_name,
            normalized_name=self.normalized_name,
            domains=sorted(self.domains),
            roles=sorted(self.roles),
            organizations=sorted(self.organizations),
            linkedin_urls=sorted(self.linkedin_urls),
            locations=sorted(self.locations),
            source_ids=list(self.source_ids),
            observations=list(self.observations),
            attributes=dict(self.attributes),
        )


class EntityCandidateGenerator:
    """Deterministic, rule-based candidate generation engine.
    
    Extracts person and organization entity candidates from search results and sources,
    normalizes attributes, retains provenance, and consolidates observations without
    forcing matches.
    """

    def generate_candidates(
        self,
        prospect: ProspectContext,
        sources: List[Any],
        search_results: Optional[List[Any]] = None,
    ) -> List[DiscoveredEntityCandidate]:
        """Generate candidate entities from sources and optional search results."""
        target_person_norm = normalize_person_name(prospect.full_name)
        target_org_norm = normalize_org_name(prospect.company_name)
        target_domain = clean_domain(prospect.company_domain or "")

        accumulated_persons: List[_MutableCandidate] = []
        accumulated_orgs: List[_MutableCandidate] = []

        # 1. Inspect search results / occurrences if provided
        if search_results:
            for res in search_results:
                title = getattr(res, "title", "") or ""
                snippet = getattr(res, "snippet", "") or ""
                url = getattr(res, "url", "") or getattr(res, "original_url", "") or ""
                domain = clean_domain(getattr(res, "domain", "") or url)
                source_id = getattr(res, "source_id", None)

                self._extract_from_text(
                    title=title,
                    text=f"{title}\n{snippet}",
                    url=url,
                    domain=domain,
                    source_id=source_id,
                    target_person_norm=target_person_norm,
                    target_org_norm=target_org_norm,
                    target_domain=target_domain,
                    prospect=prospect,
                    accumulated_persons=accumulated_persons,
                    accumulated_orgs=accumulated_orgs,
                )

        # 2. Inspect full fetched sources
        for src in sources:
            title = getattr(src, "title", "") or ""
            raw_text = getattr(src, "raw_text", "") or ""
            url = getattr(src, "url", "") or getattr(src, "final_url", "") or ""
            domain = clean_domain(getattr(src, "domain", "") or url)
            source_id = getattr(src, "source_id", None)

            self._extract_from_text(
                title=title,
                text=f"{title}\n{raw_text}",
                url=url,
                domain=domain,
                source_id=source_id,
                target_person_norm=target_person_norm,
                target_org_norm=target_org_norm,
                target_domain=target_domain,
                prospect=prospect,
                accumulated_persons=accumulated_persons,
                accumulated_orgs=accumulated_orgs,
            )

        # Freeze candidates
        result = [c.freeze() for c in accumulated_persons] + [c.freeze() for c in accumulated_orgs]
        logger.info(
            "Candidate generation complete: %d persons, %d orgs discovered",
            len(accumulated_persons),
            len(accumulated_orgs),
        )
        return result

    def _extract_from_text(
        self,
        title: str,
        text: str,
        url: str,
        domain: str,
        source_id: Optional[str],
        target_person_norm: str,
        target_org_norm: str,
        target_domain: str,
        prospect: ProspectContext,
        accumulated_persons: List[_MutableCandidate],
        accumulated_orgs: List[_MutableCandidate],
    ) -> None:
        """Extract person and organization candidates from a given title and text body."""
        # A. LinkedIn URL Detection
        found_linkedin_urls = LINKEDIN_URL_PATTERN.findall(text)
        if LINKEDIN_URL_PATTERN.search(url):
            found_linkedin_urls.append(url)

        # B. Check for Structured LinkedIn Profile in Title: "Jane Doe - VP - Acme | LinkedIn"
        if "linkedin.com" in domain or "linkedin" in title.lower():
            self._extract_linkedin_title(
                title=title,
                url=url,
                domain=domain,
                source_id=source_id,
                target_person_norm=target_person_norm,
                target_org_norm=target_org_norm,
                accumulated_persons=accumulated_persons,
                accumulated_orgs=accumulated_orgs,
            )

        # C. Target Prospect mentions in text
        if prospect.full_name:
            target_regex = re.compile(re.escape(prospect.full_name), re.IGNORECASE)
            for m in target_regex.finditer(text):
                start = max(0, m.start() - 100)
                end = min(len(text), m.end() + 150)
                snippet = text[start:end].strip()

                # Extract roles and orgs associated in this snippet
                extracted_roles, extracted_orgs = self._extract_role_and_org_from_snippet(snippet)

                # Collect employer/target domains
                obs_domains: List[str] = []
                if domain and target_domain and (domain == target_domain or domain.endswith("." + target_domain)):
                    obs_domains.append(target_domain)

                # Collect linkedin
                matching_li = [u for u in found_linkedin_urls if prospect.full_name.lower().replace(" ", "") in u.lower()]

                obs = CandidateObservation(
                    source_id=source_id,
                    url=url,
                    snippet=snippet,
                    extraction_method="target_name_match",
                )
                self._record_person_candidate(
                    raw_name=m.group(0),
                    normalized_name=target_person_norm,
                    observation=obs,
                    domains=obs_domains,
                    roles=extracted_roles or ([prospect.designation] if prospect.designation else []),
                    organizations=extracted_orgs or [prospect.company_name],
                    linkedin_urls=matching_li,
                    accumulated_persons=accumulated_persons,
                )

        # D. Target Organization mentions or domain matching
        if domain and target_domain and (domain == target_domain or domain.endswith("." + target_domain)):
            obs = CandidateObservation(
                source_id=source_id,
                url=url,
                snippet=title or f"Domain match: {domain}",
                extraction_method="target_domain_match",
            )
            self._record_org_candidate(
                raw_name=prospect.company_name,
                normalized_name=target_org_norm,
                observation=obs,
                domains=[domain, target_domain],
                accumulated_orgs=accumulated_orgs,
            )
        elif prospect.company_name and re.search(r"\b" + re.escape(prospect.company_name) + r"\b", text, re.IGNORECASE):
            obs = CandidateObservation(
                source_id=source_id,
                url=url,
                snippet=title or f"Organization mention: {prospect.company_name}",
                extraction_method="target_org_match",
            )
            self._record_org_candidate(
                raw_name=prospect.company_name,
                normalized_name=target_org_norm,
                observation=obs,
                domains=[domain] if domain else [],
                accumulated_orgs=accumulated_orgs,
            )

        # E. Executive Role Patterns in Text
        for pattern in ROLE_PATTERNS:
            for match in pattern.finditer(text):
                raw_person = match.group("name").strip()
                raw_role = match.group("role").strip()
                norm_p = normalize_person_name(raw_person)

                # Filter out noise
                if self._is_noise(raw_person, norm_p):
                    continue

                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 100)
                snippet = text[start:end].strip()

                extracted_roles, extracted_orgs = self._extract_role_and_org_from_snippet(snippet)
                all_roles = list(set([raw_role] + extracted_roles))

                obs = CandidateObservation(
                    source_id=source_id,
                    url=url,
                    snippet=snippet,
                    extraction_method="role_pattern",
                )
                self._record_person_candidate(
                    raw_name=raw_person,
                    normalized_name=norm_p,
                    observation=obs,
                    domains=[domain] if domain else [],
                    roles=all_roles,
                    organizations=extracted_orgs,
                    accumulated_persons=accumulated_persons,
                )

    def _extract_linkedin_title(
        self,
        title: str,
        url: str,
        domain: str,
        source_id: Optional[str],
        target_person_norm: str,
        target_org_norm: str,
        accumulated_persons: List[_MutableCandidate],
        accumulated_orgs: List[_MutableCandidate],
    ) -> None:
        """Extract candidate from structured title such as 'Jane Doe - VP - Acme | LinkedIn'."""
        clean_title = re.sub(r"\s*\|\s*LinkedIn\b", "", title, flags=re.IGNORECASE).strip()
        parts = [p.strip() for p in re.split(r"\s*[-–|]\s*", clean_title) if p.strip()]

        if len(parts) >= 1:
            raw_person = parts[0]
            norm_person = normalize_person_name(raw_person)
            if not self._is_noise(raw_person, norm_person):
                role = parts[1] if len(parts) >= 2 else None
                org = parts[2] if len(parts) >= 3 else None

                obs = CandidateObservation(
                    source_id=source_id,
                    url=url,
                    snippet=title,
                    extraction_method="linkedin_title",
                )
                self._record_person_candidate(
                    raw_name=raw_person,
                    normalized_name=norm_person,
                    observation=obs,
                    domains=[domain] if domain else [],
                    roles=[role] if role else [],
                    organizations=[org] if org else [],
                    linkedin_urls=[url] if "linkedin.com" in url else [],
                    accumulated_persons=accumulated_persons,
                )

                if org:
                    norm_org = normalize_org_name(org)
                    if not self._is_noise(org, norm_org):
                        org_obs = CandidateObservation(
                            source_id=source_id,
                            url=url,
                            snippet=title,
                            extraction_method="linkedin_title_org",
                        )
                        self._record_org_candidate(
                            raw_name=org,
                            normalized_name=norm_org,
                            observation=org_obs,
                            domains=[],
                            accumulated_orgs=accumulated_orgs,
                        )

    def _extract_role_and_org_from_snippet(self, snippet: str) -> Tuple[List[str], List[str]]:
        """Extract roles and org mentions from a contextual snippet."""
        roles: List[str] = []
        orgs: List[str] = []

        # Role extraction
        role_matches = re.findall(
            r"\b(CEO|CTO|CFO|COO|President|Founder|Co-Founder|VP of [A-Za-z\s]+|Director of [A-Za-z\s]+|Head of [A-Za-z\s]+|Chief [A-Za-z\s]+ Officer)\b",
            snippet,
            re.IGNORECASE,
        )
        for rm in role_matches:
            roles.append(rm.strip())

        # Org extraction: 'at Acme', 'of Acme', 'joined Acme'
        org_matches = re.findall(
            r"\b(?:at|of|joined|with)\s+([A-Z][A-Za-z0-9&]+(?:\s+[A-Z][A-Za-z0-9&]+){0,2})\b",
            snippet,
        )
        for om in org_matches:
            norm_om = normalize_org_name(om)
            if not self._is_noise(om, norm_om):
                orgs.append(om.strip())

        return roles, orgs

    def _is_noise(self, raw: str, normalized: str) -> bool:
        """Check if an entity name is generic noise or boilerplate."""
        if not normalized or len(normalized) <= 2:
            return True
        if normalized in NOISE_WORDS:
            return True
        if raw.isdigit():
            return True
        # If all tokens are noise words
        tokens = normalized.split()
        if all(t in NOISE_WORDS for t in tokens):
            return True
        return False

    def _record_person_candidate(
        self,
        raw_name: str,
        normalized_name: str,
        observation: CandidateObservation,
        domains: List[str],
        roles: List[str],
        organizations: List[str],
        accumulated_persons: List[_MutableCandidate],
        linkedin_urls: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
    ) -> None:
        """Consolidate or create person candidate, preserving distinct candidates on conflicts."""
        clean_orgs = {normalize_org_name(o) for o in organizations if o}
        clean_domains = {clean_domain(d) for d in domains if d}
        clean_lus = set(linkedin_urls or [])

        # Check existing candidates for safe consolidation
        for existing in accumulated_persons:
            # 1. Names must match normalized
            if existing.normalized_name != normalized_name:
                continue

            # 2. Check for conflict in organization affiliation
            existing_clean_orgs = {normalize_org_name(o) for o in existing.organizations if o}
            if clean_orgs and existing_clean_orgs and not (clean_orgs & existing_clean_orgs):
                # Conflicting organizations: do not merge! Keep separate.
                continue

            # 3. Check for conflict in domains
            # If organizations match, different publisher/article domains do not constitute a conflict
            if clean_orgs and existing_clean_orgs and (clean_orgs & existing_clean_orgs):
                pass
            elif clean_domains and existing.domains and not (clean_domains & existing.domains):
                # Conflicting domains without matching org: do not merge! Keep separate.
                continue

            # 4. Check for conflict in LinkedIn URLs
            if clean_lus and existing.linkedin_urls and not (clean_lus & existing.linkedin_urls):
                # Conflicting LinkedIn profiles: do not merge! Keep separate.
                continue

            # Compatible: consolidate into existing candidate
            existing.add_observation(
                observation=observation,
                domains=domains,
                roles=roles,
                organizations=organizations,
                linkedin_urls=linkedin_urls,
                locations=locations,
            )
            return

        # No compatible candidate found: create a new candidate
        candidate = _MutableCandidate(
            entity_type=EntityType.PERSON,
            raw_name=raw_name,
            normalized_name=normalized_name,
        )
        candidate.add_observation(
            observation=observation,
            domains=domains,
            roles=roles,
            organizations=organizations,
            linkedin_urls=linkedin_urls,
            locations=locations,
        )
        accumulated_persons.append(candidate)

    def _record_org_candidate(
        self,
        raw_name: str,
        normalized_name: str,
        observation: CandidateObservation,
        domains: List[str],
        accumulated_orgs: List[_MutableCandidate],
    ) -> None:
        """Consolidate or create organization candidate."""
        clean_domains = {clean_domain(d) for d in domains if d}

        for existing in accumulated_orgs:
            # Merge if normalized name matches OR domain matches
            name_match = existing.normalized_name == normalized_name
            domain_match = bool(clean_domains and existing.domains and (clean_domains & existing.domains))

            if name_match or domain_match:
                existing.add_observation(
                    observation=observation,
                    domains=domains,
                )
                return

        candidate = _MutableCandidate(
            entity_type=EntityType.ORGANIZATION,
            raw_name=raw_name,
            normalized_name=normalized_name,
        )
        candidate.add_observation(
            observation=observation,
            domains=domains,
        )
        accumulated_orgs.append(candidate)
