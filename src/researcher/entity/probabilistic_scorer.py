from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.context.models import ProspectContext
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    clean_domain,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.entity.probabilistic_models import (
    CandidateMatchAssessment,
    FeatureScore,
    ResolutionOutcome,
)
from src.researcher.models import EntityRecord, EntityType

logger = logging.getLogger(__name__)

# Common English nickname pairs (bidirectional)
NICKNAMES: Dict[str, Set[str]] = {
    "robert": {"bob", "rob", "bobby", "robbie"},
    "william": {"bill", "billy", "will", "willy", "liam"},
    "james": {"jim", "jimmy", "jamie"},
    "michael": {"mike", "mikey"},
    "david": {"dave", "davy"},
    "daniel": {"dan", "danny"},
    "christopher": {"chris"},
    "alexander": {"alex", "alec"},
    "thomas": {"tom", "tommy"},
    "richard": {"rick", "ricky", "dick"},
    "stephen": {"steve", "steven"},
    "anthony": {"tony"},
    "samuel": {"sam", "sammy"},
    "benjamin": {"ben", "benny"},
    "joseph": {"joe", "joey"},
    "matthew": {"matt", "matty"},
    "nicholas": {"nick", "nicky"},
    "andrew": {"andy", "drew"},
    "edward": {"ed", "eddie", "ted", "teddy"},
    "elizabeth": {"liz", "beth", "eliza", "lizzie"},
    "jennifer": {"jen", "jenny"},
    "katherine": {"kate", "katie", "kathy"},
    "margaret": {"maggie", "meg", "peggy"},
    "jonathan": {"john", "jon"},
    "charles": {"charlie", "chuck"},
    "kenneth": {"ken", "kenny"},
}


def are_names_nickname_match(name_a: str, name_b: str) -> bool:
    """Check if two first names represent standard informal nickname variations."""
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    if not a or not b or a == b:
        return False
    if b in NICKNAMES.get(a, set()):
        return True
    if a in NICKNAMES.get(b, set()):
        return True
    for canonical, nicks in NICKNAMES.items():
        if a in nicks and b in nicks:
            return True
    return False


def compute_levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio between 0.0 and 1.0.

    Pure Python dynamic programming implementation with O(min(N, M)) memory.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)

    # Ensure s1 is shorter
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    previous_row = list(range(len1 + 1))
    for i, c2 in enumerate(s2):
        current_row = [i + 1] * (len1 + 1)
        for j, c1 in enumerate(s1):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    dist = previous_row[len1]
    return max(0.0, 1.0 - (dist / max_len))


def compute_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def is_acronym_match(short_text: str, long_text: str) -> bool:
    """Check if short_text forms an acronym of long_text."""
    s = re.sub(r"[^\w]", "", short_text).lower()
    tokens = [t.lower() for t in re.split(r"[\s\-_]+", long_text) if t]
    if len(s) < 2 or len(tokens) < len(s):
        return False
    acronym = "".join(t[0] for t in tokens if t)
    return s == acronym or s in acronym


def are_locations_compatible(loc_a: str, loc_b: str) -> Tuple[bool, float]:
    """Evaluate geographical compatibility between two location strings.

    Returns:
        (is_compatible, similarity_score)
    """
    if not loc_a or not loc_b:
        return (True, 0.0)  # Missing, neutral

    clean_a = re.sub(r"[^\w\s]", " ", loc_a.lower())
    clean_b = re.sub(r"[^\w\s]", " ", loc_b.lower())
    toks_a = {t for t in clean_a.split() if len(t) > 2}
    toks_b = {t for t in clean_b.split() if len(t) > 2}

    if not toks_a or not toks_b:
        return (True, 0.0)

    # Identical
    if clean_a.strip() == clean_b.strip():
        return (True, 1.0)

    # Substring / containment (e.g. "San Francisco, CA" and "California")
    if clean_a.strip() in clean_b.strip() or clean_b.strip() in clean_a.strip():
        return (True, 0.8)

    # Token overlap
    overlap = len(toks_a & toks_b)
    if overlap > 0:
        ratio = overlap / min(len(toks_a), len(toks_b))
        return (True, min(1.0, 0.6 + 0.4 * ratio))

    # Known mutually exclusive distinct countries / major regions
    return (False, 0.0)


class ProbabilisticCandidateScorer:
    """Explainable candidate pair matching and feature attribution engine.

    Design Principles:
      - Controlled Feature Groups: Correlated signals (e.g. exact name, token Jaccard,
        edit distance) are grouped into a single Name Evidence score so redundant features
        cannot be counted as independent corroborating evidence.
      - Entity-Type-Specific Weighting: Different weights and feature applicability are used
        for Person vs Organization resolution. For example, domain matching is primary evidence
        for organizations, but contextual employer evidence for persons.
      - Explicit Missing Attribute Handling: Unobserved attributes are recorded as missing
        (neutral), NOT penalized as negative evidence.
      - Bounded Heuristic Scoring: The composite score is explicitly a `match_score` in [0.0, 1.0],
        not presented as a calibrated statistical posterior probability.
    """

    def __init__(
        self,
        person_name_weight: float = 0.45,
        person_org_weight: float = 0.35,
        person_ctx_weight: float = 0.20,
        org_name_weight: float = 0.45,
        org_domain_weight: float = 0.40,
        org_ctx_weight: float = 0.15,
    ) -> None:
        self.person_name_weight = person_name_weight
        self.person_org_weight = person_org_weight
        self.person_ctx_weight = person_ctx_weight
        self.org_name_weight = org_name_weight
        self.org_domain_weight = org_domain_weight
        self.org_ctx_weight = org_ctx_weight

    def score_candidate(
        self,
        candidate: DiscoveredEntityCandidate,
        target_prospect: Optional[ProspectContext] = None,
        target_entity: Optional[EntityRecord] = None,
    ) -> CandidateMatchAssessment:
        """Evaluate a candidate against a target prospect or entity record."""
        target_type = (
            EntityType.PERSON
            if target_prospect is not None
            else (target_entity.entity_type if target_entity else candidate.entity_type)
        )

        target_name = (
            target_prospect.full_name
            if target_prospect
            else (target_entity.raw_name if target_entity else None)
        )

        target_id = (
            target_entity.entity_id
            if target_entity
            else (getattr(target_prospect, "context_id", getattr(target_prospect, "prospect_id", None)) if target_prospect else None)
        )

        # 1. Hard Disqualification: Incompatible Entity Types
        if candidate.entity_type != target_type:
            disq_feature = FeatureScore(
                feature_name="entity_type_compatibility",
                feature_group="negative",
                raw_target_value=target_type.value,
                raw_candidate_value=candidate.entity_type.value,
                similarity=0.0,
                weight=1.0,
                contribution=-1.0,
                is_missing=False,
                is_negative_penalty=True,
                rationale=f"Incompatible entity types: Target is {target_type.value}, Candidate is {candidate.entity_type.value}",
            )
            return CandidateMatchAssessment(
                candidate_id=candidate.candidate_id,
                target_id=target_id,
                candidate_name=candidate.raw_name,
                target_name=target_name,
                entity_type=candidate.entity_type,
                match_score=0.0,
                feature_scores=[disq_feature],
                positive_evidence=[],
                negative_evidence=[f"Incompatible entity types ({target_type.value} vs {candidate.entity_type.value})"],
                missing_features=[],
                outcome=ResolutionOutcome.NO_MATCH,
                is_disqualified=True,
                rationale="Candidate disqualified due to entity type mismatch.",
            )

        if target_type == EntityType.PERSON:
            return self._score_person_candidate(
                candidate=candidate,
                target_prospect=target_prospect,
                target_entity=target_entity,
                target_id=target_id,
                target_name=target_name,
            )
        else:
            return self._score_organization_candidate(
                candidate=candidate,
                target_prospect=target_prospect,
                target_entity=target_entity,
                target_id=target_id,
                target_name=target_name,
            )

    def _score_person_candidate(
        self,
        candidate: DiscoveredEntityCandidate,
        target_prospect: Optional[ProspectContext],
        target_entity: Optional[EntityRecord],
        target_id: Optional[str],
        target_name: Optional[str],
    ) -> CandidateMatchAssessment:
        feature_scores: List[FeatureScore] = []
        positive_evidence: List[str] = []
        negative_evidence: List[str] = []
        missing_features: List[str] = []

        # Target attributes
        t_name = (target_prospect.full_name if target_prospect else (target_entity.raw_name if target_entity else ""))
        t_org = target_prospect.company_name if target_prospect else ""
        t_dom = clean_domain(target_prospect.company_domain) if (target_prospect and target_prospect.company_domain) else ""
        t_role = target_prospect.designation if target_prospect else ""
        t_loc = target_prospect.location if target_prospect else ""
        t_linkedin = target_prospect.linkedin_url if target_prospect else (target_entity.linkedin_url if target_entity else None)

        # ---------------------------------------------------------
        # Group 1: Person Name Evidence (Combined Max Weight: self.person_name_weight)
        # ---------------------------------------------------------
        norm_t_name = normalize_person_name(t_name)
        norm_c_name = candidate.normalized_name or normalize_person_name(candidate.raw_name)

        t_tokens = [t for t in norm_t_name.split() if t]
        c_tokens = [t for t in norm_c_name.split() if t]

        name_exact = (norm_t_name == norm_c_name and bool(norm_t_name))
        lev_sim = compute_levenshtein_similarity(norm_t_name, norm_c_name)
        tok_jaccard = compute_token_jaccard(set(t_tokens), set(c_tokens))

        # Check nickname on first names if last names match
        nickname_match = False
        if len(t_tokens) >= 2 and len(c_tokens) >= 2:
            if t_tokens[-1] == c_tokens[-1]:  # same surname
                if are_names_nickname_match(t_tokens[0], c_tokens[0]):
                    nickname_match = True

        if name_exact:
            name_score = 1.0
            name_desc = "Exact normalized name match"
            positive_evidence.append(f"Exact person name match ('{candidate.raw_name}')")
        elif nickname_match:
            name_score = 0.90
            name_desc = f"First name nickname match with matching surname ('{norm_c_name}' vs '{norm_t_name}')"
            positive_evidence.append(name_desc)
        else:
            # Controlled non-redundant combination: blend token overlap and edit distance
            name_score = 0.60 * tok_jaccard + 0.40 * lev_sim
            name_desc = f"Fuzzy name alignment (Jaccard: {tok_jaccard:.2f}, Levenshtein: {lev_sim:.2f})"
            if name_score >= 0.70:
                positive_evidence.append(f"Fuzzy name alignment ('{candidate.raw_name}', similarity: {name_score:.2f})")
            elif name_score < 0.35:
                negative_evidence.append(f"Substantial name disparity ('{candidate.raw_name}' vs '{t_name}')")

        name_contribution = name_score * self.person_name_weight
        feature_scores.append(
            FeatureScore(
                feature_name="person_name_group",
                feature_group="name",
                raw_target_value=t_name,
                raw_candidate_value=candidate.raw_name,
                similarity=round(name_score, 3),
                weight=self.person_name_weight,
                contribution=round(name_contribution, 3),
                is_missing=False,
                is_negative_penalty=False,
                rationale=name_desc,
            )
        )

        # ---------------------------------------------------------
        # Group 2: Employer Organization & Domain Evidence (Combined Max Weight: self.person_org_weight)
        # Note: Person's employer is contextual evidence, not identity proof of person.
        # ---------------------------------------------------------
        has_target_org_info = bool(t_org or t_dom)
        has_cand_org_info = bool(candidate.organizations or candidate.domains)

        if not has_target_org_info or not has_cand_org_info:
            missing_features.append("employer_organization_evidence")
            org_score = 0.0
            org_contribution = 0.0
            feature_scores.append(
                FeatureScore(
                    feature_name="person_employer_group",
                    feature_group="organization",
                    raw_target_value=f"{t_org or ''} ({t_dom or ''})",
                    raw_candidate_value=f"{','.join(candidate.organizations)} ({','.join(candidate.domains)})",
                    similarity=0.0,
                    weight=self.person_org_weight,
                    contribution=0.0,
                    is_missing=True,
                    is_negative_penalty=False,
                    rationale="Employer organization or candidate affiliation unobserved (neutral prior)",
                )
            )
        else:
            # Check domain match
            cand_domains = {clean_domain(d) for d in candidate.domains if d}
            domain_match = bool(t_dom and t_dom in cand_domains)

            # Check org name match
            norm_t_org = normalize_org_name(t_org) if t_org else ""
            cand_orgs_norm = {normalize_org_name(o) for o in candidate.organizations if o}
            org_name_match = bool(norm_t_org and norm_t_org in cand_orgs_norm)

            # Check fuzzy org name
            best_org_sim = 0.0
            if norm_t_org:
                for c_org in cand_orgs_norm:
                    sim = compute_levenshtein_similarity(norm_t_org, c_org)
                    if sim > best_org_sim:
                        best_org_sim = sim

            if domain_match and org_name_match:
                org_score = 1.0
                org_desc = f"Confirmed employer organization name ('{t_org}') and domain ('{t_dom}')"
                positive_evidence.append(org_desc)
            elif domain_match:
                org_score = 0.95
                org_desc = f"Confirmed employer domain match ('{t_dom}')"
                positive_evidence.append(org_desc)
            elif org_name_match:
                org_score = 0.90
                org_desc = f"Confirmed employer organization name match ('{t_org}')"
                positive_evidence.append(org_desc)
            elif best_org_sim >= 0.80:
                org_score = best_org_sim * 0.85
                org_desc = f"Fuzzy employer organization match (similarity: {best_org_sim:.2f})"
                positive_evidence.append(org_desc)
            else:
                # Conflicting verified employer!
                org_score = 0.0
                org_desc = (
                    f"Conflicting employer: Target is at '{t_org}' ({t_dom}), "
                    f"Candidate is at '{', '.join(candidate.organizations)}' ({', '.join(candidate.domains)})"
                )
                negative_evidence.append(org_desc)

            org_contribution = org_score * self.person_org_weight
            feature_scores.append(
                FeatureScore(
                    feature_name="person_employer_group",
                    feature_group="organization",
                    raw_target_value=f"{t_org or ''} ({t_dom or ''})",
                    raw_candidate_value=f"{','.join(candidate.organizations)} ({','.join(candidate.domains)})",
                    similarity=round(org_score, 3),
                    weight=self.person_org_weight,
                    contribution=round(org_contribution, 3),
                    is_missing=False,
                    is_negative_penalty=False,
                    rationale=org_desc,
                )
            )

        # ---------------------------------------------------------
        # Group 3: Contextual Evidence (Role, Location, LinkedIn) (Combined Max Weight: self.person_ctx_weight)
        # ---------------------------------------------------------
        ctx_sub_scores: List[float] = []

        # Role
        if t_role and candidate.roles:
            norm_t_role = re.sub(r"[^\w\s]", " ", t_role.lower())
            best_role_sim = 0.0
            for r in candidate.roles:
                norm_c_role = re.sub(r"[^\w\s]", " ", r.lower())
                sim = compute_token_jaccard(set(norm_t_role.split()), set(norm_c_role.split()))
                if sim > best_role_sim:
                    best_role_sim = sim
            if best_role_sim >= 0.70:
                positive_evidence.append(f"Compatible professional role ('{candidate.roles[0]}')")
            ctx_sub_scores.append(best_role_sim)
        else:
            missing_features.append("role_title")

        # Location
        if t_loc and candidate.locations:
            is_compat, loc_sim = are_locations_compatible(t_loc, candidate.locations[0])
            if is_compat and loc_sim >= 0.60:
                positive_evidence.append(f"Geographically compatible location ('{candidate.locations[0]}')")
                ctx_sub_scores.append(loc_sim)
            elif not is_compat:
                negative_evidence.append(f"Conflicting locations ('{candidate.locations[0]}' vs '{t_loc}')")
                ctx_sub_scores.append(0.0)
            else:
                ctx_sub_scores.append(loc_sim)
        else:
            missing_features.append("location")

        # LinkedIn Profile URL (direct identifier)
        if t_linkedin and candidate.linkedin_urls:
            clean_t_li = re.sub(r"/+$", "", t_linkedin.strip().lower())
            li_match = any(clean_t_li in re.sub(r"/+$", "", lu.strip().lower()) for lu in candidate.linkedin_urls)
            if li_match:
                positive_evidence.append("Verified LinkedIn profile URL match")
                ctx_sub_scores.append(1.0)
            else:
                ctx_sub_scores.append(0.0)
        elif not t_linkedin and not candidate.linkedin_urls:
            pass
        else:
            missing_features.append("linkedin_url")

        if ctx_sub_scores:
            ctx_score = sum(ctx_sub_scores) / len(ctx_sub_scores)
            ctx_desc = f"Contextual alignment across observed attributes: {ctx_score:.2f}"
        else:
            ctx_score = 0.0
            ctx_desc = "Contextual attributes unobserved (neutral prior)"

        ctx_contribution = ctx_score * self.person_ctx_weight
        feature_scores.append(
            FeatureScore(
                feature_name="person_context_group",
                feature_group="context",
                raw_target_value=f"{t_role or ''} | {t_loc or ''}",
                raw_candidate_value=f"{','.join(candidate.roles)} | {','.join(candidate.locations)}",
                similarity=round(ctx_score, 3),
                weight=self.person_ctx_weight,
                contribution=round(ctx_contribution, 3),
                is_missing=len(ctx_sub_scores) == 0,
                is_negative_penalty=False,
                rationale=ctx_desc,
            )
        )

        # ---------------------------------------------------------
        # Group 4: Negative Evidence Penalties
        # ---------------------------------------------------------
        negative_penalty = 0.0
        # If both employer organizations are explicitly known and completely disjoint
        if has_target_org_info and has_cand_org_info and org_score == 0.0:
            penalty = 0.45
            negative_penalty += penalty
            feature_scores.append(
                FeatureScore(
                    feature_name="conflicting_employer_penalty",
                    feature_group="negative",
                    raw_target_value=t_org,
                    raw_candidate_value=",".join(candidate.organizations),
                    similarity=0.0,
                    weight=penalty,
                    contribution=-penalty,
                    is_missing=False,
                    is_negative_penalty=True,
                    rationale=f"Candidate confirmed at contradictory employer organization ('{','.join(candidate.organizations)}')",
                )
            )

        # Calculate Final Bounded Match Score
        positive_sum = name_contribution + org_contribution + ctx_contribution
        raw_score = positive_sum - negative_penalty
        match_score = max(0.0, min(1.0, raw_score))

        # Outcome determination for individual pair
        is_disqualified = (negative_penalty >= 0.40 and match_score < 0.30) or (name_score < 0.20)
        if is_disqualified:
            outcome = ResolutionOutcome.NO_MATCH
            rationale = f"Candidate disqualified by contradictory evidence (Score: {match_score:.2f})."
        elif match_score >= 0.75:
            outcome = ResolutionOutcome.MATCH
            rationale = f"Strong multi-attribute match (Score: {match_score:.2f})."
        elif match_score <= 0.30:
            outcome = ResolutionOutcome.NO_MATCH
            rationale = f"Low alignment across observable attributes (Score: {match_score:.2f})."
        else:
            outcome = ResolutionOutcome.UNRESOLVED
            rationale = f"Insufficient evidence to confirm or refute identity (Score: {match_score:.2f})."

        return CandidateMatchAssessment(
            candidate_id=candidate.candidate_id,
            target_id=target_id,
            candidate_name=candidate.raw_name,
            target_name=target_name,
            entity_type=EntityType.PERSON,
            match_score=round(match_score, 3),
            feature_scores=feature_scores,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            missing_features=missing_features,
            outcome=outcome,
            is_disqualified=is_disqualified,
            rationale=rationale,
        )

    def _score_organization_candidate(
        self,
        candidate: DiscoveredEntityCandidate,
        target_prospect: Optional[ProspectContext],
        target_entity: Optional[EntityRecord],
        target_id: Optional[str],
        target_name: Optional[str],
    ) -> CandidateMatchAssessment:
        feature_scores: List[FeatureScore] = []
        positive_evidence: List[str] = []
        negative_evidence: List[str] = []
        missing_features: List[str] = []

        t_name = (
            target_prospect.company_name
            if target_prospect
            else (target_entity.raw_name if target_entity else "")
        )
        t_dom = (
            clean_domain(target_prospect.company_domain)
            if (target_prospect and target_prospect.company_domain)
            else (clean_domain(target_entity.domain) if (target_entity and target_entity.domain) else "")
        )
        t_loc = target_prospect.location if target_prospect else ""

        # ---------------------------------------------------------
        # Group 1: Organization Name Evidence (Weight: self.org_name_weight)
        # ---------------------------------------------------------
        norm_t_name = normalize_org_name(t_name)
        norm_c_name = candidate.normalized_name or normalize_org_name(candidate.raw_name)

        exact_org_name = (norm_t_name == norm_c_name and bool(norm_t_name))
        lev_sim = compute_levenshtein_similarity(norm_t_name, norm_c_name)
        tok_jaccard = compute_token_jaccard(set(norm_t_name.split()), set(norm_c_name.split()))
        acronym_match = is_acronym_match(norm_t_name, norm_c_name) or is_acronym_match(norm_c_name, norm_t_name)

        if exact_org_name:
            name_score = 1.0
            name_desc = "Exact normalized organization name match"
            positive_evidence.append(f"Exact organization name match ('{candidate.raw_name}')")
        elif acronym_match:
            name_score = 0.90
            name_desc = f"Organization acronym match ('{candidate.raw_name}' vs '{t_name}')"
            positive_evidence.append(name_desc)
        else:
            name_score = 0.60 * tok_jaccard + 0.40 * lev_sim
            name_desc = f"Fuzzy organization name similarity: {name_score:.2f}"
            if name_score >= 0.75:
                positive_evidence.append(f"Strong organization name similarity ('{candidate.raw_name}', {name_score:.2f})")
            elif name_score < 0.35:
                negative_evidence.append(f"Organization name disparity ('{candidate.raw_name}' vs '{t_name}')")

        name_contribution = name_score * self.org_name_weight
        feature_scores.append(
            FeatureScore(
                feature_name="org_name_group",
                feature_group="name",
                raw_target_value=t_name,
                raw_candidate_value=candidate.raw_name,
                similarity=round(name_score, 3),
                weight=self.org_name_weight,
                contribution=round(name_contribution, 3),
                is_missing=False,
                is_negative_penalty=False,
                rationale=name_desc,
            )
        )

        # ---------------------------------------------------------
        # Group 2: Organization Domain Evidence (Weight: self.org_domain_weight)
        # ---------------------------------------------------------
        has_target_dom = bool(t_dom)
        cand_domains = {clean_domain(d) for d in candidate.domains if d}
        has_cand_dom = bool(cand_domains)

        if not has_target_dom or not has_cand_dom:
            missing_features.append("organization_web_domain")
            dom_score = 0.0
            dom_desc = "Web domain unobserved on one or both records (neutral prior)"
        else:
            if t_dom in cand_domains:
                dom_score = 1.0
                dom_desc = f"Exact verified web domain match ('{t_dom}')"
                positive_evidence.append(dom_desc)
            else:
                # Subdomain or parent domain relationship
                subdomain_match = any(
                    cd.endswith("." + t_dom) or t_dom.endswith("." + cd)
                    for cd in cand_domains
                )
                if subdomain_match:
                    dom_score = 0.85
                    dom_desc = f"Subdomain or parent domain alignment ('{t_dom}' vs '{','.join(cand_domains)}')"
                    positive_evidence.append(dom_desc)
                else:
                    dom_score = 0.0
                    dom_desc = f"Conflicting verified domains: Target '{t_dom}' vs Candidate '{','.join(cand_domains)}'"
                    negative_evidence.append(dom_desc)

        dom_contribution = dom_score * self.org_domain_weight
        feature_scores.append(
            FeatureScore(
                feature_name="org_domain_group",
                feature_group="domain",
                raw_target_value=t_dom,
                raw_candidate_value=",".join(cand_domains),
                similarity=round(dom_score, 3),
                weight=self.org_domain_weight,
                contribution=round(dom_contribution, 3),
                is_missing=not has_target_dom or not has_cand_dom,
                is_negative_penalty=False,
                rationale=dom_desc,
            )
        )

        # ---------------------------------------------------------
        # Group 3: Contextual Evidence (Location, Leadership) (Weight: self.org_ctx_weight)
        # Shared address or executive does NOT prove two organizations are identical!
        # ---------------------------------------------------------
        ctx_sub_scores: List[float] = []
        if t_loc and candidate.locations:
            is_compat, loc_sim = are_locations_compatible(t_loc, candidate.locations[0])
            # Shared location capped at 0.50 contextual support to prevent conflating coworking/same building
            capped_loc_sim = loc_sim * 0.50
            ctx_sub_scores.append(capped_loc_sim)
            if is_compat and loc_sim >= 0.70:
                positive_evidence.append(f"Contextual location compatibility ('{candidate.locations[0]}')")
        else:
            missing_features.append("location")

        if ctx_sub_scores:
            ctx_score = sum(ctx_sub_scores) / len(ctx_sub_scores)
            ctx_desc = f"Contextual organization alignment: {ctx_score:.2f}"
        else:
            ctx_score = 0.0
            ctx_desc = "Contextual attributes unobserved (neutral prior)"

        ctx_contribution = ctx_score * self.org_ctx_weight
        feature_scores.append(
            FeatureScore(
                feature_name="org_context_group",
                feature_group="context",
                raw_target_value=t_loc,
                raw_candidate_value=",".join(candidate.locations),
                similarity=round(ctx_score, 3),
                weight=self.org_ctx_weight,
                contribution=round(ctx_contribution, 3),
                is_missing=len(ctx_sub_scores) == 0,
                is_negative_penalty=False,
                rationale=ctx_desc,
            )
        )

        # ---------------------------------------------------------
        # Group 4: Negative Evidence Penalties
        # ---------------------------------------------------------
        negative_penalty = 0.0
        # Conflicting verified domains for an organization is strong negative evidence
        if has_target_dom and has_cand_dom and dom_score == 0.0:
            penalty = 0.60
            negative_penalty += penalty
            feature_scores.append(
                FeatureScore(
                    feature_name="conflicting_domain_penalty",
                    feature_group="negative",
                    raw_target_value=t_dom,
                    raw_candidate_value=",".join(cand_domains),
                    similarity=0.0,
                    weight=penalty,
                    contribution=-penalty,
                    is_missing=False,
                    is_negative_penalty=True,
                    rationale=f"Conflicting primary web domain ('{t_dom}' vs '{','.join(cand_domains)}')",
                )
            )

        positive_sum = name_contribution + dom_contribution + ctx_contribution
        raw_score = positive_sum - negative_penalty
        match_score = max(0.0, min(1.0, raw_score))

        is_disqualified = (negative_penalty >= 0.50 and match_score < 0.30) or (name_score < 0.20)
        if is_disqualified:
            outcome = ResolutionOutcome.NO_MATCH
            rationale = f"Organization disqualified by contradictory domain/identity evidence (Score: {match_score:.2f})."
        elif match_score >= 0.75:
            outcome = ResolutionOutcome.MATCH
            rationale = f"Strong organization identity match (Score: {match_score:.2f})."
        elif match_score <= 0.30:
            outcome = ResolutionOutcome.NO_MATCH
            rationale = f"Low alignment across organization identity attributes (Score: {match_score:.2f})."
        else:
            outcome = ResolutionOutcome.UNRESOLVED
            rationale = f"Insufficient evidence to confirm or refute organization identity (Score: {match_score:.2f})."

        return CandidateMatchAssessment(
            candidate_id=candidate.candidate_id,
            target_id=target_id,
            candidate_name=candidate.raw_name,
            target_name=target_name,
            entity_type=EntityType.ORGANIZATION,
            match_score=round(match_score, 3),
            feature_scores=feature_scores,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            missing_features=missing_features,
            outcome=outcome,
            is_disqualified=is_disqualified,
            rationale=rationale,
        )
