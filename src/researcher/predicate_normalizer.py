from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.molecular_models import Modality, NormalizedPredicate, Polarity


class PredicateNormalizationResult(BaseModel):
    """Outcome of attempting to normalize a raw predicate into controlled taxonomy."""

    model_config = ConfigDict(frozen=True)

    raw_predicate: str
    normalized_predicate: Optional[NormalizedPredicate] = None
    is_normalized: bool = False
    modality: Modality = Modality.DEFINITIVE
    polarity: Polarity = Polarity.POSITIVE


class PredicateNormalizer:
    """Conservative, deterministic predicate normalizer for core business events."""

    # Unambiguous verb phrase mappings to (NormalizedPredicate, Modality, Polarity)
    _EXACT_MAPPINGS: Dict[str, Tuple[NormalizedPredicate, Modality, Polarity]] = {
        # Headcount & Hiring Planned vs Executed
        "plans to hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "plan to hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "planning to hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "plans to add": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "targeting hiring": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "targets hiring": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "aims to recruit": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "aims to hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "expects to add": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "looking to hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "will hire": (NormalizedPredicate.HIRING_PLANNED, Modality.PLANNED, Polarity.POSITIVE),
        "hire": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "hired": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "recruited": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "onboarded": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "added employees": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "brought on": (NormalizedPredicate.HIRING_EXECUTED, Modality.DEFINITIVE, Polarity.POSITIVE),

        # Headcount Metrics
        "employs": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "employing": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "has headcount of": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "headcount of": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "workforce of": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "team of": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "reached headcount": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),
        "staff of": (NormalizedPredicate.EMPLOYEE_COUNT, Modality.DEFINITIVE, Polarity.POSITIVE),

        # Funding & Capital
        "raised": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "secured funding": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "closed round": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "bagged funding": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "bagged investment": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "raised funding": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "secured investment": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "announced funding": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "closed series": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "raised series": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "secured series": (NormalizedPredicate.FUNDING_RAISED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "plans to raise": (NormalizedPredicate.FUNDING_RAISED, Modality.PLANNED, Polarity.POSITIVE),

        # Facility & Geographic Expansion
        "open": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "opened": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "inaugurated": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "launched facility": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "opened facility": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "opened office": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "set up office": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "established office": (NormalizedPredicate.FACILITY_OPENED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "plans to open": (NormalizedPredicate.FACILITY_OPENED, Modality.PLANNED, Polarity.POSITIVE),
        "planning to open": (NormalizedPredicate.FACILITY_OPENED, Modality.PLANNED, Polarity.POSITIVE),

        "close": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "closed": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "closed office": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "closed facility": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "shut down facility": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "shuttered office": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "exited": (NormalizedPredicate.FACILITY_CLOSED, Modality.DEFINITIVE, Polarity.POSITIVE),

        "headquartered in": (NormalizedPredicate.HEADQUARTERED_IN, Modality.DEFINITIVE, Polarity.POSITIVE),
        "based in": (NormalizedPredicate.HEADQUARTERED_IN, Modality.DEFINITIVE, Polarity.POSITIVE),
        "headquarters in": (NormalizedPredicate.HEADQUARTERED_IN, Modality.DEFINITIVE, Polarity.POSITIVE),

        "operates in": (NormalizedPredicate.OPERATING_IN, Modality.DEFINITIVE, Polarity.POSITIVE),
        "operating in": (NormalizedPredicate.OPERATING_IN, Modality.DEFINITIVE, Polarity.POSITIVE),
        "present in": (NormalizedPredicate.OPERATING_IN, Modality.DEFINITIVE, Polarity.POSITIVE),

        # Corporate Development / M&A / Partnership
        "acquire": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "acquired": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "bought": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "purchased": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "took over": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "completed acquisition of": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "plans to acquire": (NormalizedPredicate.ACQUISITION_ACQUIRED, Modality.PLANNED, Polarity.POSITIVE),

        "partner": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "partnered with": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "formed partnership with": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "entered agreement with": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "teamed up with": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "collaborated with": (NormalizedPredicate.PARTNERSHIP_FORMED, Modality.DEFINITIVE, Polarity.POSITIVE),

        # Leadership & Executive Changes
        "appoint": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "appointed": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "named": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "joined as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "joined": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "promoted to": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "serves as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "serving as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "elected as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "appointed as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "hired as": (NormalizedPredicate.LEADERSHIP_APPOINTED, Modality.DEFINITIVE, Polarity.POSITIVE),

        "resign": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "resigned": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "stepped down": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "left": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "departed": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "ousted": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "fired": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "resigned as": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "stepped down as": (NormalizedPredicate.LEADERSHIP_DEPARTED, Modality.DEFINITIVE, Polarity.POSITIVE),

        # Product / Tech Launch
        "launch": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "launched": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "released": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "unveiled": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "introduced": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "rolled out": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "shipped": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.DEFINITIVE, Polarity.POSITIVE),
        "plans to launch": (NormalizedPredicate.PRODUCT_LAUNCHED, Modality.PLANNED, Polarity.POSITIVE),
    }

    # Explicit negation markers that clearly invert polarity
    _NEGATION_PREFIXES = (
        "did not ", "does not ", "will not ", "has not ", "have not ", "no plans to ", "denied plans to ", "never "
    )

    # Explicit conditional markers
    _CONDITIONAL_MARKERS = ("subject to", "pending approval", "contingent upon", "conditioned on", "conditional on")

    def normalize(self, raw_predicate: str) -> PredicateNormalizationResult:
        """Deterministically normalize raw predicate phrase into controlled vocabulary."""
        cleaned = raw_predicate.strip().lower()
        if not cleaned:
            return PredicateNormalizationResult(
                raw_predicate=raw_predicate,
                normalized_predicate=None,
                is_normalized=False,
                modality=Modality.DEFINITIVE,
                polarity=Polarity.POSITIVE,
            )

        # 1. Check for explicit negation prefix
        polarity = Polarity.POSITIVE
        modality_override: Optional[Modality] = None
        predicate_core = cleaned
        for neg in self._NEGATION_PREFIXES:
            if cleaned.startswith(neg):
                polarity = Polarity.NEGATIVE
                predicate_core = cleaned[len(neg):].strip()
                if "plans to" in neg:
                    modality_override = Modality.PLANNED
                break

        # 2. Check for conditional marker and strip conditional clause
        for cond in self._CONDITIONAL_MARKERS:
            if cond in predicate_core:
                modality_override = Modality.CONDITIONAL
                predicate_core = re.split(rf"\b{re.escape(cond)}\b", predicate_core)[0].strip()
                break

        # 3. Direct match on core predicate in exact dictionary
        if predicate_core in self._EXACT_MAPPINGS:
            norm_pred, base_modality, _ = self._EXACT_MAPPINGS[predicate_core]
            if modality_override == Modality.PLANNED and norm_pred == NormalizedPredicate.HIRING_EXECUTED:
                norm_pred = NormalizedPredicate.HIRING_PLANNED
            return PredicateNormalizationResult(
                raw_predicate=raw_predicate,
                normalized_predicate=norm_pred,
                is_normalized=True,
                modality=modality_override or base_modality,
                polarity=polarity,
            )

        # 4. Check for leading "plans to / planning to / targets / aims to" prefix if not already in dictionary
        if predicate_core.startswith("plans to ") or predicate_core.startswith("planning to "):
            sub_verb = re.sub(r"^(?:plans to|planning to)\s+", "", predicate_core).strip()
            if sub_verb in self._EXACT_MAPPINGS:
                norm_pred, _, _ = self._EXACT_MAPPINGS[sub_verb]
                # Invariant: Planned actions must carry Modality.PLANNED!
                # For hiring, map to HIRING_PLANNED
                if norm_pred == NormalizedPredicate.HIRING_EXECUTED:
                    norm_pred = NormalizedPredicate.HIRING_PLANNED
                return PredicateNormalizationResult(
                    raw_predicate=raw_predicate,
                    normalized_predicate=norm_pred,
                    is_normalized=True,
                    modality=Modality.PLANNED,
                    polarity=polarity,
                )

        # 5. Fallback for unmapped predicates (First-class citizen)
        # Preserve raw wording, indicate is_normalized=False, do not force category
        inferred_modality = modality_override or Modality.DEFINITIVE
        if any(w in predicate_core for w in ("plans to", "targets", "aims to", "expects to", "intends to", "will ")):
            inferred_modality = Modality.PLANNED

        return PredicateNormalizationResult(
            raw_predicate=raw_predicate,
            normalized_predicate=None,
            is_normalized=False,
            modality=inferred_modality,
            polarity=polarity,
        )
