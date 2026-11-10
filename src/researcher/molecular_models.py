from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from src.researcher.models import ConfidenceLevel, VerificationStatus

logger = logging.getLogger(__name__)

MOLECULAR_SCHEMA_VERSION = 1


class ValueType(str, Enum):
    STRING = "string"
    ENTITY_REF = "entity_ref"
    NUMBER = "number"
    CURRENCY = "currency"
    DATE = "date"
    LOCATION = "location"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class MolecularValue(BaseModel):
    """Explicit, predictable serialization contract for factual objects/complements."""

    model_config = ConfigDict(frozen=True)

    value_type: ValueType = ValueType.STRING
    raw_value: str = Field(description="Original unparsed value string as extracted")
    normalized_value: Optional[Union[float, int, str, bool]] = Field(
        default=None, description="Parsed numeric, boolean, or canonical string representation"
    )
    unit: Optional[str] = Field(
        default=None, description="Measurement unit or currency code (e.g. 'USD', 'EUR', 'engineers', '%')"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value_type": self.value_type.value,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MolecularValue:
        v_type_str = data.get("value_type", ValueType.STRING.value)
        try:
            v_type = ValueType(v_type_str)
        except ValueError:
            v_type = ValueType.UNKNOWN

        return cls(
            value_type=v_type,
            raw_value=str(data.get("raw_value", "")),
            normalized_value=data.get("normalized_value"),
            unit=data.get("unit"),
        )


class Modality(str, Enum):
    DEFINITIVE = "definitive"    # Factual assertion of current state or completed event
    PLANNED = "planned"          # Forward-looking intent, target, plan, forecast
    CONDITIONAL = "conditional"  # Contingent upon external event, approval, or condition


class Polarity(str, Enum):
    POSITIVE = "positive"        # Affirmative assertion
    NEGATIVE = "negative"        # Explicit negation or denial


class NormalizedPredicate(str, Enum):
    # Corporate & Operations
    HEADQUARTERED_IN = "headquartered_in"
    FACILITY_OPENED = "facility_opened"
    FACILITY_CLOSED = "facility_closed"
    OPERATING_IN = "operating_in"

    # Capital & Corporate Development
    FUNDING_RAISED = "funding_raised"
    ACQUISITION_ACQUIRED = "acquisition_acquired"
    PARTNERSHIP_FORMED = "partnership_formed"

    # Headcount & Leadership
    EMPLOYEE_COUNT = "employee_count"
    HIRING_PLANNED = "hiring_planned"
    HIRING_EXECUTED = "hiring_executed"
    LEADERSHIP_APPOINTED = "leadership_appointed"
    LEADERSHIP_DEPARTED = "leadership_departed"

    # Commercial & Product
    PRODUCT_LAUNCHED = "product_launched"


class MolecularFact(BaseModel):
    """An atomic, independently addressable proposition derived from research evidence."""

    model_config = ConfigDict(frozen=True)

    molecular_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_canonical_id: str = Field(description="Foreign key to parent CanonicalFact.fact_id")
    atom_text: str = Field(description="Self-contained textual statement of this specific proposition")

    # Subject
    subject_entity_id: Optional[str] = Field(default=None, description="Resolved entity ID if known")
    subject_name: str = Field(description="Entity name as extracted or resolved")

    # Predicate
    raw_predicate: str = Field(description="Original verb or relation phrase")
    normalized_predicate: Optional[NormalizedPredicate] = Field(
        default=None, description="Controlled predicate if unambiguously mapped; None if custom/unknown"
    )
    is_normalized: bool = Field(default=False)

    # Object / Complement
    object_value: MolecularValue = Field(description="Structured object or complement value")
    object_entity_id: Optional[str] = Field(default=None, description="Referenced entity if object is an entity")

    # Modality & Polarity
    modality: Modality = Field(default=Modality.DEFINITIVE)
    polarity: Polarity = Field(default=Polarity.POSITIVE)

    # Qualifiers & Temporal Scope
    qualifiers: Dict[str, Any] = Field(
        default_factory=dict, description="Structured qualifiers (e.g. round, series, location, role)"
    )
    temporal_scope: Optional[str] = Field(default=None, description="Textual temporal frame (e.g. 'Q3 2024')")
    event_date: Optional[date] = Field(default=None, description="Consensus event date if known")

    # Epistemic & Derivation Metadata
    is_decomposed: bool = Field(default=False, description="True if split from a compound statement")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    verification_status: VerificationStatus = Field(default=VerificationStatus.SUPPORTED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecular_id": self.molecular_id,
            "parent_canonical_id": self.parent_canonical_id,
            "atom_text": self.atom_text,
            "subject_entity_id": self.subject_entity_id,
            "subject_name": self.subject_name,
            "raw_predicate": self.raw_predicate,
            "normalized_predicate": self.normalized_predicate.value if self.normalized_predicate else None,
            "is_normalized": self.is_normalized,
            "object_value": self.object_value.to_dict(),
            "object_entity_id": self.object_entity_id,
            "modality": self.modality.value,
            "polarity": self.polarity.value,
            "qualifiers": self.qualifiers,
            "temporal_scope": self.temporal_scope,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "is_decomposed": self.is_decomposed,
            "confidence": self.confidence.value,
            "verification_status": self.verification_status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MolecularFact:
        # Normalized predicate
        norm_pred_str = data.get("normalized_predicate")
        norm_pred = None
        if norm_pred_str:
            try:
                norm_pred = NormalizedPredicate(norm_pred_str)
            except ValueError:
                norm_pred = None

        # Modality
        mod_str = data.get("modality", Modality.DEFINITIVE.value)
        try:
            modality = Modality(mod_str)
        except ValueError:
            modality = Modality.DEFINITIVE

        # Polarity
        pol_str = data.get("polarity", Polarity.POSITIVE.value)
        try:
            polarity = Polarity(pol_str)
        except ValueError:
            polarity = Polarity.POSITIVE

        # Confidence
        conf_str = data.get("confidence", ConfidenceLevel.MEDIUM.value)
        try:
            confidence = ConfidenceLevel(conf_str)
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        # Verification Status
        ver_str = data.get("verification_status", VerificationStatus.SUPPORTED.value)
        try:
            ver_status = VerificationStatus(ver_str)
        except ValueError:
            ver_status = VerificationStatus.SUPPORTED

        # Event date
        ev_date = None
        if data.get("event_date"):
            try:
                ev_date = date.fromisoformat(str(data["event_date"]))
            except ValueError:
                ev_date = None

        # Object value
        obj_val_data = data.get("object_value")
        if isinstance(obj_val_data, dict):
            object_value = MolecularValue.from_dict(obj_val_data)
        elif isinstance(obj_val_data, str):
            object_value = MolecularValue(value_type=ValueType.STRING, raw_value=obj_val_data)
        else:
            object_value = MolecularValue(value_type=ValueType.UNKNOWN, raw_value="")

        return cls(
            molecular_id=data.get("molecular_id", str(uuid.uuid4())),
            parent_canonical_id=data.get("parent_canonical_id", ""),
            atom_text=data.get("atom_text", ""),
            subject_entity_id=data.get("subject_entity_id"),
            subject_name=data.get("subject_name", ""),
            raw_predicate=data.get("raw_predicate", ""),
            normalized_predicate=norm_pred,
            is_normalized=bool(data.get("is_normalized", norm_pred is not None)),
            object_value=object_value,
            object_entity_id=data.get("object_entity_id"),
            modality=modality,
            polarity=polarity,
            qualifiers=dict(data.get("qualifiers") or {}),
            temporal_scope=data.get("temporal_scope"),
            event_date=ev_date,
            is_decomposed=bool(data.get("is_decomposed", False)),
            confidence=confidence,
            verification_status=ver_status,
        )


def serialize_molecular_facts(facts: Sequence[MolecularFact]) -> str:
    """Deterministic, versioned JSON serialization for molecular facts."""
    envelope = {
        "schema_version": MOLECULAR_SCHEMA_VERSION,
        "facts": [f.to_dict() for f in facts],
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True)


def deserialize_molecular_facts(payload_json: Optional[str]) -> Tuple[MolecularFact, ...]:
    """Safe, defensive deserialization handling legacy NULLs, empty payloads, and malformed JSON."""
    if not payload_json or not payload_json.strip():
        return ()

    try:
        data = json.loads(payload_json)
    except Exception as exc:
        logger.warning("Failed to decode molecular_data_json: %s", exc)
        return ()

    if not isinstance(data, dict):
        logger.warning("Invalid molecular_data_json envelope type: expected dict, got %s", type(data))
        return ()

    raw_facts = data.get("facts")
    if not isinstance(raw_facts, list):
        return ()

    results: List[MolecularFact] = []
    for item in raw_facts:
        if isinstance(item, dict):
            try:
                results.append(MolecularFact.from_dict(item))
            except Exception as exc:
                logger.warning("Failed to deserialize individual molecular fact item: %s", exc)

    return tuple(results)
