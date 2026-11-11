import pytest
from pydantic import ValidationError

from src.researcher.evidence_request.models import (
    EvidenceGapReason,
    EvidenceRequest,
    EvidenceRequestStatus,
)
from src.researcher.evidence_request.validator import (
    BiasedEvidenceRequestError,
    check_commercial_bias,
    validate_hypothesis_blindness,
)


def test_valid_neutral_request_serialization_roundtrip():
    req = EvidenceRequest(
        run_id="run_test_123",
        target_entity_id="ent_company_456",
        target_entity_name="Acme Corp",
        requested_predicate="infrastructure_requirement::kubernetes",
        neutral_question="Does Acme Corp currently deploy or manage production Kubernetes clusters?",
        neutral_search_terms=[
            "Acme Corp Kubernetes",
            "Acme Corp infrastructure EKS GKE",
        ],
        disconfirming_search_terms=[
            "Acme Corp on-premise datacenter VMware",
            "Acme Corp serverless only",
            "Acme Corp no Kubernetes",
        ],
        valid_from="2025-01-01",
        valid_to="2026-11-01",
        gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
        provenance_stage="signal_eligibility_product_gate",
        motivated_by_signal_id="sig_infra_exp",
    )

    data = req.model_dump()
    assert data["target_entity_id"] == "ent_company_456"
    assert data["gap_reason"] == "missing_prerequisite"
    assert data["status"] == "pending"
    assert len(data["disconfirming_search_terms"]) == 3

    json_str = req.model_dump_json()
    deserialized = EvidenceRequest.model_validate_json(json_str)
    assert deserialized.request_id == req.request_id
    assert deserialized.neutral_question == req.neutral_question
    assert deserialized.disconfirming_search_terms == req.disconfirming_search_terms


@pytest.mark.parametrize(
    "prohibited_term",
    [
        "our product",
        "why they should buy",
        "needs our",
        "pitch",
        "cold email",
        "pain point",
        "pain points",
        "dissatisfied with",
        "our platform",
        "reasons to buy",
        "prove they need",
    ],
)
def test_rejection_of_commercial_persuasion_in_question(prohibited_term: str):
    # Direct validator check raises BiasedEvidenceRequestError
    with pytest.raises(BiasedEvidenceRequestError) as exc_info:
        validate_hypothesis_blindness(
            neutral_question=f"Does Acme have any {prohibited_term} for data management?",
            neutral_search_terms=["Acme data management"],
            disconfirming_search_terms=["Acme no data management"],
        )
    assert prohibited_term.lower() in str(exc_info.value).lower()

    # Model instantiation wraps error in ValidationError
    with pytest.raises(ValidationError) as model_exc:
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="product_fit",
            neutral_question=f"Does Acme have any {prohibited_term} for data management?",
            neutral_search_terms=["Acme data management"],
            disconfirming_search_terms=["Acme no data management"],
            gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
            provenance_stage="strategy",
        )
    assert prohibited_term.lower() in str(model_exc.value).lower()


@pytest.mark.parametrize(
    "prohibited_term",
    [
        "our solution",
        "why they should purchase",
        "cold email outreach",
        "sales hook",
    ],
)
def test_rejection_of_commercial_persuasion_in_neutral_search_terms(prohibited_term: str):
    with pytest.raises(BiasedEvidenceRequestError) as exc_info:
        validate_hypothesis_blindness(
            neutral_question="Did Acme partner with cloud providers?",
            neutral_search_terms=[f"Acme {prohibited_term}"],
            disconfirming_search_terms=["Acme standalone"],
        )
    assert prohibited_term.lower() in str(exc_info.value).lower()

    with pytest.raises(ValidationError):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="partnership",
            neutral_question="Did Acme partner with cloud providers?",
            neutral_search_terms=[f"Acme {prohibited_term}"],
            disconfirming_search_terms=["Acme standalone"],
            gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
            provenance_stage="strategy",
        )


@pytest.mark.parametrize(
    "prohibited_term",
    [
        "pain points",
        "pitch",
        "our tool",
        "dissatisfied with",
    ],
)
def test_rejection_of_commercial_persuasion_in_disconfirming_search_terms(prohibited_term: str):
    with pytest.raises(BiasedEvidenceRequestError) as exc_info:
        validate_hypothesis_blindness(
            neutral_question="What vendor does Acme use for logging?",
            neutral_search_terms=["Acme logging vendor"],
            disconfirming_search_terms=[f"Acme {prohibited_term}"],
        )
    assert prohibited_term.lower() in str(exc_info.value).lower()

    with pytest.raises(ValidationError):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="vendor_satisfaction",
            neutral_question="What vendor does Acme use for logging?",
            neutral_search_terms=["Acme logging vendor"],
            disconfirming_search_terms=[f"Acme {prohibited_term}"],
            gap_reason=EvidenceGapReason.MISSING_PREREQUISITE,
            provenance_stage="strategy",
        )


def test_required_contract_fields_validation():
    # Empty target_entity_id
    with pytest.raises(ValidationError, match="target_entity_id"):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="",
            requested_predicate="works_for",
            neutral_question="Does John work at Acme?",
            neutral_search_terms=["John Acme"],
            disconfirming_search_terms=["John left Acme"],
            gap_reason=EvidenceGapReason.UNRESOLVED_EMPLOYMENT,
            provenance_stage="strategy",
        )

    # Empty requested_predicate
    with pytest.raises(ValidationError, match="requested_predicate"):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="   ",
            neutral_question="Does John work at Acme?",
            neutral_search_terms=["John Acme"],
            disconfirming_search_terms=["John left Acme"],
            gap_reason=EvidenceGapReason.UNRESOLVED_EMPLOYMENT,
            provenance_stage="strategy",
        )

    # Empty neutral question
    with pytest.raises(ValidationError, match="neutral_question"):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="works_for",
            neutral_question="   ",
            neutral_search_terms=["John Acme"],
            disconfirming_search_terms=["John left Acme"],
            gap_reason=EvidenceGapReason.UNRESOLVED_EMPLOYMENT,
            provenance_stage="strategy",
        )

    # Missing disconfirming search terms
    with pytest.raises(ValidationError, match="disconfirming_search_term"):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="works_for",
            neutral_question="Does John work at Acme?",
            neutral_search_terms=["John Acme"],
            disconfirming_search_terms=[],
            gap_reason=EvidenceGapReason.UNRESOLVED_EMPLOYMENT,
            provenance_stage="strategy",
        )


def test_temporal_scope_validation():
    # Inverted valid_from and valid_to
    with pytest.raises(ValidationError, match="cannot be later than valid_to"):
        EvidenceRequest(
            run_id="run_1",
            target_entity_id="ent_1",
            requested_predicate="headquartered_in",
            neutral_question="Where is Acme headquartered?",
            neutral_search_terms=["Acme headquarters location"],
            disconfirming_search_terms=["Acme closed headquarters"],
            valid_from="2026-05-01",
            valid_to="2024-01-01",
            gap_reason=EvidenceGapReason.MATERIAL_CONTRADICTION,
            provenance_stage="strategy",
        )

    # Valid temporal scope
    req = EvidenceRequest(
        run_id="run_1",
        target_entity_id="ent_1",
        requested_predicate="headquartered_in",
        neutral_question="Where is Acme headquartered?",
        neutral_search_terms=["Acme headquarters location"],
        disconfirming_search_terms=["Acme closed headquarters"],
        valid_from="2024-01-01",
        valid_to="2026-05-01",
        gap_reason=EvidenceGapReason.MATERIAL_CONTRADICTION,
        provenance_stage="strategy",
    )
    assert req.valid_from == "2024-01-01"
    assert req.valid_to == "2026-05-01"
