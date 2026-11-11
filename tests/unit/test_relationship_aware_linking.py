from __future__ import annotations

import json
import tempfile
import uuid
from typing import Tuple
import pytest

from src.context.models import (
    CapabilityNode,
    CompatibilityStatus,
    ProductContract,
    ProductContext,
    ProspectContext,
)
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.researcher.entity.candidates import (
    DiscoveredEntityCandidate,
    normalize_org_name,
    normalize_person_name,
)
from src.researcher.entity.probabilistic_models import (
    RelationshipPredicate,
    ResolutionOutcome,
    TypedEntityRelationship,
)
from src.researcher.entity.probabilistic_resolver import ProbabilisticEntityResolver
from src.researcher.extraction.service import FactExtractionService
from src.researcher.models import (
    CanonicalFact,
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    EvidenceRecord,
    FactCategory,
    ResearchAccumulator,
    ResearchPackage,
    ResolutionStatus,
)
from src.signal_strategy.eligibility import SignalEligibilityEvaluator
from src.signal_strategy.models import (
    CandidateSignal,
    EligibilityStatus,
    GateIdentifier,
)


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


def _setup_session(state_store: StateStore) -> Tuple[str, str]:
    session_id = str(uuid.uuid4())
    prospect_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())

    with state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name="Target Prospect",
            company_name="Target Corp",
            company_domain="targetcorp.com",
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id=prospect_id,
            context_id=context_id,
            status="in_progress",
        )
    return session_id, prospect_id


def _build_test_contract() -> ProductContract:
    return ProductContract(
        product_name="CloudAudit Pro",
        contract_version="2.1.0",
        description="Automated continuous cloud compliance auditing and remediation",
        value_proposition="Reduces SOC2 audit prep time from months to continuous real-time monitoring.",
        capabilities=[
            CapabilityNode(
                name="Continuous SOC2 Monitoring",
                description="Real-time scanning of AWS/GCP infrastructure for SOC2 controls",
                supported_use_cases=["SOC2 Type II prep", "Cloud security audit"],
                required_prerequisites=["aws_or_gcp_cloud_environment"],
                target_operational_problems=["audit evidence manual collection"],
                target_buyer_personas=["Head of Security", "CISO", "VP Engineering"],
            ),
        ],
        is_exhaustive=False,
    )


def test_typed_temporal_relationships_with_validity(test_state_store: StateStore):
    with test_state_store.uow() as uow:
        # Upsert sources
        uow.sources.upsert_source(
            source_id="src-news-1",
            canonical_url="https://news.com/satya",
            domain="news.com",
        )
        uow.sources.upsert_source(
            source_id="src-acq-1",
            canonical_url="https://news.com/gh",
            domain="news.com",
        )

        # Create entities
        uow.entities.create_entity(
            entity_id="person-satya",
            entity_type="person",
            raw_name="Satya Nadella",
            canonical_name="Satya Nadella",
            resolution_status="resolved",
        )
        uow.entities.create_entity(
            entity_id="org-msft",
            entity_type="organization",
            raw_name="Microsoft Corporation",
            canonical_name="Microsoft",
            resolution_status="resolved",
        )

        # 1. Leadership role with temporal validity bounds
        uow.entities.add_relationship(
            relationship_id="rel-ceo-1",
            from_entity_id="person-satya",
            to_entity_id="org-msft",
            relationship_type=RelationshipPredicate.LEADERSHIP_ROLE.value,
            valid_from="2014-02-04",
            valid_until=None,
            source_id="src-news-1",
        )

        # 2. Subsidiary relationship with temporal bounds
        uow.entities.create_entity(
            entity_id="org-gh",
            entity_type="organization",
            raw_name="GitHub Inc",
            canonical_name="GitHub",
            resolution_status="resolved",
        )
        uow.entities.add_relationship(
            relationship_id="rel-sub-1",
            from_entity_id="org-gh",
            to_entity_id="org-msft",
            relationship_type=RelationshipPredicate.SUBSIDIARY_OF.value,
            valid_from="2018-10-26",
            valid_until=None,
            source_id="src-acq-1",
        )

        rels_satya = uow.entities.get_relationships("person-satya")
        assert len(rels_satya) == 1
        assert rels_satya[0]["relationship_type"] == "leadership_role"
        assert rels_satya[0]["valid_from"] == "2014-02-04"
        assert rels_satya[0]["valid_until"] is None

        rels_gh = uow.entities.get_relationships("org-gh")
        assert len(rels_gh) == 1
        assert rels_gh[0]["relationship_type"] == "subsidiary_of"
        assert rels_gh[0]["to_entity_id"] == "org-msft"


def test_distinct_entities_sharing_parent_org_remain_distinct():
    """Distinct entities (e.g. sister business units or distinct people) sharing a parent or address are not collapsed."""
    resolver = ProbabilisticEntityResolver()

    # Target is Azure division
    target_division = EntityRecord(
        entity_id="org-azure",
        raw_name="Microsoft Azure",
        canonical_name="Azure Cloud",
        entity_type=EntityType.ORGANIZATION,
    )

    # Candidate is Xbox division sharing parent 'Microsoft'
    candidate_xbox = DiscoveredEntityCandidate(
        candidate_id="cand-xbox",
        raw_name="Xbox Entertainment",
        normalized_name=normalize_org_name("Xbox Entertainment"),
        entity_type=EntityType.ORGANIZATION,
        source_ids=["src-xbox"],
        organizations=["Microsoft Corporation"],
        domains=["xbox.com"],
    )

    res = resolver.resolve_candidates(
        candidates=[candidate_xbox],
        target_entity=target_division,
        entity_type=EntityType.ORGANIZATION,
    )

    # Must NOT merge Xbox into Azure!
    assert res.outcome != ResolutionOutcome.MATCH
    assert res.selected_candidate_id is None


def test_fact_attribution_uncertainty_preservation(test_state_store: StateStore):
    """When entity resolution is AMBIGUOUS, facts preserve candidate IDs without premature binding.
    When RESOLVED, facts are linked. When UNRESOLVED, attribution remains unresolved."""
    session_id, _ = _setup_session(test_state_store)

    # Insert sources
    with test_state_store.uow() as uow:
        uow.sources.upsert_source(source_id="s-src-1", canonical_url="https://sec.gov/10k", domain="sec.gov")
        uow.sources.create_snapshot(
            snapshot_id="snap-1",
            source_id="s-src-1",
            fetch_status="success",
            raw_text="Acme disclosed Series B funding.",
        )

    # 1. Fact attributed to an AMBIGUOUS entity
    ambig_entity = EntityRecord(
        entity_id="ent-ambig-1",
        raw_name="Acme Tech",
        entity_type=EntityType.ORGANIZATION,
        resolution_status=ResolutionStatus.AMBIGUOUS,
        disambiguation_notes="Conflicting: Acme US vs Acme UK",
    )
    with test_state_store.uow() as uow:
        uow.entities.create_entity(
            entity_id="ent-ambig-1",
            entity_type="organization",
            raw_name="Acme Tech",
            resolution_status="ambiguous",
        )

    class MockExtractor:
        def __init__(self, fact_to_return):
            self.fact = fact_to_return

        def extract(self, **kwargs):
            return [self.fact]

    fact_ambig = CanonicalFact(
        fact_id="fact-ambig-1",
        claim_text="Acme raised $20M in Series B funding",
        fact_category=FactCategory.FUNDING,
        entity_ids=["ent-ambig-1"],
        evidence_records=[EvidenceRecord(source_id="s-src-1", verbatim_excerpt="Acme raised $20M in Series B")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    svc = FactExtractionService(state_store=test_state_store, extractor=MockExtractor(fact_ambig))
    persisted = svc.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id="snap-1",
        known_entities=[ambig_entity],
    )

    assert len(persisted) == 1
    assert persisted[0].entity_resolution_status == "ambiguous"
    assert "ent-ambig-1" in persisted[0].ambiguous_candidate_ids

    # Verify database state: Fact exists, but fact_entity_links MUST NOT link to ambiguous entity!
    with test_state_store.uow() as uow:
        db_fact = uow.assertions.get_canonical_fact("fact-ambig-1")
        assert db_fact is not None
        assert db_fact["entity_resolution_status"] == "ambiguous"
        assert json.loads(db_fact["ambiguous_candidates_json"]) == ["ent-ambig-1"]

        # Fact is linked to candidate entity in fact_entity_links for candidate auditability
        linked_entities = uow.assertions.get_entities_for_fact("fact-ambig-1")
        assert len(linked_entities) == 1
        assert linked_entities[0]["entity_id"] == "ent-ambig-1"

        # But prospect facts query retrieves 0 facts because ent-ambig-1 is not resolved
        prospect_facts = uow.assertions.get_prospect_facts(session_id, "ent-ambig-1")
        assert len(prospect_facts) == 0

        # And candidate facts query retrieves the ambiguous candidate fact
        candidate_facts = uow.assertions.get_candidate_facts(session_id)
        assert len(candidate_facts) == 1
        assert candidate_facts[0]["fact_id"] == "fact-ambig-1"

    # 2. Fact attributed to a RESOLVED entity
    resolved_entity = EntityRecord(
        entity_id="ent-resolved-1",
        raw_name="Apex Corp",
        entity_type=EntityType.ORGANIZATION,
        resolution_status=ResolutionStatus.RESOLVED,
    )
    with test_state_store.uow() as uow:
        uow.entities.create_entity(
            entity_id="ent-resolved-1",
            entity_type="organization",
            raw_name="Apex Corp",
            resolution_status="resolved",
        )

    fact_resolved = CanonicalFact(
        fact_id="fact-res-1",
        claim_text="Apex Corp migrated cloud workloads to AWS",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent-resolved-1"],
        evidence_records=[EvidenceRecord(source_id="s-src-1", verbatim_excerpt="Apex Corp migrated cloud workloads to AWS")],
        composite_confidence=ConfidenceLevel.HIGH,
    )

    svc_res = FactExtractionService(state_store=test_state_store, extractor=MockExtractor(fact_resolved))
    persisted_res = svc_res.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id="snap-1",
        known_entities=[resolved_entity],
    )

    assert len(persisted_res) == 2
    # Reloaded fact preserved ambiguous status
    assert persisted_res[0].entity_resolution_status == "ambiguous"
    assert "ent-ambig-1" in persisted_res[0].ambiguous_candidate_ids
    # New fact has resolved status
    assert persisted_res[1].entity_resolution_status == "match"

    with test_state_store.uow() as uow:
        db_fact_res = uow.assertions.get_canonical_fact("fact-res-1")
        assert db_fact_res["entity_resolution_status"] == "match"

        linked_res = uow.assertions.get_entities_for_fact("fact-res-1")
        assert len(linked_res) == 1
        assert linked_res[0]["entity_id"] == "ent-resolved-1"


def test_signal_eligibility_identity_gate_strictly_fails_on_ambiguity():
    """Signals backed only by facts with ambiguous entity resolution strictly fail the identity gate."""
    evaluator = SignalEligibilityEvaluator()

    prospect = ProspectContext(
        prospect_id="p-elig",
        first_name="Diana",
        last_name="Prince",
        full_name="Diana Prince",
        company_name="Wayne Enterprises",
        company_domain="wayne.com",
    )

    product = ProductContext(
        product_name="CloudAudit Pro",
        description="Cloud compliance auditing",
        capabilities=["Continuous SOC2 Monitoring"],
        value_proposition="Continuous monitoring",
        contract=_build_test_contract(),
    )

    # Fact with ambiguous entity attribution
    fact_ambig = CanonicalFact(
        fact_id="f-ambig-elig",
        claim_text="Wayne Enterprises deployed AWS cloud services",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent-ambig-9"],
        evidence_records=[EvidenceRecord(source_id="s-1", verbatim_excerpt="Wayne Enterprises deployed AWS")],
        entity_resolution_status="ambiguous",
        ambiguous_candidate_ids=["cand-1", "cand-2"],
    )

    signal = CandidateSignal(
        signal_id="sig-ambig",
        label="Cloud Security Opportunity",
        description="Wayne Enterprises AWS migration",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-ambig-elig"],
    )

    # 1. When supporting facts have ambiguous attribution and neither entity is resolved:
    eligibility = evaluator.evaluate(
        signal=signal,
        facts=[fact_ambig],
        prospect=prospect,
        product=product,
    )

    assert eligibility.is_eligible is False
    assert eligibility.status == EligibilityStatus.INELIGIBLE
    assert eligibility.identity_gate.passed is False
    assert "ambiguous or unresolved" in eligibility.identity_gate.reason

    # 2. When prospect entity is explicitly ambiguous:
    ambig_prospect_entity = EntityRecord(
        entity_id="ent-p-ambig",
        raw_name="Diana Prince",
        entity_type=EntityType.PERSON,
        resolution_status=ResolutionStatus.AMBIGUOUS,
        disambiguation_notes="Multiple people with name Diana Prince found",
    )

    fact_normal = CanonicalFact(
        fact_id="f-norm",
        claim_text="Wayne Enterprises operating aws_or_gcp_cloud_environment infrastructure",
        evidence_records=[EvidenceRecord(source_id="s-1", verbatim_excerpt="infra")],
    )
    sig_normal = CandidateSignal(
        signal_id="sig-norm",
        label="AWS Infrastructure",
        description="Wayne Enterprises AWS infrastructure",
        target_capability="Continuous SOC2 Monitoring",
        supporting_fact_ids=["f-norm"],
    )

    eligibility_p = evaluator.evaluate(
        signal=sig_normal,
        facts=[fact_normal],
        prospect=prospect,
        product=product,
        prospect_entity=ambig_prospect_entity,
    )

    assert eligibility_p.is_eligible is False
    assert eligibility_p.status == EligibilityStatus.INELIGIBLE
    assert eligibility_p.identity_gate.passed is False
    assert "ambiguous or unresolved" in eligibility_p.identity_gate.reason
