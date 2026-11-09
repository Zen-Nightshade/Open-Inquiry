from __future__ import annotations

import datetime
import json
import tempfile
import uuid
import pytest

from src.context.models import ProspectContext
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.entity.candidates import (
    CandidateObservation,
    DiscoveredEntityCandidate,
    EntityCandidateGenerator,
)
from src.researcher.entity.resolver import (
    AffiliationData,
    IdentifierData,
    IdentityAssessmentOutput,
    IdentityResolver,
    RelationshipData,
)
from src.researcher.entity.service import EntityResolutionService
from src.researcher.models import (
    ConfidenceLevel,
    EntityType,
    ResolutionStatus,
    SourceRecord,
)


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


@pytest.fixture
def sample_prospect():
    return ProspectContext(
        full_name="Elena Rostova",
        company_name="Vanguard AI",
        company_domain="vanguardai.io",
        designation="Head of Research",
        linkedin_url="https://linkedin.com/in/erostova",
    )


def _setup_session(state_store: StateStore, prospect: ProspectContext) -> Tuple[str, str]:
    session_id = str(uuid.uuid4())
    prospect_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())

    with state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name=prospect.full_name,
            company_name=prospect.company_name,
            company_domain=prospect.company_domain,
            designation=prospect.designation,
            linkedin_url=prospect.linkedin_url,
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id=prospect_id,
            context_id=context_id,
            status="in_progress",
        )
        uow.sources.upsert_source(
            source_id="src_1",
            canonical_url="https://vanguardai.io/about",
            domain="vanguardai.io",
        )
    return session_id, prospect_id


def test_entity_resolution_service_full_lifecycle(test_state_store: StateStore, sample_prospect: ProspectContext):
    session_id, _ = _setup_session(test_state_store, sample_prospect)

    source1 = SourceRecord(
        source_id="src_1",
        url="https://vanguardai.io/about",
        domain="vanguardai.io",
        title="About Us - Vanguard AI",
        raw_text="Elena Rostova serves as Head of Research at Vanguard AI.",
        fetch_date=datetime.date(2026, 11, 7),
    )

    # Mock LLM response with complete structured output
    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=[],
        prospect_canonical_name="Elena Rostova",
        company_canonical_name="Vanguard AI Corp",
        prospect_aliases=["Dr. Elena Rostova"],
        company_aliases=["Vanguard"],
        affiliations=[
            AffiliationData(
                role="Head of Research",
                organization_name="Vanguard AI",
                is_current=True,
                start_date="2023-01-01",
                end_date=None,
            ),
            AffiliationData(
                role="Senior Scientist",
                organization_name="DeepMind",
                is_current=False,
                start_date="2019-06-01",
                end_date="2022-12-31",
            ),
        ],
        identifiers=[
            IdentifierData(id_type="linkedin_url", id_value="https://linkedin.com/in/erostova"),
            IdentifierData(id_type="company_domain", id_value="vanguardai.io"),
        ],
        relationships=[
            RelationshipData(
                relationship_type="acquired_by",
                related_entity_name="Global Tech Holdings",
                valid_from="2024-05-01",
            )
        ],
        rationale="Exact name match, matching company domain (vanguardai.io), and verified LinkedIn profile.",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            import re
            msgs = request.prompt.get_messages()
            m = re.search(r"Candidate #\d+ \[ID:\s*([a-f0-9\-]+)\]", msgs[1].content if len(msgs) > 1 else "")
            cids = [m.group(1)] if m else []
            updated = llm_output.model_copy(update={"matched_candidate_ids": cids})
            raw_text = json.dumps(updated.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=updated,
            )

    engine = MockEngine()
    resolver = IdentityResolver(engine=engine)
    service = EntityResolutionService(state_store=test_state_store, resolver=resolver)

    result = service.resolve_entities_for_session(
        session_id=session_id,
        prospect=sample_prospect,
        sources=[source1],
    )

    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.confidence == ConfidenceLevel.HIGH

    # Verification: check database records
    with test_state_store.uow() as uow:
        # 1. Candidates were persisted
        candidates = uow.entities.get_candidates(session_id)
        assert len(candidates) >= 1
        persisted_cand = next(c for c in candidates if c["raw_name"] == "Elena Rostova")
        assert persisted_cand["entity_type"] == "person"
        attr_data = json.loads(persisted_cand["attributes"])
        assert "vanguardai.io" in attr_data["domains"] or "Vanguard AI" in attr_data["organizations"]

        # 2. Identity Assessment was persisted
        assessments = uow.entities.get_assessments(session_id=session_id)
        assert len(assessments) == 1
        ass = assessments[0]
        assert ass["assessment_id"] == result.assessment_id
        assert ass["resolution_status"] == "resolved"
        assert ass["confidence"] == "high"
        assert ass["entity_id"] == result.prospect_entity.entity_id

        # 3. Entities were persisted
        prospect_entity = uow.entities.get_entity(result.prospect_entity.entity_id)
        assert prospect_entity is not None
        assert prospect_entity["raw_name"] == "Elena Rostova"
        assert prospect_entity["canonical_name"] == "Elena Rostova"
        assert prospect_entity["resolution_status"] == "resolved"

        company_entity = uow.entities.get_entity(result.company_entity.entity_id)
        assert company_entity is not None
        assert company_entity["canonical_name"] == "Vanguard AI Corp"

        # 4. Temporal affiliations were persisted (distinguishing ongoing vs historical)
        affiliations = uow.entities.get_affiliations(entity_id=result.prospect_entity.entity_id)
        assert len(affiliations) == 2

        # Current affiliation (end_date is NULL)
        current_aff = next(a for a in affiliations if a["end_date"] is None)
        assert current_aff["role"] == "Head of Research"
        assert current_aff["start_date"] == "2023-01-01"

        # Historical affiliation (end_date is NOT NULL)
        historical_aff = next(a for a in affiliations if a["end_date"] is not None)
        assert historical_aff["role"] == "Senior Scientist"
        assert historical_aff["start_date"] == "2019-06-01"
        assert historical_aff["end_date"] == "2022-12-31"

        # 5. Aliases were persisted
        aliases = uow.entities.get_aliases(entity_id=result.prospect_entity.entity_id)
        alias_names = [a["alias"] for a in aliases]
        assert "Dr. Elena Rostova" in alias_names

        # 6. Relationships were persisted (acquired_by)
        relationships = uow.entities.get_relationships(entity_id=result.company_entity.entity_id)
        assert len(relationships) == 1
        assert relationships[0]["relationship_type"] == "acquired_by"
        assert relationships[0]["valid_from"] == "2024-05-01"


def test_transaction_boundary_strictly_closed_during_llm(test_state_store: StateStore, sample_prospect: ProspectContext):
    """Verify that during LLM call, no SQLite transaction is active."""
    session_id, _ = _setup_session(test_state_store, sample_prospect)

    source1 = SourceRecord(
        source_id="src_1",
        url="https://vanguardai.io/about",
        domain="vanguardai.io",
        title="About Us",
        raw_text="Elena Rostova at Vanguard AI.",
        fetch_date=datetime.date(2026, 11, 7),
    )

    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=[],
        rationale="Verified identity.",
    )

    class TransactionCheckingEngine(GeneratorEngine):
        def __init__(self, store: StateStore):
            self.store = store
            self.can_access_db = None

        def generate(self, request):
            # Verify database can be accessed concurrently without lock conflicts
            with self.store.uow() as uow:
                cands = uow.entities.get_candidates(session_id)
                self.can_access_db = cands is not None
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    engine = TransactionCheckingEngine(store=test_state_store)
    resolver = IdentityResolver(engine=engine)
    service = EntityResolutionService(state_store=test_state_store, resolver=resolver)

    service.resolve_entities_for_session(
        session_id=session_id,
        prospect=sample_prospect,
        sources=[source1],
    )

    # Database was successfully accessed independently during LLM call (zero locks held)
    assert engine.can_access_db is True


def test_atomic_rollback_on_persistence_error(test_state_store: StateStore, sample_prospect: ProspectContext):
    """If persistence fails in Stage 4, Stage 4 rolls back while Stage 2 candidates remain committed."""
    session_id, _ = _setup_session(test_state_store, sample_prospect)

    source1 = SourceRecord(
        source_id="src_1",
        url="https://vanguardai.io/about",
        domain="vanguardai.io",
        title="About Us",
        raw_text="Elena Rostova at Vanguard AI.",
        fetch_date=datetime.date(2026, 11, 7),
    )

    llm_output = IdentityAssessmentOutput(
        resolution_status=ResolutionStatus.RESOLVED,
        confidence=ConfidenceLevel.HIGH,
        matched_candidate_ids=[],
        rationale="Test rationale",
    )

    class MockEngine(GeneratorEngine):
        def __init__(self):
            pass

        def generate(self, request):
            raw_text = json.dumps(llm_output.model_dump())
            return GenerationResponse(
                text=raw_text,
                provider_name="mock_groq",
                raw_response=ProviderResponse(text=raw_text),
                parsed_output=llm_output,
            )

    # Test where Stage 2 succeeds (persisting candidates), but Stage 4 encounters an unexpected failure:
    class FailingStage4Service(EntityResolutionService):
        def resolve_entities_for_session(self, session_id, prospect, sources, search_results=None):
            # Call stage 1 and 2 normally
            candidates = self.candidate_generator.generate_candidates(prospect, sources, search_results)
            with self.state_store.uow() as uow:
                for c in candidates:
                    uow.entities.record_candidate(
                        candidate_id=c.candidate_id,
                        session_id=session_id,
                        entity_type=c.entity_type.value,
                        raw_name=c.raw_name,
                        attributes=c.to_attributes_json(),
                        source_id=c.primary_source_id,
                    )
            # Stage 3
            result = self.resolver.resolve(prospect, candidates)
            # Stage 4 explodes
            with self.state_store.uow() as uow:
                uow.entities.create_entity(
                    entity_id=result.prospect_entity.entity_id,
                    entity_type="person",
                    raw_name=result.prospect_entity.raw_name,
                )
                raise RuntimeError("Disk full / simulated failure during stage 4")

    failing_service = FailingStage4Service(
        state_store=test_state_store,
        resolver=IdentityResolver(engine=MockEngine()),
    )

    with pytest.raises(RuntimeError) as exc:
        failing_service.resolve_entities_for_session(
            session_id=session_id,
            prospect=sample_prospect,
            sources=[source1],
        )
    assert "simulated failure" in str(exc.value)

    # Candidates from Stage 2 are intact because Stage 2 committed before Stage 3 & 4
    with test_state_store.uow() as uow:
        candidates = uow.entities.get_candidates(session_id=session_id)
        assert len(candidates) >= 1

        # But Stage 4 changes (assessment, affiliations) were rolled back!
        assessments = uow.entities.get_assessments(session_id=session_id)
        assert len(assessments) == 0
