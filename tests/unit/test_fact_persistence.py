import datetime
import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.context.models import ProspectContext
from src.db.connection import get_connection
from src.db.schema import init_schema
from src.db.state_store import StateStore
from src.generator.engine.engine import GeneratorEngine
from src.generator.engine.request import GenerationRequest
from src.generator.engine.response import GenerationResponse
from src.generator.provider.provider_response import ProviderResponse
from src.researcher.extraction.extractor import (
    FactExtractionOutput,
    FactExtractor,
    RawExtractedEvidence,
    RawExtractedFact,
)
from src.researcher.extraction.service import FactExtractionService
from src.researcher.models import (
    ConfidenceLevel,
    EntityRecord,
    EntityType,
    FactCategory,
    ResolutionStatus,
)


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


def _setup_session_and_snapshot(
    state_store: StateStore,
    raw_text: str,
    domain: str = "vanguardai.io",
    title: str = "Vanguard AI Series B Press Release",
) -> Tuple[str, str, str, str]:
    session_id = str(uuid.uuid4())
    prospect_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    source_id = str(uuid.uuid4())
    snapshot_id = str(uuid.uuid4())
    entity_id = str(uuid.uuid4())

    with state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name="Elena Rostova",
            company_name="Vanguard AI",
            company_domain=domain,
            designation="Head of Research",
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id=prospect_id,
            context_id=context_id,
            status="in_progress",
        )
        uow.entities.create_entity(
            entity_id=entity_id,
            entity_type="organization",
            raw_name="Vanguard AI",
            canonical_name="Vanguard AI Technologies Inc.",
            resolution_status="resolved",
            confidence="high",
        )
        uow.sources.upsert_source(
            source_id=source_id,
            canonical_url=f"https://{domain}/press/series-b",
            domain=domain,
            credibility_score=0.9,
        )
        uow.sources.create_snapshot(
            snapshot_id=snapshot_id,
            source_id=source_id,
            session_id=session_id,
            raw_text=raw_text,
            title=title,
            fetch_status="success",
            http_status_code=200,
            duration_ms=150,
        )

    return session_id, source_id, snapshot_id, entity_id


class MockLLMEngine(GeneratorEngine):
    def __init__(self, output: FactExtractionOutput, on_generate_callback=None):
        self.output = output
        self.on_generate_callback = on_generate_callback

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.on_generate_callback:
            self.on_generate_callback()
        raw_json = json.dumps(self.output.model_dump())
        return GenerationResponse(
            text=raw_json,
            provider_name="mock_groq",
            raw_response=ProviderResponse(text=raw_json),
            parsed_output=self.output,
        )


def test_extract_and_persist_facts_happy_path(test_state_store: StateStore):
    raw_text = (
        "SAN FRANCISCO -- Vanguard AI raised $50 million in Series B funding led by Sequoia Capital. "
        "The company plans to expand its distributed machine learning research team."
    )
    session_id, source_id, snapshot_id, entity_id = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI raised $50 million in Series B funding.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                event_date="2026-05-01",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI raised $50 million in Series B funding led by Sequoia Capital.",
                        event_date="2026-05-01",
                    )
                ],
            )
        ]
    )

    known_entities = [
        EntityRecord(
            entity_id=entity_id,
            raw_name="Vanguard AI",
            canonical_name="Vanguard AI Technologies Inc.",
            entity_type=EntityType.ORGANIZATION,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        )
    ]

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    persisted = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
        known_entities=known_entities,
    )

    assert len(persisted) == 1
    fact = persisted[0]
    assert fact.claim_text == "Vanguard AI raised $50 million in Series B funding."
    assert fact.fact_category == FactCategory.FUNDING
    assert fact.entity_ids == [entity_id]

    # Verify DB persistence directly
    with test_state_store.uow() as uow:
        # 1. Canonical facts
        fact_rows = uow.assertions.get_facts_for_session(session_id)
        assert len(fact_rows) == 1
        assert fact_rows[0]["fact_id"] == fact.fact_id
        assert fact_rows[0]["claim_text"] == fact.claim_text
        assert fact_rows[0]["fact_category"] == "funding"

        # 2. Fact-Entity link
        linked_entities = uow.assertions.get_entities_for_fact(fact.fact_id)
        assert len(linked_entities) == 1
        assert linked_entities[0]["entity_id"] == entity_id

        # 3. Evidence passage and Fact-Evidence link
        evidence_rows = uow.assertions.get_evidence_for_fact(fact.fact_id)
        assert len(evidence_rows) == 1
        assert evidence_rows[0]["snapshot_id"] == snapshot_id
        assert evidence_rows[0]["excerpt_validated"] == 1
        assert evidence_rows[0]["support_type"] == "supporting"
        assert "50 million" in evidence_rows[0]["verbatim_excerpt"]

        # 4. Assessment history entry
        cursor = uow.conn.execute(
            "SELECT * FROM assessment_history WHERE session_id = ?;", (session_id,)
        )
        hist_rows = cursor.fetchall()
        assert len(hist_rows) == 1
        assert hist_rows[0]["finding_type"] == "facts_extracted"


def test_conflict_detection_and_persistence(test_state_store: StateStore):
    """When contradictory facts regarding the same round occur, both are preserved and linked in conflicts table."""
    raw_text = (
        "Report A states that Vanguard AI closed a $50 million Series B round. "
        "Earlier rumors suggested Vanguard AI raised a $30 million Series B round."
    )
    session_id, source_id, snapshot_id, entity_id = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI closed a $50 million Series B round.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI closed a $50 million Series B round.",
                    )
                ],
            ),
            RawExtractedFact(
                claim_text="Vanguard AI raised a $30 million Series B round.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI raised a $30 million Series B round.",
                    )
                ],
            ),
        ]
    )

    known_entities = [
        EntityRecord(
            entity_id=entity_id,
            raw_name="Vanguard AI",
            canonical_name="Vanguard AI Technologies Inc.",
            entity_type=EntityType.ORGANIZATION,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        )
    ]

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    persisted = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
        known_entities=known_entities,
    )

    # Both facts must be preserved separately
    assert len(persisted) == 2

    with test_state_store.uow() as uow:
        # Both facts exist in DB
        fact_rows = uow.assertions.get_facts_for_session(session_id)
        assert len(fact_rows) == 2

        # Conflict row is created
        conflicts = uow.assertions.get_conflicts_for_session(session_id)
        assert len(conflicts) == 1
        assert conflicts[0]["status"] == "open"
        assert "conflicting metrics for series b" in conflicts[0]["description"].lower()

        # Both facts are linked to the conflict
        conflict_facts = uow.assertions.get_conflict_facts(conflicts[0]["conflict_id"])
        assert len(conflict_facts) == 2
        conflict_fact_ids = {f["fact_id"] for f in conflict_facts}
        assert conflict_fact_ids == {persisted[0].fact_id, persisted[1].fact_id}


def test_hallucinated_excerpt_discarded_before_persistence(test_state_store: StateStore):
    """If an LLM produces a fact with an excerpt not in raw_text, it is discarded before hitting the DB."""
    raw_text = "Vanguard AI released an open-source benchmarking suite for AI models."
    session_id, source_id, snapshot_id, _ = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI acquired DeepMind Robotics for $2 billion.",
                fact_category=FactCategory.STRATEGY,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI acquired DeepMind Robotics for $2 billion.",
                    )
                ],
            )
        ]
    )

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    persisted = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
    )

    assert len(persisted) == 0

    with test_state_store.uow() as uow:
        facts = uow.assertions.get_facts_for_session(session_id)
        assert len(facts) == 0
        cursor = uow.conn.execute("SELECT COUNT(*) FROM evidence_passages;")
        assert cursor.fetchone()[0] == 0


def test_llm_call_strictly_outside_db_transaction(test_state_store: StateStore):
    """External LLM generation must occur while holding ZERO database transactions."""
    raw_text = "Vanguard AI announced new research breakthroughs in neural compiler optimizations."
    session_id, source_id, snapshot_id, _ = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI announced new research breakthroughs.",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI announced new research breakthroughs in neural compiler optimizations.",
                    )
                ],
            )
        ]
    )

    callback_called = False

    def verify_no_transaction():
        nonlocal callback_called
        callback_called = True
        # Try to write to the database on a separate connection to verify no write lock is held
        with test_state_store.uow() as uow:
            cursor = uow.conn.execute("SELECT COUNT(*) FROM canonical_facts;")
            assert cursor.fetchone()[0] == 0

    engine = MockLLMEngine(llm_output, on_generate_callback=verify_no_transaction)
    extractor = FactExtractor(engine=engine)
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    persisted = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
    )

    assert callback_called
    assert len(persisted) == 1


def test_persistence_failure_rolls_back_atomically(test_state_store: StateStore, monkeypatch):
    """If an exception occurs during Step 5, the entire transaction is rolled back."""
    raw_text = "Vanguard AI published benchmark performance data."
    session_id, source_id, snapshot_id, _ = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI published benchmark performance data.",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI published benchmark performance data.",
                    )
                ],
            )
        ]
    )

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    # Monkeypatch create_evidence_passage to simulate a crash during the transaction
    from src.db.repositories.source_repo import SourceRepository

    def crashing_create_evidence_passage(*args, **kwargs):
        raise RuntimeError("Simulated disk error during passage persistence")

    monkeypatch.setattr(SourceRepository, "create_evidence_passage", crashing_create_evidence_passage)

    with pytest.raises(RuntimeError, match="Simulated disk error"):
        service.extract_and_persist_facts_for_session(
            session_id=session_id,
            snapshot_id=snapshot_id,
        )

    # Verify atomic rollback: neither the canonical fact nor the passage exists
    with test_state_store.uow() as uow:
        facts = uow.assertions.get_facts_for_session(session_id)
        assert len(facts) == 0
        cursor = uow.conn.execute("SELECT COUNT(*) FROM evidence_passages;")
        assert cursor.fetchone()[0] == 0


def test_repeated_processing_is_idempotent_and_does_not_duplicate_or_crash(test_state_store: StateStore):
    """Running extraction repeatedly on the same source/snapshot does not crash, duplicate rows, or overwrite history."""
    raw_text = "Vanguard AI released its flagship open weights foundation model."
    session_id, source_id, snapshot_id, _ = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI released its flagship open weights foundation model.",
                fact_category=FactCategory.PRODUCT_LAUNCH,
                subject_entity_name="Vanguard AI",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI released its flagship open weights foundation model.",
                    )
                ],
            )
        ]
    )

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    # First run
    persisted_1 = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
    )
    assert len(persisted_1) == 1

    # Second run (repeated processing)
    persisted_2 = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
    )
    assert len(persisted_2) == 1

    with test_state_store.uow() as uow:
        # Canonical facts must NOT be duplicated
        fact_rows = uow.assertions.get_facts_for_session(session_id)
        assert len(fact_rows) == 1

        # Evidence passages must NOT be duplicated
        cursor = uow.conn.execute("SELECT COUNT(*) FROM evidence_passages;")
        assert cursor.fetchone()[0] == 1

        # Assessment history must retain both iterations (not overwritten)
        hist = uow.assertions.get_assessment_history(session_id)
        assert len(hist) == 2
        assert [h["iteration"] for h in hist] == [1, 2]


def test_target_prospect_attribution_persistence_and_reload(test_state_store: StateStore):
    """Candidate-level assertions are preserved, distinguishable from prospect facts, and attribution state survives reload."""
    raw_text = (
        "Elena Rostova joined Vanguard AI as VP of Engineering. "
        "Separately, another person named Elena Rostova (Ambiguous Candidate) founded BioCorp in 2019."
    )
    session_id, source_id, snapshot_id, org_id = _setup_session_and_snapshot(
        test_state_store, raw_text=raw_text
    )

    prospect_entity_id = str(uuid.uuid4())
    candidate_entity_id = str(uuid.uuid4())

    with test_state_store.uow() as uow:
        # Create resolved target prospect entity
        uow.entities.create_entity(
            entity_id=prospect_entity_id,
            entity_type="person",
            raw_name="Elena Rostova",
            canonical_name="Elena Rostova",
            resolution_status="resolved",
            confidence="high",
        )
        # Create ambiguous candidate entity discovered during research
        uow.entities.create_entity(
            entity_id=candidate_entity_id,
            entity_type="person",
            raw_name="Elena Rostova (Ambiguous Candidate)",
            canonical_name="Elena Rostova (BioCorp)",
            resolution_status="ambiguous",
            confidence="low",
        )

    known_entities = [
        EntityRecord(
            entity_id=prospect_entity_id,
            entity_type=EntityType.PERSON,
            raw_name="Elena Rostova",
            canonical_name="Elena Rostova",
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        ),
        EntityRecord(
            entity_id=candidate_entity_id,
            entity_type=EntityType.PERSON,
            raw_name="Elena Rostova (Ambiguous Candidate)",
            canonical_name="Elena Rostova (BioCorp)",
            resolution_status=ResolutionStatus.AMBIGUOUS,
            confidence=ConfidenceLevel.LOW,
        ),
    ]

    llm_output = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Elena Rostova joined Vanguard AI as VP of Engineering.",
                fact_category=FactCategory.HIRING,
                subject_entity_name="Elena Rostova",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Elena Rostova joined Vanguard AI as VP of Engineering.",
                    )
                ],
            ),
            RawExtractedFact(
                claim_text="Elena Rostova (Ambiguous Candidate) founded BioCorp in 2019.",
                fact_category=FactCategory.ORGANIZATION_CHANGE,
                subject_entity_name="Elena Rostova (Ambiguous Candidate)",
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Elena Rostova (Ambiguous Candidate) founded BioCorp in 2019.",
                    )
                ],
            ),
        ]
    )

    extractor = FactExtractor(engine=MockLLMEngine(llm_output))
    service = FactExtractionService(state_store=test_state_store, extractor=extractor)

    persisted = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snapshot_id,
        known_entities=known_entities,
    )

    assert len(persisted) == 2

    # Verification before reload
    with test_state_store.uow() as uow:
        # Both facts are preserved in canonical_facts
        all_facts = uow.assertions.get_facts_for_session(session_id)
        assert len(all_facts) == 2

        # Prospect facts query retrieves ONLY the resolved target prospect's fact
        prospect_facts = uow.assertions.get_prospect_facts(session_id, prospect_entity_id)
        assert len(prospect_facts) == 1
        assert "joined Vanguard AI" in prospect_facts[0]["claim_text"]

        # Candidate facts query retrieves ONLY the ambiguous candidate's assertion
        candidate_facts = uow.assertions.get_candidate_facts(session_id)
        assert len(candidate_facts) == 1
        assert "founded BioCorp" in candidate_facts[0]["claim_text"]

    # Database reload simulation (fresh StateStore connection to same SQLite file)
    reloaded_store = StateStore(db_path=test_state_store.db_path)
    with reloaded_store.uow() as uow:
        # Confirm prospect attribution survives reload
        prospect_facts_reloaded = uow.assertions.get_prospect_facts(session_id, prospect_entity_id)
        assert len(prospect_facts_reloaded) == 1
        assert prospect_facts_reloaded[0]["fact_id"] == prospect_facts[0]["fact_id"]

        # Confirm candidate assertion survives reload and is NOT silently attributed to target prospect
        candidate_facts_reloaded = uow.assertions.get_candidate_facts(session_id)
        assert len(candidate_facts_reloaded) == 1
        assert candidate_facts_reloaded[0]["fact_id"] == candidate_facts[0]["fact_id"]

        # Check entity links: prospect fact links to prospect_entity_id, candidate fact links to candidate_entity_id
        prospect_links = uow.assertions.get_entities_for_fact(prospect_facts[0]["fact_id"])
        assert [e["entity_id"] for e in prospect_links] == [prospect_entity_id]

        candidate_links = uow.assertions.get_entities_for_fact(candidate_facts[0]["fact_id"])
        assert [e["entity_id"] for e in candidate_links] == [candidate_entity_id]



