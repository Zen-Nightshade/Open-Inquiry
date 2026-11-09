import datetime
import tempfile
import uuid
from typing import List
import pytest

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


class MockLLMEngine(GeneratorEngine):
    def __init__(self, output_sequence: List[FactExtractionOutput]):
        self.output_sequence = list(output_sequence)
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        output = self.output_sequence.pop(0) if self.output_sequence else FactExtractionOutput(facts=[])
        raw_json = output.model_dump_json()
        return GenerationResponse(
            text=raw_json,
            provider_name="mock_groq",
            raw_response=ProviderResponse(text=raw_json),
            parsed_output=output,
        )


@pytest.fixture
def test_state_store():
    with tempfile.NamedTemporaryFile(suffix=".db") as tf:
        conn = get_connection(tf.name)
        init_schema(conn)
        conn.close()
        yield StateStore(db_path=tf.name)


def test_integrated_phase_e_workflow(test_state_store: StateStore):
    """Integrated test exercising the full Phase E pipeline:
    
    1. Snapshots with syndicated content from two news domains.
    2. Extraction with excerpt validation and entity linking.
    3. Origin group analysis: syndicated wire releases do NOT inflate confidence.
    4. Two-layer model preservation: canonical facts and multiple distinct evidence passages.
    5. Persistence with short bounded transactions and assessment history.
    6. Repeated execution testing idempotency (no duplicate facts or passages).
    7. Database reload verifying persistence and prospect attribution queries.
    """
    session_id = str(uuid.uuid4())
    prospect_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    prospect_entity_id = str(uuid.uuid4())
    company_entity_id = str(uuid.uuid4())
    src_tc = str(uuid.uuid4())
    src_biz = str(uuid.uuid4())
    snap_tc = str(uuid.uuid4())
    snap_biz = str(uuid.uuid4())
    origin_group_id = str(uuid.uuid4())

    text_tc = (
        "Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital on May 15, 2026. "
        "Elena Rostova currently serves as Head of Research at Vanguard AI."
    )
    text_biz = (
        "Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital on May 15, 2026. "
        "The funding will accelerate enterprise compiler development."
    )

    # 1. Setup session, entities, sources, origin syndication group, and snapshots
    with test_state_store.uow() as uow:
        uow.outreach.create_prospect(
            prospect_id=prospect_id,
            full_name="Elena Rostova",
            company_name="Vanguard AI",
            company_domain="vanguardai.io",
            designation="Head of Research",
        )
        uow.research.create_session(
            session_id=session_id,
            prospect_id=prospect_id,
            context_id=context_id,
            status="in_progress",
        )
        uow.entities.create_entity(
            entity_id=prospect_entity_id,
            entity_type="person",
            raw_name="Elena Rostova",
            canonical_name="Elena Rostova",
            resolution_status="resolved",
            confidence="high",
        )
        uow.entities.create_entity(
            entity_id=company_entity_id,
            entity_type="organization",
            raw_name="Vanguard AI",
            canonical_name="Vanguard AI Inc.",
            resolution_status="resolved",
            confidence="high",
        )
        # Two distinct domains
        uow.sources.upsert_source(
            source_id=src_tc,
            canonical_url="https://techcrunch.com/2026/05/vanguard-ai-series-b",
            domain="techcrunch.com",
            credibility_score=0.85,
        )
        uow.sources.upsert_source(
            source_id=src_biz,
            canonical_url="https://bizjournals.com/vanguard-ai-round",
            domain="bizjournals.com",
            credibility_score=0.80,
        )
        # Establish wire release syndication origin group
        uow.sources.create_origin_group(
            group_id=origin_group_id,
            canonical_source_id=src_tc,
            description="wire_release",
        )
        uow.sources.add_origin_member(group_id=origin_group_id, source_id=src_tc)
        uow.sources.add_origin_member(group_id=origin_group_id, source_id=src_biz)

        # Create snapshots
        uow.sources.create_snapshot(
            snapshot_id=snap_tc,
            source_id=src_tc,
            session_id=session_id,
            raw_text=text_tc,
            title="TechCrunch: Vanguard AI raises Series B",
            fetch_status="success",
            http_status_code=200,
            duration_ms=120,
        )
        uow.sources.create_snapshot(
            snapshot_id=snap_biz,
            source_id=src_biz,
            session_id=session_id,
            raw_text=text_biz,
            title="BizJournals: Vanguard AI $50M Round",
            fetch_status="success",
            http_status_code=200,
            duration_ms=130,
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
            entity_id=company_entity_id,
            entity_type=EntityType.ORGANIZATION,
            raw_name="Vanguard AI",
            canonical_name="Vanguard AI Inc.",
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=ConfidenceLevel.HIGH,
        ),
    ]

    # Mock LLM outputs for the extraction steps
    output_snapshot_1 = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                primary_event_date="2026-05-15",
                confidence=ConfidenceLevel.MEDIUM,
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital on May 15, 2026.",
                        event_date="2026-05-15",
                        confidence=ConfidenceLevel.MEDIUM,
                    )
                ],
            ),
            RawExtractedFact(
                claim_text="Elena Rostova currently serves as Head of Research at Vanguard AI.",
                fact_category=FactCategory.HIRING,
                subject_entity_name="Elena Rostova",
                confidence=ConfidenceLevel.HIGH,
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Elena Rostova currently serves as Head of Research at Vanguard AI.",
                        confidence=ConfidenceLevel.HIGH,
                    )
                ],
            ),
        ]
    )

    output_snapshot_2 = FactExtractionOutput(
        facts=[
            RawExtractedFact(
                claim_text="Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital.",
                fact_category=FactCategory.FUNDING,
                subject_entity_name="Vanguard AI",
                primary_event_date="2026-05-15",
                confidence=ConfidenceLevel.MEDIUM,
                evidence=[
                    RawExtractedEvidence(
                        verbatim_excerpt="Vanguard AI closed a $50 million Series B funding round led by Sequoia Capital on May 15, 2026.",
                        event_date="2026-05-15",
                        confidence=ConfidenceLevel.MEDIUM,
                    )
                ],
            )
        ]
    )

    # Sequence: snap_tc, snap_biz, and then repeated snap_tc for idempotency
    engine = MockLLMEngine([output_snapshot_1, output_snapshot_2, output_snapshot_1])
    service = FactExtractionService(state_store=test_state_store, extractor=FactExtractor(engine=engine))

    # --- Step 1: Extract & persist from Snapshot 1 (TechCrunch) ---
    facts_1 = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snap_tc,
        known_entities=known_entities,
    )
    assert len(facts_1) == 2
    funding_fact_1 = next(f for f in facts_1 if f.fact_category == FactCategory.FUNDING)
    assert funding_fact_1.composite_confidence == ConfidenceLevel.MEDIUM
    assert len(funding_fact_1.evidence_records) == 1

    # --- Step 2: Extract & persist from Snapshot 2 (BizJournals - Syndicated) ---
    facts_2 = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snap_biz,
        known_entities=known_entities,
    )
    assert len(facts_2) == 2
    funding_fact_2 = next(f for f in facts_2 if f.fact_category == FactCategory.FUNDING)
    # Syndicated source must NOT inflate composite confidence to HIGH
    assert funding_fact_2.composite_confidence == ConfidenceLevel.MEDIUM
    assert funding_fact_2.composite_confidence != ConfidenceLevel.HIGH
    # Both distinct evidence records are preserved under the single canonical fact
    assert len(funding_fact_2.evidence_records) == 2
    sources_in_ev = {e.source_id for e in funding_fact_2.evidence_records}
    assert sources_in_ev == {src_tc, src_biz}

    # --- Step 3: Repeated processing of Snapshot 1 (Idempotency Check) ---
    facts_repeat = service.extract_and_persist_facts_for_session(
        session_id=session_id,
        snapshot_id=snap_tc,
        known_entities=known_entities,
    )
    assert len(facts_repeat) == 2

    # --- Step 4: Database Reload & Persistence Integrity Verification ---
    reloaded_store = StateStore(db_path=test_state_store.db_path)
    with reloaded_store.uow() as uow:
        # Check canonical facts in DB
        db_facts = uow.assertions.get_facts_for_session(session_id)
        assert len(db_facts) == 2

        # Check total evidence passages in DB (exactly 3: 2 for funding, 1 for Elena; zero duplicates)
        cursor = uow.conn.execute("SELECT COUNT(*) FROM evidence_passages WHERE snapshot_id IN (?, ?);", (snap_tc, snap_biz))
        assert cursor.fetchone()[0] == 3

        # Check evidence passages linked to the funding canonical fact
        funding_db_fact = next(f for f in db_facts if f["fact_category"] == "funding")
        assert funding_db_fact["composite_confidence"] == "medium"
        linked_ev = uow.assertions.get_evidence_for_fact(funding_db_fact["fact_id"])
        assert len(linked_ev) == 2
        ev_sources = {e["source_id"] for e in linked_ev}
        assert ev_sources == {src_tc, src_biz}

        # Check target prospect attribution retrieval
        prospect_facts = uow.assertions.get_prospect_facts(session_id, prospect_entity_id)
        assert len(prospect_facts) == 1
        assert "Head of Research" in prospect_facts[0]["claim_text"]
        assert prospect_facts[0]["composite_confidence"] == "high"

        # Check candidate facts retrieval (zero candidate facts because both entities were resolved)
        cand_facts = uow.assertions.get_candidate_facts(session_id)
        assert len(cand_facts) == 0

        # Check assessment history (3 distinct iterations recorded)
        hist = uow.assertions.get_assessment_history(session_id)
        assert len(hist) == 3
        assert [h["iteration"] for h in hist] == [1, 2, 3]
