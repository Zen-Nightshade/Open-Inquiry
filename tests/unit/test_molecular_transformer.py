from datetime import date
import pytest

from src.researcher.models import CanonicalFact, ConfidenceLevel, EvidenceRecord, FactCategory, VerificationStatus
from src.researcher.molecular_models import Modality, NormalizedPredicate, ValueType
from src.researcher.molecular_transformer import MolecularTransformer, parse_molecular_value


@pytest.fixture
def transformer():
    return MolecularTransformer()


def test_parse_molecular_value_types():
    """Verify numeric, currency, location, and string parsing."""
    # Currency
    v_curr = parse_molecular_value("$50 million in Series B")
    assert v_curr.value_type == ValueType.CURRENCY
    assert v_curr.normalized_value == 50000000.0
    assert v_curr.unit == "USD"

    v_eur = parse_molecular_value("€25.5M")
    assert v_eur.value_type == ValueType.CURRENCY
    assert v_eur.normalized_value == 25500000.0
    assert v_eur.unit == "EUR"

    # Number with unit
    v_num = parse_molecular_value("500 engineers")
    assert v_num.value_type == ValueType.NUMBER
    assert v_num.normalized_value == 500.0
    assert v_num.unit == "engineers"

    # Location
    v_loc = parse_molecular_value("in Munich, Germany")
    assert v_loc.value_type == ValueType.LOCATION
    assert v_loc.normalized_value == "Munich, Germany"

    # Plain text
    v_str = parse_molecular_value("advanced generative compiler")
    assert v_str.value_type == ValueType.STRING


def test_coordinated_clause_decomposition_success(transformer):
    """Verify bounded decomposition splits explicit coordinating conjunctions with distinct predicates."""
    fact = CanonicalFact(
        fact_id="fact_compound",
        claim_text="Acme Corp opened a facility in Hyderabad and plans to hire 500 engineers.",
        fact_category=FactCategory.STRATEGY,
        entity_ids=["ent_acme"],
        primary_event_date=date(2025, 3, 1),
        composite_confidence=ConfidenceLevel.HIGH,
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_original",
                source_id="src_press",
                verbatim_excerpt="Acme Corp opened a facility in Hyderabad and plans to hire 500 engineers.",
            )
        ],
    )

    atoms = transformer.transform(fact, subject_name_override="Acme Corp")
    assert len(atoms) == 2

    # Atom 1: Facility Opened
    atom1 = atoms[0]
    assert atom1.parent_canonical_id == "fact_compound"
    assert atom1.normalized_predicate == NormalizedPredicate.FACILITY_OPENED
    assert atom1.modality == Modality.DEFINITIVE
    assert atom1.is_decomposed is True
    assert atom1.confidence == ConfidenceLevel.HIGH
    assert atom1.verification_status == VerificationStatus.SUPPORTED

    # Atom 2: Hiring Planned
    atom2 = atoms[1]
    assert atom2.parent_canonical_id == "fact_compound"
    assert atom2.normalized_predicate == NormalizedPredicate.HIRING_PLANNED
    assert atom2.modality == Modality.PLANNED
    assert atom2.object_value.value_type == ValueType.NUMBER
    assert atom2.object_value.normalized_value == 500.0
    assert atom2.object_value.unit == "engineers"
    assert atom2.is_decomposed is True
    assert atom2.confidence == ConfidenceLevel.HIGH


def test_ambiguous_clause_fallback_preserves_single_atom(transformer):
    """When a claim contains subordinate, relative, or ambiguous clauses, retain as single atom."""
    # Sentence with relative clause ("who joined after...")
    complex_text = "Acme Corp appointed Sarah Connor who joined after leading engineering at rival firm."
    fact = CanonicalFact(
        fact_id="fact_complex",
        claim_text=complex_text,
        fact_category=FactCategory.HIRING,
        entity_ids=["ent_acme"],
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_complex",
                source_id="src_news",
                verbatim_excerpt=complex_text,
            )
        ],
    )

    atoms = transformer.transform(fact, subject_name_override="Acme Corp")
    # Must NOT split into questionable atoms
    assert len(atoms) == 1
    assert atoms[0].is_decomposed is False
    assert atoms[0].parent_canonical_id == "fact_complex"
    assert atoms[0].atom_text == complex_text


def test_no_verification_inflation_or_synthetic_spans(transformer):
    """Decomposition inherits confidence and does not fabricate synthetic evidence spans."""
    fact_medium = CanonicalFact(
        fact_id="fact_med",
        claim_text="Vanguard AI launched a new compiler and partnered with Google Cloud.",
        fact_category=FactCategory.PRODUCT_LAUNCH,
        composite_confidence=ConfidenceLevel.MEDIUM,
        evidence_records=[
            EvidenceRecord(
                evidence_id="ev_single",
                source_id="src_1",
                verbatim_excerpt="Vanguard AI launched a new compiler and partnered with Google Cloud.",
                confidence=ConfidenceLevel.MEDIUM,
            )
        ],
    )

    atoms = transformer.transform(fact_medium, subject_name_override="Vanguard AI")
    assert len(atoms) == 2
    for atom in atoms:
        # Confidence must remain MEDIUM, not upgraded to HIGH
        assert atom.confidence == ConfidenceLevel.MEDIUM
        assert atom.parent_canonical_id == "fact_med"
