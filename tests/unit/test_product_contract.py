from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.cli.main import app as cli_app
from src.common.provenance import OutputProvenance, ProvenanceOrigin
from src.context.models import (
    CapabilityNode,
    CompatibilityResult,
    CompatibilityStatus,
    ConstraintType,
    ProductConstraint,
    ProductContract,
    ProductContext,
    ProspectContext,
    OutreachContext,
)
from src.db.state_store import StateStore
from src.orchestration import ApplicationOrchestrator, RunResult, RunStatus

runner = CliRunner()


# =============================================================================
# 1. Model Validation Tests
# =============================================================================

def test_valid_product_contract_creation():
    cap1 = CapabilityNode(
        name="Distributed Checkpointing",
        description="Near-zero overhead fault recovery",
        target_operational_problems=["GPU cluster preemption", "Long model training failures"],
        target_buyer_personas=["VP Engineering", "Head of AI Infrastructure"],
        supported_use_cases=["LLM pretraining", "Distributed fine-tuning"],
        scope=["Kubernetes >= 1.26", "NVIDIA H100", "NVIDIA A100"],
        required_prerequisites=["Kubernetes", "Shared NVMe storage"],
        hard_exclusions=["Windows Server", "Local single-node desktop"],
        verified_proof_points=["Cut failover checkpoint time from 15m to 12s at 1,024 GPUs"],
    )
    constraint = ProductConstraint(
        name="Network Bandwidth Requirement",
        constraint_type=ConstraintType.PREREQUISITE,
        description="Requires RoCEv2 or InfiniBand interconnect",
        details=[">= 400 Gbps interconnect"],
    )
    contract = ProductContract(
        product_name="ComputeScale",
        contract_version="1.0.0",
        description="Distributed GPU orchestration engine",
        value_proposition="Cut AI training cloud compute waste by 40%",
        capabilities=[cap1],
        global_constraints=[constraint],
        hard_exclusions=["MongoDB", "Single-tenant bare metal without container runtime"],
        is_exhaustive=False,
    )

    assert contract.product_name == "ComputeScale"
    assert len(contract.capabilities) == 1
    assert contract.capabilities[0].name == "Distributed Checkpointing"
    assert contract.capabilities[0].provenance.origin == ProvenanceOrigin.MANUAL_INPUT
    assert contract.is_exhaustive is False


def test_empty_or_malformed_capability_rejected():
    # Empty capability name
    with pytest.raises(ValidationError):
        CapabilityNode(name="")

    # Whitespace only capability name
    with pytest.raises(ValidationError):
        CapabilityNode(name="   ")

    # Empty product name in contract
    with pytest.raises(ValidationError):
        ProductContract(
            product_name="   ",
            description="Valid description",
            value_proposition="Valid value prop",
            capabilities=[],
        )

    # Empty description in contract
    with pytest.raises(ValidationError):
        ProductContract(
            product_name="Valid Name",
            description="",
            value_proposition="Valid value prop",
            capabilities=[],
        )

    # Empty value proposition in contract
    with pytest.raises(ValidationError):
        ProductContract(
            product_name="Valid Name",
            description="Valid description",
            value_proposition="   ",
            capabilities=[],
        )


def test_invalid_constraint_rejected():
    with pytest.raises(ValidationError):
        ProductConstraint(
            name="   ",
            constraint_type=ConstraintType.HARD_EXCLUSION,
        )


def test_duplicate_capability_names_rejected():
    cap1 = CapabilityNode(name="Cluster Telemetry")
    cap2 = CapabilityNode(name="cluster telemetry")  # Case-insensitive duplicate

    with pytest.raises(ValidationError, match="Duplicate capability name"):
        ProductContract(
            product_name="ComputeScale",
            description="Valid desc",
            value_proposition="Valid VP",
            capabilities=[cap1, cap2],
        )


def test_capability_prerequisite_contradiction_rejected():
    # A capability cannot list the same item as both a prerequisite and an exclusion
    with pytest.raises(ValidationError, match="both a prerequisite and a hard exclusion"):
        CapabilityNode(
            name="Cloud Connector",
            required_prerequisites=["AWS VPC"],
            hard_exclusions=["AWS VPC"],
        )


def test_contract_capability_vs_global_exclusion_contradiction_rejected():
    # Contract hard_exclusion forbids MongoDB, but capability claims MongoDB support
    cap = CapabilityNode(
        name="MongoDB Sync",
        description="Syncs data to MongoDB",
    )
    with pytest.raises(ValidationError, match="directly conflicts with declared hard exclusion"):
        ProductContract(
            product_name="DataPipe",
            description="Data pipeline",
            value_proposition="Real-time syncing",
            capabilities=[cap],
            hard_exclusions=["MongoDB Sync"],
        )


def test_contract_capability_use_case_vs_global_exclusion_rejected():
    cap = CapabilityNode(
        name="Data Export",
        supported_use_cases=["MongoDB Replication"],
    )
    with pytest.raises(ValidationError, match="conflicts with hard exclusion"):
        ProductContract(
            product_name="DataPipe",
            description="Data pipeline",
            value_proposition="Real-time syncing",
            capabilities=[cap],
            hard_exclusions=["MongoDB Replication"],
        )


def test_contract_capability_prerequisite_vs_global_exclusion_rejected():
    cap = CapabilityNode(
        name="High Throughput Bus",
        required_prerequisites=["On-Premise SAN"],
    )
    with pytest.raises(ValidationError, match="prerequisite.*conflicts with hard exclusion"):
        ProductContract(
            product_name="DataPipe",
            description="Data pipeline",
            value_proposition="Real-time syncing",
            capabilities=[cap],
            hard_exclusions=["On-Premise SAN"],
        )


def test_contract_serialization_round_trip():
    cap = CapabilityNode(
        name="Dynamic Gang Scheduling",
        target_operational_problems=["Gang preemption deadlocks"],
        scope=["Kubernetes"],
        required_prerequisites=["Slurm or Kubernetes"],
        hard_exclusions=["Mesos"],
        verified_proof_points=["Eliminated 99.8% of gang preemption stalls"],
    )
    contract = ProductContract(
        product_name="SchedulerPro",
        contract_version="2.1.0",
        description="Advanced scheduler",
        value_proposition="Zero gang preemption stalls",
        capabilities=[cap],
        hard_exclusions=["Windows Server"],
        is_exhaustive=True,
    )

    json_data = contract.model_dump_json()
    deserialized = ProductContract.model_validate_json(json_data)

    assert deserialized.product_name == contract.product_name
    assert deserialized.contract_version == "2.1.0"
    assert deserialized.is_exhaustive is True
    assert len(deserialized.capabilities) == 1
    assert deserialized.capabilities[0].name == "Dynamic Gang Scheduling"
    assert deserialized.capabilities[0].hard_exclusions == ["Mesos"]


# =============================================================================
# 2. Compatibility Semantics Tests
# =============================================================================

@pytest.fixture
def sample_contract() -> ProductContract:
    cap_checkpoint = CapabilityNode(
        name="Distributed Checkpointing",
        required_prerequisites=["Shared NVMe Storage", "Kubernetes"],
        scope=["Kubernetes 1.24+"],
        hard_exclusions=["Local Disk Only"],
    )
    cap_telemetry = CapabilityNode(
        name="eBPF Telemetry",
        required_prerequisites=[],
        scope=["Linux Kernel >= 5.8"],
        hard_exclusions=["Windows Server"],
    )
    return ProductContract(
        product_name="ComputeScale",
        description="GPU infrastructure optimization platform",
        value_proposition="Cut GPU waste by 40%",
        capabilities=[cap_checkpoint, cap_telemetry],
        hard_exclusions=["MongoDB", "Local Desktop Deployments"],
        is_exhaustive=False,
    )


def test_compatibility_explicitly_supported_no_prerequisites(sample_contract):
    # eBPF Telemetry has no prerequisites -> VERIFIED_COMPATIBLE
    res = sample_contract.evaluate_compatibility("eBPF Telemetry")
    assert res.status == CompatibilityStatus.VERIFIED_COMPATIBLE
    assert "verified compatible" in res.reason.lower()
    assert res.matched_capability_id is not None
    assert res.matched_scope == ["Linux Kernel >= 5.8"]


def test_compatibility_explicit_hard_exclusion(sample_contract):
    # Global hard exclusion: MongoDB
    res_global = sample_contract.evaluate_compatibility("MongoDB Support")
    assert res_global.status == CompatibilityStatus.INCOMPATIBLE
    assert "hard exclusion" in res_global.reason.lower()
    assert "mongodb" in res_global.violated_exclusions[0].lower()

    # Capability-level hard exclusion: Windows Server for eBPF Telemetry
    res_cap_excl = sample_contract.evaluate_compatibility(
        "eBPF Telemetry",
        context_attributes=["Windows Server 2022"],
    )
    assert res_cap_excl.status == CompatibilityStatus.INCOMPATIBLE
    assert "windows server" in res_cap_excl.violated_exclusions[0].lower()


def test_compatibility_with_satisfied_prerequisites(sample_contract):
    # Distributed Checkpointing requires Shared NVMe Storage and Kubernetes
    res = sample_contract.evaluate_compatibility(
        "Distributed Checkpointing",
        known_conditions={
            "shared nvme storage": True,
            "kubernetes": True,
        },
    )
    assert res.status == CompatibilityStatus.VERIFIED_COMPATIBLE
    assert len(res.missing_prerequisites) == 0
    assert len(res.unknown_conditions) == 0


def test_compatibility_with_unmet_prerequisite(sample_contract):
    # Distributed Checkpointing with explicitly unsatisfied prerequisite (False)
    res = sample_contract.evaluate_compatibility(
        "Distributed Checkpointing",
        known_conditions={
            "shared nvme storage": False,  # Confirmed absent
            "kubernetes": True,
        },
    )
    assert res.status == CompatibilityStatus.INCOMPATIBLE
    assert "unsatisfied prerequisites" in res.reason.lower()
    assert "Shared NVMe Storage" in res.missing_prerequisites


def test_compatibility_with_unknown_prerequisite(sample_contract):
    # Distributed Checkpointing without providing condition status -> CONDITIONAL_ON_UNKNOWN
    res = sample_contract.evaluate_compatibility("Distributed Checkpointing")
    assert res.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
    assert "unknown prerequisites" in res.reason.lower()
    assert "Shared NVMe Storage" in res.missing_prerequisites
    assert "Kubernetes" in res.missing_prerequisites
    assert len(res.unknown_conditions) == 2


def test_compatibility_undeclared_capability_non_exhaustive(sample_contract):
    # In a non-exhaustive contract (is_exhaustive=False), undeclared capability is CONDITIONAL_ON_UNKNOWN
    res = sample_contract.evaluate_compatibility("Automated Right-Sizing")
    assert res.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
    assert "not declared" in res.reason.lower()
    assert res.status != CompatibilityStatus.VERIFIED_COMPATIBLE
    assert res.status != CompatibilityStatus.INCOMPATIBLE


def test_compatibility_undeclared_capability_exhaustive():
    # In an exhaustive contract (is_exhaustive=True), undeclared capability is strictly INCOMPATIBLE
    exhaustive_contract = ProductContract(
        product_name="StrictApp",
        description="Strict closed-world product",
        value_proposition="Only does declared features",
        capabilities=[CapabilityNode(name="Feature A")],
        is_exhaustive=True,
    )
    res = exhaustive_contract.evaluate_compatibility("Feature B")
    assert res.status == CompatibilityStatus.INCOMPATIBLE
    assert "exhaustive product contract" in res.reason.lower()


def test_unknown_never_defaults_to_compatible_or_incompatible(sample_contract):
    # Verify strict invariant: unknown conditions do NOT default to compatible or incompatible
    res = sample_contract.evaluate_compatibility("Unseen Capability X")
    assert res.status == CompatibilityStatus.CONDITIONAL_ON_UNKNOWN
    assert res.status != CompatibilityStatus.VERIFIED_COMPATIBLE
    assert res.status != CompatibilityStatus.INCOMPATIBLE


# =============================================================================
# 3. ProductContext Adapter Tests
# =============================================================================

def test_product_context_legacy_adapter():
    legacy_product = ProductContext(
        product_name="LegacyTool",
        description="Plain text tool",
        capabilities=["Feature Alpha", "Feature Beta"],
        value_proposition="Simple productivity",
        limitations=["No air-gapped support"],
    )
    # Automatically creates a valid non-exhaustive ProductContract
    contract = legacy_product.get_contract()
    assert contract.product_name == "LegacyTool"
    assert contract.is_exhaustive is False
    assert len(contract.capabilities) == 2
    assert contract.capabilities[0].name == "Feature Alpha"
    assert contract.capabilities[0].provenance.origin == ProvenanceOrigin.MANUAL_INPUT
    assert "No air-gapped support" in contract.hard_exclusions

    # Compatibility check works on adapted contract
    res = contract.evaluate_compatibility("Feature Alpha")
    assert res.status == CompatibilityStatus.VERIFIED_COMPATIBLE

    res_excl = contract.evaluate_compatibility("No air-gapped support")
    assert res_excl.status == CompatibilityStatus.INCOMPATIBLE


def test_product_context_with_explicit_contract():
    cap = CapabilityNode(
        name="Structured Extraction",
        required_prerequisites=["PDF parser"],
    )
    contract = ProductContract(
        product_name="DocAI",
        description="Document intelligence",
        value_proposition="Extract text fast",
        capabilities=[cap],
        is_exhaustive=True,
    )
    product = ProductContext(
        product_name="DocAI",
        description="Document intelligence",
        value_proposition="Extract text fast",
        contract=contract,
    )
    # Capabilities automatically populated from contract
    assert product.capabilities == ["Structured Extraction"]
    assert product.get_contract() == contract


# =============================================================================
# 4. Application Boundary: API Tests
# =============================================================================

def test_api_create_run_with_structured_product_contract():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_test.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        payload = {
            "prospect": {
                "full_name": "Elena Rostova",
                "company_name": "Vanguard AI",
            },
            "product": {
                "product_name": "ComputeScale",
                "description": "GPU platform",
                "value_proposition": "Cut GPU waste",
                "contract": {
                    "contract_version": "1.2.0",
                    "capabilities": [
                        {
                            "name": "Dynamic Gang Scheduling",
                            "required_prerequisites": ["Kubernetes"],
                            "hard_exclusions": ["Windows"],
                        }
                    ],
                    "hard_exclusions": ["MongoDB"],
                    "is_exhaustive": True,
                },
            },
            "outreach": {
                "objective": "Briefing",
                "sender_identity": "Alex Mercer",
            },
        }

        with patch("src.api.routes.runs._background_execute_run"):
            response = client.post("/api/v1/runs", json=payload)
            assert response.status_code == 202
            run_id = response.json()["run_id"]

            # Verify persisted snapshot in database contains structured contract
            saved_run = store.get_run(run_id)
            assert saved_run is not None

            # Retrieve product version snapshot
            pv = store.get_latest_version_for_product("ComputeScale")
            assert pv is not None
            retrieved_ctx = store.get_product_context(pv["version_id"])
            assert retrieved_ctx is not None
            assert retrieved_ctx.contract is not None
            assert retrieved_ctx.contract.is_exhaustive is True
            assert retrieved_ctx.contract.capabilities[0].name == "Dynamic Gang Scheduling"


def test_api_create_run_with_invalid_contract_rejected():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "api_test.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        app = create_app(db_path=db_path)
        client = TestClient(app)

        # Contradictory contract: capability name conflicts with hard_exclusion
        bad_payload = {
            "prospect": {
                "full_name": "Elena Rostova",
                "company_name": "Vanguard AI",
            },
            "product": {
                "product_name": "ComputeScale",
                "description": "GPU platform",
                "value_proposition": "Cut GPU waste",
                "contract": {
                    "capabilities": [
                        {
                            "name": "MongoDB Connector",
                        }
                    ],
                    "hard_exclusions": ["MongoDB Connector"],
                },
            },
            "outreach": {
                "objective": "Briefing",
                "sender_identity": "Alex Mercer",
            },
        }

        response = client.post("/api/v1/runs", json=bad_payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data


# =============================================================================
# 5. Application Boundary: CLI Tests
# =============================================================================

def test_cli_run_with_contract_file():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "cli_test.db")
        contract_path = Path(td) / "contract.json"

        contract_data = {
            "product_name": "ComputeScale CLI",
            "contract_version": "1.0.0",
            "description": "CLI test product",
            "value_proposition": "Cut GPU waste",
            "capabilities": [
                {
                    "name": "Dynamic Gang Scheduling",
                    "required_prerequisites": ["Kubernetes"],
                }
            ],
            "hard_exclusions": ["MongoDB"],
            "is_exhaustive": False,
        }
        contract_path.write_text(json.dumps(contract_data), encoding="utf-8")

        dummy_result = MagicMock()
        dummy_result.status = RunStatus.COMPLETED
        dummy_result.strategy_result = None
        dummy_result.stage_records = []
        dummy_result.run_id = "run-contract-cli-test"

        with patch.object(ApplicationOrchestrator, "run", return_value=dummy_result) as mock_run:
            res = runner.invoke(
                cli_app,
                [
                    "run",
                    "-n", "Elena Rostova",
                    "-c", "Vanguard AI",
                    "--contract-file", str(contract_path),
                    "--db-path", db_path,
                    "--non-interactive",
                ],
            )
            assert res.exit_code == 0
            assert mock_run.called
            # Verify ProductContext passed into orchestrator run contains the contract
            call_kwargs = mock_run.call_args.kwargs
            passed_product: ProductContext = call_kwargs["product"]
            assert passed_product.product_name == "ComputeScale CLI"
            assert passed_product.contract is not None
            assert passed_product.contract.capabilities[0].name == "Dynamic Gang Scheduling"


def test_cli_run_with_invalid_contract_file_fails_cleanly():
    with tempfile.TemporaryDirectory() as td:
        contract_path = Path(td) / "bad_contract.json"
        # Contradictory contract
        bad_data = {
            "product_name": "BadApp",
            "description": "Bad app",
            "value_proposition": "Bad VP",
            "capabilities": [{"name": "Feature Conflict"}],
            "hard_exclusions": ["Feature Conflict"],
        }
        contract_path.write_text(json.dumps(bad_data), encoding="utf-8")

        res = runner.invoke(
            cli_app,
            [
                "run",
                "-n", "Elena Rostova",
                "-c", "Vanguard AI",
                "--contract-file", str(contract_path),
                "--non-interactive",
            ],
        )
        assert res.exit_code != 0
        assert "Invalid product contract file" in res.output


# =============================================================================
# 6. Database Persistence & Backward Compatibility Tests
# =============================================================================

def test_persistence_fresh_database_and_retrieval():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "persist.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        cap = CapabilityNode(
            name="Distributed Cache",
            scope=["Redis >= 7.0"],
            required_prerequisites=["Redis"],
            hard_exclusions=["Memcached"],
        )
        contract = ProductContract(
            product_name="CachePro",
            description="Caching engine",
            value_proposition="Sub-millisecond latency",
            capabilities=[cap],
            is_exhaustive=True,
        )
        product = contract.to_context()

        vid = store.save_product_context(product)
        assert vid is not None

        # Verify round-trip deserialization from SQLite
        loaded_ctx = store.get_product_context(vid)
        assert loaded_ctx is not None
        assert loaded_ctx.product_name == "CachePro"
        assert loaded_ctx.contract is not None
        assert loaded_ctx.contract.is_exhaustive is True
        assert loaded_ctx.contract.capabilities[0].name == "Distributed Cache"
        assert loaded_ctx.contract.capabilities[0].hard_exclusions == ["Memcached"]


def test_persistence_legacy_snapshot_backward_compatibility():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "legacy.db")
        store = StateStore(db_path=db_path)
        store.initialize()

        # Simulate a legacy snapshot_json stored before M1.1 without contract field
        legacy_json = json.dumps({
            "product_name": "OldTool",
            "description": "Pre-M1.1 tool",
            "capabilities": ["Legacy Ingestion", "CSV Export"],
            "value_proposition": "Old value prop",
            "icp": "Legacy ICP",
            "differentiators": ["Simple"],
            "limitations": ["No cloud storage"],
        })

        with store.uow() as uow:
            uow.products.create_product_version(
                version_id="legacy-v1",
                product_name="OldTool",
                snapshot_json=legacy_json,
            )

        # Deserialize using get_product_context
        loaded = store.get_product_context("legacy-v1")
        assert loaded is not None
        assert loaded.product_name == "OldTool"
        assert loaded.contract is None  # Legacy record has no explicit contract
        assert loaded.capabilities == ["Legacy Ingestion", "CSV Export"]

        # get_contract() seamlessly wraps it on-the-fly
        contract = loaded.get_contract()
        assert contract.product_name == "OldTool"
        assert len(contract.capabilities) == 2
        assert contract.capabilities[0].name == "Legacy Ingestion"
        assert contract.is_exhaustive is False
        assert "No cloud storage" in contract.hard_exclusions
