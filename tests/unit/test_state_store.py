from __future__ import annotations

import tempfile
from pathlib import Path

from src.context.models import OutreachContext, ProductContext, ProspectContext
from src.db.state_store import StateStore


def test_state_store_integration():
    with tempfile.NamedTemporaryFile() as tf:
        store = StateStore(db_path=tf.name)
        # 1. Initialize DB schema
        store.initialize()
        status = store.get_database_status()
        assert status["is_schema_valid"] is True
        assert status["expected_tables_found"] == 36

        # 2. Save Prospect domain model
        prospect = ProspectContext(
            full_name="Grace Hopper",
            company_name="US Navy",
            company_domain="navy.mil",
            designation="Rear Admiral",
            location="Arlington, VA",
        )
        pid = store.save_prospect(prospect)
        assert pid is not None

        retrieved_p = store.get_prospect(pid)
        assert retrieved_p is not None
        assert retrieved_p["full_name"] == "Grace Hopper"
        assert retrieved_p["company_domain"] == "navy.mil"

        # 3. Save ProductContext domain model
        product = ProductContext(
            product_name="COBOL Compiler",
            description="First human-readable data processing language",
            capabilities=["Automatic translation", "Data division support"],
            value_proposition="Portable business software",
        )
        vid = store.save_product_context(product)
        assert vid is not None

        retrieved_v = store.get_product_version(vid)
        assert retrieved_v is not None
        assert "COBOL Compiler" in retrieved_v["snapshot_json"]

        # 4. Save OutreachContext domain model
        outreach = OutreachContext(
            objective="Introduce standardized programming",
            sender_identity="Grace Hopper",
            desired_cta="Schedule technical review",
        )
        cid = store.save_outreach_context(outreach, prospect_id=pid, version_id=vid)
        assert cid is not None

        retrieved_c = store.get_outreach_context(cid)
        assert retrieved_c is not None
        assert "Introduce standardized programming" in retrieved_c["context_json"]
