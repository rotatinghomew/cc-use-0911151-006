"""端到端稳定性：同一证据集合在任意导入顺序/重复导入下，簇、决策、放量状态一致，
合并后证据对评议可见且历史簇编号可追溯，固件版本与地区含义保留。"""
from __future__ import annotations

import itertools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from field_feedback.observations.report import ContractError, Decision, DomainRecord
from field_feedback.product_loop import Application, run, verify
from field_feedback.triage.cluster_registry import ClusterRegistry
from field_feedback.triage.multilingual_clusters import clusters


def observation(key: str, symptom: str, country: str = "th", firmware: str = "1",
                model: str = "r1", state: str = "reported") -> DomainRecord:
    return DomainRecord(key, {
        "country": country, "crop_type": "durian", "device_model": model,
        "firmware_version": firmware, "symptom_code": symptom, "severity": 7,
        "weather": "wet", "terrain": "slope", "operator_skill": 2,
        "reproduction_rate": 0.8, "affected_units": 4, "translation_confidence": 0.9,
    }, state)


def table_counts(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("events", "decisions", "record_versions", "cluster_atoms",
                              "cluster_links", "cluster_merges", "rollout_intents", "fleet_deliveries")}
    finally:
        connection.close()


class StableClusteringTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            observation("a", "stuck arm", "th", "1"),
            observation("b", "卡住 arm", "th", "1"),
            observation("c", "slow lift", "vn", "2"),
            observation("d", "lambat lift", "vn", "2"),
            observation("e", "stuck arm", "id", "3"),
        ]

    def test_clustering_is_order_independent(self):
        baseline = [[r.record_id for r in group] for group in clusters(self.records)]
        for order in itertools.permutations(self.records):
            variant = [[r.record_id for r in group] for group in clusters(list(order))]
            self.assertEqual(variant, baseline)

    def test_synonym_symptoms_still_merge_across_languages(self):
        groups = clusters(self.records)
        members = {tuple(group[0].record_id for group in [g]) for g in groups}
        flat = sorted(member for group in groups for member in
                      [r.record_id for r in group])
        self.assertEqual(flat, ["a", "b", "c", "d", "e"])
        stuck_group = next(group for group in groups if {r.record_id for r in group} >= {"a", "b"})
        self.assertTrue({"a", "b", "e"}.issubset({r.record_id for r in stuck_group}))

    def test_registry_ids_stable_across_orders_and_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "db.sqlite3"
            app = Application(database)
            for record in self.records:
                app.ingest(record)
            app.cluster_registry.reconcile(app.repository.list_records())
            first = {view["cluster_id"]: view["members"] for view in app.cluster_registry.clusters()}
            restarted = Application(database).cluster_registry.clusters()
            second = {view["cluster_id"]: view["members"] for view in restarted}
            self.assertEqual(first, second)

    def test_firmware_and_region_are_preserved_not_flattened(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(Path(directory) / "db.sqlite3")
            for record in self.records:
                app.ingest(record)
            app.cluster_registry.reconcile(app.repository.list_records())
            view = app.cluster_registry.cluster_for_record("a")
            self.assertIn("1", view["firmware_versions"])
            self.assertIn("3", view["firmware_versions"])
            self.assertIn("th", view["countries"])
            self.assertIn("id", view["countries"])


class MergeLineageTests(unittest.TestCase):
    def test_merged_evidence_visible_while_historical_cluster_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(Path(directory) / "db.sqlite3")
            first = observation("a", "stuck arm", "th", "1")
            second = observation("b", "卡住 arm", "th", "1")
            app.process(first)
            app.process(second)
            review = app.evidence_review.capture("rev-1", ["a", "b"])
            historical_id = review["groups"][0]["cluster_id"]

            bridge = observation("c", "stuck arm", "id", "3")
            app.process(bridge)
            merged = app.cluster_registry.cluster_for_record("a")
            self.assertEqual(set(merged["members"]), {"a", "b", "c"})
            self.assertTrue(app.cluster_registry.merges())

            explanation = app.evidence_review.explain("rev-1")
            group = explanation["groups"][0]
            self.assertEqual(set(group["members"]), {"a", "b", "c"})
            self.assertEqual(group["historical_cluster_id"], historical_id)
            resolved = app.cluster_registry.resolve_cluster(historical_id)
            self.assertEqual(set(resolved["members"]), {"a", "b", "c"})
            self.assertIn(historical_id, [group["historical_cluster_id"]])

    def test_review_remains_immutable_and_conflicting_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Application(Path(directory) / "db.sqlite3")
            record = observation("r", "camera")
            app.ingest(record)
            body = app.evidence_review.capture("rev", ["r"])
            self.assertEqual(Application(Path(directory) / "db.sqlite3").evidence_review.capture("rev", ["r"]), body)
            app.ingest(observation("r", "thermal"))
            with self.assertRaises(ContractError):
                app.evidence_review.capture("rev", ["r"])


class IdempotentImportTests(unittest.TestCase):
    def _records(self):
        return [
            observation("a", "stuck arm", "th", "1"),
            observation("b", "卡住 arm", "th", "1"),
            observation("c", "slow lift", "vn", "2"),
        ]

    def test_duplicate_import_creates_no_new_artifacts_or_state_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "db.sqlite3"
            app = Application(database)
            for record in self._records():
                app.process(record)
            before = table_counts(database)
            states_before = {r.record_id: (r.version, r.state) for r in app.repository.list_records()}
            replay = Application(database)
            for record in self._records():
                replay.process(record)
            self.assertEqual(table_counts(database), before)
            states_after = {r.record_id: (r.version, r.state) for r in replay.repository.list_records()}
            self.assertEqual(states_before, states_after)

    def test_reordered_batch_produces_identical_decisions_and_states(self):
        fixture = {"records": [dict(record_id=r.record_id, payload=r.payload, state=r.state, source=r.source)
                               for r in self._records()]}
        summaries = []
        with tempfile.TemporaryDirectory() as directory:
            for name, order in (("forward", fixture["records"]), ("reverse", list(reversed(fixture["records"])))):
                db = Path(directory) / f"{name}.sqlite3"
                out = Path(directory) / f"{name}-out"
                run_fixture = {"records": order}
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(run_fixture))
                run(path, db, out)
                connection = sqlite3.connect(db)
                summaries.append({
                    "decisions": {row[0]: (row[1], round(row[2], 9)) for row in
                                  connection.execute("SELECT record_id,outcome,score FROM decisions ORDER BY record_id")},
                    "states": dict(connection.execute("SELECT record_id,state FROM records ORDER BY record_id")),
                    "versions": dict(connection.execute("SELECT record_id,version FROM records ORDER BY record_id")),
                    "counts": table_counts(db),
                })
        self.assertEqual(summaries[0]["decisions"], summaries[1]["decisions"])
        self.assertEqual(summaries[0]["states"], summaries[1]["states"])
        self.assertEqual(summaries[0]["versions"], summaries[1]["versions"])
        self.assertEqual(summaries[0]["counts"], summaries[1]["counts"])


class FleetConsistencyTests(unittest.TestCase):
    def test_rollout_state_and_scopes_survive_reprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "db.sqlite3"
            app = Application(database)
            scope_th = app.fleet_delivery.register("th", "m1", "v1", [f"t{i}" for i in range(10)])
            scope_vn = app.fleet_delivery.register("vn", "m1", "v2", [f"v{i}" for i in range(10)])
            r1 = observation("r1", "stuck arm", "th", "v1", model="m1", state="canary")
            r2 = observation("r2", "lambat lift", "vn", "v2", model="m1", state="canary")
            app.repository.save_record(r1)
            app.repository.save_record(r2)
            d1 = Decision("r1", "approve", 0.8, ("ok",), (), {})
            d2 = Decision("r2", "approve", 0.8, ("ok",), (), {})
            app.repository.save_decision(d1, r1)
            app.repository.save_decision(d2, r2)
            app.fleet_delivery.reconcile(r1, d1)
            app.fleet_delivery.reconcile(r2, d2)
            before = app.fleet_delivery.inspect()
            # repeated reconcile must not add intents or touch deliveries
            app.fleet_delivery.reconcile(r1, d1)
            app.fleet_delivery.reconcile(r2, d2)
            self.assertEqual(app.fleet_delivery.inspect(), before)
            deliveries = app.fleet_delivery.inspect()["deliveries"]
            self.assertFalse(any(row["scope_id"] == scope_vn and row["unit_id"].startswith("t") for row in deliveries))
            self.assertFalse(any(row["scope_id"] == scope_th and row["unit_id"].startswith("v") for row in deliveries))
            self.assertEqual(sum(row["scope_id"] == scope_th for row in deliveries), 1)
            self.assertEqual(sum(row["scope_id"] == scope_vn for row in deliveries), 1)

class AcceptanceFixtureTests(unittest.TestCase):
    def test_acceptance_run_verifies_and_is_order_stable(self):
        fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "acceptance.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            forward_db, reverse_db = Path(directory) / "f.sqlite3", Path(directory) / "r.sqlite3"
            forward_out, reverse_out = Path(directory) / "f-out", Path(directory) / "r-out"
            forward_fx, reverse_fx = Path(directory) / "f.json", Path(directory) / "r.json"
            reverse_fixture = dict(fixture, records=list(reversed(fixture["records"])))
            forward_fx.write_text(json.dumps(fixture))
            reverse_fx.write_text(json.dumps(reverse_fixture))
            run(forward_fx, forward_db, forward_out)
            run(reverse_fx, reverse_db, reverse_out)
            self.assertTrue(verify(forward_db, forward_out)["ok"])
            cf, cr = sqlite3.connect(forward_db), sqlite3.connect(reverse_db)
            query = "SELECT record_id,outcome,round(score,9) FROM decisions ORDER BY record_id"
            self.assertEqual(cf.execute(query).fetchall(), cr.execute(query).fetchall())
            self.assertEqual(cf.execute("SELECT record_id,state,version FROM records ORDER BY record_id").fetchall(),
                             cr.execute("SELECT record_id,state,version FROM records ORDER BY record_id").fetchall())


if __name__ == "__main__":
    unittest.main()
