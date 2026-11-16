from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from field_feedback.observations.report import ContractError, Decision, DomainRecord, records_from_fixture, stable_hash
from field_feedback.history.release_archive import Repository
from field_feedback.releases.progressive_delivery import advance_for_outcome, decide
from field_feedback.insights.regional_digest import export_bundle, verify_bundle
from field_feedback.triage.multilingual_clusters import derive_context
from field_feedback.triage.cluster_registry import ClusterRegistry
from field_feedback.releases.fleet_delivery import FleetDelivery
from field_feedback.releases.evidence_review import EvidenceReview


class Application:
    def __init__(self, database: str | Path, context: Mapping[str, Any] | None = None):
        self.repository = Repository(database)
        self.cluster_registry = ClusterRegistry(self.repository)
        self.fleet_delivery = FleetDelivery(self.repository)
        self.evidence_review = EvidenceReview(self.repository, self.fleet_delivery, self.cluster_registry)
        self.context = dict(context or {})

    def ingest(self, record: DomainRecord) -> DomainRecord:
        current = self.repository.get_record(record.record_id)
        if current is None:
            self.repository.save_record(record)
            return record
        if current.fingerprint == record.fingerprint:
            return current
        if current.payload == record.payload:
            # 重复导入同一证据（导入文件仍带初始版本/状态）：保留已推进的工作流状态与版本，
            # 绝不因重放制造新记录版本或再次改变放量状态。
            return current
        replacement = DomainRecord(record.record_id, record.payload, current.state,
                                   current.version + 1, record.occurred_at, record.source)
        self.repository.save_record(replacement, expected_version=current.version)
        return replacement

    def _decide_with_full_evidence(self, record: DomainRecord) -> Decision:
        # 上下文始终基于当前库内的完整证据集合；list_records 按 record_id 排序，
        # 聚类顺序无关，因此同一证据集合无论导入顺序如何都推出同一决策。
        context = dict(self.context)
        context.update(derive_context(record, self.repository))
        return decide(record, context)

    def _bind_decision(self, decision: Decision, basis: DomainRecord, final_record: DomainRecord) -> None:
        """决策依据的是评议时的报告，但其可追溯锚点更新为最终落库版本，
        这样重复导入同一证据时 current_decision_for 能命中而不会重复评议或重复放量。"""
        self.repository.save_decision(decision, basis)
        if final_record.fingerprint != basis.fingerprint:
            self.repository.attach_decision_to_record(decision.fingerprint, final_record)

    def process(self, record: DomainRecord) -> tuple[DomainRecord, Decision]:
        stored = self.ingest(record)
        existing = self.repository.current_decision_for(stored)
        if existing is not None:
            # 重复导入：决策、状态与放量名额都已落库，重放不得再次改变放量状态。
            self.fleet_delivery.reconcile(stored, existing)
            return stored, existing
        # 新证据先完成簇登记，再基于完整证据集合评议，与批量导入走同一条链路。
        self.cluster_registry.reconcile(self.repository.list_records())
        decision = self._decide_with_full_evidence(stored)
        advanced = advance_for_outcome(stored, decision)
        if advanced.fingerprint != stored.fingerprint:
            self.repository.save_record(advanced, expected_version=stored.version)
        self._bind_decision(decision, stored, advanced)
        self.fleet_delivery.reconcile(advanced, decision)
        return advanced, decision

    def run_fixture(self, fixture: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
        records = records_from_fixture(fixture)
        fixture_hash = stable_hash(fixture)
        run_id = self.repository.begin_run(fixture_hash, len(records))
        completed = self.repository.completed_run(run_id)
        if completed is not None:
            return completed
        # 先整体入库并登记问题簇，再逐条评议：每条记录看到的是同一完整证据集合，
        # 重排导入顺序不会改变簇、上下文权重、决策或最终放量状态。
        for record in records:
            self.ingest(record)
        self.cluster_registry.reconcile(self.repository.list_records())
        decisions: dict[str, Decision] = {}
        final_records: list[DomainRecord] = []
        for record in records:
            stored = self.repository.get_record(record.record_id)
            assert stored is not None
            existing = self.repository.current_decision_for(stored)
            if existing is not None:
                decision = existing
                final_record = stored
            else:
                decision = self._decide_with_full_evidence(stored)
                advanced = advance_for_outcome(stored, decision)
                if advanced.fingerprint != stored.fingerprint:
                    self.repository.save_record(advanced, expected_version=stored.version)
                    final_record = advanced
                else:
                    final_record = stored
                self._bind_decision(decision, stored, final_record)
                self.fleet_delivery.reconcile(final_record, decision)
            final_records.append(final_record)
            decisions[record.record_id] = decision
        snapshot = self.repository.snapshot()
        snapshot["fleet_delivery"] = self.fleet_delivery.inspect()
        snapshot["clusters"] = self.cluster_registry.clusters()
        if snapshot["chain_errors"]:
            raise ContractError("audit chain verification failed")
        manifest = export_bundle(output, final_records, decisions, snapshot, self.repository.event_rows())
        result = {"run_id": run_id, "fixture_hash": fixture_hash, "records": len(final_records),
                  "decisions": len(decisions), "manifest": manifest, "snapshot": snapshot}
        self.repository.finish_run(run_id, result)
        return result

    def inspect(self) -> dict[str, Any]:
        return {"snapshot": self.repository.snapshot(), "records": [record.__dict__ for record in self.repository.list_records()],
                "decisions": self.repository.decision_rows(), "fleet_delivery": self.fleet_delivery.inspect(),
                "clusters": self.cluster_registry.clusters()}


def load_fixture(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read fixture: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("fixture root must be an object")
    return value


def run(fixture_path: str | Path, database: str | Path, output: str | Path,
        context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return Application(database, context).run_fixture(load_fixture(fixture_path), output)


def verify(database: str | Path, output: str | Path) -> dict[str, Any]:
    repository = Repository(database)
    chain_errors = repository.verify_chain()
    bundle_errors = verify_bundle(output)
    return {"ok": not chain_errors and not bundle_errors, "chain_errors": chain_errors,
            "bundle_errors": bundle_errors, "snapshot": repository.snapshot()}
