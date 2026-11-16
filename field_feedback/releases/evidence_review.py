"""发布评议冻结现场报告及问题归并依据，送达前检查报告版本。

问题簇编号来自持久化 ClusterRegistry：同一证据集合在任何导入顺序下得到相同分组与编号；
当后续报告把两个旧簇连接起来时，待审建议看到合并后的完整证据，而历史评议引用的原簇
编号仍可通过合并谱系解析并解释。
"""
import json

from field_feedback.observations.report import ContractError, canonical_json, stable_hash
from field_feedback.triage.multilingual_clusters import clusters
from field_feedback.triage.cluster_registry import ClusterRegistry


class EvidenceReview:
    def __init__(self, repository, fleet, registry: ClusterRegistry | None = None):
        self.repository = repository
        self.fleet = fleet
        self.registry = registry or ClusterRegistry(repository)
        with repository.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS release_reviews("
                      "review_id TEXT PRIMARY KEY,evidence_hash TEXT NOT NULL,body_json TEXT NOT NULL)")

    def _build_body(self, report_ids):
        rows = [self.repository.get_record(i) for i in report_ids]
        if any(r is None for r in rows):
            raise ContractError("报告不存在")
        # 先排序再聚类：即便调用方乱序传入，分组与证据哈希也保持确定。
        ordered = sorted(rows, key=lambda record: record.record_id)
        self.registry.reconcile(ordered)
        groups = []
        for group in clusters(ordered):
            members = sorted(record.record_id for record in group)
            view = self.registry.cluster_for_record(members[0])
            cluster_id = view["cluster_id"] if view is not None else stable_hash(members)[:24]
            groups.append({
                "cluster_id": cluster_id,
                "content_fingerprint": view["content_fingerprint"] if view is not None else None,
                "members": members,
                "symptom_tokens": view["symptom_tokens"] if view is not None else (),
                "firmware_versions": view["firmware_versions"] if view is not None else (),
                "countries": view["countries"] if view is not None else (),
                "device_models": view["device_models"] if view is not None else (),
            })
        groups.sort(key=lambda item: item["cluster_id"])
        return {'reports': {r.record_id: r.fingerprint for r in ordered}, 'groups': groups}

    def capture(self, review_id, report_ids):
        if not review_id or not report_ids or len(set(report_ids)) != len(report_ids):
            raise ContractError("评议与报告范围无效")
        body = self._build_body(report_ids)
        h = stable_hash(body)
        with self.repository.transaction() as c:
            old = c.execute("SELECT * FROM release_reviews WHERE review_id=?", (review_id,)).fetchone()
            if old:
                if old['evidence_hash'] == h:
                    return self._normalized(json.loads(old['body_json']))
                stored = json.loads(old['body_json'])
                # 报告本身（指纹）被改写才是不可覆盖的依据冲突；仅问题簇合并导致的分组变化
                # 不覆盖原评议——原编号经合并谱系仍可解释，合并后证据通过 explain 查看。
                if stored.get('reports') != body['reports']:
                    raise ContractError("评议依据不能覆盖")
                return self._normalized(stored)
            c.execute("INSERT INTO release_reviews VALUES(?,?,?)", (review_id, h, canonical_json(body)))
        return self._normalized(body)

    @staticmethod
    def _normalized(body):
        # 与落库后重载的键顺序保持一致，保证重复评议返回完全相等的结构。
        return json.loads(json.dumps(body, sort_keys=True, ensure_ascii=False))

    def explain(self, review_id):
        """返回评议当前应看到的合并证据，以及每个被引用簇的合并谱系。"""
        with self.repository.connect() as c:
            row = c.execute("SELECT body_json FROM release_reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            raise ContractError("评议不存在")
        body = json.loads(row[0])
        resolved_groups = []
        for group in body['groups']:
            try:
                resolved = self.registry.resolve_cluster(group['cluster_id'])
            except ContractError:
                resolved = {"cluster_id": group['cluster_id'], "members": group['members'], "lineage": ()}
            resolved_groups.append({
                "cluster_id": resolved["cluster_id"],
                "historical_cluster_id": group['cluster_id'],
                "content_fingerprint": resolved.get("content_fingerprint", group.get("content_fingerprint")),
                "members": resolved.get("members", group['members']),
                "firmware_versions": resolved.get("firmware_versions", group.get("firmware_versions", ())),
                "countries": resolved.get("countries", group.get("countries", ())),
                "device_models": resolved.get("device_models", group.get("device_models", ())),
                "symptom_tokens": resolved.get("symptom_tokens", group.get("symptom_tokens", ())),
                "lineage": resolved.get("lineage", ()),
            })
        resolved_groups.sort(key=lambda item: item["cluster_id"])
        return {"review_id": review_id, "reports": body["reports"], "groups": resolved_groups,
                "merges": self.registry.merges()}

    def deliver(self, review_id, record_id, decision):
        with self.repository.connect() as c:
            row = c.execute("SELECT body_json FROM release_reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            raise ContractError("评议不存在")
        body = json.loads(row[0])
        if record_id not in body['reports']:
            raise ContractError("评议未引用当前报告")
        for rid, h in body['reports'].items():
            current = self.repository.get_record(rid)
            if current is None or current.fingerprint != h:
                raise ContractError("评议报告已改变")
        record = self.repository.get_record(record_id)
        self.fleet.reconcile(record, decision)
        return body
