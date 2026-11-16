"""持久化问题簇登记处。

簇身份由证据内容（多语言症状、固件版本、地区、机型、成员）决定，与导入顺序无关；
当新证据把两个旧簇连接起来时，合并事件被记录，旧簇编号仍可解析到当前簇，
历史评议引用的原簇因此始终可以解释，而待审建议看到的是合并后的完整证据。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from field_feedback.observations.report import ContractError, DomainRecord, canonical_json, stable_hash, utc_now
from field_feedback.triage.multilingual_clusters import _UnionFind, cluster_identity, similarity, tokens


SCHEMA = """
CREATE TABLE IF NOT EXISTS cluster_atoms(
    atom_id TEXT NOT NULL,
    record_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    firmware_version TEXT NOT NULL,
    country TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_links(
    left_atom_id TEXT NOT NULL,
    right_atom_id TEXT NOT NULL,
    PRIMARY KEY(left_atom_id, right_atom_id)
);
CREATE TABLE IF NOT EXISTS cluster_merges(
    merge_id TEXT PRIMARY KEY,
    surviving_cluster_id TEXT NOT NULL,
    absorbed_cluster_id TEXT NOT NULL,
    bridge_record_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def atom_content(record: DomainRecord) -> dict[str, Any]:
    """原子簇（单条证据）的内容键：固件版本与地区是显式组成部分，不会被稳定编号抹平。"""
    return {
        "record_id": record.record_id,
        "symptom_tokens": tuple(sorted(tokens(record.payload.get("symptom_code", "")))),
        "firmware_version": str(record.payload.get("firmware_version", "")),
        "country": str(record.payload.get("country", "")),
        "device_model": str(record.payload.get("device_model", "")),
    }


def atom_id_for(record: DomainRecord) -> str:
    return stable_hash(atom_content(record))[:24]


class ClusterRegistry:
    def __init__(self, repository: Any, threshold: float = 0.45):
        self.repository = repository
        self.threshold = threshold
        with repository.connect() as connection:
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------
    def _load_atoms(self, connection) -> dict[str, dict[str, Any]]:
        return {row["record_id"]: dict(row) for row in connection.execute("SELECT * FROM cluster_atoms")}

    def _load_links(self, connection) -> set[tuple[str, str]]:
        return {tuple(sorted((row["left_atom_id"], row["right_atom_id"])))
                for row in connection.execute("SELECT left_atom_id,right_atom_id FROM cluster_links")}

    def _finder_with(self, atom_ids: Iterable[str], links: Iterable[tuple[str, str]]) -> _UnionFind:
        finder = _UnionFind(atom_ids)
        for left, right in links:
            if left in finder.parent and right in finder.parent:
                finder.union(left, right)
        return finder

    def reconcile(self, records: Iterable[DomainRecord]) -> list[dict[str, Any]]:
        """把一批证据并入登记处，返回当前问题簇视图（按 cluster_id 排序，顺序无关）。

        重复导入同一证据是幂等的：内容哈希不变就不会产生新原子、新边、新合并或新事件；
        同一报告的固件版本/地区或症状变化会被识别为新内容并按新证据重新接线，
        历史合并记录保留，旧簇编号仍可追溯。
        """
        rows = sorted(records, key=lambda record: record.record_id)
        if not rows:
            return self.clusters()
        with self.repository.transaction() as connection:
            atoms = self._load_atoms(connection)
            links = self._load_links(connection)
            now = utc_now()
            current_atom: dict[str, str] = {}
            for record in rows:
                content = atom_content(record)
                content_hash = stable_hash(content)
                atom_id = atom_id_for(record)
                existing = atoms.get(record.record_id)
                if existing is None:
                    connection.execute(
                        "INSERT INTO cluster_atoms VALUES(?,?,?,?,?,?)",
                        (atom_id, record.record_id, content_hash,
                         content["firmware_version"], content["country"], now))
                    atoms[record.record_id] = {"atom_id": atom_id, "record_id": record.record_id,
                                               "content_hash": content_hash}
                elif existing["atom_id"] != atom_id or existing["content_hash"] != content_hash:
                    old_atom = existing["atom_id"]
                    stale = {edge for edge in links if old_atom in edge}
                    for edge in stale:
                        connection.execute("DELETE FROM cluster_links WHERE left_atom_id=? AND right_atom_id=?", edge)
                    links -= stale
                    connection.execute(
                        "UPDATE cluster_atoms SET atom_id=?,content_hash=?,firmware_version=?,country=?,created_at=? "
                        "WHERE record_id=?",
                        (atom_id, content_hash, content["firmware_version"], content["country"], now, record.record_id))
                    atoms[record.record_id] = {"atom_id": atom_id, "record_id": record.record_id,
                                               "content_hash": content_hash}
                current_atom[record.record_id] = atom_id

            # 先按已持久化的边重建合并关系，再逐条加入本批证据产生的相似边。
            finder = self._finder_with((entry["atom_id"] for entry in atoms.values()), links)
            new_links: set[tuple[str, str]] = set()
            bridges: dict[tuple[str, str], tuple[str, ...]] = {}
            for index, left in enumerate(rows):
                for right in rows[index + 1:]:
                    if similarity(left, right) >= self.threshold:
                        edge = tuple(sorted((current_atom[left.record_id], current_atom[right.record_id])))
                        new_links.add(edge)
                        bridges[edge] = tuple(sorted(set(bridges.get(edge, ())) | {left.record_id, right.record_id}))
            for edge in sorted(new_links):
                if edge not in links:
                    connection.execute("INSERT OR IGNORE INTO cluster_links VALUES(?,?)", edge)
                    links.add(edge)
                left_root, right_root = finder.find(edge[0]), finder.find(edge[1])
                finder.union(edge[0], edge[1])
                if left_root != right_root:
                    # 两个原本独立的问题簇被当前证据连接：记录合并谱系，存活簇取最小原子编号，
                    # 被吸收簇编号永久保留，历史评议引用仍可解析到合并后的簇。
                    surviving, absorbed_cluster = sorted((left_root, right_root))
                    merge_id = stable_hash([surviving, absorbed_cluster, list(bridges[edge])])[:24]
                    if not connection.execute("SELECT 1 FROM cluster_merges WHERE merge_id=?", (merge_id,)).fetchone():
                        connection.execute(
                            "INSERT INTO cluster_merges VALUES(?,?,?,?,?)",
                            (merge_id, surviving, absorbed_cluster,
                             canonical_json(list(bridges[edge])), now))
        return self.clusters()

    def _components(self, connection) -> list[list[dict[str, Any]]]:
        atoms = self._load_atoms(connection)
        finder = self._finder_with((entry["atom_id"] for entry in atoms.values()),
                                   self._load_links(connection))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in atoms.values():
            grouped.setdefault(finder.find(entry["atom_id"]), []).append(entry)
        return list(grouped.values())

    def clusters(self) -> list[dict[str, Any]]:
        with self.repository.connect() as connection:
            components = self._components(connection)
            record_ids = sorted({entry["record_id"] for entries in components for entry in entries})
        records_by_id = {record.record_id: record for record in
                         (self.repository.get_record(rid) for rid in record_ids)
                         if record is not None}
        views: list[dict[str, Any]] = []
        for entries in components:
            members = sorted(entry["record_id"] for entry in entries)
            records = [records_by_id[rid] for rid in members if rid in records_by_id]
            if not records:
                continue
            view = cluster_identity(records)
            # 持久簇编号取并查集根原子（最小原子标识）：合并后存活簇编号不变，被吸收簇仍可解析。
            view["cluster_id"] = min(entry["atom_id"] for entry in entries)
            views.append(view)
        return sorted(views, key=lambda view: view["cluster_id"])

    def cluster_for_record(self, record_id: str) -> dict[str, Any] | None:
        for view in self.clusters():
            if record_id in view["members"]:
                return view
        return None

    def lineage(self, cluster_id: str) -> tuple[str, ...]:
        """返回被并入该簇的全部历史簇编号（按合并记录稳定排序）。"""
        with self.repository.connect() as connection:
            rows = connection.execute(
                "SELECT absorbed_cluster_id FROM cluster_merges WHERE surviving_cluster_id=? "
                "ORDER BY created_at, merge_id", (cluster_id,)).fetchall()
        return tuple(row["absorbed_cluster_id"] for row in rows)

    def resolve_cluster(self, cluster_id: str) -> dict[str, Any]:
        """把任意当前或历史簇编号解析为当前簇，并给出完整合并谱系。"""
        current = {view["cluster_id"]: view for view in self.clusters()}
        if cluster_id in current:
            view = dict(current[cluster_id])
            view["lineage"] = self.lineage(cluster_id)
            return view
        with self.repository.connect() as connection:
            row = connection.execute(
                "SELECT surviving_cluster_id FROM cluster_merges WHERE absorbed_cluster_id=? "
                "ORDER BY created_at DESC, merge_id LIMIT 1", (cluster_id,)).fetchone()
        if row is None:
            raise ContractError("问题簇编号不存在")
        resolved = self.resolve_cluster(row["surviving_cluster_id"])
        lineage = (cluster_id,) + tuple(cid for cid in resolved["lineage"] if cid != cluster_id)
        resolved["lineage"] = lineage
        return resolved

    def merges(self) -> list[dict[str, Any]]:
        with self.repository.connect() as connection:
            rows = connection.execute(
                "SELECT merge_id,surviving_cluster_id,absorbed_cluster_id,bridge_record_ids_json,created_at "
                "FROM cluster_merges ORDER BY created_at, merge_id").fetchall()
        return [dict(row) | {"bridge_record_ids": json.loads(row["bridge_record_ids_json"])} for row in rows]
