from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from field_feedback.observations.report import DomainRecord, bounded, finite_number, normalized_text, stable_hash


SYNONYMS = {"stuck": "blocked", "jammed": "blocked", "卡住": "blocked", "atascado": "blocked",
            "slow": "latency", "延迟": "latency", "lambat": "latency", "overheat": "thermal", "过热": "thermal"}


def tokens(value: Any) -> tuple[str, ...]:
    raw = re.findall(r"[a-z0-9]+|[一-鿿]+", normalized_text(value))
    return tuple(SYNONYMS.get(item, item) for item in raw if item)


def signature(record: DomainRecord) -> tuple[str, ...]:
    values = [record.payload.get("symptom_code", ""), record.payload.get("weather", ""),
              record.payload.get("terrain", ""), record.payload.get("device_model", "")]
    merged = sorted({token for value in values for token in tokens(value)})
    return tuple(merged)


def similarity(left: DomainRecord, right: DomainRecord) -> float:
    a, b = set(signature(left)), set(signature(right))
    lexical = len(a & b) / max(1, len(a | b))
    same_model = left.payload.get("device_model") == right.payload.get("device_model")
    same_version = left.payload.get("firmware_version") == right.payload.get("firmware_version")
    regional = left.payload.get("country") == right.payload.get("country")
    return bounded(lexical * 0.55 + same_model * 0.20 + same_version * 0.15 + regional * 0.10)


class _UnionFind:
    """确定性并查集：根始终取最小原子标识，保证任何合并顺序得到同一根。"""

    def __init__(self, members: Iterable[str]):
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        root = member
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[member] != root:
            self.parent[member], member = root, self.parent[member]
        return root

    def union(self, left: str, right: str) -> tuple[str, str, str]:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root, right_root, left_root
        root, absorbed = sorted((left_root, right_root))
        self.parent[absorbed] = root
        return root, absorbed, root


def _sorted_records(records: Iterable[DomainRecord]) -> list[DomainRecord]:
    return sorted(records, key=lambda record: record.record_id)


def clusters(records: Iterable[DomainRecord], threshold: float = 0.45) -> list[list[DomainRecord]]:
    """按相似边求连通分量。

    相似度是记录对的纯函数，先按稳定标识排序再建边，并查集根取最小标识，
    因此同一证据集合无论以何种顺序导入，都得到完全相同的分组与排列；
    多语言症状（同义词归并后）仍可跨越语言、地区与固件版本形成同一问题簇。
    """
    rows = _sorted_records(records)
    if not rows:
        return []
    finder = _UnionFind(record.record_id for record in rows)
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if similarity(left, right) >= threshold:
                finder.union(left.record_id, right.record_id)
    grouped: dict[str, list[DomainRecord]] = defaultdict(list)
    for record in rows:
        grouped[finder.find(record.record_id)].append(record)
    return [grouped[root] for root in sorted(grouped)]


def cluster_identity(group: Iterable[DomainRecord]) -> dict[str, Any]:
    """问题簇的可追溯身份。

    身份包含固件版本与地区集合：归并不会把这些含义抹平，簇可以跨版本/地区聚合症状，
    但每个簇携带的 firmware_versions/countries 始终可核对；content_fingerprint 是身份
    内容的稳定哈希，对同一证据集合在任何导入顺序下都一致，成员变化时随之变化，
    便于识别“同一问题簇的证据集合发生了合并/扩充”。
    """
    rows = _sorted_records(group)
    member_ids = tuple(record.record_id for record in rows)
    symptom_tokens = tuple(sorted({token for record in rows for token in tokens(record.payload.get("symptom_code", ""))}))
    firmware_versions = tuple(sorted({str(record.payload.get("firmware_version", "")) for record in rows}))
    countries = tuple(sorted({str(record.payload.get("country", "")) for record in rows}))
    device_models = tuple(sorted({str(record.payload.get("device_model", "")) for record in rows}))
    identity = {"members": member_ids, "symptom_tokens": symptom_tokens,
                "firmware_versions": firmware_versions, "countries": countries, "device_models": device_models}
    return {"content_fingerprint": stable_hash(identity)[:24], "members": member_ids,
            "symptom_tokens": symptom_tokens, "firmware_versions": firmware_versions,
            "countries": countries, "device_models": device_models}


def canary_health(records: Iterable[DomainRecord]) -> dict[str, float]:
    rows = list(records)
    affected = sum(max(0.0, finite_number(row.payload.get("affected_units", 0.0))) for row in rows)
    reproduced = sum(bounded(finite_number(row.payload.get("reproduction_rate", 0.0))) for row in rows)
    translation = sum(bounded(finite_number(row.payload.get("translation_confidence", 0.0))) for row in rows)
    severity = sum(bounded(finite_number(row.payload.get("severity", 0.0)) / 10.0) for row in rows)
    count = max(1, len(rows))
    return {"affected": affected, "reproduction": reproduced / count, "translation": translation / count,
            "severity": severity / count, "rollback_risk": bounded((severity + reproduced) / (2.0 * count))}


def regional_distribution(records: Iterable[DomainRecord]) -> dict[str, float]:
    counts = Counter(str(row.payload.get("country", "unknown")) for row in records)
    total = sum(counts.values())
    concentration = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    return {"regions": float(len(counts)), "concentration": concentration, "diversity": bounded(1.0 - concentration)}


def derive_context(record: DomainRecord, repository: Any) -> dict[str, float]:
    rows = repository.list_records() or [record]
    groups = clusters(rows)
    health = canary_health(rows)
    regional = regional_distribution(rows)
    return {"duplicate_density_weight": 0.7 + (1.0 - len(groups) / max(1, len(rows))) * 0.7,
            "reproduction_confidence_weight": 0.6 + health["reproduction"] * 0.5,
            "regional_impact_weight": 0.6 + regional["diversity"] * 0.5,
            "rollback_risk_weight": 0.8 + health["rollback_risk"] * 0.6,
            "translation_gap_weight": 1.2 - health["translation"] * 0.4}
