"""发布评议冻结现场报告及问题归并依据，送达前检查报告版本。"""
import json
from field_feedback.observations.report import ContractError,canonical_json,stable_hash
from field_feedback.triage.multilingual_clusters import clusters

class EvidenceReview:
    def __init__(self,repository,fleet):
        self.repository=repository;self.fleet=fleet
        with repository.connect() as c:c.execute("CREATE TABLE IF NOT EXISTS release_reviews(review_id TEXT PRIMARY KEY,evidence_hash TEXT NOT NULL,body_json TEXT NOT NULL)")

    def capture(self,review_id,report_ids):
        if not review_id or not report_ids or len(set(report_ids))!=len(report_ids):raise ContractError("评议与报告范围无效")
        rows=[self.repository.get_record(i) for i in report_ids]
        if any(r is None for r in rows):raise ContractError("报告不存在")
        groups=clusters(rows)
        body={'reports':{r.record_id:r.fingerprint for r in rows},'groups':[
            {'cluster_id':stable_hash(sorted(r.record_id for r in group)), 'members':sorted(r.record_id for r in group)} for group in groups]}
        h=stable_hash(body)
        with self.repository.transaction() as c:
            old=c.execute("SELECT * FROM release_reviews WHERE review_id=?",(review_id,)).fetchone()
            if old:
                if old['evidence_hash']!=h:raise ContractError("评议依据不能覆盖")
                return json.loads(old['body_json'])
            c.execute("INSERT INTO release_reviews VALUES(?,?,?)",(review_id,h,canonical_json(body)))
        return body

    def deliver(self,review_id,record_id,decision):
        with self.repository.connect() as c:row=c.execute("SELECT body_json FROM release_reviews WHERE review_id=?",(review_id,)).fetchone()
        if row is None:raise ContractError("评议不存在")
        body=json.loads(row[0])
        if record_id not in body['reports']:raise ContractError("评议未引用当前报告")
        for rid,h in body['reports'].items():
            current=self.repository.get_record(rid)
            if current is None or current.fingerprint!=h:raise ContractError("评议报告已改变")
        self.fleet.reconcile(self.repository.get_record(record_id),decision)
        return body
