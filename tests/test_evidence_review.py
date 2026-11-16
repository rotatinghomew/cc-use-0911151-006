import tempfile,unittest
from pathlib import Path
from field_feedback.product_loop import Application
from field_feedback.observations.report import DomainRecord,ContractError

class EvidenceReviewTests(unittest.TestCase):
    def test_capture_is_durable_and_conflicting_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=Application(Path(d)/'db');r=DomainRecord('r',{'symptom_code':'camera','device_model':'m'})
            a.ingest(r);body=a.evidence_review.capture('review',['r'])
            self.assertEqual(body['groups'][0]['members'],['r'])
            self.assertEqual(Application(Path(d)/'db').evidence_review.capture('review',['r']),body)
            a.ingest(DomainRecord('r',dict(r.payload,symptom_code='thermal')))
            with self.assertRaises(ContractError):a.evidence_review.capture('review',['r'])
            with self.assertRaises(ContractError):a.evidence_review.deliver('review','r',None)
