from mri_risk.experiment import dataset,evaluate
def test_shape():
 b,d,y=dataset(20,2);assert b.shape==d.shape==(20,6) and len(y)==20
def test_metrics(): assert all(0<=v<=1 for m in evaluate(2).values() for v in m.values())

