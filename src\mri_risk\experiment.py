import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss

def dataset(n=1400,seed=27):
 r=np.random.default_rng(seed);base=r.normal(size=(n,6));delta=r.normal(size=(n,6));logit=.4*base[:,0]-.3*base[:,2]+1.2*delta[:,0]-.9*delta[:,1]+.7*delta[:,4]-1;prob=1/(1+np.exp(-logit));y=r.binomial(1,prob);return base,delta,y
def evaluate(seed=27):
 b,d,y=dataset(seed=seed);cut=950;out={}
 for name,x in {"single_visit":b,"longitudinal":np.c_[b,d]}.items():
  m=LogisticRegression(max_iter=500).fit(x[:cut],y[:cut]);p=m.predict_proba(x[cut:])[:,1];out[name]={"auroc":float(roc_auc_score(y[cut:],p)),"brier_score":float(brier_score_loss(y[cut:],p))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
