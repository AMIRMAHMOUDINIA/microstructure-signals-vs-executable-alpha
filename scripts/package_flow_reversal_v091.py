from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

FILES=[
"results/flow_reversal_v091/qa.json",
"results/flow_reversal_v091/pre_final_tfi_regression.csv",
"results/flow_reversal_v091/pre_final_partial_regression.csv",
"results/flow_reversal_v091/pre_final_auc.csv",
"results/flow_reversal_v091/pre_final_monthly.csv",
"results/flow_reversal_v091/gate.json",
"results/flow_reversal_v091/final_oos_tfi_regression.csv",
"results/flow_reversal_v091/final_oos_partial_regression.csv",
"results/flow_reversal_v091/final_oos_auc.csv",
"results/flow_reversal_v091/final_oos_monthly.csv",
"results/flow_reversal_v091/FLOW_REVERSAL_SUMMARY.md",
"FLOW_REVERSAL_PROTOCOL_V091.md",
"V09_DATA_SOURCE_POSTMORTEM.md",
]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("."));ap.add_argument("--output",type=Path,default=Path("microstructure_flow_reversal_results_v091.zip"));a=ap.parse_args()
 items=[(r,a.root/r) for r in FILES if (a.root/r).exists() and (a.root/r).is_file()]
 man={"protocol":"v0.9.1","files":[{"path":r,"bytes":p.stat().st_size,"sha256":sha(p)} for r,p in items]}
 with zipfile.ZipFile(a.output,"w",zipfile.ZIP_DEFLATED) as z:
  for r,p in items:z.write(p,r)
  z.writestr("FLOW_REVERSAL_RESULTS_MANIFEST.json",json.dumps(man,indent=2))
 print(f"Packaged {len(items)} files: {a.output}")
if __name__=="__main__":main()
