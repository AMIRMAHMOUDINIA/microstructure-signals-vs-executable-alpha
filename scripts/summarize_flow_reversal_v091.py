from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def f(x, d=4):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,default=Path("results/flow_reversal_v091"))
    p.add_argument("--output",type=Path,default=Path("results/flow_reversal_v091/FLOW_REVERSAL_SUMMARY.md"))
    a=p.parse_args()

    qa=json.loads((a.input_dir/"qa.json").read_text())
    gate=json.loads((a.input_dir/"gate.json").read_text())
    uni=pd.read_csv(a.input_dir/"pre_final_tfi_regression.csv")
    part=pd.read_csv(a.input_dir/"pre_final_partial_regression.csv")
    auc=pd.read_csv(a.input_dir/"pre_final_auc.csv")
    _mon=pd.read_csv(a.input_dir/"pre_final_monthly.csv")

    lines=[
        "# Study III v0.9.1 — 2025 Trade-Flow Reversal Replication","",
        "## Data QA","",
        f"- Observed minutes: **{qa['observed_minutes']:,}**",
        f"- Completeness: **{f(100*qa['completeness'],3)}%**","",
        "## Aggregate TFI_5m → future 10m close return","",
        "| Period | N | Beta (bps/unit) | HAC p | Spearman rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for _,r in uni.iterrows():
        lines.append(f"| {r['period']} | {int(r['n']):,} | {f(r['beta_bps_per_unit'])} | {f(r['hac_p'],5)} | {f(r['spearman_rho'],5)} |")

    lines += ["","## Incremental control for past 5m return","",
              "| Period | TFI beta (bps/unit) | HAC p | Past-return beta |",
              "|---|---:|---:|---:|"]
    for _,r in part.iterrows():
        lines.append(f"| {r['period']} | {f(r['tfi_beta_bps_per_unit'])} | {f(r['tfi_hac_p'],5)} | {f(r['past_ret_beta'])} |")

    lines += ["","## Parameter-free directional ranking","",
              "| Period | -TFI5 AUC | -PastRet5 AUC | ΔAUC | Bootstrap 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for _,r in auc.iterrows():
        lines.append(
            f"| {r['period']} | {f(r['flow_reversal_auc'])} | {f(r['price_reversal_auc'])} | "
            f"{f(r['delta_auc_flow_vs_price_reversal'])} | "
            f"[{f(r['delta_auc_ci_low_95'])}, {f(r['delta_auc_ci_high_95'])}] |"
        )

    lines += ["","## Gate","",f"**{'PASS' if gate['gate_pass'] else 'FAIL'}**",""]
    for k,v in gate["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")

    fp=a.input_dir/"final_oos_auc.csv"
    if gate["gate_pass"] and fp.exists() and fp.stat().st_size>1:
        r=pd.read_csv(fp).iloc[0]
        lines += ["","## Jul–Dec 2025 final OOS","",
                  f"- Flow-reversal AUC: **{f(r['flow_reversal_auc'])}**",
                  f"- Price-reversal AUC: **{f(r['price_reversal_auc'])}**",
                  f"- ΔAUC: **{f(r['delta_auc_flow_vs_price_reversal'])}**",
                  f"- Paired day-bootstrap 95% CI: **[{f(r['delta_auc_ci_low_95'])}, {f(r['delta_auc_ci_high_95'])}]**"]
    else:
        lines += ["","**Jul–Dec 2025 final OOS was not opened.**"]

    a.output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__": main()
