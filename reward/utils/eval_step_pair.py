# tools/eval_step_pair.py
# 依赖: utils.compute_3D.compute_metrics / MetricsResult / IMPROVED_ALIGN_DEFAULT / ICP_REFINE_DEFAULT / NUM_POINTS_DEFAULT

from typing import Dict, Any
import json
import sys

# 若本文件与 utils 不在同级，可按需调整 sys.path
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from compute_3D import (
    compute_metrics,
    MetricsResult,
    IMPROVED_ALIGN_DEFAULT,
    ICP_REFINE_DEFAULT,
    NUM_POINTS_DEFAULT,
)

def eval_step_pair(
    gt_step_path: str,
    pred_step_path: str,
    improved_align: bool = IMPROVED_ALIGN_DEFAULT,
    icp_refine: bool = ICP_REFINE_DEFAULT,
    num_points: int = NUM_POINTS_DEFAULT,
) -> Dict[str, Any]:
    """
    输入两个 STEP 文件路径，计算 3D 指标。
    返回字典: {ok, reason, iou, iogt, cd, hd, gt_step_path, pred_step_path}
    """
    res: MetricsResult = compute_metrics(
        gt_step_path=gt_step_path,
        pred_step_path=pred_step_path,
        improved_align=improved_align,
        icp_refine=icp_refine,
        num_points=num_points,
    )
    return {
        "ok": bool(res.ok),
        "reason": getattr(res, "reason", ""),
        "iou": getattr(res, "iou", None),
        "iogt": getattr(res, "iogt", None),
        "cd": getattr(res, "cd", None),
        "hd": getattr(res, "hd", None),
        "gt_step_path": gt_step_path,
        "pred_step_path": pred_step_path,
        "improved_align": improved_align,
        "icp_refine": icp_refine,
        "num_points": num_points,
    }

def main():
    """
    命令行用法:
        python tools/eval_step_pair.py <GT_STEP> <PRED_STEP> [--no-improved] [--no-icp] [--points 200000]
    输出 JSON 到标准输出。
    """
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two STEP files and output 3D metrics as JSON.")
    parser.add_argument("gt_step", type=str, help="Path to ground-truth STEP file")
    parser.add_argument("pred_step", type=str, help="Path to predicted STEP file")
    parser.add_argument("--no-improved", action="store_true", help="Disable improved alignment")
    parser.add_argument("--no-icp", action="store_true", help="Disable ICP refine")
    parser.add_argument("--points", type=int, default=NUM_POINTS_DEFAULT, help="Number of sampled points")
    args = parser.parse_args()

    out = eval_step_pair(
        gt_step_path=args.gt_step,
        pred_step_path=args.pred_step,
        improved_align=(not args.no_improved),
        icp_refine=(not args.no_icp),
        num_points=args.points,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
