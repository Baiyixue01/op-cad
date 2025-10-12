import os
import numpy as np
from tqdm import tqdm
import json
from utils.python_run import execute_script
from utils.chamfdist_hausdist import compare_mesh_chamfer_with_rotation_only

def evaluate_all_models(code_dir, gt_dir, output_dir, num_points=8192, angles=[0, 90, 180, 270]):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    total_valid_gt = 0
    invalid_pred_count = 0

    code_files = sorted([f for f in os.listdir(code_dir) if f.endswith(".py")])

    for fname in tqdm(code_files, desc="Evaluating models"):
        idx = os.path.splitext(fname)[0]
        pred_code_path = os.path.join(code_dir, fname)
        gt_code_path = os.path.join(gt_dir, f"{idx}.py")

        # 1. 执行 ground truth 代码
        try:
            gt_stl_result = execute_script(gt_code_path)
        except Exception as e:
            print(f"[SKIP] GT execution failed for {idx}: {e}")
            continue  # Ground truth 失败，跳过该样本

        total_valid_gt += 1  # 计入有效GT

        # 2. 执行预测代码
        try:
            pred_stl_result = execute_script(pred_code_path)
        except Exception as e:
            print(f"[INVALID] Prediction execution failed for {idx}: {e}")
            invalid_pred_count += 1
            continue  # 预测失败，算无效

        # 3. 对比 chamfer + hausdorff（旋转枚举）
        cd, hd = compare_mesh_chamfer_with_rotation_only(
            pred_stl_result, gt_stl_result, num_points=num_points, angles=angles
        )

        # 4. 记录结果
        results.append({"id": idx, "cd": cd, "hd": hd})

    # 5. 保存总表
    result_path = os.path.join(output_dir, "evaluation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    # 6. 统计
    valid_results = results
    all_cd = np.array([r["cd"] for r in valid_results])
    all_hd = np.array([r["hd"] for r in valid_results])

    print("\n🔎 Evaluation Summary:")
    print(f"Valid GT Count: {total_valid_gt}")
    print(f"Invalid Pred Count: {invalid_pred_count}")
    print(f"Invalid Rate: {invalid_pred_count / total_valid_gt * 100:.2f}%")

    if len(valid_results) > 0:
        print(f"Average Chamfer: {np.mean(all_cd):.6f}")
        print(f"Average Hausdorff: {np.mean(all_hd):.6f}")

        print("\n📉 Lowest CD:")
        for r in sorted(valid_results, key=lambda x: x["cd"])[:10]:
            print(f"  {r['id']}: {r['cd']:.6f}")

        print("\n📈 Highest CD:")
        for r in sorted(valid_results, key=lambda x: x["cd"], reverse=True)[:10]:
            print(f"  {r['id']}: {r['cd']:.6f}")

        print("\n📉 Lowest HD:")
        for r in sorted(valid_results, key=lambda x: x["hd"])[:10]:
            print(f"  {r['id']}: {r['hd']:.6f}")

        print("\n📈 Highest HD:")
        for r in sorted(valid_results, key=lambda x: x["hd"], reverse=True)[:10]:
            print(f"  {r['id']}: {r['hd']:.6f}")

        # 7. 保存统计摘要
        summary = {
            "valid_gt_count": total_valid_gt,
            "invalid_pred_count": invalid_pred_count,
            "invalid_rate_percent": round(invalid_pred_count / total_valid_gt * 100, 2),
            "average_cd": float(np.mean(all_cd)),
            "average_hd": float(np.mean(all_hd)),
            "lowest_cd": sorted(valid_results, key=lambda x: x["cd"])[:10],
            "highest_cd": sorted(valid_results, key=lambda x: x["cd"], reverse=True)[:10],
            "lowest_hd": sorted(valid_results, key=lambda x: x["hd"])[:10],
            "highest_hd": sorted(valid_results, key=lambda x: x["hd"], reverse=True)[:10]
        }

        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

