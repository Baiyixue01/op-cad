#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cadquery as cq
import numpy as np
import os
import json
import argparse
from typing import Tuple, Optional
from tqdm import tqdm
from itertools import permutations

# ========== 可选：全局默认开关（命令行也可覆盖） ==========
IMPROVED_ALIGN = False


# ---------- 工具函数 ----------
def generate_24_rotations():
    """
    生成 24 个保持右手系的正交旋转（轴置换×符号，det=+1）。
    """
    I = np.eye(3)
    perms = list(permutations([0, 1, 2], 3))  # 6 种轴置换
    # 偶数负号的4种符号组合，保证 det=+1
    signs = [
        np.array([ 1, 1, 1]),
        np.array([ 1,-1,-1]),
        np.array([-1, 1,-1]),
        np.array([-1,-1, 1]),
    ]
    R_list = []
    for p in perms:
        P = I[:, list(p)]
        for s in signs:
            S = np.diag(s)
            R = P @ S
            if np.linalg.det(R) > 0.0:
                R_list.append(R)
    return R_list


def detect_rotational_symmetry_axis(eigs, tol_ratio: float = 0.03) -> Optional[int]:
    """
    基于特征值差异判断近旋转对称，并返回“疑似对称轴”索引（0/1/2），None 表示非近对称。
    约定：eigs 为升/降序均可，本函数内部做排序。
    """
    l1, l2, l3 = np.sort(eigs)[::-1]  # l1>=l2>=l3
    # λ2≈λ3 ⇒ 圆柱类，轴≈最大特征值方向 v1 → index 0
    if abs(l2 - l3) / max(l2 + l3, 1e-12) < tol_ratio:
        return 0
    # λ1≈λ2 ⇒ 扁平盘/板，轴≈最小特征值方向 v3 → index 2
    if abs(l1 - l2) / max(l1 + l2, 1e-12) < tol_ratio:
        return 2
    return None


def rodrigues_from_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    由单位轴 axis 和角度 theta（弧度）构造旋转矩阵（Rodrigues）。
    """
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


# ---------- IoU 对齐与计算 ----------
def cq_align_shapes(source: cq.Workplane,
                    target: cq.Workplane,
                    improved: bool = False) -> Tuple[Optional[cq.Workplane], float, np.ndarray, np.ndarray]:
    """
    使用质心 + 惯量主轴对齐；可选启用改良版搜索。
    返回：对齐后的 source、IoU、原始 c_source、c_target。
    """
    # 质心
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    # 惯量矩（对角化得到主轴）
    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    # 体积（计算尺度）
    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    # 特征分解：I = V diag(λ) V^T
    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    # 等比尺度（惯量/体积）归一化
    s_source = np.sqrt(np.abs(I_p_source).sum() / max(v_source, 1e-12))
    s_target = np.sqrt(np.abs(I_p_target).sum() / max(v_target, 1e-12))

    normalized_source = source.translate(-c_source).val().scale(1.0 / max(s_source, 1e-12))
    normalized_target = target.translate(-c_target).val().scale(1.0 / max(s_target, 1e-12))

    # 候选旋转
    R_candidates = []
    if not improved:
        # 原始“简配”版本（保持兼容）
        Rs = np.zeros((4, 3, 3))
        Rs[0] = I_v_target @ I_v_source.T
        for i in range(3):
            alignment = 1 - 2 * np.array([i > 0, (i + 1) % 2, i % 3 <= 1])
            Rs[i + 1] = I_v_target @ (alignment[None, :] * I_v_source).T
        R_candidates = [Rs[k] for k in range(4)]
        angle_list = [0]
        sym_axis_idx = None
    else:
        # 改良版：24 旋转 + 近对称轴扫描
        for R24 in generate_24_rotations():
            R_candidates.append(I_v_target @ R24 @ I_v_source.T)
        sym_axis_idx = detect_rotational_symmetry_axis(I_p_target, tol_ratio=0.03)
        angle_list = list(range(0, 360, 15)) if sym_axis_idx is not None else [0]

    best_IOU = 0.0
    best_T = None

    # 目标主轴基（用于构造绕轴旋转）
    Vt = I_v_target

    for R in R_candidates:
        for deg in angle_list:
            R_final = R
            if improved and sym_axis_idx is not None and deg != 0:
                theta = np.deg2rad(deg)
                axis = Vt[:, sym_axis_idx]
                R_axis = rodrigues_from_axis_angle(axis, theta)
                R_final = R_axis @ R

            # 4x4 齐次矩阵
            T = np.eye(4)
            T[:3, :3] = R_final

            # 应用到归一化后的 source
            aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))

            # 布尔体积求 IoU
            try:
                intersect = aligned_source.intersect(normalized_target)
                union = aligned_source.fuse(normalized_target)
                uvol = max(union.Volume(), 1e-12)
                IOU = max(min(intersect.Volume() / uvol, 1.0), 0.0)
            except Exception:
                IOU = 0.0  # 布尔失败时记 0

            if IOU > best_IOU:
                best_IOU = IOU
                best_T = T

    if best_T is not None:
        aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
        return cq.Workplane(aligned_source), best_IOU, c_source, c_target
    else:
        return None, best_IOU, c_source, c_target


# ---------- 其他原有工具 ----------
def find_image_by_question_id(jsonl_path, target_question_id):
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if data.get("question_id") == target_question_id:
                return data.get("image")[:-6]
    return None


def average_non_none(values):
    filtered_values = [v for v in values if v is not None]
    print(f"Number of Nones: {len(values) - len(filtered_values)}")
    return sum(filtered_values) / len(filtered_values) if filtered_values else None


# ---------- 主流程 ----------
def main(model_path, test_set_name, improved_align: bool = False):
    model_name = model_path.split("/")[-1]
    model_generated_steps_dir = f"./inference/inference_results/{model_name}/{test_set_name}/model_step/"
    ground_truth_generated_steps_dir = "./inference/test100_gt_steps/"
    test_jsonl = f"./inference/{test_set_name}.jsonl"

    all_ious = []
    gt_steps = []
    model_steps = []
    model_steps_aligned = []

    files = sorted(os.listdir(model_generated_steps_dir))
    for g in tqdm(files, desc="Computing IoU"):
        if not g.lower().endswith(".step"):
            continue
        question_id = g[:-5]
        orig_id = find_image_by_question_id(test_jsonl, int(question_id))
        if orig_id is None:
            raise ValueError(f"Can't find original ID in test set for {g}")

        gt_step = cq.importers.importStep(os.path.join(ground_truth_generated_steps_dir, orig_id + ".step"))
        model_generated_step = cq.importers.importStep(os.path.join(model_generated_steps_dir, g))
        gt_steps.append(gt_step)
        model_steps.append(model_generated_step)

        try:
            aligned_model_generated, IOU, _, _ = cq_align_shapes(
                model_generated_step, gt_step, improved=improved_align
            )
            model_steps_aligned.append(aligned_model_generated)
            all_ious.append(IOU)
        except Exception as e:
            print(f"[warn] issue on {g}: {e}")
            all_ious.append(None)

    avg_iou = average_non_none(all_ious)
    print(f"Model's average IoU score: {avg_iou}")

    iou_result_file = f"./inference/inference_results/{model_name}/{test_set_name}/cad_iou_results.txt"
    os.makedirs(os.path.dirname(iou_result_file), exist_ok=True)
    with open(iou_result_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test set: {test_set_name}\n")
        f.write(f"Improved Align: {improved_align}\n")
        f.write(f"Average IoU: {avg_iou}\n")
        f.write(f"Number of valid steps: {len([v for v in all_ious if v is not None])} / {len(files)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute model's IoU score.")
    parser.add_argument("--model_path", type=str, required=True, help="Model to compute IoU for.")
    parser.add_argument("--test_set_name", type=str, required=True, help="Name of the test set.")
    parser.add_argument("--improved_align", action="store_true",
                        help="Use improved alignment (PCA + 24 rotations + optional axis sweep).")
    args = parser.parse_args()

    # 同步全局或直接传参都行；这里直接传参
    main(args.model_path, args.test_set_name, improved_align=bool(args.improved_align))
