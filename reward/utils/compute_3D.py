#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cadquery as cq
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import time

# ---------------------- 基础工具 ----------------------
def _resample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """将输入点云重采样到固定点数。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise RuntimeError("Point cloud is empty or has invalid shape")

    if len(pts) == num_points:
        return pts

    replace = len(pts) < num_points
    indices = np.random.choice(len(pts), size=num_points, replace=replace)
    return pts[indices]


def step_to_stl(step_path: str, stl_path: str) -> None:
    """把 STEP 用 CadQuery 导出为 STL（三角网）"""
    shape = cq.importers.importStep(step_path)  # 可能是 Compound
    cq.exporters.export(shape, stl_path)


def load_mesh_as_points(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    """读取三角网，并进行表面均匀采样为点云坐标（N, 3）"""
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(points, dtype=np.float64)


def load_npy_as_points(npy_path: str, num_points: int = 2048) -> np.ndarray:
    """读取 .npy 点云，并重采样到固定点数。"""
    pts = np.load(npy_path)
    return _resample_points(pts, num_points)


def sample_and_normalize_from_step(step_path: str, num_points: int = 2048) -> np.ndarray:
    """STEP → STL → 点云，并做中心化+单位球缩放（独立归一化）"""
    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tmp.stl")
        step_to_stl(step_path, stl_path)
        pts = load_mesh_as_points(stl_path, num_points=num_points)

    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts

def sample_from_step(step_path: str, num_points: int = 2048) -> np.ndarray:
    """STEP → STL → 点云（不做 normalization）"""
    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tmp.stl")
        step_to_stl(step_path, stl_path)
        pts = load_mesh_as_points(stl_path, num_points=num_points)

    return pts


def sample_points_from_path(path: str, num_points: int = 2048) -> np.ndarray:
    """按文件类型读取点云；支持 STEP/STP/NPY。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".step", ".stp"}:
        return sample_and_normalize_from_step(path, num_points=num_points)
        # return sample_from_step(path, num_points=num_points)
    if ext == ".npy":
        print(f"读取点云：{path}")
        return load_npy_as_points(path, num_points=num_points)
    raise RuntimeError(f"Unsupported 3D input format: {path}")


def _rotation_matrix_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """按 XYZ 欧拉角（单位：度）生成旋转矩阵，等价于 Rx @ Ry @ Rz。"""
    xr, yr, zr = np.radians([x_deg, y_deg, z_deg])

    cx, sx = np.cos(xr), np.sin(xr)
    cy, sy = np.cos(yr), np.sin(yr)
    cz, sz = np.cos(zr), np.sin(zr)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return rx @ ry @ rz


def _rotate_points(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return points @ rot.T


def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    t1 = cKDTree(points1)
    t2 = cKDTree(points2)
    d1, _ = t1.query(points2)
    d2, _ = t2.query(points1)
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    t1 = cKDTree(points1)
    t2 = cKDTree(points2)
    d1, _ = t1.query(points2)
    d2, _ = t2.query(points1)
    return float(max(np.max(d1), np.max(d2)))


# ---------------------- 评估（旋转枚举） ----------------------
def compare_step_chamfer_with_rotation_only(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 2048,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """只做欧拉角枚举（X/Y/Z 轴）、不做 ICP。"""
    if save_vis:
        # 统一保留参数以兼容旧调用，但不再执行可视化。
        _ = vis_prefix

    src = sample_points_from_path(step_path_1, num_points=num_points)
    tgt = sample_points_from_path(step_path_2, num_points=num_points)

    best_cd = float("inf")
    best_score = float("inf")
    best_align = None
    best_angles = (0, 0, 0)

    for x in angles:
        for y in angles:
            for z in angles:
                rot = _rotation_matrix_xyz(x, y, z)
                rotated = _rotate_points(src, rot)

                cd = chamfer_distance(rotated, tgt)
                hd = hausdorff_distance(rotated, tgt)
                score = cd + hd

                if score < best_score:
                    best_score = score
                    best_cd = cd
                    best_align = rotated
                    best_angles = (x, y, z)

    best_hd = hausdorff_distance(best_align, tgt)
    return best_cd, best_hd, best_angles

def compare_step_chamfer_no_rotation(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 2048,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis",
    
):
    """不做旋转枚举，直接比较两个 STEP 的 Chamfer / Hausdorff。"""
    if save_vis:
        # 保留参数以兼容旧调用，但不执行可视化
        _ = vis_prefix

    src = sample_points_from_path(step_path_1, num_points=num_points)
    tgt = sample_points_from_path(step_path_2, num_points=num_points)

    cd = chamfer_distance(src, tgt)
    hd = hausdorff_distance(src, tgt)

    return cd, hd, (0, 0, 0)


def compare_step_chamfer_with_icp_rotation(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 2048,
    angles: list[int] = (0, 90, 180, 270),
    max_corr: float = 0.01,
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """兼容旧接口：当前实现已去除 open3d/ICP，退化为纯旋转枚举。"""
    _ = max_corr, save_vis, vis_prefix
    return compare_step_chamfer_with_rotation_only(
        step_path_1=step_path_1,
        step_path_2=step_path_2,
        num_points=num_points,
        angles=angles,
        save_vis=False,
    )


@dataclass
class MetricsResult:
    cd: Optional[float]
    hd: Optional[float]
    best_euler_angle: Optional[tuple[int, int, int]]
    ok: bool
    reason: str = ""


def get_cd_hd(
    pred_step_path: str,
    gt_step_path: str,
    num_points: int = 2048,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis",
):
    """计算 CD/HD，去除 open3d 依赖，默认禁用可视化。"""
    cd, hd, best_angles = compare_step_chamfer_no_rotation(
        step_path_1=pred_step_path,
        step_path_2=gt_step_path,
        num_points=num_points,
        angles=angles,
        save_vis=save_vis,
        vis_prefix=vis_prefix,
    )
    return MetricsResult(cd=cd, hd=hd, best_euler_angle=best_angles, ok=True)

if __name__ == "__main__":
    import time
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # 开始前内存
    mem_before = process.memory_info().rss / 1024**2  # MB
    STEP_A = "/data/baiyixue/CAD/op_oriented_step/02815_index_14/13_14_15_16/next_model.step"
    STEP_B = "/data/baiyixue/CAD/op_oriented_step_pc_normalized/02815_index_14/13_14_15_16/next_model.npy"
    # STEP_B = "/data/baiyixue/CAD/op_oriented_step/02815_index_14/13_14_15_16/next_model.step"

    NUM_POINTS = 2048
    ANGLES = [0, 90, 180, 270]

    t0 = time.time()
    # cd, hd, ang = compare_step_chamfer_with_rotation_only(
    #     STEP_A, STEP_B, num_points=NUM_POINTS, angles=ANGLES
    # )
    cd, hd, ang = compare_step_chamfer_no_rotation(
        STEP_A, STEP_B, num_points=NUM_POINTS, angles=ANGLES
    )
    t1 = time.time()

    # 结束后内存
    mem_after = process.memory_info().rss / 1024**2  # MB

    print(f"Chamfer Distance: {cd:.6f}")
    print(f"Hausdorff Distance: {hd:.6f}")
    print(f"Elapsed: {t1 - t0:.3f}s")

    print(f"[RAM] before: {mem_before:.2f} MB")
    print(f"[RAM] after : {mem_after:.2f} MB")
    print(f"[RAM] delta : {mem_after - mem_before:.2f} MB")
