#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import copy
import tempfile
import numpy as np
import open3d as o3d
import cadquery as cq
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from typing import Optional
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord, Camera


# ---------------------- 基础工具 ----------------------
def step_to_stl(step_path: str, stl_path: str) -> None:
    """把 STEP 用 CadQuery 导出为 STL（三角网）"""
    shape = cq.importers.importStep(step_path)  # 可能是 Compound
    cq.exporters.export(shape, stl_path)


def load_mesh_as_pcd(mesh_path: str, num_points: int = 8192) -> o3d.geometry.PointCloud:
    """读取三角网，并用泊松盘采样为点云"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    return mesh.sample_points_poisson_disk(number_of_points=num_points)


def sample_and_normalize_from_step(step_path: str, num_points: int = 8192) -> o3d.geometry.PointCloud:
    """STEP → STL → 点云，并做中心化+单位球缩放（独立归一化）"""
    with tempfile.TemporaryDirectory() as td:
        stl_path = os.path.join(td, "tmp.stl")
        step_to_stl(step_path, stl_path)
        pcd = load_mesh_as_pcd(stl_path, num_points=num_points)

    pts = np.asarray(pcd.points)
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def chamfer_distance(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> float:
    a = np.asarray(pcd1.points)
    b = np.asarray(pcd2.points)
    t1 = cKDTree(a); t2 = cKDTree(b)
    d1, _ = t1.query(b); d2, _ = t2.query(a)
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


def hausdorff_distance(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> float:
    a = np.asarray(pcd1.points)
    b = np.asarray(pcd2.points)
    t1 = cKDTree(a); t2 = cKDTree(b)
    d1, _ = t1.query(b); d2, _ = t2.query(a)
    return float(max(np.max(d1), np.max(d2)))


def compute_per_point_distance(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud) -> np.ndarray:
    tree = cKDTree(np.asarray(tgt.points))
    d, _ = tree.query(np.asarray(src.points))
    return d


def apply_color_by_distance(pcd: o3d.geometry.PointCloud, distances: np.ndarray, vmax: Optional[float] = None):
    if vmax is None:
        vmax = np.percentile(distances, 95)  # 抗极端值
    colors = plt.get_cmap("jet")(np.clip(distances / max(vmax, 1e-12), 0, 1))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_and_save_offscreen(pcd: o3d.geometry.PointCloud, output_path: str, width=512, height=512):
    renderer = OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # 白色背景

    material = MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 3.0  # 点大小

    scene.add_geometry("pcd", pcd, material)

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    diameter = np.linalg.norm(extent) if np.linalg.norm(extent) > 0 else 1.0

    cam_dist = diameter * 1.5
    cam_eye = center + np.array([0, 0, cam_dist])
    cam_up = [0, 1, 0]

    scene.camera.look_at(center, cam_eye, cam_up)
    scene.camera.set_projection(60.0, width / height, 0.1, 100.0, Camera.FovType.Vertical)

    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)


# ---------------------- 评估（旋转枚举 + 可选 ICP） ----------------------
def compare_step_chamfer_with_rotation_only(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis"
):
    """只做欧拉角枚举（X/Y/Z 轴）、不做 ICP"""
    src = sample_and_normalize_from_step(step_path_1, num_points=num_points)
    tgt = sample_and_normalize_from_step(step_path_2, num_points=num_points)

    best_cd = float("inf")
    best_score = float("inf")
    best_align = None
    best_angles = (0, 0, 0)

    for x in angles:
        Rx = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(x), 0, 0))
        for y in angles:
            Ry = o3d.geometry.get_rotation_matrix_from_xyz((0, np.radians(y), 0))
            for z in angles:
                Rz = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(z)))

                rotated = copy.deepcopy(src)
                rotated.rotate(Rx @ Ry @ Rz, center=(0, 0, 0))

                cd = chamfer_distance(rotated, tgt)
                hd = hausdorff_distance(rotated, tgt)
                score = cd + hd

                if score < best_score:
                    best_score = score
                    best_cd = cd
                    best_align = rotated
                    best_angles = (x, y, z)

    best_hd = hausdorff_distance(best_align, tgt)

    if save_vis:
        d1 = compute_per_point_distance(best_align, tgt)
        colored_aligned = apply_color_by_distance(best_align, d1)
        render_and_save_offscreen(colored_aligned, f"{vis_prefix}_aligned_to_target.png")

        d2 = compute_per_point_distance(tgt, best_align)
        colored_tgt = apply_color_by_distance(tgt, d2)
        render_and_save_offscreen(colored_tgt, f"{vis_prefix}_target_to_aligned.png")

    return best_cd, best_hd, best_angles


def compare_step_chamfer_with_icp_rotation(
    step_path_1: str,
    step_path_2: str,
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    max_corr: float = 0.01,
    save_vis: bool = False,
    vis_prefix: str = "vis"
):
    """粗枚举（欧拉角） + ICP 微调（点到点）"""
    src = sample_and_normalize_from_step(step_path_1, num_points=num_points)
    tgt = sample_and_normalize_from_step(step_path_2, num_points=num_points)

    best_cd = float("inf")
    best_align = None
    best_angles = (0, 0, 0)

    for x in angles:
        Rx = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(x), 0, 0))
        for y in angles:
            Ry = o3d.geometry.get_rotation_matrix_from_xyz((0, np.radians(y), 0))
            for z in angles:
                Rz = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(z)))

                rotated = copy.deepcopy(src)
                rotated.rotate(Rx @ Ry @ Rz, center=(0, 0, 0))

                reg = o3d.pipelines.registration.registration_icp(
                    rotated, tgt, max_correspondence_distance=max_corr,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                aligned = copy.deepcopy(rotated).transform(reg.transformation)
                cd = chamfer_distance(aligned, tgt)

                if cd < best_cd:
                    best_cd = cd
                    best_align = aligned
                    best_angles = (x, y, z)

    best_hd = hausdorff_distance(best_align, tgt)

    if save_vis:
        d1 = compute_per_point_distance(best_align, tgt)
        colored_aligned = apply_color_by_distance(best_align, d1)
        render_and_save_offscreen(colored_aligned, f"{vis_prefix}_aligned_to_target.png")

        d2 = compute_per_point_distance(tgt, best_align)
        colored_tgt = apply_color_by_distance(tgt, d2)
        render_and_save_offscreen(colored_tgt, f"{vis_prefix}_target_to_aligned.png")

    return best_cd, best_hd, best_angles

from dataclasses import dataclass
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
    num_points: int = 8192,
    angles: list[int] = (0, 90, 180, 270),
    save_vis: bool = False,
    vis_prefix: str = "vis"
):
    """只做欧拉角枚举（X/Y/Z 轴）、不做 ICP"""
    src = sample_and_normalize_from_step(pred_step_path, num_points=num_points)
    tgt = sample_and_normalize_from_step(gt_step_path, num_points=num_points)

    best_cd = float("inf")
    best_score = float("inf")
    best_align = None
    best_angles = (0, 0, 0)

    for x in angles:
        Rx = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(x), 0, 0))
        for y in angles:
            Ry = o3d.geometry.get_rotation_matrix_from_xyz((0, np.radians(y), 0))
            for z in angles:
                Rz = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(z)))

                rotated = copy.deepcopy(src)
                rotated.rotate(Rx @ Ry @ Rz, center=(0, 0, 0))

                cd = chamfer_distance(rotated, tgt)
                hd = hausdorff_distance(rotated, tgt)
                score = cd + hd

                if score < best_score:
                    best_score = score
                    best_cd = cd
                    best_align = rotated
                    best_angles = (x, y, z)

    best_hd = hausdorff_distance(best_align, tgt)

    if save_vis:
        d1 = compute_per_point_distance(best_align, tgt)
        colored_aligned = apply_color_by_distance(best_align, d1)
        render_and_save_offscreen(colored_aligned, f"{vis_prefix}_aligned_to_target.png")

        d2 = compute_per_point_distance(tgt, best_align)
        colored_tgt = apply_color_by_distance(tgt, d2)
        render_and_save_offscreen(colored_tgt, f"{vis_prefix}_target_to_aligned.png")

    return MetricsResult(cd=best_cd, hd=best_hd, best_euler_angle=best_angles, ok=True)

# ---------------------- 简单测试入口（不使用 argparse） ----------------------
if __name__ == "__main__":
    # ====== 在这里改你的入参 ======
    STEP_A = "/home/baiyixue/project/op-cad/output_a.step"
    STEP_B = "/home/baiyixue/project/op-cad/output_b.step"

    NUM_POINTS = 8192
    ANGLES = [0, 90, 180, 270]
    USE_ICP = True           # False 则只做旋转枚举
    SAVE_VIS = True         # True 则会输出两张误差热图（需要可用的离屏渲染环境）
    VIS_PREFIX = "/home/baiyixue/project/op-cad/inference/inference_results/test"

    if SAVE_VIS:
        os.makedirs(os.path.dirname(VIS_PREFIX), exist_ok=True)

    t0 = time.time()
    if USE_ICP:
        cd, hd, ang = compare_step_chamfer_with_icp_rotation(
            STEP_A, STEP_B, num_points=NUM_POINTS, angles=ANGLES,
            max_corr=0.01, save_vis=SAVE_VIS, vis_prefix=VIS_PREFIX
        )
    else:
        cd, hd, ang = compare_step_chamfer_with_rotation_only(
            STEP_A, STEP_B, num_points=NUM_POINTS, angles=ANGLES,
            save_vis=SAVE_VIS, vis_prefix=VIS_PREFIX
        )
    t1 = time.time()

    print(f"Best Euler angles (deg): {ang}")
    print(f"Chamfer Distance: {cd:.6f}")
    print(f"Hausdorff Distance: {hd:.6f}")
    print(f"Elapsed: {t1 - t0:.3f}s")


# path = "/home/baiyixue/project/op-cad/test/inference/gemini-2.5-pro/std/code_step/00002_index_2/step0/full_path/k1_full.step"

# import cadquery as cq

# def all_edges_from_step(step_path):
#     obj = cq.importers.importStep(step_path)              # 常见返回 Workplane
#     wp  = obj if isinstance(obj, cq.Workplane) else cq.Workplane(obj=obj)

#     # 先探测层级（调试用）
#     print("solids:", wp.solids().size(),
#           "shells:", wp.shells().size(),
#           "faces:",  wp.faces().size(),
#           "edges:",  wp.edges().size())  # Workplane 栈意义上的数量
#     edge_list = wp.edges().vals()
#     print(len(edge_list))
#     edge_length = edge_list[2].Length()
#     print( edge_list[2].Length(), edge_list[2].Center(), edge_list[2].geomType(), edge_list[2].Vertices())
#     for i in edge_list[2].Vertices():
#         print(i.toTuple())
#     # edge = wp.edges().all()[0].Length()
#     # print(edge)
#     # return edge
# # 示例：打印几何信息
# edge_1 = all_edges_from_step(path)
# # edge_2 = all_edges_from_step(path)
# from cadquery import Plane, Vector, Workplane

# # --- Initial Setup (Following plane_usage_rules) ---
# origin_0 = Vector(0.0, 0.0, 0.0)
# normal_0 = Vector(0.0, 0.0, 1.0)
# x_dir_0 = Vector(1.0, 0.0, 0.0)
# plane_0 = Plane(origin=origin_0, normal=normal_0, xDir=x_dir_0)

# # This is our first result
# result_0 = Workplane(plane_0).box(20.0, 20.0, 10.0)

# # --- Fillet Operation (Following modify_rules) ---
# # Step 1: Select edges from result_0 and assign to edges_1
# edges_1 = result_0.edges("|Z")

# # Step 2: Apply operation on result_0, passing in edges_1, to create result_1
# fillet_radius = 2.0
# result_1 = edges_1.fillet(fillet_radius)

# # --- Chamfer Operation (Continuing the sequence) ---
# # Step 1: Select edges from result_1 and assign to edges_2
# edges_2 = result_1.edges("<Z")

# # Step 2: Apply operation on result_1, passing in edges_2, to create result_2
# chamfer_distance = 1.0
# result_2 = edges_2.chamfer(chamfer_distance)
# cq.exporters.export(result_2.val(),"/home/baiyixue/project/op-cad/test/test.stl")


