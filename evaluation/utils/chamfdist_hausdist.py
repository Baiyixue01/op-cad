import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import copy
import time
import matplotlib.pyplot as plt
from open3d.visualization import rendering as o3dr

def sample_and_normalize(mesh_path, num_points=8192):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    points = np.asarray(pcd.points)

    # Normalize: center + scale
    centroid = np.mean(points, axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points /= scale

    # 更新点云对象中的点
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd  # ✅ 返回 Open3D 点云对象

def hausdorff_distance(pcd1, pcd2):
    """
    计算两个点云之间的双向 Hausdorff 距离（最大最小距离）
    """
    tree1 = cKDTree(np.asarray(pcd1.points))
    tree2 = cKDTree(np.asarray(pcd2.points))

    d1, _ = tree1.query(np.asarray(pcd2.points))
    d2, _ = tree2.query(np.asarray(pcd1.points))

    hd = max(np.max(d1), np.max(d2))
    return hd
def chamfer_distance(pcd1, pcd2):
    tree1 = cKDTree(np.asarray(pcd1.points))
    tree2 = cKDTree(np.asarray(pcd2.points))
    d1, _ = tree1.query(np.asarray(pcd2.points))
    d2, _ = tree2.query(np.asarray(pcd1.points))
    return np.mean(d1**2) + np.mean(d2**2)

def compare_mesh_chamfer(mesh_path_1, mesh_path_2, num_points=8192):
    """
    对比两个 mesh 文件的几何相似度（Chamfer Distance）

    参数:
        mesh_path_1: 第一个 mesh 的文件路径
        mesh_path_2: 第二个 mesh 的文件路径
        num_points: 采样的点数（默认 8192）

    返回:
        Chamfer Distance 值（float）
    """
    pcd1 = sample_and_normalize(mesh_path_1, num_points)
    pcd2 = sample_and_normalize(mesh_path_2, num_points)
    cd = chamfer_distance(pcd1, pcd2)
    return cd
def compare_mesh_hausdorff(mesh_path_1, mesh_path_2, num_points=8192):
    """
    对比两个 mesh 的 Hausdorff Distance

    参数:
        mesh_path_1: 第一个 mesh 的路径
        mesh_path_2: 第二个 mesh 的路径
        num_points: 每个 mesh 采样点数（默认8192）

    返回:
        Hausdorff Distance（float）
    """
    pcd1 = sample_and_normalize(mesh_path_1, num_points)
    pcd2 = sample_and_normalize(mesh_path_2, num_points)
    hd = hausdorff_distance(pcd1, pcd2)
    return hd
def render_and_save_offscreen(pcd, output_path="/home/baiyixue/VLM_data_annotation/stl_test/heatmap.png", width=512, height=512):
    renderer = o3dr.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # 白色背景

    material =o3dr.MaterialRecord()
    material.shader = "defaultUnlit"

    scene.add_geometry("pcd", pcd, material)

    # 设置摄像机参数
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    diameter = np.linalg.norm(extent)

    camera_distance = diameter * 1.5  # 让相机离得足够远
    camera_eye = center + np.array([0, 0, camera_distance])
    camera_up = [0, 1, 0]

    scene.camera.look_at(center, camera_eye, camera_up)
    scene.camera.set_projection(60.0, width / height, 0.1, 100.0, o3dr.Camera.FovType.Vertical)

    # 渲染并保存
    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)
    print(f"[保存完成] {output_path}")

def compare_mesh_chamfer_with_icp_rotation(mesh_path_1, mesh_path_2, num_points=8192, angles=[0, 90, 180, 270]):
    source = sample_and_normalize(mesh_path_1, num_points)
    target = sample_and_normalize(mesh_path_2, num_points)

    best_cd = float("inf")
    best_align = None

    for angle in angles:
        rotated = copy.deepcopy(source)
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(angle)))
        rotated.rotate(R, center=(0, 0, 0))

        result_icp = o3d.pipelines.registration.registration_icp(
            rotated, target, max_correspondence_distance=0.01,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        aligned = copy.deepcopy(rotated).transform(result_icp.transformation)
        cd = chamfer_distance(aligned, target)

        if cd < best_cd:
            best_cd = cd
            best_align = aligned

    hd = hausdorff_distance(best_align, target)

    # 可视化误差热图：aligned → target
    distances_1 = compute_per_point_distance(best_align, target)
    colored_aligned = apply_color_by_distance(best_align, distances_1)
    render_and_save_offscreen(colored_aligned, output_path="/home/baiyixue/VLM_data_annotation/stl_test/aligned_to_target.png")

    # 可视化误差热图：target → aligned
    distances_2 = compute_per_point_distance(target, best_align)
    colored_target = apply_color_by_distance(target, distances_2)
    render_and_save_offscreen(colored_target, output_path="/home/baiyixue/VLM_data_annotation/stl_test/target_to_aligned.png")

    return best_cd, hd
def compare_mesh_chamfer_with_rotation_only(mesh_path_1, mesh_path_2, num_points=8192, angles=[0, 90, 180, 270]):
    source = sample_and_normalize(mesh_path_1, num_points)
    target = sample_and_normalize(mesh_path_2, num_points)

    best_cd = float("inf")
    best_align = None
    best_score = 999999
    # 枚举所有旋转组合：X, Y, Z 三轴分别取 angles 中的角度
    for x in angles:
        for y in angles:
            for z in angles:
                rotated = copy.deepcopy(source)
                R = o3d.geometry.get_rotation_matrix_from_xyz((
                    np.radians(x),
                    np.radians(y),
                    np.radians(z)
                ))
                rotated.rotate(R, center=(0, 0, 0))

                cd = chamfer_distance(rotated, target)
                hd = hausdorff_distance(rotated, target)
                score = cd+hd
                if score < best_score:
                    best_cd = cd
                    best_align = rotated
                    best_score = score
                    best_angle = angles

    # Hausdorff distance 也在最优姿态上计算
    hd = hausdorff_distance(best_align, target)
    print(best_angle)
    # 可视化误差热图：aligned → target
    distances_1 = compute_per_point_distance(best_align, target)
    colored_aligned = apply_color_by_distance(best_align, distances_1)
    render_and_save_offscreen(colored_aligned, output_path="/home/baiyixue/VLM_data_annotation/stl_test/aligned_to_target.png")

    # 可视化误差热图：target → aligned
    distances_2 = compute_per_point_distance(target, best_align)
    colored_target = apply_color_by_distance(target, distances_2)
    render_and_save_offscreen(colored_target, output_path="/home/baiyixue/VLM_data_annotation/stl_test/target_to_aligned.png")
    print()
    return best_cd, hd
def compute_per_point_distance(source_pcd, target_pcd):
    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)
    tree = cKDTree(target_pts)
    distances, _ = tree.query(source_pts)
    return distances

def apply_color_by_distance(pcd, distances, vmax=None):
    if vmax is None:
        vmax = np.percentile(distances, 95)  # 避免极端值影响可视化

    colors = plt.get_cmap('jet')(np.clip(distances / vmax, 0, 1))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

start_time = time.time()
# 使用示例
path_A="/home/baiyixue/VLM_data_annotation/stl_test/00023_index_2.stl"
path_B="/home/baiyixue/VLM_data_annotation/stl_test/model.stl"
cd, hd = compare_mesh_chamfer_with_rotation_only(path_A, path_B)
end_time = time.time()
print(f"执行时间: {end_time - start_time:.3f} 秒")
print(f"Chamfer Distance between A and B: {cd:.6f}")
print(f"Hausdorff Distance between A and B: {hd:.6f}")
