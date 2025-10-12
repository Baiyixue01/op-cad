import os
import json
import numpy as np
import pyvista as pv
import multiprocessing
from PIL import Image, ImageOps
from pyvista import Light
pv.global_theme.allow_empty_mesh = True
os.sched_setaffinity(0, {i for i in range(os.cpu_count())})
class STLProcessor:
    def __init__(self, stl_dir, json_dir, image_output_dir, log_success, log_failed, angles,slender_log = "slender_stl.txt"):
        self.stl_dir = stl_dir
        self.json_dir = json_dir
        self.image_output_dir = image_output_dir
        self.log_success = log_success
        self.log_failed = log_failed
        self.angles = angles
        self.slender_log =  os.path.join(image_output_dir,slender_log)
        # 确保输出目录存在
        os.makedirs(self.image_output_dir, exist_ok=True)

    def get_file_path(self, stl_path):
        """ 获取 JSON 及输出文件路径 """
        file_name = os.path.basename(stl_path)
        file_prefix = os.path.splitext(file_name)[0]  # "04422_index_3"
        json_path = os.path.join(self.json_dir, f"{file_prefix}.json")
        output_path = os.path.join(self.image_output_dir, f"{file_prefix}.png")
        return file_prefix, json_path, output_path

    def compute_camera(self, json_path, use_slender = False):
        """更健壮的相机计算方式，使用包围球 + 动态缩放"""
        angles = self.angles
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ JSON 文件不存在: {json_path}")
        with open(json_path, "r") as f:
            json_data = json.load(f)
        # 提取 bounding_box 信息
        bounding_box = json_data.get("properties", {}).get("bounding_box", {})
        if bounding_box:
            max_x, max_y, max_z = bounding_box["max_point"]["x"], bounding_box["max_point"]["y"], \
            bounding_box["max_point"]["z"]
            min_x, min_y, min_z = bounding_box["min_point"]["x"], bounding_box["min_point"]["y"], \
            bounding_box["min_point"]["z"]
            size_x = max_x - min_x
            size_y = max_y - min_y
            size_z = max_z - min_z
            adjust = False
            view_up = (0, 0, 1)
            # 找到最长边 & 最短边
            scale_factor = 2.0  # 可调整以控制模型大小
            sizes = [size_x, size_y, size_z]
            aspect_ratio = max(sizes) / min(s for s in sizes if s > 1e-6)  # 防止除0
            if aspect_ratio > 5 or use_slender:
                longest = max(sizes)
                small_dims = [s for s in sizes if s < longest / 2]  # 其余两个维度相对较小
                if len(small_dims) >= 2:
                    adjust = True
                # ✍️ 记录文件名

                sizes = {
                    'x': size_x,
                    'y': size_y,
                    'z': size_z,
                }
                longest_axis = max(sizes, key=sizes.get)
                stl_name = os.path.splitext(os.path.basename(json_path))[0]
                with open(self.slender_log, "a") as f:
                    f.write(f"{stl_name}, {longest_axis}\n")
                if longest_axis == 'x':
                    angles = [
                        # 第 1 环 (theta=30)
                        (45, 30), (135, 30), (225, 30), (315, 30),
                        # 第 2 环 (theta=60)
                        (35.264, 45), (35.264, 225), (90, 0), (0, 0), (0, 180)
                    ]
                    view_up = (0, 0, 0)
                elif longest_axis == 'y':
                    angles = [
                        (45, 30), (135, 30), (225, 30), (315, 30),
                        # 第 2 环 (theta=60)
                        (195, 180), (35.264, 225), (35.264, 45), (0, 90), (0, -90)
                    ]
                    view_up = (0, 0, 1)
                else:
                    angles = [
                        # 第 1 环 (theta=30)
                        (45, 30), (135, 30), (225, 30), (315, 30),
                        # 第 2 环 (theta=60)
                        (0, 90), (35.264, 225), (0, 0), (90, 0), (-90, 180)
                    ]
                    view_up = (0, -1, 0)
            # 计算模型对角线长度
            diag_length = np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])
            # 设定 scale_factor 并计算相机距离 distance

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            center_z = (max_z + min_z) / 2
            distance = scale_factor * diag_length
            camera_positions = []
            for idx, angle in enumerate(angles):
                theta, phi = angle

                theta_rad = np.radians(theta)
                phi_rad = np.radians(phi)

                x = center_x + distance * np.cos(theta_rad) * np.cos(phi_rad)
                y = center_y + distance * np.cos(theta_rad) * np.sin(phi_rad)
                z = center_z + distance * np.sin(theta_rad)
                camera_positions.append([(x, y, z), (center_x, center_y, center_z), view_up])
                # camera_positions.append([(x, y, z), (0,0,0), view_up])

        else:
            raise ValueError("JSON 文件中没有 bounding_box 数据")
        return camera_positions, adjust, [diag_length, aspect_ratio]

    def render_single_view(self,mesh, cam_pos, adjust, set_parallel_scale, scale,shadow):
        """多进程渲染单个视角（需要接收 `mesh` 和 `cam_pos`）"""
        plotter = pv.Plotter(off_screen=True)
        if shadow or set_parallel_scale:
            plotter.add_mesh(mesh, color="lightgrey",smooth_shading=True)
        else:
            plotter.add_mesh(mesh, color="lightgrey")
        plotter.camera.parallel_projection = True  # 平行投影避免透视变形

        if set_parallel_scale:
            if scale[1] < 20:
                factor = 0.2
            elif scale[1] >= 20 and scale[1] < 50:
                factor = 0.12
            elif scale[1] >= 50 and scale[1] < 80:
                factor = 0.09
            elif scale[1] >= 80:
                factor = 0.07
            plotter.camera.parallel_scale = scale[0] * factor  # 可以调成 0.5~0.6，自由调节
        plotter.camera_position = cam_pos
        if adjust:
            plotter.camera.SetViewAngle(60)
        img = plotter.screenshot(None, window_size=[1280, 720])
        plotter.close()
        img_pil = Image.fromarray(img)
        if adjust:
            # 自动裁剪白边或黑边（你可以根据你的背景色改成 (0, 0, 0)）
            cropped = self.auto_crop_background(img_pil, bg_color=(255, 255, 255), tolerance=10)
            return cropped
        else:
            return img_pil

    def auto_crop_background(self, image: Image.Image, bg_color=(255, 255, 255), tolerance=10, margin_ratio=0.05):
        """
        自动裁剪图像背景，同时保持原图的宽高比例，并额外保留一定边缘空白
        :param image: PIL.Image 输入图像
        :param bg_color: 背景颜色（默认白）
        :param tolerance: 容差范围
        :param margin_ratio: 留白比例（0.05 表示上下左右各留 5%）
        :return: 最终裁剪并填充后的图像
        """
        img_np = np.array(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            diff = np.abs(img_np - bg_color)
            mask = np.any(diff > tolerance, axis=2)
        else:
            diff = np.abs(img_np - bg_color[0])
            mask = diff > tolerance

        coords = np.argwhere(mask)
        if coords.size == 0:
            return image.copy()  # 全是背景

        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0) + 1

        # 增加边缘留白区域
        h, w = img_np.shape[:2]
        margin_w = int((bottom_right[1] - top_left[1]) * margin_ratio)
        margin_h = int((bottom_right[0] - top_left[0]) * margin_ratio)

        # 添加边界保护，防止越界
        left = max(0, top_left[1] - margin_w)
        upper = max(0, top_left[0] - margin_h)
        right = min(w, bottom_right[1] + margin_w)
        lower = min(h, bottom_right[0] + margin_h)

        cropped = image.crop((left, upper, right, lower))

        # 保持原图比例填充
        original_width, original_height = image.size
        target_aspect = original_width / original_height
        cropped_width, cropped_height = cropped.size
        cropped_aspect = cropped_width / cropped_height

        if cropped_aspect > target_aspect:
            new_height = int(cropped_width / target_aspect)
            new_width = cropped_width
        else:
            new_width = int(cropped_height * target_aspect)
            new_height = cropped_height

        # 居中填充裁剪结果
        result = Image.new("RGB", (new_width, new_height), bg_color)
        paste_x = (new_width - cropped_width) // 2
        paste_y = (new_height - cropped_height) // 2
        result.paste(cropped, (paste_x, paste_y))

        # 缩放回原图尺寸
        result = result.resize((original_width, original_height), Image.BICUBIC)

        return result

    def render_image(self, stl_path, json_path, save_path, shadow = False, use_slender = False):
        """ 渲染 STL 文件，并生成合成图像 """
        mesh = pv.read(stl_path)
        try:
            mesh = mesh.subdivide(4)
        except:
            print("⚠️ Subdivision failed, skipping.")
        camera_positions, adjust, scale = self.compute_camera(json_path,use_slender=use_slender)
        images = []
        set_parallel_scale = False
        for idx, cam in enumerate(camera_positions):
            if adjust:
                if idx in [7, 8]:
                    set_parallel_scale = True
                    adjust = False
            images.append(self.render_single_view(mesh, cam, adjust, set_parallel_scale, scale,shadow))
        # 合成 3x3 图片
        img_width, img_height = images[0].size
        combined_image = Image.new("RGB", (img_width * 3, img_height * 3), "white")

        for i, img in enumerate(images):
            x_offset = (i % 3) * img_width
            y_offset = (i // 3) * img_height
            img = ImageOps.expand(img, border=5, fill="black")
            combined_image.paste(img, (x_offset, y_offset))

        combined_image.save(save_path)

    def load_processed_files(self, log_file):
        """ 读取 log.txt，加载已处理的 STL 文件列表 """
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
            return set()

        with open(log_file, "r") as f:
            return set(line.strip() for line in f.readlines())

    def save_to_log(self, log_file, file_prefix, error_msg=None):
        """ 记录 STL 处理状态到对应的 log 文件 """
        with open(log_file, "a") as f:
            if error_msg:
                f.write(f"{file_prefix} - 失败: {error_msg}\n")
            else:
                f.write(f"{file_prefix}\n")

    def process_stl_file(self, stl_path, skip_check_have_processed_files = False, outpath = None, shadow = False, use_slender=False):
        """ 处理单个 STL 文件 """
        processed_files = self.load_processed_files(self.log_success)
        file_prefix, json_path, output_path = self.get_file_path(stl_path)
        if not skip_check_have_processed_files:
            if file_prefix in processed_files:
                print(f"🔄 跳过 {stl_path} (已成功处理)")
                return
            try:
                self.render_image(stl_path, json_path, output_path, shadow = shadow,use_slender=use_slender)
                # **成功后记录到 log_success.txt 并删除 STL**
                self.save_to_log(self.log_success, file_prefix)
                os.remove(stl_path)
                print(f"✅ 处理完成: {stl_path}")
            except Exception as e:
                print(f"⚠️ 处理 {stl_path} 失败: {e}")
                self.save_to_log(self.log_failed, file_prefix, error_msg=str(e))
                raise  # 🚨 关键：重新抛出异常，让外层可以捕获
        else:
            self.render_image(stl_path, json_path, output_path,shadow = shadow, use_slender=use_slender)
    def process_all_stl_files(self):
        """
            遍历 STL 文件夹，匹配对应 JSON，计算距离并批量渲染 STL
            """
        processed_files = self.load_processed_files(log_success)
        stl_files = [f for f in os.listdir(self.stl_dir) if f.endswith(".stl")]
        # **固定进程池，避免频繁创建销毁**
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = []

            for stl_file in stl_files:
                try:
                    stl_path = os.path.join(self.stl_dir, stl_file)
                    file_prefix, json_index_path, output_path = self.get_file_path(stl_path)
                    if file_prefix in processed_files:
                        print(f"🔄 跳过 {stl_file} (已成功处理)")
                        continue
                    results.append(
                        pool.apply_async(self.render_image, args=(stl_path, json_index_path, output_path)))
                    save_to_log(log_success, file_prefix)
                except Exception as e:
                    print(f"⚠️ 处理 {stl_file} 失败: {e}")
                    self.save_to_log(log_failed, file_prefix, error_msg=str(e))
                    os.remove(stl_path)  # 删除！！！！注意
            # **等待所有任务完成**
            for result in results:
                result.get()

        print("✅ 所有 STL 处理完成！")

def get_file_path(stl_path, json_d, output_d):
    file_name = os.path.basename(stl_path)
    file_prefix = os.path.splitext(file_name)[0]  # "04422_index_3"
    json_path = os.path.join(json_d, f"{file_prefix}.json")
    output_path = os.path.join(output_d, f"{file_prefix}.png")
    return file_prefix,json_path, output_path
def compute_camera(json_path, angles):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON 文件不存在: {json_path}")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    # 提取 bounding_box 信息
    bounding_box = json_data.get("properties", {}).get("bounding_box", {})
    if bounding_box:
        max_x, max_y, max_z = bounding_box["max_point"]["x"], bounding_box["max_point"]["y"], bounding_box["max_point"]["z"]
        min_x, min_y, min_z = bounding_box["min_point"]["x"], bounding_box["min_point"]["y"], bounding_box["min_point"]["z"]
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        adjust = False
        view_up=(0,0,1)
        # 找到最长边 & 最短边
        scale_factor = 2.0  # 可调整以控制模型大小
        sizes = [size_x, size_y, size_z]
        aspect_ratio = max(sizes) / min(s for s in sizes if s > 1e-6)  # 防止除0
        if aspect_ratio>10:
            adjust = True
            sizes = {
                'x': size_x,
                'y': size_y,
                'z': size_z,
            }
            longest_axis = max(sizes, key=sizes.get)
            if longest_axis == 'x':
                angles = [
                    # 第 1 环 (theta=30)
                    (45, 30), (135, 30), (225, 30), (315, 30),
                    # 第 2 环 (theta=60)
                    (195, 180), (35.264, 225),(0, 90),(0, 0),(0, 180)
                ]
            elif longest_axis == 'y':
                angles = [
                    # 第 1 环 (theta=30)
                    (45, 30), (135, 30), (225, 30), (315, 30),
                    # 第 2 环 (theta=60)
                    (195, 180), (35.264, 225),(0, 0), (0, 90), (0, -90)
                ]
            else:
                angles = [
                    # 第 1 环 (theta=30)
                    (45, 30), (135, 30), (225, 30), (315, 30),
                    # 第 2 环 (theta=60)
                    (195, 180), (35.264, 225),(0, 0), (90, 0),(-90, 0)
                ]
                view_up = (0, -1, 0)
        # 计算模型对角线长度
        diag_length = np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])
        # 设定 scale_factor 并计算相机距离 distance

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        center_z = (max_z + min_z) / 2
        distance = scale_factor * diag_length
        camera_positions = []
        for idx, angle in enumerate(angles):
            theta, phi = angle

            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi)

            x = center_x + distance * np.cos(theta_rad) * np.cos(phi_rad)
            y = center_y + distance * np.cos(theta_rad) * np.sin(phi_rad)
            z = center_z + distance * np.sin(theta_rad)
            camera_positions.append([(x, y, z), (center_x, center_y, center_z), view_up])
            #camera_positions.append([(x, y, z), (0,0,0), view_up])

    else:
        raise ValueError("JSON 文件中没有 bounding_box 数据")
    return camera_positions, adjust, diag_length

def render_single_view(mesh, cam_pos, adjust,set_parallel_scale,diag_length):
    """多进程渲染单个视角（需要接收 `mesh` 和 `cam_pos`）"""
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(mesh, color="lightgrey")
    plotter.camera.parallel_projection = True  # 平行投影避免透视变形
    if set_parallel_scale:
        plotter.add_mesh(mesh, color="lightgrey", smooth_shading=True)
        plotter.camera.parallel_scale = diag_length * 0.55  # 可以调成 0.5~0.6，自由调节
    plotter.render()
    plotter.camera_position = cam_pos
    if adjust:
        plotter.camera.SetViewAngle(60)
    img = plotter.screenshot(None, window_size=[1280, 720])
    plotter.close()
    img_pil = Image.fromarray(img)
    if adjust:
    # 自动裁剪白边或黑边（你可以根据你的背景色改成 (0, 0, 0)）
        cropped = auto_crop_background(img_pil, bg_color=(255, 255, 255), tolerance=10)
        return cropped
    else:
        return img_pil
def auto_crop_background(image: Image.Image, bg_color=(255, 255, 255), tolerance=10):
    """
    自动裁剪图像背景，同时保持原图的宽高比例（通过填充方式）
    :param image: PIL.Image 输入图像
    :param bg_color: 背景颜色，默认白色
    :param tolerance: 背景容差
    :return: 裁剪 + 保持比例后的图像
    """
    img_np = np.array(image)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        diff = np.abs(img_np - bg_color)
        mask = np.any(diff > tolerance, axis=2)
    else:
        diff = np.abs(img_np - bg_color[0])
        mask = diff > tolerance

    coords = np.argwhere(mask)
    if coords.size == 0:
        return image.copy()  # 全是背景，不裁剪

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0) + 1

    cropped = image.crop((*top_left[::-1], *bottom_right[::-1]))

    # 保持原图比例
    original_width, original_height = image.size
    target_aspect = original_width / original_height
    cropped_width, cropped_height = cropped.size
    cropped_aspect = cropped_width / cropped_height

    if cropped_aspect > target_aspect:
        # 裁剪图太宽，按宽度算出应该的高度
        new_height = int(cropped_width / target_aspect)
        new_width = cropped_width
    else:
        # 裁剪图太高，按高度算出应该的宽度
        new_width = int(cropped_height * target_aspect)
        new_height = cropped_height

    # 创建新图并居中粘贴裁剪结果
    result = Image.new("RGB", (new_width, new_height), bg_color)
    paste_x = (new_width - cropped_width) // 2
    paste_y = (new_height - cropped_height) // 2
    result.paste(cropped, (paste_x, paste_y))

    # 最终 resize 回原始大小（可选）
    result = result.resize((original_width, original_height), Image.BICUBIC)

    return result
def render_image_parallel(stl_path, json_path, angles, save_path):
    """ **多进程渲染 STL 文件** """
    mesh = pv.read(stl_path)
    try:
        mesh = mesh.subdivide(4)
    except:
        print("⚠️ Subdivision failed, skipping.")
    camera_positions,adjust, diag_length = compute_camera(json_path, angles)
    images = []
    set_parallel_scale = False
    for idx, cam in enumerate(camera_positions):
        if adjust:
           if idx in [7,8]:
               set_parallel_scale  = True
               adjust = False
        images.append(render_single_view(mesh, cam, adjust,set_parallel_scale,diag_length))

    # **合成 3x3 图片**
    img_width, img_height = images[0].size
    combined_image = Image.new("RGB", (img_width * 3, img_height * 3), "white")

    for i, img in enumerate(images):
        x_offset = (i % 3) * img_width
        y_offset = (i // 3) * img_height
        img = ImageOps.expand(img, border=5, fill="black")
        combined_image.paste(img, (x_offset, y_offset))

    combined_image.save(save_path)

def load_processed_files(log_file):
    """ 读取 log.txt，加载已处理的 STL 文件列表 """
    if not os.path.exists(log_file):
        # 创建空日志文件，避免后续操作出错
        open(log_file, 'w').close()
        return set()

    with open(log_file, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_to_log(log_file, file_prefix, error_msg=None):
    """ 记录 STL 处理状态到对应的 log 文件 """
    with open(log_file, "a") as f:
        if error_msg:
            f.write(f"{file_prefix} - 失败: {error_msg}\n")
        else:
            f.write(f"{file_prefix}\n")

def main():
    """
    遍历 STL 文件夹，匹配对应 JSON，计算距离并批量渲染 STL
    """
    processor = STLProcessor(
        stl_dir=stl_dir,
        json_dir="/home/baiyixue/data/CADParser_data/json",
        image_output_dir="/home/baiyixue/project/DeepCAD/CADparser/image_adj/",
        log_success="/home/baiyixue/project/DeepCAD/CADparser/image/log_success.txt",
        log_failed="/home/baiyixue/project/DeepCAD/CADparser/image/log_failed.txt",
        # angles=[(45, 30), (135, 30), (225, 30), (315, 30),
        #         (0, 90), (90, 0), (195, 180), (35.264, 225), (35.264, 45)]
        angles=[(45, 30), (135, 30), (225, 30), (315, 30),
                (0, 90), (90, 0), (195, 180), (35.264, 225), (35.264, 45)]
    )
    skip_check_have_processed_files = True
    use_slender = True
    stl_files = [f for f in os.listdir(stl_dir) if f.endswith(".stl")]
    # **固定进程池，避免频繁创建销毁**
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = []

        for stl_file in stl_files:
            stl_path = os.path.join(stl_dir, stl_file)
            processor.process_stl_file(stl_path, skip_check_have_processed_files=skip_check_have_processed_files, use_slender=use_slender)

        # **等待所有任务完成**
        for result in results:
            result.get()

    print("✅ 所有 STL 处理完成！")

def extract_code_ids(input_path):
    """读取文件路径列表，并提取路径中编号（去除扩展名的文件名）"""
    code_ids = []
    with open(input_path, 'r') as file:
        for line in file:
            filepath = line.strip()
            if filepath:
                filename = os.path.basename(filepath)         # 获取文件名，例如 02911_index_3.py
                code_id = os.path.splitext(filename)[0]       # 去掉扩展名，得到 02911_index_3
                code_ids.append(code_id)
    return code_ids
# 运行主程序
if __name__ == "__main__":
    stl_dir = "/home/baiyixue/project/DeepCAD/CADparser/stl/"  # STL 文件夹
    json_dir = "/home/baiyixue/data/CADParser_data/json"  # JSON 文件夹
    image_output_dir = "/home/baiyixue/project/DeepCAD/CADparser/image_adj"  # 输出文件夹
    log_success = "/home/baiyixue/project/DeepCAD/CADparser/image/log_success.txt"  # 记录成功处理的 STL
    log_failed = "/home/baiyixue/project/DeepCAD/CADparser/image/log_failed.txt"  # 记录失败的 STL

    os.makedirs(image_output_dir, exist_ok=True)
    main()