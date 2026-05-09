import os
import subprocess
import multiprocessing
import sys
import re
from render_image import STLProcessor

# === 配置 ===
folder_path = "/home/baiyixue/project/DeepCAD/CADparser/adjusted_code_revolve_axis/"
save_path = "/home/baiyixue/project/DeepCAD/CADparser/stl/"
error_save_path = "/home/baiyixue/project/DeepCAD/CADparser/"
scripts_to_run = [f for f in os.listdir(folder_path) if f.endswith('.py')]
input_txt_path = "/home/baiyixue/project/DeepCAD/CADparser/filtered_files.txt"
# scripts_to_run = []
# with open(input_txt_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()  # 读取所有行，保存在列表中
#     for line in lines:
#         scripts_to_run.append(line[:-1])
# scripts_to_run = ["/home/baiyixue/project/DeepCAD/CADparser/code/03189_index_4.py"]
# 设置路径

processed_log_path = os.path.join(error_save_path, "stl_processed_files.log")
error_log_file = os.path.join(error_save_path, "Cadparser_error.txt")
skip_check_have_processed_files = True
render_or_not = True
image_save_path = "/home/baiyixue/project/DeepCAD/CADparser/image_adj/"
# === 多卡参数 ===
gpu_ids = [0, 2]           # 可用 GPU
processes_per_gpu = 24     # 每张 GPU 跑几个进程
total_slots = len(gpu_ids) * processes_per_gpu

def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_processed_file(log_path, file_path):
    with open(log_path, 'a') as f:
        f.write(f"{file_path}\n")

def replace_export_filename(script_path):
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"exportStl\((.*?)\)"
    replacement = f"r'{save_path}/{script_name}.stl'"
    match = re.search(pattern, content)
    if match:
        content = content.replace(match.group(1), replacement)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"{save_path}/{script_name}.stl"
    else:
        print(f"No exportStl() found in {script_path}")
        return None

def execute_script_on_gpu(script_path, in_progress_dict, lock, gpu_id="0,1",render_image=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stl_path = replace_export_filename(script_path)
    if render_image:
        processor = STLProcessor(
            stl_dir=save_path,
            json_dir="/home/baiyixue/data/CADParser_data/json",
            image_output_dir=image_save_path ,
            log_success="/home/baiyixue/project/DeepCAD/CADparser/image/log_success.txt",
            log_failed="/home/baiyixue/project/DeepCAD/CADparser/image/log_failed.txt",
            angles=[(45, 30), (135, 30), (225, 30), (315, 30),
                    (0, 90), (90, 0), (195, 180), (35.264, 225), (35.264, 45)]
        )

    with lock:
        processed_files = load_processed_files(processed_log_path)
        if not skip_check_have_processed_files:
            if script_path in processed_files or in_progress_dict.get(script_path):
                print(f"[GPU {gpu_id}] 跳过已处理或正在处理: {script_path}")
                return
        in_progress_dict[script_path] = True

    try:
        print(f"[GPU {gpu_id}] 运行脚本: {script_path}")
        subprocess.run(['python', script_path], check=True, text=True, capture_output=True, timeout=180)
        if render_image:
            try:
                processor.process_stl_file(stl_path,skip_check_have_processed_files=skip_check_have_processed_files)
                with lock:
                    save_processed_file(processed_log_path, script_path)
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"{script_path} - Error: {e}\n")
    except subprocess.TimeoutExpired:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - Timeout\n")
    except subprocess.CalledProcessError as e:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - Error: {e.stderr}\n")
    except Exception as e:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - Exception: {str(e)}\n")
    finally:
        with lock:
            if script_path in in_progress_dict:
                del in_progress_dict[script_path]
    return stl_path

def worker(gpu_id, script_list, in_progress_dict, lock):
    for script in script_list:
        script_path = os.path.join(folder_path, script)
        execute_script_on_gpu(script_path, in_progress_dict, lock, gpu_id=gpu_id)

def main():
    manager = multiprocessing.Manager()
    in_progress_dict = manager.dict()
    lock = manager.Lock()

    chunks = [scripts_to_run[i::total_slots] for i in range(total_slots)]
    processes = []

    for i, script_chunk in enumerate(chunks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        p = multiprocessing.Process(target=worker, args=(gpu_id, script_chunk, in_progress_dict, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def execute_script(script_path, gpu_id="0,1", render_image=False):
    """
    顺序执行一个 Python 建模脚本，导出 STL 并可选渲染图像。

    参数:
        script_path (str): Python 脚本路径
        gpu_id (str): 用于渲染/推理的 CUDA GPU ID（默认 "0,1"）
        render_image (bool): 是否启用 STL 渲染

    返回:
        stl_path (str): 生成的 STL 文件路径
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    stl_path = replace_export_filename(script_path)

    if render_image:
        processor = STLProcessor(
            stl_dir=save_path,
            json_dir="/home/baiyixue/data/CADParser_data/json",
            image_output_dir=image_save_path,
            log_success="/home/baiyixue/project/DeepCAD/CADparser/image/log_success.txt",
            log_failed="/home/baiyixue/project/DeepCAD/CADparser/image/log_failed.txt",
            angles=[(45, 30), (135, 30), (225, 30), (315, 30),
                    (0, 90), (90, 0), (195, 180), (35.264, 225), (35.264, 45)]
        )

    try:
        print(f"🚀 正在执行脚本: {script_path}")
        subprocess.run(['python', script_path], check=True, text=True, capture_output=True, timeout=180)

        if render_image:
            try:
                processor.process_stl_file(stl_path, skip_check_have_processed_files=True)
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"{script_path} - 图像渲染错误: {e}\n")

    except subprocess.TimeoutExpired:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - Timeout\n")

    except subprocess.CalledProcessError as e:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - 程序报错: {e.stderr}\n")

    except Exception as e:
        with open(error_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{script_path} - 未知异常: {str(e)}\n")

    return stl_path


if __name__ == "__main__":
    main()
