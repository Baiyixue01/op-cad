#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from typing import List, Dict, Tuple, Optional

import cadquery as cq


# =========================
#         工具函数
# =========================

def union_step_list(step_list: List[str], out_path: str, do_clean: bool = True) -> None:
    """
    将多个 STEP（每个为一个 stepX/3D.step）做布尔并集后导出到 out_path。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    solids = []
    for p in step_list:
        try:
            wp = cq.importers.importStep(p)
            val = wp.val()
            ss = list(val.Solids()) if hasattr(val, "Solids") else []
            if not ss and val.__class__.__name__.lower() == "solid":
                ss = [val]
            print(f"📦 {os.path.basename(os.path.dirname(p))}/3D.step -> solids: {len(ss)}")
            solids.extend(ss)
        except Exception as e:
            print(f"⚠️ 读取失败 {p}: {e}")

    if not solids:
        raise RuntimeError("未从任何 STEP 中提取到 solid。")

    # 迭代 union
    result = cq.Workplane().add(solids[0])
    for s in solids[1:]:
        try:
            result = result.union(cq.Workplane().add(s))
        except Exception as e:
            print(f"⚠️ union 失败，尝试 clean 后重试：{e}")
            try:
                result = result.clean().union(cq.Workplane().add(s))
            except Exception as e2:
                print(f"⚠️ 重试仍失败，跳过该 solid：{e2}")

    if do_clean:
        result = result.clean()

    cq.exporters.export(result, out_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ 合并完成：{out_path}")
    else:
        print(f"❌ 导出失败：{out_path}")


def _parse_group_info_txt(gi_path: str) -> Dict[str, List[int]]:
    """
    解析 group_info.txt，返回 { 'step1': [1,2,3], 'step0': [0], ... }
    适配如下格式示例：
        step0: [{0: 'Sketch-Revolve pair 0'}]
        step1: [{1: 'Sketch-Extrude pair 1'}, {2: 'Sketch-Extrude pair 2'}, {3: 'Sketch-Extrude pair 3'}]
    （宽松解析：用正则抽取每个 stepX 段内所有 {num: ...} 的数字 key）
    """
    if not os.path.exists(gi_path):
        raise FileNotFoundError(f"group_info.txt 不存在: {gi_path}")

    with open(gi_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # 找每个 step 段： step<k> : [ ... ]
    # DOTALL: 允许跨行；非贪婪匹配到下一个 'step\d+:' 或文件结尾
    pattern = re.compile(r"(step\d+)\s*:\s*\[(.*?)\](?=\s*step\d+\s*:|\s*\Z)", re.S | re.I)
    out: Dict[str, List[int]] = {}

    for m in pattern.finditer(txt):
        step_key = m.group(1).strip()  # e.g., "step1"
        block = m.group(2)

        # 在 block 中找 {number: 的 number
        idxs = [int(n) for n in re.findall(r"\{\s*(\d+)\s*:", block)]
        # 去重并排序（稳定）
        idxs = sorted(set(idxs), key=lambda x: idxs.index(x)) if idxs else []
        out[step_key] = idxs

    # 兜底：若没匹配到任何 step 段，尝试单行简易解析
    if not out:
        for line in txt.splitlines():
            m = re.match(r"^\s*(step\d+)\s*:\s*\[(.*)\]\s*$", line.strip(), flags=re.I)
            if not m:
                continue
            step_key = m.group(1)
            block = m.group(2)
            idxs = [int(n) for n in re.findall(r"\{\s*(\d+)\s*:", block)]
            out[step_key] = idxs

    return out


def _build_merge_target_name(indices: List[int]) -> str:
    """
    根据一组 step 索引生成目标目录名，例如 [1,2,3] -> 'step1_2_3'
    """
    if not indices:
        raise ValueError("indices 为空，无法生成合并目标名")
    return "step" + "_".join(str(i) for i in indices)


def _step_dir(group_dir: str, idx: int) -> str:
    """返回某一步的目录路径：<group_dir>/step<idx>"""
    return os.path.join(group_dir, f"step{idx}")


def _step_file(group_dir: str, idx: int) -> str:
    """返回某一步的 3D.step 路径：<group_dir>/step<idx>/3D.step"""
    return os.path.join(_step_dir(group_dir, idx), "3D.step")


# =========================
#      主处理流程
# =========================

def merge_groups_under(root_dir: str, overwrite: bool = False) -> None:
    """
    遍历 root_dir 下的所有子目录，寻找每个目录中的 group_info.txt，
    对其中每个 'stepK: [ {i: ...}, {j: ...}, ... ]' 的条目：
      - 若包含 >=2 个索引，则将对应 step 目录下的 3D.step 合并为 stepK_i_j_.../3D.step
      - 若仅包含 1 个索引，跳过（无需合并）
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"root_dir 非目录：{root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "group_info.txt" not in filenames:
            continue

        group_dir = dirpath
        gi_path = os.path.join(group_dir, "group_info.txt")
        print(f"\n==== 处理 {group_dir} ====")

        try:
            mapping = _parse_group_info_txt(gi_path)
        except Exception as e:
            print(f"⚠️ 解析失败 {gi_path}: {e}")
            continue

        if not mapping:
            print("⚠️ 未在 group_info.txt 中解析到任何 step 条目，跳过")
            continue

        # 对每个 stepK 的组合执行合并
        for step_key, idxs in mapping.items():
            # 只对长度 >= 2 的组合进行合并
            if not idxs or len(idxs) < 2:
                continue

            # 目标目录名：以该 step_key 的数字 + idxs 组成，例如 step1: [1,2,3] -> step1_2_3
            # 注意：idxs 已包含 step_key 的数字（通常如此），直接用 idxs 构建结果名
            target_dir_name = _build_merge_target_name(idxs)
            target_dir = os.path.join(group_dir, target_dir_name)
            target_file = os.path.join(target_dir, "3D.step")

            # 如果已存在且不覆盖，跳过
            if (not overwrite) and os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                print(f"⏩ 已存在且不覆盖：{target_file}")
                continue

            # 收集输入 step 文件列表
            step_files = [_step_file(group_dir, i) for i in idxs]
            missing = [p for p in step_files if not os.path.exists(p)]
            if missing:
                print(f"⚠️ 缺少以下输入 STEP，跳过：\n    " + "\n    ".join(missing))
                continue

            # 执行合并
            try:
                os.makedirs(target_dir, exist_ok=True)
                print(f"➡️ 合并 {step_key}: {idxs} -> {target_file}")
                union_step_list(step_files, target_file, do_clean=True)
            except Exception as e:
                print(f"❌ 合并失败 {step_key} {idxs}: {e}")


# =========================
#         命令行
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python merge_steps.py <root_dir> [--overwrite]")
        sys.exit(1)

    root = sys.argv[1]
    overwrite_flag = ("--overwrite" in sys.argv[2:])
    merge_groups_under(root, overwrite=overwrite_flag)
