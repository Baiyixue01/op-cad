# ---- 新增：多进程所需 ----
import multiprocessing as mp
from functools import partial
import os, traceback, json, time
import pandas as pd
import numpy as np
import cadquery as cq
from model_call.call_model import get_model_candidates, MODEL
from model_call.prompt import build_incremental_cq_prompt
from utils.post_code_process import build_iso_code, build_integrated_code
import os, re, ast
from typing import Optional, Tuple, List
from utils.compute_3D import get_cd_hd,MetricsResult
from functools import lru_cache
_G = {}
_dedup_map = None
DEDUP_CSV = None 
OP_ORIENT_DIR = None  



# ===================== 工具初始化 =====================

# ==== argparse: 统一配置入口 ====
import argparse
from datetime import datetime

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Op-CAD 多进程评测脚本（可指定 test 名/模型名 与 cop/非cop 模式）"
    )
    # 运行标识
    p.add_argument("--test-name", default=None,
                help="本次测试名称；若未提供，则从 config/model.json 读取 model 字段")
    p.add_argument("--mode", choices=["std","cop"], default="std",
                   help="运行模式：std=非COP（默认），cop=COP增量代码模式")
    p.add_argument("--resume", action='store_true', default=True,
                   help="开启断点续跑（summary.csv 存在则跳过已完成）")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="不开启断点续跑（强制全部重新运行）")
    p.add_argument("--seed", type=int, default=42, help="全局随机种子")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda", help="推理设备")

    # 输入/输出路径
    p.add_argument("--prompts-csv", default="./data/prompt.csv", help="至少包含 group_index,prompt_text 的 CSV")
    p.add_argument("--out-root", default="./inference", help="输出根目录，程序会在其下创建 test_name 子目录")

    # 代码/GT 目录
    p.add_argument("--pre-code-dir", default="./data/pre_code", help="非COP：前序代码目录")
    p.add_argument("--cop-pre-code-dir",default="./data/pre_code_cop", help="COP：前序代码目录（增量链）")
    p.add_argument("--gt-image-dir", required=True, help="GT 渲染图根目录")
    p.add_argument("--gt-single-step-dir", required=True, help="单步GT STEP根目录")
    p.add_argument("--op-orient-dir", required=True, help="整体形状（累计到 stepN）的 STEP 根目录")
    p.add_argument("--dedup-csv", required=True, help="去重映射 CSV（group_index, duplicate_of_group_index）")

    # 评测与渲染参数
    p.add_argument("--k", type=int, default=2, help="pass@k 的 k")
    p.add_argument("--pass-metric", choices=["cosine","iou"], default="cosine", help="主评价指标（若使用图像相似）")
    p.add_argument("--cos-threshold", type=float, default=0.90, help="cosine 阈值（仅在 cosine 模式下有效）")
    p.add_argument("--image-size", type=int, default=518, help="DINO 输入尺寸")
    p.add_argument("--fivecrop", action="store_true", help="若你在 render_to_png 做了多视角，可关闭这里；否则可开")

    # 多进程
    p.add_argument("--nproc", type=int, default=max(1, mp.cpu_count()-1), help="进程数")
    p.add_argument("--write-every", type=int, default=3, help="每处理多少个样本落地一次")

    # 产物保存开关
    p.add_argument("--save-step", action="store_true", default=True, help="保存中间 step（isolated）")
    p.add_argument("--no-save-step", dest="save_step", action="store_false", help="不保存中间 step")
    p.add_argument("--save-render", action="store_true", default=True, help="保存渲染 png")
    p.add_argument("--no-save-render", dest="save_render", action="store_false", help="不保存渲染 png")

    # DINO
    p.add_argument("--dino-model-id", default="facebook/dinov2-base",
                   help="DINOv2 模型ID（如 facebook/dinov2-large）")

    return p

def apply_args(args):
    global PROMPTS_CSV, OUT_DIR, PRE_CODE_DIR, COP_PRE_CODE_DIR, GT_IMAGE_DIR, DEDUP_CSV
    global GT_SINGLE_STEP_DIR, OP_ORIENT_DIR, K, DEVICE, PASS_METRIC, COS_THRESHOLD
    global IMAGE_SIZE, FIVECROP, COP, SAVE_STEP, SAVE_RENDER, TMP_DIR, RESUME
    global SEED, DINO_MODEL_ID, NPROC, WRITE_EVERY

    # ===== 运行标识 & 目录结构 =====
    # 目录：<out-root>/<test-name>__<mode>/（附加时间戳避免覆盖，可按需去掉）
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.test_name:  # 用户没传参
        args.test_name = MODEL.replace("/", "_")
    mode_tag = args.mode  # "std" or "cop"
    run_dir = os.path.join(args.out_root, args.test_name, mode_tag)
    os.makedirs(run_dir, exist_ok=True)

    PROMPTS_CSV = args.prompts_csv
    OUT_DIR = run_dir

    PRE_CODE_DIR = args.pre_code_dir
    COP_PRE_CODE_DIR = args.cop_pre_code_dir
    GT_IMAGE_DIR = args.gt_image_dir
    DEDUP_CSV = args.dedup_csv

    GT_SINGLE_STEP_DIR = args.gt_single_step_dir
    OP_ORIENT_DIR = args.op_orient_dir

    K = args.k
    DEVICE = args.device
    PASS_METRIC = args.pass_metric
    COS_THRESHOLD = args.cos_threshold
    IMAGE_SIZE = args.image_size
    FIVECROP = args.fivecrop
    COP = (args.mode == "cop")

    SAVE_STEP = args.save_step
    SAVE_RENDER = args.save_render
    TMP_DIR = os.path.join(run_dir, "code_step")

    RESUME = args.resume
    SEED = args.seed
    DINO_MODEL_ID = args.dino_model_id

    NPROC = args.nproc
    WRITE_EVERY = args.write_every

    # 便于在日志里检索
    print(f"[RUN] test={args.test_name}  mode={args.mode}  out_dir={OUT_DIR}")
    print(f"[RUN] prompts={PROMPTS_CSV}  nproc={NPROC}  resume={RESUME}  seed={SEED}")

#================== multi-process 相关工具 ==================
def _assert_worker_env():
    must_keys = [
        "re_step", "dedup",
        "pre_code_dir", "cop_pre_code_dir",
        "tmp_dir",
    ]
    missing = [k for k in must_keys if not _G.get(k)]
    if missing:
        raise RuntimeError(f"Worker env missing keys: {missing}. "
                           f"Did you pass all initargs and set them in init_worker?")

def init_worker(device, dino_model_id, image_size,
                dedup_csv, op_orient_dir,
                pre_code_dir, cop_pre_code_dir,
                tmp_dir):
    global DEDUP_CSV, OP_ORIENT_DIR
    DEDUP_CSV = dedup_csv
    OP_ORIENT_DIR = op_orient_dir

    import re, logging, os
    log_path = _G.get("run_log_path", "run.log")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s][%(process)d] %(levelname)s: %(message)s')

    _G["re_step"] = re.compile(r"step(\d+)")
    _G["dedup"] = load_dedup_map()  # 依赖 DEDUP_CSV

    # —— 统一注入目录与临时路径 —— #
    _G["pre_code_dir"] = pre_code_dir
    _G["cop_pre_code_dir"] = cop_pre_code_dir
    _G["tmp_dir"] = tmp_dir

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    _assert_worker_env()

@lru_cache(maxsize=100000)
def _resolve_gt_paths_cached(pid, img_dir, single_dir):
    return resolve_gt_paths(pid, img_dir, single_dir)

def atomic_write_df(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def write_final_stats(cand_csv_path: str, summary_csv_path: str, out_txt_path: str, k: int):
    if not (os.path.exists(cand_csv_path) and os.path.exists(summary_csv_path)):
        print(f"[WARN] stats skipped: missing {cand_csv_path} or {summary_csv_path}")
        return

    cands = pd.read_csv(cand_csv_path)
    summ  = pd.read_csv(summary_csv_path)
    cands.columns = [c.lower() for c in cands.columns]
    summ.columns  = [c.lower() for c in summ.columns]

    # -------- pass@k --------
    grp_has_ok = cands.groupby("group_index")["metric_ok"].apply(lambda s: int((s.fillna(0) == 1).any()))
    total_groups = int(grp_has_ok.shape[0])
    pass_groups  = int((grp_has_ok == 1).sum())
    pass_at_k = (pass_groups / total_groups) if total_groups > 0 else 0.0

    # -------- best 的 CD/HD（仅通过组）--------
    best = summ[summ["k_index"] == "summary_best"].copy()
    ok_best = best[best["cd"].notna() & best["hd"].notna()]
    cd_mean   = float(np.nanmean(ok_best["cd"])) if not ok_best.empty else float("nan")
    hd_mean   = float(np.nanmean(ok_best["hd"])) if not ok_best.empty else float("nan")
    cd_median = float(np.nanmedian(ok_best["cd"])) if not ok_best.empty else float("nan")
    hd_median = float(np.nanmedian(ok_best["hd"])) if not ok_best.empty else float("nan")

    # -------- 无效率（candidate 级）--------
    total_cands   = int(cands.shape[0])
    n_pred_exist  = int((cands.get("pred_step_exists", 0).fillna(0) == 1).sum())
    n_metric_ok   = int((cands.get("metric_ok", 0).fillna(0) == 1).sum())

    execution_ineff = 1.0 - (n_pred_exist / total_cands) if total_cands > 0 else float("nan")
    geometry_ineff  = 1.0 - (n_metric_ok  / total_cands) if total_cands > 0 else float("nan")

    # —— 新增：总无效率 ——
    overall_candidate_ineff = 1.0 - (n_metric_ok / total_cands) if total_cands > 0 else float("nan")  # == geometry_ineff
    overall_group_ineff     = 1.0 - pass_at_k

    # -------- 错误分布（全局）--------
    err_dist = cands["err_group"].fillna("other").value_counts().to_dict()
    for key in ("syntax", "geom_op", "other"):
        err_dist.setdefault(key, 0)

    # -------- 输出 --------
    lines = []
    lines.append("==== Final Evaluation Stats ====")
    lines.append(f"Total groups        : {total_groups}")
    lines.append(f"Candidates per group: k={k}")
    lines.append("")
    lines.append(f"pass@k              : {pass_at_k:.6f}  ({pass_groups}/{total_groups})")
    lines.append(f"overall_group_ineff : {overall_group_ineff:.6f}  # 1 - pass@k")
    lines.append("")
    lines.append("CD/HD over PASSED groups (use summary_best):")
    lines.append(f"  mean(cd)          : {cd_mean:.8f}")
    lines.append(f"  median(cd)        : {cd_median:.8f}")
    lines.append(f"  mean(hd)          : {hd_mean:.8f}")
    lines.append(f"  median(hd)        : {hd_median:.8f}")
    lines.append("")
    lines.append("Inefficiency (candidate-level):")
    lines.append(f"  execution_ineff   : {execution_ineff:.6f}  # 1 - (#pred_step_exists / #all_candidates)")
    lines.append(f"  geometry_ineff    : {geometry_ineff:.6f}  # 1 - (#metric_ok / #all_candidates)")
    lines.append(f"  overall_cand_ineff: {overall_candidate_ineff:.6f}  # 1 - (#success_candidates / #all_candidates)")
    lines.append("")
    lines.append("Error groups (counts over all candidates):")
    lines.append(f"  syntax            : {err_dist.get('syntax',0)}")
    lines.append(f"  geom_op           : {err_dist.get('geom_op',0)}")
    lines.append(f"  other             : {err_dist.get('other',0)}")
    lines.append("")

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[stats] wrote final stats to {out_txt_path}")

#================= 执行错误类型识别 =======================
import re, json
from collections import Counter

GEOM_KWS = (
    "extrude", "revolve", "loft", "sweep", "shell", "fillet", "chamfer",
    "offset2d", "offset 2d", "boolean", "cut", "union", "intersect", "fuse",
    "workplane", "wire", "edge", "face", "solid", "thickness",
    "BRep", "TopoDS", "OCC", "gp_", "BOPAlgo", "Geom_", "BRepAlgoAPI",
    "CQ", "cadquery"
)

def parse_exec_error(err_text: str):
    """抽取异常类型/信息/行号"""
    if not err_text:
        return "UnknownError", "", None
    head = err_text.splitlines()[0]
    m = re.match(r"^([A-Za-z_][\w\.]*)\s*:\s*(.*)$", head)
    err_type = m.group(1) if m else "UnknownError"
    err_msg  = m.group(2) if m else head
    line_no = None
    for ln in reversed(err_text.splitlines()):
        mm = re.search(r'File ".*?", line (\d+)', ln)
        if mm:
            try:
                line_no = int(mm.group(1)); break
            except: pass
    return err_type, err_msg, line_no

def classify_err_3way(exec_ok: int, err_type: str, err_msg: str, pred_step_exists: bool, metric_reason: str):
    """
    返回 "syntax" | "geom_op" | "other"
    规则：
      1) SyntaxError/IndentationError => syntax
      2) 执行失败但异常消息命中几何关键词或OCC/CQ类 => geom_op
      3) 执行成功但没产出STEP/评测提示pred_step_missing => geom_op
      4) 其他 => other
    """
    t = (err_type or "").lower()
    m = (err_msg or "").lower()
    mr = (metric_reason or "").lower()

    if exec_ok == 0 and (t in ("syntaxerror", "indentationerror")):
        return "syntax"

    if exec_ok == 0:
        # 典型几何/布尔失败：OCC 栈、BRep、TopoDS、CQ操作关键词
        if any(k.lower() in t or k.lower() in m for k in GEOM_KWS):
            return "geom_op"

    # 代码执行成功但没有几何产物或评测说缺失预测STEP，也归为几何类
    if exec_ok == 1 and (not pred_step_exists or "pred_step_missing" in mr):
        return "geom_op"

    return "other"


##====================step文件路径处理=====================
def _combo_names_from_indices(indices: List[int]) -> List[str]:
    """
    给 [1,2,3] 生成若干可能的组合目录名候选：["1_2_3", "1-2-3", "1,2,3"]
    也会包含最后一个索引的单独候选（如 "3"）用于兜底。
    """
    s = [str(i) for i in indices]
    combos = []
    if indices:
        combos.append("_".join(s))
        combos.append("-".join(s))
        combos.append(",".join(s))
        combos.append(s[-1])  # 兜底：只用最后一个操作编号
    return combos

def _pick_full_step_path(op_orient_group_dir: str, expected_indices: List[int]) -> Optional[str]:
    """
    在 /data/.../op_orientated_step/<group>/ 下，按多种约定尝试找到“整体形状”的 step：
      - <combo>/next_model.step
      - <combo>/3D.step
      - <combo>/<last>/next_model.step
      - <last>/next_model.step
      - <last>/3D.step
    其中 combo 来自 indices 组合（例如 "1_2"、"1-2" 等），last 为 indices 的最后一个编号。
    """
    if not os.path.isdir(op_orient_group_dir):
        return None
    combos = _combo_names_from_indices(expected_indices)

    candidates = []
    for c in combos:
        last = c.split("_")[-1].split("-")[-1].split(",")[-1] if c else None
        # 优先级从高到低依次添加
        candidates += [
            os.path.join(op_orient_group_dir, c, "next_model.step"),
            os.path.join(op_orient_group_dir, c, "3D.step"),
        ]
        if last:
            candidates += [
                os.path.join(op_orient_group_dir, c, last, "next_model.step"),
                os.path.join(op_orient_group_dir, last, "next_model.step"),
                os.path.join(op_orient_group_dir, last, "3D.step"),
            ]

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _numbers_in_folder_suffix(folder_name: str, step_prefix: str) -> List[int]:
    """
    'step1_2_3' with step_prefix 'step1' -> [2,3]
    兼容 'step1_1_2', 'step1_2_3_4' 等
    """
    suf = folder_name[len(step_prefix):]  # e.g. '_2_3'
    nums = re.findall(r"\d+", suf)
    return [int(x) for x in nums]

def _parse_group_info_txt(path: str) -> dict:
    """
    解析 GT_SINGLE_STEP_DIR/<group>/group_info.txt
    支持形如:
      step0: [{0: 'Sketch-Extrude pair 0'}]
      step1: [{1: 'Sketch-Extrude pair 1'}, {2: 'Sketch-Extrude pair 2'}]
    返回: {"step0":[0], "step1":[1,2], ...}
    """
    out = {}
    if not os.path.exists(path):
        print(f"[WARN] group_info not found: {path}")
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"^step(\d+)\s*:\s*(\[.*\])\s*$", s)
            if not m:
                continue
            step = f"step{m.group(1)}"
            try:
                arr = ast.literal_eval(m.group(2))
                # arr 形如 [{1:'xxx'}, {2:'yyy'}]
                idxs = []
                for d in arr:
                    if isinstance(d, dict) and d:
                        key = next(iter(d.keys()))
                        try:
                            idxs.append(int(key))
                        except:
                            pass
                out[step] = idxs
            except Exception as e:
                print(f"[WARN] parse line failed: {s} -> {e}")
    return out

def _extract_group_and_step(pid: str) -> Tuple[str, str]:
    """
    pid: '00003_index_4/step1' -> ('00003_index_4', 'step1')
    """
    parts = str(pid).split("/")
    group = "/".join(parts[:-1]) if len(parts) > 1 else ""
    step  = parts[-1]  # 'step1'
    return group, step

def _pick_step_folder(group_dir: str, step: str, expected_indices: List[int]) -> Optional[str]:
    """
    选择与 step 对应的目录:
      - 兼容两种命名： 'stepN' 以及 'stepN_i_j_k'
      - 优先匹配：目录名中的数字集合 == expected_indices
      - 次之：数字集合 ⊇ expected_indices（超集）
      - 仍无：若存在精确 'stepN' 目录则选之
      - 最后：任取最短集合、字母序稳定的一个
    """
    if not os.path.isdir(group_dir):
        print(f"[WARN] group dir not found: {group_dir}")
        return None

    expected = set(expected_indices or [])
    cand_dirs = []
    step_prefix = step  # e.g. 'step0'

    for name in os.listdir(group_dir):
        full = os.path.join(group_dir, name)
        if not os.path.isdir(full):
            continue
        if not (name == step_prefix or name.startswith(step_prefix + "_")):
            continue
        step_file = os.path.join(full, "3D.step")
        if not os.path.exists(step_file):
            continue

        # 目录名等于 'stepN'（无后缀）时，作为一个候选：
        #   为了参与“完全相等匹配”，我们把它的数字集合当作 expected（如果 expected 非空），
        #   否则当作空集处理。
        if name == step_prefix:
            nums_set = expected if expected else set()
        else:
            nums_set = set(_numbers_in_folder_suffix(name, step_prefix))

        cand_dirs.append((name, nums_set, full))

    if not cand_dirs:
        return None

    # 1) 完全相等
    exact = [c for c in cand_dirs if c[1] == expected]
    if exact:
        exact.sort(key=lambda x: (len(x[1]), x[0]))
        return exact[0][2]

    # 2) 最小超集
    supersets = [c for c in cand_dirs if expected and expected.issubset(c[1])]
    if supersets:
        supersets.sort(key=lambda x: (len(x[1]), x[0]))
        return supersets[0][2]

    # 3) 若存在精确 'stepN' 目录（即 name==stepN），作为兜底
    step_only = [c for c in cand_dirs if c[0] == step_prefix]
    if step_only:
        step_only.sort(key=lambda x: x[0])
        return step_only[0][2]

    # 4) 最终兜底：取集合最短、字母序最小
    cand_dirs.sort(key=lambda x: (len(x[1]), x[0]))
    return cand_dirs[0][2]


def load_dedup_map() -> dict:
    """读取去重映射表：返回 {group_index: canonical_group_index}"""
    global _dedup_map
    if _dedup_map is None:
        if not os.path.exists(DEDUP_CSV):
            print(f"[WARN] 去重文件未找到：{DEDUP_CSV}")
            _dedup_map = {}
        else:
            df = pd.read_csv(DEDUP_CSV)
            df.columns = [c.lower() for c in df.columns]
            if "group_index" in df.columns and "duplicate_of_group_index" in df.columns:
                mapping = {}
                for _, r in df.iterrows():
                    g = str(r["group_index"]).strip()
                    d = str(r["duplicate_of_group_index"]).strip()
                    if d and d.lower() != "nan":
                        mapping[g] = d
                _dedup_map = mapping
            else:
                print("[WARN] 去重表缺少必要列：group_index, duplicate_of_group_index")
                _dedup_map = {}
    return _dedup_map


def resolve_gt_paths(pid: str, GT_IMAGE_DIR: str, GT_SINGLE_STEP_DIR: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    返回: (gt_img_path, gt_single_step_path, gt_full_step_path)
    - single: 原本的“该步的 isolated 形状”（GT_SINGLE_STEP_DIR）
    - full  : “累计到该步的整体形状”（OP_ORIENT_DIR），按多种目录约定自动匹配
    """
    dedup = load_dedup_map()

    group, step = _extract_group_and_step(pid)
    group_base = re.sub(r"/step\d+$", "", group.strip())

    # 去重映射
    if group_base in dedup:
        base_used = dedup[group_base]
        print(f"[INFO] {group_base} 是重复项，使用去重后的 {base_used}")
    else:
        base_used = group_base

    # ========= 单步（原有逻辑） =========
    group_dir = os.path.join(GT_SINGLE_STEP_DIR, base_used.replace("/", os.sep))
    gi_path   = os.path.join(group_dir, "group_info.txt")
    gt_img    = os.path.join(GT_IMAGE_DIR, f"{base_used}/{step}", "3D_isometric.png")

    m = _parse_group_info_txt(gi_path)
    expected = m.get(step, [])
    step_dir = _pick_step_folder(group_dir, step, expected)
    gt_single = os.path.join(step_dir, "3D.step") if step_dir else None

    if gt_single and not os.path.exists(gt_single):
        print(f"[WARN] expected 3D.step not found: {gt_single}")
        gt_single = None
    #TODO：需要根据去重后的 base_used 重定向 gt_img 路径
    # if not os.path.exists(gt_img):
    #     print(f"[WARN] gt image missing: {gt_img}")
    #     gt_img = None
    # ========= 整体（新增逻辑） =========
    op_orient_group_dir = os.path.join(OP_ORIENT_DIR, base_used.replace("/", os.sep))
    gt_full = _pick_full_step_path(op_orient_group_dir, expected)

    if gt_full is None:
        # 兜底打印方便排查
        print(f"[WARN] full-step not found under {op_orient_group_dir} for indices={expected}")

    return gt_img, gt_single, gt_full

#==================== DINO 初始化 =====================

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# _dino_model = None
# _transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.CenterCrop(IMAGE_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
# ])

# def ensure_dino(device="cuda"):
#     global _dino_model
#     if _dino_model is None:
#         _dino_model = Dinov2Model.from_pretrained(DINO_MODEL_ID)
#         _dino_model.eval().to(device)
#     return _dino_model

# @torch.no_grad()
# def dino_cosine(img_path_a, img_path_b, device="cuda"):
#     model = ensure_dino(device)
#     img_a = Image.open(img_path_a).convert("RGB")
#     img_b = Image.open(img_path_b).convert("RGB")
#     ta = _transform(img_a).unsqueeze(0).to(device)
#     tb = _transform(img_b).unsqueeze(0).to(device)
#     fa = model(pixel_values=ta).last_hidden_state[:,0,:]
#     fb = model(pixel_values=tb).last_hidden_state[:,0,:]
#     fa = F.normalize(fa, dim=1); fb = F.normalize(fb, dim=1)
#     return float(F.cosine_similarity(fa, fb).item())

def _load_prev_code_from_dir(group_index: str, base_dir: str) -> str:
    """
    读取前序代码：
    1) 优先在 PRE_CODE_DIR/<group_index 去掉最后. 一段>/ 下找 <group_index 用 _ 连接>.py
       例: group_index = '00002_index_2/step1'
           -> 子目录: PRE_CODE_DIR/00002_index_2
           -> 文件名: 00002_index_2_step1.py
    2) 若不存在，回退到 PRE_CODE_DIR/<文件名> 直接平铺
    3) 仍不存在则返回空串
    """
    try:
        parts = str(group_index).split('/')
        fname = "_".join(parts) + ".py"
        subdir = os.path.join(base_dir, *parts[:-1]) if len(parts) > 1 else base_dir

        path1 = os.path.join(subdir, fname)              # 优先：对应文件夹下
        path2 = os.path.join(base_dir, fname)            # 回退：平铺

        if os.path.exists(path1):
            with open(path1, "r", encoding="utf-8") as f:
                return f.read()
        if os.path.exists(path2):
            with open(path2, "r", encoding="utf-8") as f:
                return f.read()

        print(f"[WARN] prev_code not found for {group_index}: tried {path1} and {path2}")
        return ""
    except Exception as e:
        print(f"[WARN] prev_code error for {group_index}: {e}")
        return ""

# ===================== 执行/几何有效性 =====================
def safe_exec_from_path(py_path: str, globals_dict=None):
    """执行保存到磁盘的 Python/CadQuery 脚本；返回 (ok, locals, err)。"""
    glb = {"cq": cq, "np": np}
    if globals_dict:
        glb.update(globals_dict)
    loc = {}
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
        # 用文件名做 compile，有更准的报错行号/路径
        exec(compile(src, py_path, "exec"), glb, loc)
        return True, loc, ""
    except Exception as e:
        return False, {}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

def geometry_valid(shape_obj):
    """快速几何有效性：非空、体积>0、可算惯性矩"""
    if shape_obj is None:
        return False, {"reason": "result_is_none"}
    try:
        vol = float(cq.Shape.computeMass(shape_obj))
        if vol <= 0:
            return False, {"reason": "non_positive_volume"}
        _ = cq.Shape.matrixOfInertia(shape_obj)
        return True, {"volume": vol}
    except Exception as e:
        return False, {"reason": f"geom_exception:{type(e).__name__}"}

def process_one(r, K, COP, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR):
    _assert_worker_env()  # 保险

    pid = str(r["group_index"])
    m = _G["re_step"].search(pid)
    step_num = int(m.group(1)) if m else -1
    first_step = (step_num == 0)

    gt_img, gt_single_step, gt_full_step = _resolve_gt_paths_cached(
        pid, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR
    )

    base_dir = _G["cop_pre_code_dir"] if COP else _G["pre_code_dir"]
    prev_code = _load_prev_code_from_dir(pid, base_dir)

    prev_path = r.get("prev_code_path", None)
    if isinstance(prev_path, str) and os.path.exists(prev_path):
        try:
            prev_code = open(prev_path, "r", encoding="utf-8").read()
        except Exception as e:
            print(f"[WARN] failed to read prev_code_path for {pid}: {e}")

    prompt = build_incremental_cq_prompt(
        previous_code=prev_code,
        operation_instruction=r["prompt_text"],
        link_mode=None,
        images=None,
        image_prompt=None,
        next_var_name="result",
        allow_comments=False,
        add_size_guidelines=True
    )

    cands = get_model_candidates(prompt, K)

    per_cand_rows, summary_rows = [], []
    tmp_dir = _G["tmp_dir"]
    for k_idx, code in enumerate(cands):
        full_save_path = os.path.join(tmp_dir, pid, "full_path")
        os.makedirs(full_save_path, exist_ok=True)
        full_step_path = os.path.join(full_save_path, f"k{k_idx}_full.step")
        full_py = os.path.join(full_save_path, f"k{k_idx}_full.py")

        integrated_code, final_lhs = build_integrated_code(prev_code, code, full_step_path, first_step=first_step)
        with open(full_py, "w", encoding="utf-8") as f:
            f.write(integrated_code)

        ok_int, loc_int, err2 = safe_exec_from_path(full_py)
        int_exec_ok = 1 if ok_int else 0

        # 解析异常
        exec_err_type = exec_err_msg = ""
        exec_err_line = None
        if not ok_int:
            et, em, el = parse_exec_error(err2)
            exec_err_type, exec_err_msg, exec_err_line = et, em[:300], el  # 截断避免过长

        pred_step_exists = ok_int and os.path.exists(full_step_path)

        if pred_step_exists and gt_full_step:
            res = get_cd_hd(pred_step_path=full_step_path, gt_step_path=gt_full_step)
        else:
            miss = "gt_step_missing" if not gt_full_step else "pred_step_missing"
            res = MetricsResult(None, None, None, ok=False, reason=miss)

        metric_reason = getattr(res, "reason", "")
        err_group = classify_err_3way(
            int_exec_ok, exec_err_type, exec_err_msg, pred_step_exists, metric_reason
        )

        int_reason = ""
        if not ok_int:
            int_reason = f"exec_error:{exec_err_type}"
        if not res.ok and getattr(res, "reason", ""):
            int_reason = (int_reason + "; " if int_reason else "") + res.reason

        row = {
            "group_index": pid,
            "k_index": k_idx,
            "exec_ok": int_exec_ok,
            "pred_step_path": full_step_path if pred_step_exists else "",
            "cd": res.cd,
            "hd": res.hd,
            "pre2gt_angle": getattr(res, "best_euler_angle", None),
            "reason": int_reason if (not res.ok or not ok_int) else "",
            "pred_step_exists": int(pred_step_exists),
            "metric_ok": int(res.ok and (res.cd is not None) and (res.hd is not None)),
            # 新增：细节与三类标签
            "exec_err_type": exec_err_type,
            "exec_err_line": exec_err_line,
            "exec_err_msg": exec_err_msg,
            "err_group": err_group,   # <-- "syntax" | "geom_op" | "other"
        }
        per_cand_rows.append(row)

    # ====== 本样本的 summary 统计 ======
    valid = [r for r in per_cand_rows if r["metric_ok"] == 1]
    best_row = None
    if valid:
        best_row = min(valid, key=lambda r: (r["cd"] if r["cd"] is not None else 1e9) + (r["hd"] if r["hd"] is not None else 1e9))

    n_total = len(per_cand_rows)
    n_exec_ok = sum(r["exec_ok"] == 1 for r in per_cand_rows)
    n_pred_exist = sum(r["pred_step_exists"] == 1 for r in per_cand_rows)
    n_metric_ok = sum(r["metric_ok"] == 1 for r in per_cand_rows)

    cd_mean = float(np.nanmean([r["cd"] for r in valid])) if valid else np.nan
    hd_mean = float(np.nanmean([r["hd"] for r in valid])) if valid else np.nan
    dist = Counter((r.get("err_group") or "other") for r in per_cand_rows if r["exec_ok"] == 0 or not r["pred_step_exists"])

    # 按三类统计本样本分布
    dist = Counter((r.get("err_group") or "other") for r in per_cand_rows if r["exec_ok"] == 0 or not r["pred_step_exists"])
    summary_rows.append({
        "group_index": pid,
        "k_index": "summary_err_groups",
        "exec_ok": None,
        "pred_step_path": "",
        "cd": np.nan,
        "hd": np.nan,
        "pre2gt_angle": None,
        "reason": "",
        "n_total": n_total,
        "n_exec_ok": n_exec_ok,
        "n_pred_exist": n_pred_exist,
        "n_metric_ok": n_metric_ok,
        "err_groups_json": json.dumps(dist, ensure_ascii=False, sort_keys=True)  # 例如 {"geom_op":2,"syntax":1}
    })
    summary_rows.append({
        "group_index": pid,
        "k_index": "summary_best",
        "exec_ok": None,
        "pred_step_path": (best_row or {}).get("pred_step_path", ""),
        "cd": (best_row or {}).get("cd", np.nan),
        "hd": (best_row or {}).get("hd", np.nan),
        "pre2gt_angle": (best_row or {}).get("pre2gt_angle", None),
        "reason": "",
        "n_total": n_total,
        "n_exec_ok": n_exec_ok,
        "n_pred_exist": n_pred_exist,
        "n_metric_ok": n_metric_ok,
    })
    summary_rows.append({
        "group_index": pid,
        "k_index": "summary_mean_valid",
        "exec_ok": None,
        "pred_step_path": "",
        "cd": cd_mean,
        "hd": hd_mean,
        "pre2gt_angle": None,
        "reason": "",
        "n_total": n_total,
        "n_exec_ok": n_exec_ok,
        "n_pred_exist": n_pred_exist,
        "n_metric_ok": n_metric_ok,
    })
    return per_cand_rows, summary_rows, pid


def main_parallel():
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_dir(TMP_DIR)

    cand_out_path = os.path.join(OUT_DIR, "cands.csv")
    summary_out_path = os.path.join(OUT_DIR, "summary.csv")

    df = pd.read_csv(PROMPTS_CSV)
    df.columns = [c.lower() for c in df.columns]
    required = {"group_index", "prompt_text"}
    miss = required - set(df.columns)
    if miss:
        raise KeyError(f"prompts.csv 缺少列: {miss}")

    # RESUME：从 summary.csv 读取已完成样本
    done_group_indexs = set()
    if RESUME and os.path.exists(summary_out_path):
        old = pd.read_csv(summary_out_path)
        if {"group_index", "k_index"}.issubset(old.columns):
            done_group_indexs = set(old[old["k_index"] == "summary_best"]["group_index"].astype(str))

    # 过滤掉已完成的样本
    pend_rows = [r for _, r in df.iterrows() if str(r["group_index"]) not in done_group_indexs]
    print(f"[INFO] total={len(df)}, resume-skip={len(done_group_indexs)}, to-run={len(pend_rows)}")

    # 结果缓冲区
    all_cand_rows, all_summary_rows = [], []
    buffer_count = 0

    # 用 imap_unordered 边取边落地（可中断恢复）
    ctx = mp.get_context("spawn")  # CUDA/稳定性更好
    with ctx.Pool(
        processes=NPROC,
        maxtasksperchild=1,
        initializer=init_worker,
        initargs=(
            DEVICE, DINO_MODEL_ID, IMAGE_SIZE,
            DEDUP_CSV, OP_ORIENT_DIR,
            PRE_CODE_DIR, COP_PRE_CODE_DIR,
            TMP_DIR,
        ),
    ) as pool:
        worker = partial(process_one, K=K, COP=COP,
                        GT_IMAGE_DIR=GT_IMAGE_DIR, GT_SINGLE_STEP_DIR=GT_SINGLE_STEP_DIR)
        for per_cand_rows, summary_rows, pid in pool.imap_unordered(worker, pend_rows, chunksize=8):
            all_cand_rows.extend(per_cand_rows)
            all_summary_rows.extend(summary_rows)
            buffer_count += 1

            # # 每处理 WRITE_EVERY 个样本，就写一次
            # if buffer_count % WRITE_EVERY == 0:
            #     pd.DataFrame(all_cand_rows).to_csv(cand_out_path, index=False)
            #     pd.DataFrame(all_summary_rows).to_csv(summary_out_path, index=False)
            #     print(f"[flush] wrote {len(all_summary_rows)} summary rows, {len(all_cand_rows)} cand rows")
            if buffer_count % WRITE_EVERY == 0:
                atomic_write_df(pd.DataFrame(all_cand_rows), cand_out_path)
                atomic_write_df(pd.DataFrame(all_summary_rows), summary_out_path)
                print(f"[flush] wrote {len(all_summary_rows)} summary rows, {len(all_cand_rows)} cand rows")

    # 最后一批写盘
    atomic_write_df(pd.DataFrame(all_cand_rows), cand_out_path)
    atomic_write_df(pd.DataFrame(all_summary_rows), summary_out_path)
    print("[done] all results saved.")

    final_txt = os.path.join(OUT_DIR, "final_stats.txt")
    write_final_stats(cand_out_path, summary_out_path, final_txt, k=K)

if __name__ == "__main__":
    # 解析参数
    parser = build_arg_parser()
    args = parser.parse_args()

    # 线程库限核（OpenBLAS/MKL）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 应用参数到全局
    apply_args(args)

    # 跑
    main_parallel()
