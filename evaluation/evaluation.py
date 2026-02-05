# ---- 新增：多进程所需 ----
import multiprocessing as mp
from functools import partial
import os, traceback, json, time
import pandas as pd
import numpy as np
import cadquery as cq
from model_call import call_model as cm
import os, re, ast
from typing import Optional, Tuple, List
# Linux 下建议显式设置以避免 OpenBLAS 抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ==== argparse: 统一配置入口 ====
import argparse
from datetime import datetime
_dedup_map = None
THINKING = True   # === NEW: thinking ===
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
    p.add_argument("--split-json", default=None,
                   help="数据划分JSON（包含 train/val/test 等列表），只评测 split-key 对应列表中的 group_index")
    p.add_argument("--split-key", default="test",
                   help="split-json 中的键名（默认 test）")

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
    p.add_argument("--gt-edges-dir", required=True, help="GT 边目录（之前抽取的 JSON 根目录，例如 /home/.../gt_edges_json）")

    # 评测与渲染参数
    p.add_argument("--k", type=int, default=2, help="pass@k 的 k")
    p.add_argument("--pass-metric", choices=["cosine","iou"], default="cosine", help="主评价指标（若使用图像相似）")
    p.add_argument("--cos-threshold", type=float, default=0.90, help="cosine 阈值（仅在 cosine 模式下有效）")
    p.add_argument("--image-size", type=int, default=518, help="DINO 输入尺寸")
    p.add_argument("--fivecrop", action="store_true", help="若你在 render_to_png 做了多视角，可关闭这里；否则可开")

    # 多进程
    p.add_argument("--nproc", type=int, default=128, help="进程数")
    p.add_argument("--write-every", type=int, default=1, help="每处理多少个样本落地一次")

    # 产物保存开关
    p.add_argument("--save-step", action="store_true", default=True, help="保存中间 step（isolated）")
    p.add_argument("--no-save-step", dest="save_step", action="store_false", help="不保存中间 step")
    p.add_argument("--save-render", action="store_true", default=True, help="保存渲染 png")
    p.add_argument("--no-save-render", dest="save_render", action="store_false", help="不保存渲染 png")
    # === 新增：是否写 summary ===
    p.add_argument("--write-summary", action="store_true", default=True, help="写入 summary.csv（默认开）")
    p.add_argument("--no-summary", dest="write_summary", action="store_false", help="不写 summary.csv（同时忽略基于 summary 的 resume）")

    # DINO
    p.add_argument("--dino-model-id", default="facebook/dinov2-base",
                   help="DINOv2 模型ID（如 facebook/dinov2-large）")
    p.add_argument(
        "--repair-csv", default=None,
        help="修正模式：逐行读取该CSV（必须含 group_index；可选 k_index、prev_code_path），"
             "若同一 pid 同时含 k_index=0 和 1，则用 K=2 生成并覆盖两个槽"
    )

    # ==== 模型配置覆盖 ====
    p.add_argument("--gen-mode", choices=["local","api","auto"], default=None,
               help="生成模式覆盖：local/api/auto（覆盖 config.json）")
    p.add_argument("--provider", choices=["openai","http","local","siliconflow","vllm"], default=None,
                help="指定API提供方或本地：openai/http/local（覆盖 config.json 的 enabled）")
    p.add_argument("--openai-model", default=None, help="覆盖 OpenAI 模型名，如 gpt-4o")
    p.add_argument("--http-model", default=None, help="覆盖 HTTP 模型名，如 gpt-4o-mini")
    p.add_argument("--gen-temperature", type=float, default=None, help="覆盖生成温度")
    p.add_argument("--gen-timeout", type=int, default=None, help="覆盖生成超时秒数")
    p.add_argument("--vllm-endpoint-key", default="port1",
                help="当 provider=vllm 时，指定使用 config.json 中 'vllm.endpoints' 下的哪个 key (默认按 config.json 的 strategy 选)")
    p.add_argument("--thinking", action="store_true", default=False,
                help="启用 thinking 模式：请求模型返回含思维/轨迹的答案，并在输出目录打标签")
    
    # === One-shot few-shot 开关 ===
    p.add_argument("--oneshot", action="store_true", default=False,
                   help="启用 one-shot few-shot 示例拼接到 Prompt 中")
    p.add_argument("--oneshot-csv", default=None,
                   help="one_shot.csv 路径（需含: group_index, picked_as, 可选 answer）")
    p.add_argument("--meta-csv", default="/home/baiyixue/project/op-cad/data/data_indication_out.csv",
                   help="含 final_score/op 的 meta 表，用于计算 bin2")
    p.add_argument("--bool-csv", default="/home/baiyixue/project/data_render/data/op_orientation/grouped_op_pairs_bool.csv",
                   help="bool_op 表(1=add, -1=cut)")

    
    return p

def apply_args(args):
    global PROMPTS_CSV, OUT_DIR, PRE_CODE_DIR, COP_PRE_CODE_DIR, GT_IMAGE_DIR, DEDUP_CSV, GT_EDGES_DIR
    global GT_SINGLE_STEP_DIR, OP_ORIENT_DIR, K, DEVICE, PASS_METRIC, COS_THRESHOLD
    global IMAGE_SIZE, FIVECROP, COP, SAVE_STEP, SAVE_RENDER, TMP_DIR, RESUME
    global SEED, DINO_MODEL_ID, NPROC, WRITE_EVERY
    global WRITE_SUMMARY
    global THINKING
    global ONESHOT_ON, ONESHOT_CSV, META_CSV, BOOL_CSV

    ONESHOT_ON  = bool(getattr(args, "oneshot", False))
    ONESHOT_CSV = getattr(args, "oneshot_csv", None)
    META_CSV    = getattr(args, "meta_csv", None)
    BOOL_CSV    = getattr(args, "bool_csv", None)

    # ===== 运行标识 & 目录结构 =====
    # 目录：<out-root>/<test-name>__<mode>/（附加时间戳避免覆盖，可按需去掉）
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm.set_runtime_config(
    gen_mode=args.gen_mode,
    provider=args.provider,
    vllm_endpoint_key=getattr(args, "vllm_endpoint_key", None), # <-- 新增这行
    openai_model=args.openai_model,
    http_model=args.http_model,
    temperature=args.gen_temperature,
    timeout_s=args.gen_timeout
    )

    # 如果 test_name 没传，这里再用“最新的”模型名作为默认
    if not args.test_name:
        args.test_name = str(cm.MODEL).replace("/", "_")

    mode_tag = args.mode  # "std" or "cop"
    # mode_tag_with_think = mode_tag + ("_thinking" if args.thinking else "")
    run_dir = os.path.join(args.out_root, args.test_name, mode_tag)
    os.makedirs(run_dir, exist_ok=True)
    if "Qwen3-8B" in args.test_name:
        THINKING = bool(args.thinking)
    else:
        THINKING = True

    PROMPTS_CSV = args.prompts_csv
    OUT_DIR = run_dir

    WRITE_SUMMARY = args.write_summary

    PRE_CODE_DIR = args.pre_code_dir
    COP_PRE_CODE_DIR = args.cop_pre_code_dir
    GT_IMAGE_DIR = args.gt_image_dir
    DEDUP_CSV = args.dedup_csv
    GT_EDGES_DIR = args.gt_edges_dir

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
    TMP_DIR = os.path.join(run_dir, "middle_step")

    RESUME = args.resume
    SEED = args.seed
    DINO_MODEL_ID = args.dino_model_id

    NPROC = args.nproc
    WRITE_EVERY = args.write_every

    # 便于在日志里检索
    print(f"[RUN] test={args.test_name}  mode={args.mode}"
         f"  thinking={THINKING}  out_dir={OUT_DIR}")
    print(f"[RUN] prompts={PROMPTS_CSV}  nproc={NPROC}  resume={RESUME}  seed={SEED}")




# ===================== 工具初始化 =====================

# ===== One-shot 相关 =====
_meta_map = None      # pid -> {op, final_score, bin2}
_bool_map = None      # pid -> {bool_op}
_oneshot_tbl = None   # picked_as -> row(dict)
_bin2_thr = None      # 全局中位数

def _load_meta_map(path: str):
    """加载 meta: {pid: {op, final_score, bin2(自动)}} & 全局中位数阈值"""
    global _meta_map, _bin2_thr
    if _meta_map is not None:
        return _meta_map
    df = pd.read_csv(path)
    df["group_index"] = df["group_index"].astype(str)
    # 计算二档（若无现成 bin2 列）
    if "bin2" in df.columns:
        df["bin2"] = df["bin2"].astype(str).str.lower()
        # 若 bin2 有空，仍用中位数补
        if df["bin2"].isna().any():
            m = df["final_score"].astype(float)
            _bin2_thr = float(m.median())
            df.loc[df["bin2"].isna(), "bin2"] = np.where(m <= _bin2_thr, "low", "high")
    else:
        m = df["final_score"].astype(float)
        _bin2_thr = float(m.median())
        df["bin2"] = np.where(m <= _bin2_thr, "low", "high")
    _meta_map = df.set_index("group_index")[["op","final_score","bin2"]].to_dict(orient="index")
    return _meta_map

def _load_bool_map(path: str):
    """加载 bool_op: {pid: {bool_op}}；无则返回空"""
    global _bool_map
    if _bool_map is not None:
        return _bool_map
    try:
        d = pd.read_csv(path)
        key = "group_index" if "group_index" in d.columns else ("pid" if "pid" in d.columns else None)
        if not key or "bool_op" not in d.columns:
            _bool_map = {}
            return _bool_map
        d[key] = d[key].astype(str)
        _bool_map = d.set_index(key)[["bool_op"]].to_dict(orient="index")
        return _bool_map
    except Exception:
        _bool_map = {}
        return _bool_map

def _load_oneshot_tbl(path: str):
    """加载 one_shot.csv，按 picked_as 建索引"""
    global _oneshot_tbl
    if _oneshot_tbl is not None:
        return _oneshot_tbl
    df = pd.read_csv(path)
    df["group_index"] = df["group_index"].astype(str)
    if "picked_as" not in df.columns:
        raise KeyError("one_shot.csv 需要列 picked_as")
    _oneshot_tbl = df.groupby("picked_as").apply(lambda x: x.iloc[0]).to_dict(orient="index")
    return _oneshot_tbl

def _classify_key_for_pid(pid: str, meta_map: dict, bool_map: dict) -> Optional[str]:
    """给当前 pid 产出分类键：extrude_add_low / extrude_step0_high / chamfer_fillet_low 等"""
    if pid not in meta_map:
        return None
    info = meta_map[pid]
    op = str(info["op"]).strip().lower()
    bin2 = str(info.get("bin2","low")).strip().lower()
    is_step0 = str(pid).endswith("/step0")
    if op in ("extrude", "revolve"):
        if is_step0:
            return f"{op}_step0_{bin2}"
        bo = (bool_map.get(pid, {}) or {}).get("bool_op", None)
        if bo == 1:
            return f"{op}_add_{bin2}"
        elif bo == -1:
            return f"{op}_cut_{bin2}"
        else:
            # 无 bool_op 时，退化为 op_bin2（可按需改更激进的回退）
            return f"{op}_add_{bin2}"
    elif op == "chamfer_fillet":
        return f"chamfer_fillet_{bin2}"
    else:
        return None

def _build_few_shot_for_pid(pid: str, pmap: dict, cop_mode: bool) -> Optional[dict]:
    """返回一个 few-shot 示例 dict: {label, prev_code, instruction, answer}"""
    if not ONESHOT_ON or not ONESHOT_CSV or not META_CSV:
        return None
    meta_map = _load_meta_map(META_CSV)
    bool_map = _load_bool_map(BOOL_CSV) if BOOL_CSV else {}
    key = _classify_key_for_pid(pid, meta_map, bool_map)
    if not key:
        return None
    tbl = _load_oneshot_tbl(ONESHOT_CSV)
    row = tbl.get(key)
    if not row:
        return None
    ex_pid = str(row["group_index"])
    if ex_pid == pid:
        return None  # 避免拿当前样本做示例
    # 取示例 prev_code 与 instruction
    prev_code = _load_prev_code_from_dir(ex_pid, COP_PRE_CODE_DIR if cop_mode else PRE_CODE_DIR)
    ex_instr = (pmap.get(ex_pid, {}) or {}).get("prompt_text", "")
    ex_ans = str(row.get("answer","") or "")
    return {"label": key, "prev_code": prev_code, "instruction": ex_instr, "answer": ex_ans}

def _normalize_validity(row: dict) -> dict:
    """
    规则：
    A. “有效”以是否成功产出几何指标为准：exec_ok_* = metric_ok_*
    B. 兜底：
       - reason_single == 'pred_step_missing'（忽略前后空格/大小写）→ 两个 exec 置 0
       - reason_single 含 'metric_exception:RuntimeError:Empty mesh'（忽略大小写）→ 两个 exec 置 0
    """
    # ---- A. 用 metric_ok_* 覆盖 exec_ok_* ----
    if "metric_ok_single" in row:
        row["exec_ok_single"] = int(bool(row.get("metric_ok_single", 0)))
    if "metric_ok_full" in row:
        row["exec_ok_full"] = int(bool(row.get("metric_ok_full", 0)))

    # ---- B. 兜底规则（基于 reason_single）----
    rs = (row.get("reason_single", "") or "").strip().lower()

    # 精确等于 pred_step_missing
    if rs == "pred_step_missing":
        if "exec_ok_single" in row: row["exec_ok_single"] = 0
        if "exec_ok_full"   in row: row["exec_ok_full"]   = 0
        return row

    # 含 metric_exception:RuntimeError:Empty mesh
    if "metric_exception:runtimeerror:empty mesh" in rs:
        if "exec_ok_single" in row: row["exec_ok_single"] = 0
        if "exec_ok_full"   in row: row["exec_ok_full"]   = 0
        return row

    return row

def _compute_cf_iou_metrics(pred_f, pred_c, gt_f, gt_c):
    """
    仅计算并返回 7 个字段：
    - cf_num_pred_fillet, cf_num_pred_chamfer
    - cf_num_gt_fillet,   cf_num_gt_chamfer
    - cf_hits_fillet,     cf_hits_chamfer
    - cf_iou  （= (hits_f + hits_c) / (|pred_f ∪ gt_f| + |pred_c ∪ gt_c|)）
    """
    # 去重
    pred_f = _dedup_edges(pred_f); pred_c = _dedup_edges(pred_c)
    gt_f   = _dedup_edges(gt_f);   gt_c   = _dedup_edges(gt_c)

    # 逐类匹配
    m_f, pred_fu, gt_fu = _match_edges(pred_f, gt_f)
    m_c, pred_cu, gt_cu = _match_edges(pred_c, gt_c)

    hits_f = len(m_f)
    hits_c = len(m_c)

    # ∪ 的基数：|A ∪ B| = |A| + |B| - |A ∩ B|；这里用匹配数近似 |A ∩ B|
    union_f = len(pred_f) + len(gt_f) - hits_f
    union_c = len(pred_c) + len(gt_c) - hits_c
    union_all = max(union_f + union_c, 0)

    iou = (hits_f + hits_c) / union_all if union_all > 0 else 1.0  # 双空集约定得 1.0

    return {
        "cf_num_pred_fillet": len(pred_f),
        "cf_num_pred_chamfer": len(pred_c),
        "cf_num_gt_fillet": len(gt_f),
        "cf_num_gt_chamfer": len(gt_c),
        "cf_hits_fillet": hits_f,
        "cf_hits_chamfer": hits_c,
        "cf_iou": iou,
    }


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



def _pick_single_step_path(group_dir: str, indices: List[int]) -> Optional[str]:
    """
    在 <group_dir> 下，为“单步（isolated）形状”挑选 3D.step 路径。
    规则：
    - 若 indices 只有一个数 i -> 期望目录名：step{i}/3D.step
    - 若 indices 含多个数 [i,j,k,...] -> 期望目录名：step{i}_{j}_{k}_.../3D.step
      （兼容常见变体：stepi-j-k、stepi,j,k）
    - 若严格命名不存在，则在 <group_dir> 内搜索所有以 'step{i}' 开头的目录，
      取“包含 indices 集合”的最佳匹配（优先集合完全相等，其次最小超集，其次最短集合）。

    返回：命中的 3D.step 完整路径；若找不到返回 None
    """
    if not os.path.isdir(group_dir) or not indices:
        return None

    # -------- 1) 构造“理想命名”的候选路径（优先尝试） --------
    def _combo_candidates(idxs: List[int]) -> List[str]:
        s = [str(x) for x in idxs]
        if len(idxs) == 1:
            # 单个：step{i}
            return [f"step{s[0]}"]
        # 多个：stepi_j_k... （并兼容 '-' / ','）
        return [f"step{'_'.join(s)}", f"step{'-'.join(s)}", f"step{','.join(s)}"]

    ideal_dirs = _combo_candidates(indices)
    for d in ideal_dirs:
        p = os.path.join(group_dir, d, "3D.step")
        if os.path.exists(p):
            return p
    return None

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

def _load_prompts_map():
    """返回 {pid: {prompt_text:..., op:..., prev_code_path:...}}"""
    dfp = pd.read_csv(PROMPTS_CSV)
    dfp.columns = [c.lower() for c in dfp.columns]
    need = {"group_index", "prompt_text", "op"}
    if not need.issubset(dfp.columns):
        raise KeyError(f"prompts.csv 需要列：{need}")
    # 防空&去重
    dfp = dfp.dropna(subset=["group_index"]).copy()
    dfp["group_index"] = dfp["group_index"].astype(str).str.strip()
    dfp = dfp.drop_duplicates(subset=["group_index"], keep="last")
    return dfp.set_index("group_index").to_dict(orient="index")


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
   
    assert OP_ORIENT_DIR, "OP_ORIENT_DIR not set. Call apply_args() first."
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
        # print(f"[INFO] {group_base} 是重复项，使用去重后的 {base_used}")
    else:
        base_used = group_base

    # ========= 单步（原有逻辑） =========
    group_dir = os.path.join(GT_SINGLE_STEP_DIR, base_used.replace("/", os.sep))
    gi_path   = os.path.join(group_dir, "group_info.txt")
    gt_img    = os.path.join(GT_IMAGE_DIR, f"{base_used}/{step}", "3D_isometric.png")

    m = _parse_group_info_txt(gi_path)
    expected = m.get(step, [])
    gt_single = _pick_single_step_path(group_dir, expected)

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
        # ---- step0 直接返回 ----
        if re.search(r"/step0$", group_index):
            return ""
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
# ========= chamfer/fillet 评测：加载GT、解析生成代码、匹配 =========
import math
from scipy.optimize import linear_sum_assignment  # pip install scipy

TAU = 2 * math.pi

def _radius_est(edge: dict):
    if (edge.get("geomType") or "").upper() == "CIRCLE":
        L = edge.get("length")
        if L and L > 0:
            return L / TAU
    return None

def _edge_feat(edge: dict):
    center = np.array(edge.get("center") or [np.nan, np.nan, np.nan], dtype=float)
    L = edge.get("length")
    r = _radius_est(edge)
    t = (edge.get("geomType") or "").upper()
    verts = np.array(edge.get("vertices") or [], dtype=float)
    v_mean = np.mean(verts, axis=0) if len(verts) else center  # 没有就退回 center
    return center, L, r, t, v_mean

def _edge_cost(e_pred, e_gt, w_c=1.0, w_l=5.0, w_r=2.0, w_v=0.5):
    c_p, L_p, r_p, t_p, v_p = _edge_feat(e_pred)
    c_g, L_g, r_g, t_g, v_g = _edge_feat(e_gt)
    if t_p != t_g:
        return 1e9
    if (np.any(np.isnan(c_p)) or np.any(np.isnan(c_g)) or 
        np.any(np.isnan(v_p)) or np.any(np.isnan(v_g))):
        return 1e9

    dc = np.linalg.norm(c_p - c_g)               # center 距离
    dv = np.linalg.norm(v_p - v_g)               # 顶点均值距离
    if (L_p is None) or (L_g is None):
        return 1e9
    dl = abs(L_p - L_g)
    rel_len_err = dl / max(L_g, 1e-9)

    dr = 0.0
    if t_p == "CIRCLE":
        if (r_p is None) or (r_g is None):
            return 1e9
        dr = abs(r_p - r_g)

    # 软阈值（保守）：任一超过阈值直接判大成本
    if dc > 1e-3:                                   # 位置
        return 1e9
    if dv > 1e-3:                                   # 顶点均值
        return 1e9
    if (dl > 1e-2) and (rel_len_err > 0.02):        # 长度
        return 1e9
    if t_p == "CIRCLE" and dr > 2e-3:               # 半径
        return 1e-9 + w_r*dr + w_c*dc + w_l*dl + w_v*dv

    return w_c*dc + w_v*dv + w_l*dl + w_r*dr


def _match_edges(pred_edges: list, gt_edges: list):
    """返回 (matches, pred_unmatched_idx, gt_unmatched_idx)，matches 元素为 (i,j,cost)"""
    if (not pred_edges) or (not gt_edges):
        return [], list(range(len(pred_edges))), list(range(len(gt_edges)))
    n, m = len(pred_edges), len(gt_edges)
    C = np.full((n, m), 1e9, dtype=float)
    for i, ep in enumerate(pred_edges):
        for j, eg in enumerate(gt_edges):
            C[i, j] = _edge_cost(ep, eg)
    row_ind, col_ind = linear_sum_assignment(C)
    matches, pred_un, gt_un = [], set(range(n)), set(range(m))
    for i, j in zip(row_ind, col_ind):
        if C[i, j] < 1e8:
            matches.append((i, j, float(C[i, j])))
            pred_un.discard(i)
            gt_un.discard(j)
    return matches, sorted(pred_un), sorted(gt_un)

def _dedup_edges(edges: list, tol_center=1e-6, tol_len=1e-4):
    """简单去重：按 (geomType, center 四舍五入, length 四舍五入) 作为 key"""
    seen, out = set(), []
    for e in edges:
        t = (e.get("geomType") or "").upper()
        c = e.get("center") or [0,0,0]
        L = e.get("length") or 0.0
        key = (t,
               round(c[0] if c else 0.0, int(abs(math.log10(tol_center)))),
               round(c[1] if c else 0.0, int(abs(math.log10(tol_center)))),
               round(c[2] if c else 0.0, int(abs(math.log10(tol_center)))),
               round(L, int(abs(math.log10(tol_len)))))
        if key not in seen:
            seen.add(key); out.append(e)
    return out

def _load_gt_edges_for_pid(gt_dir: str, pid: str):
    """pid: 'xxxx_index_y/stepk' -> 聚合 GT 中该步所有 fillet/chamfer 边"""
    base, step = _extract_group_and_step(pid)  # base='xxxx_index_y', step='stepk'
    step_dir = os.path.join(gt_dir, base, step)  # 我们的抽取器写的是 step{k} 目录
    fillet_edges, chamfer_edges = [], []
    if not os.path.isdir(step_dir):
        return fillet_edges, chamfer_edges
    for fn in os.listdir(step_dir):
        if not fn.endswith(".json"): continue
        op_tag = os.path.splitext(fn)[0]  # Fillet_3 / Chamfer_4
        full = os.path.join(step_dir, fn)
        try:
            data = json.load(open(full, "r", encoding="utf-8"))
        except Exception:
            continue
        # 可能是单对象，也可能包含 'edges' 数组
        if isinstance(data, dict):
            edges = data.get("edges", [])
            if op_tag.lower().startswith("fillet"):
                fillet_edges.extend(edges)
            elif op_tag.lower().startswith("chamfer"):
                chamfer_edges.extend(edges)
    return fillet_edges, chamfer_edges

# ---------- 解析输出代码中的多块 "edges select → chamfer/fillet" ----------
_EDGES_HEADER = re.compile(r"^\s*#\s*edges\s+select\s*$", re.IGNORECASE)
_OP_HEADER    = re.compile(r"^\s*#\s*operation(?:\s+(?P<kind>fillet|chamfer))?\s*$", re.IGNORECASE)
_OP_HEADER_LEGACY = re.compile(r"^\s*#\s*chamfer/fillet\s*$", re.IGNORECASE)
_ASSIGN_EDGES = re.compile(r"^\s*(?P<lhs>\w+)\s*=\s*(?P<rhs>.+?\.edges\s*\(.*\))\s*$", re.IGNORECASE)
_APPLY_LINE   = re.compile(r".*?\b(?P<var>\w+)\.(?P<op>fillet|chamfer)\s*\((?P<args>[^)]*)\)\s*$", re.IGNORECASE)

def _parse_modify_blocks(gen_code: str):
    """
    返回 list[ { 'select_lines': [...], 'apply_line': '...', 'kind': 'fillet|chamfer' } ]
    基于约定的输出格式：
      #edges select
      <one or more selection lines>
      #operation [fillet|chamfer]   # 新格式（可省略类型）
      <one operation line>
    兼容旧格式：
      #edges select
      ...
      #chamfer/fillet
      ...
    """
    blocks = []
    lines = gen_code.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if _EDGES_HEADER.match(lines[i] or ""):
            i += 1
            select_lines = []
            # 收集 selection 段，直到遇到 operation 标头（新/旧任一）
            while i < n and not (_OP_HEADER.match(lines[i] or "") or _OP_HEADER_LEGACY.match(lines[i] or "")):
                if lines[i].strip():
                    select_lines.append(lines[i])
                i += 1

            # 读取 operation 标头（新优先）
            hdr_m = None
            if i < n:
                hdr_m = _OP_HEADER.match(lines[i] or "") or _OP_HEADER_LEGACY.match(lines[i] or "")
                if hdr_m:
                    i += 1

            # 下一行应为 apply（取第一行非空）
            apply_line = ""
            while i < n:
                L = lines[i].strip()
                if not L:
                    i += 1
                    continue
                apply_line = lines[i]
                i += 1
                break

            # 决定 kind：先看 operation 标头是否携带类型，再从 apply 行回退推断
            kind = None
            if hdr_m and hdr_m.re is _OP_HEADER:
                try:
                    kind = hdr_m.group("kind")
                except IndexError:
                    kind = None
            if not kind:
                m = _APPLY_LINE.match(apply_line or "")
                if m:
                    kind = m.group("op").lower()

            if select_lines and kind in ("fillet", "chamfer"):
                blocks.append({"select_lines": select_lines, "apply_line": apply_line, "kind": kind})
        else:
            i += 1
    return blocks

# def _eval_pred_edges_from_blocks(prev_code: str, gen_code: str):
#     """
#     执行 prev_code + 每个 selection RHS，得到 predicted 边集合（按 kind 分桶）。
#     不执行 fillet/chamfer，只 eval 边选择表达式。
#     """
#     fillet_pred, chamfer_pred = [], []
#     # 准备执行环境
#     glb = {"cq": cq, "np": np}
#     loc = {}
#     # 先执行 previous_code（已有 result_*）
#     try:
#         exec(prev_code, glb, loc)
#     except Exception as e:
#         return fillet_pred, chamfer_pred, f"prev_exec_error:{e}"
#     blocks = _parse_modify_blocks(gen_code)
#     # 逐块处理
#     for b in blocks:
#         # 逐行执行 selection 之前的赋值以定义变量（但只 eval RHS）
#         for line in b["select_lines"]:
#             m = _ASSIGN_EDGES.match(line.strip())
#             if not m:
#                 # 允许一些准备行（若有），直接 exec
#                 try:
#                     exec(line, glb, loc)
#                 except Exception:
#                     pass
#                 continue
#             lhs, rhs = m.group("lhs"), m.group("rhs")
#             try:
#                 sel = eval(rhs, glb, loc)   # result_x.edges(...)
#                 vals = list(sel.vals())
#                 # 把 Edge 转 dict
#                 vec = []
#                 for i, e in enumerate(vals):
#                     try:
#                         center = e.Center(); length = e.Length(); g = e.geomType()
#                         verts = [(float(v.X), float(v.Y), float(v.Z)) for v in e.Vertices()]
#                         vec.append({"edge_index": i, "length": float(length),
#                                     "center": (float(center.x), float(center.y), float(center.z)),
#                                     "geomType": g, "vertices": verts})
#                     except Exception:
#                         continue
#                 if b["kind"] == "fillet":
#                     fillet_pred.extend(vec)
#                 else:
#                     chamfer_pred.extend(vec)
#                 # 也把选择结果放到上下文，以防后续代码引用
#                 loc[lhs] = sel
#             except Exception:
#                 continue
#     return fillet_pred, chamfer_pred, ""

def _eval_pred_edges_from_blocks(prev_code: str, gen_code: str):
    """
    新版本：
    - 不再依赖 #edges select / #operation 注释；
    - 直接在 gen_code 中查找所有 edge_var = xxx.edges(...)
    - 再根据 edge_var 后续是否被 .fillet() / .chamfer() 使用，决定归入哪一类。
    """
    fillet_pred, chamfer_pred = [], []

    # ---------- 1. 执行前序代码，构建 result_* 场景 ----------
    glb = {"cq": cq, "np": np}
    loc = {}
    try:
        exec(prev_code, glb, loc)
    except Exception as e:
        return fillet_pred, chamfer_pred, f"prev_exec_error:{e}"

    lines = gen_code.splitlines()

    # ---------- 2. 先扫一遍，记录 edge_var 对应的 op 类型 ----------
    # 例如： shape_1 = edge_1.chamfer(0.25)  → edge_1 -> "chamfer"
    edge_op_map = {}   # edge_var -> "fillet" / "chamfer"
    for line in lines:
        m_apply = _APPLY_LINE.match(line.strip() if line else "")
        if not m_apply:
            continue
        var = m_apply.group("var")
        op  = m_apply.group("op").lower()   # fillet / chamfer
        if op in ("fillet", "chamfer"):
            edge_op_map[var] = op

    # ---------- 3. 再扫一遍，执行所有 edge_var = xxx.edges(...) ----------
    for line in lines:
        m_sel = _ASSIGN_EDGES.match(line.strip() if line else "")
        if not m_sel:
            continue

        lhs = m_sel.group("lhs")   # edge_1 / edges_to_modify 之类
        rhs = m_sel.group("rhs")   # result_2.edges(...)

        try:
            sel = eval(rhs, glb, loc)     # result_2.edges(...)
            vals = list(sel.vals())
        except Exception:
            # 某一行解析失败就跳过，不影响其他行
            continue

        vec = []
        for i, e in enumerate(vals):
            try:
                center = e.Center()
                length = e.Length()
                g = e.geomType()
                verts = [(float(v.X), float(v.Y), float(v.Z)) for v in e.Vertices()]
                vec.append({
                    "edge_index": i,
                    "length": float(length),
                    "center": (float(center.x), float(center.y), float(center.z)),
                    "geomType": g,
                    "vertices": verts,
                })
            except Exception:
                continue

        # 根据 edge_var 后面被什么 op 使用，决定丢到哪个 bucket
        op_kind = edge_op_map.get(lhs, None)

        if op_kind == "chamfer":
            chamfer_pred.extend(vec)
        elif op_kind == "fillet":
            fillet_pred.extend(vec)
        else:
            # 如果这个 edge_var 没有任何 chamfer/fillet 调用记录，
            # 可以选择：
            #   1) 忽略（continue），或者
            #   2) 默认归入某一类，这里我选默认归到 fillet，也可以改成跳过。
            fillet_pred.extend(vec)

    return fillet_pred, chamfer_pred, ""

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
        exec(compile(src, py_path, "exec"), glb, glb)
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

def _compute_summary(rows: List[dict], pid: str, op_kind: str) -> dict:
    """
    汇总单个 pid 的评测结果：
    - 对 Chamfer/Fillet 操作：仅统计 CF 指标。
    - 对几何操作（Extrude/Revolve/Cut 等）：统计 cd/hd 指标。
    """
    import numpy as np

    # 判断是否 CF 操作（Chamfer / Fillet）
    if op_kind == "chamfer_fillet":
        cf_rows = [r for r in rows if "cf_iou" in r]
        cf_iou_mean = np.nanmean([r.get("cf_iou", np.nan) for r in cf_rows]) if cf_rows else np.nan
        # 也可加一个覆盖度：至少有一条预测或GT
        return {
            "group_index": pid,
            "k_index": "summary",
            "op_type": "Chamfer/Fillet",
            "cf_iou_mean": cf_iou_mean,
            "n_total": len(rows),
        }

    # ======= 否则走原几何汇总逻辑 =======
    def _best_and_mean(rows, cd_key, hd_key, ok_key):
        valid = [r for r in rows if r.get(ok_key) == 1 and
                 (r.get(cd_key) is not None) and (r.get(hd_key) is not None)]
        if valid:
            best_row = min(valid, key=lambda r: (r.get(cd_key, 1e9) or 1e9) + (r.get(hd_key, 1e9) or 1e9))
            cd_mean = float(np.nanmean([r[cd_key] for r in valid]))
            hd_mean = float(np.nanmean([r[hd_key] for r in valid]))
        else:
            best_row, cd_mean, hd_mean = None, np.nan, np.nan
        return best_row, cd_mean, hd_mean, len(valid)

    best_s, cd_mean_s, hd_mean_s, n_valid_s = _best_and_mean(rows, "cd_single", "hd_single", "metric_ok_single")
    best_f, cd_mean_f, hd_mean_f, n_valid_f = _best_and_mean(rows, "cd_full", "hd_full", "metric_ok_full")

    n_total = len(rows)
    n_exec_ok_single = sum(int(r.get("exec_ok_single", 0) == 1) for r in rows)
    n_exec_ok_full   = sum(int(r.get("exec_ok_full",   0) == 1) for r in rows)
    n_pred_exist_single = sum(int(r.get("pred_single_exists", 0) == 1) for r in rows)
    n_pred_exist_full   = sum(int(r.get("pred_full_exists",   0) == 1) for r in rows)

    return {
        "group_index": pid,
        "k_index": "summary",
        "op_type": "Geometry",
        "best_k_single": (best_s or {}).get("k_index", None),
        "cd_single_best": (best_s or {}).get("cd_single", np.nan),
        "hd_single_best": (best_s or {}).get("hd_single", np.nan),
        "cd_single_mean": cd_mean_s,
        "hd_single_mean": hd_mean_s,
        "best_k_full": (best_f or {}).get("k_index", None),
        "cd_full_best": (best_f or {}).get("cd_full", np.nan),
        "hd_full_best": (best_f or {}).get("hd_full", np.nan),
        "cd_full_mean": cd_mean_f,
        "hd_full_mean": hd_mean_f,
        "n_total": n_total,
        "n_exec_ok_single": n_exec_ok_single,
        "n_exec_ok_full": n_exec_ok_full,
        "n_pred_exist_single": n_pred_exist_single,
        "n_pred_exist_full": n_pred_exist_full,
        "n_metric_ok_single": n_valid_s,
        "n_metric_ok_full": n_valid_f,
    }

def _safe_get_cd_hd(pred_step_path, gt_step_path, *, num_points=None, angles=None):
    """包装 get_cd_hd，任何异常都转成 MetricsResult(ok=False, reason=...)；支持固定角度。"""
    from utils.compute_3D import get_cd_hd, MetricsResult
    try:
        if angles is None:
            return get_cd_hd(pred_step_path=pred_step_path, gt_step_path=gt_step_path)
        else:
            return get_cd_hd(pred_step_path=pred_step_path, gt_step_path=gt_step_path, angles=angles)
    except Exception as e:
        reason = f"metric_exception:{type(e).__name__}:{e}"
        return MetricsResult(None, None, None, ok=False, reason=reason)


def process_one(r, K, COP, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR, GT_EDGES_DIR):
    """处理单个样本，返回 (per_cand_rows, summary_rows, pid)。per_cand_rows 含 single/full 两类，用 kind 区分。"""
    pid = str(r["group_index"])
    import re, os
    from utils.compute_3D import get_cd_hd, MetricsResult
    from model_call.call_model import get_model_candidates
    from model_call.prompt import build_incremental_cq_prompt_infer, build_incremental_cq_prompt
    from utils.post_code_process import build_iso_code, build_integrated_code

    # -------- 基本信息 --------
    m = re.search(r"step(\d+)", pid)
    step_num = int(m.group(1)) if m else -1
    first_step = (step_num == 0)

    gt_img, gt_single_step, gt_full_step = resolve_gt_paths(pid, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR)

    # 前序代码
    prev_code = _load_prev_code_from_dir(pid, COP_PRE_CODE_DIR if COP else PRE_CODE_DIR)
    prev_path = r.get("prev_code_path", None)
    if isinstance(prev_path, str) and os.path.exists(prev_path):
        try:
            prev_code = open(prev_path, "r", encoding="utf-8").read()
        except Exception as e:
            print(f"[WARN] failed to read prev_code_path for {pid}: {e}")

    # Prompt & 候选代码
    op_kind = str(r.get("op", "")).lower()

    # === NEW: few-shot 示例（按当前 pid 分类在 one_shot.csv 中挑一条）===
    few_shot = None
    try:
        pmap_all = _load_prompts_map()   # 你已有的函数
        few = _build_few_shot_for_pid(pid, pmap_all, COP)
        if few:
            few_shot = [few]             # 目前只放1条；后续可扩展为多条
    except Exception as e:
        few_shot = None
        print(f"[ONESHOT] build few-shot failed for {pid}: {e}")

    # prompt = build_incremental_cq_prompt(
    #     previous_code=prev_code,
    #     operation_instruction=r["prompt_text"],
    #     link_mode=None,
    #     images=None,
    #     image_prompt=None,
    #     next_var_name="result",
    #     allow_comments=False,
    #     add_size_guidelines=True,
    #     op_kind=op_kind,
    #     few_shots=few_shot,              # <<< NEW
    # )
    prompt = build_incremental_cq_prompt_infer(
        previous_code=prev_code,
        operation_instruction=r["prompt_text"],
        link_mode=None,
        images=None,
        image_prompt=None,
        next_var_name="result",
        allow_comments=False,
        add_size_guidelines=True,
        op_kind=op_kind,
        few_shots=few_shot,              # <<< NEW
    )

    # print(prompt)
    # print(f"[INFO] Prompt for {pid} length (chars): {len(prompt)}") # 打印长度
    cands = get_model_candidates(prompt, K, thinking=THINKING)

    # 结果收集
    per_cand_rows: List[dict] = []
    summary_rows: List[dict] = []

    # 路径
    base_tmp_dir = os.path.join(TMP_DIR, pid)
    full_save_path = os.path.join(base_tmp_dir, "full_path")
    single_save_path = os.path.join(base_tmp_dir, "single_step")
    os.makedirs(full_save_path, exist_ok=True)
    os.makedirs(single_save_path, exist_ok=True)

    for k_idx, item in enumerate(cands):
        if isinstance(item, dict):
            code = item.get("code", "") or ""
            gen_backend = item.get("backend", "")
            gen_in_tok = item.get("input_tokens", None)
            gen_out_tok = item.get("output_tokens", None)
            gen_tot_tok = item.get("total_tokens", None)
            gen_error  = item.get("err", "")
        else:
            code = str(item or "")
            gen_backend = getattr(cm, "MODEL", "auto")
            gen_in_tok = gen_out_tok = gen_tot_tok = None
            gen_error = ""

        # 统一：把这些字段放在一个 dict，后面拼 row 用
        gen_meta = {
            "gen_backend": gen_backend,
            "gen_input_tokens": gen_in_tok,
            "gen_output_tokens": gen_out_tok,
            "gen_total_tokens": gen_tot_tok,
            "gen_error": gen_error,
            "gen_prompt_len": len(str(prompt))  # 可选：记录 prompt 字符长度
        }

        # ========= 新增：生成代码为空，直接记一条错误记录并跳过 =========
        if not code or not code.strip():
            if op_kind == "chamfer_fillet":
                # CF 模式：只管 full 分支
                row = {
                    "group_index": pid,
                    "k_index": k_idx,
                    "exec_ok_full": 0,
                    "pred_full_path": "",
                    "cd_full": np.nan,
                    "hd_full": np.nan,
                    "reason_full": "empty_code",
                    "pred_full_exists": 0,
                    "metric_ok_full": 0,
                    # CF 指标全置 0 / NaN
                    "cf_num_pred_fillet": 0,
                    "cf_num_pred_chamfer": 0,
                    "cf_num_gt_fillet": 0,
                    "cf_num_gt_chamfer": 0,
                    "cf_hits_fillet": 0,
                    "cf_hits_chamfer": 0,
                    "cf_iou": np.nan,
                    **gen_meta,
                }
                per_cand_rows.append(row)
            else:
                # 普通几何操作
                row = {
                    "group_index": pid,
                    "k_index": k_idx,
                    "exec_ok_single": 0,
                    "exec_ok_full": 0,
                    "pred_single_path": "",
                    "pred_full_path": "",
                    "cd_single": np.nan,
                    "hd_single": np.nan,
                    "cd_full": np.nan,
                    "hd_full": np.nan,
                    "angle_single": None,
                    "angle_full": None,
                    "reason_single": "empty_code",
                    "reason_full": "empty_code",
                    "pred_single_exists": 0,
                    "pred_full_exists": 0,
                    "metric_ok_single": 0,
                    "metric_ok_full": 0,
                    **gen_meta,
                }
                row = _normalize_validity(row)
                per_cand_rows.append(row)

            # 不再往下执行本轮 k 的建模 / 渲染 / 度量
            continue
        # ========= 空代码早退逻辑结束 =========

        if op_kind == "chamfer_fillet":
            # 只构建 full.py （整合代码）
            base_tmp_dir = os.path.join(TMP_DIR, pid)
            full_save_path = os.path.join(base_tmp_dir, "full_path")
            os.makedirs(full_save_path, exist_ok=True)
            full_step_path = os.path.join(full_save_path, f"k{k_idx}_full.step")
            full_py = os.path.join(full_save_path, f"k{k_idx}_full.py")

            # 生成整合代码（不生成 single）
            integrated_code, final_lhs = build_integrated_code(prev_code, code, full_step_path, first_step=False)
            with open(full_py, "w", encoding="utf-8") as f:
                f.write(integrated_code)

            # 执行一次即可
            ok_full, loc_full, err_full = safe_exec_from_path(full_py)
            pred_full_step_exists = ok_full and os.path.exists(full_step_path)

            # CF 边级匹配（从代码层提取选边）
            pred_f_edges, pred_c_edges, parse_err = _eval_pred_edges_from_blocks(prev_code, code)
            gt_f_edges, gt_c_edges = _load_gt_edges_for_pid(GT_EDGES_DIR, pid)

            cfm = {}
            if not parse_err:
                cfm = _compute_cf_iou_metrics(pred_f_edges, pred_c_edges, gt_f_edges, gt_c_edges)

            # === 保存预测边信息（与 GT 格式一致，可留可去） ===
            if not parse_err:
                cf_pred_json = {
                    "group_index": pid,
                    "k_index": k_idx,
                    "op_tag": "Pred_FilletChamfer",
                    "op_kind": "Chamfer/Fillet",
                    "num_edges_fillet": len(pred_f_edges),
                    "num_edges_chamfer": len(pred_c_edges),
                    "edges_fillet": pred_f_edges,
                    "edges_chamfer": pred_c_edges,
                }
                pred_json_path = os.path.join(full_save_path, f"k{k_idx}_pred_edges.json")
                with open(pred_json_path, "w", encoding="utf-8") as f:
                    json.dump(cf_pred_json, f, ensure_ascii=False, indent=2)

            row = {
                "group_index": pid,
                "k_index": k_idx,
                "exec_ok_full": int(ok_full),
                "pred_full_path": full_step_path if pred_full_step_exists else "",
                "cd_full": np.nan,
                "hd_full": np.nan,
                "reason_full": f"exec_error:{err_full.splitlines()[-1] if err_full else ''}" if not ok_full else "",
                "pred_full_exists": int(pred_full_step_exists),
                "metric_ok_full": int(ok_full),
                **cfm,       # ←← 只包含 7 个 CF 字段
                **gen_meta,
            }
            per_cand_rows.append(row)
            continue

        # 输出文件路径
        full_step_path = os.path.join(full_save_path,   f"k{k_idx}_full.step")
        single_step_path = os.path.join(single_save_path, f"k{k_idx}_single.step")
        full_py = os.path.join(full_save_path,   f"k{k_idx}_full.py")
        single_py = os.path.join(single_save_path, f"k{k_idx}_single.py")

        # 生成代码
        single_code, info_shape = build_iso_code(prev_code, code, single_step_path,first_step=first_step)   # info_shape['lhs'] 可用
        integrated_code, final_lhs = build_integrated_code(prev_code, code, full_step_path, first_step=first_step)

        # 落盘
        with open(single_py, "w", encoding="utf-8") as f:
            f.write(single_code)
        with open(full_py, "w", encoding="utf-8") as f:
            f.write(integrated_code)

        # 执行 single 与 full
        ok_single, loc_single, err_single = safe_exec_from_path(single_py)

        # 无论 single 成功与否，都独立执行 full
        ok_full, loc_full, err_full = safe_exec_from_path(full_py)

        # 预测文件存在性
        pred_single_step_exists = ok_single and os.path.exists(single_step_path)
        pred_full_step_exists = ok_full and os.path.exists(full_step_path)

        # ===== 评测 SINGLE =====
        if pred_single_step_exists and gt_single_step:
            res_s = _safe_get_cd_hd(pred_step_path=single_step_path, gt_step_path=gt_single_step)
        else:
            miss = "gt_step_missing" if not gt_single_step else "pred_step_missing"
            res_s = MetricsResult(None, None, None, ok=False, reason=miss)

        reason_s = ""
        if not ok_single:
            reason_s = f"single_exec_error:{(err_single.splitlines()[-1] if err_single else 'unknown')}"
        if not res_s.ok and getattr(res_s, "reason", ""):
            reason_s = (reason_s + "; " if reason_s else "") + res_s.reason

        # ===== 评测 FULL =====
        if pred_full_step_exists and gt_full_step:
            if first_step and ok_single:
                # step0: full == single，直接复用 single 指标（不再重复计算）
                res_f = MetricsResult(cd=res_s.cd,
                                    hd=res_s.hd,
                                    best_euler_angle=res_s.best_euler_angle,
                                    ok=res_s.ok,
                                    reason="inherit_from_single_step0")
            else:
                # step≥1：固定 0°（不做枚举、不做兜底）
                res_f = _safe_get_cd_hd(pred_step_path=full_step_path,
                                        gt_step_path=gt_full_step,
                                        angles=[0])
        else:
            miss = "gt_step_missing" if not gt_full_step else "pred_step_missing"
            res_f = MetricsResult(None, None, None, ok=False, reason=miss)

        reason_f = ""
        if ok_single and not ok_full:
            reason_f = f"full_exec_error:{(err_full.splitlines()[-1] if err_full else 'unknown')}"
        if not res_f.ok and getattr(res_f, "reason", ""):
            reason_f = (reason_f + "; " if reason_f else "") + res_f.reason

        row = {
            "group_index": pid,
            "k_index": k_idx,
            "exec_ok_single": int(ok_single),
            "exec_ok_full": int(ok_full),
            "pred_single_path": single_step_path if pred_single_step_exists else "",
            "pred_full_path": full_step_path if pred_full_step_exists else "",
            "cd_single": res_s.cd,
            "hd_single": res_s.hd,
            "cd_full": res_f.cd,
            "hd_full": res_f.hd,
            "angle_single": getattr(res_s, "best_euler_angle", None),
            "angle_full": getattr(res_f, "best_euler_angle", None),
            "reason_single": reason_s if not res_s.ok else "",
            "reason_full": reason_f if not res_f.ok else "",
            "pred_single_exists": int(pred_single_step_exists),
            "pred_full_exists": int(pred_full_step_exists),
            "metric_ok_single": int(res_s.ok and (res_s.cd is not None) and (res_s.hd is not None)),
            "metric_ok_full": int(res_f.ok and (res_f.cd is not None) and (res_f.hd is not None)),
            **gen_meta,                 # ←← 新增：写到 cands.csv
        }
        row = _normalize_validity(row)
        per_cand_rows.append(row)

    # ====== summary ======
    summary_row = _compute_summary(per_cand_rows, pid, op_kind) if WRITE_SUMMARY else None
    summary_rows = [summary_row] if WRITE_SUMMARY else []
    return per_cand_rows, summary_rows, pid

def _repair_worker(args):
    """并行子进程执行：返回 (per_cand_rows, summary_rows, pid)"""
    fake_row, Kfix, COP, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR, GT_EDGES_DIR = args
    return process_one(
        fake_row,
        K=Kfix,
        COP=COP,
        GT_IMAGE_DIR=GT_IMAGE_DIR,
        GT_SINGLE_STEP_DIR=GT_SINGLE_STEP_DIR,
        GT_EDGES_DIR=GT_EDGES_DIR,
    )

# ======= 新增：安全原子写 =======
import tempfile, shutil, os, pandas as pd

def _write_csv_atomic(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        df.to_csv(tmp_path, index=False)
        # 原子替换
        os.replace(tmp_path, path)
        # 最保险：sync 目录项（可选，部分系统才有用）
        try:
            dir_fd = os.open(os.path.dirname(path), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main_repair_parallel():
    """
    实时写盘的 repair 模式：
    - 每完成一个 pid，就把该 pid 的 per_cand_rows 合并到 cands.csv（覆盖相同 group_index,k_index）
    - 同时重算该 pid 的 summary 并写回 summary.csv（覆盖该 pid 的 summary 行）
    - 采用原子写盘，避免中途中断导致损坏
    """
    global args
    from tqdm import tqdm

    os.makedirs(OUT_DIR, exist_ok=True); ensure_dir(TMP_DIR)
    cand_out_path = os.path.join(OUT_DIR, "cands.csv")
    summary_out_path = os.path.join(OUT_DIR, "summary.csv")

    # 读 prompts 映射（用于获取 op）
    pmap = _load_prompts_map()

    # 读 repair.csv：需要 group_index；可选 k_index、prev_code_path
    dfr = pd.read_csv(args.repair_csv)
    dfr.columns = [c.lower() for c in dfr.columns]
    if "group_index" not in dfr.columns:
        raise KeyError("repair-csv 需要列：group_index（可选：k_index, prev_code_path）")
    dfr = dfr.dropna(subset=["group_index"]).copy()
    dfr["group_index"] = dfr["group_index"].astype(str).str.strip()
    if "k_index" in dfr.columns:
        dfr["k_index"] = pd.to_numeric(dfr["k_index"], errors="coerce").astype("Int64")
    else:
        dfr["k_index"] = pd.Series([0]*len(dfr), dtype="Int64")

    # 组织任务：一个 pid -> 修哪些槽，Kfix
    to_fix = {}
    extra_prev = {}
    for _, r in dfr.iterrows():
        pid = r["group_index"]
        if pid not in pmap:
            print(f"[WARN] {pid} 不在 prompts.csv，跳过")
            continue
        kslot = int(r["k_index"]) if pd.notna(r["k_index"]) else 0
        to_fix.setdefault(pid, set())
        if kslot in (0, 1):
            to_fix[pid].add(kslot)
        if "prev_code_path" in dfr.columns and pd.notna(r.get("prev_code_path", None)):
            extra_prev[pid] = str(r["prev_code_path"]).strip()

    # 预载现有 CSV（若存在）
    if os.path.exists(cand_out_path):
        cands = pd.read_csv(cand_out_path)
        cands["group_index"] = cands["group_index"].astype(str)
        cands["k_index"] = pd.to_numeric(cands["k_index"], errors="coerce").astype("Int64")
    else:
        cands = pd.DataFrame()

    if WRITE_SUMMARY and os.path.exists(summary_out_path):
        sums = pd.read_csv(summary_out_path)
        sums["group_index"] = sums["group_index"].astype(str)
    else:
        sums = pd.DataFrame()

    # 生成任务包
    tasks = []
    slot_map = {}  # pid -> (slots, Kfix, base_task)
    for pid, slots in to_fix.items():
        meta = pmap[pid]
        base_task = {
            "group_index": pid,
            "prompt_text": meta["prompt_text"],
            "op": meta.get("op", ""),
            "prev_code_path": extra_prev.get(pid, None),
        }
        if {0,1}.issubset(slots) or slots == {0,1}:
            Kfix = 2
        else:
            Kfix = 1
        tasks.append((pid, slots, base_task, Kfix))
        slot_map[pid] = (slots, Kfix, base_task)

    # 并行执行 + 实时落盘
    job_args = []
    for pid, slots, base_task, Kfix in tasks:
        fake_row = {
            "group_index": pid,
            "prompt_text": base_task["prompt_text"],
            "op": base_task["op"],
            "prev_code_path": base_task["prev_code_path"],
        }
        job_args.append((fake_row, Kfix, COP, GT_IMAGE_DIR, GT_SINGLE_STEP_DIR, GT_EDGES_DIR))

    print("[Repair] 实时记录模式启动...")
    with mp.Pool(processes=NPROC,maxtasksperchild=10) as pool:
        for per_cand_rows, _, pid in tqdm(
            pool.imap_unordered(_repair_worker, job_args, chunksize=1),
            total=len(job_args), desc="[Repair-Streaming]", ncols=100
        ):
            slots, Kfix, base_task = slot_map[pid]

            if not per_cand_rows:
                print(f"[WARN] {pid} 无生成结果，跳过", flush=True)
                continue

            # 规范 k_index（覆盖到需要的槽）
            if Kfix == 2:
                per_cand_rows = per_cand_rows[:2]
                for i, r in enumerate(per_cand_rows):
                    r["k_index"] = i  # 0,1
            else:
                # 只修一个槽
                kslot = list(slots)[0]
                for r in per_cand_rows:
                    r["k_index"] = kslot

            new_df = pd.DataFrame(per_cand_rows)
            new_df["group_index"] = new_df["group_index"].astype(str)
            new_df["k_index"] = pd.to_numeric(new_df["k_index"], errors="coerce").astype("Int64")

            # —— 合并到 cands（覆盖相同 pid+k）——
            if not cands.empty:
                touched = set(zip(new_df["group_index"].astype(str), new_df["k_index"].astype(int)))
                # 只在 pid 命中时做精确删除，避免全表 apply 慢
                mask_pid = cands["group_index"].astype(str).isin(new_df["group_index"].unique())
                if mask_pid.any():
                    idx_drop = []
                    for idx, row in cands[mask_pid].iterrows():
                        key = (str(row["group_index"]), int(row["k_index"]) if pd.notna(row["k_index"]) else -1)
                        if key in touched:
                            idx_drop.append(idx)
                    if idx_drop:
                        cands = cands.drop(index=idx_drop)
                cands = pd.concat([cands, new_df], ignore_index=True)
            else:
                cands = new_df.copy()

            # —— 立刻写出 cands.csv（原子）——
            _write_csv_atomic(cands, cand_out_path)

            # —— 重算该 pid 的 summary，并覆盖写回（如开启）——
            if WRITE_SUMMARY:
                # 拿最新的该 pid 的 k0/1 行参与 summary
                rows_pid = cands[
                    (cands["group_index"] == pid) &
                    (cands["k_index"].isin([0, 1]))
                ].to_dict(orient="records")

                op_kind = str(pmap.get(pid, {}).get("op", "geometry")).lower()
                new_sum = _compute_summary(rows_pid, pid, op_kind)

                if not sums.empty and {"group_index","k_index"}.issubset(sums.columns):
                    sums = sums[~(
                        (sums["group_index"].astype(str) == pid) &
                        (sums["k_index"].astype(str).isin(["summary","summary_best"]))
                    )]
                sums = pd.concat([sums, pd.DataFrame([new_sum])], ignore_index=True)
                _write_csv_atomic(sums, summary_out_path)

            print(f"[✓ 实时写盘] {pid} 已合并到 cands.csv"
                  + (" 并更新 summary.csv" if WRITE_SUMMARY else ""), flush=True)

    print("[REPAIR] 全部任务完成（实时记录已开启，文件随进度写出）。")
from tqdm import tqdm 
def main_parallel():
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_dir(TMP_DIR)

    cand_out_path = os.path.join(OUT_DIR, "cands.csv")
    summary_out_path = os.path.join(OUT_DIR, "summary.csv")

    df = pd.read_csv(PROMPTS_CSV)
    df.columns = [c.lower() for c in df.columns]
    required = {"group_index", "prompt_text","op"}
    miss = required - set(df.columns)
    if miss:
        raise KeyError(f"prompts.csv 缺少列: {miss}")
    
     # ===== 新增：按 split-json 过滤 =====
    if getattr(args, "split_json", None):
        if not os.path.exists(args.split_json):
            raise FileNotFoundError(f"--split-json 文件不存在: {args.split_json}")
        with open(args.split_json, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        key = getattr(args, "split_key", "test")
        if key not in split_obj:
            raise KeyError(f"--split-key={key} 不在 {args.split_json} 中的键集合: {list(split_obj.keys())}")
        # 支持：直接是列表；或是字典里再套列表
        if isinstance(split_obj[key], dict):
            # 如果是 { "items": [...] } 之类，尝试拼成一个列表
            # 你也可以按实际结构改这里
            white_list = []
            for v in split_obj[key].values():
                if isinstance(v, list):
                    white_list.extend(map(str, v))
        else:
            white_list = list(map(str, split_obj[key]))

        white_set = set(white_list)
        n_before = len(df)
        df = df[df["group_index"].astype(str).isin(white_set)].copy()
        print(f"[SPLIT] split-json={args.split_json} key={key}  kept={len(df)}/{n_before}")

    # RESUME：从 summary.csv 读取已完成样本
# 结果缓冲区 (!!! 提前初始化 !!!)
    all_cand_rows = []
    all_summary_rows = []

    # RESUME：从 summary.csv 读取已完成样本
    done_group_indexs = set()
    if RESUME and WRITE_SUMMARY and os.path.exists(summary_out_path):
        print(f"[RESUME] Loading existing {summary_out_path} for resume and appending...")
        try:
            old_summary_df = pd.read_csv(summary_out_path)
            if {"group_index", "k_index"}.issubset(old_summary_df.columns):
                # 1. Populate done_group_indexs for skipping
                mask = old_summary_df["k_index"].isin(["summary", "summary_best"])
                done_group_indexs = set(old_summary_df.loc[mask, "group_index"].astype(str))
                
                # 2. Load existing data for appending
                all_summary_rows = old_summary_df.to_dict(orient="records") # <-- 装载旧数据
                print(f"[RESUME] Loaded {len(all_summary_rows)} summary rows. Found {len(done_group_indexs)} completed tasks.")
        except Exception as e:
            print(f"[WARN] Failed to load {summary_out_path}, will start fresh. Error: {e}")
    # 基于 summary 的完成列表，加载 cands.csv
    if RESUME and os.path.exists(cand_out_path):
        print(f"[RESUME] Loading existing {cand_out_path} for appending...")
        try:
            old_cands_df = pd.read_csv(cand_out_path)
            
            if not old_cands_df.empty and not done_group_indexs:
                    print(f"[WARN] {cand_out_path} exists, but no completed tasks found in summary. Will overwrite cands.csv.")
            elif not old_cands_df.empty:
                old_cands_df["group_index"] = old_cands_df["group_index"].astype(str)
                
                # 关键：只保留那些 "已完成" 任务的 cands
                # 这样，"未完成" (pending) 的任务被重新运行后，旧的 cand (if any) 会被丢弃
                cands_to_keep = old_cands_df[old_cands_df["group_index"].isin(done_group_indexs)]
                all_cand_rows = cands_to_keep.to_dict(orient="records") # <-- 装载旧数据
                print(f"[RESUME] Loaded and kept {len(all_cand_rows)} candidate rows from completed tasks.")
        except Exception as e:
            print(f"[WARN] Failed to load {cand_out_path}, will start fresh. Error: {e}")

    # 过滤掉已完成的样本
    pend_rows = [r for _, r in df.iterrows() if str(r["group_index"]) not in done_group_indexs]
    n_total = len(pend_rows)
    print(f"[INFO] total={len(df)}, resume-skip={len(done_group_indexs)}, to-run={n_total}")

    # 结果缓冲区 (!!! 此处的初始化已删除 !!!)
    # all_cand_rows, all_summary_rows = [], []  <-- 确保这行已被删除或注释掉
    buffer_count = 0

    # ---- 带进度条 ----
    with mp.Pool(processes=NPROC,maxtasksperchild=10) as pool:
        worker = partial(process_one, K=K, COP=COP,
                        GT_IMAGE_DIR=GT_IMAGE_DIR,
                        GT_SINGLE_STEP_DIR=GT_SINGLE_STEP_DIR,
                        GT_EDGES_DIR=GT_EDGES_DIR)
        with tqdm(total=n_total, desc="[Progress]", dynamic_ncols=True) as pbar:
            it = pool.imap_unordered(worker, pend_rows, chunksize=1)
            while True:
                try:
                    per_cand_rows, summary_rows, pid = next(it)
                except StopIteration:
                    break
                except Exception as e:
                    # 记录一条错误summary，不要中断
                    if WRITE_SUMMARY:                                              # === 新增开关 ===
                        err_row = {"group_index": "unknown", "k_index": "summary",
                                   "op_type": "Unknown", "n_total": 0,
                                   "error": f"worker_loop_exception:{type(e).__name__}:{e}"}
                        all_summary_rows.append(err_row)
                    pbar.update(1)
                    continue

                # 正常累加与落盘
                all_cand_rows.extend(per_cand_rows)
                if WRITE_SUMMARY:                                                  # === 新增开关 ===
                    all_summary_rows.extend(summary_rows)
                buffer_count += 1
                pbar.update(1)
                if buffer_count % WRITE_EVERY == 0:
                    pd.DataFrame(all_cand_rows).to_csv(cand_out_path, index=False)
                    if WRITE_SUMMARY:                                              # === 新增开关 ===
                        pd.DataFrame(all_summary_rows).to_csv(summary_out_path, index=False)
                    pbar.set_postfix({"saved": len(all_summary_rows)})


    # ---- 最后一批写盘 ----
    pd.DataFrame(all_cand_rows).to_csv(cand_out_path, index=False)
    if WRITE_SUMMARY:                                                              # === 新增开关 ===
        pd.DataFrame(all_summary_rows).to_csv(summary_out_path, index=False)
    print("[done] all results saved.")

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_args(args)

    if getattr(args, "repair_csv", None):
        main_repair_parallel()   # ← 修正模式
    else:
        main_parallel()          # ← 常规评测
    # group_dir = "/data/baiyixue/CAD/step_files/00002_index_2"  # 例子
    # print(_pick_single_step_path(group_dir, [0]))       # 期望 step0/3D.step
    # print(_pick_single_step_path(group_dir, [1,2,3]))   # 期望 step1_2_3/3D.step（或兼容变体）
