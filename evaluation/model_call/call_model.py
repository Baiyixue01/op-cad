import json, time, random, re, traceback, os
from typing import List, Dict, Any
from .config_loader import load_config
from typing import Optional


RATE_LIMIT_WAIT_S = 65          # 429 时等待 60 秒
RATE_LIMIT_MAX_RETRIES = 6      # 429 最多重试 6 次（总等待不超过 6 分钟）

# ===== 全局配置 =====
CFG = load_config("config.json")  # 全局加载一次
MODEL = "auto"
# MODE = str(CFG["gen"]["mode"]).lower()
# if MODE == "local":
#     MODEL = "local"
# elif MODE == "api":
#     if CFG["openai"]["enabled"]:
#         MODEL = CFG["openai"]["model"]
#     elif CFG["http"]["enabled"]:
#         MODEL = CFG["http"]["model"]
#     else:
#         raise RuntimeError("API mode enabled but no API backend selected.")
# else:
#     # auto / 其它 → 走 _try_local_then_api
#     MODEL = "auto"

def _resolve_gen_params() -> Dict[str, Any]:
    g = CFG.get("gen", {}) or {}
    # 默认值（兜底）
    params = {
        "temperature": float(g.get("temperature", 0.7)),
        "top_p": float(g.get("top_p", 1.0)) if g.get("top_p") is not None else None,
        "top_k": int(g.get("top_k", 40)) if g.get("top_k") is not None else None,
        "repetition_penalty": float(g.get("repetition_penalty", 1.0)) if g.get("repetition_penalty") is not None else None,
        "presence_penalty": float(g.get("presence_penalty", 0.0)) if g.get("presence_penalty") is not None else None,
        "out_seq_length": int(g.get("out_seq_length", 4096)),
        "greedy": bool(g.get("greedy", False)),
    }

    # greedy 优先：统一约定 temperature=0, top_k=1（top_p 保持 1.0）
    if params["greedy"]:
        params["temperature"] = 0.0
        params["top_k"] = 1
        if params["top_p"] is None:
            params["top_p"] = 1.0

    return params


def _adopt_siliconflow_into_http(CFG):
    """将 CFG['siliconflow'] 的配置拷贝/覆盖到 CFG['http']，供统一的 HTTP 调用使用。"""
    CFG.setdefault("http", {})
    sf = CFG.get("siliconflow", {}) or {}
    if not sf:
        return
    # 映射常用字段
    if sf.get("base_url"):   CFG["http"]["base_url"] = sf["base_url"]
    if sf.get("headers"):    CFG["http"]["headers"]  = sf["headers"]
    if sf.get("model"):      CFG["http"]["model"]    = sf["model"]

def _adopt_vllm_into_http(CFG, endpoint_key: Optional[str] = None):
    """
    将 CFG['vllm'] 的配置拷贝/覆盖到 CFG['http']。
    支持新的多端点 'endpoints' 结构，并根据策略或传入的 key 选择。
    """
    CFG.setdefault("http", {})
    vllm = CFG.get("vllm", {}) or {}
    if not vllm:
        return

    endpoints = vllm.get("endpoints", {})

    # ---- 1. 检查是否使用新的 'endpoints' 结构 ----
    if isinstance(endpoints, dict) and endpoints:
        key_to_use = None
        keys = list(endpoints.keys()) # 获取所有可用的 key
        if not keys:
            print("[WARN] vllm.endpoints is defined but empty. Falling back.")
            # (继续执行下面的旧逻辑)
        else:
            # 1a. 优先使用命令行传入的 key
            if endpoint_key and endpoint_key in endpoints:
                key_to_use = endpoint_key

            # 1b. 否则，按 config.json 中的策略选择
            if not key_to_use:
                strategy = vllm.get("strategy", "default")

                if strategy == "round_robin":
                    # 简单的轮询 (在多进程中，基于PID的取模是一个很好的无锁分发)
                    idx = os.getpid() % len(keys) 
                    key_to_use = keys[idx]
                else: 
                    # 默认策略: "default"
                    key_to_use = vllm.get("default_key")
                    if not key_to_use or key_to_use not in endpoints:
                         print(f"[WARN] vllm.default_key='{key_to_use}' not found in endpoints. Using first key '{keys[0]}'.")
                         key_to_use = keys[0] # 兜底选第一个

            # 1c. 拿到选中的端点配置
            chosen_endpoint = endpoints.get(key_to_use, {})
            print(f"[vLLM] Using endpoint key: '{key_to_use}' -> {chosen_endpoint.get('base_url')}")

            # 1d. 映射到 http
            if chosen_endpoint.get("base_url"):
                CFG["http"]["base_url"] = chosen_endpoint["base_url"]
            if chosen_endpoint.get("headers"):
                CFG["http"]["headers"] = chosen_endpoint["headers"]
            if chosen_endpoint.get("model"):
                CFG["http"]["model"] = chosen_endpoint["model"]
            return # --- 配置完成，退出 ---

    # ---- 2. 回退到旧的单体 vllm 配置 (如果 'endpoints' 不存在或为空) ----
    print("[vLLM] Using legacy single vLLM config (no valid 'endpoints' structure found).")
    if vllm.get("base_url"):  CFG["http"]["base_url"] = vllm["base_url"]
    if vllm.get("headers"):   CFG["http"]["headers"]  = vllm["headers"]
    if vllm.get("model"):     CFG["http"]["model"]    = vllm["model"]
# ====================

def set_runtime_config(
    *,
    gen_mode: Optional[str] = None,
    provider: Optional[str] = None,
    vllm_endpoint_key: Optional[str] = None, # <-- 新增这行
    openai_model: Optional[str] = None,
    http_model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout_s: Optional[int] = None
):
    """
    运行时更新生成配置；优先级高于 config.json。
    修改后会同步刷新全局 MODEL 选择逻辑。
    """
    global CFG, MODEL

    # 1) 覆盖 gen.mode
    if gen_mode:
        CFG.setdefault("gen", {})
        CFG["gen"]["mode"] = gen_mode

    # 2) 覆盖 provider（openai/http/local/siliconflow）
    if provider:
        p = provider.lower().strip()
        CFG.setdefault("openai", {}).setdefault("enabled", False)
        CFG.setdefault("http",     {}).setdefault("enabled", False)
        CFG.setdefault("siliconflow", {}).setdefault("enabled", False)
        CFG.setdefault("vllm",   {}).setdefault("enabled", False) # <-- 新增

        if p == "openai":
            CFG["openai"]["enabled"] = True
            CFG["http"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = False

        elif p == "http":
            CFG["openai"]["enabled"] = False
            CFG["http"]["enabled"] = True
            CFG["siliconflow"]["enabled"] = False

        elif p in ("siliconflow", "sf"): 
            CFG["openai"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = True
            CFG["http"]["enabled"] = True
            CFG["vllm"]["enabled"] = False # <-- 新增
            _adopt_siliconflow_into_http(CFG)

        elif p == "vllm":
            CFG["openai"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = False
            CFG["vllm"]["enabled"] = True
            CFG["http"]["enabled"] = True # 核心：启用 http 通道
            _adopt_vllm_into_http(CFG, vllm_endpoint_key) # 核心：用 vllm 配置覆盖 http

        elif p == "local":
            CFG["openai"]["enabled"] = False
            CFG["http"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = False
            gen_mode = gen_mode or "local"
            CFG.setdefault("gen", {})["mode"] = gen_mode
            CFG["vllm"]["enabled"] = False # <-- 新增

    # 3) 模型名覆盖（优先显式入参）
    if openai_model:
        CFG.setdefault("openai", {})["model"] = openai_model
    if http_model:
        CFG.setdefault("http", {})["model"] = http_model

    # 若当前 provider 是 siliconflow，但未手动指定 http_model，则用 siliconflow.model
    if CFG.get("siliconflow", {}).get("enabled") and not http_model:
        sfm = CFG.get("siliconflow", {}).get("model")
        if sfm:
            CFG.setdefault("http", {})["model"] = sfm

    # 4) 其他超参
    if temperature is not None:
        CFG.setdefault("gen", {})["temperature"] = float(temperature)
    if timeout_s is not None:
        CFG.setdefault("gen", {})["timeout_s"] = int(timeout_s)

    # 5) 刷新 MODEL（保持你原来的逻辑）
    mode = str(CFG.get("gen", {}).get("mode", "auto")).lower()
    if mode == "local":
        MODEL = "local"
    elif mode == "api":
        if CFG.get("openai", {}).get("enabled"):
            MODEL = CFG["openai"]["model"]
        elif CFG.get("http", {}).get("enabled"):
            # http 可能来自 siliconflow，也可能是自定义 http
            MODEL = CFG.get("http", {}).get("model") or "http"
        else:
            raise RuntimeError("API mode enabled but no API backend selected.")
    else:
        MODEL = "auto"
# ===== 本地模型（占位）=====
def run_local_model(prompt: str) -> str:
    # TODO: 替换为你的本地推理
    code = f"""
import cadquery as cq
result = cq.Workplane("XY").circle(10).extrude(20)
"""
    return code.strip()

CODE_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*?)```", re.DOTALL)
THINK_BLOCKS = [
    (re.compile(r"<think>.*?</think>", re.DOTALL), ""),              # 常见思维块
    (re.compile(r"<thoughts>.*?</thoughts>", re.DOTALL), ""),
    (re.compile(r"<!--\s*BEGIN THOUGHTS\s*-->.*?<!--\s*END THOUGHTS\s*-->", re.DOTALL), ""),
    (re.compile(r"^Thoughts:.*?(?=\n[A-Z][a-zA-Z]+:|\Z)", re.DOTALL | re.IGNORECASE | re.MULTILINE), ""),  # “Thoughts:”段
]
LANG_WHITELIST = {"python", "py", "cadquery", ""}  # 允许空语言围栏（``` … ```）

def _strip_thinking(text: str) -> str:
    out = text or ""
    for pat, repl in THINK_BLOCKS:
        out = pat.sub(repl, out)
    return out

def _score_candidate(code: str) -> int:
    """简单评分：更像 CadQuery/Python 的得分更高；用于多候选择优。"""
    s = code.strip()
    score = 0
    if "import cadquery as cq" in s: score += 5
    if "cq.Workplane" in s: score += 3
    if "result" in s: score += 1
    if "# === GENERATED CODE" in s or "# === ISO export" in s: score += 2
    if len(s) > 0: score += min(len(s)//200, 5)  # 太短的降权
    return score

def _extract_code_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1) 去掉思维/日志块（不影响正常内容）
    cleaned = _strip_thinking(text)

    # 2) 先尝试 JSON 外壳（thinking 模式常见）：{"code": "..."} 或 {"final_code": "..."}
    try:
        obj = json.loads(cleaned)
        for key in ("code", "final_code", "python", "script"):
            if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].strip():
                return obj[key].strip()
    except Exception:
        pass

    # 3) 抓取所有 fenced code，按语言白名单与启发式评分择优
    cands = []
    for m in CODE_FENCE_RE.finditer(cleaned):
        lang = (m.group("lang") or "").strip().lower()
        body = (m.group("body") or "").strip()
        if lang in LANG_WHITELIST and body:
            cands.append(body)
    if cands:
        cands.sort(key=_score_candidate, reverse=True)
        return cands[0].strip()

    # 4) 无 fenced 时的兜底：从含 CadQuery/Python 特征的行开始截到末尾
    lines = cleaned.splitlines()
    for i, ln in enumerate(lines):
        if ("import cadquery as cq" in ln) or ("cq.Workplane" in ln) or ln.strip().startswith("from cadquery"):
            return "\n".join(lines[i:]).strip()

    # 5) 最终兜底：原样去首尾
    return cleaned.strip()


# ===== OpenAI backend =====
def _build_openai_client():
    from openai import OpenAI
    import httpx

    oai = CFG.get("openai", {})
    api_key = oai.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url = oai.get("base_url", "https://api.openai.com/v1")

    timeout_s = float(CFG["gen"].get("timeout_s", 180))
    # 放到 client 初始化，而不是 .create(...) 里
    client = OpenAI(
        api_key=api_key or None,
        base_url=base_url,
        timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0),
        max_retries=2,  # SDK 级别轻量重试
    )
    return client, api_key


def _gen_via_openai(prompt: str, thinking: bool = False) -> Dict[str, Any]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError, APIError

    client, api_key = _build_openai_client()
    if not api_key:
        return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                "backend":"openai", "err":"missing_openai_api_key"}

    model = CFG["openai"]["model"]
    gp = _resolve_gen_params()

    tries = 0
    last_err = ""
    while tries < RATE_LIMIT_MAX_RETRIES:
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=gp["temperature"],
                top_p=gp["top_p"],                         # OpenAI 支持
                presence_penalty=gp["presence_penalty"],   # OpenAI 支持
                max_tokens=gp["out_seq_length"],           # OpenAI 支持
                # 注：OpenAI 不支持 top_k / repetition_penalty，这里不要传
            )
            txt = (rsp.choices[0].message.content or "").strip()
            code = _extract_code_from_text(txt)
            usage = getattr(rsp, "usage", None)
            return {
                "code": code,
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "backend": "openai",
                "err": "",
            }
        except RateLimitError as e:
            tries += 1
            wait_s = RATE_LIMIT_WAIT_S
            try:
                h = getattr(e, "response", None)
                if h and hasattr(h, "headers"):
                    ra = h.headers.get("Retry-After")
                    if ra:
                        wait_s = int(ra)
            except Exception:
                pass
            print(f"[429] OpenAI 限流，第 {tries}/{RATE_LIMIT_MAX_RETRIES} 次等待 {wait_s}s 之后重试")
            time.sleep(wait_s)
            continue
        except (APITimeoutError, APIConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(3); tries += 1; continue
        except APIError as e:
            last_err = f"{type(e).__name__}: {e}"
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
            "backend":"openai", "err":"rate_limit_exceeded"}




# ===== HTTP backend =====
def _get_tokenizer():
    # 懒加载，全局复用
    global _TOK
    try:
        _TOK
        return _TOK
    except NameError:
        pass

    from transformers import AutoTokenizer
    # 优先从 vllm/http 配置拿模型名；找不到就退到通用 Qwen3
    m = (CFG.get("vllm", {}).get("tokenizer_model")
         or CFG.get("http", {}).get("tokenizer_model")
         or CFG.get("http", {}).get("model")
         or "Qwen/Qwen3-8B")
    _TOK = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    return _TOK


def _gen_via_http(prompt: str, thinking: bool = False) -> Dict[str, Any]:
    import requests, json

    try:
        http = CFG.get("http", {})
        if not http.get("enabled", False):
            return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                    "backend":"http", "err":"http_backend_disabled"}

        base = (http.get("base_url") or "").rstrip("/")
        headers = dict(http.get("headers") or {})
        headers.setdefault("Content-Type", "application/json")
        model = http.get("model", "gpt-4")
        timeout_s = int(CFG["gen"].get("timeout_s", 1800))

        gp = _resolve_gen_params()

        # 公共采样参数（大多数 vLLM/自建接口会识别）
        sampling = {
            "temperature": gp["temperature"],
            "top_p": gp["top_p"],
            "top_k": gp["top_k"],
            "repetition_penalty": gp["repetition_penalty"],
            "presence_penalty": gp["presence_penalty"],
            "max_tokens": gp["out_seq_length"],      # 有些实现用 max_new_tokens；若需要可同时传
            # "max_new_tokens": gp["out_seq_length"],
        }
        # 删除 None 项，避免某些服务端报错
        sampling = {k: v for k, v in sampling.items() if v is not None}

        if thinking:
            # chat-style
            endpoint = base
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                **sampling,
            }
        else:
            # completions-style：手工渲染对话模板
            tok = _get_tokenizer()
            messages = [{"role": "user", "content": prompt}]
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            endpoint = base
            payload = {
                "model": model,
                "prompt": rendered,
                **sampling,
            }

        tries = 0
        while tries < RATE_LIMIT_MAX_RETRIES:
            try:
                r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=timeout_s)
                if r.status_code == 429:
                    tries += 1
                    wait_s = int(r.headers.get("Retry-After", RATE_LIMIT_WAIT_S))
                    print(f"[429] HTTP 限流，第 {tries}/{RATE_LIMIT_MAX_RETRIES} 次等待 {wait_s}s 之后重试")
                    time.sleep(wait_s)
                    continue
                r.raise_for_status()
                rsp = r.json()

                if thinking:
                    txt = (rsp.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
                else:
                    txt = (rsp.get("choices",[{}])[0].get("text","") or "").strip()

                code = _extract_code_from_text(txt)
                usage = rsp.get("usage", {}) or {}
                return {
                    "code": code,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "backend": "http",
                    "err": "",
                }

            except requests.HTTPError as e:
                return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                        "backend":"http", "err": f"HTTPError:{e}"}
            except Exception as e:
                tries += 1
                if tries >= 3:
                    return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                            "backend":"http", "err": f"{type(e).__name__}: {e}"}
                time.sleep(3)

        return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                "backend":"http", "err":"rate_limit_exceeded"}

    except Exception as e:
        return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                "backend":"http", "err": f"{type(e).__name__}: {e}"}






# ===== 混合兜底：local -> openai -> http =====
def _try_local_then_api(prompt: str) -> Dict[str, Any]:
    # 1) local
    try:
        code = run_local_model(prompt)
        return {
            "code": _extract_code_from_text(code),
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "backend": "local",
            "err": "",
        }
    except Exception as e:
        print(f"[WARN] local model failed: {e}")

    # 2) openai
    if CFG["openai"]["enabled"]:
        ret = _gen_via_openai(prompt)
        if ret.get("code"):
            return ret
        else:
            print(f"[WARN] openai backend failed: {ret.get('err')}")

    # 3) http
    if CFG["http"]["enabled"]:
        ret = _gen_via_http(prompt)
        if ret.get("code") or ret.get("err") == "":
            return ret
        else:
            print(f"[WARN] http backend failed: {ret.get('err')}")

    # 全部失败
    return {
        "code": "",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "backend": "none",
        "err": "no_backend",
    }


def get_model_candidates(prompt: str, k: int, *, thinking: bool = False):
    mode = str(CFG["gen"]["mode"]).lower()
    results: List[Dict[str, Any]] = []
    pid = os.getpid()

    # 用 “收集到k条为止”的循环，遇到 429 的占位错误就继续重试，不append
    while len(results) < (k or 1):
        try:
            if mode == "local":
                code = run_local_model(prompt)
                result = {
                    "code": _extract_code_from_text(code),
                    "input_tokens": None, "output_tokens": None, "total_tokens": None,
                    "backend": "local", "err": ""
                }
            elif mode == "api":
                if CFG["openai"]["enabled"]:
                    result = _gen_via_openai(prompt,thinking=thinking)
                elif CFG["http"]["enabled"]:
                    result = _gen_via_http(prompt,thinking=thinking)
                else:
                    raise RuntimeError("API mode enabled but no API backend selected.")
            else:
                result = _try_local_then_api(prompt)

            # --- 关键逻辑：429 产生的占位错误，跳过记录，继续获取下一条 ---
            if result.get("err") == "rate_limit_exceeded":
                # 这里不追加、不计数，直接继续（_gen 内部已等待过）
                continue

            result.setdefault("err", "")
            results.append(result)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[PID {pid}] ERROR: {err_msg}")
            traceback.print_exc()
            results.append({
                "code": "", "input_tokens": None, "output_tokens": None, "total_tokens": None,
                "backend": mode, "err": err_msg
            })

        time.sleep(2)  # 可保留轻微间隔，降低触发几率
    return results
