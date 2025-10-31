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

def set_runtime_config(
    *,
    gen_mode: Optional[str] = None,
    provider: Optional[str] = None,
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
        CFG.setdefault("http",   {}).setdefault("enabled", False)
        CFG.setdefault("siliconflow", {}).setdefault("enabled", False)

        if p == "openai":
            CFG["openai"]["enabled"] = True
            CFG["http"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = False

        elif p == "http":
            CFG["openai"]["enabled"] = False
            CFG["http"]["enabled"] = True
            CFG["siliconflow"]["enabled"] = False

        elif p in ("siliconflow", "sf"):  # 兼容缩写
            # 核心：启用 http 通道，但从 siliconflow 节读取配置并写回 http
            CFG["openai"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = True
            CFG["http"]["enabled"] = True
            _adopt_siliconflow_into_http(CFG)

        elif p == "local":
            CFG["openai"]["enabled"] = False
            CFG["http"]["enabled"] = False
            CFG["siliconflow"]["enabled"] = False
            gen_mode = gen_mode or "local"
            CFG.setdefault("gen", {})["mode"] = gen_mode

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

def _extract_code_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, flags=re.S)
    code = (m.group(1) if m else text).strip()
    return code


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


def _gen_via_openai(prompt: str) -> Dict[str, Any]:
    from openai import APIConnectionError, APITimeoutError, RateLimitError, APIError

    client, api_key = _build_openai_client()
    if not api_key:
        return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                "backend":"openai", "err":"missing_openai_api_key"}

    model = CFG["openai"]["model"]
    temperature = float(CFG["gen"]["temperature"])

    # 明确处理 429：不返回错误，内部等待并重试
    tries = 0
    last_err = ""
    while tries < RATE_LIMIT_MAX_RETRIES:
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
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
            # 429：吞掉、等待、继续；不把这次算作失败记录
            tries += 1
            wait_s = RATE_LIMIT_WAIT_S
            # 若能拿到 Retry-After，优先
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
            # 短暂网络问题：温和退避
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(3)
            tries += 1
            continue

        except APIError as e:
            # 其他 APIError（非429）直接返回，让上层记录
            last_err = f"{type(e).__name__}: {e}"
            break

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            break

    # 走到这里要么多次 429 要么持续失败；为避免“记录429”，标注成可识别错误码，交给上层决定是否跳过
    return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
            "backend":"openai", "err":"rate_limit_exceeded"}



# ===== HTTP backend =====
def _gen_via_http(prompt: str) -> Dict[str, Any]:
    import requests, json

    try:
        http = CFG.get("http", {})
        if not http.get("enabled", False):
            return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                    "backend":"http", "err":"http_backend_disabled"}

        url = (http.get("base_url") or "").rstrip("/")
        headers = http.get("headers")
        model = http.get("model", "gpt-4")
        temperature = float(CFG["gen"].get("temperature", 0.7))
        timeout_s = int(CFG["gen"].get("timeout_s", 1800))
        if not url:
            return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                    "backend":"http", "err":"missing_base_url_or_endpoint"}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        tries = 0
        while tries < RATE_LIMIT_MAX_RETRIES:
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
                if r.status_code == 429:
                    tries += 1
                    wait_s = RATE_LIMIT_WAIT_S
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try: wait_s = int(ra)
                        except: pass
                    print(f"[429] HTTP 限流，第 {tries}/{RATE_LIMIT_MAX_RETRIES} 次等待 {wait_s}s 之后重试")
                    time.sleep(wait_s)
                    continue  # 不返回、不落盘
                r.raise_for_status()
                rsp = r.json()

                choices = rsp.get("choices") or []
                txt = (choices[0]["message"]["content"] if choices else "").strip()
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
                # 非 429 的 HTTP 错误：直接返回，让上层记录
                return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                        "backend":"http", "err": f"HTTPError:{e}"}
            except Exception as e:
                # 其它异常：温和重试几次
                tries += 1
                if tries >= 3:
                    return {"code":"", "input_tokens":None, "output_tokens":None, "total_tokens":None,
                            "backend":"http", "err": f"{type(e).__name__}: {e}"}
                time.sleep(3)

        # 多次 429 仍未成功
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


def get_model_candidates(prompt: str, k: int = None) -> List[Dict[str, Any]]:
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
                    result = _gen_via_openai(prompt)
                elif CFG["http"]["enabled"]:
                    result = _gen_via_http(prompt)
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
