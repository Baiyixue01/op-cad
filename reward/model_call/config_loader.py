# config_loader.py
import os, json, copy
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "gen": {
    "mode": "auto",
    "k": 5,
    "temperature": 0.7,
    "timeout_s": 45,
    "greedy": False,
    "top_p": 1.0,
    "top_k": 40,
    "repetition_penalty": 1.0,
    "presence_penalty": 2.0,
    "out_seq_length": 32768
    },
    "openai": {
        "enabled": True,
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "organization": "",
        "project": ""
    },
    "http": {
        "enabled": True,
        "endpoint": "http://127.0.0.1:8000/generate",
        "headers": {"Content-Type": "application/json"}
    },
    "paths": {
        "pre_code_dir": "./previous_codes",
        "prompts_csv": "./prompts.csv",
        "out_csv": "./out_eval.csv",
        "tmp_dir": "./_tmp_eval"
    }
}

_SECRET_KEYS = {
    ("openai", "api_key"),
    ("http", "headers", "Authorization")
}

def _mask(k: str, v: str) -> str:
    if not isinstance(v, str) or not v:
        return str(v)
    if len(v) <= 8: return "****"
    return v[:4] + "..." + v[-4:]

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """
    支持用环境变量覆盖敏感/常用字段：
    - OPENAI_API_KEY
    - OPENAI_BASE_URL
    - OPENAI_MODEL
    - HTTP_ENDPOINT
    - GEN_MODE / GEN_K / GEN_TEMPERATURE / GEN_TIMEOUT_S
    """
    env_map = {
        ("openai", "api_key"): os.getenv("OPENAI_API_KEY"),
        ("openai", "base_url"): os.getenv("OPENAI_BASE_URL"),
        ("openai", "model"): os.getenv("OPENAI_MODEL"),
        ("http", "endpoint"): os.getenv("HTTP_ENDPOINT"),
        ("gen", "mode"): os.getenv("GEN_MODE"),
        ("gen", "k"): os.getenv("GEN_K"),
        ("gen", "temperature"): os.getenv("GEN_TEMPERATURE"),
        ("gen", "timeout_s"): os.getenv("GEN_TIMEOUT_S"),
    }
    for path, val in env_map.items():
        if val is None: 
            continue
        # 基本类型转换
        if path[-1] in ("k", "timeout_s"):
            try: val = int(val)
            except: continue
        if path[-1] in ("temperature",):
            try: val = float(val)
            except: continue
        # 写入
        cur = cfg
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = val

def load_config(path: str = None) -> Dict[str, Any]:
    """
    加载配置优先序：
    1) 显式 path
    2) 环境变量 CONFIG_PATH
    3) 默认 ./config.json
    然后应用环境变量覆盖；做基本校验；返回 dict
    """
    path = path or os.getenv("CONFIG_PATH") or "./config.json"
    print(path)
    cfg = copy.deepcopy(_DEFAULTS)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        _deep_update(cfg, user)
    else:
        print(f"[WARN] config file not found: {path}, using defaults")

    _apply_env_overrides(cfg)
    _validate_config(cfg)
    _log_config_safe(cfg, path)
    return cfg

def _validate_config(cfg: Dict[str, Any]) -> None:
    # 模式合法性
    mode = str(cfg["gen"]["mode"]).lower()
    if mode not in ("local", "api", "auto"):
        raise ValueError(f"gen.mode must be one of local/api/auto, got: {mode}")

    # 若 API 模式启用但没任何后端
    if mode in ("api", "auto"):
        if not cfg.get("openai", {}).get("enabled") and not cfg.get("http", {}).get("enabled"):
            raise ValueError("API mode requires at least one backend enabled (openai/http).")

    # 若启用 openai 必须有 key
    if cfg.get("openai", {}).get("enabled") and not cfg["openai"].get("api_key"):
        print("[WARN] openai.enabled=True but openai.api_key is empty.")

def _log_config_safe(cfg: Dict[str, Any], path: str) -> None:
    print(f"[CONFIG] loaded from: {path}")
    g = cfg["gen"]
    print(
        "[CONFIG] gen: "
        f"mode={g.get('mode')} k={g.get('k')} "
        f"temp={g.get('temperature')} top_p={g.get('top_p')} top_k={g.get('top_k')} "
        f"rep_pen={g.get('repetition_penalty')} pres_pen={g.get('presence_penalty')} "
        f"greedy={g.get('greedy')} max_tokens={g.get('out_seq_length')} "
        f"timeout={g.get('timeout_s')}s"
    )

    if cfg["openai"]["enabled"]:
        print("[CONFIG] openai: enabled=True "
              f"base_url={cfg['openai'].get('base_url')} "
              f"model={cfg['openai'].get('model')} "
              f"api_key={_mask('api_key', cfg['openai'].get('api_key',''))}")
    else:
        print("[CONFIG] openai: enabled=False")

    if cfg["http"]["enabled"]:
        auth = cfg["http"].get("headers", {}).get("Authorization", "")
        print("[CONFIG] http: enabled=True "
              f"base_url={cfg['http'].get('base_url') or cfg['http'].get('endpoint')} "
              f"model={cfg['http'].get('model')} "
              f"auth={_mask('Authorization', auth)}")
    else:
        print("[CONFIG] http: enabled=False")

