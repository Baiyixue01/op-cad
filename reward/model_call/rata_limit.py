# rate_limit.py
import time, random, multiprocessing as mp

class MPGlobalRateLimiter:
    """
    多进程共享：同时限制 RPM 和 TPM（1 分钟滑窗）。
    所有请求在发送前先 acquire() 排号，拿到许可再发。
    """
    def __init__(self, rpm:int, tpm:int, manager=None):
        self.rpm = int(rpm)
        self.tpm = int(tpm)
        self.m = manager or mp.Manager()
        self.req_ts = self.m.list()   # 每次请求的时间戳
        self.tok_ts = self.m.list()   # (ts, tokens_used)
        self.lock  = self.m.Lock()

    def _gc(self, now):
        cutoff = now - 60
        for L in (self.req_ts, self.tok_ts):
            i = 0
            while i < len(L):
                ts = L[i] if L is self.req_ts else L[i][0]
                if ts < cutoff:
                    L.pop(i)
                else:
                    i += 1

    def acquire(self, est_tokens:int):
        est_tokens = max(1, int(est_tokens or 1))
        while True:
            with self.lock:
                now = time.time()
                self._gc(now)
                used_rpm = len(self.req_ts)
                used_tpm = sum(t for _, t in list(self.tok_ts))
                if used_rpm < self.rpm and used_tpm + est_tokens <= self.tpm:
                    self.req_ts.append(now)
                    self.tok_ts.append((now, est_tokens))
                    return
                # 下一次最短等待（取 RPM/TPM 的最大等待）
                wait_rpm = max(0.0, 60 - (now - self.req_ts[0])) if self.req_ts else 0.0
                wait_tpm = 0.0
                if self.tok_ts:
                    acc = 0
                    for ts, tk in list(self.tok_ts):
                        acc += tk
                        if used_tpm - acc + est_tokens <= self.tpm:
                            wait_tpm = max(0.0, 60 - (now - ts))
                            break
                wait_s = max(wait_rpm, wait_tpm, 0.25)
            time.sleep(wait_s + random.uniform(0, 0.2))  # 轻微抖动，避免齐步撞限