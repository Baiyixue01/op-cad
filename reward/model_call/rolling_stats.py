# rolling_stats.py
import multiprocessing as mp

class MPRollingStats:
    """
    多进程共享滚动统计（不限制 max_tokens）：
    - 分别统计 in_tokens / out_tokens / latency
    - 提供平均值（也可按需改成指数滑动平均 EMA）
    """
    def __init__(self, manager=None):
        m = manager or mp.Manager()
        self.lock = m.Lock()
        self.n = m.Value('i', 0)
        self.sum_in = m.Value('d', 0.0)
        self.sum_out = m.Value('d', 0.0)
        self.sum_latency = m.Value('d', 0.0)

    def add_io(self, in_tokens:int, out_tokens:int, latency_s:float):
        with self.lock:
            self.n.value += 1
            self.sum_in.value += max(0, in_tokens or 0)
            self.sum_out.value += max(0, out_tokens or 0)
            self.sum_latency.value += max(0.0, latency_s or 0.0)

    def averages_io(self, default_in=1500, default_out=2048, default_L=2.0):
        with self.lock:
            n = self.n.value
            if n <= 0:
                return default_in, default_out, default_L
            return (
                self.sum_in.value / n,
                self.sum_out.value / n,
                self.sum_latency.value / n
            )
