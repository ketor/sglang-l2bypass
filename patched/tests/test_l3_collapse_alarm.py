"""七期: L3 命中崩塌告警的行为测试。

守护的故障签名: exist 报页在 L3 → device GET 整批 0 命中 → 丢 marker 重算。
单次是合法降级；连续发生 = L3 对该实例已失效，而且**完全静默**。
本测试锁住"何时该叫、何时不该叫"，避免它退化成噪音或哑巴。
"""
import ast, os, sys, types, unittest

PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HRC = os.path.join(PATCHED, "mem_cache", "hiradix_cache.py")


class _Clock:
    def __init__(self): self.t = 1000.0
    def monotonic(self): return self.t


clock = _Clock()
LOGS = []


def _extract(name):
    with open(HRC) as f:
        tree = ast.parse(f.read(), filename=HRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"time": clock,
          "logger": types.SimpleNamespace(
              error=lambda *a: LOGS.append(a[0] % a[1:]),
              warning=lambda *a, **k: None, info=lambda *a, **k: None)}
    mod = ast.Module(body=[fn], type_ignores=[]); ast.fix_missing_locations(mod)
    exec(compile(mod, HRC, "exec"), ns)
    return ns[name]


NOTE = _extract("_note_l3_load_outcome")


class C:
    note = NOTE
    def __init__(self):
        self._l3_load_ok = 0; self._l3_load_fail = 0
        self._l3_collapse_logged_at = 0.0


class Alarm(unittest.TestCase):
    def setUp(self):
        LOGS.clear(); clock.t = 1000.0

    def test_silent_below_min_sample(self):
        """样本不足时不叫——否则启动初期几次正常降级就会误报。"""
        c = C()
        for _ in range(7):
            c.note(ok=False)
        self.assertEqual(LOGS, [])

    def test_silent_when_failures_are_a_minority(self):
        """偶发降级是正常的，不该叫。"""
        c = C()
        for _ in range(20):
            c.note(ok=True)
        for _ in range(3):
            c.note(ok=False)
        self.assertEqual(LOGS, [])

    def test_fires_on_sustained_collapse(self):
        c = C()
        for _ in range(8):
            c.note(ok=False)
        self.assertEqual(len(LOGS), 1)
        self.assertIn("L3 HIT COLLAPSE", LOGS[0])
        self.assertIn("8/8", LOGS[0])

    def test_throttled_to_one_per_minute(self):
        c = C()
        for _ in range(8):
            c.note(ok=False)
        for _ in range(50):
            c.note(ok=False)
        self.assertEqual(len(LOGS), 1, "一分钟内只应叫一次")
        clock.t += 61
        c.note(ok=False)
        self.assertEqual(len(LOGS), 2)

    def test_message_carries_triage_instructions(self):
        """告警必须自带下一步——否则半夜看到它的人无从下手。"""
        c = C()
        for _ in range(8):
            c.note(ok=False)
        m = LOGS[0]
        self.assertIn("dfkv_cache_miss_total", m)
        self.assertIn("MISS/SHORT", m)

    def test_both_call_sites_wired(self):
        """成功侧也必须计数，否则只有分母没有分子，比率恒为 100%。"""
        with open(HRC) as f:
            src = f.read()
        self.assertIn("self._note_l3_load_outcome(ok=True)", src)
        self.assertIn("self._note_l3_load_outcome(ok=False)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
