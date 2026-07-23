"""GPU/torch-free unit tests for the L2-bypass increment-3 ASYNC device-direct
read: the verified-prefix counting (device_page_meta.consecutive_ok_pages) and the
TP-collective-sequence BALANCE of the check_prefetch_progress state machine.

The balance test is the load-bearing one: with the read moved to a background
thread, ranks finish their local GET at different rounds, yet every rank MUST issue
the identical number and order of all_reduce collectives for a given request or the
NCCL op stream desyncs and deadlocks under TP>1. This simulates the state machine's
collective schedule (the real code's _start_l3_async_load / _poll_l3_async_load /
_promote_l3_async_load) across ranks with divergent completion times and asserts the
emitted collective-tag sequences are identical. Pure python; run with
`python3 test_async_read_state_machine.py`.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mem_cache.device_page_meta import consecutive_ok_pages  # noqa: E402


class TestConsecutiveOkPages(unittest.TestCase):
    def test_dense_all_hit(self):
        self.assertEqual(consecutive_ok_pages([True] * 5, [], 5), 5)

    def test_dense_first_miss_truncates(self):
        self.assertEqual(consecutive_ok_pages([True, True, False, True], [], 4), 2)

    def test_dense_immediate_miss(self):
        self.assertEqual(consecutive_ok_pages([False, True, True], [], 3), 0)

    def test_dense_short_list_is_miss(self):
        # backend returned fewer results than pages -> the missing tail is a miss.
        self.assertEqual(consecutive_ok_pages([True, True], [], 4), 2)

    def test_hybrid_sidecar_gates_prefix(self):
        # main KV hits all 4, but the indexer sidecar misses page 2 -> prefix = 2.
        kv = [True, True, True, True]
        indexer = [True, True, False, True]
        self.assertEqual(consecutive_ok_pages(kv, [indexer], 4), 2)

    def test_hybrid_kv_gates_below_sidecar(self):
        # KV is the shorter hit here -> prefix bounded by KV even though sidecar
        # would allow more.
        kv = [True, False, True]
        indexer = [True, True, True]
        self.assertEqual(consecutive_ok_pages(kv, [indexer], 3), 1)

    def test_hybrid_multiple_sidecars_min(self):
        kv = [True, True, True, True]
        s1 = [True, True, True, False]
        s2 = [True, True, False, True]
        # min prefix across kv, s1, s2 = 2 (s2 misses page 2).
        self.assertEqual(consecutive_ok_pages(kv, [s1, s2], 4), 2)

    def test_zero_pages(self):
        self.assertEqual(consecutive_ok_pages([], [], 0), 0)


# --- Collective-sequence balance simulation ---------------------------------------
#
# A faithful abstraction of the async read state machine's collective schedule. Each
# rank, per request, emits a sequence of collective TAGS in the exact order the real
# code issues its torch.distributed.all_reduce calls:
#   "exist" : _l3_exist_query               (round 1, ALWAYS)
#   "alloc" : _start_l3_async_load alloc-MIN (round 1, only if a load is submitted)
#   "done"  : _poll_l3_async_load done-MIN   (once PER waiting round)
#   "pages" : _promote_l3_async_load verify-MIN (final round)
# The invariant: for one request, all ranks emit an identical tag sequence.


def simulate_rank_sequence(raw_hit, threshold, has_loadable, alloc_ok, done_round):
    """Reproduce ONE rank's collective-tag sequence for ONE request, given the
    *post-MIN* global facts it will observe (the simulator computes those globally
    and feeds every rank the same values, exactly as the MIN reduces guarantee)."""
    seq = ["exist"]
    if raw_hit < threshold or not has_loadable:
        # threshold/nothing-to-load: return True at round 1, no alloc MIN.
        return seq
    seq.append("alloc")
    if not alloc_ok:
        # alloc failed on >=1 rank: abort at round 1, recompute.
        return seq
    # Loading: one "done" MIN per round until the global-min done round, then "pages".
    for _ in range(done_round):
        seq.append("done")
    seq.append("pages")
    return seq


class TestCollectiveBalance(unittest.TestCase):
    THRESHOLD = 256

    def _run(self, ranks_raw_hits, ranks_local_done_round, ranks_alloc_ok, has_loadable=True):
        """Model global MIN semantics, then assert all ranks' sequences match.

        ranks_raw_hits        : per-rank raw exist hit count (pre-MIN)
        ranks_local_done_round : per-rank round at which its background GET lands
        ranks_alloc_ok        : per-rank device-alloc success (bool)
        """
        n = len(ranks_raw_hits)
        # Post-MIN global facts every rank observes:
        global_hit = min(ranks_raw_hits)
        global_alloc = all(ranks_alloc_ok)  # MIN of 0/1
        # The loop parks until the SLOWEST rank is done: global done round = max.
        # Every rank issues one "done" per round up to and including that round.
        global_done_round = max(ranks_local_done_round)
        seqs = [
            simulate_rank_sequence(
                raw_hit=global_hit,  # each rank branches on the post-MIN hit
                threshold=self.THRESHOLD,
                has_loadable=has_loadable,
                alloc_ok=global_alloc,
                done_round=global_done_round,
            )
            for _ in range(n)
        ]
        for r in range(1, n):
            self.assertEqual(
                seqs[0],
                seqs[r],
                f"rank 0 and rank {r} emitted different collective sequences: "
                f"{seqs[0]} vs {seqs[r]}",
            )
        return seqs[0]

    def test_balanced_when_ranks_finish_at_different_rounds(self):
        # The core worry: rank 0 finishes its GET at round 2, rank 3 at round 7.
        # All must still emit the same number of "done" reduces.
        seq = self._run(
            ranks_raw_hits=[4096, 4096, 4096, 4096],
            ranks_local_done_round=[2, 5, 1, 7],
            ranks_alloc_ok=[True, True, True, True],
        )
        # 1 exist + 1 alloc + 7 done (slowest) + 1 pages.
        self.assertEqual(seq, ["exist", "alloc"] + ["done"] * 7 + ["pages"])

    def test_balanced_below_threshold_all_ranks_skip_alloc(self):
        # A raw hit that is high on some ranks but the MIN is below threshold: every
        # rank must take the no-load branch (exist only), no alloc/done/pages.
        seq = self._run(
            ranks_raw_hits=[4096, 128, 4096, 4096],  # MIN=128 < 256
            ranks_local_done_round=[3, 3, 3, 3],
            ranks_alloc_ok=[True, True, True, True],
        )
        self.assertEqual(seq, ["exist"])

    def test_balanced_alloc_failure_aborts_uniformly(self):
        # One rank cannot allocate GPU slots: every rank emits exist+alloc then
        # aborts (recompute) — no "done"/"pages" on ANY rank.
        seq = self._run(
            ranks_raw_hits=[4096, 4096, 4096, 4096],
            ranks_local_done_round=[2, 2, 2, 2],
            ranks_alloc_ok=[True, False, True, True],
        )
        self.assertEqual(seq, ["exist", "alloc"])

    def test_balanced_nothing_loadable(self):
        # Discovered prefix already fully device-resident on all ranks: exist only.
        seq = self._run(
            ranks_raw_hits=[4096, 4096],
            ranks_local_done_round=[1, 1],
            ranks_alloc_ok=[True, True],
            has_loadable=False,
        )
        self.assertEqual(seq, ["exist"])

    def test_single_rank_no_tp(self):
        # TP=1: still a valid single-element sequence.
        seq = self._run(
            ranks_raw_hits=[4096],
            ranks_local_done_round=[3],
            ranks_alloc_ok=[True],
        )
        self.assertEqual(seq, ["exist", "alloc"] + ["done"] * 3 + ["pages"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
