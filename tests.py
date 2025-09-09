import unittest
import numpy as np

# Import the module we are testing
import duopoly as algo

MID = (algo.P_MIN + algo.P_MAX) / 2.0

def get_period(info):
    """Handles both old ('period') and new ('t') state shapes."""
    if 'period' in info:
        return info['period']
    if 't' in info:
        return info['t']
    raise KeyError("No 'period' or 't' in info dump")

class TestPricingBot(unittest.TestCase):
    def test_coldstart_midprice(self):
        # On the first call, we should return the safe mid-price.
        price, info = algo.p({})
        self.assertTrue(np.isfinite(price))
        self.assertAlmostEqual(price, MID, places=6)
        self.assertEqual(get_period(info), 1)

    def test_bounds_and_comp_cap(self):
        # 1) price stays within [P_MIN, P_MAX]
        # 2) price is strictly below the competitor when competitor is lower
        price, info = algo.p({})
        ctx = {'sales': 20.0, 'competitor_price': 40.0}
        price2, info2 = algo.p(ctx, info_dump=info)
        self.assertGreaterEqual(price2, algo.P_MIN - 1e-9)
        self.assertLess(price2, 40.0)  # strictly below competitor (pc - 0.01)
        self.assertEqual(get_period(info2), 2)

    def test_period_rollover_and_rb_capacity(self):
        # RingBuffer should be bounded and period should roll correctly.
        info = None
        steps = 150
        last_q = None
        for t in range(steps):
            ctx = {} if t == 0 else {'sales': last_q, 'competitor_price': 55.0}
            price, info = algo.p(ctx, info_dump=info)
            last_q = 30.0  # constant feedback
        self.assertEqual(get_period(info), steps)
        # first step has no feedback; rb count is steps-1 but bounded by RB_SIZE
        expected = min(algo.RB_SIZE, steps - 1)
        self.assertEqual(info['rb']['count'], expected)

    def test_cost_floor(self):
        # Never price below cost + margin.
        price, info = algo.p({})
        ctx = {'sales': 25.0, 'competitor_price': 80.0, 'cost': 60.0}
        price2, info2 = algo.p(ctx, info_dump=info)
        self.assertGreaterEqual(price2, 60.0 + 0.1 - 1e-9)

if __name__ == '__main__':
    unittest.main()
