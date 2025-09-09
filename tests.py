
import unittest
import math
import numpy as np

# Import the module we are testing
import duopoly as algo

MID = (algo.P_MIN + algo.P_MAX) / 2.0

class TestPricingBot(unittest.TestCase):
    def test_coldstart_midprice(self):
        # This test verifies that on the very first call, our policy returns the safe, default price.
        # This is a critical check for ensuring the agent behaves predictably before it has any data.
        price, info = algo.p({})
        self.assertTrue(np.isfinite(price))
        self.assertAlmostEqual(price, MID, places=6)
        self.assertEqual(info['period'], 1)

    def test_bounds_and_comp_cap(self):
        # This test ensures two important business rules are followed:
        # 1. The price is always within the min/max bounds.
        # 2. The price is always slightly below the competitor's price if the competitor is lower.
        price, info = algo.p({})
        # next call with competitor below mid
        ctx = {'sales': 20.0, 'competitor_price': 40.0}
        price2, info2 = algo.p(ctx, info_dump=info)
        self.assertGreaterEqual(price2, algo.P_MIN - 1e-9)
        self.assertLess(price2, 40.0)  # must be strictly below competitor (pc - 0.01)
        self.assertEqual(info2['period'], 2)

    def test_period_rollover_and_rb_capacity(self):
        # This is a stress test for the memory management. It simulates many periods to
        # make sure that the RingBuffer correctly overwrites old data and doesn't grow beyond its capacity.
        info = None
        steps = 150
        last_q = None
        for t in range(steps):
            ctx = {} if t == 0 else {'sales': last_q, 'competitor_price': 55.0}
            price, info = algo.p(ctx, info_dump=info)
            last_q = 30.0  # simple constant feedback
        self.assertEqual(info['period'], steps)
        # first step has no feedback; rb count is steps-1 but bounded by RB_SIZE
        expected = min(algo.RB_SIZE, steps - 1)
        self.assertEqual(info['rb']['count'], expected)

    def test_cost_floor(self):
        # This test verifies that the agent never price below floor cost, which is a fundamental business constraint.
        price, info = algo.p({})
        ctx = {'sales': 25.0, 'competitor_price': 80.0, 'cost': 60.0}
        price2, info2 = algo.p(ctx, info_dump=info)
        self.assertGreaterEqual(price2, 60.0 + 0.1 - 1e-9)

if __name__ == '__main__':
    unittest.main()
