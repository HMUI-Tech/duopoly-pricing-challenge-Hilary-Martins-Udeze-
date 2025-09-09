# Simple performance test for p() latency: prints avg seconds per call

import time, duopoly as algo

info = None
N = 5000
t0 = time.time()
for i in range(N):
    ctx = {'period_id': i, 'sales': 30.0, 'competitor_price': 55.0} if i else {'period_id': 0}
    _, info = algo.p(ctx, info_dump=info)
dt = time.time() - t0

avg_s_per_call = dt / N
print("avg_s_per_call:", avg_s_per_call)
