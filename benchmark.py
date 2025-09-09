# This script is a simple performance test to verify that the core pricing function is fast enough to meet the latency requirements.
# It runs the `p()` function a large number of times and reports the average time per call.

import time, duopoly as algo
info=None
N=5000
t0=time.time()
for i in range(N):
    ctx = {'period_id': i, 'sales': 30.0, 'competitor_price': 55.0} if i else {'period_id':0}
    _, info = algo.p(ctx, info_dump=info)
dt = time.time()-t0
print("avg_ms_per_call:", (dt/N)*1000)
