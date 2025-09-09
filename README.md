# Duopoly Pricing Agent – Business & Technical Summary

## Introduction
When I designed this pricing agent, I didn’t want it to just “work” under competition rules.  
I wanted something that a **business manager could trust**, that a **data scientist could defend**, and that an **engineer could run** without breaking constraints.  

Every choice was about balance:
- **Profit vs. Exploration**
- **Safety vs. Adaptability**
- **Simplicity vs. Power**

---

## 1. State Design – Lean but Capable
At first, I considered keeping *all history* so the agent could see every past event.  
But I realized this would blow up memory and make the agent slow.  

I settled on a **ring buffer (128 steps)** with incremental statistics (EWMA means and variances).  

- **Why 128?** Large enough to capture meaningful patterns, but small enough to adapt quickly.  
- **Advantages:** Constant-time updates, bounded memory, smooth adaptation.  
- **Drawbacks:** Long-term seasonality beyond ~128 steps is lost.  
- **Business value:** Predictable performance, never stalls over time.

---

## 2. Demand Learning – Fast and Adaptive
I wanted a demand model that adapts in real time.  
- **Tried:** Full regressions → too slow.  
- **Tried:** Exponential smoothing → light, but ignored elasticity.  

**Final choice:** A **decayed OLS regression** updating running sums (Sx, Sy, Sxx, Sxy).  

- **Advantages:** Adapts to shocks (recent data weighted more), efficient constant-time updates, interpretable.  
- **Drawbacks:** Assumes linear demand curve.  
- **Business value:** Forecasts are simple, explainable, and fast enough for the 0.2s runtime budget.

---

## 3. Pricing Policy – A Hybrid Approach
I went through three phases:

1. **Greedy OLS** – stable, but blind to new opportunities.  
2. **Pure UCB bandit** – curious, but wasted money on bad prices.  
3. **Hybrid (final choice)** – OLS “favorite” price, occasionally challenged by UCB.

### Parameter Trade-offs

| Parameter | Tried | Outcome | Final Choice |
|-----------|-------|---------|--------------|
| **Window (W)** | 50 (too noisy), 100 (too slow) | W=80 balanced noise vs. adaptation | **W=80** |
| **Grid (K)** | 21 (coarse), 41 (slower, little gain) | K=31 gave fine precision | **K=31** |
| **Exploration (ε)** | 0.10 (jittery), 0.02 (converged too early) | 0.05 kept balance | **ε=0.05** |
| **UCB Constant (C)** | 1.5 (over-explored), 0.8 (too cautious) | 1.0 balanced | **C=1.0** |

- **Advantages:** Learns quickly, stabilizes later, robust under shocks.  
- **Drawbacks:** May still under-explore if shocks are very rare.  
- **Business value:** Fast early learning, later stability for profit protection.  
- **Unique angle:** Few candidates combine **bandit exploration** with **regression accuracy**.

---

## 4. Cold Start & Safety Rails
During the first rounds, the agent uses a **safe mid-price** until data arrives.  

Safety rules:
- Never below cost + buffer.  
- Never too far above competitor.  
- Always within [P_MIN, P_MAX].  

- **Business value:** No reckless losses or unrealistic pricing, even when demand data is missing.  
- **Drawback:** More conservative start.

---

## 5. Development Journey – Resilience
During testing, the agent misbehaved when the competitor had **no capacity**.  
Instead of ignoring it, I added a **specific override check** to handle the scenario.  

- **Business value:** This kind of resilience prevents embarrassing failures in production.  
- **Interview value:** Shows I can anticipate and fix edge cases.

---

## 6. Transparency – The Decision Card
I added a **Decision Card** inside the state object.  

Each round, it stores:
- OLS price, UCB price, chosen price  
- Exploration rate ε, UCB bonus  
- Applied constraints (cost floor, competitor cap)  
- A short reason string (e.g., `"OLS within Δ=0.5 of UCB; smoother path chosen"`)

- **Business value:** A manager or analyst can audit pricing decisions without needing to read the code.

---

## 7. What I Rejected
- **Unbounded history** → risk of crashes.  
- **Deep RL / heavy optimizers** → too slow, disallowed, unpredictable.  
- **Fixed rules only** → too rigid, no adaptation.  

---

## 8. Business Takeaway
This agent maximizes revenue **safely under constraints**:
- Learns from real-time sales & competitor prices.  
- Balances exploration with profit focus.  
- Avoids reckless moves (cost floor, safe bounds).  
- Explains decisions clearly to both engineers and managers.  

It is **lean, adaptive, and trustworthy** — fit for both technical evaluation and real-world business use.  

---

## Appendix – Technical Footnotes
For less technical readers, here are the main formulas:

- **EWMA (Exponentially Weighted Moving Average)**  
  \[
  \mu_t = \alpha x_t + (1-\alpha)\mu_{t-1}
  \]  
  *Gives more weight to recent sales data.*

- **Decayed OLS Regression**  
  Regression with weights that decay over time: recent data counts more than old data.  
  Coefficient estimate:
  \[
  \hat{\beta} = \frac{S_{xy}}{S_{xx}}
  \]

- **UCB (Upper Confidence Bound)**  
  \[
  p = \hat{r} + C \sqrt{\frac{\ln t}{n}}
  \]  
  *Chooses between exploring less-tested prices and exploiting the best-known price.*

- **ε-greedy**  
  With small probability ε, try a random price to avoid getting stuck.

---
