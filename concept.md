# Comprehensive Mathematical Pipeline & Deployment Guide for HMADRL

## 1. Introduction to the Architecture

Hierarchical Multi-Agent Deep Reinforcement Learning (HMADRL) operates via a tripartite architecture:
1. **The Top Manager:** An RL (PPO) or MoE (Mixture of Experts) router determining capital allocation across macro-domains.
2. **The Stochastic Controller:** A classical quantitative math layer applying Merton portfolio theory and mean-reverting (Ornstein-Uhlenbeck) dynamics to constrain the AI.
3. **The Domain Managers:** Specific RL agents (SAC) that determine micro-allocations (individual stock weights) within their respective domains.

This document serves as an extremely detailed step-by-step mathematical trace of a single training iteration, including dummy data to precisely illustrate the calculations.

---

## 2. Phase 1: Data Ingestion and Feature Engineering

Imagine our universe consists of 4 stocks: AAPL, MSFT (Tech) and JPM, BAC (Finance).
At time $T=120$, we want to generate features based on a lookback window of $L=20$ days.

### 2.1 Price and Return Data
Suppose the raw Adjusted Close prices for the last 20 days are $P_{t-20} \dots P_{t}$.
For AAPL:
- $P_{t-20} = 150.00$
- $P_{t} = 165.00$

### 2.2 Feature Calculation (data_api.py)
**Momentum ($\mu$):**
$$ \mu_{AAPL} = \frac{P_{t}}{P_{t-20}} - 1.0 = \frac{165.00}{150.00} - 1.0 = 0.10 \ (10\%) $$

**Volatility ($\sigma$):**
Calculated as the annualized standard deviation of daily returns. Suppose daily std dev of AAPL returns is 0.015 (1.5%).
$$ \sigma_{AAPL} = 0.015 \times \sqrt{252} \approx 0.238 \ (23.8\%) $$

**Liquidity ($L$):**
Average volume over the lookback. Suppose average volume is 50,000,000 shares.
$$ L_{AAPL} = 50,000,000 $$

We compute this for all 4 stocks. Let's assume the following features:
- **AAPL:** $\mu = 0.10$, $\sigma = 0.238$, $L = 5e7$
- **MSFT:** $\mu = 0.08$, $\sigma = 0.210$, $L = 3e7$
- **JPM:** $\mu = -0.02$, $\sigma = 0.180$, $L = 1e7$
- **BAC:** $\mu = -0.05$, $\sigma = 0.200$, $L = 2e7$

---

## 3. Phase 2: Dynamic Clustering (The Leakage-Free Domains)

When `use_learned_clusters` is True, we don't rely on static "Tech" or "Finance" labels. Instead, we cluster dynamically based on correlation.

### 3.1 Correlation Matrix
Using the training window (e.g., 504 days), we compute the Pearson correlation matrix.
Suppose the correlation matrix $C$ is:
| Stock | AAPL | MSFT | JPM | BAC |
|-------|------|------|-----|-----|
| AAPL  | 1.0  | 0.85 | 0.20| 0.15|
| MSFT  | 0.85 | 1.0  | 0.25| 0.20|
| JPM   | 0.20 | 0.25 | 1.0 | 0.90|
| BAC   | 0.15 | 0.20 | 0.90| 1.0 |

### 3.2 Distance Matrix
Distance is $D = 1.0 - C$.
| Stock | AAPL | MSFT | JPM | BAC |
|-------|------|------|-----|-----|
| AAPL  | 0.0  | 0.15 | 0.80| 0.85|
| MSFT  | 0.15 | 0.0  | 0.75| 0.80|
| JPM   | 0.80 | 0.75 | 0.0 | 0.10|
| BAC   | 0.85 | 0.80 | 0.10| 0.0 |

### 3.3 K-Medoids Initialization
1. Choose highest variance stock as first seed. Say AAPL.
2. Choose second seed to maximize distance from AAPL. Max distance is BAC (0.85).
Seeds: [AAPL, BAC].

3. Assign remaining stocks to nearest seed.
- MSFT distance to AAPL = 0.15. Distance to BAC = 0.80. MSFT -> AAPL.
- JPM distance to AAPL = 0.80. Distance to BAC = 0.10. JPM -> BAC.

Clusters formed:
- **cluster_00:** [AAPL, MSFT]
- **cluster_01:** [JPM, BAC]

---

## 4. Phase 3: State Construction

We build the state representations for the RL agents.

### 4.1 Domain Aggregation
For `cluster_00` (Tech), we average the features of AAPL and MSFT.
- $\mu_{c00} = (0.10 + 0.08) / 2 = 0.09$
- $\sigma_{c00} = (0.238 + 0.210) / 2 = 0.224$
- $L_{c00} = (5e7 + 3e7) / 2 = 4e7$

For `cluster_01` (Finance):
- $\mu_{c01} = (-0.02 + -0.05) / 2 = -0.035$
- $\sigma_{c01} = (0.180 + 0.200) / 2 = 0.190$
- $L_{c01} = (1e7 + 2e7) / 2 = 1.5e7$

### 4.2 Top-Level State Vector
The `TopLevelState` concatenates domain features, current allocations, and global macro variables.
$S_{top} = [ \mu_{c00}, \mu_{c01}, \sigma_{c00}, \sigma_{c01}, L_{c00}, L_{c01}, \text{alloc}_{c00}, \text{alloc}_{c01}, \text{macro}_{vol}, \dots ]$

Assuming equal initial allocation (0.5, 0.5) and normalized features:
$S_{top} = [0.09, -0.035, 0.224, 0.190, 0.8 (norm), 0.3 (norm), 0.5, 0.5, \dots]$

---

## 5. Phase 4: Top Manager Action (PPO)

The `RLTopManager` receives $S_{top}$. It runs an ensemble of PPO networks.
Let's assume the ensemble size is 1 for simplicity.
The policy network (MLP or Transformer) outputs a Dirichlet or Softmax distribution over domains.

### 5.1 Network Forward Pass
Logits:
- $L_{c00} = 2.5$
- $L_{c01} = 0.5$

Softmax probabilities (Base Weights):
- $W_{c00} = \frac{e^{2.5}}{e^{2.5} + e^{0.5}} \approx \frac{12.18}{12.18 + 1.64} = 0.881$
- $W_{c01} = 1.0 - 0.881 = 0.119$

### 5.2 Uncertainty Attenuation
If the ensemble disagrees, uncertainty $\sigma_{ensemble}$ is high.
$W'_{i} = W_i \cdot e^{-\alpha \cdot \sigma_{ensemble}}$
Assume low uncertainty, so weights remain $0.881$ and $0.119$.

### 5.3 Hold Steps
The network also outputs a continuous value for hold steps, scaled to `[min_hold, max_hold]`.
Assume output maps to $H_{top} = 5$ days.

Action: $\{ \text{cluster\_00}: 0.881, \text{cluster\_01}: 0.119 \}$, Hold: 5.

---

## 6. Phase 5: Stochastic Control (The Mathematical Bridge)

This is where classical finance overrides pure RL to ensure mathematical safety. The top weights are transformed into `DomainControlSignal`.

### 6.1 Merton Fraction Calculation
For `cluster_00`:
- $\mu = 0.09$
- $\sigma = 0.224$
- Risk Aversion $\gamma = 3.0$

*Note: This is the inconsistency identified!* $\mu$ is 20-day return, $\sigma$ is annualized. For this manual trace, we follow the code exactly.
$$ merton_{c00} = \frac{\mu}{\gamma \cdot \sigma^2} = \frac{0.09}{3.0 \cdot (0.224)^2} = \frac{0.09}{3.0 \cdot 0.050} = \frac{0.09}{0.150} = 0.60 $$

For `cluster_01`:
$$ merton_{c01} = \frac{-0.035}{3.0 \cdot (0.190)^2} = \frac{-0.035}{0.108} = -0.324 $$

### 6.2 Ornstein-Uhlenbeck Latent Dynamics
The latent state $x$ reverts to the Merton fraction.
$$ x_{t+1} = x_t + \kappa (w^* - x_t) + \sigma \cdot 0.1 \cdot \epsilon $$
Assume $x_t = 0$, $\kappa = 0.18$, $\epsilon = 0.5$ (random noise).

For `cluster_00`:
$$ x_{t+1} = 0 + 0.18 \cdot (0.60 - 0) + 0.224 \cdot 0.1 \cdot 0.5 = 0.108 + 0.0112 = 0.1192 $$

### 6.3 Budget Scoring
The control term scales the raw RL budget:
$C_{term} = 0.5 \cdot \tanh(x_{t+1}) + 0.5 \cdot \text{confidence}$
Assume confidence $\approx 0.8$.
$C_{term} = 0.5 \cdot \tanh(0.1192) + 0.4 \approx 0.5 \cdot 0.118 + 0.4 = 0.459$

Final Budget Score (Simplified):
$Score_{c00} = W_{c00} \cdot (0.6 + 0.4 \cdot C_{term}) \approx 0.881 \cdot (0.6 + 0.183) = 0.881 \cdot 0.783 = 0.689$

The Stochastic Controller scales the raw 0.881 down to 0.689 (relative) due to the OU dynamics and market stress modeling.

---

## 7. Phase 6: Domain Manager Action (SAC)

Each domain has an independent Soft Actor-Critic agent.

For `cluster_00` (Tech), the agent receives $S_{domain}$ containing AAPL and MSFT features, plus the control signal (Budget=0.689).
SAC outputs a distribution over the 2 stocks.
Because it's off-policy, it samples from a Gaussian policy (Reparameterization Trick).
- $W_{AAPL} = 0.60$
- $W_{MSFT} = 0.40$

For `cluster_01` (Finance):
- $W_{JPM} = 0.70$
- $W_{BAC} = 0.30$

### 7.1 Final Portfolio Construction
The global portfolio weights are the product of the controlled domain budget and the domain stock weights.
Normalize budgets: $B_{c00} = 0.85$, $B_{c01} = 0.15$ (after normalization).
- Global AAPL = $0.85 \times 0.60 = 0.51 \ (51\%)$
- Global MSFT = $0.85 \times 0.40 = 0.34 \ (34\%)$
- Global JPM = $0.15 \times 0.70 = 0.105 \ (10.5\%)$
- Global BAC = $0.15 \times 0.30 = 0.045 \ (4.5\%)$

Total global weights = $1.0$.

---

## 8. Phase 7: Environment Step and Reward Math

We step the environment forward by 1 day to $T=121$.
Observe next day returns:
- AAPL: $+2.0\%$ (0.02)
- MSFT: $+1.0\%$ (0.01)
- JPM: $-1.0\%$ (-0.01)
- BAC: $-2.0\%$ (-0.02)

### 8.1 Gross Return Calculation
$$ R_{gross} = (0.51 \times 0.02) + (0.34 \times 0.01) + (0.105 \times -0.01) + (0.045 \times -0.02) $$
$$ R_{gross} = 0.0102 + 0.0034 - 0.00105 - 0.0009 = 0.01165 \ (1.165\%) $$

### 8.2 Turnover Calculation
Assume previous weights were all 25% (0.25).
Turnover $T = \sum | W_{new} - W_{old} |$
- AAPL: $|0.51 - 0.25| = 0.26$
- MSFT: $|0.34 - 0.25| = 0.09$
- JPM: $|0.105 - 0.25| = 0.145$
- BAC: $|0.045 - 0.25| = 0.205$
$$ T = 0.26 + 0.09 + 0.145 + 0.205 = 0.70 \ (70\%) $$

### 8.3 Cost Deductions
- Transaction Cost BPS = 5.0 (0.0005)
- Slippage BPS = 2.5 (0.00025)
$ Cost = T \times (0.0005 + 0.00025) = 0.70 \times 0.00075 = 0.000525 $

### 8.4 Normalized Reward (Highlighting Inconsistency)
Code executes:
$$ Cost_{norm} = \frac{Cost}{1.0 + |R_{gross}|} = \frac{0.000525}{1.0 + 0.01165} = 0.000518 $$
*(Bug: Dividing by gross return distorts economic reality).*

$$ R_{final} = R_{gross} - Cost_{norm} - \dots $$
$$ R_{final} = 0.01165 - 0.000518 = 0.011132 $$

---

## 9. Phase 8: Model Backpropagation (Learning)

The final reward $R_{final} = 0.011132$ is dispatched back to the agents.

### 9.1 Top Manager (PPO) Update
- Computes Advantage $A_t = R_{final} + \gamma V(S_{t+1}) - V(S_t)$
- Updates policy network to maximize $\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)$
- High reward reinforces the decision to overweight `cluster_00`.

### 9.2 Domain Managers (SAC) Update
- The experience $(S_{domain}, A_{domain}, R_{final}, S_{domain\_next})$ is stored in the Replay Buffer.
- SAC updates Q-networks to minimize temporal difference error.
- SAC updates Policy network to maximize expected Q-value plus entropy bonus.

---

## 10. The Walk-Forward Execution Loop (Pipeline Strategy)

The above iteration happens over and over in a **sliding window** fashion.

1. **Window 1 (2010 - 2012):**
   - Cluster stocks dynamically based on 2010-2012 correlation.
   - Initialize new Neural Networks.
   - Run the 8-phase loop above for thousands of steps.
   - Test on 2013 (Deterministic out-of-sample). Record returns.
2. **Window 2 (2011 - 2013):**
   - Cluster stocks dynamically based on 2011-2013 correlation.
   - Initialize *brand new* Neural Networks (The Tabula Rasa problem).
   - Run the loop again.
   - Test on 2014. Record returns.

At the end, the test returns are stitched together to form a highly rigorous, non-leaking equity curve spanning the entire decade.

---

## 11. Live Deployment Architecture

If an institution wishes to deploy HMADRL into a live production trading environment, the standard `model.load("weights.pth")` paradigm **will not work** due to the dynamic clustering and the Tabula Rasa problem.

### 11.1 Infrastructure Setup
1. **Data Pipeline:** An automated chron job pulling daily Adjusted Close and Volume data via `yfinance` or Bloomberg API.
2. **Compute:** A GPU-enabled server (AWS EC2 / GCP Compute) capable of training a PPO/SAC ensemble in under a few hours.
3. **Execution Engine:** A broker API (e.g., Interactive Brokers) to execute the final target weights.

### 11.2 Daily / Rebalance Operational Flow
Instead of just inferencing a pre-trained model, you must **train a new model every rebalance period**.

**Step 1: Data Fetching (T-0)**
At market close (4:00 PM EST), fetch the last 504 days of price data up to today.

**Step 2: Dynamic Clustering (T-0)**
Run `_learned_cluster_map` on the 504-day window to determine the current market correlation clusters. 
- *Why?* Because what constituted a "highly correlated group" 6 months ago might have broken down today.

**Step 3: Just-In-Time Training (T-0 to T+1)**
- Initialize a completely fresh Hierarchical Agent (`RLTopManager` + `SAC Domain Managers`).
- Run the training loop (`_train_step`) over the 504-day window for several epochs (e.g., `fine_tune_rounds = 5`).
- Wait for the models to converge (loss plateau).

**Step 4: Inference for Tomorrow**
- Feed today's market state $S_t$ into the newly trained, converged agent in **deterministic mode**.
- Extract the Final Stock Weights (e.g., 51% AAPL, 34% MSFT, 10.5% JPM, 4.5% BAC).

**Step 5: Trade Execution (T+1)**
At market open the next day, execute trades to align the actual portfolio with the target weights.

### 11.3 Why JIT Training is Mandatory Here
If you save the model from 2024 and try to run it in 2026, the inputs to the neural network will be structurally invalid. The network trained node 0 to trade "Cluster 00". In 2024, Cluster 00 was Tech. In 2026, Cluster 00 might be Energy. The neural network's learned representations are tied to the transient identity of the clusters. 

Therefore, **Just-In-Time (JIT) Retraining** on a rolling window is the only mathematically and structurally sound way to deploy this specific codebase.

---

## 12. Complete Resolution of Mathematical Inconsistencies

If attempting to fork or productize this repository, the following mathematical patches are mandatory to prevent catastrophic capital loss:

### 12.1 The Merton Patch
In `hmadrl/stochastic_control.py`, the temporal scales must match.
```python
# CURRENT FLAWED CODE:
# mu is lookback return (e.g., 120 days)
# sigma is annualized daily volatility
merton = mu / (gamma * sigma * sigma)

# PATCH REQUIRED:
# Convert mu to annualized return
annualization_factor = 252.0 / lookback_days
mu_annualized = mu * annualization_factor
merton = mu_annualized / (gamma * sigma * sigma)
```
Without this patch, the stochastic controller chronically underestimates the Merton fraction, leading to excessive cash drags or unnecessarily tight risk controls.

### 12.2 The Cost Normalization Patch
In `hmadrl/pipeline.py`, costs are artificially deflated during high-volatility environments.
```python
# CURRENT FLAWED CODE:
cost_norm = (tx + sl + tp) / (1.0 + abs(gross))

# PATCH REQUIRED:
# Costs must hit the bottom line purely.
net_return = gross - tx - sl
turnover_penalty = tp * turnover  # Abstract algorithmic penalty
reward = net_return - turnover_penalty - downside_penalty
```

### 12.3 Architecture Refactor for Continuous Learning (Optional)
To avoid the computationally heavy JIT Retraining, the architecture must be refactored to use **Permutation Invariant Networks** (like Deep Sets or Transformers without positional encoding for clusters).
If the state representation encodes the *properties* of the cluster rather than assigning it a fixed integer ID, a single global RL model could be trained across 20 years of data and deployed instantly in production without requiring nightly retraining. 

---

## 13. Summary
The HMADRL framework is a brilliantly conceived fusion of classical quantitative stochastic control and modern multi-agent deep reinforcement learning. Its strict adherence to leakage-free walk-forward optimization makes it academically rigorous. However, due to its localized dynamic clustering, it operates as a sophisticated sliding-window optimizer rather than a generalized global intelligence. Deploying it requires significant automated ML-Ops infrastructure to support daily Just-In-Time retraining.
