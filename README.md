# 🏈 Fantasy Football Player Valuation & Trade Target Engine (`ffbEval`)

An end-to-end Data Science & Machine Learning pipeline that uses NFL play-by-play opportunity metrics, position-specific **Expected Fantasy Points ($xFP$)**, **Empirical Bayesian Rate Shrinkage**, **Zero-Leakage Defense vs. Position (DvP) Schedule Adjustments**, and **Out-of-Time Walk-Forward Regression Models** to identify **undervalued (Buy Low)** and **overvalued (Sell High)** players in fantasy football.

---

## ⚡ Quickstart & Pipeline Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Data & Modeling Pipeline (Optional — Precomputed CSVs Included in `data/`)
```bash
# Step 1: Fetch live NFL weekly stats, schedules, and injury reports (2023–present)
python 01_fetch_data.py

# Step 2: Compute Position-Specific xFP, Bayesian Shrinkage, and Zero-Leakage DvP Adjustments
python 02_feature_engineering.py

# Step 3 (Optional): Run Position-Specific Walk-Forward Model Benchmarks & VIF/AIC/BIC Audits
python 03_train_qb_model.py
python 03_train_rb_model.py
python 03_train_wr_model.py
python 03_train_te_model.py

# Step 4: Generate Position-Decoupled Projections & Percentile Valuation Signals
python 04_generate_buy_sell_signals.py
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📂 Repository Structure

* [`01_fetch_data.py`](01_fetch_data.py): Dynamic Polars ingestion via `nflreadpy` for historical + live NFL seasons (weekly stats, injury reports, and upcoming opponent schedule lookups).
* [`02_feature_engineering.py`](02_feature_engineering.py): Computes position-specific Expected Fantasy Points ($xFP$), Empirical Bayesian Shrinkage, and Zero-Leakage Defense vs. Position (DvP) & Schedule-Adjusted rolling features.
* [`03_train_qb_model.py`](03_train_qb_model.py): Quarterback walk-forward benchmark suite (Dual-threat rushing + passing air yards + DvP).
* [`03_train_rb_model.py`](03_train_rb_model.py): Running Back walk-forward benchmark suite (VIF audit, Stepwise Bilateral AIC, Pure Opportunity + DvP).
* [`03_train_wr_model.py`](03_train_wr_model.py): Wide Receiver walk-forward benchmark suite (Target share, air yards, Pure Opportunity + DvP).
* [`03_train_te_model.py`](03_train_te_model.py): Tight End walk-forward benchmark suite (Stepwise BIC vs. AIC on Pure Opportunity infrastructure).
* [`04_generate_buy_sell_signals.py`](04_generate_buy_sell_signals.py): Trains the 4 decoupled position champion models, preserves neutral-matchup per-game projections on bye weeks, and assigns dynamic positional quantile valuation signals.
* [`optimize_bayesian_priors.py`](optimize_bayesian_priors.py): Empirical Bayes ANOVA variance decomposition ($\sigma_{\text{within}}^2 / \sigma_{\text{between}}^2$) to derive positional stabilization constants ($m_{\text{pos}}$).
* [`app.py`](app.py): Interactive Streamlit web application featuring a 4-quadrant Plotly scatter plot, position-filtered trade target leaderboard, player trajectory charts, and a 1-click live NFL data refresh button.
* [`DIRECTORY.md`](DIRECTORY.md): Living project directory and statistical reference log.

---

## 📌 1. FUTURE ENHANCEMENTS (Planned Roadmap)

1. **Position-Specific Model Training & Tuning (Completed across QB, RB, WR, TE)**:
   * Built 4 decoupled regression models (QB, RB, WR, TE) tailored to the unique drivers of each position group rather than using a single global model.
2. **Dual-Threat Quarterback Engine (Completed for QB Model)**:
   * Explicitly modeled QB rushing opportunities (carries, rushing yards, rushing TDs) to capture the elite fantasy ceilings of mobile QBs (e.g., Lamar Jackson, Josh Allen, Jalen Hurts).
3. **Opponent Matchup Strength — Defense vs. Position / DvP (Completed)**:
   * Integrated zero-leakage Bayesian DvP volume/efficiency differentials and schedule-adjusted rolling metrics across all 4 positions.
4. **Field Location & Red-Zone Distance Weighting (Next Up)**:
   * Upgrade expected points ($xFP$) by incorporating field position distance-to-goal (`yardline_100`). A carry at the 2-yard line carries a ~45% TD probability ($2.70\text{ xFP}$), whereas a carry at mid-field carries a ~2% TD probability ($0.12\text{ xFP}$).
5. **Streamlit Dashboard UX Overhaul**:
   * Refine the visual layout, user experience, and interactive workflows in `app.py` before public portfolio deployment.
6. **GitHub Pages Portfolio Integration (`aclink88.github.io`)**:
   * Deploy `aclink88/ffbEval` to Streamlit Community Cloud and embed the live app in-place via `?embed=true` iframe inside `samples.html` / `readMeRender.js` alongside the rendered `README.md`.
   * Update `readMeRender.js` relative image path resolution from `/master/` to `/HEAD/` so repositories using either `main` or `master` render images automatically.

---

## 🏗️ 2. SYSTEM ARCHITECTURE & IMPLEMENTATION

### **A. Expected Fantasy Points ($xFP$) Methodology & Multipliers**
The coefficients in our $xFP$ equations are empirical regression coefficients derived from historical NFL play-by-play events (`nflverse` dataset).

* **Receiving $xFP$ (WR / TE / RB targets)**:
  $$\text{xFP}_{\text{receiving}} = (0.60 \times \text{Targets}) + (0.08 \times \text{Air Yards}) + (0.30 \times \text{Targets})$$
  * **$0.60$ per Target**: Base completion & receptions value in PPR leagues (~65% catch rate $\times$ 1.0 PPR pt).
  * **$0.08$ per Air Yard**: Accounts for target depth downfield (e.g., a target 20 yards downfield adds $1.60$ expected points).
  * **$0.30$ per Target (TD Expectation)**: Average target-to-TD conversion (~5% $\times$ 6.0 pts).

* **Rushing $xFP$ (RB / QB carries)**:
  $$\text{xFP}_{\text{rushing}} = (0.65 \times \text{Carries}) + (0.18 \times \text{Carries})$$
  * **$0.65$ per Carry**: Base yardage value (~4.2 YPC $\times$ 0.1 pts/yd plus first-down value).
  * **$0.18$ per Carry (TD Expectation)**: Average carry-to-TD conversion (~3% $\times$ 6.0 pts).

* **Passing $xFP$ (QB Pass Attempts & Air Yards)**:
  $$\text{xFP}_{\text{passing}} = (0.20 \times \text{Attempts}) + (0.025 \times \text{Passing Air Yards}) + (0.18 \times \text{Attempts})$$
  * **$0.20$ per Attempt**: Base passing volume & YAC value.
  * **$0.025$ per Air Yard**: Passing yardage rate ($0.04$ pts/yd scaled by completion probability downfield).
  * **$0.18$ per Attempt (TD Expectation)**: Average QB passing TD conversion (~4.5% $\times$ 4.0 pts per passing TD).

### **B. Empirical Bayesian Rate Shrinkage**
To handle small sample size noise (e.g., a backup QB playing 1 half or a backup RB getting 10 carries), we apply **Empirical Bayesian Shrinkage**:

$$\hat{\theta}_{\text{adjusted}} = w \cdot \bar{X}_{\text{observed}} + (1 - w) \cdot \mu_{\text{prior}}, \quad \text{where } w = \frac{n}{n + m_{\text{pos}}}$$

**Empirically Derived Regularization Constants ($m_{\text{pos}}$)** via ANOVA variance decomposition ($\sigma_{\text{within}}^2 / \sigma_{\text{between}}^2$):
* **Running Back (RB)**: $m = 1.1\text{ weeks}$ (~15-20 touches) — Volume stabilizes fastest; role changes are real immediately.
* **Wide Receiver (WR)**: $m = 1.4\text{ weeks}$ (~12-15 targets) — Requires ~1.4 weeks of target volume before fully trusting signal.
* **Tight End (TE)**: $m = 1.5\text{ weeks}$ (~8-10 targets) — High TD variance requires moderate shrinkage.
* **Quarterback (QB)**: $m = 1.8\text{ weeks}$ (~55-60 dropbacks) — High weekly scoring variance; backup QBs are shrunk ~85% on 1st game.

### **C. Position-Percentile Valuation Taxonomy**
Players are classified using dynamic **positional quantiles** ($P_{25}, P_{50}, P_{75}, P_{90}$) to avoid arbitrary static cutoffs:
1. **`FANTASY ALPHA`**: Top 10% Actual Output ($P_{90+}$) AND Top 10% Expected Volume ($P_{90+}$) with $\text{Efficiency Gap} \le 3.0$ (e.g. Josh Allen, Lamar Jackson).
2. **`ELITE BUY TARGET`**: Top producers whose expected volume is so high ($P_{85+}$ with $xFP \gg \text{Actual}$) that they have upside to reach Fantasy Alpha tier (e.g. Tyler Shough).
3. **`HIGH-VALUE PRODUCER`**: High scoring players whose TD efficiency rate exceeds standard volume expectations, but who are locked into a permanent high-value role (e.g. Dallas Goedert effect).
4. **`TRUE BUY LOW`**: Below-median actual producers getting solid expected volume ($xFP \gg \text{Actual}$).
5. **`TRUE SELL HIGH`**: Fluke fantasy output on low expected volume.
6. **`PERPETUAL TRAP`**: Below-median producers who historically underperform $xFP$ (low efficiency is their norm).

---

## 🏃 3. MODEL DEEP DIVE FOR RBs (Running Back Multicollinearity & Drivers)

Evaluating Running Backs requires balancing **Rushing Volume** (carries) with **High-Value Receiving Workload** (targets/receptions).

### **A. Multicollinearity Challenges in RB Features**
When building linear models for Running Backs, two major structural sources of multicollinearity arise:
1. **Single-Week vs. Multi-Week Volume**:
   * `carries` (Week $N$) and `roll3_carries` (3-week average) are strongly correlated ($\rho \approx 0.85 - 0.92$).
2. **Opportunity vs. Output**:
   * `targets` (opportunity) and `receptions` (output) have a near-perfect linear relationship ($\rho \approx 0.90$).

### **B. How Different Models Handle RB Multicollinearity**
* **Classical OLS**: Suffers from high Variance Inflation Factors ($\text{VIF} > 15$), making unregularized coefficients unstable.
* **Stepwise Bilateral AIC & Lasso ($L_1$)**: Actively **prunes redundant collinear variables** (e.g., keeping `targets` while dropping `receptions`).
* **Ridge Regression ($L_2$)**: Retains all variables but **shrinks coefficients** to split weight evenly between collinear pairs.
* **Tree-Based Models (Random Forest & XGBoost)**: Completely **immune to multicollinearity** because decision trees split on orthogonal single-variable thresholds rather than matrix inverses $(\mathbf{X}^T\mathbf{X})^{-1}$.

### **C. Fast Stabilization Rate for Running Backs ($m_{\text{RB}} = 1.1$)**
Running Back volume is heavily dictated by team depth charts and game script. Because RB volume stabilizes rapidly, our Bayesian shrinkage constant for RBs is set to $m = 1.1\text{ weeks}$ (~15-20 touches). When a backup RB takes over a backfield, their new volume signal is trusted immediately.

---

## 🎯 4. MODEL DEEP DIVE FOR WRs (Wide Receiver Target Share & Air Yards)

Evaluating Wide Receivers requires disentangling **Opportunity Volume** (`targets`, `roll3_targets`), **Downfield Depth** (`receiving_air_yards`, `roll3_receiving_air_yards`), and **Efficiency Output** (`receptions`, `receiving_yards`).

### **A. Stepwise Bilateral AIC Feature Selection for WRs**
Out of 15 available volume and efficiency predictors, Bilateral AIC selected **5 key features**:
$$\mathcal{S}_{\text{AIC, WR}} = \{\text{roll3\_targets}, \text{roll3\_actual\_ppr}, \text{receptions}, \text{rushing\_yards}, \text{receiving\_yards}\}$$

* **`roll3_targets` & `roll3_actual_ppr`**: Capture sustained baseline target volume and baseline scoring role over a 3-week window.
* **`receptions` & `receiving_yards`**: Capture single-week usage and yards per reception efficiency.
* **`rushing_yards`**: Captures high-ceiling dual-threat WR usage (gadget plays, jet sweeps, end-arounds like Deebo Samuel / Zay Flowers).

### **B. WR Model Walk-Forward Performance Comparison (2025–2026 Test Set)**

| WR Model Variant | Num Features | RMSE | MAE | Spearman $r_s$ | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WR Naive Baseline (Roll3 Avg)** | 1 | 6.6155 | 4.6430 | 0.5856 | 0.5388 |
| **WR Support Vector Regressor (SVR)** | 15 | 6.3110 | **4.2873** | 0.6121 | 0.5607 |
| **WR Random Forest (AIC 5 Features)** | 5 | 6.1455 | 4.4953 | 0.6209 | 0.5700 |
| **WR Random Forest (Full 15 Set)** | 15 | 6.1557 | 4.4988 | 0.6209 | 0.5679 |
| **WR Lasso Selection (L1)** | 15 | **6.1351** | 4.5173 | **0.6227** | **0.5711** |
| **WR Ridge Regression (L2)** | 15 | 6.1371 | 4.5135 | 0.6225 | 0.5708 |
| **Bilateral AIC OLS Model** | 5 | 6.1411 | 4.5155 | 0.6222 | 0.5701 |

### **C. Key Architectural Insights for Wide Receivers**
1. **Lasso & Ridge Win Rank Correlation ($r_s = 0.6227$)**: Linear shrinkage methods excel at ranking WR performance because target volume and receiving yards scale smoothly with fantasy output.
2. **SVR Wins Point-Prediction Error ($\text{MAE} = 4.2873$)**: SVR's $\epsilon$-insensitive loss function effectively ignores small week-to-week target noise, making it the most robust model for predicting exact point totals.
3. **Random Forest AIC Pruning**: Restricting Random Forest to the 5 AIC features improved both MAE (`4.4953` vs `4.4988`) and Pearson correlation (`0.5700` vs `0.5679`), demonstrating that pruning noisy features improves tree prediction quality.

---

## 🏈 5. MODEL DEEP DIVE FOR TEs (Pure Opportunity Architecture & Pure Feature Selection)

To eliminate **circular logic** (using past fantasy points to predict future fantasy points) and create a **true competitive advantage**, the TE model uses a **Pure Opportunity Architecture**. It strips out backward-looking point totals (`fantasy_points_ppr`, `roll3_actual_ppr`) and forces models to rely strictly on underlying volume, target depth, and expected point infrastructure ($xFP$).

### **A. Stepwise BIC & AIC Pure Opportunity Selection for TEs**
When restricting candidate features to pure opportunity metrics:
* **Stepwise BIC Selection (Extreme Parsimony)**:
  $$\mathcal{S}_{\text{BIC, TE}} = \{\text{roll3\_targets}, \text{receiving\_yards}\}$$
* **Stepwise AIC Selection (Expanded Opportunity Signal)**:
  $$\mathcal{S}_{\text{AIC, TE}} = \{\text{roll3\_targets}, \text{receiving\_yards}, \text{roll3\_expected\_ppr}, \text{efficiency\_delta\_roll3}\}$$

### **B. TE Pure Opportunity Walk-Forward Performance (2025–2026 Test Set)**

| TE Model Variant | Num Features | RMSE | MAE | Spearman $r_s$ | Pearson $r$ | Architecture Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TE Naive Baseline (Roll3 Expected)** | 1 | 5.4926 | **3.6259** | 0.5789 | 0.5278 | Pure Opportunity |
| **TE AIC Pure Opportunity OLS** | 4 | **5.2762** | 3.7339 | **0.5885** | **0.5373** | Pure Opportunity |
| **TE BIC Pure Opportunity OLS** | 2 | 5.2768 | 3.7340 | 0.5875 | 0.5372 | Pure Opportunity |
| **TE Ridge Regression (L2 Opp)** | 8 | 5.2759 | 3.7391 | 0.5872 | 0.5373 | Pure Opportunity |
| **TE Lasso Selection (L1 Opp)** | 8 | 5.2786 | 3.7399 | 0.5857 | 0.5366 | Pure Opportunity |
| **TE Random Forest (BIC Features)** | 2 | 5.3269 | 3.7655 | 0.5764 | 0.5247 | Pure Opportunity |
| **TE Support Vector Regressor (SVR)** | 8 | 5.5311 | 3.6660 | 0.5737 | 0.5187 | Pure Opportunity |

### **C. Key Architectural Insights for Tight Ends**
1. **Superior Predictive Signal**: The **Pure Opportunity AIC Model** (`roll3_targets`, `receiving_yards`, `roll3_expected_ppr`, `efficiency_delta_roll3`) outperforms the old output-driven model in Spearman rank correlation (**$r_s = 0.5885$** vs $0.5876$).
2. **Identifies True Buy-Low Candidates**: By evaluating TEs solely on target share and air-yard depth rather than actual fantasy points scored, the model identifies players whose opportunity is rising before the fantasy points show up on the box score.
3. **BIC vs AIC Parsimony**: BIC selected a minimalist 2-feature model (`roll3_targets` + `receiving_yards`) that performs nearly identically ($r_s = 0.5875$), proving that 3-week rolling target volume is the primary underlying engine of TE fantasy success.

---

## 📚 6. LEARNING REFERENCES (Statistical & Modeling Concepts)

### **A. Stepwise AIC Selection Directions (Forward, Backward, Bilateral)**
1. **Forward Selection**: Starts empty, adds features one-by-one based on maximum AIC reduction. Cannot drop features once added.
2. **Backward Elimination**: Starts with full model, removes features one-by-one. Suffers if initial full-model multicollinearity causes matrix singularity (`SingularMatrixWarning`).
3. **Bilateral (Both / Stepwise Forward-Backward)**: Evaluates both additions and removals at every step. Can **drop previously added features** if new features make them redundant.

### **B. Ridge ($L_2$) vs. Lasso ($L_1$) Regularization**
* **Ridge ($L_2$)**: Shrinks coefficients toward zero, but **NEVER sets any coefficient to zero**. Best when all features carry signal.
* **Lasso ($L_1$)**: Shrinks coefficients **all the way to EXACTLY ZERO**. Best for automatic variable elimination.

### **C. BIC vs. AIC Criteria**
* **AIC**: Penalty $= 2k$. Minimizes prediction error; favors slightly larger models.
* **BIC**: Penalty $= k \ln(n)$. Heavily penalizes complexity for datasets with $n > 8$; favors simple, parsimonious models.

### **D. Bayesian Shrinkage Intuition**
$$\hat{\theta}_{\text{adjusted}} = w \cdot \bar{X}_{\text{observed}} + (1 - w) \cdot \mu_{\text{prior}}, \quad w = \frac{n}{n + m}$$
Uses **Opportunity Units** (Pass Dropbacks for QBs, Touches for RBs, Targets for WRs) so mid-game replacements are rate-scaled accurately.

### **E. In-Sample Likelihood Optimization vs. Domain-Constrained Feature Selection**
* **Why Unconstrained AIC/BIC Can Greedily Select Suboptimal Features**:
  Stepwise AIC and BIC optimize strictly for historical log-likelihood ($\ln L$) on training data:
  $$\text{AIC} = 2k - 2\ln(L), \quad \text{BIC} = k\ln(n) - 2\ln(L)$$
  When backward-looking scoring outputs (like `fantasy_points_ppr`) are present in candidate feature pools, AIC/BIC greedily select them because they soak up immediate historical variance. However, once past points are selected, the algorithm's complexity penalty ($+2k$ or $+k\ln(n)$) prevents underlying opportunity metrics (like `receiving_air_yards` or `targets`) from entering the model because their marginal likelihood gain is deemed statistically "redundant."
* **The Domain-Constrained Solution**:
  To build models with **predictive edge and domain utility** (such as identifying Buy-Low targets before box-score points follow), data scientists must impose **domain constraints on candidate feature pools**. Restricting candidate inputs to pure opportunity infrastructure forces AIC and BIC to find the optimal combination of true underlying drivers rather than circular box-score outputs.

---

## 🛡️ 7. DEFENSE VS. POSITION (DvP) & SCHEDULE ADJUSTMENT ENGINE

To avoid the three classic pitfalls of raw "Points Allowed per Game" (schedule bias, touchdown flukes, and lookahead data leakage), our pipeline implements a **Zero-Leakage Bayesian DvP & Schedule-Adjustment Engine** across [`01_fetch_data.py`](01_fetch_data.py) and [`02_feature_engineering.py`](02_feature_engineering.py).

### **A. Decomposing Defensive Allowances into Volume vs. Efficiency**
For every `(season, week, opponent_team, position)` group, we aggregate total points allowed and decompose them into two orthogonal components:
1. **Defensive Volume Allowed ($xFP$ Differential)**:
   $$\Delta\text{DvP}_{\text{xFP}} = \text{credibility}_{\text{def}} \times \left(\overline{\text{xFP}}_{\text{allowed, roll4}} - \mu_{\text{league, xFP}}\right)$$
   Measures whether a defense funnels volume (targets, carries, pass attempts) to a specific position.
2. **Defensive Efficiency Allowed ($\text{Actual} - xFP$ Differential)**:
   $$\Delta\text{DvP}_{\text{eff}} = \text{credibility}_{\text{def}} \times \left(\overline{\text{Eff}}_{\text{allowed, roll4}} - \mu_{\text{league, eff}}\right)$$
   Measures whether a defense allows players to outperform their expected volume via blown coverages, missed tackles (YAC), or high red-zone TD conversion rates.

### **B. Zero-Leakage Lagged Windows & Empirical Bayesian Shrinkage ($m_{\text{def}} = 3.0$)**
* All defensive rolling windows use strictly lagged 4-week histories (`.shift(1)`) so that Week $t$ predictions never see Week $t$ outcomes.
* Early-season defensive noise (Weeks 1–3) is shrunk toward the league-average positional allowance using a Bayesian prior weight of $m_{\text{def}} = 3.0\text{ games}$:
  $$\text{credibility}_{\text{def}} = \frac{n_{\text{games}}}{n_{\text{games}} + 3.0}$$

### **C. Backward Schedule Adjustment vs. Forward Matchup Adjustment**
1. **Backward-Looking Schedule Adjustment (`sched_adj_xfp`, `roll3_sched_adj_xfp`, `roll3_sched_adj_actual_ppr`)**:
   Credits players who faced lockdown defenses over the past 3 weeks and discounts players who padded stats against soft defenses:
   $$\text{sched\_adj\_xfp}_{t} = \text{xFP}_{t} - \left(s_{\text{pos}} \cdot \Delta\text{DvP}_{\text{curr\_opp, xFP}}\right)$$
2. **Forward-Looking Matchup Adjustment (`next_opp_dvp_xfp_adj`, `next_opp_dvp_eff_adj`, `next_opp_dvp_total_adj`)**:
   Uses `nflreadpy.load_schedules()` to map every player's next scheduled opponent (`effective_next_opponent`) in Week $t+1$ and attaches that defense's latest Bayesian-shrunk positional allowance entering the matchup.

### **D. Cross-Position DvP Benchmark Findings (2025–2026 Walk-Forward Test Set)**
| Position | Stepwise AIC Selection Behavior | Best DvP Model Variant | Key Metric Improvement |
| :--- | :--- | :--- | :--- |
| **QB** | Swapped raw `roll3_expected_ppr` $\to$ **`roll3_sched_adj_xfp`** + added **`next_opp_dvp_xfp_adj`** | **Lasso (With DvP)** / **Random Forest (With DvP)** | Spearman $r_s$: `0.3707` $\to$ **`0.3833`**; RF RMSE: `8.0960` $\to$ **`8.0612`** |
| **RB** | Swapped raw `roll3_actual_ppr` $\to$ **`roll3_sched_adj_actual_ppr`** + added **`next_opp_dvp_xfp_adj`** | **Pure Opp + DvP AIC OLS (7 Features)** | Lowest overall RB RMSE (**`6.3682`**) & $r_s = 0.7136$ without using past points |
| **WR** | Selected both **`roll3_sched_adj_xfp`** and **`next_opp_dvp_xfp_adj`** in Pure Opp pool | **Lasso (With DvP)** / **Pure Opp + DvP OLS** | Best WR RMSE (**`6.1645`**) & Spearman $r_s$ (**`0.6316`**); SVR MAE = **`4.3029`** |
| **TE** | Retained 4 pure opportunity features (offensive scheme dominates TE usage over matchup) | **AIC Pure Opp OLS** / **SVR (With DvP)** | Best TE Spearman $r_s$ (**`0.5933`**); DvP boosted SVR $r_s$ (`0.5737` $\to$ `0.5859`) |


---

## 📅 8. SEASON $S$ WEEK 1 PROJECTION METHODOLOGY (Offseason Retention & Rookie Priors)

When rolling windows (`roll3_*`) and target shifts (`.shift(-1)`) are partitioned strictly by `['player_id', 'season']`, a player's first row in Season $S$ (Week 1) only predicts Week 2—leaving Week 1 without a pre-game projection. To generate mathematically grounded **Season $S$ Week 1 Projections**, we distinguish between **Returning Veterans** and **Incoming Rookies**:

### **A. Returning Players: Empirical Offseason Retention Weight ($w_{\text{offseason, pos}}$)**
Rather than picking an arbitrary heuristic (e.g., 35% shrinkage) to discount a player's end-of-season baseline over the offseason, we derive the exact **Minimum-MSE Linear Shrinkage Weight** from historical season-to-season transitions ($2023 \to 2024$ and $2024 \to 2025$).

For each position group (`QB`, `RB`, `WR`, `TE`), we regress a returning player's **Season $S$ Week 1 Opportunity ($\text{xFP}_{S, \text{wk1}}$)** against their **Season $S-1$ Final 3-Week Rolling Opportunity ($\overline{\text{xFP}}_{S-1, \text{end}}$)**:

$$w_{\text{offseason, pos}} = \frac{\text{Cov}\left(\text{xFP}_{S, \text{wk1}}, \, \overline{\text{xFP}}_{S-1, \text{end}}\right)}{\text{Var}\left(\overline{\text{xFP}}_{S-1, \text{end}}\right)}$$

The player's pre-Week 1 shrunk feature baseline entering Season $S$ is then:
$$\hat{X}_{S, \text{pre-wk1}} = w_{\text{offseason, pos}} \cdot \bar{X}_{S-1, \text{end}} + \left(1 - w_{\text{offseason, pos}}\right) \cdot \mu_{\text{pos}}$$
* **Why Position-Specific Weights Matter**: Quarterbacks and #1 Wide Receivers exhibit higher year-over-year role stability (higher $w_{\text{offseason}}$) than Running Backs, whose workloads are more heavily impacted by offseason free agency, coaching changes, and draft additions.

### **B. Incoming Rookies: Draft Capital ($\ln(\text{Pick})$) & Depth Chart Stratification**
Assigning every incoming rookie the flat positional mean ($\mu_{\text{pos}}$) creates a severe flaw: an elite top-5 overall draft pick and Week 1 starter (e.g., **Jeremiyah Love**, 3rd overall pick) would receive the exact same Week 1 projection as a 7th-round depth piece (e.g., **Eli Heidenreich**).

To differentiate rookies entering Week 1, we construct a **Stratified Rookie Prior** using NFL Draft Capital (`nfl.load_draft_picks()` / `nfl.load_players()`) and Week 1 Depth Chart rank (`nfl.load_depth_charts()`):
1. **Log-Draft-Pick Opportunity Curve**:
   Historically, rookie Week 1 opportunity ($xFP$) decays logarithmically with overall draft slot ($\text{Pick } \in [1, 260]$, with undrafted free agents assigned $\text{Pick } = 300$):
   $$\mu_{\text{rookie, xFP}}(\text{Pick}, \text{Pos}) = \alpha_{\text{pos}} - \beta_{\text{pos}} \cdot \ln(\text{Pick})$$
   * **Top-10 Overall Pick (Round 1)**: Projects for immediate featured starter volume (~14–16 $\text{xFP}$ for RBs, ~12–14 $\text{xFP}$ for WRs).
   * **Day 2 Pick (Rounds 2–3)**: Projects for rotational / committee volume (~7–9 $\text{xFP}$).
   * **Day 3 / UDFA (Rounds 4–7+)**: Projects for backup / special-teams volume (~1.5–3.0 $\text{xFP}$).
2. **Week 1 Depth Chart Multiplier (`depth_team`)**:
   Incorporating the Week 1 depth chart rank (`depth_team == 1` starter vs. `depth_team >= 2` backup) ensures that a late-round rookie who earns a starting job in training camp is upgraded accordingly, while a backup remains anchored to reserve volume.


### **C. Mathematical Deep Dive: Why Draft Capital Decays Logarithmically & Conditional Bayesian Priors**

#### **1. Why Does Draft Capital Decay Logarithmically Instead of Linearly?**
Imagine trying to model rookie Week 1 Expected Fantasy Points ($xFP$) with a standard linear equation:
$$\text{xFP} = \alpha - \beta \cdot (\text{Pick})$$
In a linear model, the slope $\beta$ is constant across the entire draft. That would imply the talent and opportunity gap between **Pick #1 and Pick #21** (20 picks apart in Round 1) is identical to the gap between **Pick #200 and Pick #220** (20 picks apart in Round 6/7).

In reality, NFL talent and coaching commitment follow a **Pareto / Power-Law distribution** (heavily right-skewed):
* **Top of Round 1 (Picks 1–10)**: Teams draft generational outliers and build their offensive game plan around them immediately. The drop-off from Pick #2 to Pick #15 is steep—Pick #2 is drafted to be an immediate 70%+ snap-share bellcow, whereas Pick #15 may already be in a committee.
* **Late Rounds (Picks 150–260)**: Players are developmental depth and special-teamers. There is virtually zero difference in expected Week 1 opportunity between Pick #180 and Pick #220.

**How the Natural Logarithm ($\ln(\text{Pick})$) Solves This Mathematically:**
When we take the derivative (rate of change) of the logarithmic curve with respect to pick number:
$$\mu_{\text{xFP}}(\text{Pick}) = \alpha_{\text{pos}} - \beta_{\text{pos}} \cdot \ln(\text{Pick})$$
$$\frac{d(\mu_{\text{xFP}})}{d(\text{Pick})} = -\frac{\beta_{\text{pos}}}{\text{Pick}}$$
Notice that the marginal rate of decay $-\frac{\beta_{\text{pos}}}{\text{Pick}}$ is **inversely proportional to the pick number**:
* **At Pick #2**: The slope is $-\frac{\beta}{2}$ (very steep drop-off per pick).
* **At Pick #20**: The slope is $-\frac{\beta}{20}$ ($10\times$ flatter than at Pick #2).
* **At Pick #200**: The slope is $-\frac{\beta}{200}$ ($100\times$ flatter—almost completely horizontal).

Furthermore, because $\ln(2^{\text{k}})$ steps linearly with each doubling of pick number, **every time the pick number doubles** (Pick 2 $\to$ Pick 4 $\to$ Pick 8 $\to$ Pick 16 $\to$ Pick 32 $\to$ Pick 64), expected Week 1 opportunity drops by a constant amount $\beta_{\text{pos}} \cdot \ln(2) \approx 0.693\,\beta_{\text{pos}}$. This matches the famous **Jimmy Johnson / Chase Stuart NFL Draft Trade Value Chart**, which also follows an exponential/logarithmic decay curve.

#### **2. How Empirical Bayes Conditional Priors Work for Rookies ($n = 0 \to n \ge 1$)**
In a standard (unconditional) Empirical Bayes model, every player in a position group shrinks toward the same global positional mean $\mu_{\text{pos}}$:
$$\hat{\theta}_{i} = \left(\frac{n}{n + m_{\text{pos}}}\right) \bar{X}_{i} + \left(\frac{m_{\text{pos}}}{n + m_{\text{pos}}}\right) \mu_{\text{pos}}$$
When a rookie enters **Week 1**, their observed NFL sample size is **$n = 0$**, so their credibility weight is $w = \frac{0}{0 + m_{\text{pos}}} = 0$, and $100\%$ of their projection comes from the prior ($1 - w = 1.0$).

In a **Hierarchical / Conditional Empirical Bayes** framework, we replace the flat global prior $\mu_{\text{pos}}$ with a **Covariate-Conditioned Prior** $\mu_{0, i} = E[\theta_i \mid \text{Pick}_i, \text{Depth}_i]$:
$$\mu_{0, i} = \left(\alpha_{\text{pos}} - \beta_{\text{pos}} \cdot \ln(\text{Pick}_i)\right) \times \gamma_{\text{depth}}$$
where $\gamma_{\text{depth}}$ is the depth-chart multiplier (e.g., $\gamma > 1$ if listed as the Week 1 `depth_team == 1` starter, $\gamma < 1$ if buried at `depth_team >= 3`).

Then, as the rookie actually plays NFL games in Weeks 1, 2, and 3 ($n = 1, 2, 3$), the Bayesian update smoothly blends their **observed NFL box-score opportunity ($\bar{X}_i$)** with their **Draft-Capital Prior ($\mu_{0, i}$)**:
$$\hat{\theta}_{i, \text{week } t} = \underbrace{\left(\frac{n}{n + m_{\text{pos}}}\right) \bar{X}_{i}}_{\text{Observed NFL Volume}} + \underbrace{\left(\frac{m_{\text{pos}}}{n + m_{\text{pos}}}\right) \mu_{0, i}}_{\text{Draft Capital & Depth Prior}}$$
* **Entering Week 1 ($n = 0$)**: $100\%$ of the projection is driven by $\mu_{0, i}$ (so Jeremiyah Love at Pick #3 projects as a high-volume RB1 while Eli Heidenreich in Round 7 projects as a deep reserve).
* **Entering Week 2 ($n = 1$, with $m_{\text{RB}} = 1.1$)**: The rookie's actual Week 1 usage now carries $\frac{1}{1 + 1.1} = 47.6\%$ weight, while their draft capital prior carries $52.4\%$ weight.
* **By Week 4+ ($n = 3$)**: Observed 3-week rolling NFL volume carries $\frac{3}{3 + 1.1} = 73.2\%$ weight, naturally phasing out draft capital once real NFL snap/touch data is established!


---

### 8.5 Dynamic Runtime Calibration & Empirical 2026 Week 1 Verification

Rather than hardcoding static offseason retention weights or draft-capital slopes that could drift as NFL coaching tendencies evolve, `04_generate_buy_sell_signals.py` (`compute_dynamic_offseason_and_rookie_priors`) dynamically fits both formulas across all active historical seasons in the dataset at runtime:

| Position | Dynamic Veteran Retention ($w_{\text{offseason, pos}}$) | Dynamic Rookie Draft Curve ($\mu_{\text{xFP}}(\text{Pick})$) |
| :--- | :---: | :--- |
| **QB** | `0.500` | $26.10 - 4.34 \cdot \ln(\text{Pick})$ |
| **RB** | `0.786` | $18.10 - 2.87 \cdot \ln(\text{Pick})$ |
| **WR** | `0.759` | $13.94 - 2.00 \cdot \ln(\text{Pick})$ |
| **TE** | `0.451` | $11.60 - 1.91 \cdot \ln(\text{Pick})$ |

#### 1. Incoming 2026 Rookies (Stratified by Overall Draft Pick)
| Player | Pos | Team | Round / Pick | Pre-Week 1 Proj xFP | Pre-Week 1 Model Pred (PPR) | Actual Week 1 PPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jeremiyah Love** | RB | ARI | Rd 1, Pick #3 | `14.95` | **`13.54`** | `13.00` |
| **Carnell Tate** | WR | TEN | Rd 1, Pick #4 | `11.16` | **`10.31`** | `7.80` |
| **Kenyon Sadiq** | TE | NYJ | Rd 1, Pick #16 | `6.30` | **`7.32`** | `11.40` |
| **Makai Lemon** | WR | PHI | Rd 1, Pick #20 | `7.93` | **`7.75`** | `2.50` |
| **KC Concepcion** | WR | CLE | Rd 1, Pick #24 | `7.56` | **`7.46`** | `7.80` |
| **Denzel Boston** | WR | CLE | Rd 2, Pick #39 | `6.59` | **`6.69`** | `13.90` |
| **Nate Boerkircher** | TE | JAX | Rd 2, Pick #56 | `3.91` | **`5.07`** | `1.90` |
| **Antonio Williams** | WR | WAS | Rd 3, Pick #71 | `5.39` | **`5.74`** | `16.40` |

#### 2. Returning Veterans (Top 2026 Week 1 Projections)
| Player | Pos | Team | Pre-Week 1 Proj xFP | Pre-Week 1 Model Pred (PPR) | Actual Week 1 PPR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Josh Allen** | QB | BUF | `24.04` | **`19.01`** | `35.66` |
| **Puka Nacua** | WR | LA | `23.72` | **`18.82`** | `12.40` |
| **Bo Nix** | QB | DEN | `22.88` | **`18.61`** | `5.44` |
| **Caleb Williams** | QB | CHI | `23.70` | **`17.76`** | `37.26` |
| **Justin Herbert** | QB | LAC | `22.33` | **`17.70`** | `13.26` |


---

## 9. Future Enhancements & Roadmap

### 9.1 Mid-Season IR Activations & Delayed Season Debuts (The "Zach Charbonnet" Case)

While the current Season $S$ debut logic (`pl.col("week").rank("ordinal").over(["player_id", "season"]) == 1`) automatically routes a returning player's first active game of the season through the veteran offseason retention formula, three specialized enhancements are planned to handle players returning mid-season from long-term Injured Reserve (IR) or Physically Unable to Perform (PUP) lists (e.g., Zach Charbonnet returning in 2026 Week 8 after tearing his ACL in the 2025 playoffs):

1. **Pre-Debut Roster Injection (`01_fetch_data.py`)**:
   * **Current Limitation**: `nfl.load_player_stats(summary_level="week")` only creates a player-week row *after* a player logs a box-score stat in Season $S$. A player who spent Weeks 1–7 on IR has `0` rows in `season == 2026` prior to Week 8 kickoff, meaning they do not appear in the `2026` dashboard view on Wednesday before their debut game.
   * **Planned Enhancement**: Cross-reference `nfl.load_rosters_weekly()` / `nfl.load_injuries()` for the active season and inject a synthetic pre-debut baseline row for any veteran or drafted rookie elevated to the active 53-man roster ahead of Week $t+1$ who has not yet recorded a box-score row in Season $S$.
2. **Injury-Exit Game Filtering in Prior-Season Baselines ($\overline{\text{xFP}}_{S-1,\text{end}}$)**:
   * **Current Limitation**: In Charbonnet's final game of 2025 (Week 20), he exited early with a torn ACL (`4.15 xFP` / `2.0 PPR`), dragging his 3-game end-of-season average down from `17.63 xFP` (Weeks 17–18) to `13.13 xFP`.
   * **Planned Enhancement**: Join weekly snap counts (`nfl.load_snap_counts()`) or injury-exit tags to down-weight or exclude games where a player exited in the first half due to injury when computing $\overline{\text{xFP}}_{S-1,\text{end}}$.
3. **IR Pitch-Count / Ramp-Up Decay Factor ($\gamma_{\text{IR}}$)**:
   * **Current Limitation**: The dynamic veteran retention weight ($w_{\text{offseason, RB}} = 0.786$) is calibrated on healthy Week 1 starters. Applying `78.6%` retention to a player making their season debut in Week 8 off a major knee reconstruction ignores both medical snap limits ("pitch counts") and backfield role consolidation by healthy teammates during Weeks 1–7.
   * **Planned Enhancement**: Introduce an empirical **Missed-Time / IR Activation Discount ($\gamma_{\text{IR}}(\Delta \text{Weeks Missed})$)** that scales down Week 1 activation volume projections when a player's season debut occurs in Week $t \ge 5$ following a multi-week IR/PUP stint, gradually releasing the discount over their first 3 active games back.
