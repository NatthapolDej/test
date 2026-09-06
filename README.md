# Machine Learning for Cross-Sectional Stock Selection

This project develops and evaluates machine-learning models for ranking stocks by their expected next-month performance. The objective is to identify stocks likely to enter the top quintile of next-month returns relative to the S&P 500 proxy, SPY.

The project uses historical daily price and volume data to construct monthly momentum, volatility, trend, market-risk, and liquidity features. Logistic regression, random forest, and histogram gradient boosting models are evaluated using chronological out-of-sample predictions and realistic transaction-cost assumptions.

> This project is intended for research and educational purposes. It is not investment advice.

## Project objectives

The project investigates the following questions:

1. Can historical price, volatility, trend, beta, and liquidity characteristics predict the cross-sectional ranking of next-month stock returns?
2. Do nonlinear models outperform a regularized logistic-regression baseline?
3. Can predictive performance be converted into profitable long-only or long–short portfolios after transaction costs?
4. How stable are the signals across years and different market regimes?
5. How does the machine-learning strategy compare with conventional 12–1 momentum?

## Methodology

The strategy follows this monthly process:

1. Download adjusted daily OHLCV data.
2. Calculate stock characteristics using only information available at each month-end.
3. Apply price and liquidity screens.
4. Winsorize and standardize features within each monthly stock cross-section.
5. Define the prediction target using next-month relative returns.
6. Train models using historical data only.
7. Generate out-of-sample scores for each stock.
8. Rank stocks by their predicted scores.
9. Construct long-only and long–short portfolios.
10. Deduct turnover-based transaction costs.
11. Evaluate predictive and portfolio performance.

The timing convention is:

$$
\text{features at month-end }t
\longrightarrow
\text{prediction score at }t
\longrightarrow
\text{return during month }t+1.
$$

## Data

Daily adjusted price and volume data are downloaded through `yfinance`.

The default configuration includes:

```python
start = '2010-01-01'
benchmark = 'SPY'
first_test_year = 2017
top_fraction = 0.20
min_price = 5.0
min_dollar_volume = 10_000_000
cost_bps = 10.0
random_state = 42
```

Downloaded data are cached locally to reduce repeated API calls.

### Investable-universe filters

A stock-month observation must satisfy:

$$
\text{price}\geq \$5
$$

and:

$$
\text{average daily dollar volume}\geq \$10{,}000{,}000.
$$

These filters reduce exposure to low-priced and illiquid securities whose historical returns may be difficult to implement in practice.

## Features

The model uses the following monthly stock characteristics.

### Momentum

* One-month momentum
* Three-month momentum
* Six-month momentum
* Twelve-minus-one-month momentum

The 12–1 momentum feature excludes the most recent month:

```math
\operatorname{Mom}_{12-1,t}
=
\frac{P_{t-1}}{P_{t-12}} - 1
```

### Volatility and downside risk

* Three-month annualized volatility
* Twelve-month annualized volatility
* Six-month downside volatility
* Twelve-month drawdown

For example, three-month annualized volatility is:

$$
\sigma_{3m,t}
=
\operatorname{Std}(r_{t-62},\ldots,r_t)\sqrt{252}.
$$

$\sigma_{3m,t} = \text{Std} r_{t-62},\ldots,r_t\sqrt{252}.$
### Trend and return consistency

* Distance from the 50-day moving average
* Distance from the 200-day moving average
* Fraction of positive-return days over the latest 63 trading days

### Market exposure

Six-month rolling beta is estimated as:

$$
\beta_{i,t}
=
\frac{
\operatorname{Cov}(r_{i},r_m)
}{
\operatorname{Var}(r_m)
}.
$$

### Liquidity

* Logarithm of average daily dollar volume
* Amihud illiquidity

The Amihud measure captures absolute price movement relative to trading activity:

$$
ILLIQ_{i,t}
=
\frac{|r_{i,t}|}{\text{dollar volume}_{i,t}}.
$$

Higher Amihud values indicate greater estimated price impact and lower liquidity.

## Cross-sectional preprocessing

Features are processed separately within each month.

First, observations are winsorized using the 1st and 99th cross-sectional percentiles. The winsorized feature is then standardized:

$$
z_{i,t}
=
\frac{x_{i,t}-\mu_t}{\sigma_t},
$$

where \(\mu_t\) and \(\sigma_t\) are calculated across stocks available at date \(t\).

Cross-sectional standardization allows the model to interpret each characteristic relative to the contemporaneous stock universe. It also places features with different units on comparable scales.

Missing feature values are handled inside each model pipeline using median imputation.

## Prediction target

The next-month stock return is:

$$
R_{i,t+1}
=
\frac{P_{i,t+1}}{P_{i,t}}-1.
$$

Relative return is defined as:

$$
R_{i,t+1}^{relative}
=
R_{i,t+1}-R_{m,t+1},
$$

where \(R_{m,t+1}\) is the next-month SPY return.

Stocks are ranked by relative return within each date. The classification target equals one for stocks in the highest future-return group:

$$
Y_{i,t}
=
\begin{cases}
1, & \text{if stock }i\text{ is in the future top quintile},\\
0, & \text{otherwise}.
\end{cases}
$$

Future returns and target labels are used only for model training and evaluation. They are not included among the prediction features.

## Models

### Logistic regression

The logistic-regression pipeline contains:

1. Median imputation
2. Standard scaling
3. Regularized logistic regression

The model estimates:

$$
P(Y=1\mid X)
=
\frac{1}{
1+\exp[-(\beta_0+\boldsymbol{\beta}'X)]
}.
$$

Class weights are balanced to account for the smaller positive class.

### Random forest

The random forest combines 300 randomized decision trees. Tree depth and minimum leaf size are restricted to reduce overfitting.

The model can capture nonlinear relationships and interactions such as high momentum being useful only when volatility is low.

### Histogram gradient boosting

Histogram gradient boosting builds decision trees sequentially. Each new tree attempts to reduce the classification errors remaining from the previous trees.

Histogram binning makes split selection computationally efficient, while a low learning rate, restricted leaf count, and L2 regularization limit model complexity.

## Out-of-sample design

The primary backtest uses chronological yearly testing. Each model is fitted using only observations whose labels would have been available before the test period.

For example:

```text
Training formation dates: through November 2016
Test formation dates:     January–December 2017
```

The model is trained once for each test year and remains fixed during that year.

Two training-window designs can be compared:

### Expanding window

The model retains all eligible historical observations:

```text
2017 test: beginning of data → November 2016
2018 test: beginning of data → November 2017
2019 test: beginning of data → November 2018
```

### Rolling window

The model retains only a fixed number of recent months:

```text
2017 test: recent 60 months through November 2016
2018 test: recent 60 months through November 2017
2019 test: recent 60 months through November 2018
```

An expanding window provides more training data, while a rolling window may adapt more quickly to changing market regimes.

## Prediction diagnostics

Model predictions are evaluated using several complementary metrics.

### ROC-AUC

ROC-AUC measures the probability that an actual future top-quintile stock receives a higher model score than a non-top-quintile stock:

$$
AUC
=
P(S_{\text{winner}}>S_{\text{non-winner}}).
$$

An AUC of `0.50` represents random ordering.

### Top-quintile precision

Precision measures the proportion of model-selected stocks that actually enter the realized top quintile:

$$
\text{Precision}
=
\frac{TP}{TP+FP}.
$$

Because approximately 20% of stocks receive positive labels, random selection has expected precision near `0.20`.

Precision lift is:

$$
\text{Precision lift}
=
\frac{\text{model precision}}
{\text{positive-class rate}}.
$$

### Information coefficient

Monthly Information Coefficient is calculated using cross-sectional Spearman rank correlation:

$$
IC_t
=
\operatorname{Corr}_{rank}
\left(
S_{i,t},
R_{i,t+1}^{relative}
\right).
$$

A positive IC means stocks receiving higher model scores generally produce higher subsequent relative returns.

Reported IC statistics include:

* Mean monthly IC
* IC standard deviation
* Fraction of months with positive IC
* IC information ratio

## Portfolio construction

### Long-only portfolio

The long-only strategy invests equally in the highest-ranked fraction of stocks:

$$
\sum_iw_{i,t}=1.
$$

It tests whether the model can identify attractive stocks but remains exposed to broad market movements.

### Long–short portfolio

The long–short strategy allocates:

$$
\sum_{i\in L_t}w_{i,t}=+0.5
$$

to high-score stocks and:

$$
\sum_{i\in S_t}w_{i,t}=-0.5
$$

to low-score stocks.

Therefore:

$$
\text{net exposure}=0,
\qquad
\text{gross exposure}=1.
$$

This portfolio is dollar-neutral but not necessarily beta-neutral or sector-neutral.

### Momentum benchmark

A conventional long-only 12–1 momentum strategy is constructed using the same dates and investable observations. This provides a transparent benchmark for evaluating whether machine learning adds value beyond a traditional ranking signal.

## Turnover and transaction costs

Monthly traded notional is approximated as:

$$
TO_t
=
\sum_i
|w_{i,t}-w_{i,t-1}|.
$$

$$ T_{0,t} = \sum_i |w_{i,t}-w_{i,t-1}|$$
Transaction cost is:

$$
C_t
=
TO_t
\frac{c_{\mathrm{bps}}}{10{,}000}.
$$

The default assumption is:

$$
c_{\mathrm{bps}}=10.
$$

Net portfolio return is:

$$
R_{p,t}^{net}
=
R_{p,t}^{gross}-C_t.
$$

This cost model captures a basic turnover penalty but does not fully represent bid–ask variation, nonlinear market impact, short-borrow fees, financing costs, taxes, or execution delay.

## Performance metrics

Portfolio performance is evaluated using:

* Compound annual growth rate
* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Calmar ratio
* Fraction of positive months
* Average monthly turnover
* Estimated annual transaction-cost drag

Compound annual growth rate is:

$$
CAGR
=
\left(
\frac{V_T}{V_0}
\right)^{1/T}-1.
$$

Maximum drawdown is:

$$
MDD
=
\min_t
\left(
\frac{W_t}{\max_{s\leq t}W_s}-1
\right).
$$

The zero-risk-free-rate Sharpe ratio is:

$$
SR
=
\frac{\bar r_m}{s_m}\sqrt{12}.
$$

For a more precise historical analysis, the contemporaneous Treasury-bill return can be subtracted from each monthly portfolio return.

## Installation

Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate

pip install numpy pandas scipy scikit-learn matplotlib seaborn yfinance jupyter
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Usage

Start JupyterLab:

```bash
jupyter lab
```

Open the main notebook and run the cells in order.

The workflow will:

1. Download or load cached market data.
2. Construct the monthly panel.
3. Generate cross-sectional features and targets.
4. Evaluate individual signals.
5. Fit the classification models.
6. Generate out-of-sample predictions.
7. Construct ranked portfolios.
8. Produce diagnostic and performance summaries.

## Results

Add the final out-of-sample results after running the complete notebook:

| Model                       | ROC-AUC | Top-quintile precision | Mean monthly IC | IC positive fraction |
| --------------------------- | ------: | ---------------------: | --------------: | -------------------: |
| Logistic regression         |     TBD |                    TBD |             TBD |                  TBD |
| Random forest               |     TBD |                    TBD |             TBD |                  TBD |
| Histogram gradient boosting |     TBD |                    TBD |             TBD |                  TBD |

Portfolio results:

| Strategy                | CAGR | Annual volatility | Sharpe | Maximum drawdown | Average turnover |
| ----------------------- | ---: | ----------------: | -----: | ---------------: | ---------------: |
| Logistic long-only      |  TBD |               TBD |    TBD |              TBD |              TBD |
| Random forest long-only |  TBD |               TBD |    TBD |              TBD |              TBD |
| HGB long-only           |  TBD |               TBD |    TBD |              TBD |              TBD |
| Traditional momentum    |  TBD |               TBD |    TBD |              TBD |              TBD |

Results should be interpreted using the complete set of diagnostics. A model with a positive ROC-AUC or IC does not necessarily produce attractive net portfolio performance.

## Limitations

This project is subject to several limitations:

* Yahoo Finance data are suitable for research but not institutional execution analysis.
* A fixed modern ticker list may introduce survivorship bias.
* Missing and delisted-security returns require careful treatment.
* The liquidity and price filters are simplified.
* Corporate-action and point-in-time constituent data may be incomplete.
* Transaction costs are represented by a constant basis-point assumption.
* Long–short results exclude stock-borrow and financing costs.
* Dollar neutrality does not guarantee beta or sector neutrality.
* Hyperparameter selection can introduce data-snooping bias.
* Historical relationships may not persist in future market regimes.
* Backtested performance does not guarantee live performance.

## Possible extensions

Future improvements could include:

* Point-in-time index membership and delisting returns
* Historical risk-free-rate data
* Sector- and beta-neutral portfolios
* Volatility-scaled portfolio weights
* Exact post-return weight-drift turnover
* Monthly rather than annual model retraining
* Nested walk-forward hyperparameter tuning
* Probability calibration
* SHAP-based model interpretation
* IC-decay analysis across forecast horizons
* Comparison with XGBoost, LightGBM, and neural networks
* Alternative markets and international equity universes

## Reproducibility

Randomized models use a fixed random seed:

```python
random_state = 42
```

Data ranges, universe definitions, feature parameters, transaction-cost assumptions, and test periods should be recorded with every reported result.

## Disclaimer

This repository is provided solely for educational and research purposes. It does not constitute financial advice, an investment recommendation, or an offer to buy or sell securities. Historical and backtested results do not guarantee future performance.
