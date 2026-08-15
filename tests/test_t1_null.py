import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp

rng = np.random.default_rng(42)  # Set a seed for reproducibility


def compute_car(
    market_returns,
    stock_returns,
    est_start,
    est_end,
    event_start,
    event_end,
    alpha_mode,
    beta_mode,
):
    # Fit the market model using OLS regression
    X = sm.add_constant(
        market_returns.loc[est_start:est_end]
    )  # Add a constant term for the intercept
    y = stock_returns.loc[est_start:est_end]
    model = sm.OLS(y, X).fit()

    if beta_mode == "estimated":
        beta = model.params.iloc[1]
    elif beta_mode == "market":
        beta = 1

    if alpha_mode == "estimated":
        alpha = model.params.iloc[0]
    elif alpha_mode == "zero":
        alpha = 0
    expected_returns = alpha + beta * market_returns.loc[event_start:event_end]
    abnormal_returns = stock_returns.loc[event_start:event_end] - expected_returns

    # Calculate cumulative abnormal returns (CAR)
    car = np.sum(abnormal_returns)

    return car


false_positive = 0
true_negative = 0

for i in range(100):
    data = []
    for i in range(844):
        market = pd.Series(rng.normal(0, 0.01, 281), index=range(-250, 31))
        stock = pd.Series(rng.normal(0, 0.02, 281), index=range(-250, 31))
        data.append((market, stock))

    car_values = [
        compute_car(market, stock, -250, -31, 1, 30, "estimated", "estimated")
        for market, stock in data
    ]

    t_stat, p_value = ttest_1samp(car_values, 0, alternative="less")
    if p_value < 0.05:
        false_positive += 1
    else:
        true_negative += 1

print(f"False positive count: {false_positive}")
print(f"True negative count: {true_negative}")
print(
    f"False positive rate: {false_positive / (false_positive + true_negative) * 100:.2f}%"
)
