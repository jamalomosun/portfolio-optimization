import numpy as np
import pandas as pd
import cvxpy as cp

def mad_portfolio(assets: pd.DataFrame, R=0.001, w_lb=0.0, w_ub=0.2):
    # Step 1: Calculate daily returns and mean return
    prices = assets.values
    returns = np.diff(np.log(prices), axis=0)
    mean_return = np.mean(returns, axis=0)
    T, n = returns.shape

    # Step 2: Define optimization variables
    w = cp.Variable(n)
    z = cp.Variable(T)

    constraints = [
        cp.sum(w) == 1,  # weights must sum to 1
        w >= w_lb,
        w <= w_ub,
        mean_return @ w >= R
    ]

    # MAD constraints
    for t in range(T):
        deviation = cp.sum(cp.multiply(w, returns[t, :] - mean_return))
        constraints += [
            z[t] >= deviation,
            z[t] >= -deviation
        ]

    # Objective: Minimize average deviation
    objective = cp.Minimize(cp.sum(z) / T)

    # Step 3: Solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Step 4: Output results
    return w.value