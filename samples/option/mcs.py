import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def longstaff_schwartz(paths, strike, r,option_type):
    cash_flows = np.zeros_like(paths)
    if option_type == "Call":
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(round(x - strike,2),0) for x in paths[i]]
    else:
        for i in range(0,cash_flows.shape[0]):
            cash_flows[i] = [max(-round(x - strike,2),0) for x in paths[i]]
    discounted_cash_flows = np.zeros_like(cash_flows)


    T = cash_flows.shape[0]-1

    for t in range(1,T):
        
        # Look at time t+1
        # Create index to only look at in the money paths at time t
        in_the_money =paths[t,:] < strike


        # Run Regression
        X = (paths[t,in_the_money])
        X2 = X*X
        Xs = np.column_stack([X,X2])
        Y = cash_flows[t-1,in_the_money]  * np.exp(-r)
        model_sklearn = LinearRegression()
        model = model_sklearn.fit(Xs, Y)
        conditional_exp = model.predict(Xs)
        continuations = np.zeros_like(paths[t,:])
        continuations[in_the_money] = conditional_exp

        # # First rule: If continuation is greater in t =0, then cash flow in t=1 is zero
        cash_flows[t,:] = np.where(continuations> cash_flows[t,:], 0, cash_flows[t,:])

        # 2nd rule: If stopped ahead of time, subsequent cashflows = 0
        exercised_early = continuations < cash_flows[t, :]
        cash_flows[0:t, :][:, exercised_early] = 0
        discounted_cash_flows[t-1,:] = cash_flows[t-1,:]* np.exp(-r * 3)

    discounted_cash_flows[T-1,:] = cash_flows[T-1,:]* np.exp(-r * 1)


    # Return final option price
    final_cfs = np.zeros((discounted_cash_flows.shape[1], 1), dtype=float)
    for i,row in enumerate(final_cfs):
        final_cfs[i] = sum(discounted_cash_flows[:,i])
    option_price = np.mean(final_cfs)
    print("mcs price", option_price)
    return option_price

def simulate_gbm(mu, sigma, S0, T, dt, num_paths):
    num_steps = int(T / dt) + 1
    times = np.linspace(0, T, num_steps)
    paths = np.zeros(( num_steps,num_paths))

    for i in range(num_paths):
        # Generate random normal increments
        dW = np.random.normal(0, np.sqrt(dt), num_steps - 1)
        # Calculate the cumulative sum of increments
        cumulative_dW = np.cumsum(dW)
        # Calculate the stock price path using the GBM formula
        paths[ 1:,i] = S0 * np.exp((mu - 0.5 * sigma**2) * times[1:] + sigma * cumulative_dW)
    return paths

np.random.seed(99)
# Parameters
mu = 0.00  # Drift (average return per unit time)
sigma = 0.2  # Volatility (standard deviation of the returns)
S0 = 1  # Initial stock price
T = 1  # Total time period (in years)
dt = 1/3  # Time increment (daily simulation)
num_paths = 1000  # Number of simulation paths

# Simulate stock price paths
paths = simulate_gbm(mu, sigma, S0, T, dt, num_paths)
paths[0,:]=S0
paths = paths[::-1]

longstaff_schwartz(paths = paths, strike =1.1, r = 0.06, option_type="Put")  


