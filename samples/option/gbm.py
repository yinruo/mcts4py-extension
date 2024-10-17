import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, dt, N):
    num_steps = int(T / dt) + 1  
    times = np.linspace(0, T, num_steps)  
    paths = np.zeros((num_steps, N))  
    paths[0] = S0  
    
    for i in range(1, num_steps):
        Z = np.random.normal(0, 1, N)
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return times, paths

S0 = 1
mu = 0.05 
sigma = 0.15
T = 5
dt = 1/2 
N = 100

times, paths = simulate_gbm(S0, mu, sigma, T, dt, N)

plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(times, paths[:, i], lw=1)
plt.title("GBM 模拟的资产价格路径")
plt.xlabel("时间 (年)")
plt.ylabel("资产价格")
plt.show()