import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, dt, N):
    """
    使用几何布朗运动模拟资产价格
    :param S0: 初始价格
    :param mu: 漂移率（预期收益率）
    :param sigma: 波动率
    :param T: 模拟的总时间
    :param dt: 每次时间步长
    :param N: 模拟的路径数量
    :return: 模拟的价格路径
    """
    num_steps = int(T / dt) + 1  # 时间步数
    times = np.linspace(0, T, num_steps)  # 时间数组
    paths = np.zeros((num_steps, N))  # 存储所有路径的数组
    paths[0] = S0  # 初始化每条路径的起点为 S0
    
    for i in range(1, num_steps):
        # 生成标准正态分布随机数
        Z = np.random.normal(0, 1, N)
        # 根据GBM公式计算下一时间步的价格
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return times, paths

# 模拟参数
S0 = 100  # 初始价格
mu = 0.05  # 漂移率
sigma = 0.2  # 波动率
T = 1  # 模拟的总时间（1年）
dt = 1/252  # 每日时间步长
N = 10  # 模拟10条路径

# 生成GBM路径
times, paths = simulate_gbm(S0, mu, sigma, T, dt, N)

# 可视化模拟的路径
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(times, paths[:, i], lw=1)
plt.title("GBM 模拟的资产价格路径")
plt.xlabel("时间 (年)")
plt.ylabel("资产价格")
plt.show()