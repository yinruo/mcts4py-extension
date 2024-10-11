import numpy as np

class BinomialTreeOption:
    def __init__(self, S0, K, r, T, sigma, dt, option_type="Call"):
        self.S0 = S0  # 初始股票价格
        self.K = K  # 行权价格
        self.r = r  # 无风险利率
        self.T = T  # 期权到期时间
        self.sigma = sigma  # 波动率
        self.n = int(T / dt)  # 树的层数（时间步）
        self.option_type = option_type  # "Call" or "Put"
        self.dt = dt # 每个时间步的时间
        self.u = np.exp(sigma * np.sqrt(self.dt))  # 股票价格上涨的比例
        self.d = 1 / self.u  # 股票价格下跌的比例
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)  # 上涨的概率
        self.discount = np.exp(-r * self.dt)  # 折现因子

    def get_payoff(self, S):
        """计算期权的收益"""
        if self.option_type == "Put":
            return max(self.K - S, 0)  # Put的收益：max(K - S, 0)
        elif self.option_type == "Call":
            return max(S - self.K, 0)  # Call的收益：max(S - K, 0)

    def price(self):
        # 初始化价格树
        price_tree = np.zeros((self.n + 1, self.n + 1))
        stock_tree = np.zeros((self.n + 1, self.n + 1))

        # 初始化最后一期的期权价值和股票价格
        for i in range(self.n + 1):
            stock_tree[i, self.n] = self.S0 * (self.u ** (self.n - i)) * (self.d ** i)
            price_tree[i, self.n] = self.get_payoff(stock_tree[i, self.n])

        # 回溯计算
        for j in range(self.n - 1, -1, -1):
            for i in range(j + 1):
                hold_value = (self.p * price_tree[i, j + 1] + (1 - self.p) * price_tree[i + 1, j + 1]) * self.discount
                stock_tree[i, j] = self.S0 * (self.u ** (j - i)) * (self.d ** i)
                exercise_value = self.get_payoff(stock_tree[i, j])
                price_tree[i, j] = max(hold_value, exercise_value)

        """ print("股票价格树：")
        self.print_tree(stock_tree)

        print("\n期权价格树：")
        self.print_tree(price_tree) """

        return price_tree[0, 0]

    def print_tree(self, tree):
        """打印树的结构"""
        for j in range(self.n + 1):
            print(f"第 {j} 期: ", end="")
            for i in range(j + 1):
                print(f"{tree[i, j]:.4f} ", end="")
            print()

if __name__ == "__main__":
    # 使和 USoptionMDP 相同的参数
    S0 = 1  # 初始股票价格
    K = 0.95  # 行权价格
    r = 0.01  # 无风险利率
    T = 5  # 到期时间1年
    sigma = 0.15  # 波动率
    dt = 1/2  # 时间步长（即每个步的时间间隔）
    n = int(T / dt)  # 二叉树的时间步数
    option_type = "Put"  # 期权类型Call代表看涨期权

    binomial_model = BinomialTreeOption(S0, K, r, T, sigma, dt, option_type)

    binomial_price = binomial_model.price()
    print(f"\n二叉树方法计算的美式期权价格为: {binomial_price:.4f}")
