import numpy as np
from typing import List, Optional  # 新增 Optional 类型

class MWUPriceUpdater:
    def __init__(self, B_max: float, L: float, N: int, epsilon: Optional[float] = None):
        """
        MWU 动态定价算法
        :param B_max: 最大可能价格（买家估值上限）
        :param L: 收益函数的 Lipschitz 常数
        :param N: 预期买家总数
        :param epsilon: 价格网格精度（可选，默认自动计算）
        """
        self.B_max = B_max
        self.L = L
        self.N = N
        
        # 自动计算 epsilon（如果未提供）
        if epsilon is None:
            epsilon_value = 1 / (L * np.sqrt(N))
        else:
            epsilon_value = epsilon
        self.epsilon = float(epsilon_value)  # 确保 epsilon 是 float 类型
        
        # 生成候选价格网格
        self.candidates = np.arange(0, B_max + self.epsilon, self.epsilon)
        self.num_candidates = len(self.candidates)
        
        # 初始化权重和学习率
        self.weights = np.ones(self.num_candidates)
        self.delta = np.sqrt(np.log(self.num_candidates) / N)

    def choose_price(self) -> float:
        """选择当前价格（按权重随机采样）"""
        total_weight = np.sum(self.weights)
        prob = self.weights / total_weight
        chosen_idx = np.random.choice(self.num_candidates, p=prob)
        return float(self.candidates[chosen_idx])

    def update_weights(self, chosen_price: float, b_n: float, Y: np.ndarray, 
                       X: np.ndarray, revenue_func) -> None:
        """更新权重（基于收益反馈）"""
        for i, c_i in enumerate(self.candidates):
            RF_i = revenue_func(X, Y, c_i, b_n)
            g_i = RF_i / self.B_max  # 归一化收益增益
            self.weights[i] *= (1 + self.delta * g_i)
