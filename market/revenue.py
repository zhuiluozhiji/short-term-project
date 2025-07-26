# shapley_allocation.py

import numpy as np
import random
from typing import List, Dict

from market.auction import allocation_function
from models.learner import train_and_predict
from utils.metrics import gain_function

class DataMarketplace:
    def __init__(self, X: np.ndarray, Y: np.ndarray, lambda_penalty: float = 0.1):
        """
        X: (样本数, 特征数) numpy array
        Y: (样本数, 标签维度) numpy array
        """
        self.X = X
        self.Y = Y
        self.M = X.shape[1]  # 特征数量
        self.lambda_penalty = lambda_penalty

    def revenue_function(self, subset: List[int], pn: float, bn: float) -> float:
        """
        计算给定特征子集 subset 在价格 pn 和买家出价 bn 下的收益（支付）
        """
        if len(subset) == 0:
            return 0.0

        X_subset = self.X[:, subset]
        Y = self.Y

        # 买家实际获得的收益增益 G(bn)
        X_alloc = allocation_function(X_subset, pn, bn)
        Y_hat = train_and_predict(X_alloc, Y)
        G_bn = gain_function(Y, Y_hat)

        # 数值积分 ∫0^bn G(z) dz
        steps = 100
        zs = np.linspace(0, bn, steps)
        integral = 0.0
        for z in zs:
            X_z = allocation_function(X_subset, pn, z)
            Y_z_hat = train_and_predict(X_z, Y)
            G_z = gain_function(Y, Y_z_hat)
            integral += G_z * (bn / steps)

        payment = bn * G_bn - integral
        payment = max(payment, 0.0)
        return payment

    def marginal_gain(self, subset: List[int], feature: int, pn: float, bn: float) -> float:
        return self.revenue_function(subset + [feature], pn, bn) - self.revenue_function(subset, pn, bn)

    def approximate_shapley(self, pn: float, bn: float, K: int = 1000) -> Dict[int, float]:
        shapley_values = {m: 0.0 for m in range(self.M)}
        for _ in range(K):
            permutation = random.sample(range(self.M), self.M)
            for i, m in enumerate(permutation):
                T = permutation[:i]
                gain = self.marginal_gain(T, m, pn, bn)
                shapley_values[m] += gain
        # 平均
        shapley_values = {m: v / K for m, v in shapley_values.items()}
        return shapley_values

    def allocate_revenue(self, pn: float, bn: float, method: str = 'approximate', K: int = 1000,
                         total_revenue: float = 1.0, enforce_non_negative=True) -> Dict[int, float]:
        if method == 'approximate':
            shapley_values = self.approximate_shapley(pn, bn, K)
        else:
            raise ValueError("Only approximate Shapley supported")

        # 引入简单相似度惩罚（如果需要，可改成传入相似度矩阵）
        payouts = {}
        for i in range(self.M):
            penalty = self.lambda_penalty * 0  # 这里示例无相似度矩阵，故为0
            payouts[i] = shapley_values[i] - penalty

        if enforce_non_negative:
            payouts = {i: max(v, 0.0) for i, v in payouts.items()}

        total = sum(payouts.values())
        if total < 1e-8:
            return {i: 0.0 for i in range(self.M)}

        # 归一化乘以总收益
        normalized_payouts = {i: (v / total) * total_revenue for i, v in payouts.items()}
        return normalized_payouts
