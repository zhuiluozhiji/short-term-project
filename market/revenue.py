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
        
    def revenue_function(self, subset: List[int], pn: float, bn: float, 
                         model_func=train_and_predict, gain_func=gain_function, steps: int = 10) -> float:
        """
        Myerson 风格支付函数：G(bn)*bn - ∫₀^bn G(z) dz
        """
        if len(subset) == 0:
            return 0.0

        X = self.X[:, subset]
        Y = self.Y

        # 防止增益为0导致积分失败
        for _ in range(10):
            X_alloc = allocation_function(X, pn, bn)
            Y_hat = model_func(X_alloc, Y)
            G_bn = gain_func(Y, Y_hat)
            if G_bn > 0:
                break
        else:
            G_bn = 0.0

        # 梯形积分 ∫₀^bn G(z) dz
        zs = np.linspace(0, bn, steps)
        dz = bn / (steps - 1)
        integral = 0.0
        prev_G = None
        for z in zs:
            for _ in range(10):
                X_z = allocation_function(X, pn, z)
                Y_z_hat = model_func(X_z, Y)
                G_z = gain_func(Y, Y_z_hat)
                if G_z <= G_bn and G_z > 0:
                    break
            else:
                G_z = 0.0

            if prev_G is not None:
                integral += 0.5 * (G_z + prev_G) * dz
            prev_G = G_z

        payment = bn * G_bn - integral
        return max(payment, 0.0)

    def marginal_gain(self, subset: List[int], feature: int, pn: float, bn: float) -> float:
        with_feature = self.revenue_function(subset + [feature], pn, bn)
        without_feature = self.revenue_function(subset, pn, bn)
        return with_feature - without_feature

    def approximate_shapley(self, pn: float, bn: float, K: int = 1000) -> Dict[int, float]:
        shapley_values = {m: 0.0 for m in range(self.M)}
        for _ in range(K):
            perm = random.sample(range(self.M), self.M)
            for i, m in enumerate(perm):
                T = perm[:i]
                gain = self.marginal_gain(T, m, pn, bn)
                shapley_values[m] += gain
        return {m: v / K for m, v in shapley_values.items()}

    def allocate_revenue(self, pn: float, bn: float, method: str = 'approximate', K: int = 1000,
                         total_revenue: float = 1.0, enforce_non_negative=True) -> Dict[int, float]:
        if method != 'approximate':
            raise ValueError("Only approximate Shapley is supported.")

        shapley_values = self.approximate_shapley(pn, bn, K)

        payouts = {}
        for i in range(self.M):
            penalty = self.lambda_penalty * 0  # 若有相似度矩阵可加上
            payouts[i] = shapley_values[i] - penalty

        if enforce_non_negative:
            payouts = {i: max(v, 0.0) for i, v in payouts.items()}

        total = sum(payouts.values())
        if total < 1e-8:
            return {i: 0.0 for i in range(self.M)}

        normalized_payouts = {i: (v / total) * total_revenue for i, v in payouts.items()}
        return normalized_payouts

