# shapley_allocation.py

import numpy as np
import random
from typing import List, Dict
from models.learner import train_and_predict
from utils.metrics import gain_function
from market.auction import revenue_function
from market.auction import allocation_function

class DataMarketplace:
    def __init__(self, num_sellers: int, X: np.ndarray, Y: np.ndarray,  pn: float, bn: float, similarity_matrix: np.ndarray = None, lambda_penalty: float = 0.1):
        self.M = num_sellers
        self.similarity_matrix = similarity_matrix if similarity_matrix is not None else np.eye(num_sellers)
        self.lambda_penalty = lambda_penalty
        # 以下是新加入的来自auction中必要的成员变量，避免函数参数冗余调用
        self.X = X
        self.Y = Y
        self.pn = pn
        self.bn = bn
    '''
    def revenue_function(self, subset: List[int]) -> float:
        # 示例：收益与特征子集大小成正比
        print(f"调用revenue.py中的同名函数")
        return len(subset) * 10.0
    '''

    '''
    def marginal_gain(self, subset: List[int], feature: int) -> float:
        return self.revenue_function(subset + [feature]) - self.revenue_function(subset)
    '''

    def marginal_gain(self, subset: List[int], feature: int) -> float:
        """
        计算特征的边际增益。
        使用 auction.py 中的 revenue_function 和 allocation_function。
        """
        # 构造包含新特征的子集
        subset_with_feature = subset + [feature]

        # 计算包含新特征的子集的收益
        revenue_with_feature = revenue_function(
            self.X[:, subset_with_feature],  # 提取子集对应的特征列
            self.Y,
            self.pn,
            self.bn,
            model_func=train_and_predict,
            gain_func=gain_function
        )

        # 计算不包含新特征的子集的收益
        revenue_without_feature = revenue_function(
            self.X[:, subset],  # 提取子集对应的特征列
            self.Y,
            self.pn,
            self.bn,
            model_func=train_and_predict,
            gain_func=gain_function
        )

        # 返回边际增益
        return revenue_with_feature - revenue_without_feature

       
        
    def approximate_shapley(self, K: int = 1000) -> Dict[int, float]:
        shapley_values = {m: 0.0 for m in range(self.M)}
        for _ in range(K):
            permutation = random.sample(range(self.M), self.M)
            for i, m in enumerate(permutation):
                T = permutation[:i]
                gain = self.marginal_gain(T, m)
                shapley_values[m] += gain
        shapley_values = {m: v / K for m, v in shapley_values.items()}
        return shapley_values

    def allocate_revenue(self, method: str = 'approximate', K: int = 1000) -> Dict[int, float]:
        if method == 'approximate':
            shapley_values = self.approximate_shapley(K)
        else:
            raise ValueError("Only approximate Shapley supported in this demo")
        total = sum(shapley_values.values())
        if total <= 0:
            return {m: 0.0 for m in range(self.M)}
        return {m: v / total for m, v in shapley_values.items()}
