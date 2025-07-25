# shapley_allocation.py

import numpy as np
import random
from typing import List, Dict

class DataMarketplace:
    def __init__(self, num_sellers: int, similarity_matrix: np.ndarray = None, lambda_penalty: float = 0.1):
        self.M = num_sellers
        self.similarity_matrix = similarity_matrix if similarity_matrix is not None else np.eye(num_sellers)
        self.lambda_penalty = lambda_penalty
        
    def revenue_function(self, subset: List[int]) -> float:
        # 示例：收益与特征子集大小成正比
        return len(subset) * 10.0

    def marginal_gain(self, subset: List[int], feature: int) -> float:
        return self.revenue_function(subset + [feature]) - self.revenue_function(subset)

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
