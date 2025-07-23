import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple
import random
from sklearn.metrics.pairwise import cosine_similarity

class DataMarketplace:
    def __init__(self, 
                 num_sellers: int,
                 similarity_matrix: np.ndarray = None,
                 lambda_penalty: float = 0.1):
        """
        初始化数据市场
        :param num_sellers: 卖家数量(特征数量)
        :param similarity_matrix: 特征相似度矩阵(用于鲁棒性方法)
        :param lambda_penalty: 相似度惩罚系数
        """
        self.M = num_sellers
        self.similarity_matrix = similarity_matrix if similarity_matrix is not None else np.eye(num_sellers)
        self.lambda_penalty = lambda_penalty
        
    def revenue_function(self, subset: List[int]) -> float:
        """
        收益函数(示例实现)
        实际应用中应替换为真实的收益计算逻辑
        :param subset: 特征子集
        :return: 该子集产生的收益
        """
        # 示例: 收益与子集大小正相关(实际应用中应使用真实模型)
        return len(subset) * 10.0
    
    def marginal_gain(self, subset: List[int], feature: int) -> float:
        """
        计算边际增益
        :param subset: 当前特征子集
        :param feature: 待添加的特征
        :return: 边际增益值
        """
        return self.revenue_function(subset + [feature]) - self.revenue_function(subset)
    
    def exact_shapley(self) -> Dict[int, float]:
        """
        精确Shapley值计算(标准分配方法)
        时间复杂度: O(M!)
        :return: 各特征的Shapley值分配
        """
        shapley_values = {m: 0.0 for m in range(self.M)}
        
        # 遍历所有可能的特征排列
        for permutation in permutations(range(self.M)):
            # 计算每个特征在排列中的边际贡献
            for i, m in enumerate(permutation):
                # 前i个特征构成的子集
                T = list(permutation[:i])
                # 计算边际增益
                gain = self.marginal_gain(T, m)
                # 累加到Shapley值
                shapley_values[m] += gain
        
        # 平均化(除以M!)
        factorial = 1
        for i in range(1, self.M+1):
            factorial *= i
        for m in shapley_values:
            shapley_values[m] /= factorial
            
        return shapley_values
    
    def approximate_shapley(self, K: int = 1000) -> Dict[int, float]:
        """
        采样近似Shapley值计算(Algorithm 2)
        时间复杂度: O(K*M)
        :param K: 采样排列数
        :return: 近似Shapley值分配
        """
        shapley_values = {m: 0.0 for m in range(self.M)}
        
        for _ in range(K):
            # 随机生成一个排列
            permutation = random.sample(range(self.M), self.M)
            # 计算每个特征在排列中的边际贡献
            for i, m in enumerate(permutation):
                T = permutation[:i]
                gain = self.marginal_gain(T, m)
                shapley_values[m] += gain
        
        # 平均化
        shapley_values = {m: v/K for m, v in shapley_values.items()}
        return shapley_values
    
    def robust_shapley(self, K: int = 1000) -> Dict[int, float]:
        """
        鲁棒性Shapley值计算(Algorithm 3)
        加入特征相似度惩罚
        :param K: 采样排列数
        :return: 鲁棒性Shapley值分配
        """
        # 先计算近似Shapley值
        shapley_values = self.approximate_shapley(K)
        
        # 计算相似度惩罚项
        similarity_penalty = {m: 0.0 for m in range(self.M)}
        for m in range(self.M):
            for m_prime in range(self.M):
                if m != m_prime:
                    similarity_penalty[m] += self.similarity_matrix[m][m_prime]
        
        # 应用惩罚(式(3)的简化实现)
        for m in shapley_values:
            shapley_values[m] -= self.lambda_penalty * similarity_penalty[m]
            # 确保值非负
            shapley_values[m] = max(0, shapley_values[m])
        
        # 重新归一化(保持总和不变)
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {m: v/total for m, v in shapley_values.items()}
        
        return shapley_values
    
    def allocate_revenue(self, method: str = 'approximate', K: int = 1000) -> Dict[int, float]:
        """
        收益分配主函数
        :param method: 分配方法('exact', 'approximate'或'robust')
        :param K: 采样排列数(仅对近似方法有效)
        :return: 各卖家分配的收益比例
        """
        if method == 'exact':
            shapley_values = self.exact_shapley()
        elif method == 'approximate':
            shapley_values = self.approximate_shapley(K)
        elif method == 'robust':
            shapley_values = self.robust_shapley(K)
        else:
            raise ValueError("Invalid method. Choose from 'exact', 'approximate' or 'robust'")
        
        # 计算总Shapley值(用于归一化)
        total = sum(shapley_values.values())
        if total <= 0:
            return {m: 0.0 for m in range(self.M)}
        
        # 归一化分配比例
        allocation = {m: v/total for m, v in shapley_values.items()}
        return allocation

# 示例用法
if __name__ == "__main__":
    from itertools import permutations
    
    # 1. 初始化市场(假设有5个卖家)
    num_sellers = 5
    marketplace = DataMarketplace(num_sellers=num_sellers)
    
    # 2. 可选: 设置特征相似度矩阵(示例使用随机相似度)
    similarity_matrix = np.random.rand(num_sellers, num_sellers)
    np.fill_diagonal(similarity_matrix, 1)  # 对角线为1(特征与自身完全相似)
    marketplace.similarity_matrix = similarity_matrix
    
    # 3. 选择分配方法并计算分配比例
    print("=== 精确Shapley分配 ===")
    exact_allocation = marketplace.allocate_revenue(method='exact')
    print(exact_allocation)
    
    print("\n=== 采样近似分配(K=1000) ===")
    approx_allocation = marketplace.allocate_revenue(method='approximate', K=1000)
    print(approx_allocation)
    
    print("\n=== 鲁棒性分配(K=1000) ===")
    robust_allocation = marketplace.allocate_revenue(method='robust', K=1000)
    print(robust_allocation)