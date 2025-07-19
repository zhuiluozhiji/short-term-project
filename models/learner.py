"""
定义简单模型 train_and_predict
对应论文：
Section 2.3 - M(X) → Ŷ 是一个任意机器学习模型，可自定义
Section 4.1 - 举例说可用 Sklearn 模型、SparkML 等
💡我们先用最简单的 LinearRegression 进行回归预测
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def train_and_predict(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    给定特征 X 和标签 Y，拟合模型并预测 Y（使用自身训练数据）
    """
    model = LinearRegression() # 创建模型实例
    model.fit(X, Y)  # 训练模型，学习 X 和 Y 的关系
    return model.predict(X)  # 使用学到的关系预测 Y
    """
    什么是 LinearRegression？
        线性回归模型：寻找特征 X 和目标 Y 之间的线性关系
        来自 sklearn：Python 最流行的机器学习库
        监督学习算法：需要训练数据（X, Y）来学习模式
    """

"""
一个例子说明：
import numpy as np
from sklearn.linear_model import LinearRegression

# 创建简单的训练数据
X = np.array([[1], [2], [3], [4], [5]])  # 特征
Y = np.array([2, 4, 6, 8, 10])           # 目标（y = 2x 的关系）

# 使用线性回归
model = LinearRegression()
model.fit(X, Y)  # 模型学习：发现 y ≈ 2x 的关系

# 查看学到的参数
print(f"系数 (斜率): {model.coef_[0]:.2f}")      # 应该接近 2
print(f"截距: {model.intercept_:.2f}")           # 应该接近 0

# 进行预测
Y_pred = model.predict(X)
print(f"预测结果: {Y_pred}")  # 应该接近 [2, 4, 6, 8, 10]
"""

