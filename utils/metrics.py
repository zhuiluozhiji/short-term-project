# utils/metrics.py
"""
Section 2.2 - 定义了 prediction gain 函数 G(Y, Ŷ) ∈ [0, 1]，用于衡量预测质量。
 )，论文中建议使用：

对回归：1 - RMSE

对分类：Accuracy
"""
import numpy as np

# 方均根
def rmse(y_true, y_pred): 
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def gain_function(y_true, y_pred, task='regression'):
    """
    G(Y, Ŷ) ∈ [0, 1]，用于衡量预测质量。
    regression: 使用 1 - RMSE 的方式。
    classification: 使用 Accuracy。
    """
    if task == 'regression':
        return 1.0 - rmse(y_true, y_pred)
    elif task == 'classification':
        return np.mean(y_true == y_pred)
    else:
        raise ValueError("Unsupported task type.")
