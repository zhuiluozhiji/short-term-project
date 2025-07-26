"""
✅ 第一阶段：完成 market/auction.py 中的 AF（分配函数）与 RF（收入函数）
"""
import numpy as np
from models.learner import train_and_predict
from utils.metrics import gain_function
'''
def allocation_function(X: np.ndarray, pn: float, bn: float, noise_std: float = 0.001) -> np.ndarray:
    """
    AF∗: 对应论文 Section 4.1 Example 4.1
    若买家出价 bn < 当前价格 pn，添加噪声；否则返回原始特征 X
    """
    if bn >= pn:
        return X.copy()
    else:
        noise = np.random.normal(0, noise_std, size=X.shape)
        return X + (pn - bn) * noise 
'''
def allocation_function(X: np.ndarray, pn: float, bn: float, noise_std: float = 1) -> np.ndarray:
    """
    AF∗: 对应论文 Section 4.1 Example 4.1
    若买家出价 bn < 当前价格 pn，添加适度噪声；否则返回原始特征 X。
    """
    if bn >= pn:
        return X.copy()
    else:
        # 根据 (pn - bn) 的比例动态调整噪声幅度，避免噪声过大
        noise_scale = min(1.0, max(0.1, (pn - bn) / (pn + bn)))  # 噪声比例限制在 [0.1, 1.0]
        noise = np.random.normal(0, noise_std * noise_scale, size=X.shape)
        X_tilde = X + noise
        # print(f"(debug) 添加噪声后的 X_tilde: {X_tilde}")
        return X_tilde


def revenue_function_origin(X: np.ndarray, Y: np.ndarray, pn: float, bn: float, 
                     model_func=train_and_predict, gain_func=gain_function, steps: int = 100) -> float:
    """
    Myerson 风格支付函数（论文 Section 4.1，式 (3)）
    收益 = 边际增益 × 出价 - 累积边际增益积分
    """
    while True:
        X_alloc = allocation_function(X, pn, bn)
        Y_hat = model_func(X_alloc, Y)
        G_bn = gain_func(Y, Y_hat)
        if G_bn > 0:
            break




    # 第二项：近似积分 ∫₀^bn G(z) dz，使用 Trapezoidal rule
    zs = np.linspace(0, bn, steps)
    dz = bn / (steps - 1)
    integral = 0.0

    prev_G = None
    for z in zs:
        X_z = allocation_function(X, pn, z)
        Y_z_hat = model_func(X_z, Y)
        G_z = gain_func(Y, Y_z_hat)

        if prev_G is not None:
            integral += 0.5 * (G_z + prev_G) * dz  # 梯形面积
        prev_G = G_z

    # print(f"[调试] 出价 bn: {bn:.2f}, G(bn): {G_bn:.4f}, ∫G(z)dz ≈ {integral:.4f}")

    return bn * G_bn - integral



def revenue_function(X: np.ndarray, Y: np.ndarray, pn: float, bn: float, 
                     model_func=train_and_predict, gain_func=gain_function, steps: int = 100) -> float:
    """
    Myerson 风格支付函数（论文 Section 4.1，式 (3)）
    收益 = 边际增益 × 出价 - 累积边际增益积分
    """
    # 第一项：买家实际的预测增益
    while True:
        X_alloc = allocation_function(X, pn, bn)
        Y_hat = model_func(X_alloc, Y)
        G_bn = gain_func(Y, Y_hat)
        if G_bn > 0:
            break
        
    # 第二项：近似积分 ∫₀^bn G(z) dz，使用 Trapezoidal rule
    zs = np.linspace(0, bn, steps)
    dz = bn / (steps - 1)
    integral = 0.0

    prev_G = None
    for z in zs:
        while True:
            X_z = allocation_function(X, pn, z)
            Y_z_hat = model_func(X_z, Y)
            G_z = gain_func(Y, Y_z_hat)
            if G_z > 0:
                break


        if prev_G is not None:
            integral += 0.5 * (G_z + prev_G) * dz  # 梯形面积
        prev_G = G_z



    revenue = bn * G_bn - integral
    return revenue, X_alloc, Y_hat, G_bn, integral

