

#### 3.1 数据分配函数（AF∗）


在论文的 Example 4.1 中，AF∗ 被具体实现为对输入特征添加噪声：

$$
\tilde{X}_j(t) = X_j(t) + \max(0, p_n - b_n) \cdot \mathcal{N}(0, \sigma^2)
$$

即：若 $b_n < p_n$，则按比例添加高斯噪声。

实现如下：

```python
def allocation_function(X: np.ndarray, pn: float, bn: float, noise_std: float = 1) -> np.ndarray:
    if bn >= pn:
        return X.copy()
    else:
        noise_scale = min(1.0, max(0.1, (pn - bn) / (pn)))
        noise = np.random.normal(0, noise_std * noise_scale, size=X.shape)
        return X + noise
```
其中，我们通过不断测试基于选定训练模型（`model = LinearRegression()`线性拟合 ）下输入数据，即改变buyer.json中传入的买家信息，发现添加噪声
过大过小都会导致和预期实际效果有差别：
- 噪声过大：会导致数据完全失去原有相关性，采用$1 - MSE$方式得到的预期收益`G`有可能因为均方差过大直接变为负数!
- 噪声过小：即使噪声干扰下的X_alloc依旧能被训练达到较高的拟合度（预期增益`G`未明显削减），尤其是对于鲁棒性更强大的训练模型（猜测）！导致出价`bn`较小的代价变得微乎其微，Myerson支付函数 
$
\text{RF}^*(p_n, b_n, Y_n) = b_n \cdot G(Y_n, \hat{Y}_n) - \int_0^{b_n} G(Y_n, \hat{Y}(z)) dz
$
右式 减数(累积边际增益积分) 和 被减数（边际增益 × 出价）大小相近，导致最后买家需要支付的revenue非常小，不符合现实情况


因此我们针对噪声做了如下优化：相当于对高斯噪声进行合理化限制
* 使用 $(pn - bn)/(pn)$ 归一化比值控制噪声幅度；
* 将噪声强度限制在合理范围 $[0.1, 1.0]$，避免噪声过小或过大，提升了鲁棒性；


这保证了 AF∗ 是一个大样本下的单调函数（出价越高，预测质量越高），满足论文对 truthful bidding 的关键假设。

---

#### 3.2 收益函数（RF∗）



论文式 (3) 给出了基于 Myerson 机制的支付函数：

$$
\text{RF}^*(p_n, b_n, Y_n) = b_n \cdot G(Y_n, \hat{Y}_n) - \int_0^{b_n} G(Y_n, \hat{Y}(z)) dz
$$
* 第一项：买家按其出价乘以模型真实的预测提升付费；
* 第二项：从 0 到出价的累计精度提升积分（作为折扣）。


* $G(Y, \hat{Y})$：衡量预测质量的“增益函数”，可采用分类准确率或 $1 - \text{RMSE}$；


实现如下：

```python
def revenue_function(X, Y, pn, bn, model_func, gain_func, steps=10):
    # 第一项：计算在出价 bn 下的预测效果
    X_alloc = allocation_function(X, pn, bn)
    Y_hat = model_func(X_alloc, Y)
    G_bn = gain_func(Y, Y_hat)

    # 第二项：使用梯形法近似积分 ∫₀^bn G(z) dz
    zs = np.linspace(0, bn, steps)
    integral = 0.0
    prev_G = None
    for z in zs:
        X_z = allocation_function(X, pn, z)
        Y_z_hat = model_func(X_z, Y)
        G_z = gain_func(Y, Y_z_hat)
        if prev_G is not None:
            integral += 0.5 * (G_z + prev_G) * (bn / (steps - 1))
        prev_G = G_z

    revenue = bn * G_bn - integral
    return revenue, X_alloc, Y_hat, G_bn, integral
```

使用 `train_and_predict` 训练模型并获得预测 $\hat{Y}$，这里我们使用线性拟合的模型 `model = LinearRegression()` 。同时使用 `gain_function` 计算精度；
```py
def gain_function(y_true, y_pred, task='regression'):
    if task == 'regression':
        return 1.0 - rmse(y_true, y_pred)
    elif task == 'classification':
        return np.mean(y_true == y_pred)
    else:
        raise ValueError("Unsupported task type.")
```

积分项我们采用梯形法数值求和来近似积分计算，通过`steps`参数来自行调控微分近似精度


该函数输出的不仅包括最终的 revenue，我们还增加了预测输出、分配后的特征 $\tilde{X}$、增益值与积分项，便于我们后续调试,以及更重要的是与后续的定价权重更新函数、主函数等提供传输关键量的接口

