1. update_weights
```py
    def update_weights(self, chosen_price: float, b_n: float, Y: np.ndarray, 
                       X: np.ndarray, revenue_func) -> None:
        """更新权重（基于收益反馈）"""
        for i, c_i in enumerate(self.candidates):
            RF_i, _,_,_,_ = revenue_func(X, Y, c_i, b_n)
            g_i = RF_i / self.B_max  # 归一化收益增益
            self.weights[i] *= (1 + self.delta * g_i)
```
不应该再次重新调用`revenue_func`，再算一遍出来的revenue可能跟主程序中的不一样了 noise随机性太大


因为main.py中是单独更新权重的：
```py
    if price_updater is not None:
        price_updater.update_weights(p_n, b, Y, X, revenue_function_debug)

```
最好将已经算出来的revenue作为参数传入update函数, 同时也能优化性能

2. revenue过于小 考虑适当增大noise来削减积分项里的G