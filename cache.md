
**论文术语**：$\mu, b, p_n, X, Y, \tilde{X}, G, \text{RF}$

---

论文关键术语

| 符号/变量       | 程序中体现                                | 实际含义              |
| ----------- | ------------------------------------ | ----------------- |
| $\mu$       | `mu = buyer["mu"]`                   | 买家愿意为 1 单位提升支付的价格 |
| $b$         | `b = mu`                             | 买家出价（我们简化设为 μ）    |
| $p_n$       | `p_n = price_updater.choose_price()` | 当前市场定价            |
| $X$         | `X = buyer["X"]`                     | 所有卖家提供的特征矩阵       |
| $\tilde{X}$ | `X_tilde = allocation_function(...)` | 根据 p 和 b 分配的数据    |
| $Y$         | `Y = buyer["Y"]`                     | 预测目标              |
| $\hat{Y}$   | `Y_hat = train_and_predict(...)`     | 预测值               |
| $G$         | `gain = gain_function(...)`          | 模型的预测效果提升         |
| RF          | `revenue = revenue_function(...)`    | 实际支付金额            |
| Shapley     | `allocate_revenue(...)`              | 卖家的边际贡献分配收益       |


