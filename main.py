# 仅仅用于参考测试

import numpy as np
from utils.io import load_buyer_data
from market.auction import allocation_function, revenue_function
from models.learner import train_and_predict
from utils.metrics import gain_function
from market.pricing import MWUPriceUpdater 
from typing import Optional 

# 设定一个固定市场价格（可由 MWU 动态决定）
FIXED_PRICE = 50.0  # p_n

def run_single_buyer(buyer, price_updater: Optional[MWUPriceUpdater] = None) -> float:
    """
    执行一个买家的完整交易流程：分配、训练、预测、收益、支付
    """
    buyer_id = buyer["buyer_id"]
    mu = buyer["mu"]           # 买家的真实估值
    X = buyer["X"]             # 特征数据
    Y = buyer["Y"]             # 标签
    b = mu                     # 本机制鼓励出价 b = mu（诚实）

    p_n = price_updater.choose_price() if price_updater is not None else 50.0


    print(f"📦 买家 {buyer_id} 到达：")
    print(f"   - 真实估值 mu = {mu:.2f}")
    print(f"   - 出价 b = {b:.2f}")
    print(f"   - 当前市场定价 p = {p_n:.2f}")

    # 分配数据（可能加噪）
    X_tilde = allocation_function(X, p_n, b)

    # 模型训练和预测
    Y_hat = train_and_predict(X_tilde, Y)

    # 计算预测增益
    gain = gain_function(Y, Y_hat)

    # 计算买家的支付金额
    revenue = revenue_function(X, Y, p_n, b)

    if price_updater is not None:
        price_updater.update_weights(p_n, b, Y, X, revenue_function)

    # 展示结果
    print(f"✅ 预测增益 G = {gain:.4f}")
    print(f"💰 买家需支付 Revenue = {revenue:.4f}")
    print(f"💡 净效用（G*b - revenue） = {gain * b - revenue:.4f}")
    print("-" * 60)
    return p_n

def main():
    # 加载 buyer 数据（支持多个）
    buyers = load_buyer_data("data/buyer.json")

    # 仅测试第一个买家（单人流程）
    if buyers:
        run_single_buyer(buyers[0])
    else:
        print("⚠️ 未找到买家数据，请检查 data/buyer_data.json")

    # 初始化 MWU 动态定价（参数需根据实际数据调整）
    price_updater = MWUPriceUpdater(
        B_max=100.0,  # 假设最大可能价格
        L=1.0,        # Lipschitz 常数（需实验估计）
        N=len(buyers)  # 买家总数
    )

    # 遍历所有买家，运行动态定价
    prices = []
    for buyer in buyers:
        p_n = run_single_buyer(buyer, price_updater)
        prices.append(p_n)

    # 输出价格动态变化
    print("\n📊 市场价格动态:")
    for i, p in enumerate(prices):
        print(f"买家 {i+1}: p = {p:.2f}")

if __name__ == "__main__":
    main()
