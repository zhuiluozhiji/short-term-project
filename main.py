# 仅仅用于参考测试

import numpy as np
from utils.io import load_buyer_data
from market.auction import allocation_function, revenue_function
from models.learner import train_and_predict
from utils.metrics import gain_function

# 设定一个固定市场价格（可由 MWU 动态决定）
FIXED_PRICE = 50.0  # p_n

def run_single_buyer(buyer, price=FIXED_PRICE):
    """
    执行一个买家的完整交易流程：分配、训练、预测、收益、支付
    """
    buyer_id = buyer["buyer_id"]
    mu = buyer["mu"]           # 买家的真实估值
    X = buyer["X"]             # 特征数据
    Y = buyer["Y"]             # 标签
    b = mu                     # 本机制鼓励出价 b = mu（诚实）

    print(f"📦 买家 {buyer_id} 到达：")
    print(f"   - 真实估值 mu = {mu:.2f}")
    print(f"   - 出价 b = {b:.2f}")
    print(f"   - 当前市场定价 p = {price:.2f}")

    # 分配数据（可能加噪）
    X_tilde = allocation_function(X, price, b)

    # 模型训练和预测
    Y_hat = train_and_predict(X_tilde, Y)

    # 计算预测增益
    gain = gain_function(Y, Y_hat)

    # 计算买家的支付金额
    revenue = revenue_function(X, Y, price, b)

    # 展示结果
    print(f"✅ 预测增益 G = {gain:.4f}")
    print(f"💰 买家需支付 Revenue = {revenue:.4f}")
    print(f"💡 净效用（G*b - revenue） = {gain * b - revenue:.4f}")
    print("-" * 60)

def main():
    # 加载 buyer 数据（支持多个）
    buyers = load_buyer_data("data/buyer.json")

    # 仅测试第一个买家（单人流程）
    if buyers:
        run_single_buyer(buyers[0])
    else:
        print("⚠️ 未找到买家数据，请检查 data/buyer_data.json")

if __name__ == "__main__":
    main()
