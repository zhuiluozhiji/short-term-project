# market_simulation.py

import numpy as np
from typing import Optional

from utils.io import load_buyer_data
from market.auction import allocation_function, revenue_function, revenue_function_origin
from models.learner import train_and_predict
from market.revenue import DataMarketplace

from utils.metrics import gain_function
from market.pricing import MWUPriceUpdater

FIXED_PRICE = 50.0

def run_single_buyer(buyer, price_updater: Optional[MWUPriceUpdater] = None):
    buyer_id = buyer["buyer_id"]
    mu = buyer["mu"]
    X = buyer["X"]
    Y = buyer["Y"]
    b = mu
    p_n = price_updater.choose_price() if price_updater is not None else FIXED_PRICE

    print(f"📦 买家 {buyer_id} 到达：")
    print(f"   - 真实估值 mu = {mu:.2f}")
    print(f"   - 出价 b = {b:.2f}")
    print(f"   - 当前市场定价 p = {p_n:.2f}")
    revenue, X_tilde, Y_hat, gain, integral = revenue_function(X, Y, p_n, b)
    # X_tilde = allocation_function(X, p_n, b)
    # Y_hat = train_and_predict(X_tilde, Y)
    print(f"   - 分配前的特征 X: {X}")
    print(f"   - 分配后的特征 X_tilde: {X_tilde}")
    print(f"   - 真实值 Y: {Y}")
    print(f"   - 预测值 Y_hat: {Y_hat}")
    # print(f"刚调用完train_and_predict")
    # print(f"刚调用完gain_function")

    print(f"✅ 预测增益 G = {gain:.4f}")

    
    if price_updater is not None:
        price_updater.update_weights(p_n, b, Y, X, revenue_function)

    print(f"💰 买家需支付 Revenue = {revenue:.4f}")
    # print(f"💡 净效用（G*b - revenue） = {gain * b - revenue:.4f}")
    print("-" * 60)
    
    return p_n, revenue, X_tilde


def main():
    buyers = load_buyer_data("data/buyer.json")
    if not buyers:
        print("⚠️ 未找到买家数据，请检查 data/buyer.json")
        return

    price_updater = MWUPriceUpdater(B_max=100.0, L=1.0, N=len(buyers))
    num_features = buyers[0]["X"].shape[1]
    seller_revenue = {i: 0.0 for i in range(num_features)}
    prices = []

    for buyer in buyers:
        p_n, revenue, X_tilde = run_single_buyer(buyer, price_updater)

        prices.append(p_n)
           
        used_features = list(np.where(X_tilde.std(axis=0) != 0)[0])
        if not used_features:
            continue

        X_used = X_tilde[:, used_features]
        Y = buyer["Y"]
        sub_market = DataMarketplace(X_used, Y)

        shapley_weights = sub_market.allocate_revenue(
            pn=p_n,
            bn=buyer["mu"],
            method='approximate',
            K=100,
            total_revenue=revenue  # 用买家的实际支付作为总收益进行分配
        )


        for local_idx, weight in shapley_weights.items():
            global_idx = used_features[local_idx]
            seller_revenue[global_idx] += revenue * weight
        
    
    print("\n📊 市场价格动态:")
    for i, p in enumerate(prices):
        print(f"买家 {i+1}: p = {p:.2f}")

    print("\n🏦 卖家总收益分配（基于边际贡献）:")
    for seller_id, rev in seller_revenue.items():
        print(f"特征 {seller_id}: 收益 = {rev:.4f}")
    
if __name__ == "__main__":
    main()
