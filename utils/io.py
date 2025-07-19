# utils/io.py (改进版)
import json
import numpy as np

def load_buyer_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 过滤掉 schema 说明
    buyers = []
    for item in data:
        # 跳过包含 _schema 的元素
        if '_schema' in item:
            continue
        
        # 转为 numpy 格式方便计算
        buyer = item.copy()
        buyer['X'] = np.array(buyer['X'])  # 特征矩阵转为 numpy 数组
        buyer['Y'] = np.array(buyer['Y'])  # 目标变量转为 numpy 数组
        buyers.append(buyer)
    
    return buyers