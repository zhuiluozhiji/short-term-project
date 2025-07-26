推荐项目结构
```
data-marketplace/
│
├── README.md                # 项目简介
├── requirements.txt         # Python 依赖
├── .gitignore
├── LICENSE
│
├── data/                    # 存放模拟/生成的数据
│   └── buyer_data.json
│
├── models/                  # 机器学习模型/预测部分
│   └── learner.py
│
├── market/                  # 拍卖、定价、收益分配逻辑
│   ├── auction.py           # 诚实机制（AF + RF）
│   ├── pricing.py           # MWU 动态定价
│   └── revenue.py           # Shapley 收益分配
│
├── utils/                   # 公共函数
│   └── metrics.py
│   └── io.py
│
└── main.py                  # 主程序入口：输入一个买家 → 运行三步流程
```