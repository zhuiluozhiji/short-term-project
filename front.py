import streamlit as st
import json
import numpy as np
import pandas as pd
from main import main as run_simulation
from io import StringIO

def main():
    st.title("📊 数据市场模拟器")
    st.markdown("""
    这是一个模拟数据市场的应用，可以输入买家数据并查看市场定价和收益分配结果。
    """)

    # 侧边栏设置
    st.sidebar.header("设置")
    fixed_price = st.sidebar.number_input("固定价格(不使用MWU时)", value=50.0, min_value=0.0)
    use_mwu = st.sidebar.checkbox("使用MWU价格更新机制", value=True)
    
    # 主界面
    tab1, tab2 = st.tabs(["输入数据", "运行模拟"])
    
    with tab1:
        st.header("买家数据输入")
        
        # 方法选择
        input_method = st.radio("输入方式", 
                               ["使用示例数据", "手动输入"], 
                               index=0)
        
        buyer_data = []
        
        if input_method == "使用示例数据":
            with open("data/buyer.json", "r",encoding='utf-8') as f:
                example_data = json.load(f)
            st.json(example_data, expanded=False)
            buyer_data = example_data[1:]  # 跳过schema
        elif input_method == "手动输入":
            num_buyers = st.number_input("买家数量", min_value=1, max_value=20, value=3)
            num_features = st.number_input("特征数量", min_value=1, max_value=10, value=3)
            
            for i in range(num_buyers):
                with st.expander(f"买家 {i+1}"):
                    buyer_id = st.number_input(f"买家ID", value=i+1, key=f"id_{i}")
                    mu = st.number_input(f"估值(μ)", min_value=0.0, value=10.0*(i+1), key=f"mu_{i}")
                    
                    st.write("特征矩阵X (6行×{num_features}列):")
                    x_rows = []
                    for row in range(6):
                        cols = st.columns(num_features)
                        x_row = []
                        for col in range(num_features):
                            val = cols[col].number_input(
                                f"行{row+1} 特征{col}", 
                                value=0.1*(row+1) + 0.1*(col+1),
                                key=f"x_{i}_{row}_{col}"
                            )
                            x_row.append(val)
                        x_rows.append(x_row)
                    
                    st.write("目标值Y (6个值):")
                    y_cols = st.columns(6)
                    y_vals = []
                    for j in range(6):
                        y_val = y_cols[j].number_input(
                            f"Y{j+1}", 
                            value=1.0*(j+1),
                            key=f"y_{i}_{j}"
                        )
                        y_vals.append(y_val)
                    
                    buyer_data.append({
                        "buyer_id": buyer_id,
                        "mu": mu,
                        "X": x_rows,
                        "Y": y_vals
                    })
    
    with tab2:
        if not buyer_data:
            st.warning("请先输入买家数据")
            return
            
        if st.button("运行模拟"):
            # 将数据保存到临时文件供主程序使用
            with open("data/buyer_temp.json", "w") as f:
                json.dump([{"_schema": "临时数据"}] + buyer_data, f)
            
            # 运行模拟并捕获输出
            st.write("## 模拟结果")
            st.write("### 买家交易详情")
            
            # 这里需要重构main.py以返回结果而不是直接打印
            # 作为临时方案，我们可以重定向stdout
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            try:
                run_simulation()  # 这会使用我们保存的临时文件
            except Exception as e:
                st.error(f"模拟出错: {str(e)}")
            finally:
                sys.stdout = old_stdout
                
            output = mystdout.getvalue()
            
            # 解析输出显示在Streamlit中
            st.text(output)


if __name__ == "__main__":
    main()