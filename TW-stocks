import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import streamlit as pd_st  # 避免命名衝突
import streamlit as st

# =========================================================================
# 頁面基本設定
# =========================================================================
st.set_page_config(page_title="台股 ETF 投資組合最佳化大腦", layout="wide")
st.title("🧠 台股與 ETF 最佳化投資組合大腦")
st.markdown("融合「歷史資產配置最佳化回測」與「盤中即時警示監控」的全方位量化系統")

# =========================================================================
# 側邊欄參數設定 (Sidebar)
# =========================================================================
st.sidebar.header("⚙️ 系統參數設定")

# 預設優異標的清單
DEFAULT_TICKERS = ["2330.TW", "2454.TW", "2317.TW", "0050.TW", "0056.TW", "00919.TW", "006208.TW"]
selected_tickers = st.sidebar.multiselect(
    "選擇觀測台股/ETF 清單 (yfinance 格式)",
    options=DEFAULT_TICKERS,
    default=DEFAULT_TICKERS
)

benchmark = st.sidebar.selectbox("選擇對比基準點 (Benchmark)", options=["0050.TW", "006208.TW"], index=0)

# 日期選擇
start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-01-01"))

# 無風險利率設定
risk_free_rate = st.sidebar.slider("台灣市場預估無風險利率 (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1) / 100

# =========================================================================
# 核心計算邏輯函數 (Optimization Brain)
# =========================================================================
def portfolio_performance(weights, annual_returns, cov_matrix):
    """計算投資組合的年化報酬率與年化波動度"""
    port_return = np.sum(annual_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def negative_sharpe_ratio(weights, annual_returns, cov_matrix, risk_free_rate):
    """計算負夏普比率（用於極大化求解）"""
    p_ret, p_vol = portfolio_performance(weights, annual_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def portfolio_volatility(weights, annual_returns, cov_matrix):
    """單純返回波動度（用於極小化變異數求解）"""
    return portfolio_performance(weights, annual_returns, cov_matrix)[1]

def optimize_portfolio(annual_returns, cov_matrix, mode="MAX_SHARPE"):
    """使用 SciPy SLSQP 求解最佳權重"""
    num_assets = len(selected_tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 權重和 = 1
    bounds = tuple((0, 1) for _ in range(num_assets))              # 不允許放空
    initial_weights = num_assets * [1. / num_assets]               # 初始均權
    
    if mode == "MAX_SHARPE":
        result = minimize(negative_sharpe_ratio, initial_weights, 
                          args=(annual_returns, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_volatility, initial_weights, 
                          args=(annual_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_mdd(return_series):
    """計算最大回撤 (Maximum Drawdown)"""
    cum_returns = (1 + return_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

# =========================================================================
# 主程式數據流
# =========================================================================
if len(selected_tickers) < 2:
    st.error("❌ 請至少選擇兩檔以上的標的進行投資組合最佳化計算！")
else:
    with st.spinner("正在從大數據庫下載台股還原收盤價歷史數據..."):
        # 確保基準點也在下載清單中
        download_list = list(set(selected_tickers + [benchmark]))
        raw_data = yf.download(download_list, start=start_date, end=end_date)['Adj Close']
        raw_data = raw_data.dropna()
        
        # 提取觀測標的每日報酬率
        df_returns = raw_data[selected_tickers].pct_change().dropna()
        
        # 計算年化指標 (以一年 252 交易日計)
        annual_returns = df_returns.mean() * 252
        cov_matrix = df_returns.cov() * 252
        
    # 呼叫大腦計算權重
    weights_max_sharpe = optimize_portfolio(annual_returns, cov_matrix, mode="MAX_SHARPE")
    weights_min_vol = optimize_portfolio(annual_returns, cov_matrix, mode="MIN_VOLATILITY")

    # 分流網頁頁籤
    tab1, tab2 = st.tabs(["📊 投資組合大腦回測", "🔔 盤中即時監控與警示大廳"])

    # ---------------------------------------------------------------------
    # TAB 1: 歷史回測與優化
    # ---------------------------------------------------------------------
    with tab1:
        st.subheader("💡 最佳化權重配置建議")
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            st.markdown("**🏆 報酬最高模式 (最大夏普比率配置)**")
            df_ms = pd.DataFrame({'標的': selected_tickers, '建議比例 (%)': np.round(weights_max_sharpe * 100, 2)})
            st.dataframe(df_ms, use_container_width=True, hide_index=True)
            
            # 圓餅圖
            fig_pie1 = go.Figure(data=[go.Pie(labels=df_ms['標的'], values=df_ms['建議比例 (%)'], hole=.3)])
            fig_pie1.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=250)
            st.plotly_chart(fig_pie1, use_container_width=True)

        with col_w2:
            st.markdown("**🛡️ 風險最低模式 (最小變異數配置)**")
            df_mv = pd.DataFrame({'標的': selected_tickers, '建議比例 (%)': np.round(weights_min_vol * 100, 2)})
            st.dataframe(df_mv, use_container_width=True, hide_index=True)
            
            # 圓餅圖
            fig_pie2 = go.Figure(data=[go.Pie(labels=df_mv['標的'], values=df_mv['建議比例 (%)'], hole=.3)])
            fig_pie2.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=250)
            st.plotly_chart(fig_pie2, use_container_width=True)

        st.separator()
        st.subheader("📈 歷史累積報酬率走勢對比 (含股息複利再投入)")

        # 計算各組合歷史表現
        df_returns['Max_Sharpe_Portfolio'] = df_returns[selected_tickers].dot(weights_max_sharpe)
        df_returns['Min_Volatility_Portfolio'] = df_returns[selected_tickers].dot(weights_min_vol)
        
        # 基準點報酬
        benchmark_returns = raw_data[benchmark].pct_change().dropna()
        
        # 合併數據計算累積報酬
        cum_df = pd.DataFrame({
            '報酬最高模式 (Max Sharpe)': (1 + df_returns['Max_Sharpe_Portfolio']).cumprod() - 1,
            '風險最低模式 (Min Volatility)': (1 + df_returns['Min_Volatility_Portfolio']).cumprod() - 1,
            f'純持有基準點 ({benchmark})': (1 + benchmark_returns).cumprod() - 1
        }).dropna() * 100  # 轉成百分比

        # 使用 Plotly 繪製高效互動圖表
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=cum_df.index, y=cum_df['報酬最高模式 (Max Sharpe)'], name='報酬最高模式 (Max Sharpe)', line=dict(color='#FF4B4B', width=2)))
        fig_line.add_trace(go.Scatter(x=cum_df.index, y=cum_df['風險最低模式 (Min Volatility)'], name='風險最低模式 (Min Volatility)', line=dict(color='#0068C9', width=2)))
        fig_line.add_trace(go.Scatter(x=cum_df.index, y=cum_df[f'純持有基準點 ({benchmark})'], name=f'純持有基準點 ({benchmark})', line=dict(color='#29B5E8', width=1.5, dash='dash')))
        
        fig_line.update_layout(title="累積報酬率趨勢 (%)", xaxis_title="日期", yaxis_title="報酬率 (%)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_line, use_container_width=True)

        # 績效量化數據看板
        st.subheader("📊 關鍵量化績效指標摘要")
        metrics_cols = st.columns(3)
        
        strategies = ['報酬最高模式 (Max Sharpe)', '風險最低模式 (Min Volatility)', f'純持有基準點 ({benchmark})']
        source_returns = [df_returns['Max_Sharpe_Portfolio'], df_returns['Min_Volatility_Portfolio'], benchmark_returns]
        
        for idx, col in enumerate(metrics_cols):
            with col:
                ret_series = source_returns[idx]
                total_return = cum_df[strategies[idx]].iloc[-1]
                mdd = calculate_mdd(ret_series) * 100
                ann_ret = (ret_series.mean() * 252) * 100
                ann_vol = (ret_series.std() * np.sqrt(252)) * 100
                sharpe = (ann_ret/100 - risk_free_rate) / (ann_vol/100)
                
                st.metric(label=strategies[idx], value=f"{total_return:.2f}% (總報酬)")
                st.markdown(f"""
                *   **年化報酬率**: {ann_ret:.2f}%
                *   **年化波動度**: {ann_vol:.2f}%
                *   **夏普比率 (Sharpe)**: {sharpe:.2f}
                *   **歷史最大回撤 (MDD)**: <span style='color:#FF4B4B'>{mdd:.2f}%</span>
                """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # TAB 2: 盤中即時監控與警示大廳
    # ---------------------------------------------------------------------
    with tab2:
        st.subheader("🔔 盤中實時監控核心與再平衡雷達")
        st.info("系統會抓取當前最新的市場市價，比對您的實施庫存與大腦目標權重的偏離度。")
        
        # 模擬/獲取目前最新的盤中價格（實務上可對接券商 API，此處利用 yfinance 抓取最新一筆）
        with st.spinner("正在同步盤中即時市價..."):
            latest_prices = {}
            for t in selected_tickers:
                ticker_obj = yf.Ticker(t)
                # 拿最後一筆現價
                latest_prices[t] = ticker_obj.history(period="1d")['Close'].iloc[-1]
        
        # 使用者輸入自訂的當前實際資產張數/比例，用以檢測再平衡
        st.markdown("##### 📥 請輸入您目前的實際庫存部位（或維持預設進行偏離度模擬測試）：")
        user_current_weights = {}
        input_cols = st.columns(len(selected_tickers))
        
        for idx, t in enumerate(selected_tickers):
            with input_cols[idx]:
                # 預設隨機給一些等權重附近的數值做示範
                user_current_weights[t] = st.number_input(f"{t} 目前權重(%)", min_value=0.0, max_value=100.0, value=100.0/len(selected_tickers), step=1.0)
        
        # 正規化使用者權重
        total_user_w = sum(user_current_weights.values())
        if total_user_w > 0:
            for t in user_current_weights:
                user_current_weights[t] = user_current_weights[t] / total_user_w
        
        # 警示計算展示
        st.separator()
        col_alert1, col_alert2 = st.columns(2)
        
        with col_alert1:
            st.markdown("##### 🎯 權重偏離度再平衡監控 (以報酬最高模式為基準)")
            has_rebalance_alert = False
