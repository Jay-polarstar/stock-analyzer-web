import streamlit as st
import pandas as pd
import numpy as np
from twelvedata import TDClient
from scipy.optimize import minimize
import plotly.express as px

# 1. 網頁基本配置
st.set_page_config(page_title="美股投資組合優化大腦 v3", page_icon="🧠", layout="wide")
st.title("🧠 🇺🇸 美股資產配置最佳化優化大腦 (自動金鑰版)")
st.caption("第二階段修復：金鑰已由內建程式碼自動帶入，免手動輸入即可執行量化分析")

# =========================================================================
# 2. 自動金鑰與側邊欄設定 (請在下方引號內填入您在 twelvedata.com 申請的金鑰)
# =========================================================================
API_KEY = "您的_TWELVE_DATA_API_KEY_貼在這裡" 

st.sidebar.header("🛠️ 配置面板")
ticker_input = st.sidebar.text_input("修改美股/ETF觀察清單:", value="SPY, QQQ, VTI, GLD, SMH, NVDA")
rf_rate = st.sidebar.slider("無風險利率 (%)", 0.0, 8.0, 2.0) / 100
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

# 3. 呼叫官方套件抓取數據
@st.cache_data(ttl=3600)
def fetch_stock_data_official(ticker_list, key):
    combined_df = pd.DataFrame()
    success_tickers = []
    
    # 防錯機制：若金鑰欄位忘記修改或留空
    if not key or "貼在這裡" in key:
        st.error("🔑 程式碼內的 API_KEY 尚未填入正確的金鑰！請先修改您的 GitHub app.py 檔案。")
        return combined_df, success_tickers
        
    try:
        td = TDClient(apikey=key)
        for ticker in ticker_list:
            try:
                # 呼叫官方 SDK API 獲取 3 年歷史日線數據
                ts = td.time_series(symbol=ticker, interval="1day", outputsize=750, order="asc")
                df = ts.as_pandas() # 官方快速轉換
                if not df.empty:
                    combined_df[ticker] = pd.to_numeric(df['close'])
                    success_tickers.append(ticker)
            except: 
                pass
    except: 
        pass
    return combined_df, success_tickers

# 4. 優化算法核心數學邏輯 (Scipy)
def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.sum(returns * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return port_return, port_volatility

def min_variance_objective(weights, returns, cov_matrix):
    _, port_volatility = portfolio_performance(weights, returns, cov_matrix)
    return port_volatility

def max_sharpe_objective(weights, returns, cov_matrix, rf):
    port_return, port_volatility = portfolio_performance(weights, returns, cov_matrix)
    sharpe = (port_return - rf) / port_volatility if port_volatility != 0 else 0
    return -sharpe

# 5. 主程式觸發邏輯
if not tickers:
    st.warning("⚠️ 請至少輸入一個有效的美股代碼！")
else:
    with st.spinner("🔍 正在自動連接金融伺服器獲取即時數據並建立矩陣..."):
        # 這裡的第二個參數改為直接傳入我們在上方定義好的私密變數 API_KEY
        data, valid_tickers = fetch_stock_data_official(tickers, API_KEY)
        
    if data.empty:
        # 如果使用者完全沒在 14 行修改金鑰，上面第 27 行會攔截，若走到這代表可能是頻率限制
        if not API_KEY or "貼在這裡" in API_KEY:
            pass
        else:
            st.error("❌ 無法獲取歷史數據。可能原因：您填寫的金鑰無效、超過免費版「每分鐘 8 次」的訪問次數限制，或美股代碼不正確。")
    else:
        failed_tickers = set(tickers) - set(valid_tickers)
        if failed_tickers:
            st.warning(f"⚠️ 提示：標的 {list(failed_tickers)} 數據獲取失敗，已自動自優化矩陣中剔除。")
            
        returns_df = data.pct_change().dropna()
        avg_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        num_assets = len(valid_tickers)
        
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        init_weights = num_assets * [1.0 / num_assets]
        
        # --- 方案 A：風險最低優化 ---
        res_min_var = minimize(min_variance_objective, init_weights, args=(avg_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
        weights_min_var = res_min_var.x
        ret_min, vol_min = portfolio_performance(weights_min_var, avg_returns, cov_matrix)
        sharpe_min = (ret_min - rf_rate) / vol_min
        
        # --- 方案 B：夏普值最高優化 ---
        res_max_sharpe = minimize(max_sharpe_objective, init_weights, args=(avg_returns, cov_matrix, rf_rate), method='SLSQP', bounds=bounds, constraints=constraints)
        weights_max_sharpe = res_max_sharpe.x
        ret_max, vol_max = portfolio_performance(weights_max_sharpe, avg_returns, cov_matrix)
        sharpe_max = (ret_max - rf_rate) / vol_max
        
        # --- 等權重組合對照組 ---
        ret_equal, vol_equal = portfolio_performance(np.array(init_weights), avg_returns, cov_matrix)
        sharpe_equal = (ret_equal - rf_rate) / vol_equal
        
        st.success("✅ 量化優化矩陣計算完成！")
        tab1, tab2 = st.tabs(["📊 方案 A：風險最低組合", "🚀 方案 B：報酬最高/夏普最大組合"])
        
        with tab1:
            st.subheader("🛡️ 總體風險（波動度）最小化的最佳配置")
            m1, m2, m3 = st.columns(3)
            m1.metric("預期年化報酬率", f"{ret_min*100:.2f}%", f"{(ret_min-ret_equal)*100:+.2f}% 較等權重")
            m2.metric("預期年化波動度 (風險)", f"{vol_min*100:.2f}%", f"{(vol_min-vol_equal)*100:+.2f}% 較等權重", delta_color="inverse")
            m3.metric("夏普值 (Sharpe Ratio)", f"{sharpe_min:.2f}", f"{sharpe_min-sharpe_equal:+.2f} 較等權重")
            
            df_pie_min = pd.DataFrame({'資產': valid_tickers, '權重(%)': weights_min_var * 100})
            df_pie_min = df_pie_min[df_pie_min['權重(%)'] > 0.01]
            fig_min = px.pie(df_pie_min, values='權重(%)', names='資產', title="風險最低資產配比圓餅圖", hole=0.4, color_discrete_sequence=px.colors.sequential.YlGnBu)
            st.plotly_chart(fig_min, use_container_width=True)
            st.dataframe(df_pie_min.set_index('資產').style.format("{:.2f}%"))
            
        with tab2:
            st.subheader("🔥 報酬與風險權衡（夏普值）最大化的最佳配置")
            k1, k2, k3 = st.columns(3)
            k1.metric("預期年化報酬率", f"{ret_max*100:.2f}%", f"{(ret_max-ret_equal)*100:+.2f}% 較等權重")
            k2.metric("預期年化波動度 (風險)", f"{vol_max*100:.2f}%", f"{(vol_max-vol_equal)*100:+.2f}% 較等權重", delta_color="inverse")
            k3.metric("夏普值 (Sharpe Ratio)", f"{sharpe_max:.2f}", f"{sharpe_max-sharpe_equal:+.2f} 較等權重")
            
            df_pie_max = pd.DataFrame({'資產': valid_tickers, '權重(%)': weights_max_sharpe * 100})
            df_pie_max = df_pie_max[df_pie_max['權重(%)'] > 0.01]
            fig_max = px.pie(df_pie_max, values='權重(%)', names='資產', title="夏普值最高資產配比圓餅圖", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_max, use_container_width=True)
            st.dataframe(df_pie_max.set_index('資產').style.format("{:.2f}%"))

        with st.expander("ℹ️ 觀看各別個股歷史收盤價走勢"):
            st.line_chart(data)
