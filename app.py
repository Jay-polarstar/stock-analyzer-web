import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json
from scipy.optimize import minimize
import plotly.express as px

# 1. 網頁基本配置
st.set_page_config(page_title="美股投資組合優化大腦", page_icon="🧠", layout="wide")

st.title("🧠 🇺🇸 美股資產配置最佳化優化大腦")
st.caption("第二階段開發：基於馬可維茨均值-變異數理論 (Mean-Variance Optimization)")

# 2. 側邊欄：讓使用者調整觀察清單與參數
st.sidebar.header("🛠️ 配置面板")

# 預設經典清單
default_tickers = ['SPY', 'QQQ', 'VTI', 'GLD', 'SMH', 'VOO', 'NVDA']
ticker_input = st.sidebar.text_input(
    "修改美股/ETF觀察清單 (用半形逗號隔開):", 
    value=", ".join(default_tickers)
)

# 無風險利率設定 (用來計算夏普值，預設為 2%)
rf_rate = st.sidebar.slider("無風險利率 (Risk-Free Rate %)", min_value=0.0, max_value=8.0, value=2.0, step=0.1) / 100

# 解析輸入的標的
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

# 3. 核心數據抓取函式 (防 commercial/阻斷崩潰機制)
@st.cache_data(ttl=3600)  # 快取資料 1 小時，避免頻繁請求被 Yahoo 封鎖
def fetch_stock_data(ticker_list):
    combined_df = pd.DataFrame()
    success_tickers = []
    
    for ticker in ticker_list:
        try:
            # 使用免金鑰 Web API 獲取 3 年歷史日線數據
            url = f"https://yahoo.com{ticker}?range=3y&interval=1d"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            
            with urllib.request.urlopen(req) as response:
                raw_data = json.loads(response.read().decode())
                
            result = raw_data['chart']['result'][0]
            timestamps = result['timestamp']
            closes = result['indicators']['quote'][0]['close']
            
            df = pd.DataFrame({'Close': closes}, index=pd.to_datetime(timestamps, unit='s'))
            df = df.dropna()
            
            if not df.empty:
                # 只保留收盤價，並以標的名稱命名欄位
                combined_df[ticker] = df['Close']
                success_tickers.append(ticker)
        except Exception:
            # 個別標的失敗不中斷整體程式，網頁會跳出警告提示
            pass
            
    return combined_df, success_tickers

# 4. 優化算法核心數學邏輯 (Scipy 實作)
def portfolio_performance(weights, returns, cov_matrix):
    # 計算投資組合的預期年化報酬率與年化波動度
    port_return = np.sum(returns * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return port_return, port_volatility

def min_variance_objective(weights, returns, cov_matrix):
    # 最小化波動度的目標函數：回傳組合波動度
    _, port_volatility = portfolio_performance(weights, returns, cov_matrix)
    return port_volatility

def max_sharpe_objective(weights, returns, cov_matrix, rf):
    # 最大化夏普值的目標函數：因為 minimize 是找最小值，所以將夏普值乘上負號
    port_return, port_volatility = portfolio_performance(weights, returns, cov_matrix)
    sharpe = (port_return - rf) / port_volatility if port_volatility != 0 else 0
    return -sharpe

# 5. 主程式觸發邏輯
if not tickers:
    st.warning("⚠️ 請至少輸入一個有效的美股代碼！")
else:
    with st.spinner("🔍 正在從國際金融伺服器獲取即時數據並建立矩陣..."):
        data, valid_tickers = fetch_stock_data(tickers)
        
    # 檢查是否有抓到任何數據
    if data.empty:
        st.error("❌ 無法獲取任何輸入標的的歷史數據，請確認代碼是否輸入正確。")
    else:
        # 如果有部分標的失敗，顯示黃色警告提示
        failed_tickers = set(tickers) - set(valid_tickers)
        if failed_tickers:
            st.warning(f"⚠️ 提示：標的 {list(failed_tickers)} 數據獲取失敗，已自動自優化矩陣中剔除。")
            
        # 計算每日報酬率
        returns_df = data.pct_change().dropna()
        
        # 計算平均每日報酬與共變異數矩陣
        avg_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        num_assets = len(valid_tickers)
        
        # 設定邊界條件：每檔股票權重必須在 0% 到 100% 之間 (不開槓桿、不放空)
        bounds = tuple((0, 1) for _ in range(num_assets))
        # 設定約束條件：所有權重相加必須剛好等於 1 (100% 滿倉)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        # 初始權重平均分配
        init_weights = num_assets * [1.0 / num_assets]
        
        # --- 方案 A：風險最低優化 ---
        res_min_var = minimize(
            min_variance_objective, init_weights, 
            args=(avg_returns, cov_matrix), 
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        weights_min_var = res_min_var.x
        ret_min, vol_min = portfolio_performance(weights_min_var, avg_returns, cov_matrix)
        sharpe_min = (ret_min - rf_rate) / vol_min
        
        # --- 方案 B：夏普值最高優化 ---
        res_max_sharpe = minimize(
            max_sharpe_objective, init_weights, 
            args=(avg_returns, cov_matrix, rf_rate), 
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        weights_max_sharpe = res_max_sharpe.x
        ret_max, vol_max = portfolio_performance(weights_max_sharpe, avg_returns, cov_matrix)
        sharpe_max = (ret_max - rf_rate) / vol_max
        
        # --- 等權重組合 (作為優化前後的對照組) ---
        ret_equal, vol_equal = portfolio_performance(np.array(init_weights), avg_returns, cov_matrix)
        sharpe_equal = (ret_equal - rf_rate) / vol_equal
        
        # 6. 網頁前端結果呈現
        st.success("✅ 量化優化矩陣計算完成！")
        
        # 分流切換面板
        tab1, tab2 = st.tabs(["📊 方案 A：風險最低組合 (Minimum Variance)", "🚀 方案 B：報酬最高/夏普最大組合 (Max Sharpe Ratio)"])
        
        with tab1:
            st.subheader("🛡️ 總體風險（波動度）最小化的最佳配置")
            
            # 數據儀表板
            m1, m2, m3 = st.columns(3)
            m1.metric("預期年化報酬率", f"{ret_min*100:.2f}%", f"{(ret_min-ret_equal)*100:+.2f}% 較等權重")
            m2.metric("預期年化波動度 (風險)", f"{vol_min*100:.2f}%", f"{(vol_min-vol_equal)*100:+.2f}% 較等權重", delta_color="inverse")
            m3.metric("夏普值 (Sharpe Ratio)", f"{sharpe_min:.2f}", f"{sharpe_min-sharpe_equal:+.2f} 較等權重")
            
            # 繪製圓餅圖
            df_pie_min = pd.DataFrame({'資產': valid_tickers, '權重(%)': weights_min_var * 100})
            df_pie_min = df_pie_min[df_pie_min['權重(%)'] > 0.01] # 隱藏過小的權重
            fig_min = px.pie(df_pie_min, values='權重(%)', names='資產', title="風險最低資產配比圓餅圖", hole=0.4, color_discrete_sequence=px.colors.sequential.YlGnBu)
            st.plotly_chart(fig_min, use_container_width=True)
            
            # 顯示表格資料
            st.dataframe(df_pie_min.set_index('資產').style.format("{:.2f}%"))
            
        with tab2:
            st.subheader("🔥 同等風險下，高風報比(夏普值)最大化的最佳配置")
            
            # 數據儀表板
            k1, k2, k3 = st.columns(3)
            k1.metric("預期年化報酬率", f"{ret_max*100:.2f}%", f"{(ret_max-ret_equal)*100:+.2f}% 較等權重")
            k2.metric("預期年化波動度 (風險)", f"{vol_max*100:.2f}%", f"{(vol_max-vol_equal)*100:+.2f}% 較等權重", delta_color="inverse")
            k3.metric("夏普值 (Sharpe Ratio)", f"{sharpe_max:.2f}", f"{sharpe_max-sharpe_equal:+.2f} 較等權重")
            
            # 繪製圓餅圖
            df_pie_max = pd.DataFrame({'資產': valid_tickers, '權重(%)': weights_max_sharpe * 100})
            df_pie_max = df_pie_max[df_pie_max['權重(%)'] > 0.01]
            fig_max = px.pie(df_pie_max, values='權重(%)', names='資產', title="夏普值最高資產配比圓餅圖", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_max, use_container_width=True)
            
            # 顯示表格資料
            st.dataframe(df_pie_max.set_index('資產').style.format("{:.2f}%"))

        # 基礎個股數據參考
        with st.expander("ℹ️ 觀看各別個股 3 年日線歷史收盤價走勢"):
            st.line_chart(data)
