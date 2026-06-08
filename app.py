python
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 設定網頁標題與外觀
st.set_page_config(
    page_title="美股量化分析與風險評估工具",
    page_icon="📈",
    layout="wide"
)

# 標題與說明
st.title("📈 美股歷史數據與風險指標分析工具")
st.markdown("""
這個網頁版工具能協助您快速分析任何美股標的過去數年的表現。
系統將自動計算**年化報酬率 (CAGR)**、**年化波動度**以及**歷史最大回撤 (Max Drawdown)**，並提供互動式圖表。
""")

# 2. 側邊欄輸入參數
st.sidebar.header("⚙️ 參數設定")
ticker = st.sidebar.text_input("輸入美股代碼 (例如: AAPL, TSLA, VOO, QQQ)", value="AAPL").upper().strip()
years = st.sidebar.slider("分析歷史年期", min_value=1, max_value=10, value=5, step=1)

# 開始計算按鈕
start_analysis = st.sidebar.button("🚀 開始分析", type="primary")

# 3. 核心分析邏輯
def analyze_stock_data(ticker_symbol, years_count):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years_count * 365)
    
    # 抓取數據
    try:
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"❌ 無法取得 {ticker_symbol} 的數據，請檢查代碼是否正確。")
            return None
    except Exception as e:
        st.error(f"❌ 擷取數據時發生錯誤: {e}")
        return None
        
    # 處理 yfinance 可能返回的 MultiIndex
    df = pd.DataFrame()
    if isinstance(stock_data.columns, pd.MultiIndex):
        df['Close'] = stock_data['Close'][ticker_symbol]
    else:
        df['Close'] = stock_data['Close']
        
    df = df.dropna()
    
    # 計算每日報酬率
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 計算量化指標
    trading_days_per_year = 252
    total_trading_days = len(df)
    actual_years = total_trading_days / trading_days_per_year
    
    # A. 年化報酬率 (CAGR)
    initial_val = float(df['Close'].iloc[0])
    final_val = float(df['Close'].iloc[-1])
    cagr = (final_val / initial_val) ** (1 / actual_years) - 1
    
    # B. 年化波動度
    daily_vol = df['Daily_Return'].std()
    annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
    
    # C. 歷史最大回撤
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
    max_drawdown = df['Drawdown'].min()
    max_dd_date = df['Drawdown'].idxmin()
    
    return {
        "df": df,
        "cagr": cagr,
        "volatility": annualized_vol,
        "max_dd": max_drawdown,
        "max_dd_date": max_dd_date,
        "actual_years": actual_years,
        "start_date": df.index[0],
        "end_date": df.index[-1]
    }

# 4. 網頁主要內容顯示
if start_analysis or ticker:
    with st.spinner("正在下載並分析歷史數據，請稍候..."):
        results = analyze_stock_data(ticker, years)
        
        if results is not None:
            df = results["df"]
            
            # A. 頂部數據看板 (Metric Cards)
            st.subheader(f"📊 {ticker} 分析報告 ({results['start_date'].strftime('%Y-%m-%d')} ~ {results['end_date'].strftime('%Y-%m-%d')})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="📈 年化報酬率 (CAGR)", 
                    value=f"{results['cagr'] * 100:.2f}%",
                    help="複合年均增長率，代表平均每年的投資報酬率。"
                )
            with col2:
                st.metric(
                    label="📉 年化波動度 (Volatility)", 
                    value=f"{results['volatility'] * 100:.2f}%",
                    help="衡量股價波動劇烈程度的指標，波動度越高代表風險越大。"
                )
            with col3:
                st.metric(
                    label="🚨 歷史最大回撤 (Max Drawdown)", 
                    value=f"{results['max_dd'] * 100:.2f}%",
                    delta=f"發生於 {results['max_dd_date'].strftime('%Y-%m-%d')}",
                    delta_color="inverse",
                    help="在此期間內，資產從最高點跌落至最低點的最大幅度。"
                )
            
            # B. 繪製互動式 Plotly 圖表
            st.write("---")
            st.subheader("📈 互動式走勢與回撤分析圖")
            
            # 建立雙子圖 (共享 X 軸)
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.08,
                row_heights=[0.6, 0.4]
            )
            
            # 上圖：收盤價
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Close'], 
                    name="收盤價 (USD)", 
                    line=dict(color="#1f77b4", width=2)
                ),
                row=1, col=1
            )
            
            # 下圖：回撤區間
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df['Drawdown'] * 100, 
                    name="回撤 (%)", 
                    line=dict(color="#d62728", width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(214, 39, 40, 0.2)'
                ),
                row=2, col=1
            )
            
            # 標記最大回撤點
            fig.add_trace(
                go.Scatter(
                    x=[results['max_dd_date']],
                    y=[results['max_dd'] * 100],
                    mode="markers+text",
                    name="最大回撤點",
                    marker=dict(color="black", size=10, symbol="triangle-down"),
                    text=[f"Max DD: {results['max_dd']*100:.2f}%"],
                    textposition="bottom center",
                    textfont=dict(color="black", size=12, family="Arial")
                ),
                row=2, col=1
            )
            
            # 圖表版面配置優化
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_yaxes(title_text="股價 (USD)", row=1, col=1)
            fig.update_yaxes(title_text="回撤幅度 (%)", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # C. 數據預覽
            st.write("---")
            st.subheader("📂 歷史數據預覽 (最近 10 筆交易日)")
            st.dataframe(df[['Close', 'Daily_Return']].tail(10).style.format({
                'Close': '${:.2f}',
                'Daily_Return': '{:.2%}'
            }), use_container_width=True)
