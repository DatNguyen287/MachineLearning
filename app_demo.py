import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
import sys
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Import tất cả class models từ train.py
from train import *

# Đưa các class vào module __main__ để pickle tìm được
if __name__ == '__main__':
    sys.modules['__main__'].NYAStackingPredictor = NYAStackingPredictor
    sys.modules['__main__'].RidgeRegression = RidgeRegression
    sys.modules['__main__'].RandomForestRegressor = RandomForestRegressor
    sys.modules['__main__'].GradientBoosting = GradientBoosting
    sys.modules['__main__'].KNNRegressor = KNNRegressor
    sys.modules['__main__'].DecisionTreeRegressor = DecisionTreeRegressor
    sys.modules['__main__'].Node = Node

def crawl_nya_data(end_date):
    """Đọc dữ liệu NYA từ file CSV và lọc theo ngày"""
    try:
        df = pd.read_csv("NYA_1_month_clean.csv")
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        df = df[df['Date'] <= pd.Timestamp(end_date)]
        
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {str(e)}")
        st.info("Vui lòng đảm bảo file NYA_1_month_clean.csv tồn tại trong thư mục project")
        return None

@st.cache_resource
def load_predictor():
    model_path = Path(__file__).parent / 'nya_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Model chưa được lưu")
        st.info("Vui lòng chạy Test.ipynb để train và lưu model")
        return None


st.set_page_config(
    page_title="Dự đoán giá cổ phiếu NYA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Dự đoán giá cổ phiếu NYA")
st.write("Stacking Ensemble - 4 thuật toán Machine Learning")

with st.sidebar:
    st.header("Cài đặt")
    st.write("**Mô hình:** Ridge, RF, GB, KNN + Stacking")
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dữ liệu NYA",
        type=['csv'],
        help="File cần có: Date, Open, High, Low, Close, Volume"
    )
    
    if uploaded_file:
        st.success(f"Đã tải: {uploaded_file.name}")

if uploaded_file is not None:
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    try:
        df_recent = pd.read_csv(uploaded_file)
        
        if 'Adj Close' not in df_recent.columns:
            df_recent['Adj Close'] = df_recent['Close']
        
        if 'Date' in df_recent.columns:
            df_recent['Date'] = pd.to_datetime(df_recent['Date'])
            df_recent = df_recent.sort_values('Date').reset_index(drop=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số điểm dữ liệu", len(df_recent))
        with col2:
            st.metric("Từ ngày", df_recent['Date'].min().strftime('%d/%m/%Y'))
        with col3:
            st.metric("Đến ngày", df_recent['Date'].max().strftime('%d/%m/%Y'))
        
        st.divider()
        
        with st.spinner('Đang tính toán các đặc trưng...'):
            df = df_recent.copy()
            
            df['Price_Current'] = df['Adj Close']
            df['Return_1d'] = df['Adj Close'].pct_change()
            df['Return_5d'] = df['Adj Close'].pct_change(5)
            
            for i in [1, 2, 3, 5, 10]:
                df[f'Return_Lag_{i}'] = df['Return_1d'].shift(i)
            
            close_prev = df['Adj Close'].shift(1)
            df['MA_5'] = close_prev.rolling(window=5).mean()
            df['MA_10'] = close_prev.rolling(window=10).mean()
            df['MA_20'] = close_prev.rolling(window=20).mean()
            df['Price_over_MA10'] = close_prev / (df['MA_10'] + 1e-8)
            
            df['Volatility_5'] = df['Return_1d'].rolling(window=5).std()
            df['Volatility_10'] = df['Return_1d'].rolling(window=10).std()
            df['Volatility_20'] = df['Return_1d'].rolling(window=20).std()
            
            df['Volume_MA_5'] = df['Volume'].shift(1).rolling(window=5).mean()
            df['Volume_ratio'] = df['Volume'] / (df['Volume_MA_5'] + 1e-8)
            
            df['Momentum_5'] = df['Adj Close'] - df['Adj Close'].shift(5)
            df['Momentum_10'] = df['Adj Close'] - df['Adj Close'].shift(10)
            df['Momentum_pct_5'] = df['Momentum_5'] / (df['Adj Close'].shift(5) + 1e-8)
            df['Momentum_pct_10'] = df['Momentum_10'] / (df['Adj Close'].shift(10) + 1e-8)
            df['Price_Range'] = (df['High'] - df['Low']) / (df['Low'] + 1e-8)
            
            rolling_high_10 = df['High'].rolling(window=10).max()
            rolling_low_10 = df['Low'].rolling(window=10).min()
            df['HL_position_10'] = (df['Close'] - rolling_low_10) / (rolling_high_10 - rolling_low_10 + 1e-8)
            
            rolling_high_20 = df['High'].rolling(window=20).max()
            rolling_low_20 = df['Low'].rolling(window=20).min()
            df['HL_position_20'] = (df['Close'] - rolling_low_20) / (rolling_high_20 - rolling_low_20 + 1e-8)
            
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            df = df.dropna()
        
        if len(df) == 0:
            st.error("Không đủ dữ liệu! Cần ít nhất 20-30 ngày dữ liệu lịch sử.")
        else:
            exclude_cols = ['Date', 'Index', 'Close', 'Adj Close', 'Open', 'High', 'Low', 
                            'Volume', 'CloseUSD', 'Target_Return', 'Price_Current', 'Price_Next']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            today_features = df[feature_cols].iloc[-1:].values
            today_price = df['Price_Current'].iloc[-1]
            today_date = df['Date'].iloc[-1] if 'Date' in df.columns else 'Latest'
            
            with st.spinner('Đang dự đoán...'):
                predictions = {}
                base_names = ['Ridge', 'RF', 'GradientBoosting', 'KNN']
                
                for name in base_names:
                    if name in predictor.models:
                        pred_return = predictor.models[name].predict(today_features)[0]
                        predictions[name] = pred_return
                
                test_preds = np.array([[predictions[name] for name in base_names]])
                stacking_return = predictor.meta_model.predict(test_preds)[0]
                predictions['Stacking'] = stacking_return
            
            st.markdown("## Kết quả dự đoán")
            
            pred_price = today_price * (1 + stacking_return)
            direction = "TĂNG" if stacking_return > 0 else "GIẢM"
            
            next_date = today_date + timedelta(days=1)
            while next_date.weekday() >= 5:  
                next_date = next_date + timedelta(days=1)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Ngày dữ liệu",
                    value=today_date.strftime('%d/%m/%Y') if isinstance(today_date, pd.Timestamp) else str(today_date)
                )
            
            with col2:
                st.metric(
                    label="Giá hiện tại",
                    value=f"${today_price:,.2f}"
                )
            
            with col3:
                st.metric(
                    label="Dự đoán ngày kế tiếp",
                    value=next_date.strftime('%d/%m/%Y') if isinstance(next_date, pd.Timestamp) else str(next_date),
                    delta=direction
                )
            
            with col4:
                st.metric(
                    label="Giá dự đoán",
                    value=f"${pred_price:,.2f}",
                    delta=f"{stacking_return*100:+.3f}%"
                )
            
            st.divider()
            
            st.markdown("### Kết quả từng mô hình")
            
            cols = st.columns(len(base_names))
            for idx, name in enumerate(base_names):
                with cols[idx]:
                    ret = predictions[name]
                    price = today_price * (1 + ret)
                    st.metric(
                        label=name,
                        value=f"{ret*100:+.3f}%",
                        delta=f"${price:,.2f}"
                    )
            
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Độ tin cậy")
                up_votes = sum(1 for p in predictions.values() if p > 0)
                down_votes = len(predictions) - up_votes
                agreement = max(up_votes, down_votes) / len(predictions) * 100
                
                if agreement >= 80:
                    confidence = "Cao"
                elif agreement >= 60:
                    confidence = "Trung bình"
                else:
                    confidence = "Thấp"
                
                st.metric("Mức độ đồng thuận", f"{agreement:.1f}%")
                st.markdown(f"**{up_votes}** mô hình dự đoán tăng, **{down_votes}** dự đoán giảm")
                st.markdown(f"**Độ tin cậy:** {confidence}")
            
            with col2:
                st.markdown("### Đánh giá rủi ro")
                std_pred = np.std(list(predictions.values()))
                st.metric("Độ phân tán", f"±{std_pred*100:.3f}%")
                
                if std_pred < 0.002:
                    risk_msg = "**Độ phân tán thấp** - Các mô hình đồng thuận cao"
                elif std_pred < 0.005:
                    risk_msg = "**Độ phân tán trung bình** - Có sự khác biệt giữa các mô hình"
                else:
                    risk_msg = "**Độ phân tán cao** - Các mô hình có ý kiến khác nhau đáng kể"
                
                st.markdown(risk_msg)
            
            st.divider()
            st.markdown("### So sánh kết quả các mô hình")
            
            chart_data = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Tỷ suất sinh lời (%)': [v * 100 for v in predictions.values()]
            })
            
            st.bar_chart(chart_data.set_index('Model'))
            
            with st.expander("Xem chi tiết bảng dự đoán"):
                detail_df = pd.DataFrame({
                    'Mô hình': list(predictions.keys()),
                    'Tỷ suất (%)': [f"{v*100:+.3f}" for v in predictions.values()],
                    'Giá dự đoán ($)': [f"{today_price * (1 + v):,.2f}" for v in predictions.values()],
                    'Xu hướng': ['Tăng' if v > 0 else 'Giảm' for v in predictions.values()]
                })
                st.dataframe(detail_df, use_container_width=True)
            
            st.divider()
            result_dict = {
                'Date': today_date.strftime('%Y-%m-%d') if isinstance(today_date, pd.Timestamp) else str(today_date),
                'Current_Price': today_price,
                'Predicted_Return': stacking_return,
                'Predicted_Price': pred_price,
                'Direction': direction,
                'Confidence': agreement,
                'Variance': std_pred
            }
            
            st.download_button(
                label="Tải xuống kết quả (JSON)",
                data=pd.DataFrame([result_dict]).to_json(orient='records', indent=2),
                file_name=f"nya_prediction_{today_date.strftime('%Y%m%d') if isinstance(today_date, pd.Timestamp) else 'latest'}.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
        st.exception(e)

else:
    st.info("Tải file CSV lên để bắt đầu dự đoán")
    
    st.markdown("""
    ### Hướng dẫn:
    1. **Chuẩn bị file CSV** với các cột: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
    2. **Cần ít nhất 20-30 ngày** dữ liệu để tính features
    3. **Upload file** → App tự động dự đoán **ngày tiếp theo** sau ngày cuối cùng trong file
    
    ### Ví dụ:
    - File có dữ liệu đến **04/01/2026**
    - App sẽ dự đoán giá ngày **05/01/2026** (hôm sau)
    
    ### Cập nhật dữ liệu mới:
    - Chạy notebook **Craw_1_Month.ipynb** để tải dữ liệu mới nhất
    - Hoặc thêm dòng mới vào file CSV có sẵn
    """)
    
    st.divider()
    st.caption("File mẫu: NYA_1_month_clean.csv")

st.divider()
st.caption("Đồ án Học Máy - Chỉ phục vụ mục đích học tập")
