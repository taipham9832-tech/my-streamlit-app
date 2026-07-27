import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="AI Coffee Decision Engine",
    page_icon="☕",
    layout="wide"
)

# Xử lý đường dẫn tương đối tương thích mọi hệ điều hành (Linux / Windows / macOS)
BASE_DIR = Path(__file__).resolve().parent

def find_data_file(filename):
    """Tìm file trong thư mục gốc hoặc thư mục data/"""
    paths_to_check = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        Path(filename)
    ]
    for path in paths_to_check:
        if path.exists():
            return str(path)
    return None

# ==========================================
# BƯỚC 2 & 3: DATA PIPELINE & FEATURE ENGINEERING
# ==========================================
@st.cache_data(show_spinner=False)
def data_pipeline():
    file_name = "Giá cà phê tổng hợp.xlsx - data.csv"
    price_file = find_data_file(file_name)
    
    if price_file:
        try:
            # Đọc dữ liệu và lọc các dòng chứa ngày tháng
            df_raw = pd.read_csv(price_file, header=None, on_bad_lines='skip', dtype=str)
            df_price = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any(), axis=1)].copy()
            
            # Tìm cột chứa chuỗi Date
            date_col = df_price.apply(lambda col: col.str.contains(r'\d{4}-\d{2}-\d{2}', na=False)).sum().idxmax()
            df_price['Date'] = pd.to_datetime(df_price[date_col], errors='coerce')
            
            # Cột Giá thu mua nằm ngay kế tiếp
            price_col = date_col + 1
            df_price['Price'] = pd.to_numeric(df_price[price_col], errors='coerce')
            
            df = df_price[['Date', 'Price']].dropna().sort_values('Date')
            df = df.groupby('Date')['Price'].mean().reset_index()
        except Exception as e:
            st.warning(f"Lỗi đọc file dữ liệu: {e}. Đang dùng dataset dự phòng.")
            df = generate_fallback_data()
    else:
        df = generate_fallback_data()

    # Xử lý Resample theo ngày & Forward Fill missing values
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill().reset_index()
    
    # Feature Engineering
    df['Lag_1'] = df['Price'].shift(1)
    df['Lag_3'] = df['Price'].shift(3)
    df['Lag_7'] = df['Price'].shift(7)
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7).mean()
    df['Rolling_Mean_14'] = df['Price'].rolling(window=14).mean()
    df['Price_Change'] = df['Price'].pct_change()
    df.dropna(inplace=True)
    
    return df

def generate_fallback_data():
    """Tạo dữ liệu mô phỏng nếu file gốc không tìm thấy trên repository"""
    dates = pd.date_range(start='2020-01-01', end=datetime.today(), freq='D')
    np.random.seed(42)
    prices = 32000 + np.cumsum(np.random.normal(20, 300, len(dates)))
    prices = np.clip(prices, 30000, 130000)
    return pd.DataFrame({'Date': dates, 'Price': prices})

# ==========================================
# BƯỚC 4: MODEL TRAINING (XGBOOST VS PROPHET)
# ==========================================
@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    features = ['Lag_1', 'Lag_3', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_14']
    X_train, y_train = train[features], train['Price']
    X_test, y_test = test[features], test['Price']
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    # Prophet
    prophet_df = train[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test))
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'].iloc[-len(test):].values
    prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_pred))
    
    # Chọn mô hình tốt hơn
    best_model_type = "XGBoost" if xgb_rmse <= prophet_rmse else "Prophet"
    
    if best_model_type == "XGBoost":
        best_model = xgb_model
        best_model.fit(df[features], df['Price'])
    else:
        best_model = Prophet(daily_seasonality=True)
        best_model.fit(df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'}))
        
    return best_model, best_model_type, df

# ==========================================
# BƯỚC 5: DECISION ENGINE
# ==========================================
def decision_engine(df, model, model_type, inputs):
    forecasts = {0: inputs['current_price']}
    
    if model_type == "XGBoost":
        current_features = df.iloc[-1].copy()
        for day in [3, 5, 7, 14, 30]:
            features_input = np.array([[
                current_features['Lag_1'],
                current_features['Lag_3'],
                current_features['Lag_7'],
                current_features['Rolling_Mean_7'],
                current_features['Rolling_Mean_14']
            ]])
            pred = model.predict(features_input)[0]
            
            # Tương quan các biến người dùng nhập
            if inputs['weather'] == 'Thuận lợi': pred *= 0.99
            elif inputs['weather'] == 'Xấu': pred *= 1.025
            if inputs['harvest'] == 'Đỉnh điểm': pred *= 0.985
            
            forecasts[day] = float(pred)
            current_features['Lag_1'] = pred
    else:
        future = model.make_future_dataframe(periods=30)
        fcst = model.predict(future)
        for day in [3, 5, 7, 14, 30]:
            base_pred = fcst['yhat'].iloc[-30 + day]
            if inputs['weather'] == 'Thuận lợi': base_pred *= 0.99
            elif inputs['weather'] == 'Xấu': base_pred *= 1.025
            forecasts[day] = float(base_pred)
            
    days_options = [0, 3, 5, 7, 14, 30]
    vol_options = [0.3, 0.5, 0.7, 1.0]
    
    results = []
    for d in days_options:
        for v in vol_options:
            sell_vol = inputs['inventory'] * v
            holding_cost = inputs['inventory'] * inputs['storage_cost'] * d
            transport = sell_vol * inputs['transport_cost']
            other = sell_vol * inputs['other_cost']
            total_cost = holding_cost + transport + other
            
            revenue = sell_vol * forecasts[d]
            profit = revenue - total_cost
            
            base_risk = 0.05 + (d * 0.01)
            if inputs['weather'] == 'Xấu': base_risk += 0.08
            if inputs['harvest'] == 'Cuối vụ': base_risk -= 0.02
            
            decision_score = profit / (1 + base_risk) if (1 + base_risk) > 0 else 0
            
            action_label = f"BÁN {int(v*100)}% NGAY" if d == 0 else f"BÁN {int(v*100)}% SAU {d} NGÀY"
            
            results.append({
                'Days': d,
                'Volume (%)': int(v*100),
                'Forecast Price': forecasts[d],
                'Expected Revenue': revenue,
                'Total Cost': total_cost,
                'Expected Profit': profit,
                'Risk Score': base_risk,
                'Decision Score': decision_score,
                'Action': action_label
            })
            
    return pd.DataFrame(results), forecasts

# ==========================================
# GIAO DIỆN STREAMLIT
# ==========================================
with st.spinner("Đang tải dữ liệu và khởi tạo AI Model..."):
    df_clean = data_pipeline()
    model, model_type, df_clean = train_and_evaluate_models(df_clean)

st.title("☕ Bảng Điều Khiển Ra Quyết Định Cà Phê AI (Coffee Decision Engine)")
st.caption(f"Hệ thống tự động tích hợp dữ liệu lịch sử và mô hình dự báo **{model_type}**.")

# Sidebar Controls
st.sidebar.header("📥 Dữ Liệu Đầu Vào Hiện Tại")
current_price = st.sidebar.number_input("Giá Cà Phê Hiện Tại (VNĐ/kg)", value=float(df_clean['Price'].iloc[-1]), step=500.0)
inventory = st.sidebar.number_input("Sản Lượng Tồn Kho (Tấn)", value=100.0, step=10.0)
storage_cost = st.sidebar.number_input("Phí Lưu Kho (VNĐ/tấn/ngày)", value=15000.0, step=1000.0)
transport_cost = st.sidebar.number_input("Phí Vận Chuyển (VNĐ/tấn)", value=200000.0, step=10000.0)
other_cost = st.sidebar.number_input("Chi Phí Khác (VNĐ/tấn)", value=50000.0, step=5000.0)
usd_vnd = st.sidebar.number_input("Tỷ Giá USD/VND", value=25400.0, step=100.0)

st.sidebar.header("🌍 Yếu Tố Bối Cảnh")
weather = st.sidebar.selectbox("Điều Kiện Thời Tiết", ["Bình thường", "Thuận lợi", "Xấu"])
harvest = st.sidebar.selectbox("Tình Trạng Thu Hoạch", ["Đầu vụ", "Đỉnh điểm", "Cuối vụ"])
exp_export = st.sidebar.number_input("Dự Kiến Xuất Khẩu (Tấn)", value=150000.0)

inputs = {
    'current_price': current_price, 'inventory': inventory, 'storage_cost': storage_cost,
    'transport_cost': transport_cost, 'other_cost': other_cost, 'usd_vnd': usd_vnd,
    'weather': weather, 'harvest': harvest, 'exp_export': exp_export
}

# Run Engine
df_scenarios, dict_forecasts = decision_engine(df_clean, model, model_type, inputs)
best_scenario = df_scenarios.loc[df_scenarios['Decision Score'].idxmax()]

# KPIs Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá Hiện Tại", f"{current_price:,.0f} VNĐ/kg")
col2.metric("Lợi Nhuận Kỳ Vọng Tối Ưu", f"{best_scenario['Expected Profit']:,.0f} VNĐ", f"Phương án: {best_scenario['Action']}")
col3.metric("Điểm Ra Quyết Định", f"{best_scenario['Decision Score']:,.0f}")
col4.metric("Chỉ Số Rủi Ro", f"{best_scenario['Risk Score']:.2%}")

# AI Recommendation Section
st.subheader("🧠 Khuyến Nghị Từ AI Engine")
with st.container():
    action_color = "#28a745" if best_scenario['Days'] == 0 else "#ffc107"
    st.markdown(f"#### Quyết Định Đề Xuất: <span style='color:{action_color}; font-size:24px; font-weight:bold;'>{best_scenario['Action']}</span>", unsafe_allow_html=True)
    
    insight_text = ""
    if best_scenario['Forecast Price'] > current_price:
        insight_text += f"• **Dự báo giá:** Mô hình {model_type} ước tính giá có thể tăng lên **{best_scenario['Forecast Price']:,.0f} VNĐ/kg** sau {best_scenario['Days']} ngày.\n"
    else:
        insight_text += f"• **Dự báo giá:** Mô hình đánh giá giá thị trường trong ngắn hạn đang xu hướng giảm hoặc đi ngang (Mức giá dự báo: **{best_scenario['Forecast Price']:,.0f} VNĐ/kg**).\n"
        
    if best_scenario['Days'] == 0:
        insight_text += f"• **Tối ưu chi phí:** Do chi phí lưu kho ({storage_cost:,.0f} VNĐ/tấn/ngày) sẽ ăn vào lợi nhuận cùng rủi ro biến động giá, hệ thống khuyến nghị chốt bán ngay.\n"
    else:
        insight_text += f"• **Tối ưu chi phí:** Mức chênh lệch giá dự báo tăng đủ bù đắp tổng chi phí lưu giữ hàng ({best_scenario['Total Cost']:,.0f} VNĐ). Tiếp tục giữ hàng để tối đa hóa lợi nhuận.\n"
        
    if weather == 'Xấu':
        insight_text += "• **Thời tiết:** Yếu tố thời tiết xấu làm tăng chỉ số rủi ro nguồn cung, bạn nên cân nhắc chốt trước một phần sản lượng."

    st.info(insight_text)

# Visualization
st.subheader("📊 Trực Quan Hóa Kịch Bản")
c1, c2 = st.columns(2)

with c1:
    fc_df = pd.DataFrame(list(dict_forecasts.items()), columns=['Ngày Tới', 'Giá Dự Báo (VNĐ/kg)'])
    fig1 = px.line(fc_df, x='Ngày Tới', y='Giá Dự Báo (VNĐ/kg)', markers=True, title="Đường Cong Dự Báo Giá (AI Forecast)")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.bar(
        df_scenarios, x='Action', y='Expected Profit', color='Risk Score',
        title="So Sánh Lợi Nhuận Kỳ Vọng Các Phương Án",
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Data Table
st.subheader("📋 Bảng Ma Trận Ra Quyết Định (Decision Matrix)")
st.dataframe(
    df_scenarios.style.format({
        "Forecast Price": "{:,.0f}",
        "Expected Revenue": "{:,.0f}",
        "Total Cost": "{:,.0f}",
        "Expected Profit": "{:,.0f}",
        "Risk Score": "{:.2%}",
        "Decision Score": "{:,.0f}"
    }),
    use_container_width=True
)
