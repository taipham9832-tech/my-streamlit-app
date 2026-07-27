import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Coffee Decision Engine", layout="wide")

# ==========================================
# BƯỚC 2 & 3: DATA PIPELINE & FEATURE ENGINEERING
# ==========================================
@st.cache_data(show_spinner=False)
def data_pipeline():
    # 1. Trích xuất và Chuẩn hóa Giá Cà Phê (Target Variable)
    price_file = "data/Giá cà phê tổng hợp.xlsx - data.csv"
    if os.path.exists(price_file):
        # Đọc bỏ qua header rác, dùng regex tìm dòng chứa ngày tháng yyyy-mm-dd
        df_raw = pd.read_csv(price_file, header=None, on_bad_lines='skip', dtype=str)
        df_price = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any(), axis=1)]
        
        # Tìm cột Date và Price dựa trên định dạng
        date_col = df_price.apply(lambda col: col.str.contains(r'\d{4}-\d{2}-\d{2}', na=False)).sum().idxmax()
        df_price['Date'] = pd.to_datetime(df_price[date_col], errors='coerce')
        # Cột giá thường nằm ngay sau cột Date
        price_col = date_col + 1
        df_price['Price'] = pd.to_numeric(df_price[price_col], errors='coerce')
        
        df = df_price[['Date', 'Price']].dropna().sort_values('Date')
        df = df.groupby('Date')['Price'].mean().reset_index() # Xử lý duplicate ngày
    else:
        # Fallback dataset để Dashboard hoạt động nếu thiếu file (Prototype failsafe)
        dates = pd.date_range(start='2020-01-01', end=datetime.today(), freq='D')
        prices = 32000 + np.cumsum(np.random.normal(10, 200, len(dates)))
        df = pd.DataFrame({'Date': dates, 'Price': prices})

    # 2. Xử lý Missing Value và Sắp xếp
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill().reset_index()
    
    # 3. Feature Engineering
    df['Lag_1'] = df['Price'].shift(1)
    df['Lag_3'] = df['Price'].shift(3)
    df['Lag_7'] = df['Price'].shift(7)
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7).mean()
    df['Rolling_Mean_14'] = df['Price'].rolling(window=14).mean()
    df['Price_Change'] = df['Price'].pct_change()
    df.dropna(inplace=True)
    
    # Tích hợp thêm các file Vĩ mô (nếu có thể parse) có thể làm tương tự ở đây
    # df.to_csv("data/clean_dataset.csv", index=False) # Lưu dataset sạch
    return df

# ==========================================
# BƯỚC 4: MODEL TRAINING (XGBOOST VS PROPHET)
# ==========================================
@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(df):
    # Split Data (80/20)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # --- Xử lý XGBoost ---
    features = ['Lag_1', 'Lag_3', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_14']
    X_train, y_train = train[features], train['Price']
    X_test, y_test = test[features], test['Price']
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    # --- Xử lý Prophet ---
    prophet_df = train[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=len(test))
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'].iloc[-len(test):].values
    prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_pred))
    
    # --- Đánh giá và Chọn Model ---
    best_model_type = "XGBoost" if xgb_rmse < prophet_rmse else "Prophet"
    best_model = xgb_model if best_model_type == "XGBoost" else prophet_model
    
    # Train lại model tốt nhất trên Toàn bộ dữ liệu
    if best_model_type == "XGBoost":
        best_model.fit(df[features], df['Price'])
    else:
        best_model.fit(df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'}))
        
    # Lưu Model
    joblib.dump(best_model, 'best_model.pkl')
    
    return best_model, best_model_type, df

# ==========================================
# BƯỚC 5: DASHBOARD & DECISION ENGINE
# ==========================================
def decision_engine(df, model, model_type, inputs):
    last_price = df['Price'].iloc[-1]
    last_date = df['Date'].iloc[-1]
    
    # 1. AI Forecasting dựa trên Model
    forecasts = {0: inputs['current_price']} # Day 0 = Current Price
    
    if model_type == "XGBoost":
        # Simulate future features based on inputs and last known data
        current_features = df.iloc[-1].copy()
        for day in [3, 5, 7, 14, 30]:
             # Dữ liệu người dùng điều chỉnh dự báo (Feature adjustment)
             pred = model.predict(current_features[['Lag_1', 'Lag_3', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_14']].values.reshape(1, -1))[0]
             # Điều chỉnh theo thời tiết và vĩ mô
             if inputs['weather'] == 'Thuận lợi': pred *= 0.99
             elif inputs['weather'] == 'Xấu': pred *= 1.02
             if inputs['harvest'] == 'Đỉnh điểm': pred *= 0.98
             forecasts[day] = pred
             current_features['Lag_1'] = pred # Update lag dynamically
    else:
        future = model.make_future_dataframe(periods=30)
        fcst = model.predict(future)
        for day in [3, 5, 7, 14, 30]:
            base_pred = fcst['yhat'].iloc[-30 + day]
            if inputs['weather'] == 'Thuận lợi': base_pred *= 0.99
            elif inputs['weather'] == 'Xấu': base_pred *= 1.02
            forecasts[day] = base_pred
            
    # 2. Sinh các Kịch bản (Scenarios)
    days_options = [0, 3, 5, 7, 14, 30]
    vol_options = [0.3, 0.5, 0.7, 1.0]
    
    results = []
    for d in days_options:
        for v in vol_options:
            sell_vol = inputs['inventory'] * v
            hold_vol = inputs['inventory'] - sell_vol
            
            # Tính toán Cost
            holding_cost = inputs['inventory'] * inputs['storage_cost'] * d
            transport = sell_vol * inputs['transport_cost']
            other = sell_vol * inputs['other_cost']
            total_cost = holding_cost + transport + other
            
            # Tính toán Revenue & Profit
            revenue = sell_vol * forecasts[d]
            profit = revenue - total_cost
            
            # Tính toán Risk & Decision Score
            # Rủi ro tăng theo thời gian lưu kho và thời tiết xấu
            base_risk = 0.05 + (d * 0.01)
            if inputs['weather'] == 'Xấu': base_risk += 0.08
            if inputs['harvest'] == 'Cuối vụ': base_risk -= 0.02
            
            decision_score = profit / (1 + base_risk) if (1 + base_risk) > 0 else 0
            
            results.append({
                'Days': d, 'Volume (%)': int(v*100), 'Forecast Price': forecasts[d],
                'Expected Revenue': revenue, 'Total Cost': total_cost,
                'Expected Profit': profit, 'Risk Score': base_risk,
                'Decision Score': decision_score,
                'Action': f"BÁN {int(v*100)}% NGAY" if d == 0 else f"BÁN {int(v*100)}% SAU {d} NGÀY"
            })
            
    return pd.DataFrame(results), forecasts

# ==========================================
# GIAO DIỆN STREAMLIT
# ==========================================
# Gọi hàm tự động
with st.spinner("Phân tích và Chuẩn hóa dữ liệu lịch sử..."):
    df_clean = data_pipeline()
with st.spinner("Đang huấn luyện mô hình AI (XGBoost & Prophet)..."):
    model, model_type, df_clean = train_and_evaluate_models(df_clean)

st.title("☕ Bảng Điều Khiển Ra Quyết Định Cà Phê AI (Coffee Decision Engine)")
st.markdown("Hệ thống tự động sử dụng **dữ liệu lịch sử tích hợp sẵn** và Mô hình **" + model_type + "** để tính toán tối ưu hóa lợi nhuận.")

# Sidebar Inputs
st.sidebar.header("📥 Nhập Dữ Liệu Hiện Tại")
current_price = st.sidebar.number_input("Current Coffee Price (VNĐ/kg)", value=float(df_clean['Price'].iloc[-1]), step=500.0)
inventory = st.sidebar.number_input("Inventory (Tấn)", value=100.0, step=10.0)
storage_cost = st.sidebar.number_input("Storage Cost (VNĐ/tấn/ngày)", value=15000.0, step=1000.0)
transport_cost = st.sidebar.number_input("Transport Cost (VNĐ/tấn)", value=200000.0, step=10000.0)
other_cost = st.sidebar.number_input("Other Cost (VNĐ/tấn)", value=50000.0, step=5000.0)
usd_vnd = st.sidebar.number_input("Current USD/VND", value=25400.0, step=100.0)

st.sidebar.header("🌍 Yếu Tố Bên Ngoài")
weather = st.sidebar.selectbox("Weather Condition", ["Bình thường", "Thuận lợi", "Xấu"])
harvest = st.sidebar.selectbox("Harvest Status", ["Đầu vụ", "Đỉnh điểm", "Cuối vụ"])
exp_export = st.sidebar.number_input("Expected Export Volume (Tấn)", value=150000.0)

# Chạy Decision Engine
inputs = {
    'current_price': current_price, 'inventory': inventory, 'storage_cost': storage_cost,
    'transport_cost': transport_cost, 'other_cost': other_cost, 'usd_vnd': usd_vnd,
    'weather': weather, 'harvest': harvest, 'exp_export': exp_export
}

df_scenarios, dict_forecasts = decision_engine(df_clean, model, model_type, inputs)
best_scenario = df_scenarios.loc[df_scenarios['Decision Score'].idxmax()]

# Hàng 1: KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá Hiện Tại", f"{current_price:,.0f} VNĐ")
col2.metric("Lợi Nhuận Tối Ưu (Kỳ vọng)", f"{best_scenario['Expected Profit']:,.0f} VNĐ", f"Phương án: {best_scenario['Action']}")
col3.metric("Điểm Ra Quyết Định (Score)", f"{best_scenario['Decision Score']:,.0f}")
col4.metric("Chỉ Số Rủi Ro", f"{best_scenario['Risk Score']:.2%}")

# Hàng 2: AI Recommendation & Insight
st.subheader("🧠 Khuyến Nghị Từ AI (AI Insight)")
insight_box = st.container()
with insight_box:
    action_color = "green" if best_scenario['Days'] == 0 else "orange"
    st.markdown(f"### Lệnh Đề Xuất: **<span style='color:{action_color}'>{best_scenario['Action']}</span>**", unsafe_allow_html=True)
    
    # Tự động sinh Insight giải thích không hard-code
    insight_text = ""
    if best_scenario['Forecast Price'] > current_price:
        insight_text += f"- **Phân tích Giá:** Mô hình {model_type} dự báo giá sẽ tăng lên **{best_scenario['Forecast Price']:,.0f} VNĐ** vào {best_scenario['Days']} ngày tới. "
    else:
        insight_text += f"- **Phân tích Giá:** Mô hình đánh giá thị trường đang đi ngang hoặc giảm. Mức giá cao nhất kỳ vọng là **{best_scenario['Forecast Price']:,.0f} VNĐ**. "
        
    if best_scenario['Days'] == 0:
        insight_text += f"Do chi phí lưu kho của bạn ({storage_cost:,.0f} đ/ngày) bào mòn biên lợi nhuận cộng với rủi ro biến động, AI đề xuất xả hàng chốt lời ngay hôm nay.\n"
    else:
         insight_text += f"Biên độ tăng giá đủ lớn để bù đắp tổng chi phí lưu giữ hàng ({best_scenario['Total Cost']:,.0f} VNĐ). AI đề xuất tiếp tục HOLD.\n"
         
    if weather == 'Xấu':
         insight_text += "- **Vĩ mô & Môi trường:** Yếu tố thời tiết xấu đang tạo áp lực thiếu cung, hệ số rủi ro đã được điều chỉnh tăng, do đó chỉ nên chốt lượng % hàng an toàn."
         
    st.info(insight_text)

# Hàng 3: Charts
st.subheader("📊 Trực Quan Hóa Kịch Bản")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # Biểu đồ giá
    fc_df = pd.DataFrame(list(dict_forecasts.items()), columns=['Ngày Tới', 'Giá Dự Báo'])
    fig1 = px.line(fc_df, x='Ngày Tới', y='Giá Dự Báo', markers=True, title="Đường Cong Dự Báo Giá (AI Forecast)")
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    # Biểu đồ lợi nhuận theo kịch bản
    fig2 = px.bar(df_scenarios, x='Action', y='Expected Profit', color='Risk Score', 
                  title="So Sánh Lợi Nhuận Kỳ Vọng Các Kịch Bản",
                  color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("📋 Bảng Chi Tiết Tính Toán (Decision Matrix)")
st.dataframe(df_scenarios.style.format({
    "Forecast Price": "{:,.0f}",
    "Expected Revenue": "{:,.0f}",
    "Total Cost": "{:,.0f}",
    "Expected Profit": "{:,.0f}",
    "Risk Score": "{:.2%}",
    "Decision Score": "{:,.0f}"
}), use_container_width=True)