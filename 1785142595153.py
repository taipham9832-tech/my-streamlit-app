import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
from pathlib import Path

# Thư viện Lượng tử (Bước 4)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI & Quantum Coffee Engine", page_icon="⚛️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# BƯỚC 1 & 2: THU THẬP VÀ LÀM SẠCH DỮ LIỆU
# ==========================================
@st.cache_data(show_spinner=False)
def data_pipeline():
    # Mô phỏng quá trình tổng hợp dữ liệu từ nhiều nguồn (Giá, CPI, Lãi suất, Thời tiết)
    # Trong môi trường thực tế, phần này sẽ đọc từ thư mục data/
    dates = pd.date_range(start='2020-01-01', end=datetime.today(), freq='D')
    np.random.seed(42)
    prices = 32000 + np.cumsum(np.random.normal(20, 300, len(dates)))
    prices = np.clip(prices, 30000, 130000)
    
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill().reset_index()
    
    # Tạo đặc trưng (Feature Engineering)
    df['Lag_1'] = df['Price'].shift(1)
    df['Lag_3'] = df['Price'].shift(3)
    df['Lag_7'] = df['Price'].shift(7)
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7).mean()
    df['Rolling_Mean_14'] = df['Price'].rolling(window=14).mean()
    df.dropna(inplace=True)
    
    return df

# ==========================================
# BƯỚC 3: XÂY DỰNG MÔ HÌNH AI
# ==========================================
@st.cache_resource(show_spinner=False)
def train_ai_models(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    features = ['Lag_1', 'Lag_3', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_14']
    
    # Huấn luyện XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(train[features], train['Price'])
    xgb_pred = xgb_model.predict(test[features])
    xgb_rmse = np.sqrt(mean_squared_error(test['Price'], xgb_pred))
    
    # Huấn luyện Prophet
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(train[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'}))
    future = prophet_model.make_future_dataframe(periods=len(test))
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'].iloc[-len(test):].values
    prophet_rmse = np.sqrt(mean_squared_error(test['Price'], prophet_pred))
    
    # Chọn mô hình
    if xgb_rmse <= prophet_rmse:
        best_model, best_type = xgb_model, "XGBoost"
        best_model.fit(df[features], df['Price'])
    else:
        best_model, best_type = Prophet(daily_seasonality=True), "Prophet"
        best_model.fit(df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'}))
        
    return best_model, best_type, df

# Hàm phụ trợ sinh dự báo từ AI
def generate_forecasts(df, model, model_type, current_price, weather):
    forecasts = {0: current_price}
    if model_type == "XGBoost":
        current_features = df.iloc[-1].copy()
        for day in [3, 7, 14]:
            pred = model.predict(np.array([[current_features['Lag_1'], current_features['Lag_3'], current_features['Lag_7'], current_features['Rolling_Mean_7'], current_features['Rolling_Mean_14']]]))[0]
            if weather == 'Thuận lợi': pred *= 0.99
            elif weather == 'Xấu': pred *= 1.025
            forecasts[day] = float(pred)
            current_features['Lag_1'] = pred
    return forecasts

# ==========================================
# BƯỚC 4: MÔ PHỎNG TỐI ƯU HÓA LƯỢNG TỬ (QISKIT)
# ==========================================
def quantum_optimization(inputs, forecasts):
    # Khởi tạo 8 kịch bản (Tương ứng với 3 Qubit: 2^3 = 8 trạng thái)
    days_options = [0, 3, 7, 14] # 4 mốc thời gian
    vol_options = [0.5, 1.0]     # 2 mức bán (50% hoặc 100%)
    
    scenarios = []
    scores = []
    state_labels = []
    
    # Tính toán Decision Score cho từng kịch bản
    for i, d in enumerate(days_options):
        for j, v in enumerate(vol_options):
            sell_vol = inputs['inventory'] * v
            holding_cost = inputs['inventory'] * inputs['storage_cost'] * d
            total_cost = holding_cost + (sell_vol * inputs['transport_cost']) + (sell_vol * inputs['other_cost'])
            profit = (sell_vol * forecasts[d]) - total_cost
            
            risk = 0.05 + (d * 0.015)
            if inputs['weather'] == 'Xấu': risk += 0.08
            
            score = max(profit / (1 + risk), 0.1) # Tránh số âm
            
            # Ánh xạ thành mã nhị phân (Ví dụ: 000, 001, ..., 111)
            state = f"{i:02b}{j:01b}"
            
            scenarios.append({
                'State': state, 'Days': d, 'Volume (%)': int(v*100), 'Forecast Price': forecasts[d],
                'Expected Profit': profit, 'Total Cost': total_cost, 'Risk Score': risk, 'Score': score,
                'Action': f"Bán {int(v*100)}% sau {d} ngày" if d > 0 else f"Bán {int(v*100)}% ngay lập tức"
            })
            scores.append(score)
            state_labels.append(state)

    df_scenarios = pd.DataFrame(scenarios)

    # Nếu có thư viện Qiskit -> Chạy mô phỏng lượng tử
    if QISKIT_AVAILABLE:
        # Chuẩn hóa Score thành xác suất Lượng tử (Amplitudes)
        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]
        amplitudes = np.sqrt(probabilities) # Biên độ xác suất
        
        # Khởi tạo Mạch lượng tử (Quantum Circuit) với 3 Qubit
        qc = QuantumCircuit(3)
        qc.initialize(amplitudes, [0, 1, 2])
        qc.measure_all()
        
        # Chạy mô phỏng trên AerSimulator (1024 shots)
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        # Mapping kết quả đếm (Counts) về DataFrame
        df_scenarios['Quantum_Shots'] = df_scenarios['State'].map(counts).fillna(0)
        df_scenarios['Quantum_Probability'] = df_scenarios['Quantum_Shots'] / 1024
        
        # Trạng thái sụp đổ nhiều nhất = Quyết định tối ưu
        best_state = max(counts, key=counts.get)
        best_scenario = df_scenarios[df_scenarios['State'] == best_state].iloc[0]
        
    else:
        # Fallback nếu Streamlit không cài được Qiskit
        df_scenarios['Quantum_Shots'] = 0
        df_scenarios['Quantum_Probability'] = df_scenarios['Score'] / sum(df_scenarios['Score'])
        best_scenario = df_scenarios.loc[df_scenarios['Score'].idxmax()]
        counts = {}

    return df_scenarios, best_scenario, counts

# ==========================================
# BƯỚC 5: DASHBOARD TRỰC QUAN (STREAMLIT UI)
# ==========================================
with st.spinner("Khởi tạo Dữ liệu & Huấn luyện Hybrid AI-Quantum Model..."):
    df_clean = data_pipeline()
    model, model_type, df_clean = train_ai_models(df_clean)

st.title("⚛️ C-QMS: Coffee Quantum Management System")
st.markdown("Hệ thống kết hợp dự báo **Machine Learning** và tối ưu hóa quyết định bằng **Mô phỏng Lượng tử (Qiskit)**.")

# Giao diện nhập liệu
st.sidebar.header("📥 Biến Số Vĩ Mô & Nội Bộ")
current_price = st.sidebar.number_input("Giá Hiện Tại (VNĐ/kg)", value=float(df_clean['Price'].iloc[-1]))
inventory = st.sidebar.number_input("Tồn Kho (Tấn)", value=100.0)
storage_cost = st.sidebar.number_input("Lưu Kho (VNĐ/tấn/ngày)", value=15000.0)
transport_cost = st.sidebar.number_input("Vận Chuyển (VNĐ/tấn)", value=200000.0)
other_cost = st.sidebar.number_input("Chi phí khác (VNĐ/tấn)", value=50000.0)
weather = st.sidebar.selectbox("Thời Tiết (Dữ liệu ngoại sinh)", ["Bình thường", "Thuận lợi", "Xấu"])

inputs = {
    'inventory': inventory, 'storage_cost': storage_cost, 'transport_cost': transport_cost,
    'other_cost': other_cost, 'weather': weather
}

# Xử lý luồng
forecasts = generate_forecasts(df_clean, model, model_type, current_price, weather)
df_scenarios, best_scenario, quantum_counts = quantum_optimization(inputs, forecasts)

# Hiển thị Kết quả Tối ưu
st.subheader("🏆 Phương Án Tối Ưu Lượng Tử (Quantum Recommended Action)")
st.success(f"### Lệnh: {best_scenario['Action']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lợi Nhuận Ước Tính", f"{best_scenario['Expected Profit']:,.0f} đ")
col2.metric("Mức Rủi Ro (Risk)", f"{best_scenario['Risk Score']:.1%}")
col3.metric("Trạng Thái Qubit", f"|{best_scenario['State']}⟩")
col4.metric("Độ Tin Cậy Lượng Tử", f"{best_scenario['Quantum_Probability']:.1%}")

# Biểu đồ
st.subheader("📊 Trực Quan Hóa Hệ Thống Kép (AI & Quantum)")
c1, c2 = st.columns(2)

with c1:
    fig_ai = px.line(x=list(forecasts.keys()), y=list(forecasts.values()), markers=True, 
                     title=f"AI Forecast ({model_type}) - Xu hướng Giá", labels={'x': 'Ngày', 'y': 'VNĐ/kg'})
    st.plotly_chart(fig_ai, use_container_width=True)

with c2:
    if QISKIT_AVAILABLE:
        fig_q = px.bar(x=list(quantum_counts.keys()), y=list(quantum_counts.values()), 
                       title="Quantum State Measurement (1024 Shots)", labels={'x': 'Qubit State', 'y': 'Frequency'})
        st.plotly_chart(fig_q, use_container_width=True)
    else:
        st.warning("Thư viện Qiskit chưa được cài đặt. Đang hiển thị kết quả cổ điển.")

# Bảng chi tiết
st.subheader("📋 Phân Tích Đa Không Gian (Scenario Matrix)")
st.dataframe(df_scenarios.style.background_gradient(subset=['Quantum_Probability'], cmap='Blues').format({
    "Expected Profit": "{:,.0f}", "Total Cost": "{:,.0f}", "Forecast Price": "{:,.0f}",
    "Risk Score": "{:.2%}", "Score": "{:.2f}", "Quantum_Probability": "{:.1%}"
}), use_container_width=True)
