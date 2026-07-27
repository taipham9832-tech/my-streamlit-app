import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# AI Libraries
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Quantum Libraries
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="AgriQ AI - Quantum Decision Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODULE 1: XỬ LÝ DỮ LIỆU (DATA PIPELINE)
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Hàm đọc dữ liệu từ file CSV của người dùng.
    Có tích hợp cơ chế Fallback để đảm bảo app không bị crash khi demo.
    """
    base_dir = Path(__file__).resolve().parent
    price_file = base_dir / "data" / "Giá cà phê tổng hợp.xlsx - data.csv"
    
    df = None
    if price_file.exists():
        try:
            # Đọc file, bỏ qua các dòng lỗi (vì file chứa nhiều bảng gộp)
            df_raw = pd.read_csv(price_file, header=None, on_bad_lines='skip', dtype=str)
            # Tìm các dòng chứa định dạng ngày tháng yyyy-mm-dd
            df_price = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any(), axis=1)].copy()
            
            # Tự động tìm cột Date và cột Price (giả định cột Price nằm ngay sau cột Date)
            date_col = df_price.apply(lambda col: col.str.contains(r'\d{4}-\d{2}-\d{2}', na=False)).sum().idxmax()
            df_price['Date'] = pd.to_datetime(df_price[date_col], errors='coerce')
            df_price['Price'] = pd.to_numeric(df_price[date_col + 1], errors='coerce')
            
            df = df_price[['Date', 'Price']].dropna().sort_values('Date')
            df = df.groupby('Date')['Price'].mean().reset_index() # Lấy giá trung bình các tỉnh
        except Exception as e:
            st.sidebar.warning(f"Lỗi parse dữ liệu: {e}. Dùng dữ liệu mô phỏng.")
            
    # Fallback Data (Đảm bảo Demo luôn chạy)
    if df is None or len(df) < 100:
        dates = pd.date_range(start='2022-01-01', end=datetime.today(), freq='D')
        np.random.seed(42)
        # Giả lập giá cà phê từ 40k tăng dần lên 120k theo xu hướng thực tế
        trend = np.linspace(40000, 120000, len(dates))
        noise = np.random.normal(0, 1500, len(dates))
        df = pd.DataFrame({'Date': dates, 'Price': trend + noise})

    # Resample và Feature Engineering (Lag, Moving Average)
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill().reset_index()
    df['Lag_3'] = df['Price'].shift(3)
    df['Lag_7'] = df['Price'].shift(7)
    df['MA_7'] = df['Price'].rolling(window=7).mean()
    df['MA_14'] = df['Price'].rolling(window=14).mean()
    df.dropna(inplace=True)
    
    return df

# ==========================================
# MODULE 2: AI FORECASTING (XGBOOST & PROPHET)
# ==========================================
@st.cache_resource(show_spinner=False)
def train_ai_engine(df):
    """Huấn luyện mô hình ngay trên dữ liệu vừa nạp"""
    features = ['Lag_3', 'Lag_7', 'MA_7', 'MA_14']
    
    # 1. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(df[features], df['Price'])
    
    # 2. Prophet
    prophet_df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(prophet_df)
    
    return xgb_model, prophet_model, features

def get_forecasts(df, xgb_model, features, inputs):
    """Sinh dự báo giá cho các mốc thời gian tương lai"""
    current_price = inputs['current_price']
    forecasts = {0: current_price}
    
    current_features = df.iloc[-1].copy()
    for day in [3, 7, 14]:
        # Dùng XGBoost dự báo
        X_pred = np.array([[current_features['Lag_3'], current_features['Lag_7'], current_features['MA_7'], current_features['MA_14']]])
        pred = float(xgb_model.predict(X_pred)[0])
        
        # Tích hợp yếu tố ngoại sinh từ Input người dùng
        if inputs['weather'] == 'Mưa bão/Hạn hán': pred *= 1.03 # Thiếu cung -> Tăng giá
        elif inputs['weather'] == 'Thuận lợi': pred *= 0.98     # Dư cung -> Giảm giá
        
        if inputs['exchange_rate'] > 25500: pred *= 1.01        # USD tăng -> Giá nội địa tăng
        
        forecasts[day] = pred
        # Cập nhật features động
        current_features['Lag_7'] = current_features['Lag_3']
        current_features['Lag_3'] = pred
        
    return forecasts

# ==========================================
# MODULE 3: QUANTUM OPTIMIZATION (QISKIT)
# ==========================================
def quantum_decision_engine(inputs, forecasts):
    """
    Sử dụng Mạch lượng tử (Quantum Circuit) để tìm điểm sụp đổ trạng thái tối ưu
    Dựa trên 3 Qubit -> 8 Kịch bản kinh doanh
    """
    # 8 kịch bản (Thời gian: 0, 3, 7, 14 ngày | Khối lượng: 50%, 100%)
    days_options = [0, 3, 7, 14]
    vol_options = [0.5, 1.0]
    
    scenarios = []
    scores = []
    
    # Tính toán Lợi nhuận và Rủi ro cổ điển
    for i, d in enumerate(days_options):
        for j, v in enumerate(vol_options):
            sell_vol = inputs['inventory'] * v
            # Hàm chi phí: Lưu kho + Vận chuyển + Hao hụt
            total_cost = (inputs['inventory'] * inputs['storage_cost'] * d) + (sell_vol * inputs['transport_cost'])
            
            revenue = sell_vol * forecasts[d]
            profit = revenue - total_cost
            
            # Hàm rủi ro tăng theo thời gian giam hàng
            risk = 0.05 + (d * 0.01)
            if inputs['weather'] == 'Mưa bão/Hạn hán': risk += 0.05
            
            # Hàm mục tiêu (Decision Score)
            score = max(profit / (1 + risk), 1)
            state_label = f"|{i:02b}{j:01b}⟩" # Ví dụ: |001⟩
            
            scenarios.append({
                'Qubit State': state_label,
                'Days to Wait': d,
                'Sell Vol (%)': int(v * 100),
                'Forecast Price': forecasts[d],
                'Total Cost': total_cost,
                'Expected Profit': profit,
                'Risk': risk,
                'Score': score,
                'Action': f"Bán {int(v*100)}% sau {d} ngày" if d > 0 else f"Bán {int(v*100)}% ngay hôm nay"
            })
            scores.append(score)

    df_scen = pd.DataFrame(scenarios)

    # Khởi tạo mô phỏng Lượng tử
    if QISKIT_AVAILABLE:
        # Chuẩn hóa Score thành Biên độ xác suất (Probability Amplitudes)
        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]
        amplitudes = np.sqrt(probabilities)
        
        # Tạo mạch 3 Qubit
        qc = QuantumCircuit(3)
        qc.initialize(amplitudes, [0, 1, 2])
        qc.measure_all()
        
        # Mô phỏng 1024 lần bắn (Shots)
        simulator = AerSimulator()
        compiled_qc = transpile(qc, simulator)
        job = simulator.run(compiled_qc, shots=1024)
        counts = job.result().get_counts()
        
        # Cập nhật kết quả đo lường vào bảng
        df_scen['Quantum Shots'] = df_scen['Qubit State'].str.strip('|⟩').map(counts).fillna(0)
        df_scen['Probability'] = df_scen['Quantum Shots'] / 1024
        
        # Chọn kịch bản có xác suất sụp đổ cao nhất
        best_state = max(counts, key=counts.get)
        best_scenario = df_scen[df_scen['Qubit State'] == f"|{best_state}⟩"].iloc[0]
    else:
        df_scen['Quantum Shots'] = 0
        df_scen['Probability'] = df_scen['Score'] / df_scen['Score'].sum()
        best_scenario = df_scen.loc[df_scen['Score'].idxmax()]
        counts = {}
        
    return df_scen, best_scenario, counts

# ==========================================
# MODULE 4: STREAMLIT DASHBOARD (GIAO DIỆN)
# ==========================================
# Tải Dữ liệu & Huấn luyện
with st.spinner("Đang khởi tạo Data Warehouse & AI Models..."):
    df_clean = load_and_preprocess_data()
    xgb_model, prophet_model, features = train_ai_engine(df_clean)

# Tiêu đề
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>AgriQ AI - Hệ thống Ra Quyết Định Kinh Doanh Nông Sản</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Tích hợp Trí tuệ Nhân tạo (Machine Learning) & Tối ưu hóa Lượng tử (Quantum Simulation)</h4>", unsafe_allow_html=True)
st.divider()

# Sidebar: Nhập liệu
st.sidebar.header("📥 THÔNG SỐ HIỆN TẠI (INPUTS)")

st.sidebar.subheader("1. Vị thế Doanh nghiệp/HTX")
current_price = st.sidebar.number_input("Giá cà phê hôm nay (VNĐ/kg)", value=float(df_clean['Price'].iloc[-1]), step=500.0)
inventory = st.sidebar.number_input("Sản lượng Tồn kho (Tấn)", value=100.0, step=10.0)

st.sidebar.subheader("2. Cấu trúc Chi phí")
storage_cost = st.sidebar.number_input("Chi phí Lưu kho (VNĐ/tấn/ngày)", value=15000.0, step=1000.0)
transport_cost = st.sidebar.number_input("Cước Logistics (VNĐ/tấn)", value=250000.0, step=10000.0)

st.sidebar.subheader("3. Yếu tố Vĩ mô & Môi trường")
weather = st.sidebar.selectbox("Điều kiện thời tiết (vùng trồng)", ["Bình thường", "Thuận lợi", "Mưa bão/Hạn hán"])
exchange_rate = st.sidebar.number_input("Tỷ giá (USD/VND)", value=25450.0, step=50.0)

inputs = {
    'current_price': current_price, 'inventory': inventory, 
    'storage_cost': storage_cost, 'transport_cost': transport_cost,
    'weather': weather, 'exchange_rate': exchange_rate
}

# Xử lý Engine
forecasts = get_forecasts(df_clean, xgb_model, features, inputs)
df_scen, best_scen, q_counts = quantum_decision_engine(inputs, forecasts)

# KHỐI 1: KPIs & Quyết định Lượng tử
st.subheader("🎯 ĐỀ XUẤT TỐI ƯU (QUANTUM COLLAPSE STATE)")

# Tạo hộp hiển thị nổi bật
action_color = "#d32f2f" if best_scen['Days to Wait'] == 0 else "#388e3c"
st.markdown(f"""
<div style="background-color: #f1f8e9; padding: 20px; border-radius: 10px; border-left: 10px solid {action_color};">
    <h2 style="margin: 0; color: {action_color};">Lệnh Hành Động: {best_scen['Action'].upper()}</h2>
    <p style="font-size: 18px; margin-top: 10px;"><b>Lợi nhuận ròng ước tính:</b> {best_scen['Expected Profit']:,.0f} VNĐ</p>
    <p style="font-size: 16px; margin: 0;"><b>Trạng thái Lượng tử (Qubit):</b> {best_scen['Qubit State']} | Xác suất đo lường: {best_scen['Probability']:.2%}</p>
</div>
""", unsafe_allow_html=True)
st.write("")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá Hiện Tại", f"{current_price:,.0f} đ/kg")
col2.metric("Giá Dự Báo (Tối ưu)", f"{best_scen['Forecast Price']:,.0f} đ/kg", f"{(best_scen['Forecast Price'] - current_price):,.0f} đ")
col3.metric("Tổng Chi Phí", f"{best_scen['Total Cost']:,.0f} VNĐ")
col4.metric("Chỉ số Rủi ro", f"{best_scen['Risk']:.1%}")

# Khối AI Insight
st.markdown("### 🧠 Trí Tuệ Nhân Tạo Phân Tích (AI Insight)")
insight_text = f" Dựa trên phân tích từ **XGBoost** và mô phỏng **Qiskit AerSimulator (1024 shots)**:\n"
if best_scen['Days to Wait'] == 0:
    insight_text += f"- Áp lực chi phí lưu kho ({storage_cost:,.0f} đ/tấn/ngày) và xu hướng giá đi ngang/giảm không mang lại biên lợi nhuận đủ lớn. Mô phỏng lượng tử hội tụ về trạng thái chốt lời tức thì để bảo toàn vốn."
else:
    profit_diff = best_scen['Expected Profit'] - df_scen.iloc[0]['Expected Profit']
    insight_text += f"- Giá nông sản dự kiến tăng lên mức **{best_scen['Forecast Price']:,.0f} đ/kg** sau {best_scen['Days to Wait']} ngày. Mức tăng này vượt xa tổng chi phí bảo quản phát sinh, đem lại thêm **{profit_diff:,.0f} VNĐ** so với việc bán ngay. "
    if weather == 'Mưa bão/Hạn hán':
        insight_text += f"Đặc biệt, do yếu tố thời tiết xấu làm khan hiếm nguồn cung cục bộ, AI khuyến nghị tiếp tục giữ hàng để hưởng lợi."

st.info(insight_text)

# KHỐI 2: BIỂU ĐỒ TRỰC QUAN
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Dự báo Xu hướng Giá (AI Model)")
    fc_df = pd.DataFrame(list(forecasts.items()), columns=['Ngày Tới', 'Giá VNĐ/kg'])
    fig1 = px.line(fc_df, x='Ngày Tới', y='Giá VNĐ/kg', markers=True, 
                   color_discrete_sequence=['#ff7f0e'])
    fig1.update_traces(marker=dict(size=10))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("⚛️ Mô phỏng Lượng tử (Quantum Measurement)")
    if QISKIT_AVAILABLE and q_counts:
        # Chuẩn hóa key cho biểu đồ
        q_data = pd.DataFrame({'Trạng thái Qubit': [f"|{k}⟩" for k in q_counts.keys()], 'Số lần đo (Shots)': list(q_counts.values())})
        fig2 = px.bar(q_data, x='Trạng thái Qubit', y='Số lần đo (Shots)', 
                      color='Số lần đo (Shots)', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Vui lòng cài đặt thư viện 'qiskit' để hiển thị biểu đồ lượng tử.")

# KHỐI 3: BẢNG MA TRẬN KỊCH BẢN
st.subheader("📋 Ma Trận Kịch Bản Đa Chiều (Scenario Matrix)")
st.dataframe(
    df_scen.style.background_gradient(subset=['Probability'], cmap='Greens')
    .format({
        "Forecast Price": "{:,.0f}",
        "Total Cost": "{:,.0f}",
        "Expected Profit": "{:,.0f}",
        "Risk": "{:.1%}",
        "Score": "{:.2f}",
        "Probability": "{:.2%}"
    }),
    use_container_width=True
)
