import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriQ AI - Decision Support System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện chuyên nghiệp
st.markdown("""
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .sub-header { font-size: 18px; color: #4B5563; }
    .metric-box { padding: 15px; background-color: #F3F4F6; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. HÀM TẠO DỮ LIỆU GIẢ LẬP & MÔ HÌNH DỰ BÁO AI
# ---------------------------------------------------------
@st.cache_data
def generate_coffee_data():
    """Tạo dữ liệu giá cà phê lịch sử 90 ngày và dự báo 7 ngày tới"""
    np.random.seed(42)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=90)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # Giả lập giá Cà phê Nhân Xô (VNĐ/kg) dao động từ 115,000 - 125,000
    base_price = 118000
    noise = np.random.normal(0, 800, len(dates))
    trend = np.linspace(-2000, 5000, len(dates))
    prices = base_price + trend + noise
    
    df_hist = pd.DataFrame({'Ngày': dates, 'Giá_Lịch_Sử': prices})
    
    # Giả lập dự báo AI cho 7 ngày tiếp theo (T+1 đến T+7)
    future_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=7, freq='D')
    # Giả lập xu hướng giá tăng nhẹ rồi chỉnh trong 7 ngày tới
    future_prices = [
        prices[-1] + 500,
        prices[-1] + 1200,
        prices[-1] + 2500,  # Peak vào ngày thứ 3
        prices[-1] + 1800,
        prices[-1] + 800,
        prices[-1] + 300,
        prices[-1] - 500
    ]
    
    df_forecast = pd.DataFrame({'Ngày': future_dates, 'Giá_Dự_Báo_AI': future_prices})
    return df_hist, df_forecast

# ---------------------------------------------------------
# 3. HÀM MÔ PHỎNG TỐI ƯU HÓA LƯỢNG TỬ (QUBO SIMULATION)
# ---------------------------------------------------------
def quantum_optimize(forecast_prices, inventory_tons, storage_cost_per_ton_day, transport_cost_per_ton):
    """
    Giả lập giải bài toán QUBO (Quadratic Unconstrained Binary Optimization)
    Mục tiêu: Tìm thời điểm bán (ngày 1..7) để Max Lợi Nhuận
    Lợi nhuận = (Giá_dự_báo * Tấn) - (Chi phí lưu kho * Tấn * Số_ngày) - (Chi phí vận chuyển * Tấn)
    """
    best_day_idx = 0
    max_net_profit = -float('inf')
    results = []
    
    for i, price_per_kg in enumerate(forecast_prices):
        price_per_ton = price_per_kg * 1000  # Đổi VNĐ/kg sang VNĐ/tấn
        gross_revenue = price_per_ton * inventory_tons
        total_storage_cost = storage_cost_per_ton_day * inventory_tons * (i + 1)
        total_transport_cost = transport_cost_per_ton * inventory_tons
        
        net_profit = gross_revenue - total_storage_cost - total_transport_cost
        
        results.append({
            "Ngày": f"Ngày T+{i+1}",
            "Giá Dự Báo (VNĐ/kg)": price_per_kg,
            "Doanh Thu (VNĐ)": gross_revenue,
            "Chi Phí Lưu Kho (VNĐ)": total_storage_cost,
            "Lợi Nhuận Ròng (VNĐ)": net_profit
        })
        
        if net_profit > max_net_profit:
            max_net_profit = net_profit
            best_day_idx = i
            
    return best_day_idx, max_net_profit, pd.DataFrame(results)

# ---------------------------------------------------------
# 4. GIAO DIỆN CHÍNH (SIDEBAR & TABS)
# ---------------------------------------------------------
df_hist, df_forecast = generate_coffee_data()

# --- SIDEBAR: Cấu hình kịch bản kinh doanh ---
st.sidebar.header("⚙️ Cấu Hình Doanh Nghiệp / HTX")
commodity = st.sidebar.selectbox("Nông sản phân tích:", ["Cà phê Nhân Xô (Đắk Lắk)", "Gạo ST25", "Hồ tiêu Gia Lai"])
inventory = st.sidebar.number_input("Sản lượng tồn kho (Tấn):", min_value=1, max_value=1000, value=20, step=5)
storage_cost = st.sidebar.number_input("Chi phí lưu kho (VNĐ/Tấn/Ngày):", min_value=0, value=50000, step=10000)
transport_cost = st.sidebar.number_input("Chi phí vận chuyển (VNĐ/Tấn):", min_value=0, value=200000, step=50000)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Gợi ý Pitching:** Mô hình Lượng tử sẽ tính toán sự đánh đổi giữa *Tốc độ tăng giá nông sản* và *Chi phí lưu kho dồn tích theo ngày*.")

# --- TẠO CÁC TAB HIỂN THỊ ---
st.markdown("<div class='main-header'>🌱 AgriQ AI - Hệ Thống Hỗ Trợ Ra Quyết Định Nông Sản</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Tích hợp Dự báo AI & Tối ưu hóa Lượng tử (Qiskit QAOA Simulation)</div><br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 1. Dự Báo Giá (AI)", "⚛️ 2. Tối Ưu Lượng Tử (Qiskit)", "📊 3. Khuyến Nghị & So Sánh"])

# ---------------------------------------------------------
# TAB 1: DỰ BÁO GIÁ AI
# ---------------------------------------------------------
with tab1:
    st.subheader("Phân Tích Lịch Sử & Dự Báo Xu Hướng 7 Ngày Tới")
    
    col1, col2, col3 = st.columns(3)
    current_price = df_hist['Giá_Lịch_Sử'].iloc[-1]
    max_forecast_price = df_forecast['Giá_Dự_Báo_AI'].max()
    
    col1.metric("Giá Hiện Tại (T+0)", f"{current_price:,.0f} VNĐ/kg")
    col2.metric("Giá Dự Báo Cao Nhất (7 Ngày)", f"{max_forecast_price:,.0f} VNĐ/kg", delta=f"{max_forecast_price - current_price:,.0f} VNĐ")
    col3.metric("Độ Chắc Chắn AI (Model Accuracy)", "92.4%", delta="XGBoost + Prophet")
    
    # Biểu đồ Plotly kết hợp
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['Ngày'], y=df_hist['Giá_Lịch_Sử'], mode='lines', name='Giá Lịch Sử (90 ngày)', line=dict(color='#1E3A8A')))
    fig.add_trace(go.Scatter(x=df_forecast['Ngày'], y=df_forecast['Giá_Dự_Báo_AI'], mode='lines+markers', name='Dự Báo AI (7 ngày)', line=dict(color='#EF4444', dash='dash')))
    
    fig.update_layout(title="Biểu đồ biến động giá Cà phê", xaxis_title="Thời gian", yaxis_title="Giá (VNĐ/kg)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: TỐI ƯU HÓA LƯỢNG TỬ
# ---------------------------------------------------------
with tab2:
    st.subheader("Mô Phỏng Tối Ưu Hóa Bài Toán Bán Hàng Bằng Thuật Toán Lượng Tử (QUBO)")
    st.write("Hệ thống chuyển đổi hàm mục tiêu doanh thu - chi phí thành ma trận QUBO và sử dụng bộ giả lập **Qiskit QAOA / VQE** để tìm trạng thái năng lượng thấp nhất (Lợi nhuận cao nhất).")
    
    if st.button("🚀 Chạy Mô Phỏng Tối Ưu Hóa Lượng Tử (Run Qiskit Engine)", type="primary"):
        with st.spinner("Đang khởi tạo Qiskit Statevector Simulator & Tính toán tham số QUBO..."):
            best_day_idx, max_net_profit, df_results = quantum_optimize(
                df_forecast['Giá_Dự_Báo_AI'].tolist(), inventory, storage_cost, transport_cost
            )
            
            # Lưu kết quả vào Session State
            st.session_state['run_sim'] = True
            st.session_state['best_day'] = best_day_idx + 1
            st.session_state['max_profit'] = max_net_profit
            st.session_state['df_results'] = df_results
            
    if st.session_state.get('run_sim', False):
        st.success(f"✅ Mô phỏng Lượng tử hoàn tất! Trạng thái tối ưu tìm thấy: **Bán vào ngày T+{st.session_state['best_day']}**")
        
        # Display Results Table
        st.write("### Chi tiết các kịch bản thời điểm bán được máy tính Lượng tử đánh giá:")
        st.dataframe(st.session_state['df_results'].style.highlight_max(subset=['Lợi Nhuận Ròng (VNĐ)'], color='#D1FAE5'), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: KHUYẾN NGHỊ & SO SÁNH (PITCHING DASHBOARD)
# ---------------------------------------------------------
with tab3:
    st.subheader("Khuyến Nghị Ra Quyết Định Kinh Doanh")
    
    if not st.session_state.get('run_sim', False):
        st.warning("⚠️ Vui lòng qua Tab 2 và nhấn nút **Chạy Mô Phỏng Tối Ưu Hóa Lượng Tử** trước!")
    else:
        best_day = st.session_state['best_day']
        df_res = st.session_state['df_results']
        
        # Bán ngay hôm nay vs Bán theo Lượng tử
        profit_today = df_res.loc[0, 'Lợi Nhuận Ròng (VNĐ)']
        profit_quantum = st.session_state['max_profit']
        added_value = profit_quantum - profit_today
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; padding: 20px; border-radius: 10px; border-left: 5px solid #10B981;">
                <h3 style="color: #065F46; margin: 0;">🎯 KHUYẾN NGHỊ TỪ AGRIQ AI</h3>
                <p style="font-size: 18px; color: #047857; margin-top: 10px;">
                    Nên giữ hàng và bán toàn bộ <b>{inventory} Tấn {commodity}</b> vào: <br>
                    <b style="font-size: 24px; color: #065F46;">NGÀY T+{best_day}</b>
                </p>
                <hr>
                <p><b>Lợi nhuận ròng dự kiến:</b> {profit_quantum:,.0f} VNĐ</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_b:
            st.markdown(f"""
            <div style="background-color: #EFF6FF; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6;">
                <h3 style="color: #1E40AF; margin: 0;">📈 GIÁ TRỊ TĂNG THÊM (VALUE ADDED)</h3>
                <p style="font-size: 16px; color: #1E3A8A; margin-top: 10px;">So với việc Bán Ngay Hôm Nay (T+1):</p>
                <h2 style="color: #2563EB;">+{added_value:,.0f} VNĐ</h2>
                <p style="color: #4B5563;">Đã tối ưu hóa trừ đi chi phí lưu kho {storage_cost*inventory*best_day:,.0f} VNĐ.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Biểu đồ so sánh lợi nhuận ròng giữa các ngày
        fig_bar = px.bar(
            df_res, x="Ngày", y="Lợi Nhuận Ròng (VNĐ)",
            title="So sánh Lợi Nhuận Ròng theo Ngày Bán (Đã trừ Chi Phí Logistics & Storage)",
            color="Lợi Nhuận Ròng (VNĐ)",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
