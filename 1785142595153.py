import yfinance as yf

# ---------------------------------------------------------
# THÊM ĐOẠN NÀY VÀO DƯỚI PHẦN SIDEBAR HIỆN TẠI
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("📡 Cập Nhật Giá Thực Tế (T+0)")

# Cho phép người dùng chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu giá hôm nay:", 
    ("Tự động (Sàn ICE Quốc tế)", "Thủ công (Thương lái báo)")
)

# Biến lưu giá thực tế hôm nay
today_price_real = 118000 # Giá mặc định dự phòng

if data_source == "Tự động (Sàn ICE Quốc tế)":
    with st.sidebar.status("Đang kết nối API toàn cầu...", expanded=True) as status:
        try:
            # Mã KC=F là Cà phê Arabica trên Yahoo Finance (Tính bằng US Cents / pound)
            coffee_ticker = yf.Ticker("KC=F")
            hist_data = coffee_ticker.history(period="1d")
            
            if not hist_data.empty:
                price_cents_lb = hist_data['Close'].iloc[-1]
                
                # CÔNG THỨC CHUYỂN ĐỔI: Cents/lb -> VNĐ/kg
                # 1 kg = 2.20462 lbs. Tỷ giá giả định 1 USD = 25,400 VNĐ
                usd_vnd_rate = 25400
                price_vnd_kg = (price_cents_lb / 100) * 2.20462 * usd_vnd_rate
                
                today_price_real = round(price_vnd_kg, -2) # Làm tròn đến hàng trăm
                status.update(label="Đã đồng bộ giá thành công!", state="complete")
                st.sidebar.success(f"✅ Giá ICE quy đổi: {today_price_real:,.0f} VNĐ/kg")
            else:
                status.update(label="Lỗi dữ liệu. Chuyển sang thủ công.", state="error")
        except Exception as e:
            status.update(label="Mạng lỗi. Dùng tính năng thủ công.", state="error")

else:
    # Phương án Nhập thủ công (Cứu tinh khi đi thi mất mạng)
    today_price_real = st.sidebar.number_input(
        "Nhập giá thu mua tại địa phương (VNĐ/kg):", 
        min_value=50000, max_value=200000, value=122000, step=1000
    )
    st.sidebar.success(f"📝 Đã ghi nhận giá nội địa: {today_price_real:,.0f} VNĐ/kg")

# Lưu giá trị này vào Session State để phần AI Dự báo (Tab 1) 
# và Tối ưu Lượng tử (Tab 2) có thể lấy ra sử dụng
st.session_state['current_real_price'] = today_price_real
