import streamlit as st
import google.generativeai as genai

# 1. Khởi tạo cấu hình Gemini từ Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Chọn model (gemini-1.5-flash tốc độ rất nhanh, phù hợp làm demo)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_expert_advice(quantum_result, market_news):
    """
    Hàm gọi Gemini để tổng hợp kết quả lượng tử và tin tức thị trường
    """
    prompt = f"""
    Bạn là một chuyên gia tư vấn chiến lược nông sản tại Việt Nam.
    Hệ thống AI và Lượng tử của chúng tôi vừa đưa ra quyết định sau cho HTX Cà phê:
    {quantum_result}
    
    Tin tức thị trường vĩ mô hiện tại:
    {market_news}
    
    Hãy viết một báo cáo tham mưu ngắn gọn (dưới 150 chữ), giải thích lý do tại sao HTX nên theo chiến lược lượng tử này. Dùng ngôn ngữ dễ hiểu, thuyết phục, nhấn mạnh vào việc giảm rủi ro lưu kho và tối ưu dòng tiền.
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- GIAO DIỆN STREAMLIT ---
st.title("AgriQ AI - Tối ưu hóa Nông sản")

# Mô phỏng dữ liệu đầu ra từ mô hình Lượng tử (Qiskit) đã chạy xong
quantum_output = """
- Tuần 1: Bán 30 tấn (Xả hàng thu tiền mặt trả nợ).
- Tuần 2-4: Trữ 70 tấn.
- Tuần 5: Bán 70 tấn (Dự báo đạt đỉnh giá cục bộ, hết khấu hao kho).
- Tổng biên lợi nhuận dự kiến: +18%.
"""

market_context = "El Nino đang gây hạn hán nhẹ tại Brazil, tỷ giá USD/VND đang neo ở mức cao."

st.subheader("1. Lộ trình xuất kho (Đề xuất bởi Qiskit)")
st.info(quantum_output)

st.subheader("2. Phân tích từ Chuyên gia AI (Gemini)")

# Nút kích hoạt Gemini
if st.button("Sinh báo cáo phân tích chiến lược", type="primary"):
    with st.spinner("Đang tổng hợp dữ liệu vĩ mô và lượng tử..."):
        try:
            # Gọi hàm tương tác API
            advice = generate_expert_advice(quantum_output, market_context)
            
            # Hiển thị kết quả đẹp mắt
            st.success("Đã hoàn tất phân tích!")
            st.markdown("### 📝 Khuyến nghị hành động")
            st.write(advice)
            
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi kết nối API: {e}")
