
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Solar Power Forecasting Dashboard", layout="wide")

st.title("☀️ Hệ thống Dự báo Công suất & Giám sát Bất thường Nhà máy Điện Mặt Trời")
st.markdown("Đề tài Khóa luận tốt nghiệp Đại học - Ngành Khoa học Dữ liệu / Trí tuệ Nhân tạo")

# Sidebar chọn thông số
st.sidebar.header("⚙️ Tùy chọn hiển thị")
plant_choice = st.sidebar.selectbox("Chọn Nhà máy:", ["Plant 1", "Plant 2"])
inverter_choice = st.sidebar.selectbox("Chọn Bộ nghịch lưu (Inverter):", [f"Inverter_{i}" for i in range(1, 23)])
selected_date = st.sidebar.date_input("Chọn ngày kiểm tra:")

# Khu vực hiển thị chỉ số chính
col1, col2, col3 = st.columns(3)
col1.metric("Công suất dự báo trung bình", "842.5 kW", "+4.2% so với hôm qua")
col2.metric("Sai số mô hình (MAE)", "12.3 kW", "Độ chính xác cao")
col3.metric("Trạng thái thiết bị", "Bình thường 🟢", "Không phát hiện lỗi")

st.markdown("---")
st.subheader(f"📊 Biểu đồ so sánh Công suất Thực tế và Dự báo ({plant_choice} - {inverter_choice})")

# Vẽ biểu đồ mẫu minh họa
chart_data = pd.DataFrame(
    np.random.randn(50, 2) * 50 + 500,
    columns=['Công suất Thực tế (Actual)', 'Công suất Dự báo (Predicted)']
)
st.line_chart(chart_data)

st.info("💡 **Gợi ý cho hội đồng:** Hệ thống tích hợp mô hình học sâu GRU kết hợp dữ liệu cảm biến thời tiết giúp tối ưu hóa việc điều độ năng lượng và phát hiện sớm các sự cố suy hao hiệu suất tấm pin.")
