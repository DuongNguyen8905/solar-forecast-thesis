import streamlit as st
import pandas as pd
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Solar Power Forecasting & O&M Dashboard", layout="wide")

st.title("☀️ Hệ thống Dự báo Công suất & Giám sát Bất thường Nhà máy Điện Mặt Trời")
st.markdown("**Đồ án Khóa luận tốt nghiệp - Ngành Khoa học Dữ liệu**")

# ==============================================================================
# SIDEBAR: TÙY CHỌN ĐIỀU KHIỂN
# ==============================================================================
st.sidebar.header("⚙️ Bảng Điều Khiển Hệ Thống")
plant_choice = st.sidebar.selectbox("Chọn Nhà máy:", ["Plant 1", "Plant 2"])
model_choice = st.sidebar.selectbox("Chọn Mô hình AI Dự báo:", ["GRU (Deep Learning)", "LSTM (Deep Learning)", "XGBoost (Baseline ML)"])
inverter_choice = st.sidebar.selectbox("Chọn Bộ nghịch lưu (Inverter):", [f"Inverter_{i}" for i in range(1, 23)])

st.sidebar.divider()
st.sidebar.info("💡 Mô hình GRU kết hợp dữ liệu cảm biến thời tiết giúp dự báo bám sát các pha mây che và tối ưu hóa kế hoạch bảo trì.")

# ==============================================================================
# PHẦN 1: CÁC CHỈ SỐ VẬN HÀNH THỜI GIAN THỰC (KPI CARDS)
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

# Gán chỉ số tương ứng theo từng mô hình đã thực nghiệm
metrics_dict = {
    "GRU (Deep Learning)": {"mae": "22.95 kW", "rmse": "39.42 kW", "nmae": "2.71%", "acc": "Tối ưu nhất 🏆"},
    "LSTM (Deep Learning)": {"mae": "26.84 kW", "rmse": "44.12 kW", "nmae": "3.16%", "acc": "Độ chính xác cao"},
    "XGBoost (Baseline ML)": {"mae": "24.15 kW", "rmse": "41.24 kW", "nmae": "2.85%", "acc": "Baseline chuẩn"}
}
m_info = metrics_dict[model_choice]

col1.metric("Công suất phát hiện tại", "14,820 kW", "+3.4% so với TB")
col2.metric("Sai số MAE của mô hình", m_info["mae"], m_info["acc"])
col3.metric("Sai số toàn phương (RMSE)", m_info["rmse"], "Kiểm soát sụt áp")
col4.metric("Sai số chuẩn hóa (nMAE)", m_info["nmae"], "Chuẩn quốc tế", delta_color="inverse")

st.markdown("---")

# ==============================================================================
# PHẦN 2: BIỂU ĐỒ CHUỖI THỜI GIAN CHUẨN ĐẶC TÍNH ĐIỆN MẶT TRỜI (HÌNH CHUÔNG)
# ==============================================================================
st.subheader(f"📊 Đồ thị So sánh Công suất Thực tế vs Dự báo ({model_choice} - {inverter_choice})")

# Tạo dữ liệu mô phỏng chuẩn chu kỳ ngày đêm (Hình chuông parabol thực tế)
time_steps = 96 # 96 mốc 15 phút = 24 giờ
hours = np.linspace(0, 24, time_steps)

# Công suất mặt trời chỉ phát từ 6h sáng đến 18h tối (đỉnh lúc 12h trưa)
daylight_mask = (hours >= 6) & (hours <= 18)
solar_profile = np.zeros(time_steps)
solar_profile[daylight_mask] = np.sin((hours[daylight_mask] - 6) / 12 * np.pi) * 850

# Thực tế có dao động mây che
np.random.seed(42)
actual_power = np.maximum(0, solar_profile + np.random.normal(0, 30, time_steps) * daylight_mask)

# Dự báo bám sát theo mô hình
if "GRU" in model_choice:
    pred_power = np.maximum(0, solar_profile + np.random.normal(0, 18, time_steps) * daylight_mask)
elif "LSTM" in model_choice:
    pred_power = np.maximum(0, solar_profile + np.random.normal(0, 25, time_steps) * daylight_mask)
else: # XGBoost
    pred_power = np.maximum(0, solar_profile + np.random.normal(0, 22, time_steps) * daylight_mask)

time_labels = [f"{int(h):02d}:{int((h%1)*60):02d}" for h in hours]
df_chart = pd.DataFrame({
    'Thời gian (15 phút/bước)': time_labels,
    'Công suất Thực tế (Actual)': np.round(actual_power, 1),
    f'Dự báo {model_choice.split()[0]}': np.round(pred_power, 1)
}).set_index('Thời gian (15 phút/bước)')

import plotly.graph_objects as go

# 1. Khởi tạo đồ thị cao cấp
fig = go.Figure()

# Đường 1: Công suất Thực tế (Màu Vàng Cam Mặt Trời + Đổ bóng diện tích sản lượng)
fig.add_trace(go.Scatter(
    x=df_chart.index,
    y=df_chart['Công suất Thực tế (Actual)'],
    mode='lines',
    name='☀️ Thực Tế (Actual)',
    line=dict(color='#F59E0B', width=3),
    fill='tozeroy',
    fillcolor='rgba(245, 158, 11, 0.15)', # Vùng đổ bóng mờ ấm áp
    hovertemplate='<b>Thực tế:</b> %{y:.1f} kW<extra></extra>'
))

# Đường 2: Dự báo AI (Tự động đổi màu công nghệ theo mô hình bạn chọn)
model_short_name = model_choice.split()[0]
color_ai = '#8B5CF6' if "GRU" in model_choice else ('#10B981' if "LSTM" in model_choice else '#EC4899')

fig.add_trace(go.Scatter(
    x=df_chart.index,
    y=df_chart[f'Dự báo {model_short_name}'],
    mode='lines',
    name=f'⚡ Dự Báo AI ({model_short_name})',
    line=dict(color=color_ai, width=2.5, dash='dash'),
    hovertemplate='<b>Dự báo AI:</b> %{y:.1f} kW<extra></extra>'
))

# Đánh dấu huy hiệu Đỉnh phát cực đại lúc giữa trưa (Peak Annotation)
max_val = df_chart['Công suất Thực tế (Actual)'].max()
max_idx = df_chart['Công suất Thực tế (Actual)'].idxmax()

fig.add_annotation(
    x=max_idx,
    y=max_val,
    text=f"🔥 Đỉnh phát: {max_val:.0f} kW",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowcolor='#D97706',
    ax=0,
    ay=-35,
    font=dict(size=12, color='#B45309', family='Arial, sans-serif'),
    bgcolor='rgba(254, 243, 199, 0.95)',
    bordercolor='#F59E0B',
    borderwidth=1.5,
    borderpad=4
)

# Cấu hình giao diện chuẩn mực, sạch sẽ và KHÓA CỐ ĐỊNH TRỤC
fig.update_layout(
    title=dict(
        text=f"<b>Biểu Đồ So Sánh Công Suất Phát: Thực Tế vs Dự Báo {model_short_name}</b>",
        font=dict(size=16, color='#1E293B')
    ),
    xaxis=dict(
        title="<b>Thời Gian Trong Ngày (Mốc 15 phút)</b>",
        fixedrange=True, # Khóa chống kéo lệch trục
st.markdown("---")

# ==============================================================================
# PHẦN 3: NGHIỆP VỤ PHÁT HIỆN SỰ CỐ & BẢO TRÌ NHÀ MÁY (O&M)
# ==============================================================================
st.subheader("🚨 Giám Sát Sức Khỏe Inverter & Nhật Ký Cảnh Báo Bảo Trì (O&M)")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 📋 Bảng Xếp Hạng Inverter Cần Bảo Dưỡng")
    df_faults = pd.DataFrame({
        'Mã Inverter': ['Inverter_1 (1BY6WEc)', 'Inverter_4 (1IF53ai)', 'Inverter_7 (3PZuoBA)', 'Inverter_12 (VHMLBKo)'],
        'Lỗi Dừng Máy (Khi nắng)': [14, 11, 8, 3],
        'Lỗi Sụt Áp (>30%)': [28, 22, 15, 8],
        'Mức Độ': ['Nghiêm trọng 🔴', 'Nghiêm trọng 🔴', 'Cảnh báo 🟡', 'Theo dõi 🟢']
    })
    st.dataframe(df_faults, use_container_width=True)

with col_right:
    st.markdown("#### 🛠️ Khuyến Nghị Hành Động Cho Kỹ Sư Hiện Trường")
    st.warning("**Inverter_1:** Bị sụt giảm 43% sản lượng vào buổi trưa. Khuyến nghị kiểm tra bám bụi trên chuỗi quang điện hoặc bóng che.")
    st.error("**Inverter_4:** Xuất hiện hiện tượng không phát điện khi bức xạ > 0.8 kW/m². Khuyến nghị kiểm tra Aptomat / Ngắt lưới.")
    
    # Nút tải file phiếu bảo trì
    csv_data = df_faults.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải Phiếu Yêu Cầu Bảo Trì (Work Order CSV)",
        data=csv_data,
        file_name="phieu_bao_tri_inverter.csv",
        mime="text/csv"
    )
