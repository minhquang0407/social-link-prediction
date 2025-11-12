import streamlit as st
import analytics_engine as ae
import json
import pandas as pd
import time
st.markdown(
    """
    <style>
    /* Nhắm vào "thân" (body) của sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6; /* Đổi màu nền sidebar (ví dụ: màu xám nhạt) */
    }

    /* Nhắm vào các nút bấm 'radio' trong sidebar */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        /* (Bạn có thể thêm CSS cho các nút radio ở đây) */
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Đổi font chữ của tiêu đề sidebar (ví dụ) */
    [data-testid="stSidebar"] .css-18e3th9 { 
        font-family: "Georgia", serif;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)#UI
st.sidebar.title("MENU ĐIỀU HƯỚNG")
st.sidebar.header("Phân tích Mạng xã hội")
st.sidebar.info("Phân tích mạng lưới xã hội")
choice = st.sidebar.radio(
	"Chọn một chức năng:",
	[
		"Trang chủ",
		"1. Tìm kiếm và Dự đoán",
		"2. Phân tích và Khám phá"
	],
	key = 'navigation'
)

st.sidebar.markdown("---") # Đường kẻ ngang
st.sidebar.markdown(
    "**Nhóm 3:**\n"
    "- Quân (Extractor)\n"
    "- Tân (Transformer/AI)\n"
    "- Quang (Loader/App)"
)	
if choice == "Trang chủ":
	st.title("Chào mừng đến với Trang chủ")
	st.write("Vui lòng chọn một chức năng từ thanh menu bên trái.")
	st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Social_graph.svg/1200px-Social_graph.svg.png")

elif choice == "1. Tìm kiếm và Dự đoán":
	tab1, tab2 = st.tabs([
		"1. ✈️ Sáu Bậc Xa cách", 
		"2. 🔮 Dự đoán Liên kết", 
	])
	with tab1:
		st.header("Kiểm chứng Sáu Bậc Xa Cách")
		col1, col2 = st.columns(2)
		with col1:
			name_a = st.text_input("Tên người 1")
		with col2:
			name_b = st.text_input("Tên người 2")
	

elif choice == "2. Phân tích và Khám phá":
	tab1, tab2 = st.tabs([
		"1. 📈 Phân tích Mạng lưới (Analytics)", 
		"2. 🔍 Khám phá Lân cận (Ego)"
	])
	with tab1:
			
		st.header("Phân tích Mạng lưới")
		
		st.write("Các chỉ số này được tính toán 'offline' trên toàn đồ thị")
			
		if 'analytics_done' not in st.session_state:
			st.session_state.analytics_done = False
		
		if st.button("Chạy Phân tích"):
			with st.spinner("Đang chạy tính toán... Vui lòng chờ 3 giây"):
				time.sleep(3)
			st.success("Tính toán hoàn tất!")
			st.session_state.analytics_done = True

		
		if st.session_state.analytics_done:
		#with open("data_output/analytics.json") as f:
		#analytics = json.load(f)
			st.subheader("📊 Thống kê Đường đi (Sáu Bậc Xa cách)")
		
			col1, col2, col3 = st.columns(3)
		
			col1.metric(
				label = "Số bậc Trung bình (AVG PATH)",
				value = 2
			)
		
			col2.metric(
				label = "Số bậc phổ biến (MODE PATH)",
				value = 3
			)

			col3.metric(
				label = "Đường kính (Diameter)",
				value = 4
			)
		
			st.divider()
		
			st.subheader("Phân phối Bậc (Degree Distribution)")
			
			#df_dist_degree = pd.DataFrame(
			#	analytics['degree_histogram'].items(),
			#	columns = ['Bậc', 'Số lượng']
			#).set_index('Bậc')
			#st.bar_chart(df_dist_degree)

			st.subheader("Phân phối Đường đi (Path Length Distribution)")
			# Vẽ biểu đồ 'path_length_histogram')
			st.divider()


			st.subheader("👑 Phân tích 'Quyền lực' (Centrality Top 5)")

			col_deg, col_bet, col_close, col_eig = st.columns(4)
		
			with col_deg:
				st.markdown("**1. Siêu Kết nối (Degree)**")

			with col_bet:
				st.markdown("**2. Môi giới (Betweenness)**")

			with col_close:
				st.markdown("**3. Trung tâm (Closeness)**")
        		
			with col_eig:
				st.markdown("**4. Ảnh hưởng (Eigenvector)**")









