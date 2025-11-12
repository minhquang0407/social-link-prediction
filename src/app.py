import streamlit as st
import time


def typewriter_effect(text: str, speed: float = 0.045):
    """
    Hàm này nhận một chuỗi (text) và "gõ" nó ra.
    'speed' là thời gian (giây) chờ giữa mỗi ký tự.
    """

    placeholder = st.empty()

    displayed_text = ""

    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(speed)
    placeholder.markdown(displayed_text)
#UI

if 'page' not in st.session_state:
    st.session_state.page = "Trang chủ"

st.sidebar.title("MENU ĐIỀU HƯỚNG")
st.sidebar.header("Phân tích Mạng xã hội")


if st.sidebar.button("Trang chủ"):
    st.session_state.page = "Trang chủ"

st.sidebar.markdown("---")
st.sidebar.markdown("# Các chức năng chính")
if st.sidebar.button("1. Tìm kiếm và Dự đoán"):
    st.session_state.page = "Tìm kiếm và Dự đoán"

if st.sidebar.button("2. Phân tích và Khám phá"):
    st.session_state.page = "Phân tích và Khám phá"


st.sidebar.markdown("---") # Đường kẻ ngang
st.sidebar.markdown(
    "**Nhóm 3:**\n"
    "- Quân (Extractor)\n"
    "- Tân (Transformer/AI)\n"
    "- Quang (Loader/App)"
)	
if st.session_state.page == "Trang chủ":
    typewriter_effect("# Chào mừng đến với Trang chủ")
    st.write("Vui lòng chọn một chức năng từ thanh menu bên trái.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Social_graph.svg/1200px-Social_graph.svg.png")

elif st.session_state.page == "Tìm kiếm và Dự đoán" :
    st.title("1. Tìm kiếm và Dự đoán")
    tab1, tab2 = st.tabs([
        "1. ✈️ Sáu Bậc Xa cách",
        "2. 🔮 Dự đoán Liên kết",
    ])
    with tab1:
        typewriter_effect("## Kiểm chứng Sáu Bậc Xa Cách", speed = 0.02)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tên người 1")
            name_a = st.text_input("#Hãy nhập vào đây!")
        with col2:
            st.markdown("#### Tên người 2")
            name_b = st.text_input("#Hãy nhập vào đây! ")
    with tab2:
        st.header("Dự đoán Liên kết")

elif st.session_state.page == "Phân tích và Khám phá":
    st.title("2. Phân tích và Khám phá")
    tab1, tab2 = st.tabs([
        "1. 📈 Phân tích Mạng lưới (Analytics)",
        "2. 🔍 Khám phá Lân cận (Ego)"
    ])
    with tab1:

        st.header("Phân tích Toàn bộ Mạng lưới")

        st.write("Các chỉ số này được tính toán 'offline' trên toàn đồ thị")

        if 'analytics_done' not in st.session_state:
            st.session_state.analytics_done = False

        if st.button("Chạy Phân tích"):
            with st.spinner("Đang chạy tính toán... Vui lòng chờ 3 giây"):
                time.sleep(3)
            st.success("Tính toán hoàn tất!")
            st.session_state.analytics_done = True
            if st.session_state.analytics_done:
                typewriter_effect("### 📊 Thống kê Đường đi (Sáu Bậc Xa cách)", speed=0.03)
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
                time.sleep(0.5)

                typewriter_effect("### 📊 Phân phối Bậc (Degree Distribution)", speed=0.03)
                #df_dist_degree = pd.DataFrame(
                #	analytics['degree_histogram'].items(),
                #	columns = ['Bậc', 'Số lượng']
                #).set_index('Bậc')
                #st.bar_chart(df_dist_degree)
                time.sleep(0.5)

                typewriter_effect("### 📊 Phân phối Đường đi (Path Length Distribution)", speed=0.03)                # Vẽ biểu đồ 'path_length_histogram')
                st.divider()
                time.sleep(0.5)

                typewriter_effect("### 👑 Phân tích 'Quyền lực' (Centrality Top 5)", speed=0.03)

                col_deg, col_bet, col_close, col_eig = st.columns(4)

                with col_deg:
                    st.markdown("**1. Siêu Kết nối (Degree)**")

                with col_bet:
                    st.markdown("**2. Môi giới (Betweenness)**")

                with col_close:
                    st.markdown("**3. Trung tâm (Closeness)**")

                with col_eig:
                    st.markdown("**4. Ảnh hưởng (Eigenvector)**")
    with tab2:
        st.header("Khám phá Lân cận (Ego Network)")








