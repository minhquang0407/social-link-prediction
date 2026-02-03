import streamlit as st
import time
import streamlit as st
from tabs.bfs_tab import render_bfs_tab
from tabs.ai_tab import render_ai_tab
from tabs.analytics_tab import render_analytics_tab
from tabs.ego_tab import render_ego_tab
from components.sidebar import render_sidebar
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_SCRIPT_DIR.parent

class AppRunner:
    def __init__(self, analysis_service, ai_service):
        self.analysis_service = analysis_service
        self.ai_service = ai_service

    def run(self):
        st.set_page_config(
            layout="wide",
            page_title="Social Network Analysis",
            page_icon="🕸️"
        )

        self._inject_custom_css()

        if 'page' not in st.session_state:
            st.session_state.page = "HOME"

        render_sidebar()

        self._render_main_content()


    def _inject_custom_css(self):
        st.markdown("""
                <style>
                /* Sidebar màu xanh */
                section[data-testid="stSidebar"] { background-color: #0004ffff; color: white; }
                section[data-testid="stSidebar"] * { color: white !important; }

                /* Nền chính xám nhạt */
                .stApp { background-color: #f0f2f6; color: #1a1a1a; }

                /* Tiêu đề Tab */
                .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                    font-size: 1.2rem;
                }
                </style>
                """, unsafe_allow_html=True)

    def _writer(self,text: str, speed: float = 0.03, key=None):
        if text not in st.session_state:
            placeholder = st.empty()
            displayed_text = ""
            for char in text:
                displayed_text += char
                placeholder.markdown(displayed_text + "▌")
                time.sleep(speed)
            placeholder.markdown(displayed_text)

            if text:
                st.session_state[text] = True
        else:
            st.markdown(text)
    def _render_main_content(self):
        page = st.session_state.page
        if page == "HOME":
            self._render_home()
        elif page == "SEARCH":
            st.title("1. Tìm kiếm & Phân tích")
            tab1, tab2 = st.tabs(["✈️ Sáu Bậc Xa cách", "📈 Phân tích Mạng lưới"])
            with tab1:
                render_bfs_tab(self.analysis_service)
            with tab2:
                render_analytics_tab(self.analysis_service)

        elif page == "AI":
            st.title("2. Dự đoán & Khám phá")
            tab1, tab2 = st.tabs(["🔮 Dự đoán Liên kết", "🔍 Khám phá Lân cận"])
            with tab1:
                render_ai_tab(self.ai_service)
            with tab2:
                render_ego_tab(self.analysis_service)

    def _render_home(self):
        self._writer("# 🕸️ Hệ thống Phân tích Mạng xã hội")
        st.info("Python cho Khoa học Dữ Liệu - Nhóm 3")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("""
            Chào mừng! Hệ thống này sử dụng dữ liệu từ **Wikidata** và công nghệ **Graph Neural Networks (GNN)** để:
            1.  Tìm đường đi ngắn nhất giữa hai người bất kỳ.
            2.  Dự đoán mối quan hệ tiềm năng trong tương lai.
            3.  Phân tích cấu trúc mạng lưới xã hội.
            """)
            if not self.analysis_service.graph:
                st.error("⚠️ CẢNH BÁO: Chưa tải được dữ liệu đồ thị. Vui lòng kiểm tra lại pipeline.")
        with col2:
            st.image("https://dist.neo4j.com/wp-content/uploads/example-viz.png",caption="Mô phỏng đồ thị mạng xã hội")