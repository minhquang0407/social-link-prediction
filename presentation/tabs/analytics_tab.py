import streamlit as st
import pandas as pd
import numpy as np
import time

def render_analytics_tab(analysis_service):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <h2 style="color: white; margin: 0;">📈 Phân tích & Thống kê Mạng lưới Đồ thị (Network Analytics)</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Khám phá các đặc trưng cấu trúc đồ thị mạng xã hội Wikidata bao gồm phân bố liên kết, mật độ và xếp hạng các thực thể trung tâm.
            </p>
        </div>
    """, unsafe_allow_html=True)

    g = analysis_service.graph
    if g is None:
        st.error("⚠️ Chưa có dữ liệu đồ thị để phân tích. Vui lòng kiểm tra dữ liệu đầu vào.")
        return

    with st.spinner("📊 Đang phân tích cấu trúc đồ thị toàn cục..."):
        # 1. Tính toán các chỉ số cơ bản
        n_vertices = g.vcount()
        n_edges = g.ecount()
        density = g.density()
        
        # Thành phần liên thông (sử dụng weak connection cho đồ thị có hướng)
        components = g.connected_components(mode='weak')
        n_components = len(components)
        largest_comp_size = len(components.giant().vs)

    # Hiển thị 4 cột chỉ số
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Tổng số Đỉnh (Nodes)", value=f"{n_vertices:,}")
    with col2:
        st.metric(label="Tổng số Cạnh (Edges)", value=f"{n_edges:,}")
    with col3:
        st.metric(label="Mật độ Đồ thị (Density)", value=f"{density:.6f}")
    with col4:
        st.metric(label="Thành phần Liên thông", value=f"{n_components:,}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chia 2 cột chính cho đồ thị và xếp hạng
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("### 📊 Phân bố Bậc Đồ thị (Degree Distribution)")
        st.write("""
            Biểu đồ dưới đây thể hiện số lượng đỉnh có số lượng liên kết (bậc) tương ứng. 
            Mạng xã hội thực tế thường tuân theo **Quy luật lũy thừa (Power Law)**: hầu hết các nút có bậc rất nhỏ, 
            chỉ một số ít nút (Hubs) có bậc cực lớn.
        """)
        
        # Tính bậc của các đỉnh
        degrees = g.degree()
        
        # Tạo histogram cho bậc (giới hạn từ 1 đến 50 để dễ quan sát phần lớn nút)
        counts, bin_edges = np.histogram(degrees, bins=range(1, 52))
        
        # Đóng gói dữ liệu hiển thị
        df_deg = pd.DataFrame({
            "Bậc liên kết (Degree)": [f"{int(bin_edges[i])}" for i in range(len(counts))],
            "Số lượng nút (Nodes Count)": counts
        }).set_index("Bậc liên kết (Degree)")
        
        st.area_chart(df_deg, color="#11998e", use_container_width=True)

    with c_right:
        st.markdown("### 🏆 Top 10 Thực thể Ảnh hưởng nhất (PageRank)")
        st.write("""
            **PageRank Centrality** đo lường tầm ảnh hưởng và uy tín của một nút dựa trên chất lượng và số lượng liên kết đến nó.
            Các thực thể có PageRank cao nhất thường là những nhân vật trung tâm hoặc các tổ chức lớn.
        """)
        
        # Nút bấm tính PageRank để tránh tự động chạy tốn tài nguyên
        if st.checkbox("🔥 Xem Xếp hạng Ảnh hưởng PageRank", value=True):
            with st.spinner("🤖 Đang tính toán ma trận PageRank..."):
                start_p = time.time()
                pr = g.pagerank(damping=0.85)
                end_p = time.time()
                
            # Tạo DataFrame kết quả PageRank
            nodes_labels = g.vs['label']
            nodes_types = g.vs['type']
            nodes_qids = g.vs['name']
            
            df_pr = pd.DataFrame({
                "name": nodes_labels,
                "type": nodes_types,
                "qid": nodes_qids,
                "score": pr
            })
            
            # Sắp xếp lấy Top 10
            top_10 = df_pr.sort_values(by="score", ascending=False).head(10)
            
            st.write(f"⏱️ Tính toán hoàn tất trong: **{end_p - start_p:.4f} giây**.")
            
            # Vẽ bảng Top 10 tùy biến HTML cực đẹp
            for idx, row in enumerate(top_10.itertuples()):
                icon = "👤" if row.type == 'human' else "🏢" if row.type == 'organization' else "🟢"
                score_scaled = row.score * 1000  # Nhân hệ số để dễ nhìn phần trăm
                
                st.markdown(f"""
                    <div style="background-color: #1a202c; padding: 10px 15px; border-radius: 8px; border-left: 5px solid #38ef7d; margin-bottom: 8px; color: white; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="font-size: 1.05rem;">#{idx+1} {icon} {row.name}</strong> 
                            <span style="color:#a0aec0; font-size:0.8rem;">[{row.qid}]</span>
                            <br><span style="color:#cbd5e0; font-size:0.85rem; text-transform: capitalize;">Loại: {row.type}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color:#38ef7d; font-weight:bold; font-size:1.1rem;">{score_scaled:.4f}</span>
                            <br><span style="color:#a0aec0; font-size:0.75rem;">Chỉ số uy tín</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
