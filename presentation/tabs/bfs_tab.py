import streamlit as st
import time
from pyvis.network import Network
import tempfile
import os

def streamlit_search_node(search_engine, label_text, key_prefix):
    query = st.text_input(label_text, key=f"{key_prefix}_input", placeholder="Nhập tên nhân vật (ví dụ: Ngô Bảo Châu, Elon Musk)...")
    if not query:
        return None
        
    candidates, score = search_engine.search_best(query)
    if not candidates:
        st.warning("⚠️ Không tìm thấy thực thể nào khớp với từ khóa.")
        return None
        
    df_nodes = search_engine.lookup
    
    # Tạo danh sách hiển thị
    options = []
    id_map = {}
    for idx in candidates:
        name = df_nodes.at[idx, 'name']
        ntype = df_nodes.at[idx, 'type']
        desc = df_nodes.at[idx, 'description']
        qid = df_nodes.at[idx, 'id']
        
        display_label = f"👤 {name} ({ntype}) | {desc[:60]}... ({qid})" if ntype == 'human' else f"🏢 {name} ({ntype}) | {desc[:60]}... ({qid})"
        options.append(display_label)
        id_map[display_label] = idx
        
    if len(options) == 1:
        st.success(f"✅ Đã chọn thực thể: **{df_nodes.at[candidates[0], 'name']}**")
        return candidates[0]
    else:
        selected_label = st.selectbox(
            f"🔍 Có {len(options)} kết quả khớp với '{query}'. Chọn chính xác:",
            options,
            key=f"{key_prefix}_select"
        )
        return id_map[selected_label]

def draw_pyvis_graph(path_detail):
    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    
    # Cấu hình physics
    net.barnes_hut()
    
    for i, step in enumerate(path_detail):
        # Chọn màu và icon dựa trên loại thực thể
        if step['type'] == 'human':
            color = "#00e5ff"  # Xanh cyan neon cho con người
            shape = "dot"
        elif step['type'] == 'organization':
            color = "#ff9f00"  # Cam neon cho tổ chức
            shape = "triangle"
        else:
            color = "#39ff14"  # Xanh lá neon cho loại khác
            shape = "diamond"
            
        net.add_node(
            step['idx'], 
            label=step['name'], 
            title=f"Loại: {step['type']}\nQID: {step['qid']}",
            color=color, 
            shape=shape,
            size=30 if i in [0, len(path_detail)-1] else 20  # Điểm đầu/cuối to hơn
        )
        
        # Thêm cạnh
        if i < len(path_detail) - 1:
            next_step = path_detail[i+1]
            rel_label = step.get('next_rel', 'liên kết')
            
            # Ghi nhận chiều đi (incoming / outgoing)
            direction = step.get('direction', 'outgoing')
            if direction == "incoming":
                net.add_edge(
                    next_step['idx'], 
                    step['idx'], 
                    label=rel_label, 
                    color="#ff3366", # Đỏ hồng cho chiều ngược
                    width=2, 
                    arrows="to"
                )
            else:
                net.add_edge(
                    step['idx'], 
                    next_step['idx'], 
                    label=rel_label, 
                    color="#00ffcc", # Xanh ngọc cho chiều thuận
                    width=3, 
                    arrows="to"
                )
            
    # Lưu vào file tạm và hiển thị trong Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=420)
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

def render_bfs_tab(analysis_service):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <h2 style="color: white; margin: 0;">✈️ Sáu Bậc Xa Cách (Degrees of Separation)</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Nhập 2 nhân vật bất kỳ để tìm đường đi kết nối ngắn nhất giữa họ thông qua các liên kết mạng xã hội của Wikidata.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if not analysis_service.graph:
        st.error("⚠️ Chưa nạp được cơ sở dữ liệu đồ thị. Vui lòng chạy quy trình ETL trước.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Nhân vật thứ nhất (Source)")
        idx_a = streamlit_search_node(analysis_service.search_engine, "Tìm thực thể A", "src_p")
        
    with col2:
        st.subheader("👤 Nhân vật thứ hai (Target)")
        idx_b = streamlit_search_node(analysis_service.search_engine, "Tìm thực thể B", "dst_p")
        
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    
    if idx_a is not None and idx_b is not None:
        if st.button("🚀 Bắt đầu Tìm đường đi liên kết", use_container_width=True):
            with st.spinner("🔮 Đang phân tích mạng lưới hàng triệu mối quan hệ..."):
                start_time = time.time()
                result = analysis_service.find_connection(idx_a, idx_b)
                end_time = time.time()
                
            if result.get("success"):
                duration = end_time - start_time
                st.balloons()
                
                st.success(f"🎉 **Đã tìm thấy liên kết!** Thời gian tính toán: **{duration:.4f} giây**.")
                
                path_detail = result["path_detail"]
                
                # Hiển thị số bước (Degrees of Separation)
                degrees = sum(1 for step in path_detail if step['type'] == 'human') - 1
                degrees = max(0, degrees)
                st.metric(label="Bậc xa cách (Degrees of Separation)", value=f"{degrees} Bậc")
                
                # Chia 2 cột: Bên trái là timeline hiển thị văn bản đẹp, Bên phải là PyVis
                c_left, c_right = st.columns([1, 1])
                
                with c_left:
                    st.markdown("### 📋 Luồng Liên kết Chi tiết")
                    
                    for i, step in enumerate(path_detail):
                        icon = "👤" if step['type'] == 'human' else "🏢" if step['type'] == 'organization' else "🟢"
                        
                        st.markdown(f"""
                            <div style="background-color: #1a202c; padding: 12px; border-radius: 8px; border-left: 5px solid #00c4ff; margin-bottom: 5px; color: white;">
                                <strong style="font-size: 1.1rem;">{icon} {step['name']}</strong> <span style="color:#a0aec0; font-size:0.85rem;">[{step['qid']}]</span>
                                <br><span style="color:#cbd5e0; font-size:0.9rem;">Loại thực thể: {step['type']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if i < len(path_detail) - 1:
                            next_rel = step.get('next_rel', 'liên kết')
                            direction = step.get('direction', 'outgoing')
                            arrow = "⬇️" if direction == "outgoing" else "⬆️"
                            color = "#00ffcc" if direction == "outgoing" else "#ff3366"
                            dir_text = "chiều thuận" if direction == "outgoing" else "chiều ngược"
                            
                            st.markdown(f"""
                                <div style="text-align: center; margin: 5px 0; font-size: 0.95rem; color: {color};">
                                    <strong>{arrow} ({next_rel} - {dir_text})</strong>
                                </div>
                            """, unsafe_allow_html=True)
                
                with c_right:
                    st.markdown("### 🕸️ Sơ đồ Mạng lưới Tương tác")
                    draw_pyvis_graph(path_detail)
            else:
                st.error(f"❌ {result.get('message', 'Không tìm thấy đường đi giữa hai thực thể này.')}")
    else:
        st.info("💡 Vui lòng nhập tên và chọn chính xác 2 nhân vật ở phía trên để tiến hành tìm liên kết.")
