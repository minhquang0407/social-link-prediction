import streamlit as st
import time
from pyvis.network import Network
import tempfile
import os

def streamlit_search_node(search_engine, label_text, key_prefix):
    query = st.text_input(label_text, key=f"{key_prefix}_input", placeholder="Nhập tên nhân vật cần vẽ lân cận...")
    if not query:
        return None
        
    candidates, score = search_engine.search_best(query)
    if not candidates:
        st.warning("⚠️ Không tìm thấy thực thể nào khớp với từ khóa.")
        return None
        
    df_nodes = search_engine.lookup
    
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

def draw_ego_network(g, ego_idx, max_neighbors=65):
    # Lấy láng giềng bậc 1
    neighbors = g.neighbors(ego_idx, mode='all')
    total_neighbors_count = len(neighbors)
    
    # Giới hạn số lượng hiển thị để UI mượt mà
    if len(neighbors) > max_neighbors:
        neighbors = neighbors[:max_neighbors]
        
    all_nodes = [ego_idx] + neighbors
    subg = g.subgraph(all_nodes)
    
    # Thiết lập pyvis network
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    net.barnes_hut()
    
    # Thêm nút vào mạng
    for v in subg.vs:
        original_idx = all_nodes[v.index]
        name = v['label'] if v['label'] else 'Unknown'
        ntype = v['type']
        qid = v['name']
        
        if original_idx == ego_idx:
            # Ego (Thực thể trung tâm)
            color = "#ff007f"  # Hồng neon rực rỡ
            shape = "dot"
            size = 40
        else:
            # Alters
            if ntype == 'human':
                color = "#00e5ff"  # Xanh dương cyan
                shape = "dot"
                size = 20
            elif ntype == 'organization':
                color = "#ff9f00"  # Cam neon
                shape = "triangle"
                size = 20
            else:
                color = "#39ff14"  # Xanh lá neon
                shape = "diamond"
                size = 18
                
        net.add_node(
            original_idx, 
            label=name, 
            title=f"Loại: {ntype}\nQID: {qid}",
            color=color, 
            shape=shape,
            size=size
        )
        
    # Thêm cạnh vào mạng
    for e in subg.es:
        src_local = e.source
        dst_local = e.target
        
        src_original = all_nodes[src_local]
        dst_original = all_nodes[dst_local]
        
        rel_label = e['relationship_label'] if e['relationship_label'] else 'liên kết'
        
        # Nếu cạnh nối trực tiếp với Ego thì tô đậm màu sáng
        if src_original == ego_idx or dst_original == ego_idx:
            color = "#ffffff"
            width = 3
        else:
            color = "#4a5568"  # Cạnh giữa các alter xám mờ
            width = 1
            
        net.add_edge(
            src_original, 
            dst_original, 
            label=rel_label, 
            color=color, 
            width=width, 
            arrows="to"
        )
        
    # Vẽ và render
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=520)
            
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
        
    return total_neighbors_count

def render_ego_tab(analysis_service):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #F3904F 0%, #3B4371 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <h2 style="color: white; margin: 0;">🔍 Khám phá Mạng lưới Lân cận (Ego Network Explorer)</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Tìm kiếm một thực thể để hiển thị sơ đồ mạng lưới quan hệ trực tiếp (bậc 1) xung quanh họ.
            </p>
        </div>
    """, unsafe_allow_html=True)

    g = analysis_service.graph
    if g is None:
        st.error("⚠️ Chưa có dữ liệu đồ thị để phân tích. Vui lòng kiểm tra dữ liệu đầu vào.")
        return

    st.markdown("### 🔍 Nhập tên thực thể cần khám phá")
    ego_idx = streamlit_search_node(analysis_service.search_engine, "Tìm thực thể", "ego_search")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if ego_idx is not None:
        df_nodes = analysis_service.search_engine.lookup
        ego_name = df_nodes.at[ego_idx, 'name']
        ego_type = df_nodes.at[ego_idx, 'type']
        ego_qid = df_nodes.at[ego_idx, 'id']
        ego_desc = df_nodes.at[ego_idx, 'description']
        
        # Hiển thị thông tin profile
        icon = "👤" if ego_type == 'human' else "🏢" if ego_type == 'organization' else "🟢"
        
        st.markdown(f"""
            <div style="background-color: #1a202c; padding: 15px; border-radius: 8px; border-left: 5px solid #ff007f; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0; color: #ff007f;">{icon} {ego_name}</h3>
                <p style="margin: 5px 0 0 0; color: #cbd5e0;">QID: {ego_qid} | Loại: {ego_type}</p>
                <p style="margin: 5px 0 0 0; color: #a0aec0; font-style: italic;">Mô tả: {ego_desc}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🕸️ Bắt đầu dựng Sơ đồ mạng lưới lân cận", use_container_width=True):
            with st.spinner("🤖 Đang tính toán cấu trúc subgraph lân cận..."):
                start_t = time.time()
                total_neighbors = draw_ego_network(g, ego_idx)
                end_t = time.time()
                
            st.success(f"⚡ Đã vẽ xong đồ thị lân cận trong **{end_t - start_t:.4f} giây**!")
            
            # Hiển thị thống kê
            st.write(f"📊 Thực thể **{ego_name}** có tổng cộng **{total_neighbors}** liên kết trực tiếp (Bậc 1) trong đồ thị.")
            if total_neighbors > 65:
                st.info("💡 Lưu ý: Do số lượng liên kết quá lớn, sơ đồ chỉ hiển thị 65 liên kết tiêu biểu để tối ưu hóa trực quan.")
    else:
        st.info("💡 Vui lòng nhập tên thực thể ở phía trên để tiến hành dựng Ego Network.")
