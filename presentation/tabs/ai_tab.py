import streamlit as st
import pandas as pd
import time
import torch

def streamlit_search_node(search_engine, label_text, key_prefix):
    query = st.text_input(label_text, key=f"{key_prefix}_input", placeholder="Nhập tên nhân vật (ví dụ: Ngô Bảo Châu, Barack Obama)...")
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

def render_ai_tab(ai_service):
    st.markdown("""
        <div style="background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <h2 style="color: white; margin: 0;">🔮 Dự đoán & Gợi ý Liên kết AI (GNN Prediction)</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                Sử dụng mô hình Học sâu Đồ thị (GraphSAGE) để dự đoán các mối quan hệ ẩn hoặc liên kết tiềm năng trong tương lai giữa các thực thể.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not ai_service.model:
        st.error("⚠️ Mô hình AI chưa được tải thành công. Vui lòng kiểm tra file model.pt.")
        return

    sub_tab1, sub_tab2 = st.tabs(["🔮 Dự đoán Cặp đôi", "👑 Gợi ý Kết nối & Vợ/Chồng"])

    with sub_tab1:
        st.markdown("### 🔮 Dự đoán quan hệ giữa hai thực thể bất kỳ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Thực thể nguồn (A)")
            idx_a = streamlit_search_node(ai_service.search_engine, "Tìm thực thể A", "ai_src")
            
        with col2:
            st.subheader("👤 Thực thể đích (B)")
            idx_b = streamlit_search_node(ai_service.search_engine, "Tìm thực thể B", "ai_dst")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Cấu hình dự đoán
        zero_shot = st.checkbox(
            "🔓 Chế độ Linh hoạt (Loose Mode / Zero-shot)", 
            value=False,
            help="Strict Mode chỉ dự đoán các quan hệ có kiểu dữ liệu hợp lệ trong đồ thị gốc. Loose Mode cho phép thử tất cả các loại mối quan hệ."
        )
        
        if idx_a is not None and idx_b is not None:
            if st.button("🔮 Bắt đầu dự đoán bằng GNN Model", use_container_width=True):
                df_nodes = ai_service.search_engine.lookup
                
                name_a = df_nodes.at[idx_a, 'name']
                name_b = df_nodes.at[idx_b, 'name']
                
                src_type = df_nodes.at[idx_a, 'type']
                dst_type = df_nodes.at[idx_b, 'type']
                
                # Lấy local pyg id
                id_src = df_nodes.at[idx_a, 'pyg_id']
                id_dst = df_nodes.at[idx_b, 'pyg_id']
                
                mode = 'loose' if zero_shot else 'strict'
                
                with st.spinner("🤖 Đang trích xuất embeddings và lan truyền tiến (GNN Inference)..."):
                    start_time = time.time()
                    best_rel, max_score, results = ai_service.predictor.scan_relationship(
                        id_src, id_dst, src_type, dst_type, mode=mode
                    )
                    end_time = time.time()
                    
                st.success(f"⚡ Dự đoán hoàn tất trong **{end_time - start_time:.4f} giây**!")
                
                if not results:
                    st.warning("⚠️ Không tìm thấy loại quan hệ hợp lệ nào giữa hai thực thể này ở chế độ nghiêm ngặt.")
                else:
                    # Hiển thị quan hệ chính tốt nhất
                    st.markdown(f"""
                        <div style="background-color: #162447; border-left: 5px solid #00f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
                            <span style="color: #a2a8d3; font-size: 0.9rem; text-transform: uppercase; font-weight: bold;">Mối quan hệ có khả năng xảy ra nhất</span>
                            <h2 style="color: #00f3ff; margin: 5px 0;">{best_rel.upper()}</h2>
                            <span style="color: white; font-size: 1.5rem; font-weight: bold;">{max_score:.2%}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Sắp xếp và vẽ bảng phân phối xác suất
                    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
                    
                    st.markdown("### 📊 Phân phối xác suất tất cả quan hệ")
                    for rel, score in sorted_res:
                        col_rel, col_prog, col_val = st.columns([2, 5, 1])
                        
                        with col_rel:
                            st.write(f"**{rel}**")
                        with col_prog:
                            # Đảm bảo score trong khoảng [0, 1]
                            clipped_score = max(0.0, min(1.0, float(score)))
                            st.progress(clipped_score)
                        with col_val:
                            st.write(f"**{score:.2%}**")

    with sub_tab2:
        st.markdown("### 👑 Gợi ý đối tác / Vợ chồng tiềm năng")
        
        st.markdown("""
            Tính năng này kết hợp khả năng học đồ thị của **GNN** (Soft Constraint) với các **quy tắc thực tế** (Hard Constraints) 
            như khoảng cách tuổi tác tối đa và cấm kỵ huyết thống (ví dụ: không gợi ý cha mẹ, anh chị em làm vợ chồng).
        """)
        
        idx_src = streamlit_search_node(ai_service.search_engine, "Tìm thực thể muốn nhận gợi ý", "ai_rec")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            rel_type = st.selectbox(
                "Chọn mối quan hệ gợi ý",
                ["spouse (Vợ/Chồng)", "colleague (Đồng nghiệp)", "employer (Công ty)", "educated_at (Trường học)"]
            )
        with col_c2:
            top_k = st.slider("Số lượng gợi ý (Top K)", min_value=1, max_value=20, value=5)
            
        max_age_gap = st.slider("Chênh lệch tuổi tối đa (Spouse)", min_value=5, max_value=40, value=20)
        
        if idx_src is not None:
            if st.button("👑 Bắt đầu phân tích & Gợi ý kết nối", use_container_width=True):
                df_nodes = ai_service.search_engine.lookup
                src_name = df_nodes.at[idx_src, 'name']
                src_type = df_nodes.at[idx_src, 'type']
                src_id = df_nodes.at[idx_src, 'pyg_id']
                
                cleaned_rel = rel_type.split(" ")[0].strip()
                
                with st.spinner("🤖 Đang quét toàn bộ không gian đồ thị để tính toán gợi ý tốt nhất..."):
                    # Nếu là Spouse, chạy logic ràng buộc logic
                    if cleaned_rel == 'spouse' and src_type == 'human':
                        candidates = ai_service.predictor.recommend_top_k(
                            src_id,
                            top_k=100,  # Quét rộng để lọc
                            src_type='human',
                            dst_type='human',
                            rel_name='spouse'
                        )
                        
                        valid_candidates = []
                        src_meta = ai_service.search_engine.search_backward_pyg('human', src_id)
                        src_sex = src_meta.get('sex_or_gender')
                        src_year = src_meta.get('birth_year')
                        
                        def is_valid_year(y):
                            return y is not None and not pd.isna(y)
                            
                        for cand in candidates:
                            dst_id = cand['id']
                            dst_meta = ai_service.search_engine.search_backward_pyg('human', dst_id)
                            
                            cand['sex'] = dst_meta.get('sex_or_gender', 'Unknown')
                            cand['birth_year'] = dst_meta.get('birth_year')
                            cand['name'] = dst_meta.get('name', 'Unknown')
                            cand['desc'] = dst_meta.get('description', '')
                            cand['qid'] = dst_meta.get('id', '')
                            
                            # Cấm kỵ huyết thống
                            if ai_service.check_existing_connection(src_id, dst_id, ['sibling', 'father', 'mother','rev_sibling', 'rev_father', 'rev_mother']):
                                continue
                                
                            # Phạt chênh lệch tuổi tác
                            dst_year = cand['birth_year']
                            if is_valid_year(src_year) and is_valid_year(dst_year):
                                try:
                                    age_gap = abs(int(src_year) - int(dst_year))
                                    if age_gap > max_age_gap:
                                        cand['score'] *= 0.5  # Phạt nặng
                                except (ValueError, TypeError):
                                    pass
                                    
                            valid_candidates.append(cand)
                            
                            if len(valid_candidates) >= top_k:
                                break
                                
                        results = valid_candidates
                    else:
                        # Khác spouse, chạy gợi ý GNN thuần túy
                        results = ai_service.predictor.recommend_top_k(
                            src_id, 
                            top_k=top_k, 
                            src_type=src_type,
                            rel_name=cleaned_rel
                        )
                        
                        # Điền thêm thông tin hiển thị
                        for item in results:
                            meta = ai_service.search_engine.search_backward_pyg(item['type'], item['id'])
                            item['name'] = meta.get('name', 'Unknown')
                            item['birth_year'] = meta.get('birth_year', 'Unknown')
                            item['sex'] = meta.get('sex_or_gender', 'Unknown')
                            item['desc'] = meta.get('description', '')
                            item['qid'] = meta.get('id', '')
                
                st.success("🎉 Danh sách Gợi ý của bạn đã sẵn sàng!")
                
                # Hiển thị kết quả bằng các Card thông tin rất đẹp
                for i, item in enumerate(results):
                    st.markdown(f"""
                        <div style="background-color: #1a202c; padding: 15px; border-radius: 10px; border-left: 5px solid #8e2de2; margin-bottom: 12px; color: white;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 1.2rem; color: #f3e5f5;">#{i+1} {item['name']}</strong>
                                <span style="background-color: #4a00e0; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: bold; color: white;">Score: {item['score']:.4f}</span>
                            </div>
                            <div style="margin-top: 5px; font-size: 0.9rem; color: #e2e8f0;">
                                <strong>Thông tin:</strong> QID: {item['qid']} | Năm sinh: {item['birth_year']} | Giới tính: {item['sex']}
                            </div>
                            <div style="margin-top: 5px; font-size: 0.9rem; color: #cbd5e0; font-style: italic;">
                                <strong>Mô tả:</strong> {item['desc']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
