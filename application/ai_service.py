import torch
import pickle
from core.ai import Predictor
import pandas as pd
from config.settings import NODES_DATA_PATH
from core.interfaces import ISearchEngine
class AIService:
    def __init__(self,  model, metadata, embeddings, engine:ISearchEngine, device = 'cuda'):
        self.model = model
        self.embeddings = embeddings
        self.metadata = metadata
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("AI: Đang khởi tạo bộ máy dự đoán!",flush=True)
        self.search_engine = engine
        self.predictor = Predictor(model,metadata, embeddings)
        with open(f'data_output/predicting/adjacency.pkl', 'rb') as f:
            self.adj = pickle.load(f)
        print("AI: Sẵn sàng!",flush=True)

    def predict_link_score(self, name_src, name_dst, zero_shot=False):
        src_type, id_src = self.search_engine.search_forward_pyg(name_src)
        dst_type, id_dst = self.search_engine.search_forward_pyg(name_dst)

        mode = 'loose' if zero_shot else 'strict'

        best_rel, max_score, results = self.predictor.scan_relationship(
            id_src, id_dst, src_type, dst_type, mode=mode
        )

        print(f"║ PHÂN TÍCH QUAN HỆ:{name_src} {src_type} #{id_src} vs {name_dst} {dst_type} #{id_dst} ║")

        # Sắp xếp kết quả
        sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print(f"\n➤ DỰ ĐOÁN CHÍNH: [{best_rel.upper()}] (Độ tin cậy: {max_score:.2%})")
        print("-" * 50)

        for rel, score in sorted_res:
            bar_len = int(score * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)

            status = " "
            if score > 0.8:
                status = "(Rất cao)"
            elif score > 0.5:
                status = "(Có thể)"

            print(f"  {rel:<15} : {score:.4f}  {bar} {status}")

        print("-" * 50)

    def recommendations_relationship(self, src_name, rel_name, top_k=10):
        """
        Gợi ý theo quan hệ cụ thể (VD: Tìm 'acted_in' cho Tom Cruise)
        """
        src_type, src_id = self.search_engine.search_forward_pyg(src_name)
        print(f"\nTìm Top {top_k} '{rel_name}' cho {src_name}...")

        # FIX: Gọi hàm recommend thống nhất
        results = self.predictor.recommend_top_k(
            src_id, top_k, src_type,
            dst_type=None,  # Tự động tìm dst_type dựa trên rel_name
            rel_name=rel_name
        )

        print(f"║ GỢI Ý CHO {rel_name.upper()} ║")
        for i, item in enumerate(results):
            # item: {'id', 'type', 'relation', 'score'}
            # Cần lấy tên để hiển thị đẹp
            meta = self.search_engine.search_backward_pyg(item['type'], item['id'])
            name = meta.get('name', 'Unknown')
            print(f"#{i + 1:02d} {name[:20]:<20} | {item['score']:.4f}")

    def recommendations(self, src_name, top_k=10, dst_type=None):
        """Gợi ý chung hoặc theo loại đích"""
        src_type, src_id = self.search_engine.search_forward_pyg(src_name)

        # FIX: Gọi hàm recommend thống nhất
        results = self.predictor.recommend_top_k(src_id, top_k, src_type, dst_type=dst_type)

        scope = dst_type.upper() if dst_type else "GLOBAL"
        print(f"║ GỢI Ý ({scope}) CHO {src_name} ║")
        for item in results:
            meta = self.search_engine.search_backward_pyg(item['type'], item['id'])
            print(f"{meta.get('name')} - [{item['relation']}] - {item['score']:.2f}")

    def predict_spouse_with_constraints(self, src_name, top_k=5, max_age_gap=20):
        """
        Dự đoán vợ chồng có kiểm tra ràng buộc logic (Hard Constraints).
        """
        # ---------------------------------------------------------
        # BƯỚC 1: LẤY ỨNG VIÊN TỪ GNN (Soft Constraint)
        # ---------------------------------------------------------
        # Lấy Top-50 hoặc Top-100 để có dư địa mà lọc
        # Bắt buộc dst_type='human' (Ràng buộc 1)
        src_type, src_id = self.search_engine.search_forward_pyg(src_name)
        candidates = self.predictor.recommend_top_k(
            src_id,
            top_k=100,  # Lấy dư để lọc
            src_type='human',
            dst_type='human',
            rel_name='spouse'
        )

        valid_spouses = []
        # Lấy thông tin năm sinh của nguồn (cần lookup dict lưu bên ngoài)
        # Giả sử bạn có self.node_metadata['human'][id] = {'birthYear': 1990}
        src_meta = self.search_engine.search_backward_pyg('human', src_id)
        src_sex = src_meta.get('sex_or_gender')
        src_year = src_meta.get('birth_year')

        # ---------------------------------------------------------
        # BƯỚC 2: HẬU XỬ LÝ (Hard Constraints)
        # ---------------------------------------------------------
        def is_valid_year(y):
            return y is not None and not pd.isna(y)
        for cand in candidates:
            dst_id = cand['id']
            dst_meta = self.search_engine.search_backward_pyg('human', dst_id)
            cand['sex'] = dst_meta.get('sex_or_gender', 'Unknown')
            cand['birth_year'] = dst_meta.get('birth_year')
            cand['name'] = dst_meta.get('name')
            # --- Ràng buộc 1: Khoảng cách tuổi tác ---
            # Nếu cả 2 đều có năm sinh, kiểm tra chênh lệch
            dst_year = cand['birth_year']
            if is_valid_year(src_year) and is_valid_year(dst_year):
                try:
                    age_gap = abs(int(src_year) - int(dst_year))

                    if age_gap > max_age_gap:
                        cand['score'] *= 0.5
                except (ValueError, TypeError):
                    pass

            # --- Ràng buộc 2: Quan hệ huyết thống (Taboo) ---
            # Kiểm tra xem trong đồ thị hiện tại đã có cạnh cấm kỵ chưa
            if self.check_existing_connection(src_id, dst_id, ['sibling', 'father', 'mother','rev_sibling', 'rev_father', 'rev_mother']):
                continue

            # Nếu vượt qua mọi cửa ải -> Chấp nhận
            valid_spouses.append(cand)
            # Đủ số lượng thì dừng
            if len(valid_spouses) >= top_k:
                break
        print(f"║ GỢI Ý VỢ CHỒNG CHO {src_type.upper()} #{src_id} | THÔNG TIN: {src_year}/{src_sex} ║")
        print(f"║ ỨNG VIÊN VỢ/CHỒNG CHO {src_name} ║")
        for item in valid_spouses:
            print(f"{item['name']} ({item['birth_year']}/{item['sex']}) - Score: {item['score']:.4f}")


    def has_edge(self, src_idx, dst_idx, src_type, rel, dst_type):
        """Kiểm tra cạnh tồn tại (Input là số nguyên)"""
        key = f"{src_type}__{rel}__{dst_type}"
        if key not in self.adj: return False
        return (src_idx, dst_idx) in self.adj[key] or (dst_idx, src_idx) in self.adj[key]

    def check_existing_connection(self, id_a, id_b, taboo_rels):
        """
        Kiểm tra nhanh xem giữa A và B đã có cạnh nào trong danh sách cấm chưa.
        """
        # Đây là code giả định, bạn cần implement dựa trên data thực tế
        # Check xuôi: A -> taboo -> B
        for rel in taboo_rels:
            if self.has_edge(id_a, id_b, 'human', rel, 'human'):
                return True
            if self.has_edge(id_b, id_a, 'human', rel, 'human'):
                return True
        return False