import torch
import pickle
from core.ai import Predictor
import pandas as pd
from config.settings import NODES_DATA_PATH
from core.logic import RapidFuzzySearch
class AIService:
    def __init__(self, G_full, model, data, engine:RapidFuzzySearch):
        self.G_full = G_full
        self.model = model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("AI: Đang khởi tạo bộ máy dự đoán!",flush=True)
        # self.lookup = pd.read_parquet(NODES_DATA_PATH, engine='fastparquet')
        # self.lookup  = self.lookup.reset_index(drop = False)
        # self.lookup = self.lookup.set_index(['type','pyg_id'])
        self.search_engine = engine
        self.predictor = Predictor(model, data, self.device)
        print("AI: Sẵn sàng!",flush=True)

    def predict_link_score(self, name_src, name_dst):
        src_type, id_src = self.search_engine.search_forward_pyg(name_src)
        dst_type, id_src = self.search_engine.search_forward_pyg(name_dst)

        best_rel, max_score, results = self.predictor.scan_relationship(id_a, id_b, src_type, dst_type)

        print(f"║ PHÂN TÍCH QUAN HỆ: {src_type} #{id_src} vs {dst_type} #{id_dst} ║")

        # Sắp xếp kết quả
        sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print(f"\n➤ DỰ ĐOÁN CHÍNH: [{best_rel.upper()}] (Độ tin cậy: {max_score:.2%})")
        print("-" * 50)

        for rel, score in sorted_res:
            # Logic hiển thị thanh bar
            bar_len = int(score * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)

            # Logic màu sắc (Text indication)
            status = " "
            if score > 0.8:
                status = "(Rất cao)"
            elif score > 0.5:
                status = "(Có thể)"

            print(f"  {rel:<15} : {score:.4f}  {bar} {status}")

        print("-" * 50)

    def recommendations_with_rel(self, src_name, rel_name, top_k=10):
        """In báo cáo Top-K đẹp mắt."""
        type, src_id = self.search_engine.search_forward_pyg(src_name)
        print(f"\nĐang tìm Top {top_k} '{rel_name.upper()}' cho User #{src_id}...")

        ids, scores = self.predictor.recommend_top_k_with_rel(src_id, rel_name, top_k, type)

        print(f"╔══════════════════════════════════════╗")
        print(f"║ DANH SÁCH GỢI Ý ({rel_name})             ║")
        print(f"╠══════════════════════════════════════╣")

        for rank, (uid, score) in enumerate(zip(ids, scores)):
            # Vẽ thanh bar
            bar_len = int(score * 20)
            bar = "▓" * bar_len + "░" * (20 - bar_len)

            print(f"║ #{rank + 1:02d} User {uid:<6} | {score:.4f} {bar} ║")

        print(f"╚══════════════════════════════════════╝")

    def print_recommendations(self, src_id, top_k=10, src_type='human', dst_type=None):
        results = self.recommend(src_id, top_k, src_type, dst_type)

        scope = f"LOẠI: {dst_type.upper()}" if dst_type else "TOÀN BỘ (GLOBAL)"
        print(f"\n╔══════════════════════════════════════════════════════════════╗")
        print(f"║ GỢI Ý KẾT NỐI CHO {src_type.upper()} #{src_id} | PHẠM VI: {scope:<12} ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")

        for i, item in enumerate(results):
            score = item['score']
            target = f"{item['type']} #{item['id']}"
            rel = f"[{item['relation'].upper()}]"

            bar_len = int(score * 15)
            bar = "▓" * bar_len + "░" * (15 - bar_len)

            print(f"║ #{i + 1:02d} {target:<18} | {rel:<15} | {score:.4f} {bar} ║")

        print(f"╚══════════════════════════════════════════════════════════════╝")
