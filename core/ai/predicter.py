import pandas as pd
import torch
from core.interfaces import ILinkPredictor
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from tqdm import tqdm
import torch.amp
from config.settings import NODES_DATA_PATH
class Predictor(ILinkPredictor):
    """
    Lớp dùng để tính toán và cache vector nhúng (Z) của tất cả các node ('person'),
    sau đó thực hiện tìm kiếm node tương đồng (Link Prediction).
    """
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self.model.eval()
        self.model.to(device)
        self.embeddings = None
        self.connectivity_map = self._build_connectivity_map()
    @torch.no_grad()
    def _compute_all_embeddings(self, batch_size=128):
        """
        Chạy model 1 lần để lấy vector của TẤT CẢ các node.
        Hàm này dùng để cache vector Z.
        """

        self.model.eval()
        loader = NeighborLoader(
            data = self.data,
            input_nodes = None,
            num_neighbors= [20,10],
            shuffle = False,
            num_workers = 0,
            batch_size= batch_size
        )

        temp_embs = {nt: [] for nt in self.data.node_types}

        with torch.no_grad():
            pbar = tqdm(loader, desc = "Encoding Nodes")
            for batch in pbar:
                batch = batch.to(self.device)

                with torch.amp.autocast('cuda'):
                    z_dict = self.model.encoder(batch.x_dict, batch.edge_index_dict)

                for nt, z in z_dict.items():
                    if nt in batch and batch[nt].batch_size is not None:
                        num_target = batch[nt].batch_size
                        temp_embs[nt].append(z[:num_target].cpu())

        for nt, embs in temp_embs.items():
            if embs:
                self.embeddings[nt] = torch.cat(embs, dim=0)

    def _get_score(self, src_id, dst_id, src_type, rel, dst_type):
        if src_type not in self.embeddings or dst_type not in self.embeddings:
            return 0.0

        try:
            vec_a = self.embeddings[src_type][src_id].to(self.device).unsqueeze(0)
            vec_b = self.embeddings[dst_type][dst_id].to(self.device).unsqueeze(0)
        except IndexError:
            return 0.0

        key = f"{src_type}__{rel}__{dst_type}"
        if key in self.model.decoders:
            logits = self.model.decoders[key](vec_a, vec_b)
            return torch.sigmoid(logits).item()
        else:
            return 0.0

    def scan_relationship(self, id_a, id_b, src_type = 'human', dst_type = 'human'):
        results = {}
        max_score = -1
        best_rel = None

        for et in self.data.edge_types:
            s, r, d = et
            if s == src_type and d == dst_type and not r.startswith('rev_'):
                score = self._get_score(id_a, id_b, s, r, d)
                results[r] = score

                if score > max_score:
                    max_score = score
                    best_rel = r
        return best_rel, max_score, results

    @torch.no_grad()
    def recommend_top_k_with_rel(self, src_id, rel_name, top_k=10, src_type='human'):
        """
        Tìm Top-K node đích có khả năng liên kết cao nhất với src_id theo quan hệ rel_name.
        """
        if not self.is_ready:
            raise RuntimeError("Chưa chạy .precompute_embeddings()!")
        for et in self.data.edge_types:
            s, rel, d = et
            if s == src_type and rel == rel_name:
                dst_type = d
                break
        # 1. Xác định Decoder chuyên gia
        key = f"{src_type}__{rel_name}__{dst_type}"
        if key not in self.model.decoders:
            raise ValueError(f"Không tìm thấy mô hình cho quan hệ: {key}")

        decoder = self.model.decoders[key]

        # 2. Chuẩn bị Vector nguồn (Ông A)
        try:
            # Shape: [1, Hidden_Dim]
            vec_src = self.embeddings[src_type][src_id].view(1, -1).to(self.device)
        except IndexError:
            return [], []  # ID không tồn tại

        # 3. Lấy toàn bộ Vector đích (Tất cả mọi người)
        # Shape: [Num_Candidates, Hidden_Dim]
        # Lưu ý: candidates_emb đang ở CPU
        candidates_emb = self.embeddings[dst_type]
        num_candidates = candidates_emb.size(0)

        # 4. CHẠY BATCH INFERENCE (Để không cháy VRAM)
        # Vì chỉ là phép nhân ma trận đơn giản nên batch có thể rất to
        eval_batch_size = 4096
        all_scores = []

        # Duyệt qua từng cụm ứng viên
        for i in range(0, num_candidates, eval_batch_size):
            # Cắt batch ứng viên và đưa lên GPU
            batch_dst = candidates_emb[i: i + eval_batch_size].to(self.device)

            # Mở rộng vec_src để khớp kích thước batch
            # [1, H] -> [Batch_Size, H]
            batch_src = vec_src.expand(batch_dst.size(0), -1)

            # Tính điểm qua Decoder
            # Dùng AMP cho nhanh
            with torch.amp.autocast('cuda'):
                logits = decoder(batch_src, batch_dst)
                scores = torch.sigmoid(logits).view(-1)  # Ép về 1 chiều

            # Đưa về CPU ngay lập tức để tiết kiệm VRAM
            all_scores.append(scores.cpu())

        # 5. Nối lại thành 1 tensor điểm số khổng lồ
        final_scores = torch.cat(all_scores)

        # Gán điểm -1.0 cho chính bản thân mình (để không tự gợi ý mình)
        if src_type == dst_type:
            final_scores[src_id] = -1.0

        # 6. Lấy Top K (Hàm topk của PyTorch siêu nhanh)
        # values: Điểm số, indices: ID của người được gợi ý
        values, indices = torch.topk(final_scores, k=top_k)

        return indices.numpy(), values.numpy()


    def predict_link_score(self):
        """
        Tính toán điểm liên kết (link score) giữa hai vector và chuyển thành xác suất.
        """

    @torch.no_grad()
    def _build_connectivity_map(self):
        mapping = {}
        for src, rel, dst in self.data.edge_types:
            if rel.startswith('rev_'): continue

            if src not in mapping: mapping[src] = {}
            if dst not in mapping[src]: mapping[src][dst] = []

            mapping[src][dst].append(rel)
        return mapping

    @torch.no_grad()
    def recommend_top_k(self, src_id, top_k=10, src_type='human', dst_type=None):
        """
        Hàm gợi ý đa năng:
        - Nếu dst_type=None: Tìm Top-K trên TOÀN BỘ hệ thống (Global).
        - Nếu dst_type='...': Tìm Top-K chỉ trên loại node đó (Specific).

        Returns:
            List[Dict]: Danh sách kết quả đã sort.
            Mỗi item: {'id', 'type', 'relation', 'score'}
        """
        # 1. Kiểm tra đầu vào
        if not hasattr(self, 'embeddings') or not self.embeddings:
            raise RuntimeError("Chưa có Embeddings. Hãy chạy precompute trước.")

        if src_type not in self.embeddings: return []

        try:
            vec_src = self.embeddings[src_type][src_id].view(1, -1).to(self.device)
        except IndexError:
            return []  # ID nguồn không tồn tại

        # 2. Xác định phạm vi tìm kiếm (Target Groups)
        # target_groups dạng: {dst_type: [rel_name_1, rel_name_2]}
        target_groups = {}

        if dst_type is not None:
            # CASE A: Tìm kiếm cụ thể (VD: chỉ tìm 'human')
            if src_type in self.connectivity_map and dst_type in self.connectivity_map[src_type]:
                target_groups[dst_type] = self.connectivity_map[src_type][dst_type]
            else:
                return []  # Không có đường nối giữa src và dst này
        else:
            # CASE B: Tìm kiếm toàn cục (Global)
            if src_type in self.connectivity_map:
                target_groups = self.connectivity_map[src_type]
            else:
                return []

        print(f"🌍 Đang quét liên kết từ '{src_type} #{src_id}' đến {list(target_groups.keys())}...")

        global_candidates = []

        # 3. Vòng lặp chính: Duyệt qua từng loại Node Đích
        for target_type, rel_names in target_groups.items():

            if target_type not in self.embeddings: continue

            candidates_emb = self.embeddings[target_type]  # CPU Tensor
            num_dst = candidates_emb.size(0)

            # Tensor lưu Max Score cho mỗi node đích thuộc loại này
            # (Khởi tạo -1)
            type_max_scores = torch.full((num_dst,), -1.0, dtype=torch.float32)
            type_best_rels = [None] * num_dst  # Lưu tên quan hệ tốt nhất

            # 3.1. Max-Pooling qua các quan hệ (VD: Friend vs Colleague)
            for rel_name in rel_names:
                key = f"{src_type}__{rel_name}__{target_type}"
                if key not in self.model.decoders: continue

                decoder = self.model.decoders[key]

                # Batch Inference
                batch_size = 4096
                for i in range(0, num_dst, batch_size):
                    batch_dst = candidates_emb[i: i + batch_size].to(self.device)
                    # Expand src để khớp batch
                    batch_src = vec_src.expand(batch_dst.size(0), -1)

                    with torch.amp.autocast('cuda'):
                        logits = decoder(batch_src, batch_dst)
                        scores = torch.sigmoid(logits).view(-1).cpu()

                    # Cập nhật Max Score thủ công trên CPU
                    # (Logic: Nếu score mới > score cũ thì cập nhật score và relation)
                    # Dùng slicing để gán cho nhanh
                    current_slice = slice(i, i + len(scores))

                    # Tạo mask cho những điểm tốt hơn
                    mask = scores > type_max_scores[current_slice]

                    # Update Score
                    type_max_scores[current_slice] = torch.where(
                        mask, scores, type_max_scores[current_slice]
                    )

                    # Update Relation Name (Cần loop vì đây là list string)
                    indices = torch.nonzero(mask).flatten() + i
                    for idx in indices:
                        type_best_rels[idx.item()] = rel_name

            # 3.2. Xử lý Self-loop (Không gợi ý chính mình)
            if src_type == target_type:
                type_max_scores[src_id] = -1.0

            # 3.3. Lấy Top-K cục bộ (của loại node này)
            # Lấy nhiều hơn top_k một chút để khi gộp Global không bị thiếu
            k_local = min(top_k, num_dst)
            vals, indices = torch.topk(type_max_scores, k=k_local)

            # Đưa vào danh sách tổng
            for score, idx in zip(vals, indices):
                if score > 0.0:
                    idx = idx.item()
                    global_candidates.append({
                        'score': score.item(),
                        'id': idx,
                        'type': target_type,
                        'relation': type_best_rels[idx]
                    })

        # 4. Sắp xếp Global và lấy Top-K cuối cùng
        # Sort giảm dần theo score
        global_candidates.sort(key=lambda x: x['score'], reverse=True)

        return global_candidates[:top_k]
