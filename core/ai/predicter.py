import os

import torch
import torch.amp
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from config.settings import PYG_DATA_PATH, PREDICT_DATA_PATH
from infrastructure.repositories import PyGDataRepository
from core.interfaces import ILinkPredictor


class Predictor(ILinkPredictor):
    def __init__(self, model, data = None,  metadata = None, embeddings=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        if data is not None:
            self.data = data
            self.metadata = data.metadata()
        elif metadata is not None:
            self.metadata = metadata
        self.model.eval()
        self.model.to(self.device)

        if embeddings is None and data is not None:
            if os.path.exists(str(PREDICT_DATA_PATH)):
                self.embeddings = torch.load(str(PREDICT_DATA_PATH))
            else:
                self.embeddings = self._compute_all_embeddings()
        else:
            self.embeddings = embeddings

        self.connectivity_map = self._build_connectivity_map()
        self.BIOLOGICAL_RELS = {
            'spouse', 'sibling', 'father', 'mother', 'child',
            'student_of'
        }
        self.HUMAN_SRC_ONLY = {
            'educated_at', 'student_of'
        }
    @torch.no_grad()
    def _compute_all_embeddings(self, batch_size=512):
        """Tính và trả về Embeddings (Không lưu attribute để tránh side-effect)"""
        if self.data is None:
            repo = PyGDataRepository(PYG_DATA_PATH)
            data = repo.load_data()
        else: data = self.data
        embeddings = {}

        print(f"🚀 Computing Embeddings on {self.device}...")

        for node_type in data.node_types:
            if data[node_type].num_nodes == 0: continue

            loader = NeighborLoader(
                data,
                num_neighbors=[20, 10],
                input_nodes=node_type,
                shuffle=False,
                num_workers=0,
                batch_size=batch_size
            )

            all_embs = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Enc {node_type}", leave=False):
                    batch = batch.to(self.device)
                    with torch.amp.autocast('cuda'):
                        z_dict = self.model.encoder(batch.x_dict, batch.edge_index_dict)

                    if node_type in z_dict:
                        bs = batch[node_type].batch_size
                        all_embs.append(z_dict[node_type][:bs].cpu())

            if all_embs:
                embeddings[node_type] = torch.cat(all_embs, dim=0)

        # Save 1 lần sau khi xong hết
        torch.save(embeddings, str(PREDICT_DATA_PATH))
        return embeddings

    @torch.no_grad()
    def _build_connectivity_map(self):
        # Dùng metadata (edge_types) được truyền vào init thay vì self.edge_types
        node_types, edge_types = self.metadata
        mapping = {}
        for src, rel, dst in edge_types:
            if src not in mapping: mapping[src] = {}
            if dst not in mapping[src]: mapping[src][dst] = []
            mapping[src][dst].append(rel)
        return mapping

    def scan_relationship(self, id_a, id_b, src_type, dst_type, mode='strict'):
        """
        Args:
            mode (str):
                - 'strict': Chỉ check các quan hệ có trong Metadata (Human-Org -> work_at).
                - 'loose': Check TẤT CẢ quan hệ mà model biết (Human-Org -> thử cả spouse, member_of...).
                           Dùng cho Zero-shot / Vay mượn.
        """
        results = {}
        max_score = -1
        best_rel = None

        # 1. Xác định danh sách quan hệ cần kiểm tra
        candidate_rels = set()

        if mode == 'strict':
            # Cách cũ: Chỉ lấy những gì schema cho phép
            candidate_rels = set(self.connectivity_map.get(src_type, {}).get(dst_type, []))
        else:
            # Cách mới (Zero-shot): Lấy TOÀN BỘ các quan hệ model đã học
            # Duyệt qua keys của decoder để trích xuất tên quan hệ
            for key in self.model.decoders.keys():
                rel_name = key.strip('_')
                if '__' in key.strip('_'): rel_name = key.split('__')[1]
                candidate_rels.add(rel_name)



        # 2. Duyệt và Dự đoán
        for rel in candidate_rels:
            if rel.startswith('rev_'): continue

            # --- [BỘ LỌC NGỮ NGHĨA - SEMANTIC FILTER] ---

            # Luật 1: Quan hệ Sinh học (Vợ chồng, anh em...)
            if rel in self.BIOLOGICAL_RELS:
                if src_type != 'human' or dst_type != 'human':
                    continue

            # Luật 2: Quan hệ Hành vi con người (Tác giả, Đạo diễn...)
            if rel in self.HUMAN_SRC_ONLY:
                if src_type != 'human':
                    continue
            score = self._get_score_fast(id_a, id_b, src_type, rel, dst_type)

            if score > 0.001:
                results[rel] = score
                if score > max_score:
                    max_score = score
                    best_rel = rel

        return best_rel, max_score, results

    def _get_score_fast(self, src_id, dst_id, src_type, rel, dst_type):
        """Helper function tính điểm 1 cạnh"""
        if rel.startswith('rev_'): rel = rel.replace('rev_', '')

        # Giả sử decoder lưu theo key dạng "__rel__" như bạn định nghĩa
        key = f"__{rel}__"

        if key not in self.model.decoders: return 0.0

        try:
            vec_a = self.embeddings[src_type][src_id].to(self.device).view(1, -1)
            vec_b = self.embeddings[dst_type][dst_id].to(self.device).view(1, -1)
            logits = self.model.decoders[key](vec_a, vec_b)
            return torch.sigmoid(logits).item()
        except:
            return 0.0

    @torch.no_grad()
    def recommend_top_k(self, src_id, top_k=10, src_type='human', dst_type=None, rel_name=None):
        """
        HÀM GỢI Ý THỐNG NHẤT (UNIFIED RECOMMENDATION)
        Xử lý cả 3 trường hợp:
        1. Có rel_name -> Tìm theo quan hệ cụ thể.
        2. Có dst_type -> Tìm theo loại đích (Max-pool qua các quan hệ).
        3. Không có gì -> Tìm toàn cục (Global).
        """
        if src_type not in self.embeddings: return []

        try:
            vec_src = self.embeddings[src_type][src_id].view(1, -1).to(self.device)
        except IndexError:
            return []

        # 1. Lên kế hoạch tìm kiếm (Search Plan)
        search_plan = {}  # {dst_type: [rel1, rel2]}

        if rel_name:
            # Case 1: Tìm theo quan hệ cụ thể
            node_types, edge_types = self.metadata
            for s, r, d in edge_types:
                if s == src_type and r == rel_name:
                    if dst_type is None or dst_type == d:
                        if d not in search_plan: search_plan[d] = []
                        search_plan[d].append(r)
        elif dst_type:
            # Case 2: Tìm theo loại đích
            rels = self.connectivity_map.get(src_type, {}).get(dst_type, [])
            if rels: search_plan[dst_type] = rels
        else:
            # Case 3: Global
            search_plan = self.connectivity_map.get(src_type, {})

        global_candidates = []
        eval_batch_size = 4096

        # 2. Thực thi
        for target_type, rels in search_plan.items():
            if target_type not in self.embeddings: continue

            candidates_emb = self.embeddings[target_type]
            num_candidates = candidates_emb.size(0)

            # Tensor lưu Max Score cho mỗi candidate của loại này
            best_scores = torch.full((num_candidates,), -1.0, device='cpu')
            best_rels = [None] * num_candidates

            for r_name in rels:
                if r_name.startswith('rev_'): continue  # Bỏ qua cạnh ngược nếu muốn

                key = f"__{r_name}__"  # Format key decoder
                if key not in self.model.decoders: continue
                decoder = self.model.decoders[key]

                # Batch Inference
                for i in range(0, num_candidates, eval_batch_size):
                    batch_dst = candidates_emb[i: i + eval_batch_size].to(self.device)
                    batch_src = vec_src.expand(batch_dst.size(0), -1)

                    with torch.amp.autocast('cuda'):
                        logits = decoder(batch_src, batch_dst)
                        scores = torch.sigmoid(logits).view(-1).cpu()

                    # Cập nhật Max Score
                    current_slice = slice(i, i + len(scores))
                    mask = scores > best_scores[current_slice]
                    best_scores[current_slice] = torch.where(mask, scores, best_scores[current_slice])

                    # Update Relation Name
                    indices = torch.nonzero(mask).flatten() + i
                    for idx in indices:
                        best_rels[idx.item()] = r_name

            # Self-loop check
            if src_type == target_type:
                best_scores[src_id] = -1.0

            # Local Top-K
            k_local = min(top_k, num_candidates)
            vals, indices = torch.topk(best_scores, k=k_local)

            for val, idx in zip(vals, indices):
                if val > 0.0:
                    idx = idx.item()
                    global_candidates.append({
                        'id': idx,
                        'type': target_type,
                        'relation': best_rels[idx],
                        'score': val.item()
                    })

        # 3. Global Sort
        global_candidates.sort(key=lambda x: x['score'], reverse=True)
        return global_candidates[:top_k]