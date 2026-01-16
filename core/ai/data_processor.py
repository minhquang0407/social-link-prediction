import gzip
from collections import defaultdict
import igraph as ig
import torch
import numpy as np
from networkx.readwrite.json_graph.adjacency import adjacency_data
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import ToUndirected
import pandas as pd
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm
from infrastructure.repositories import PyGDataRepository
from config.settings import PYG_DATA_PATH, ADJACENCY_PATH

class GraphDataProcessor:
    """
    1. Tạo vector đặc trưng từ thuộc tính node.
    2. Chuyển đổi đồ thị NetworkX sang PyG HeteroData.
    """

    def __init__(self):
        print("CORE: Đang tải model Sentence-BERT...")
        if torch.cuda.is_available():
            print("GPU FOUNDED! PREPARING ON GPU")
            self.device = 'cuda'
        else:
            print("GPU NOT FOUND! PREPARING ON CPU")
            self.device = 'cpu'
        self.text_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self.text_encoder.max_seq_length = 384
        self.text_encoder.tokenizer.model_max_length = 384

    import igraph as ig
    import pandas as pd
    import numpy as np

    def _precompute_structural_features(self, df_nodes, df_edges):
        """
        Tính toán trước PageRank (theo từng loại cạnh) và Total Degree.
        Output: DataFrame indexed by 'id', columns = [pr_0, pr_1, ..., degree]
        """
        print("Đang tính toán Structural Features (PageRank & Degree)...")

        all_node_ids = df_nodes['id'].unique()
        node_to_idx = {id: i for i, id in enumerate(all_node_ids)}
        num_nodes = len(all_node_ids)

        # 1. Xác định danh sách các loại quan hệ (Edge Labels)
        edge_labels = sorted(df_edges['relationship_label'].unique())
        print(f"   - Tìm thấy {len(edge_labels)} loại quan hệ.")

        # Khởi tạo ma trận kết quả: [Num_Nodes, Num_Relations]
        # Mỗi cột tương ứng với PageRank của một loại cạnh
        pr_matrix = np.zeros((num_nodes, len(edge_labels)), dtype=np.float32)

        # 2. Tính Multi-view PageRank
        for i, label in enumerate(edge_labels):
            # Lấy các cạnh thuộc loại label này
            sub_edges = df_edges[df_edges['relationship_label'] == label]

            if len(sub_edges) == 0: continue

            # Map ID sang Index để tạo graph
            # Lưu ý: dropna() để bỏ các cạnh có node không tồn tại trong df_nodes
            src_idx = sub_edges['person'].map(node_to_idx).dropna().astype(int)
            dst_idx = sub_edges['object'].map(node_to_idx).dropna().astype(int)

            edges_list = list(zip(src_idx, dst_idx))

            if not edges_list: continue

            # Tạo graph tạm bằng igraph (Directed)
            g = ig.Graph(n=num_nodes, edges=edges_list, directed=True)

            # Tính PageRank
            # damping=0.85 là chuẩn. reset=None để dùng uniform distribution
            pr_scores = g.pagerank(damping=0.85)

            # Lưu vào cột thứ i
            pr_matrix[:, i] = pr_scores

        # 3. Tính Total Degree (Trên toàn bộ đồ thị gộp)
        print("   - Đang tính Total Degree...")
        # Tạo đồ thị tổng hợp
        full_src = df_edges['person'].map(node_to_idx).dropna().astype(int)
        full_dst = df_edges['object'].map(node_to_idx).dropna().astype(int)
        full_edges = list(zip(full_src, full_dst))

        g_full = ig.Graph(n=num_nodes, edges=full_edges, directed=False)  # Degree tính vô hướng cho tổng quát
        degrees = np.array(g_full.degree())

        # 4. CHUẨN HÓA (Log Transform) - QUAN TRỌNG
        # Giúp đưa dữ liệu về khoảng dễ học hơn, tránh outlier quá lớn
        pr_matrix = np.log1p(pr_matrix)  # log(x + 1)
        degrees = np.log1p(degrees).reshape(-1, 1)

        # 5. Gộp lại thành DataFrame
        print("   - Đang đóng gói Structural Features...")
        struct_feats = np.concatenate([pr_matrix, degrees], axis=1)

        # Tạo tên cột: pr_spouse, pr_acted_in, ..., total_degree
        col_names = [f'pr_{label}' for label in edge_labels] + ['total_degree']

        df_struct = pd.DataFrame(struct_feats, index=all_node_ids, columns=col_names)

        # Giải phóng bộ nhớ
        del pr_matrix, degrees, g_full, node_to_idx

        return df_struct, len(edge_labels)
    def _create_node_features(self, df, structural_features_df):
        df['full_text'] = (
                'name: ' + df['name']
                + ', description: ' + df['description'].astype(str).fillna('')
                + ', interests: ' + df['interests'].astype(str).fillna('')
                + ', occupation: ' + df['occupation'].astype(str).fillna('')
                + ', country: ' + df['country'].astype(str).fillna('')
                + ', birthPlace: ' + df['birth_place'].astype(str).fillna('')
                + ', subtype: ' + df['sub_type'].astype(str).fillna('')
                + ', sex: ' + df['sex_or_gender'].astype(str).fillna('')
        )

        embeddings= self.text_encoder.encode(df['full_text'].tolist(), batch_size=256, show_progress_bar=True, device= self.device)

        valid_years = df['birth_year'].dropna().astype(float)

        if not valid_years.empty:
            _min_year = valid_years.min()
            _max_year = valid_years.max()
            # Tính median để bù đắp
            median_year =  valid_years.median()
        else:
            # Trường hợp dự phòng nếu cả cột birthYear đều trống
            _min_year, _max_year, median_year = 1800, 2025, 1950

        range_year = _max_year - _min_year
        if range_year == 0: range_year = 1

        is_missing = df['birth_year'].isna().astype(float).values.reshape(-1, 1)
        years = df['birth_year'].astype(float).fillna(median_year)
        years_norm = (years - _min_year) / range_year
        years_vec = years_norm.values.reshape(-1, 1)
        struct_feats = structural_features_df.reindex(df['id']).fillna(0).values

        # 4. GỘP TẤT CẢ (CONCATENATE)
        # Cấu trúc: [BERT (384) | Year (1) | IsMissing (1) | PageRank (45) | Degree (1)]
        features = np.concatenate([embeddings, years_vec, is_missing, struct_feats], axis=1)

        return torch.from_numpy(features).float()

    def sanitize_hetero_data(self,data):
        """Xóa các loại cạnh rỗng để tránh lỗi khi chạy Loader."""
        print("Đang dọn dẹp các loại cạnh rỗng...")
        edge_types_to_delete = []
        for edge_type in data.edge_types:
            if 'edge_index' not in data[edge_type] or data[edge_type].edge_index is None:
                edge_types_to_delete.append(edge_type)
            elif data[edge_type].edge_index.numel() == 0:
                edge_types_to_delete.append(edge_type)

        for et in edge_types_to_delete:
            del data[et]
        return data


    def process_graph_to_pyg(self, df_edges, df_nodes):
        """
        Chuyển đổi DataFrame sang HeteroData object.

        Args:
            df_edges (pd.DataFrame): ['person', ..., 'src_type', 'dst_type']
            df_nodes (pd.DataFrame): ['id', ..., 'type', 'pyg_id']

        Returns:
            HeteroData: Đồ thị dị thể PyG.
        """
        print("CORE: Bắt đầu chuyển đổi đồ thị sang HeteroData (Pre-typed)...")
        pyg_data = HeteroData()

        # ---------------------------------------------------------
        # 0. Chuẩn bị Mappings
        # ---------------------------------------------------------
        # Chỉ cần map ID -> PyG ID. Không cần map Type nữa.
        id_to_pygid_map = dict(zip(df_nodes['id'], df_nodes['pyg_id']))
        df_struct, num_rels = self._precompute_structural_features(df_nodes, df_edges)
        print(f"LOG: Đã chuẩn bị {df_struct.shape[1]} structural features.")
        # ---------------------------------------------------------
        # 1. Xử lý Nodes & Features (Giữ nguyên)
        # ---------------------------------------------------------
        print("CORE: Xử lý Node")
        grouped_nodes = df_nodes.groupby('type')
        for ntype, group in grouped_nodes:
            # Tạo features cho từng loại node
            print(f"LOG: Đang Xử lý loại '{ntype}': {len(group)} nodes.", flush=True)
            node_features = self._create_node_features(group, df_struct)
            pyg_data[ntype].x = node_features
            pyg_data[ntype].num_nodes = len(group)
            print(f"LOG: Xong loại '{ntype}': {len(group)} nodes.",flush= True)

        # ---------------------------------------------------------
        # 2. Xử lý Edges (Đã tối ưu hóa)
        # ---------------------------------------------------------
        print("CORE: Xử lý cạnh")

        # Map ID gốc sang PyG ID (local index) trực tiếp trên df_edges
        # Lưu ý: Ta dùng .map() để thay thế ID string bằng index int
        df_edges['src_idx'] = df_edges['person'].map(id_to_pygid_map)
        df_edges['dst_idx'] = df_edges['object'].map(id_to_pygid_map)

        # Group key: (Loại nguồn, Quan hệ, Loại đích)
        initial_len = len(df_edges)
        df_edges.dropna(subset=['src_idx', 'dst_idx'], inplace=True)
        print(f"LOG: Đã loại bỏ {initial_len - len(df_edges)} cạnh không hợp lệ (NaN).")
        adj_store = {}
        grouped_edges = df_edges.groupby(['person_type', 'relationship_label', 'object_type'])

        for (src_type, rel_label, dst_type), group in grouped_edges:
            edge_type = (str(src_type), str(rel_label), str(dst_type))

            # Lấy indices (đã chắc chắn là số nguyên do dropna)
            src_indices = group['src_idx']
            dst_indices = group['dst_idx']

            key = f"{src_type}__{rel_label}__{dst_type}"
            adj_store[key] = set(zip(src_indices , dst_indices))

            mask = src_indices.notna() & dst_indices.notna()
            src = torch.from_numpy(src_indices[mask].values.astype(np.int64))
            dst = torch.from_numpy(dst_indices[mask].values.astype(np.int64))
            # Chuyển sang Tensor
            edge_index = torch.stack([src, dst], dim=0)

            # Loại bỏ trùng lặp
            edge_index = sort_edge_index(edge_index, sort_by_row=True)
            # Gán vào HeteroData
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
            pyg_data[edge_type].edge_index = edge_index

            print(f"LOG: Đã thêm {edge_index.size(1)} cạnh loại: {edge_type}")

        # ---------------------------------------------------------
        # 3. Post-processing
        # ---------------------------------------------------------
        print("Kiểm tra chuyển đổi:")
        print(f"Kiểm tra đỉnh: {pyg_data.num_nodes} = {len(df_nodes)} - {pyg_data.num_nodes == len(df_nodes)}")
        print(f"Kiểm tra đỉnh: {pyg_data.num_edges} = {len(df_edges)} - {pyg_data.num_edges == len(df_edges)}")
        pyg_data = ToUndirected(merge=False)(pyg_data)
        pyg_data = self.sanitize_hetero_data(pyg_data)

        return pyg_data, adj_store

    def run(self, edges, nodes):
        pyg_data, adjacency= self.process_graph_to_pyg(edges, nodes)
        repo = PyGDataRepository(PYG_DATA_PATH, ADJACENCY_PATH)
        repo.save_data(pyg_data)
        repo.save_adjacency(adjacency)

