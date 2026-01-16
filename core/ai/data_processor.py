import torch
import numpy as np
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from torch_geometric.transforms import ToUndirected
import pandas as pd

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

    def _create_node_features(self, df):
        df['full_text'] = (
                'name: ' + df['name']
                + ', description: ' + df['description']
                + ', country: ' + df['country']
                + ', birthPlace: ' + df['birthPlace']
                + ', interests: ' + df['interests']
        )
        embeddings = df['full_text'].tolist()
        feature = self.text_encoder.encode(embeddings, batch_size=128, show_progress_bar=True, device= self.device)
        _max_year = 2025
        _min_year = 1800

        years = df['birthYear']

        years_norm = (years - _min_year) / (_max_year - _min_year)

        years_vec = years_norm.values.reshape(-1, 1)

        features = np.concat([feature, years_vec], axis=1)

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

    import torch
    import numpy as np
    import pandas as pd
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import ToUndirected

    def process_graph_to_pyg(self, df_edges, df_nodes):
        """
        Chuyển đổi DataFrame sang HeteroData object (Phiên bản tối ưu khi có sẵn Type).

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

        # ---------------------------------------------------------
        # 1. Xử lý Nodes & Features (Giữ nguyên)
        # ---------------------------------------------------------
        print("CORE: Xử lý Node")
        grouped_nodes = df_nodes.groupby('type')

        for ntype, group in grouped_nodes:
            # Tạo features cho từng loại node
            node_features = self._create_node_features(group)

            pyg_data[ntype].x = node_features
            pyg_data[ntype].num_nodes = len(group)
            print(f"LOG: Xong loại '{ntype}': {len(group)} nodes.")

        # ---------------------------------------------------------
        # 2. Xử lý Edges (Đã tối ưu hóa)
        # ---------------------------------------------------------
        print("CORE: Xử lý cạnh")

        # Map ID gốc sang PyG ID (local index) trực tiếp trên df_edges
        # Lưu ý: Ta dùng .map() để thay thế ID string bằng index int
        df_edges['src_idx'] = df_edges['person'].map(id_to_pygid_map)
        df_edges['dst_idx'] = df_edges['object'].map(id_to_pygid_map)

        # Data Integrity: Loại bỏ các cạnh chứa node không tồn tại trong df_nodes
        # (Ví dụ: Node bị lọc bỏ ở bước trước hoặc dữ liệu edges lỗi)
        initial_count = len(df_edges)
        valid_edges = df_edges.dropna(subset=['src_idx', 'dst_idx'])

        dropped_count = initial_count - len(valid_edges)
        if dropped_count > 0:
            print(f"WARN: Đã loại bỏ {dropped_count} cạnh 'treo' (dangling edges).")

        # Group trực tiếp dựa trên cột type có sẵn
        # Group key: (Loại nguồn, Quan hệ, Loại đích)
        grouped_edges = valid_edges.groupby(['personType', 'relationshipLabel', 'objectType'])

        for (src_type, rel_label, dst_type), group in grouped_edges:
            edge_type = (str(src_type), str(rel_label), str(dst_type))

            # Lấy indices (đã chắc chắn là số nguyên do dropna)
            src = group['src_idx'].values.astype(np.int64)
            dst = group['dst_idx'].values.astype(np.int64)

            # Chuyển sang Tensor
            edge_index = torch.stack([
                torch.from_numpy(src),
                torch.from_numpy(dst)
            ], dim=0)

            # Loại bỏ trùng lặp
            edge_index = torch.unique(edge_index, dim=1)

            # Gán vào HeteroData
            pyg_data[edge_type].edge_index = edge_index

            print(f"LOG: Đã thêm {edge_index.size(1)} cạnh loại: {edge_type}")

        # ---------------------------------------------------------
        # 3. Post-processing
        # ---------------------------------------------------------
        pyg_data = ToUndirected(merge=False)(pyg_data)
        pyg_data = self.sanitize_hetero_data(pyg_data)

        return pyg_data
