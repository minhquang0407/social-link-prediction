import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData


class GraphDataProcessor:
    """
    Hệ thống xử lý dữ liệu đồ thị:
    1. Tạo vector đặc trưng (node features) bằng SBERT và chuẩn hóa dữ liệu số.
    2. Chuyển đổi DataFrame thành cấu trúc HeteroData của PyTorch Geometric.
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"CORE: Đang khởi tạo model {model_name}...")
        # TODO: Khởi tạo self.text_encoder và cấu hình thiết bị (cuda/cpu)
        self.text_encoder = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _create_node_features(self, df: pd.DataFrame):
        """
        TODO: Xử lý vector đặc trưng cho Node
        1. Kết hợp các cột text thành một chuỗi duy nhất.
        2. Sử dụng SBERT để encode chuỗi thành embeddings.
        3. Chuẩn hóa cột 'birthYear' về khoảng [0, 1].
        4. Concatenate embedding và year_norm thành một tensor duy nhất.
        """
        # Gợi ý logic:
        # text_data = df['name'] + ...
        # feature_vec = self.text_encoder.encode(...)
        # year_norm = (year - min) / (max - min)

        print("TODO: Thực hiện logic tạo features tại đây")
        return None  # Trả về torch.FloatTensor

    def _create_nodes_data(self, df: pd.DataFrame):
        """
        TODO: Chuẩn hóa dữ liệu thô thành DataFrame chứa Nodes
        1. Tách và map các cột cho thực thể 'Person'.
        2. Tách và map các cột cho thực thể 'Object'.
        3. Hợp nhất (concat) và loại bỏ trùng lặp.
        """
        print("TODO: Thực hiện logic chuẩn hóa danh sách node tại đây")
        return None  # Trả về pd.DataFrame chứa: id, type, và các thuộc tính

    def process_graph_to_pyg(self, df: pd.DataFrame):
        """
        TODO: Chuyển đổi toàn bộ dữ liệu sang PyG HeteroData
        """
        print("CORE: Bắt đầu chuyển đổi đồ thị...")

        # BƯỚC 1: Chuẩn bị dữ liệu node tổng hợp
        df_nodes = self._create_nodes_data(df)

        pyg_data = HeteroData()
        node_mapping = {}  # Lưu {ntype: {original_id: integer_index}}
        rev_node_mapping = {}  # Lưu {ntype: {integer_index: original_id}}

        # BƯỚC 2: Xử lý Nodes & Features theo từng loại (ntype)
        print("CORE: Đang xử lý Node Features...")
        grouped_nodes = df_nodes.groupby('type')
        for ntype, group in grouped_nodes:
            # TODO:
            # 1. Tạo mapping index cho loại node này
            # 2. Gọi _create_node_features để lấy tensor đặc trưng
            # 3. Gán vào pyg_data[ntype].x
            pass

        # BƯỚC 3: Xử lý Edges (Quan hệ)
        print("CORE: Đang xử lý Edges...")
        # TODO:
        # 1. Group dữ liệu theo loại quan hệ (relationshipLabel, objectType)
        # 2. Dựa vào node_mapping để chuyển đổi ID gốc sang Index (int)
        # 3. Tạo edge_index tensor và gán vào pyg_data[edge_type].edge_index

        # Gợi ý: edge_type = ('human', rel_label, obj_type)

        print("LOG: Quá trình chuyển đổi hoàn tất.")
        return pyg_data, (node_mapping, rev_node_mapping)

