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
        """
        # 1. Kết hợp các cột text thành một chuỗi duy nhất.
        # [MATCH] Logic: text_data = df['name'] + ...
        # (Dùng fillna để tránh lỗi NaN khi cộng chuỗi)
        text_data = df['name'].fillna('') + ' ' + df.get('description', '').fillna('')

        # 2. Sử dụng SBERT để encode chuỗi thành embeddings.
        # [MATCH] Logic: feature_vec = self.text_encoder.encode(...)
        embeddings = self.text_encoder.encode(
            text_data.tolist(),
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 3. Chuẩn hóa cột 'birthYear' về khoảng [0, 1].
        if 'birthYear' in df.columns and not df['birthYear'].isnull().all():
            years = df['birthYear'].fillna(df['birthYear'].mean()).values
            min_y, max_y = years.min(), years.max()

            # [MATCH] Logic: year_norm = (year - min) / (max - min)
            # (Thêm 1e-9 để tránh chia cho 0 nếu max == min)
            year_norm = (years - min_y) / (max_y - min_y + 1e-9)

            # Chuyển sang tensor để ghép
            year_tensor = torch.tensor(year_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

            # 4. Concatenate embedding và year_norm thành một tensor duy nhất.
            # [MATCH] Concatenate
            return torch.cat([embeddings, year_tensor], dim=1)

        return embeddings  # Trả về torch.FloatTensor

    def _create_nodes_data(self, df: pd.DataFrame):
        """
        TODO: Chuẩn hóa dữ liệu thô thành DataFrame chứa Nodes
        """
        # 1. Tách và map các cột cho thực thể 'Person'.
        # [MATCH] Tách cột cho Person
        person_nodes = df[['person_id', 'person_name']].rename(
            columns={'person_id': 'id', 'person_name': 'name'}
        )
        # Lưu ý: Đặt type là 'human' để khớp với gợi ý edge_type bên dưới
        person_nodes['type'] = 'human'
        # (Nếu code cũ dùng 'person', ở đây tôi đổi thành 'human' để khớp đúng comment gợi ý của bạn)

        # 2. Tách và map các cột cho thực thể 'Object'.
        # [MATCH] Tách cột cho Object
        object_nodes = df[['object_id', 'object_name', 'object_type']].rename(
            columns={'object_id': 'id', 'object_name': 'name', 'object_type': 'type'}
        )

        # 3. Hợp nhất (concat) và loại bỏ trùng lặp.
        # [MATCH] Concat và drop_duplicates
        df_nodes = pd.concat([person_nodes, object_nodes], ignore_index=True)
        df_nodes = df_nodes.drop_duplicates(subset=['id', 'type']).reset_index(drop=True)

        return df_nodes  # Trả về pd.DataFrame chứa: id, type, và các thuộc tính

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
            ids = group['id'].values
            mapping = {id_: i for i, id_ in enumerate(ids)}
            node_mapping[ntype] = mapping
            rev_node_mapping[ntype] = {i: id_ for id_, i in mapping.items()}

            # 2. Gọi _create_node_features để lấy tensor đặc trưng
            # 3. Gán vào pyg_data[ntype].x
            # [MATCH] Gọi hàm feature và gán vào pyg_data
            pyg_data[ntype].x = self._create_node_features(group)

        # BƯỚC 3: Xử lý Edges (Quan hệ)
        print("CORE: Đang xử lý Edges...")
        # TODO:
        # 1. Group dữ liệu theo loại quan hệ (relationshipLabel, objectType)
        # [MATCH] Groupby
        for (rel_label, obj_type), group in df.groupby(['relationshipLabel', 'object_type']):
            # 2. Dựa vào node_mapping để chuyển đổi ID gốc sang Index (int)
            # Lấy mapping của 'human' (nguồn) và obj_type (đích)
            src_mapping = node_mapping['human']
            dst_mapping = node_mapping[obj_type]

            src_indices = [src_mapping[pid] for pid in group['person_id']]
            dst_indices = [dst_mapping[oid] for oid in group['object_id']]

            # 3. Tạo edge_index tensor và gán vào pyg_data[edge_type].edge_index
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

            # [MATCH] Gợi ý: edge_type = ('human', rel_label, obj_type)
            edge_type = ('human', rel_label, obj_type)
            pyg_data[edge_type].edge_index = edge_index

        print("LOG: Quá trình chuyển đổi hoàn tất.")
        return pyg_data, (node_mapping, rev_node_mapping)

