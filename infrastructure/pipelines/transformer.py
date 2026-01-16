import json
import numpy as np
import pandas as pd
import glob
import os
import re
from unidecode import unidecode
from config.settings import RAW_JSON_DIR, RAW_PARQUET_PATH, EDGES_DATA_PATH,NODES_DATA_PATH, GRAPH_PATH
from infrastructure.repositories import PickleGraphRepository
import igraph as ig
class GraphTransformer:
    def __init__(self):
        # Khởi tạo một đồ thị rỗng
        print("GraphTransformer initialized.")

    def _ingest_json_to_parquet(self,json_folder = RAW_JSON_DIR , save = True):
        json_files = glob.glob(os.path.join(str(json_folder,), "*.json"))

        dfs = []
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            try:
                file_name = os.path.splitext(file_name)[0].split('_')
                object_type = file_name[-1]
                person_type = file_name[2]
            except IndexError:
                object_type = "unknown"
                person_type = "unknown"
            try:
                print(f"Chuyển đổi quan hệ {file_name[-2]}:",end='',flush=True)
                # Đọc dữ liệu
                data = self._load_and_flatten_json(file_path)

                # Nếu file rỗng hoặc lỗi, bỏ qua
                if data.empty: continue

                data['objectType.value'] = object_type
                data['personType.value'] = person_type

                dfs.append(data)

                print("Thành công!",flush=True)
            except Exception as e:
                print(f"LỖI: {e}")


        # Gộp tất cả dữ liệu chính
        if dfs:
            dfs_final = pd.concat(dfs, ignore_index=True)
        else:
            print("Không tìm thấy dữ liệu chính!")
            return

        interest_path = json_folder / "interests"
        interest_files = glob.glob(os.path.join(str(interest_path), "*interest*.json"))

        dfs_interests = []
        for file_path in interest_files:
            # Đọc dữ liệu
            data = self._load_and_flatten_json(file_path)
            if data.empty: continue

            # 1. Chọn đúng cột cần thiết
            if 'person.value' in data.columns and 'objectLabel.value' in data.columns:
                data = data[['person.value', 'objectLabel.value']]
                data = data.rename(columns={'objectLabel.value': 'interests.value'})
                dfs_interests.append(data)

        # --- PHẦN 3: MERGE (GỘP SỞ THÍCH VÀO MAIN) ---
        if dfs_interests:
            # Gộp tất cả file interest lại
            df_interests_all = pd.concat(dfs_interests, ignore_index=True)

            df_interests_agg = df_interests_all.groupby('person.value')['interests.value'].apply(
                lambda x: ', '.join(x.dropna().astype(str).unique())
            ).reset_index()

            df_final = pd.merge(dfs_final, df_interests_agg, on='person.value', how='left')
        else:
            print("Không tìm thấy dữ liệu Interest, bỏ qua bước merge.")
            dfs_final['interests.value'] =''
            df_final = dfs_final

        # --- PHẦN 4: LƯU PARQUET ---
        os.makedirs(os.path.dirname(str(RAW_PARQUET_PATH)), exist_ok=True)
        if save:
            df_final.to_parquet(str(RAW_PARQUET_PATH), engine='pyarrow', compression='snappy')
        print(f"Đã gộp xong! Tổng số dòng: {len(df_final)}")
        return df_final

    def _load_and_flatten_json(self, raw_filepath):
        """
        Đọc, làm phẳng và dọn dẹp sơ bộ dữ liệu từ JSON.
        """
        if not os.path.exists(raw_filepath):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {raw_filepath}")
            return pd.DataFrame()

        try:
            with open(raw_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 1. Làm phẳng
            bindings = data.get('results', {}).get('bindings', [])
            if not bindings:
                return pd.DataFrame()

            df = pd.json_normalize(bindings)


            return df
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {raw_filepath}: {e}")
            return pd.DataFrame()

    def _remove_back_edges_stay_columns(self, df):
        # Trích xuất mảng để tính toán tốc độ cao
        p1 = df['person'].values
        p2 = df['object'].values
        rel = df['relationshipLabel'].values

        # Sắp xếp vector: luôn giữ (nhỏ, lớn)
        node_min = np.where(p1 < p2, p1, p2)
        node_max = np.where(p1 < p2, p2, p1)

        # Tạo DataFrame tạm thời để tận dụng hàm duplicated cực nhanh của Pandas
        temp_df = pd.DataFrame({
            'n1': node_min,
            'n2': node_max,
            'rel': rel
        }, index=df.index)  # Quan trọng: Giữ nguyên Index của df gốc

        mask = ~temp_df.duplicated(keep='first')
        return df.loc[mask]
    def _get_object_info(self, df):
        file_path = 'data_output/raw/raw_data_object.parquet'
        df_temp = pd.read_parquet(file_path, engine='pyarrow')
        df_final = pd.merge(df, df_temp, on ='id', how = 'left')
        return df_final
    def _get_person_occupation(self,df):
        file_path = 'data_output/raw/raw_data_occupation.parquet'
        df_temp = pd.read_parquet(file_path, engine='pyarrow')
        df_final = pd.merge(df, df_temp, on = 'id', how = 'left')
        return df_final
    def _create_nodes_data(self,df: pd.DataFrame):

        # 1. Chuẩn bị Person
        cols_p_map = {
            'person': 'id', 'personLabel': 'name', 'personDescription': 'description',
            'birthYear': 'birthYear', 'countryLabel': 'country',
            'birthPlaceLabel': 'birthPlace', 'interests': 'interests', 'personType' : 'type'
        }
        # Lọc cột tồn tại để tránh lỗi
        valid_cols_p = [c for c in cols_p_map.keys() if c in df.columns]
        df_person = df[valid_cols_p].rename(columns=cols_p_map)
        df_person = self._get_person_occupation(df_person)

        # 2. Chuẩn bị Object
        cols_o_map = {
            'object': 'id',
            'objectLabel': 'name',
            'objectDescription': 'description',
            'objectType': 'type'
        }
        valid_cols_o = [c for c in cols_o_map.keys() if c in df.columns]
        df_object = df[valid_cols_o].rename(columns=cols_o_map)
        df_object = self._get_object_info(df_object)

        # 3. Data final
        df_nodes = pd.concat([df_person, df_object], ignore_index=True)
        df_nodes['birthYear'] = pd.to_numeric(df_nodes['birthYear'], errors='coerce').astype('Int64')
        df_nodes.drop_duplicates('id', inplace=True, ignore_index=True, keep='first')
        df_nodes['pyg_id'] = df_nodes.groupby('type').cumcount()
        return df_nodes
    def _clean_and_process_data(self, df: pd.DataFrame, save = True):
        if df.empty:
            return df
        # 2. Đổi tên cột (Bỏ đuôi .value)
        # Chỉ đổi tên những cột quan trọng, các cột khác (.type, .xml:lang) sẽ bị lọc bỏ sau
        new_columns = {col: col.replace('.value', '') \
                       for col in df.columns \
                       if col.endswith('.value')}

        #Đổi tên cột
        df = df.rename(columns=new_columns)

        # 3. Lọc bỏ các cột metadata thừa (type, xml:lang, datatype...)
        if new_columns:
            valid_cols = list(new_columns.values())
            df = df[valid_cols]

        def join_unique(x):
            return ', '.join(x.fillna(' ').astype(str).unique())

        cols_to_group = ['countryLabel', 'birthPlaceLabel', 'birthYear']

        df_aggregated = df.groupby('person')[cols_to_group].agg(join_unique).reset_index()

        df = df.drop(columns=cols_to_group)
        df = pd.merge(df, df_aggregated, on='person', how='left')
        df[['countryLabel', 'birthPlaceLabel', 'birthYear']] = df[['countryLabel', 'birthPlaceLabel', 'birthYear']].replace('', None)

        num_col_isnull = df.isnull().sum()
        print(f"Thống kê cột có dữ liệu bị thiếu:\n{num_col_isnull}",flush=True)
        num_row_isnull = df.isnull().any(axis=1).sum()
        print(f"Tổng số dòng có dữ liệu bị thiếu: {num_row_isnull}",flush=True)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'[\r\n\t]+', ' ', regex=True)

        # Làm sạch ID (bỏ http://.../Q123 -> Q123)
        for col in ['person', 'object']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.split('/').str[-1]

        # Rác của ID
        print(len(df))

        # Rác của Label: Bắt đầu bằng Q (chưa giải mã) HOẶC là link genid
        pattern = r'^Q\d+$'

        # Tìm dòng mà ID bắt đầu bằng Q
        mask_id = df[['person', 'object']].apply(
            lambda col: col.astype(str).str.match(pattern, na=False)
        ).all(axis=1)

        df = df[mask_id]

        # Tìm dòng mà Label bị dính mã Q hoặc link genid
        mask_label = df[['personLabel', 'objectLabel']].apply(
            lambda col: col.astype(str).str.match(pattern, na=False)
        ).any(axis=1)

        df = df[~mask_label]

        print(f"Tổng số dòng rác đã loại bỏ: {(~mask_id).sum() + mask_label.sum()}")
        print(len(df))
        df = df[df['person'].notna() & (df['person'] != '')]
        print(len(df))

        # Loc cac dong bi lap
        df = df.drop_duplicates(subset=['person', 'relationshipLabel', 'object'], keep='first')
        print(len(df))

        print(f"Đang lọc các cạnh ngược")
        df = self._remove_back_edges_stay_columns(df)
        print(len(df))

        df_nodes = self._create_nodes_data(df)

        # Lay cac cot can thiet
        cols = ['person', 'personLabel','personType' , 'relationshipLabel', 'object', 'objectLabel', 'objectType']
        df_edges = df[cols]

        def to_snake_case(name: str) -> str:
            """
            Chuyển đổi chuỗi từ camelCase/PascalCase/Space-separated sang snake_case.
            """
            # Bước 1: Xử lý các cụm viết tắt và ký tự hoa liên tiếp
            # Ví dụ: personSubType -> person_Sub_Type
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)

            # Bước 2: Tách giữa ký tự thường/số và ký tự hoa
            # Ví dụ: person_Sub_Type -> person_sub_type
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

            # Bước 3: Thay thế khoảng trắng hoặc gạch ngang thành gạch dưới
            return s2.replace(" ", "_").replace("-", "_")

        df_edges.columns = [to_snake_case(col) for col in df_edges.columns]
        df_nodes.columns = [to_snake_case(col) for col in df_nodes.columns]
        print("Đã làm sạch xong!")
        if save:
            os.makedirs(os.path.dirname(str(EDGES_DATA_PATH)), exist_ok=True)
            os.makedirs(os.path.dirname(str(NODES_DATA_PATH)), exist_ok=True)
            df_edges.to_parquet(str(EDGES_DATA_PATH), engine='pyarrow', compression='snappy')
            df_nodes.to_parquet(str(NODES_DATA_PATH), engine='pyarrow', compression='snappy')
        return df_edges, df_nodes

    def build_graph(self,df_edges, df_nodes):

        # 1. Tạo đồ thị cơ bản

        # Tạo bảng mapping: Q-ID -> Index (Dùng để map cho edges)

        # Cột 'id' trong df_nodes là Q-ID (ví dụ Q123)

        mapping = pd.Series(df_nodes.index, index=df_nodes['id'])

        print(f" - Đã index {len(df_nodes)} nodes.")

        # --- BƯỚC 2: MAP EDGES (Dùng Pandas Vectorization) ---

        print(" - Đang mapping cạnh...")

        # Cần đảm bảo src và dst có cùng độ dài sau khi dropna (inner join logic)

        # Cách an toàn nhất là tạo 1 df tạm để dropna đồng bộ

        edges_temp = pd.DataFrame({

            'source': df_edges['person'],

            'target': df_edges['object'],

            'label': df_edges['relationship_label']

        })

        # Map lại trên df tạm

        edges_temp['src_idx'] = edges_temp['source'].map(mapping)

        edges_temp['dst_idx'] = edges_temp['target'].map(mapping)

        # Loại bỏ các hàng bị NaN (do node không có trong df_nodes)

        valid_edges = edges_temp.dropna(subset=['src_idx', 'dst_idx'])

        edge_list = list(zip(valid_edges['src_idx'].astype(int), valid_edges['dst_idx'].astype(int)))

        # Lấy thuộc tính quan trọng của cạnh để dùng tính trọng số sau này

        edge_attrs = {

            'relationship_label': valid_edges['label'].tolist()

        }

        print(f" - Số lượng cạnh hợp lệ: {len(edge_list)}")

        # --- BƯỚC 3: TẠO IGRAPH ---

        print(" - Đang xây dựng cấu trúc đồ thị...")

        g = ig.Graph(n=len(df_nodes), edges=edge_list, directed=True, edge_attrs=edge_attrs)

        # --- BƯỚC 4: NẠP THUỘC TÍNH NODE ---

        # Nạp tất cả thông tin từ df_nodes vào graph để sau này truy xuất

        # Lưu ý: igraph dùng thuộc tính 'name' làm định danh chuỗi (Q-ID)
        g.vs['name'] = df_nodes['id'].tolist()  # Q-ID (Q42)
        g.vs['label'] = df_nodes['name'].tolist()  # Tên hiển thị (Elon Musk)
        g.vs['type'] = df_nodes['type'].tolist()  # Loại (human, movie...)

        print("Hoàn tất chuyển đổi!")

        return g


    def run_transformer(self, raw_dir = None, force_data = True, save = True ):
        """
        Hàm điều phối chính (Orchestrator).
        """

        print(f"Chạy pipeline từ Raw Directory: {raw_dir}")
        if force_data:
            df = self._ingest_json_to_parquet(raw_dir)
            df_edges, df_nodes = self._clean_and_procces_data(df)
        else:
            df_edges = pd.read_parquet(EDGES_DATA_PATH, engine='fastparquet')
            df_nodes = pd.read_parquet(NODES_DATA_PATH, engine='fastparquet')

        relationship_graph = self.build_graph(df_edges,df_nodes)
        if save:
            graph_repo = PickleGraphRepository(GRAPH_PATH)
            graph_repo.save_graph(relationship_graph)
        return relationship_graph




