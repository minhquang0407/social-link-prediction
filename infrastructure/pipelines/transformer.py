import json
import networkx as nx
import pandas as pd
import glob
import os
from unidecode import unidecode
from config.settings import RAW_JSON_DIR, RAW_PARQUET_PATH, EDGES_DATA_PATH,NODES_DATA_PATH, GRAPH_PATH
from infrastructure.repositories import PickleGraphRepository
import igraph as ig
class GraphTransformer:
    def __init__(self):
        # Khởi tạo một đồ thị rỗng
        print("GraphTransformer initialized.")

    def _ingest_json_to_parquet(self,json_folder, save = True):
        json_files = glob.glob(os.path.join(str(RAW_JSON_DIR,), "*.json"))

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

        mean_val = int(round(df_nodes['birthYear'].mean()))
        df_nodes['birthYear'] = df_nodes['birthYear'].fillna(mean_val).astype('Int64')
        df_nodes.fillna('', inplace=True)

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

        df_country = df.groupby('person')['countryLabel'].apply(lambda x: ', '.join(x.fillna('').astype(str).unique())).reset_index()
        df = df.drop(columns= 'countryLabel')
        df = pd.merge(df,df_country, on='person',how = 'left')


        num_col_isnull = df.isnull().sum()
        print(f"Thống kê cột có dữ liệu bị thiếu:\n{num_col_isnull}",flush=True)
        num_row_isnull = df.isnull().any(axis=1).sum()
        print(f"Tổng số dòng có dữ liệu bị thiếu: {num_row_isnull}",flush=True)
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.replace(r'[\r\n\t]+', ' ', regex=True)

        df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce').astype('Int64')
        mean_val = df['birthYear'].mean()
        df['birthYear'] = df['birthYear'].fillna(mean_val)
        # Làm sạch ID (bỏ http://.../Q123 -> Q123)
        for col in ['person', 'object']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.split('/').str[-1]

        # Lọc các dòng mà có name bắt đầu bằng Q...
        qid_pattern = r'^Q\d+$'
        total_dropped = 0

        if 'personLabel' in df.columns:
            mask_invalid = df['personLabel'].astype(str).str.match(qid_pattern, na=False)
            count_invalid = mask_invalid.sum()
            if count_invalid > 0:
                df = df[~mask_invalid]
                total_dropped += count_invalid

        if 'objectLabel' in df.columns:
            mask_invalid = df['objectLabel'].astype(str).str.match(qid_pattern, na=False)
            count_invalid = mask_invalid.sum()
            if count_invalid > 0:
                df = df[~mask_invalid]
                total_dropped += count_invalid

        print(f"Số dòng bị bỏ (Name lỗi): {total_dropped}", flush=True)

        # Lọc dòng trống ID
        df = df[df['person'].notna() & (df['person'] != '')]
        # Loc cac dong bi lap

        df = df.drop_duplicates(subset=['person', 'relationshipLabel', 'object'], keep='first')
        df_nodes = self._create_nodes_data(df)

        # Lay cac cot can thiet
        cols = ['person', 'personLabel','personType' , 'relationshipLabel', 'object', 'objectLabel', 'objectType']
        df = df[cols]

        print("Đã làm sạch xong!")
        if save:
            os.makedirs(os.path.dirname(str(EDGES_DATA_PATH)), exist_ok=True)
            os.makedirs(os.path.dirname(str(NODES_DATA_PATH)), exist_ok=True)
            df.to_parquet(str(EDGES_DATA_PATH), engine='pyarrow', compression='snappy')
            df_nodes.to_parquet(str(NODES_DATA_PATH), engine='pyarrow', compression='snappy')
        return df, df_nodes

    def build_graph(self, df_edges, df_nodes):
        # 1. Tạo đồ thị cơ bản
        nodes_df = df_nodes.reset_index(drop=True)

        # Tạo bảng mapping: Q-ID -> Index (Dùng để map cho edges)
        # Cột 'id' trong df_nodes là Q-ID (ví dụ Q123)
        mapping = pd.Series(nodes_df.index, index=nodes_df['id'])

        print(f"   - Đã index {len(nodes_df)} nodes.")

        # --- BƯỚC 2: MAP EDGES (Dùng Pandas Vectorization) ---
        print("   - Đang mapping cạnh...")

        # Cần đảm bảo src và dst có cùng độ dài sau khi dropna (inner join logic)
        # Cách an toàn nhất là tạo 1 df tạm để dropna đồng bộ
        edges_temp = pd.DataFrame({
            'source': df_edges['person'],
            'target': df_edges['object'],
            'label': df_edges['relationshipLabel']
        })

        # Map lại trên df tạm
        edges_temp['src_idx'] = edges_temp['source'].map(mapping)
        edges_temp['dst_idx'] = edges_temp['target'].map(mapping)

        # Loại bỏ các hàng bị NaN (do node không có trong df_nodes)
        valid_edges = edges_temp.dropna(subset=['src_idx', 'dst_idx'])

        edge_list = list(zip(valid_edges['src_idx'].astype(int), valid_edges['dst_idx'].astype(int)))

        # Lấy thuộc tính quan trọng của cạnh để dùng tính trọng số sau này
        edge_attrs = {
            'relationshipLabel': valid_edges['label'].tolist()
        }

        print(f"   - Số lượng cạnh hợp lệ: {len(edge_list)}")

        # --- BƯỚC 3: TẠO IGRAPH ---
        print("   - Đang xây dựng cấu trúc đồ thị...")
        g = ig.Graph(n=len(nodes_df), edges=edge_list, directed=True, edge_attrs=edge_attrs)

        # --- BƯỚC 4: NẠP THUỘC TÍNH NODE ---
        # Nạp tất cả thông tin từ df_nodes vào graph để sau này truy xuất
        # Lưu ý: igraph dùng thuộc tính 'name' làm định danh chuỗi (Q-ID)
        g.vs['name'] = nodes_df['id'].tolist()  # Q-ID (Q42)
        g.vs['label'] = nodes_df['name'].tolist()  # Tên hiển thị (Elon Musk)
        g.vs['desc'] = nodes_df['description'].tolist()  # Mô tả
        g.vs['type'] = nodes_df['type'].tolist()  # Loại (human, movie...)

        # Các thuộc tính phụ (nếu cần hiển thị chi tiết)
        if 'birthYear' in nodes_df.columns:
            g.vs['birthYear'] = nodes_df['birthYear'].fillna(0).tolist()
        if 'country' in nodes_df.columns:
            g.vs['country'] = nodes_df['country'].fillna('').tolist()

        # --- BƯỚC 5: LƯU FILE ---

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




