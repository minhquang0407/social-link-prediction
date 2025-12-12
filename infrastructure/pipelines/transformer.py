import sys
import json
import networkx as nx
import pandas as pd
import glob
import os
from unidecode import unidecode
from collections import defaultdict
from pathlib import Path
import pyarrow
from config.settings import RAW_JSON_DIR, RAW_PARQUET_PATH, CLEAN_DATA_PATH, GRAPH_PATH
from infrastructure.repositories import PickleGraphRepository
class GraphTransformer:
    def __init__(self):
        # Khởi tạo một đồ thị rỗng
        print("GraphTransformer initialized.")

    def _ingest_json_to_parquet(self,json_folder):
        json_folder_path = Path(json_folder)
        json_files = glob.glob(os.path.join(str(json_folder_path), "*.json"))

        dfs = []
        for file_path in json_files:
            file_name = os.path.basename(file_path)
            try:
                file_name = os.path.splitext(file_name)[0].split('_')

                object_type = file_name[-1]
            except IndexError:
                object_type = "unknown"
            try:
                print(f"Chuyển đổi quan hệ {file_name[-2]}:",end='',flush=True)
                # Đọc dữ liệu
                data = self._load_and_flatten_json(file_path)

                # Nếu file rỗng hoặc lỗi, bỏ qua
                if data.empty: continue

                data['objectType.value'] = object_type
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

        interest_path = json_folder_path / "interest"
        interest_files = glob.glob(os.path.join(str(interest_path), "*interest*.json"))

        dfs_interests = []
        for file_path in interest_files:
            # Đọc dữ liệu
            data = self._load_and_flatten_json(file_path)
            if data.empty: continue

            # 1. SỬA LỖI CHỌN CỘT: Chọn đúng cột cần thiết
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
            df_final = dfs_final

        # --- PHẦN 4: LƯU PARQUET ---
        os.makedirs(os.path.dirname(str(RAW_PARQUET_PATH)), exist_ok=True)

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

    def _clean_and_procces_data(self, df: pd.DataFrame) -> pd.DataFrame:
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

        valid_cols = list(new_columns.values())
        df = df[valid_cols]

        num_col_isnull = df.isnull().sum()
        print(f"Thống kê cột có dữ liệu bị thiếu:\n{num_col_isnull}",flush=True)
        num_row_isnull = df.isnull().any(axis=1).sum()
        print(f"Tổng số dòng có dữ liệu bị thiếu: {num_row_isnull}",flush=True)
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.replace(r'[\r\n\t]+', ' ', regex=True)


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


        # Lưu file


        print("Đã làm sạch xong!")
        return df

    def _create_attribute_node(self, df: pd.DataFrame) -> dict:
        # -- XỬ LÝ PERSON --
        cols_person = {
            'person': 'id',
            'personLabel': 'name',
            'personDescription': 'description',
            'birthYear': 'birthYear',
            'interest': 'interests',
            'countryLabel': 'country',
            'birthPlaceLabel': 'birthPlace'
        }

        valid_p_cols = [c for c in cols_person.keys() if c in df.columns]
        df_p = df[valid_p_cols].drop_duplicates(subset=['person'])
        df_p['type'] = 'human'
        df_p['normalize_name'] = df_p['personLabel'].astype(str).apply(unidecode).str.lower()
        df_p.rename(columns=cols_person, inplace=True)

        # -- XỬ LÝ OBJECT --
        cols_object = {
            'object': 'id',
            'objectLabel': 'name',
            'objectDescription': 'description',
            'objectType': 'type'
        }
        valid_o_cols = [c for c in cols_object.keys() if c in df.columns]
        df_o = df[valid_o_cols].drop_duplicates(subset=['object']).copy()
        df_o['normalize_name'] = df_o['objectLabel'].astype(str).apply(unidecode).str.lower()
        df_o.rename(columns=cols_object, inplace=True)

        # -- VÌ PERSON VÀ OBJECT ĐỀU LÀ NODE NÊN GỘP LẠI ĐỂ THÊM 1 LẦN
        df_all = pd.concat([df_p, df_o], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['id'], keep= 'first')
        df_all = df_all.set_index('id')

        # Chuyển DF thành Dict of Dicts
        # orient='index' tạo ra: {ID_Node: {attr1: val1, attr2: val2}}
        node_attrs = df_all.to_dict(orient='index')
        return node_attrs

    def build_graph(self, df):
        # 1. Tạo đồ thị cơ bản
        G = nx.from_pandas_edgelist(
            df,
            source='person',
            target='object',
            edge_attr= 'relationshipLabel',
            create_using=nx.DiGraph
        )
        print(f"Graph Stat: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


        # --- CẬP NHẬT VÀO NETWORKX ---
        node_attrs = self._create_attribute_node(df)
        nx.set_node_attributes(G, node_attrs)
        print("Đã cập nhật thuộc tính Node thành công.")
        return G

    def run_transformer(self, raw_dir = None, force_data = True ):
        """
        Hàm điều phối chính (Orchestrator).
        """

        # --- BƯỚC 1: LẤY DỮ LIỆU CẠNH (EDGES) ---

        print(f"🚀 Chạy pipeline từ Raw Directory: {raw_dir}")
        if force_data:
            df = self._ingest_json_to_parquet(raw_dir)
            df = self._clean_and_procces_data(df)
        else:
            df = self._ingest_json_to_parquet(CLEAN_DATA_PATH)

        os.makedirs(os.path.dirname(str(CLEAN_DATA_PATH)), exist_ok=True)
        df.to_parquet(str(CLEAN_DATA_PATH), engine='pyarrow', compression='snappy')

        relationship_graph = self.build_graph(df)

        return relationship_graph




