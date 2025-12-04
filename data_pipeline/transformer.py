import networkx as nx
import pandas as pd
import json
import pickle
import unicodedata
from collections import defaultdict

class GraphTransformer:
    def __init__(self):
        """
        Khởi tạo đồ thị rỗng và dictionary chứa sở thích tạm thời
        """
        self.G = nx.Graph()
        self.person_interests_map = defaultdict(set)  # Dùng defaultdict(set)
        print("GraphTransformer initialized.")


    def clean_data(self, df):
        """
        [MỚI] Hàm làm sạch dữ liệu:
        Loại bỏ các dòng mà tên người (personLabel) bị lỗi hiển thị thành ID (ví dụ: Q12345).
        """
        if df.empty or 'personLabel' not in df.columns:
            return df

        original_count = len(df)

        # Regex: ^Q\d+$ nghĩa là Bắt đầu bằng Q, theo sau là số, và kết thúc chuỗi.
        # na=False nghĩa là nếu dữ liệu trống thì không coi là lỗi -> giữ lại.
        mask_error = df['personLabel'].astype(str).str.fullmatch(r'^Q\d+$', na=False)

        # Giữ lại những dòng KHÔNG bị lỗi (dấu ~ là phủ định)
        df_clean = df[~mask_error]

        removed_count = original_count - len(df_clean)
        if removed_count > 0:
            print(f"    [CLEAN] Đã loại bỏ {removed_count} dòng có tên lỗi dạng ID (Q123...).")
        return df_clean

    def _load_and_flatten_json(self, raw_filepath):
        """
        Load file JSON, làm phẳng VÀ LÀM SẠCH (CLEAN) ngay lập tức.
        """
        try:
            with open(raw_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data_to_normalize = None
            if isinstance(data, dict):
                data_to_normalize = data.get('results', {}).get('bindings', [])
            elif isinstance(data, list):
                data_to_normalize = data

            if not data_to_normalize:
                return pd.DataFrame()

            # 1. Làm phẳng JSON -> DataFrame
            df = pd.json_normalize(data_to_normalize)

            # 2. Làm sạch tên cột (bỏ .value)
            df.columns = [col.replace('.value', '') for col in df.columns]

            # THÊM DÒNG NÀY]
            # 3. Gọi hàm làm sạch dữ liệu ngay tại đây
            df = self.clean_data(df)

            return df

        except Exception as e:
            # Lưu ý: Một số file có thể không có cột personLabel,
            # hàm clean_data đã có check cột nên sẽ không gây lỗi.
            print(f"LỖI LOAD FILE {raw_filepath}: {e}")
            return pd.DataFrame()

    def _normalize_name(self, name):
        """
        Chuẩn hóa tên thành không dấu
        """
        if pd.isna(name) or not name:
            return "Unknown"

        # Loại bỏ dấu và chuyển về chữ thường
        name_str = str(name).strip()
        normalized = unicodedata.normalize('NFKD', name_str)
        normalized = ''.join([c for c in normalized if not unicodedata.combining(c)])
        return normalized

    def _aggregate_interests(self, file_list):
        """
        Gom tất cả sở thích từ danh sách file vào self.person_interests_map.
        Không thêm vào đồ thị ngay, chỉ lưu vào Dict để tra cứu sau.
        """
        print("Đang tổng hợp dữ liệu sở thích (Interests)...")
        for filepath, interest_obj_col in file_list:
            df = self._load_and_flatten_json(filepath)

            # Cần cột person và cột chứa sở thích
            if 'person' not in df.columns:
                continue

            # Tìm tên cột label của sở thích
            label_col = f"{interest_obj_col}Label"
            if label_col not in df.columns and 'objectLabel' in df.columns:
                label_col = 'objectLabel'

            if label_col not in df.columns:
                continue

            for _, row in df.iterrows():
                # Lấy ID person bằng cách tách phần cuối của URL
                p_id = row['person'].split('/')[-1]
                raw_interest = row[label_col]

                if p_id and raw_interest and not pd.isna(raw_interest):
                    # Không cần làm sạch label sở thích
                    self.person_interests_map[p_id].add(str(raw_interest))

        print(f"-> Đã tổng hợp sở thích cho {len(self.person_interests_map)} người.")

    def _add_generic_relation(self, df, target_node_type, rel_label):
        """
        Thêm quan hệ chung vào đồ thị.
        """
        # Các cột bắt buộc cơ bản (đã bỏ .value)
        p_id_col = "person"
        p_label_col = "personLabel"
        obj_id_col = "object"
        obj_label_col = "objectLabel"

        # Map các cột thuộc tính bổ sung
        attribute_map = {
            "personDescription": "description",
            "birthYear": "date_of_birth",
            "birthPlaceLabel": "place_of_birth",
            "countryLabel": "nationality"
        }

        # Kiểm tra cột cơ bản
        if p_id_col not in df.columns or obj_id_col not in df.columns:
            print(f"  [SKIP] File thiếu cột ID (person hoặc object).")
            return

        count = 0
        for _, row in df.iterrows():
            # 1. Xử lý PERSON (Node nguồn)
            p_id = row[p_id_col].split('/')[-1]  # Lấy phần cuối của URL

            # Lấy tên và chuẩn hóa
            p_name_raw = row.get(p_label_col, "Unknown")
            p_name = self._normalize_name(p_name_raw)

            # Tạo dict attributes cho Person
            person_attrs = {"name": p_name, "type": "Person"}

            # Thêm các thuộc tính bổ sung
            for col_df, attr_name in attribute_map.items():
                if col_df in df.columns:
                    val = row[col_df]
                    if not pd.isna(val):
                        person_attrs[attr_name] = str(val)

            # Xử lý sở thích: join thành string
            if p_id in self.person_interests_map:
                person_attrs["interests"] = ", ".join(sorted(self.person_interests_map[p_id]))
            else:
                person_attrs["interests"] = ""

            # Add Node Person
            self.G.add_node(p_id, **person_attrs)

            # 2. Xử lý OBJECT (Node đích)
            obj_id = row[obj_id_col].split('/')[-1]  # Lấy phần cuối của URL
            obj_name = row.get(obj_label_col, "Unknown")

            # Không cần chuẩn hóa tên cho object
            if pd.isna(obj_name) or not obj_name:
                obj_name = "Unknown"

            # Add Node Object (Target)
            self.G.add_node(obj_id, name=obj_name, type=target_node_type)

            # 3. Add Edge
            self.G.add_edge(p_id, obj_id, relationship=rel_label)
            count += 1

        print(f"  -> Đã thêm {count} cạnh '{rel_label}'.")

    def build_full_graph(self, config_list, interest_files_config=None):
        """
        Hàm chính xây dựng đồ thị
        """
        # BƯỚC 1: Xử lý sở thích trước
        if interest_files_config:
            self._aggregate_interests(interest_files_config)

        # BƯỚC 2: Xử lý các quan hệ chính
        print("\nBẮT ĐẦU XÂY DỰNG ĐỒ THỊ...")
        for path, target_type, rel_label in config_list:
            print(f"Đang xử lý file: {path} (Quan hệ: {rel_label})")

            df = self._load_and_flatten_json(path)

            if not df.empty:
                self._add_generic_relation(df, target_type, rel_label)
            else:
                print(f"  [WARN] File {path} rỗng hoặc lỗi.")

    def save_graph(self, output_path):
        print(f"Đang lưu đồ thị vào {output_path}...")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
            print(f"Đã lưu thành công! (Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()})")
        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")


if __name__ == "__main__":
    transformer = GraphTransformer()

    # 1. Cấu hình file Sở thích (Interests) - Dựa trên ảnh
    # Cấu trúc: ("đường_dẫn_file", "tên_cột_trong_file_gốc_để_ghép_label")
    # Lưu ý: Bạn cần mở file json ra xem cái key chứa ID sở thích là gì (ví dụ: 'instrument', 'genre', 'field')
    interest_configs = [
        ("data/raw_data_interest_instrument.json", "instrument"),
        ("data/raw_data_interest_genre.json", "genre"),
        ("data/raw_data_interest_field.json", "field"),
    ]

    # 2. Cấu hình file Quan hệ (Edges) - Dựa trên ảnh
    # Cấu trúc: ("đường_dẫn_file", "Loại_Node_Đích", "Tên_Quan_Hệ")
    relation_configs = [
        # --- Quan hệ gia đình/xã hội ---
        ("data/raw_data_spouse.json", "Person", "spouse"),
        ("data/raw_data_mother.json", "Person", "mother"),
        ("data/raw_data_father.json", "Person", "father"),
        ("data/raw_data_sibling.json", "Person", "sibling"),

        # --- Quan hệ nghề nghiệp/tác phẩm ---
        ("data/raw_data_film_actor.json", "Film", "acted_in"),
        ("data/raw_data_film_director.json", "Film", "directed"),
        ("data/raw_data_film_screenwriter.json", "Film", "wrote"),
        ("data/raw_data_music_composer.json", "MusicalWork", "composed"),
        ("data/raw_data_music_lyricist.json", "MusicalWork", "wrote_lyrics"),
        ("data/raw_data_performer.json", "MusicalWork", "performed"),
        ("data/raw_data_author.json", "Book", "wrote_book"),

        # --- Quan hệ tổ chức/tôn giáo/tư tưởng ---
        ("data/raw_data_party.json", "PoliticalParty", "member_of_party"),
        ("data/raw_data_religion.json", "Religion", "religion"),
        ("data/raw_data_group.json", "Group", "member_of_group"),
        ("data/raw_data_ideology.json", "Ideology", "political_ideology"),

        # --- Quan hệ học vấn/khác ---
        ("data/raw_data_advisor.json", "Person", "doctoral_advisor"),
        ("data/raw_data_influenced.json", "Person", "influenced_by"),
    ]

    # 3. Chạy Pipeline
    # Code sẽ tự động bỏ qua các file không tìm thấy hoặc lỗi
    transformer.build_full_graph(
        config_list=relation_configs,
        interest_files_config=interest_configs
    )

    # 4. Lưu kết quả
    transformer.save_graph("data/G_full.gpickle")