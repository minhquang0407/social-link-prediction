import pandas as pd
import networkx as nx
import json
import pickle
from pathlib import Path


def load_sparql_result(raw_filepath: str) -> pd.DataFrame:
    """
    Đọc file JSON thô từ SPARQL và "làm phẳng" (flatten) cấu trúc
    results.bindings để tạo DataFrame.
    [cite_start][cite: 89]
    """
    print(f"Đang đọc file: {raw_filepath}")
    try:
        # 1. Đọc file JSON
        with open(raw_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # [cite_start]2. Làm phẳng cấu trúc lồng nhau [cite: 90]
        # Đây là bước tạo ra các cột 'person.value', 'personLabel.value', v.v.
        df = pd.json_normalize(data['results']['bindings'])

        # 3. Dọn dẹp dữ liệu (PHẦN ĐƯỢC SỬA)
        # Chúng ta cần đổi tên các cột có đuôi '.value'

        # Tạo một dictionary để map tên cũ sang tên mới
        # Ví dụ: {'person.value': 'person', 'personLabel.value': 'personLabel'}
        rename_map = {}
        for col in df.columns:
            if col.endswith('.value'):
                # Tạo tên mới bằng cách bỏ đi 4 ký tự cuối ('.value')
                new_name = col[:-6]  # Bỏ '.value'
                rename_map[col] = new_name

        # 4. Áp dụng đổi tên cột
        df = df.rename(columns=rename_map)

        # 5. Lọc DataFrame:
        # Giờ đây chúng ta chỉ giữ lại các cột đã được đổi tên (các cột "sạch")
        # và bỏ đi các cột .type, .xml:lang

        # Lấy danh sách các tên cột "sạch" (ví dụ: ['person', 'spouse', 'personLabel', 'spouseLabel'])
        final_clean_columns = list(rename_map.values())

        # Lọc DataFrame để chỉ giữ lại các cột này
        # (Thêm_ 'if col in df.columns' để phòng trường hợp query không trả về cột nào đó)
        df_cleaned = df[[col for col in final_clean_columns if col in df.columns]]

        print(f"Làm phẳng {raw_filepath} thành công. (Số dòng: {len(df_cleaned)})")
        print(f"Các cột đã xử lý: {list(df_cleaned.columns)}")

        return df_cleaned

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {raw_filepath}")
        return pd.DataFrame()  # Trả về DF rỗng
    except KeyError:
        print(f"LỖI: Cấu trúc JSON không đúng, không tìm thấy 'results.bindings'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"LỖI không xác định khi xử lý file {raw_filepath}: {e}")
        return pd.DataFrame()


def build_graph_v0_1(df: pd.DataFrame) -> nx.Graph:
    """
    Xây dựng đồ thị v0.1 từ DataFrame (vợ/chồng).
    """
    # Khởi tạo một đồ thị vô hướng
    G = nx.Graph()

    # --- TÊN CỘT ĐÃ ĐƯỢC XÁC NHẬN ---
    # Dựa trên code wikidata_collector.py (hàm get_v0_1_data_spouse)
    col_person_id = 'person'
    col_person_name = 'personLabel'
    col_relation_id = 'spouse'
    col_relation_name = 'spouseLabel'

    print("Bắt đầu xây dựng đồ thị v0.1...")

    # Kiểm tra xem DataFrame có đủ các cột này không
    if not all(col in df.columns for col in [col_person_id, col_person_name, col_relation_id, col_relation_name]):
        print(f"LỖI: DataFrame thiếu các cột cần thiết.")
        print(f"Cần có: ['person', 'personLabel', 'spouse', 'spouseLabel']")
        print(f"Các cột hiện có: {list(df.columns)}")
        return G  # Trả về đồ thị rỗng

    # Lặp qua từng hàng (từng cặp quan hệ) trong DataFrame
    for _, row in df.iterrows():
        try:
            person_id = row[col_person_id]
            person_name = row[col_person_name]
            spouse_id = row[col_relation_id]
            spouse_name = row[col_relation_name]

            # Thêm 2 nút vào đồ thị.
            G.add_node(person_id, name=person_name)
            G.add_node(spouse_id, name=spouse_name)

            # Thêm 1 cạnh (edge) nối 2 nút
            G.add_edge(person_id, spouse_id, relationship="spouse")
        except Exception as e:
            # Bắt các lỗi khác nếu có (ví dụ: dữ liệu null)
            print(f"Cảnh báo: Bỏ qua hàng bị lỗi: {e}")
            continue

    print(f"Xây dựng đồ thị v0.1 hoàn tất.")
    print(f"-> Tổng số Nút (Nodes): {G.number_of_nodes()}")
    print(f"-> Tổng số Cạnh (Edges): {G.number_of_edges()}")
    return G


def run_v0_1_pipeline():
    """
    Hàm này thực hiện nhiệm vụ của Tuần 2:
    Chạy các hàm đã viết ở Tuần 1 để tạo ra sản phẩm.

    """
    # Đường dẫn file input (do Quân cung cấp)
    RAW_FILE = "raw_data_v0.1_spouse.json"  # [cite: 81]

    # Đường dẫn file output
    OUTPUT_GRAPH = "G_v0.1.gpickle"

    print("--- Bắt đầu Pipeline v0.1 ---")

    # 1. Extract & Transform (Load)
    df = load_sparql_result(RAW_FILE)

    if not df.empty:
        # 2. Build Graph
        G = build_graph_v0_1(df)

        if G.number_of_nodes() > 0:
            # 3. Load (Save) [cite: 105]
            print(f"Đang lưu đồ thị vào {OUTPUT_GRAPH}...")
            with open(OUTPUT_GRAPH, 'wb') as f:
                pickle.dump(G, f)


            print(f"--- Pipeline v0.1 Hoàn tất! Đã lưu đồ thị vào {OUTPUT_GRAPH} ---")
        else:
            print(f"--- Pipeline v0.1 Thất bại! Đồ thị rỗng (có thể do sai tên cột). ---")
    else:
        print(f"--- Pipeline v0.1 Thất bại! Không thể load dữ liệu từ {RAW_FILE} ---")


# Thêm đoạn này vào cuối file để có thể chạy trực tiếp
if __name__ == "__main__":
    run_v0_1_pipeline()