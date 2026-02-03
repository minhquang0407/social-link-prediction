from core.interfaces import ISearchEngine
from rapidfuzz import process, fuzz
from unidecode import unidecode
from config.settings import SEARCH_THRESHOLD

from collections import defaultdict
import pandas as pd

def build_search_index(df_nodes):
    print("LOG: Đang xây dựng chỉ mục tìm kiếm...")

    # 1. Chuẩn hóa Key (Đây là đoạn tốn thời gian nhất bắt buộc phải làm)
    # Dùng .values để lấy mảng Numpy ngay lập tức
    keys = df_nodes['name'].astype(str).str.lower().apply(unidecode).values
    # Lấy mảng ID
    ids = df_nodes.index
    types = df_nodes['type'].astype(str).values
    search_map = defaultdict(list)

    # zip trên numpy array chạy ở tốc độ C
    for k, i, t in zip(keys, ids, types):
        search_map[k].append(i)

    final_map = dict(search_map)

    print(f"LOG: Đã tạo xong Index với {len(final_map)} từ khóa.")
    return final_map


class RapidFuzzySearch(ISearchEngine):
    def __init__(self, df_nodes):

        if df_nodes is not None and not df_nodes.empty:
            self.search_map = build_search_index(df_nodes)
            self.all_keys = list(self.search_map.keys())
            self.lookup = df_nodes
        else:
            self.search_map, self.all_keys = {}, []

    def search_best(self, query: str, threshold= SEARCH_THRESHOLD):
        """
        Thực thi tìm kiếm mờ.
        """
        if not query: return None, 0

            # Chuẩn hóa input người dùng ngay lúc tìm kiếm
        clean_query = unidecode(str(query)).lower().strip()
        if clean_query in self.search_map:
            # Trả về ngay lập tức với điểm số tuyệt đối 100
            return self.search_map[clean_query], 100
        # Dùng RapidFuzz để so khớp với danh sách keys
        candidates = process.extract(
            clean_query,
            self.all_keys,
            scorer=fuzz.WRatio,
            limit=10,
            score_cutoff=threshold
        )
        if not candidates:
            return None, 0
        best_candidate = None
        best_final_score = -1

        for key, score, _ in candidates:
            clean_key = unidecode(str(key)).lower()

            # --- Logic phạt/thưởng ---
            final_score = score

            # Phạt nặng nếu kết quả quá ngắn so với query
            len_ratio = len(clean_key) / len(clean_query)
            if len_ratio < 0.5:
                final_score -= 30

            # Thưởng nếu bắt đầu đúng (Prefix match)
            if clean_key.startswith(clean_query):
                final_score += 20

            # Thưởng nếu chứa trọn vẹn (Substring match)
            elif clean_query in clean_key:
                final_score += 10

            # Cập nhật người tốt nhất
            if final_score > best_final_score:
                best_final_score = final_score
                best_candidate = key

        # Trả về kết quả
        if best_candidate and best_final_score >= threshold:
            return self.search_map[best_candidate], best_final_score

        return None, 0

    def quick_get_id(self, query_name):
        print(f"Đang tìm: '{query_name}'...")
        candidates, score = self.search_best(query_name)

        if not candidates:
            return None

        # TRƯỜNG HỢP 1: Chỉ có 1 kết quả
        if len(candidates) == 1:
            person_idx = candidates[0]
            print(f"-> Đã chọn: {self.lookup.at[person_idx, 'name']} (ID: {person_idx})")
            return int(person_idx)

        # TRƯỜNG HỢP 2: Có nhiều người trùng tên
        print(f"Có {len(candidates)} người tên giống vậy. Vui lòng chọn:")
        for i, p in enumerate(candidates):
            print(f"   [{i}] {self.lookup.at[p, 'name']} "
                  f"({self.lookup.at[p, 'type']}) "
                  f"(DESC: {self.lookup.at[p, 'description']} - ID: {p}"
            )

        try:
            choice = int(input("Nhập số thứ tự (index): "))
            print(f"-> Đã chọn {self.lookup.at[candidates[choice], 'name']} (ID: {candidates[choice]})")
            return int(candidates[choice])
        except (ValueError, IndexError):
            print("Chọn sai!!!")
            return None

    def search_forward_pyg(self, name):
        """Hàm tìm kiếm dựa trên tên cho pyg data
        input: name
        output: (type, pyg_id)
        """
        id = self.quick_get_id(name)
        pyg_id = self.lookup.at[id,'pyg_id']
        type = self.lookup.at[id,'type']
        return type, pyg_id

    def search_backward_pyg(self, type, pyg_id):
        lookup_backward = self.lookup.set_index(['type','pyg_id'])

        id = lookup_backward.loc[(type,pyg_id),'id']

        return id