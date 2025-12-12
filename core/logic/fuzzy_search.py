from core.interfaces import ISearchEngine
from rapidfuzz import process, fuzz, utils
from unidecode import unidecode


class RapidFuzzySearch(ISearchEngine):
    def __init__(self, search_index_data):

        if search_index_data:
            self.search_map = search_index_data
            self.all_keys = self.search_map.keys()
        else:
            self.search_map, self.all_keys = {}, []

    def quick_get_id(service, name_input):
        print(f"🔎 Đang tìm: '{name_input}'...")
        candidates, score = service.search_best(name_input)

        if not candidates:
            print(f"❌ Không tìm thấy ai tên là '{name_input}'")
            return None

        # TRƯỜNG HỢP 1: Chỉ có 1 kết quả (hoặc nhập chính xác tên)
        # Ví dụ: Nhập "Son Tung M-TP" và chỉ có 1 ông -> Lấy luôn
        if len(candidates) == 1:
            person = candidates[0]
            print(f"✅ Đã chọn: {person['name']} (ID: {person['id']})")
            return person['id']

        # TRƯỜNG HỢP 2: Có nhiều người trùng tên (VD: 5 ông tên "Nguyen Van A")
        # Phải hỏi người dùng chọn ông nào
        print(f"⚠️ Có {len(candidates)} người tên giống vậy. Vui lòng chọn:")
        for i, p in enumerate(candidates):
            print(f"   [{i}] {p['name']} ({p.get('type', 'Unknown')}) - ID: {p['id']}")

        try:
            choice = int(input("👉 Nhập số thứ tự (index): "))
            return candidates[choice]['id']
        except (ValueError, IndexError):
            print("❌ Chọn sai!")
            return None
    def search_best(self, query: str, threshold=60):
        """
        Thực thi tìm kiếm mờ.
        """
        if not query: return None, 0

            # Chuẩn hóa input người dùng ngay lúc tìm kiếm
        clean_query = unidecode(str(query)).lower()
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

