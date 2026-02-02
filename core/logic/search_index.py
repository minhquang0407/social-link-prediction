
import sys
from pathlib import Path
<<<<<<< HEAD
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from core.interfaces import IGraphRepository
=======
>>>>>>> 9de2b1b (FINAL)

from collections import defaultdict


def build_search_index(G):
    """
    Hàm độc lập tạo chỉ mục tìm kiếm từ đồ thị G.
<<<<<<< HEAD
    Dựa trên thuộc tính 'normalized_name' đã được tính trước.
=======
    Dựa trên thuộc tính 'normalize_name' đã được tính trước.
>>>>>>> 9de2b1b (FINAL)
    """
    print("LOG: Đang xây dựng chỉ mục tìm kiếm (Tối ưu)...")
    search_map = defaultdict(list)

    # Lặp qua các node
    for node_id, data in G.nodes(data=True):
        # 1. Lấy key chuẩn hóa (đã tính sẵn ở Transformer)
<<<<<<< HEAD
        clean_key = data.get('normalized_name')

        # 2. Lấy tên gốc
        original_name = str(data.get('name', 'Unknown'))

        if clean_key and original_name:
            node_info = {
                "id": node_id,
                "name": original_name,
                "description": str(data.get('description', ''))
            }

            # 3. Thêm vào map
            search_map[clean_key].append(node_info)

    # Trả về cả Map và List các Keys (để RapidFuzz dùng)
    return search_map, list(search_map.keys())
=======
        if data['type'] == 'person':
            clean_key = data.get('normalize_name')

            # 2. Lấy tên gốc
            original_name = str(data.get('name', 'Unknown'))

            if clean_key and original_name:
                node_info = {
                    "id": node_id,
                    "name": original_name,
                    "description": str(data.get('description', '')),
                    "type": data.get('type', 'Unknown')
                }

                # 3. Thêm vào map
                search_map[clean_key].append(node_info)

    # Trả về cả Map (để RapidFuzz dùng)
    return search_map


>>>>>>> 9de2b1b (FINAL)
