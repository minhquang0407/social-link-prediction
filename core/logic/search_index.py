
import sys
from pathlib import Path

from collections import defaultdict


def build_search_index(G):
    """
    Hàm độc lập tạo chỉ mục tìm kiếm từ đồ thị G.
    Dựa trên thuộc tính của igraph.
    """
    print("LOG: Đang xây dựng chỉ mục tìm kiếm (Tối ưu)...")
    search_map = defaultdict(list)

    # Lặp qua các node của igraph
    for v in G.vs:
        if v['type'] == 'human':
            original_name = str(v['label'] if v['label'] else 'Unknown')
            clean_key = v['name']

            if clean_key and original_name:
                node_info = {
                    "id": v.index,
                    "name": original_name,
                    "description": str(v['description'] if 'description' in v.attributes() else ''),
                    "type": v['type']
                }

                # Thêm vào map
                search_map[clean_key].append(node_info)

    # Trả về cả Map
    return search_map


