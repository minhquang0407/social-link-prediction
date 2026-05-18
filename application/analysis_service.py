import igraph
from core.algorithms.bfs import PathFinder
from core.interfaces import ISearchEngine
import numpy as np
import igraph as ig
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from tqdm import tqdm
import time
import math
import warnings

# Tắt warning nếu đồ thị bị phân mảnh (không tìm thấy đường đi)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="igraph")

# ==========================================
# 1. GLOBAL WORKER CONTEXT (Zero-Copy)
# ==========================================
# Các biến này sẽ được chia sẻ cho các process con mà không cần copy (trên Linux)
global_graph = None
global_weights = None
global_is_human_arr = None


def init_worker(graph_obj, weights_obj, is_human_arr):
    """
    Khởi tạo môi trường cho Process con.
    Gán các object lớn vào biến toàn cục để truy cập nhanh.
    """
    global global_graph, global_weights, global_is_human_arr
    global_graph = graph_obj
    global_weights = weights_obj
    global_is_human_arr = is_human_arr


def process_batch_task(batch_data):
    """
    Xử lý một lô (batch) các cặp điểm.
    Logic: Tìm đường (One-to-Many) -> Tính Human Count (Vectorized) -> Trả về Degree.
    """
    # 1. Gom nhóm theo Source Node để tối ưu Dijkstra
    grouped = defaultdict(list)
    for orig_idx, s, t in batch_data:
        grouped[s].append((t, orig_idx))

    results = []

    # 2. Xử lý từng nhóm source
    for start_node, targets_info in grouped.items():
        target_nodes = [t[0] for t in targets_info]
        original_indices = [t[1] for t in targets_info]

        # Gọi C-layer của igraph (Nhanh hơn gọi từng cái)
        paths = global_graph.get_shortest_paths(
            v=start_node,
            to=target_nodes,
            weights=global_weights,
            output="vpath",
            mode='all'
        )

        # 3. Tính toán Vector hóa (Numpy) thay vì Loop Python
        for path, orig_idx in zip(paths, original_indices):
            # Nếu không có đường đi (path rỗng)
            if not path:
                results.append((orig_idx, 0))
                continue

            # TỐI ƯU: Truy xuất mảng Numpy bằng danh sách index (Fancy Indexing)
            # Nhanh gấp ~50 lần so với sum(list comprehension)
            human_count = global_is_human_arr[path].sum()

            degree_val = max(0, human_count - 1)
            results.append((orig_idx, degree_val))

    return results
class AnalysisService:
    def __init__(self, graph, search_engine: ISearchEngine = None):
        """
        Service quản lý việc Phân tích.
        - graph: Đồ thị Igraph.
        - search_engine: Đối tượng thực thi việc tìm kiếm (đã được khởi tạo và build index).
        """
        print("Analysis: Đang khởi tạo bộ máy phân tích!", flush=True)
        self.graph = graph
        self.path_finder = PathFinder(self.graph)
        self.search_engine = search_engine
        is_human_bool = np.array(graph.vs['type']) == 'human'
        self.is_human_arr = is_human_bool.astype(np.int8)
        print("Analysis: Sẵn sàng!",flush=True)

    def nodes_info(self, query_name):
        id = self.search_engine.quick_get_id(query_name)
        for key, value in self.graph.vs[id]:
            print(f"{key}: {value if value != '' else 'unknown'}")

    def find_connection(self, id_a, id_b, draw = False):
        """
        Tìm đường đi giữa 2 người.
        """

        normalize = lambda x: x if isinstance(x, int) else self.search_engine.quick_get_id(x)

        id_a = normalize(id_a)
        id_b = normalize(id_b)

        if id_a == id_b:
            success = False
            message = "Bạn đã nhập cùng một người."
            return {"success": success, "message": message}
        # 2. Kiểm tra tồn tại trong đồ thị
        if self.graph is None:
            message = "Chưa có dữ liệu đồ thị."
            print("Chưa có dữ liệu đồ thị.")
            return {"success": False, "path": [], "message": message}

        if id_a is None:
            message = f"Chọn sai hoặc ID '{id_a}' không tồn tại."
            print(message)
            return {"success": False, "path": [], "message": message}
        if id_b is None:
            message = f"Chọn sai hoặc ID '{id_b}' không tồn tại."
            print(message)
            return {"success": False,"path": [], "message": message }

        # 3. Gọi thuật toán BFS
        result_path = []

        path_indices  = self.path_finder.find_shortest_path(id_a, id_b)
        if path_indices is not  None:
            nodes_on_path = self.graph.vs[path_indices]
            for i, node in enumerate(nodes_on_path):
                node_info = {
                    'idx': node.index,
                    'qid': node['name'],
                    'name': node['label'],
                    'type': node['type']
                }

                # Nếu không phải node cuối, tìm thông tin cạnh nối với node tiếp theo
                edge_label = None
                if i < len(path_indices) - 1:
                    u = path_indices[i]
                    v = path_indices[i + 1]

                    # Thử tìm cạnh chiều thuận u -> v
                    eid = self.graph.get_eid(u, v, error=False)

                    # Nếu không có (-1), thử tìm cạnh chiều ngược v -> u
                    if eid == -1:
                        eid = self.graph.get_eid(v, u, error=False)
                        direction = "incoming"  # Đánh dấu là đi ngược
                    else:
                        direction = "outgoing"  # Đánh dấu là đi xuôi

                    # Nếu tìm thấy cạnh (dù chiều nào)
                    if eid != -1:
                        edge_label = self.graph.es[eid]['relationship_label']
                        node_info['next_rel'] = edge_label
                        node_info['direction'] = direction
                    else:
                        node_info['next_rel'] = 'unknown'

                result_path.append(node_info)
            message = f"Đã tìm thấy liên kết!"
            print(message)
            if draw:
                self.draw_path(result_path)
            return {
                "success": True,
                "path_id": path_indices,
                "path_detail": result_path,
                "message": message
            }

        # 4. Xử lý trường hợp không có đường đi
        name_a = self.graph.vs[id_a].get('name', id_a)
        name_b = self.graph.vs[id_b].get('name', id_b)
        massage =  f"Không tìm thấy liên kết giữa **{name_a}** và **{name_b}**."
        return {
            "success": False,
            "path": [],
            "message": massage
        }
    
    def draw_path(self, path_data):
        if not path_data: return

        print("\nKẾT QUẢ TÌM ĐƯỜNG:")
        for i, step in enumerate(path_data):
            # Icon đại diện type
            icon = "👤" if step['type'] == 'human' else "🏢" if step['type'] == 'organization' else "🟢"

            print(f"{i + 1:02d}. {icon} {step['name']} [{step.get('qid', '')}]")

            # Vẽ mũi tên kết nối nếu không phải node cuối
            if i < len(path_data) - 1:
                rel = step.get('next_rel', 'liên kết với')
                print(f"      │")
                print(f"      ▼ ({rel})")



    def find_degrees(self, pair_list, weight=False):
        # 1. Gọi hàm tìm đường (Hàm thứ 2)
        all_paths = self.path_finder.find_shortest_paths_batch(pair_list, weight)
        
        # 2. Tính bậc cho từng đường đi
        final_degrees = []
        for path in all_paths:
            if not path or len(path) == 0:
                final_degrees.append(0)
                continue

            human_count = sum(self.is_human_arr[n] for n in path)
            degree_val = max(0, human_count - 1)
            final_degrees.append(degree_val)
        return final_degrees

    def compute_degrees(self, pairs, batch_size=2000, n_jobs=None):
        """
        Phiên bản tối ưu hóa chạy song song cho 100.000 cặp.
        """
        total_pairs = len(pairs)
        # Gắn index để đảm bảo kết quả trả về đúng thứ tự ban đầu
        indexed_pairs = [(i, p[0], p[1]) for i, p in enumerate(pairs)]

        # Chia batch
        batches = [
            indexed_pairs[i: i + batch_size]
            for i in range(0, total_pairs, batch_size)
        ]

        if n_jobs is None:
            n_jobs = mp.cpu_count()


        final_results = [0] * total_pairs

        # Sử dụng ProcessPoolExecutor
        with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=init_worker,
                initargs=(self.graph, None, self.is_human_arr)
        ) as executor:

            # Executor map + Tqdm progress bar
            futures = list(tqdm(
                executor.map(process_batch_task, batches),
                total=len(batches),
                desc="Tiến độ xử lý",
                unit="batch"
            ))

            # Tổng hợp kết quả
            for batch_res in futures:
                for idx, degree in batch_res:
                    final_results[idx] = degree

        return np.array(final_results)