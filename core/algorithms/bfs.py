<<<<<<< HEAD
import networkx as nx
from core.interfaces import IPathFinder


class NetworkXBFSFinder(IPathFinder):
    """
    Triển khai thuật toán tìm kiếm đường đi ngắn nhất (BFS)
    sử dụng thư viện NetworkX.
    """

    def find_path(self, graph, start_id, end_id):
        try:
            path_ids = nx.shortest_path(graph, source=start_id, target=end_id)

            path_names = [graph.nodes[n].get('name', str(n)) for n in path_ids]

            return path_ids, path_names

        except nx.NetworkXNoPath:
            return [], []

        except Exception as e:
            print(f"Lỗi thuật toán BFS: {e}")
            return [], []

=======
import igraph as ig
import pickle
import math
import os
from collections import defaultdict

class PathFinder:
    def __init__(self, graph):
        """
        Khởi tạo: Load graph và tính toán trước trọng số.
        """
        self.graph = graph
        self.weights = self._precompute_weights()

    def _precompute_weights(self):
        """
        Tính trọng số (Cost) cho cạnh.
        Cost càng cao = Càng khó đi qua = Mối quan hệ càng yếu.

        Logic tổng hợp:
        1. Blacklist: Cost = Vô cực.
        2. Hub Penalty: Node càng to (như Vietnam, Google) -> Cost càng cao.
        3. Generation Gap: Hai người cách nhau quá 20 tuổi -> Cost tăng mạnh.
        """
        # 1. Chuẩn bị dữ liệu (Bulk Fetch để tối ưu tốc độ)
        blacklist = {'influenced_by'}

        # Lấy danh sách Label của cạnh
        rels = self.graph.es['relationship_label']

        # Lấy danh sách cạnh dưới dạng tuple (source_index, target_index)
        # Cách này nhanh hơn truy cập e.source, e.target trong vòng lặp
        edges_list = self.graph.get_edgelist()

        # Lấy thuộc tính Node (xử lý trường hợp thiếu dữ liệu)
        try:
            node_types = self.graph.vs['type']
            # fillna(0) hoặc đảm bảo dữ liệu đầu vào là số
            birth_years = self.graph.vs['birth_year']
        except KeyError:
            # Fallback an toàn nếu chưa có thuộc tính
            node_types = ['unknown'] * self.graph.vcount()
            birth_years = [0] * self.graph.vcount()

        # Lấy In-Degree để tính Hub Penalty
        degrees = self.graph.degree(mode='in')

        weights = []

        # 2. Vòng lặp tính toán (Duyệt qua 9 triệu cạnh)
        for i, (src, dst) in enumerate(edges_list):
            rel = rels[i]

            # --- LOGIC 1: BLACKLIST (Chặn tuyệt đối) ---
            if rel in blacklist:
                weights.append(float('inf'))
                continue

            # --- LOGIC 2: BASE WEIGHT (Tránh Hub) ---
            # Công thức: log(degree + 1)
            # Ví dụ: Hub 1M follow -> weight ~13.8. Người thường 100 follow -> weight ~4.6
            target_deg = degrees[dst]
            base_weight = math.log(target_deg + 1)

            # --- LOGIC 3: AGE GAP PENALTY (Phạt chênh lệch tuổi) ---
            age_penalty = 0.0

            # Chỉ xét nếu cả Source và Target đều là 'human'
            if node_types[src] == 'human' and node_types[dst] == 'human':
                y_src = birth_years[src]
                y_dst = birth_years[dst]

                # Kiểm tra dữ liệu hợp lệ (khác None, khác 0, và năm sinh > 1000)
                if y_src and y_dst and y_src > 1000 and y_dst > 1000:
                    age_diff = abs(y_src - y_dst)

                    # QUY TẮC PHẠT:
                    # - Dưới 15 tuổi: Không phạt (cùng thế hệ).
                    # - Trên 15 tuổi: Bắt đầu phạt.
                    # - Cứ mỗi 10 năm chênh lệch cộng thêm 2 điểm weight.
                    if age_diff > 15:
                        # Ví dụ: Cách 35 tuổi -> (35 - 15)/5 = 4.0 điểm cộng thêm
                        # Weight lúc này sẽ ngang ngửa việc đi qua một Hub trung bình
                        age_penalty = (age_diff - 15) / 5.0

            # Tổng hợp trọng số
            final_weight = base_weight + age_penalty
            weights.append(final_weight)

        return weights
    def find_shortest_path(self, start_idx, end_idx):
        """
        Tìm đường đi ngắn nhất giữa 2 Q-ID.
        Trả về: (List[Dict], Message)
        """
        # 1. Map Q-ID sang Index

        if start_idx is None: return None
        if end_idx is None: return None

        try:
            # 2. Tìm đường bằng iGraph C-Core
            paths = self.graph.get_shortest_paths(
                v=start_idx,
                to=end_idx,
                weights=self.weights,
                mode='all'
            )

            path_indices = paths[0] # Lấy đường đầu tiên

            return path_indices


        except Exception:
            return None


    def find_shortest_paths_batch(self, pairs, weight=False ):
        """
        Nhập: list of tuples [(s1, t1), (s2, t2), ...]
        Trả về: list of paths
        """
        # Gom nhóm để tối ưu: Các cặp chung id_a chỉ cần chạy Dijkstra 1 lần
        if not weight:
            weight = None
        else:
            weight = self.weights
        grouped = defaultdict(list)
        for i, (s, t) in enumerate(pairs):
            grouped[s].append((t, i)) # Lưu index để trả về đúng thứ tự
            
        results = [None] * len(pairs)
        
        for start_node, targets_info in grouped.items():
            target_nodes = [info[0] for info in targets_info]
            indices = [info[1] for info in targets_info]
            
            # iGraph tìm từ 1 nguồn đến nhiều đích (Multi-destination)
            # Cực nhanh trên đồ thị 4 triệu nút
            paths = self.graph.get_shortest_paths(
                v=start_node, to=target_nodes, weights=weight, output="vpath", mode='all'
            )
            
            for path, original_idx in zip(paths, indices):
                results[original_idx] = path
        return results
>>>>>>> 9de2b1b (FINAL)
