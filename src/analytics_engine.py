import networkx as nx

class AnalyticsEngine:
    def __init__(self, G_full):
        self.graph = G_full 
        self.graph_lcc = None 
        self.analytics_results = {} # (Lưu kết quả offline)

    def _get_person_id(self, name_to_find):
        for node_id, data in self.graph.nodes(data = True):
            current_name = data.get('name','').lower()
            if current_name == name_to_find.lower():
               return node_id
        return None	

    # --- Module 1 (BFS) ---
    def find_path(self, name_a, name_b):
        id_a = self._get_person_id(name_a)
        id_b = self._get_person_id(name_b)
        if id_a is None: return None, f"Không tìm thấy tên '{name_a}'."
        if id_b is None: return None, f"Không tìm thấy tên '{name_b}'."
        if id_a == id_b: return None, "Bạn đã nhập hai tên giống nhau."
        try:
          path_ids = nx.shortest_path(self.graph, source = id_a, target = id_b)
        except nx.NetworkXNoPath:
          return None, f"Không tìm thấy liên kết giữa {name_a} và {name_b}."
        except Exception as e:
          return None, f"Lỗi không xác định: {e}"
        path_names = [self.graph.nodes[id]['name'] for id in path_ids]
        return path_ids, path_names
    # --- Module 4 (Ego) ---
    def get_ego_network(self, name_a):
        # Logic: Lấy ID, lấy neighbors, G.subgraph, trả về G_ego
        pass

    # --- Module 3 (Analytics Offline) ---
    def calculate_offline_stats(self):
        # (Hàm này chạy RẤT LÂU)
        # Logic: Tìm G_lcc. Tính Centrality, Community, Avg Path...
        # Lưu kết quả vào self.analytics_results (một dict)
        # (Hàm này sẽ được chạy 1 lần)
        pass

    # --- Module 2 (XAI) ---
    def get_feature_importances(self, model_features_file):
        # (Logic đọc file model_features.json)
        pass
