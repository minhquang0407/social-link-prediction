from email import message_from_string

from core.algorithms.bfs import PathFinder
from core.interfaces import ISearchEngine

class AnalysisService:
    def __init__(self, graph, search_engine: ISearchEngine):
        """
        Service quản lý việc Phân tích.
        - graph: Đồ thị Igraph.
        - search_engine: Đối tượng thực thi việc tìm kiếm (đã được khởi tạo và build index).
        """
        self.graph = graph
        self.path_finder = PathFinder(self.graph)
        self.search_engine = search_engine

    def nodes_info(self, query_name):
        id = self.search_engine.quick_get_id(query_name)
        for key, value in self.graph.nodes[id].items():
            print(f"{key}: {value}")

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
                        edge_label = self.graph.es[eid]['relationshipLabel']
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



