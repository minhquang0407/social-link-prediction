from abc import ABC, abstractmethod


# --- 1. HỢP ĐỒNG CHO KHO CHỨA ĐỒ THỊ (REPOSITORY) ---
class IGraphRepository(ABC):
    """
    Giao diện cho việc lưu trữ và truy xuất đồ thị.
    """

    @abstractmethod
    def load_graph(self):
        """
        Tải đồ thị từ nơi lưu trữ lên bộ nhớ.
        Returns:
            Object đồ thị (ví dụ: networkx.Graph) hoặc None nếu lỗi.
        """
        pass

    @abstractmethod
    def save_graph(self, G):
        """
        Lưu đồ thị xuống nơi lưu trữ.
        Args:
            G: Object đồ thị cần lưu.
        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        pass


# --- 2. HỢP ĐỒNG CHO KHO CHỨA MODEL AI (REPOSITORY) ---
class IModelRepository(ABC):
    """
    Giao diện cho việc lưu trữ và truy xuất Model AI.
    """

    @abstractmethod
    def load_model(self, model_architecture=None, device=None):
        """
        Tải model đã huấn luyện.
        Args:
            model_architecture: (Optional) Cấu trúc mạng (cho PyTorch).
            device: (Optional) CPU hoặc CUDA.
        Returns:
            Model object.
        """
        pass

    @abstractmethod
    def save_model(self, model):
        """
        Lưu model đã huấn luyện.
        Args:
            model: Model object cần lưu.
        Returns:
            bool: True nếu thành công.
        """
        pass


# --- 3. HỢP ĐỒNG CHO THUẬT TOÁN TÌM ĐƯỜNG (ALGORITHM) ---
class IPathFinder(ABC):
    """
    Giao diện cho các thuật toán tìm đường (BFS, Dijkstra, A*...).
    """

    @abstractmethod
    def find_path(self, graph, start_id, end_id):
        """
        Tìm đường đi giữa 2 điểm.
        Args:
            graph: Đồ thị.
            start_id: ID điểm bắt đầu.
            end_id: ID điểm kết thúc.
        Returns:
            tuple: (list_ids, list_names) - Danh sách ID và Tên trên đường đi.
                   Trả về ([], []) nếu không tìm thấy.
        """
        pass


# --- 4. HỢP ĐỒNG CHO CÔNG CỤ TÌM KIẾM (SEARCH ENGINE) ---
class ISearchEngine(ABC):
    """
    Giao diện cho việc tìm kiếm mờ (Fuzzy Search).
    """

    @abstractmethod
    def search_best(self, query, threshold=60):
        """
        Tìm kết quả tốt nhất cho từ khóa.
        Args:
            query: Từ khóa người dùng nhập.
            threshold: Ngưỡng điểm tối thiểu.
        Returns:
            tuple: (user_info_dict, score) hoặc (None, 0).
        """
        pass

<<<<<<< HEAD
=======
    @abstractmethod
    def quick_get_id(self, query_name):
        pass


>>>>>>> 9de2b1b (FINAL)
# --- 5. HỢP ĐỒNG CHO CÔNG CỤ DỰ ĐOÁN (PREDICTOR) ---

class ILinkPredictor(ABC):
    """
    Giao diện cho logic dự đoán liên kết.
    """

    @abstractmethod
<<<<<<< HEAD
    def predict_top_k_similar(self, target_vec,all_vectors, top_k=5)->tuple:
=======
    def recommend_top_k(self, target_vec,all_vectors, top_k=5)->tuple:
>>>>>>> 9de2b1b (FINAL)
        """
        Tìm top người có thể liên kết với 1 người cho trước
        Returns: tuple (top người, id người)
        """
        pass

<<<<<<< HEAD
    def predict_link_score(self, vec_a, vec_b) -> float:
=======
    def scan_relationship(self, id_a, id_b, src_type, dst_type, mode) -> float:
>>>>>>> 9de2b1b (FINAL)
        """
        Dự đoán điểm liên kết giữa 2 vector đặc trưng.
        Returns: float (0.0 đến 1.0)
        """
        pass

class ITrainingDataRepository(ABC):
    """
    Giao diện để lưu/tải dữ liệu đã tiền xử lý cho AI (processed_data.pt).
    """
    @abstractmethod
<<<<<<< HEAD
    def save_data(self, data, mapping):
        """Lưu data (HeteroData) và mapping (dict)"""
=======
    def save_data(self, data):
        """Lưu data (HeteroData) và mapping"""
>>>>>>> 9de2b1b (FINAL)
        pass

    @abstractmethod
    def load_data(self):
        """Trả về (data, mapping)"""
        pass