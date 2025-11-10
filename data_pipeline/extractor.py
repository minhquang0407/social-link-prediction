class WikidataExtractor:
    def __init__(self, user_agent):
        # Khởi tạo SPARQLWrapper
        self.sparql = SPARQLWrapper(...) 
        self.sparql.setReturnFormat(JSON)

    def _run_paginated_query(self, base_query, output_filename):
        # (Logic "lật trang" (while True))
        # (Hàm "private", chỉ dùng nội bộ)
        pass

    def fetch_all_relationships(self, relationship_queries, output_dir):
        # (Hàm chính)
        # Logic: Lặp qua dict 'relationship_queries'
        # (ví dụ: {"family": "SELECT..."})
        # và gọi _run_paginated_query cho mỗi cái.
        pass

