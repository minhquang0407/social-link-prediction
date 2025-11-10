import networkx as nx
import pandas as pd
import pickle

class GraphTransformer:
    def __init__(self):
        # Khởi tạo một đồ thị rỗng
        self.G = nx.Graph() 

    def _load_and_flatten_json(self, raw_filepath):
        # (Logic 'load_sparql_result')
        return df_flattened

    def _add_1_to_1_edges(self, df, relationship_label):
        # (Logic xử lý "vợ/chồng" 1-1)
        # Logic: lặp df, G.add_node(..., name=...), G.add_edge(..., label=...)
        pass

    def _add_N_to_N_edges(self, df, hub_column, relationship_label):
        # (Logic "groupby" (N-N) xử lý "cùng trường", "cùng phim"...)
        # Logic: pandas.groupby(hub_column)... lặp 2 vòng for
        pass

    def build_full_graph(self, raw_files_dict):
        # (Hàm chính)
        # Logic:
        # df_family = self._load_and_flatten_json(raw_files_dict["family"])
        # self._add_1_to_1_edges(df_family, "spouse") # (ví dụ)
        #
        # df_school = self._load_and_flatten_json(raw_files_dict["education"])
        # self._add_N_to_N_edges(df_school, "school.value", "school")
        #Làm cho tất cả các file
        #Trả về 1 G_full 
        pass
    def save_graph(self, output_path):
        # (Hàm lưu file .gpickle - dùng pickle.dump)
        pass

