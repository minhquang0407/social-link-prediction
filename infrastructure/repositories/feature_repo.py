<<<<<<< HEAD
import torch
import pickle
import os
import sys
from pathlib import Path
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent.parent
sys.path.append(str(PROJECT_DIR))
from core.interfaces import ITrainingDataRepository


class PyGDataRepository(ITrainingDataRepository):
    def __init__(self, data_path, mapping_path):
        # Nhận đường dẫn từ Settings
        self.data_path = data_path
        self.mapping_path = mapping_path

    def save_data(self, data, mapping):
        try:
            # Đảm bảo thư mục cha tồn tại
=======
import os
import gzip
import pickle

from core.interfaces import ITrainingDataRepository
import torch
from torch_geometric.data.storage import BaseStorage, GlobalStorage
from torch_geometric.data import Data
from torch_geometric.data.storage import EdgeStorage, NodeStorage
torch.serialization.add_safe_globals([
    BaseStorage,
    GlobalStorage,
    Data,
    EdgeStorage,
    NodeStorage,
    dict
])
class PyGDataRepository(ITrainingDataRepository):
    def __init__(self, data_path, adjacency_path = None):
        self.data_path = str(data_path)
        self.adjacency_path = str(adjacency_path)
    def save_data(self, data):
        try:
>>>>>>> 9de2b1b (FINAL)
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

            print(f"REPO: Đang lưu Processed Data vào {self.data_path}...")
            torch.save(data, self.data_path)

<<<<<<< HEAD
            print(f"REPO: Đang lưu Mapping vào {self.mapping_path}...")
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(mapping, f)
            return True
=======
>>>>>>> 9de2b1b (FINAL)
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return False

    def load_data(self):
<<<<<<< HEAD
        if not os.path.exists(self.data_path) or not os.path.exists(self.mapping_path):
            return None, None

        try:
            print("REPO: Đang tải Processed Data...")
            # map_location='cpu' để an toàn khi load trên máy không có GPU
            data = torch.load(self.data_path, map_location='cpu')

            with open(self.mapping_path, 'rb') as f:
                mapping = pickle.load(f)

            return data, mapping
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return None, None
=======
        if not os.path.exists(self.data_path):
            return None
        try:
            print("REPO: Đang tải Processed Data...")
            data = torch.load(self.data_path, map_location='cpu')
            return data
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return None

    def load_adjacency(self):
        if not os.path.exists(self.adjacency_path):
            return None
        try:
            print("REPO: Đang tải Processed Data...")
            with open(self.adjacency_path, 'rb') as f:
                adj = pickle.load(f)
            return adj
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return None
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return None
    def save_adjacency(self, data):
        try:
            os.makedirs(os.path.dirname(self.adjacency_path), exist_ok=True)
            with gzip.open(self.adjacency_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return False
>>>>>>> 9de2b1b (FINAL)
