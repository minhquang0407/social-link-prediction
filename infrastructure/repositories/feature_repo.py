import os
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
    def __init__(self, data_path):
        self.data_path = data_path

    def save_data(self, data):
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

            print(f"REPO: Đang lưu Processed Data vào {self.data_path}...")
            torch.save(data, self.data_path)

        except Exception as e:
            print(f"REPO ERROR: {e}")
            return False

    def load_data(self):
        if not os.path.exists(self.data_path):
            return None
        try:
            print("REPO: Đang tải Processed Data...")
            data = torch.load(self.data_path, map_location='cpu',weights_only=True)
            return data
        except Exception as e:
            print(f"REPO ERROR: {e}")
            return None
