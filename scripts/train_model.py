import sys
import os
from pathlib import Path
import itertools  # Dùng cho Grid Search

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import json

from config.settings import (
    GRAPH_PATH, MODEL_PATH, PYG_DATA_PATH, MAPPING_PATH, TRAINING_HISTORY_PATH,
    INPUT_DIM, OUTPUT_DIM, BATCH_SIZE
)
from infrastructure.repositories.graph_repo import PickleGraphRepository
from core.ai.gnn_architecture import GraphSAGE
from core.ai.data_processor import GraphDataProcessor
from infrastructure.repositories.feature_repo import PyGDataRepository
from infrastructure.repositories.model_repo import ModelRepository

# --- 1. CHUẨN BỊ DỮ LIỆU ---
def get_or_prepare_data(force_prepare=False):
    """Tải hoặc tạo mới dữ liệu PyG."""
    # TODO 1: Nếu không bắt buộc tạo lại.
    if not force_prepare:
        pass
        # TODO 2: Khởi tạo PyGDataRepository để xử lý việc tải/lưu dữ liệu PyG.
        # TODO 3: Thử tải dữ liệu HeteroData (data) và mapping từ disk.
    # TODO 4: Nếu dữ liệu PyG chưa tồn tại (data is None) hoặc Bắt buộc tạo lại PyG:
    if data is None or force_prepare:
        print("⚠️ Chưa có dữ liệu PyG. Đang xử lý từ NetworkX...")
        # TODO 5: Tải đồ thị NetworkX (G) từ PickleGraphRepository.
        # TODO 6: Khởi tạo và sử dụng GraphDataProcessor để chuyển đổi G sang PyG HeteroData.
        # TODO 7: Lưu dữ liệu PyG (data và mapping) mới tạo.

    # TODO 8: Trả về dữ liệu PyG đã sẵn sàng.
    return None


# --- 2. CÁC HÀM HUẤN LUYỆN & ĐÁNH GIÁ ---

def train_epoch(model, loader, optimizer, device, target_edge_type):
    """Chạy 1 epoch huấn luyện."""
    #TODO: Bật chế độ train cho model
    model.train()
    total_loss = 0
    total_examples = 0

    # TODO 1: Lặp qua loader với tqdm.
    for batch in tqdm(loader, desc="Training", leave=False):
        # TODO 2: Di chuyển batch sang device và reset gradient.

        # TODO 3: Forward Pass: Lấy embeddings Z dictionary (z_dict) từ model.

        # TODO 4: Trích xuất nhãn (edge_label) và chỉ mục cạnh (edge_label_index) cần dự đoán.


        # TODO 5: Decode (Tính điểm):
        #         - Lấy loại node nguồn và đích từ target_edge_type.
        #         - Lấy embeddings của node nguồn (z_src) và node đích (z_dst) tương ứng với edge_label_index.
        #         - Tính điểm liên kết (score) bằng Dot Product (sum theo dim=-1).

        # TODO 6: Tính Loss (sử dụng F.binary_cross_entropy_with_logits).

        # TODO 7: Backward Pass và cập nhật tham số.

        # TODO 8: Cập nhật tổng loss và số lượng mẫu.


    # TODO 9: Trả về Loss trung bình.
    return None


@torch.no_grad()
def evaluate(model, loader, device, target_edge_type):
    """Đánh giá mô hình (tính AUC)."""
    model.eval()
    preds = []
    ground_truths = []

    # TODO 1: Lặp qua loader với tqdm (không tính toán gradient: @torch.no_grad).
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        # TODO 2: Forward Pass: Lấy embeddings Z dictionary.
        # TODO 3: Trích xuất nhãn (edge_label) và chỉ mục cạnh (edge_label_index).
        # TODO 4: Decode (Tính điểm):
        #         - Lấy embeddings z_src, z_dst tương ứng.
        #         - Tính score, sau đó áp dụng Sigmoid để chuyển thành xác suất [0, 1].
        # TODO 5: Lưu trữ dự đoán (preds) và nhãn thực tế (ground_truths) về CPU/Numpy.

    # TODO 6: Nối (concatenate) các mảng lại và tính ROC AUC Score.
    return roc_auc_score(np.concatenate(ground_truths), np.concatenate(preds))


# --- 3. CHIẾN LƯỢC CHẠY ---

def train_one_config(data, config, device, target_edge_type, final_mode=False):
    """
    Huấn luyện mô hình với 1 bộ tham số cụ thể.
    """
    # TODO 1: Khởi tạo từ điển lịch sử và trích xuất tham số từ config.
    history = {
        "epoch": [],
        "loss": [],
        "val_auc": []
    }

    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']

    print(f"\n⚙️ Cấu hình: Hidden={hidden_dim}, LR={lr}")

    # TODO 2: Chia dữ liệu (RandomLinkSplit):
        # TODO 2a: Nếu là Final Mode, dùng toàn bộ data cho train (val_loader = None).
        # TODO 2b: Dùng RandomLinkSplit (10% Val, 10% Test) để chia data thành train/val/test.
        # TODO 2c: Khởi tạo LinkNeighborLoader cho tập Validation (không shuffle, không neg_sampling_ratio).


    # TODO 3: Khởi tạo LinkNeighborLoader cho tập Train:
    #         - Dùng train_data.
    #         - edge_label_index: Sử dụng tất cả các cạnh trong tập train (train_data[target_edge_type].edge_index).
    #         - neg_sampling_ratio=1.0.
    #         - Có shuffle.

    # TODO 4: Khởi tạo Model & Optimizer:
    #         - Khởi tạo Base GNN (GraphSAGE) với hidden_dim và OUTPUT_DIM.
    #         - Chuyển Base Model thành Hetero Model (to_hetero) và gửi sang device.
    #         - Khởi tạo Optimizer (Adam) với learning rate (lr).

    best_val_auc = 0
    best_model_state = None

    # TODO 5: Vòng lặp huấn luyện chính (Loop):
    for epoch in range(1, epochs + 1):

        # TODO 5a: Huấn luyện 1 epoch và cập nhật history/log.
        #   - LOGIC TRAIN 1 EPOCH.
        #   - Lưu lại loss và epoch


        # TODO 5b: Nếu có val_loader:
        #   - Đánh giá mô hình trên tập Val (val_auc).
        #   - Lưu lại state_dict của mô hình nếu đây là kết quả AUC tốt nhất.

        print(log_msg)

    # TODO 6: Xử lý lưu lịch sử và state cuối cùng khi Final Mode:
    #   - Lưu lịch sử huấn luyện (history) file JSON vào TRAINING_HISTORY_PATH .
    #   - Lấy state cuối cùng (thay vì best_model_state) và giả định AUC = 1.0.

    # TODO 7: Trả về AUC tốt nhất và state_dict tương ứng.
    return best_val_auc, best_model_state


def run_grid_search():
    """Chạy tìm kiếm tham số tối ưu và huấn luyện mô hình cuối cùng."""
    # TODO 1: Chuẩn bị dữ liệu, thiết bị (device) và target_edge_type.

    # TODO 2: Định nghĩa lưới tham số (param_grid) cho hidden_dim, lr, epochs.
    param_grid = {
        'hidden_dim': [64, 128],
        'lr': [0.01, 0.001],
        'epochs': [20, 50]
    }

    # TODO 3: Tạo tất cả các tổ hợp tham số từ lưới (sử dụng itertools.product).
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_auc = 0
    best_params = None

    print(f"🚀 Bắt đầu Grid Search trên {len(combinations)} cấu hình...")

    # TODO 4: Lặp qua từng cấu hình trong combinations:
    #   - Gọi train_one_config (với final_mode=False) và lấy AUC.
    #   - Cập nhật best_auc và best_params nếu tìm thấy kết quả tốt hơn.


    print(f"\n✅ Grid Search Hoàn tất. Tốt nhất: {best_params} (AUC: {best_auc:.4f})")

    print("\n🏋️ Bắt đầu Final Training (100 Epochs) với tham số tốt nhất...")
    # TODO 5: Chạy Final Training với tham số tốt nhất:
    #   - Cập nhật số epochs cho Final Training (ví dụ: 100).
    #   - Gọi train_one_config với best_params và final_mode=True.
    _, final_state = train_one_config(data, best_params, device, target_edge_type, final_mode=True)

    # TODO 6: Lưu Model cuối cùng, gọi ModelRepository và lưu lại



if __name__ == "__main__":
    # TODO: Khởi động quá trình Grid Search và Final Training.
    run_grid_search()