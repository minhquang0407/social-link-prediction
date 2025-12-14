import sys
import os
import itertools
import json
import torch
import torch.nn.functional as F
# import pandas as pd
import numpy as np
# from torch_geometric import edge_index
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# PyG Imports
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Project Imports

from config.settings import (
    GRAPH_PATH, MODEL_PATH, PYG_DATA_PATH, MAPPING_PATH,
    CLEAN_DATA_PATH, TRAINING_HISTORY_PATH, BATCH_SIZE,
    INPUT_DIM, OUTPUT_DIM
)
from infrastructure.repositories.feature_repo import PyGDataRepository
from core.ai.gnn_architecture import GraphSAGE
# from core.ai.data_processor import GraphDataProcessor


# --- 1. CÁC HÀM TIỆN ÍCH XỬ LÝ DATA ---

def sanitize_hetero_data(data):
    """
    Xóa các loại cạnh rỗng để tránh lỗi khi chạy Loader.
    """
    print("🧹 Đang dọn dẹp các loại cạnh rỗng...")
    # TODO 1: Duyệt qua data.edge_types.
    edge_types_to_del = []
    for edge_type in data.edge_types:
    # Kiểm tra xem edge_index có tồn tại hoặc có rỗng không.
        if 'edge_index' not in data[edge_type]:
            edge_types_to_del.append(edge_type)
            continue
        current_edge_index = data[edge_type].edge_index
        if current_edge_index is None or current_edge_index.numel() == 0 or current_edge_index.size(1) == 0:
            edge_types_to_del.append(edge_type)
    # Nếu rỗng thì xóa loại cạnh đó khỏi data (dùng del data[et]).
    if len(edge_types_to_del) > 0:
        for et in edge_types_to_del:
            print(f"   Đã xóa loại cạnh rỗng: {et}")
            del data[et]
    else:
        print("   Dữ liệu sạch, không tìm thấy loại cạnh rỗng.")
    return data


def get_unified_edge_index(data, src_node_type='person', dst_node_type='person'):
    """
    Gộp tất cả các loại cạnh nối giữa Person-Person lại thành một 'Siêu cạnh'
    để làm nhãn huấn luyện (Supervision Target).
    """
    print(f"🔗 Đang tổng hợp các cạnh nối giữa '{src_node_type}' và '{dst_node_type}':")
    edge_indices_list = []
    # TODO 2: Duyệt qua data.edge_types.
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
    # 1. Chỉ lấy cạnh nối src_node_type và dst_node_type.
        if src == src_node_type and dst == dst_node_type:
            # 2. Bỏ qua các cạnh nghịch đảo (bắt đầu bằng 'rev_') để tránh trùng lặp.
            if rel.startswith('rev_'):
                continue
            # 3. Thu thập edge_index vào một list.
            edge_indices_list.append(data[edge_type].edge_index)
    if not edge_indices_list:
        print("   ⚠️ Không tìm thấy cạnh nào phù hợp.")
        return torch.empty(2, 0, dtype=torch.long)
    # TODO 3: Nối (Concat) tất cả edge_index lại theo chiều ngang (dim=1).
    super_edge_index = torch.cat(edge_indices_list, dim=1)
    # TODO 4: Lọc bỏ các cạnh trùng lặp (dùng torch.unique).
    super_edge_index = torch.unique(super_edge_index, dim=1)
    # Return về super_edge_index
    return super_edge_index # Placeholder


def get_or_prepare_data():
    """Tải và chuẩn bị dữ liệu (Undirected + Sanitize)."""
    feature_repo = PyGDataRepository(PYG_DATA_PATH, MAPPING_PATH)
    data, mapping = feature_repo.load_data()

    if data is None:
        print("⚠️ Chưa có dữ liệu PyG. Vui lòng chạy ETL trước!")
        return None

    # TODO 5: Thực hiện quy trình làm sạch và chuyển đổi đồ thị:
    # 1. Gọi sanitize_hetero_data lần 1.
    data = sanitize_hetero_data(data)
    # 2. Chuyển đồ thị sang vô hướng (dùng T.ToUndirected()).
    transform = T.ToUndirected()
    data = transform(data)
    # 3. Gọi sanitize_hetero_data lần 2 (để dọn rác do ToUndirected sinh ra).
    data = sanitize_hetero_data(data)

    return data


# --- 2. CÁC HÀM TRAIN & EVAL ---

def train_epoch(model, loader, optimizer, device, target_edge_type):
    """Chạy 1 epoch huấn luyện."""
    model.train()
    total_loss = 0
    total_examples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        # TODO 6: Quan trọng - Ép kiểu dữ liệu (Data Type Casting)
        # Kiểm tra batch.x_dict, nếu là Float16 thì ép về Float32 để tránh lỗi matmul.
        for node_type in batch.x_dict:
            batch.x_dict[node_type] = batch.x_dict[node_type].float()

        # TODO 7: Forward Pass
        # 1. Đưa dữ liệu qua model để lấy z_dict (embedding).
        z_dict = model(batch.x_dict, batch.edge_index_dict)
        # 2. Lấy edge_label_index và edge_label từ batch[target_edge_type].
        edge_label_index = batch[target_edge_type].edge_label_index
        edge_label = batch[target_edge_type].edge_label
        
        # TODO 8: Decode (Tính điểm tương đồng)
        src_type, _, dst_type = target_edge_type
        # Lấy embedding của node nguồn và node đích, thực hiện Dot Product.
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        scores = (z_src * z_dst).sum(dim=-1)

        # TODO 9: Tính Loss và Backprop
        # Dùng binary_cross_entropy_with_logits.
        loss = F.binary_cross_entropy_with_logits(scores, edge_label)
        # Gọi backward() và optimizer.step().
        loss.backward()
        optimizer.step()
        # Cập nhật total_loss
        total_loss += loss.item() * edge_label.size(0)
        total_examples += edge_label.size(0)

    return total_loss / (total_examples + 1e-9)



@torch.no_grad()
def evaluate(model, loader, device, target_edge_type):
    """Đánh giá mô hình."""
    model.eval()
    preds = []
    ground_truths = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        # TODO 10: Ép kiểu dữ liệu về Float32 (tương tự train_epoch).
        for node_type in batch.x_dict:
            batch.x_dict[node_type] = batch.x_dict[node_type].float()
        # TODO 11: Forward Pass và Decode
        # Tương tự train_epoch, nhưng KHÔNG tính loss, KHÔNG backprop.
        z_dict = model(batch.x_dict, batch.edge_index_dict)
        edge_label_index = batch[target_edge_type].edge_label_index
        edge_label = batch[target_edge_type].edge_label

        src_type, _, dst_type = target_edge_type
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        # Lưu ý: Kết quả output cần qua hàm .sigmoid() để về xác suất [0, 1].
        scores = (z_src * z_dst).sum(dim=-1).sigmoid()

        # Append kết quả vào preds và ground_truths
        preds.append(scores.cpu().numpy())
        ground_truths.append(edge_label.cpu().numpy())

    if len(preds) == 0:
        return 0.0

    # TODO 12: Tính ROC AUC Score dùng sklearn
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ground_truths)

    return roc_auc_score(y_true, y_pred)


# --- 3. CHIẾN LƯỢC CHẠY ---

def train_one_config(data, config, device, final_mode=False):
    """Huấn luyện với 1 cấu hình cụ thể."""
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']

    # --- CHUẨN BỊ DỮ LIỆU ---
    # TODO 13: Gọi hàm get_unified_edge_index để tạo 'Siêu cạnh' cho việc training.
    super_edge_index = get_unified_edge_index(data, src_node_type='person', dst_node_type='person')
    target_edge_type = ('person', 'super_link', 'person')
    # TODO 14: Chia dữ liệu (Split Train/Val)
    num_edges = super_edge_index.size(1)
    perm = torch.randperm(num_edges)
    # Nếu final_mode=True: Dùng toàn bộ siêu cạnh để train.
    if final_mode:
        train_edge_index = super_edge_index
        val_loader = None
    # Nếu final_mode=False: Chia 80% train, 20% val (dùng torch.randperm).
    else:
        num_train = int(0.8 * num_edges)
        train_index = perm[:num_train]
        val_index = perm[num_train:]
        train_edge_index = super_edge_index[:, train_index]
        val_edge_index = super_edge_index[:, val_index]

    # TODO 15: Khởi tạo LinkNeighborLoader
        # - Val Loader (nếu có): shuffle=False, neg_sampling_ratio=1.0
        val_loader = LinkNeighborLoader(
            data,
            num_neighbors=[10, 5],  # Sample ít hơn cho nhanh
            edge_label_index=(target_edge_type, val_edge_index),
            edge_label=torch.ones(val_edge_index.size(1), device=data['person'].x.device),
            batch_size=BATCH_SIZE,
            shuffle=False,
            neg_sampling_ratio=1.0  # Tỉ lệ 1:1 cho tập val
        )
    # - Train Loader: shuffle=True, neg_sampling_ratio=1.0
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[20, 10],
        edge_label_index=(target_edge_type, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1), device=data['person'].x.device),
        batch_size=BATCH_SIZE,
        shuffle=True,
        neg_sampling_ratio=1.0  # Tỉ lệ 1:1 cho tập train
    )

    # Lưu ý: edge_label_index trỏ vào phần data đã split ở trên.

    # --- KHỞI TẠO MODEL ---
    # TODO 16: Khởi tạo GraphSAGE và convert sang Hetero (to_hetero).
    # Input dim lấy từ data['person'].x.shape[1].
    # input_dim = data['person'].x.shape[1] if 'person' in data else INPUT_DIM
    base_model = GraphSAGE(
        hidden_channels = hidden_dim,
        out_channels = OUTPUT_DIM
    )
    model = to_hetero(base_model, data.metadata(), aggr='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss": [], "val_auc": []}
    best_val_auc = 0
    best_model_state = None

    print(f"\n🚀 Bắt đầu train (Hidden={hidden_dim}, LR={lr})...")

    # --- TRAINING LOOP ---
    for epoch in range(1, epochs + 1):
        # TODO 17: Gọi train_epoch
        loss = train_epoch(model, train_loader, optimizer, device, target_edge_type)
        
        # Log history
        history["epoch"].append(epoch)
        history["loss"].append(float(loss))

        log_msg = f"Epoch {epoch:03d} | Loss: {loss:.4f}"

        # TODO 18: Nếu có val_loader, gọi evaluate
        # Cập nhật best_val_auc và best_model_state nếu kết quả tốt hơn.
        if val_loader is not None:
            val_auc = evaluate(model, val_loader, device, target_edge_type)
            history["val_auc"].append(float(val_auc))
            log_msg += f" | Val AUC: {val_auc:.4f}"

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
        else:
            # Final mode: Luôn cập nhật model mới nhất
            best_model_state = model.state_dict().copy()

        print(log_msg)

    # Xử lý final mode
    if final_mode:
        print(f"💾 Đang lưu lịch sử huấn luyện vào {TRAINING_HISTORY_PATH}...")
        try:
            with open(TRAINING_HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"⚠️ Không thể lưu lịch sử: {e}")
        best_val_auc = 1.0  # Placeholder cho final mode

    return best_val_auc, best_model_state


def run_grid_search():
    """Chạy Grid Search và Final Training."""
    data = get_or_prepare_data()
    if data is None: return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Running on: {device}")

    # Grid Search Configs
    param_grid = {
        'hidden_dim': [64, 128],
        'lr': [0.01],
        'epochs': [10]
    }
    
    # Tạo combinations từ param_grid
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_auc = 0
    best_params = None

    # TODO 19: Grid Search Loop
    # Duyệt qua các config trong combinations.
    for config in combinations:
    # Gọi train_one_config với final_mode=False.
        print(f"\n🧪 Testing config: {config}")
        auc, _ = train_one_config(data, config, device, final_mode=False)
    # So sánh và lưu lại config tốt nhất (best_auc).
        if auc > best_auc:
            best_auc = auc
            best_params = config
    print(f"\n🥇 Best Params: {best_params} (AUC: {best_auc:.4f})")
    
    # TODO 20: Final Training
    # Cập nhật epochs lên cao hơn (ví dụ 50).
    print("\n🏋️ Bắt đầu Final Training (50 Epochs) với tham số tốt nhất...")
    if best_params is None:
        best_params = combinations[0]  # Fallback
    final_config = best_params.copy()
    final_config['epochs'] = 50  # Tăng epoch

    # Gọi train_one_config với final_mode=True dùng best_params.
    _, final_state = train_one_config(data, final_config, device, final_mode=True)
    # Lưu model (torch.save) vào MODEL_PATH.
    print(f"💾 Đang lưu mô hình vào {MODEL_PATH}...")
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(final_state, MODEL_PATH)
        print("🎉 Hoàn tất quy trình huấn luyện!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu model: {e}")

if __name__ == "__main__":
    run_grid_search()
