import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import torch_geometric.transforms as T
from torch_geometric.utils import sort_edge_index
from torch_geometric.loader import LinkNeighborLoader
import itertools
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from infrastructure.repositories import PyGDataRepository
from config.settings import  PYG_DATA_PATH, OUTPUT_DIM, TRAINING_HISTORY_PATH, MODEL_PATH, \
    PYG_TRAINING_DATA_PATH
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
torch.set_float32_matmul_precision('medium')
from torch.cuda.amp import GradScaler, autocast


# Architecture
class GraphSAGE(torch.nn.Module):
    def __init__(self,hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv((-1,-1), hidden_channels)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv((-1,-1), out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2., dim=-1)
        return x

class InteractionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        # Input dim * 3 vì ta sẽ nối [src, dst, src*dst]
        self.lin1 = torch.nn.Linear(input_dim * 3, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, z_src, z_dst):
        # TẠO TƯƠNG TÁC MẠNH MẼ
        # 1. Đặc trưng gốc: z_src, z_dst
        # 2. Đặc trưng tương đồng: z_src * z_dst (Hadamard product)
        # Giúp model dễ dàng học được "A và B có giống nhau không?"
        combined = torch.cat([z_src, z_dst, z_src * z_dst], dim=1)

        h = self.lin1(combined)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.lin2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.lin3(h)

        # Ép về 1 chiều để tránh lỗi shape [N, 1] vs [N]
        return h.view(-1)
import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class HGTLinkPrediction(torch.nn.Module):
    def __init__(self,  hidden_channels, out_channels,data, dropout=0.5, num_heads=4, num_layers=3):
        super().__init__()

        # 1. INPUT PROJECTION (Quan trọng)
        # Biến đổi vector Text (768 dim) + Year (1 dim) về không gian chung (256 dim)
        # Giúp model học được đặc trưng riêng cho bài toán này
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            # Lấy kích thước feature đầu vào thực tế từ data
            in_dim = data[node_type].x.size(1)
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        # 2. HGT LAYERS (Thay cho SAGE)
        # HGT dùng cơ chế Attention để "đọc hiểu" feature text tốt hơn SAGE
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           heads=num_heads)
            self.convs.append(conv)

        self.dropout = torch.nn.Dropout(p=dropout)

        # 3. INTERACTION DECODER (Như đã bàn)
        self.decoders = torch.nn.ModuleDict()
        for et in data.edge_types:
            src, rel, dst = et
            if rel.startswith('rev_'): continue
            key = f"{src}__{rel}__{dst}"

            # Decoder với input là hidden_channels
            self.decoders[key] = InteractionMLP(hidden_channels, 64, 1, dropout)

    def forward(self, x_dict, edge_index_dict, target_edge_type, edge_label_index):
        # A. Projection: Ép feature text + year vào không gian Hidden
        dtype = self.kqv_lin.weights[0].dtype if hasattr(self.kqv_lin, 'weights') else torch.float32
        x_dict = {k: v.to(dtype) for k, v in x_dict.items()}
        x_start = {}
        for node_type, x in x_dict.items():
            x_start[node_type] = self.lin_dict[node_type](x).relu_()
            x_start[node_type] = self.dropout(x_start[node_type])

        # B. HGT Message Passing
        for conv in self.convs:
            x_start = conv(x_start, edge_index_dict)

        # C. Decode
        src_type, rel, dst_type = target_edge_type
        z_src = x_start[src_type][edge_label_index[0]]
        z_dst = x_start[dst_type][edge_label_index[1]]

        key = f"{src_type}__{rel}__{dst_type}"
        if key in self.decoders:
            return self.decoders[key](z_src, z_dst)
        else:
            return torch.zeros(z_src.size(0), device=z_src.device)

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, data=None, dropout=0.5):
        super().__init__()

        # Dùng DeepGraphSAGE thay vì bản thường
        self.gnn = GraphSAGE(hidden_channels, out_channels, dropout)
        self.encoder = to_hetero(self.gnn, data.metadata(), aggr='sum')

        self.decoders = torch.nn.ModuleDict()

        # Init Decoder cho từng loại cạnh
        for et in data.edge_types:
            src, rel, dst = et
            if rel.startswith('rev_'): continue

            key = f"{src}__{rel}__{dst}"
            # Decoder mới thông minh hơn
            self.decoders[key] = InteractionMLP(out_channels, 64, 1, dropout)

    def forward(self, x, edge_index, target_edge_type, edge_label_index):
        z_dict = self.encoder(x, edge_index)

        src_type, rel, dst_type = target_edge_type

        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        key = f"{src_type}__{rel}__{dst_type}"

        if key in self.decoders:
            return self.decoders[key](z_src, z_dst)
        else:
            return torch.zeros(z_src.size(0), device=z_src.device)

# Function to train
import optuna

from config.settings import BATCH_SIZE, MODEL_PATH
from infrastructure.repositories import ModelRepository
import gc



def get_or_prepare_data():
    """Tải và chuẩn bị dữ liệu (Undirected + Sanitize)."""
    feature_repo = PyGDataRepository(PYG_DATA_PATH)
    data = feature_repo.load_data()

    if data is None:
        print("Chưa có dữ liệu PyG. Vui lòng chạy ETL trước!")
        return None

    # Xóa cạnh rỗng
    #data = sanitize_hetero_data(data)

    return data


def loader_generator(data_source, target_edge_types, batch_size, shuffle=False):
    for et in target_edge_types:
        # 1. Kiểm tra nhanh (giữ nguyên logic cũ của bạn)
        if et not in data_source.edge_index_dict: pass
        if hasattr(data_source[et], 'edge_label_index') and data_source[et].edge_label_index.numel() == 0:
            continue

        # 2. Chuẩn bị nhãn
        lbl_index = data_source[et].edge_label_index
        lbl_ones = torch.ones(lbl_index.size(1), dtype=torch.float32)

        # 3. Khởi tạo Loader (Chỉ tốn RAM tại thời điểm này)

        loader = LinkNeighborLoader(
            data_source,
            num_neighbors=[15, 10],
            edge_label_index=(et, lbl_index),
            edge_label=lbl_ones,
            neg_sampling_ratio=1.0,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,        # Tăng lên 4 hoặc 6 (tùy số nhân CPU của bạn)
        )
        # 4. Trả về để dùng ngay
        yield et, loader


def train_epoch(model, data, optimizer, device, target_edge_types, scaler, batch_size=BATCH_SIZE):
    model.train()
    total_loss = 0
    total_examples = 0
    count = 1
    max = len(target_edge_types)
    data_loader_gen = loader_generator(data, target_edge_types, batch_size, shuffle=True)
    for edge_type, loader in data_loader_gen:
        pbar = tqdm(loader, desc="Training", leave=False)
        pbar.set_postfix({
            "relationship": f" {edge_type[1]} | {count}/{max}"
        })
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Với single-type loader, ta biết chắc chắn cạnh cần dự đoán là edge_type
            # PyG tự động gán vào edge_label_index của edge_type đó trong batch
            edge_label_index = batch[edge_type].edge_label_index
            edge_label = batch[edge_type].edge_label
            with torch.amp.autocast('cuda'):
                # Forward
                out = model(batch.x_dict, batch.edge_index_dict, edge_type, edge_label_index)

                # Loss
                loss = F.binary_cross_entropy_with_logits(out, edge_label)

            scaler.scale(loss).backward()

            # 3. Optimizer Step
            scaler.step(optimizer)
            scaler.update()  # Cập nhật lại hệ số scale cho lần sau

            total_loss += loss.item() * edge_label.size(0)
            total_examples += edge_label.size(0)
        count += 1
        del loader, pbar
        gc.collect()
    return total_loss / (total_examples + 1e-6)


@torch.no_grad()
def evaluate(model, data, device, target_edge_types, batch_size=BATCH_SIZE):
    """
    Đánh giá mô hình trên tập Val hoặc Test.

    Args:
        model: Mô hình GNN đã huấn luyện.
        loaders: List các LinkNeighborLoader (mỗi loader ứng với 1 loại cạnh).
        device: 'cuda' hoặc 'cpu'.
        target_edge_types: List các loại cạnh tương ứng với loaders.

    Returns:
        score: Chỉ số ROC-AUC (0.0 -> 1.0).
    """
    model.eval()  # Chuyển model sang chế độ đánh giá (tắt Dropout, khóa BatchNorm)

    preds = []
    ground_truths = []

    # 1. Duyệt song song qua từng cặp (Loại cạnh, Loader tương ứng)
    # Lưu ý: target_edge_types và loaders phải có cùng độ dài và thứ tự
    count = 1
    max = len(target_edge_types)
    data_loader_gen = loader_generator(data, target_edge_types, batch_size, shuffle=False)
    for edge_type, loader in data_loader_gen:

        pbar = tqdm(loader, desc="Validation", leave=False)
        pbar.set_postfix({
            "relationship": f" {edge_type[1]} | {count}/{max}"
        })
        for batch in pbar:
            batch = batch.to(device)

            # Kiểm tra an toàn: Batch có chứa nhãn cho loại cạnh này không?
            if not hasattr(batch[edge_type], 'edge_label_index') or batch[edge_type].edge_label_index.numel() == 0:
                continue

            # Lấy dữ liệu "đề thi"
            edge_label_index = batch[edge_type].edge_label_index
            edge_label = batch[edge_type].edge_label

            # 3. Forward Pass
            # Truyền đúng edge_type để model biết dùng trọng số nào (nếu có chia tách)
            # Output model thường là Logits (chưa qua Sigmoid)
            with torch.amp.autocast('cuda'):
                out = model(batch.x_dict, batch.edge_index_dict, edge_type, edge_label_index)
                # Sigmoid cũng nên nằm trong context này (hoặc không, tùy ý, nhưng forward model bắt buộc phải có)
                out = torch.sigmoid(out)

            # 4. Lưu lại kết quả (Đưa về CPU và Numpy để tính toán bằng Sklearn)
            preds.append(out.cpu().numpy())
            ground_truths.append(edge_label.cpu().numpy())
        count += 1
        del loader, pbar
    # 5. Xử lý trường hợp không có dữ liệu (tránh lỗi crash)
    if len(preds) == 0:
        print("Not data for validation.")
        return 0.0

    # 6. Gộp tất cả các mảng numpy lại thành 1 mảng dài duy nhất
    final_preds = np.concatenate(preds)
    final_labels = np.concatenate(ground_truths)

    if np.isnan(final_preds).any():
        print("❌ LỖI NGHIÊM TRỌNG: Model output chứa NaN!")
        return 0.0

    # Kiểm tra xem Labels có đủ 2 lớp (0 và 1) không
    unique_labels = np.unique(final_labels)
    if len(unique_labels) < 2:
        print(f"⚠️ CẢNH BÁO: Tập Label chỉ chứa 1 loại nhãn duy nhất: {unique_labels}")
        print("-> Lý do: Loader Validation chưa bật 'neg_sampling_ratio'!")
        return 0.0

    # 7. Tính ROC-AUC Score
    try:
        return roc_auc_score(final_labels, final_preds)
    except ValueError as e:
        print(f"❌ SKLEARN ERROR: {e}")
        # In thêm thống kê để biết tại sao
        print(f"Min Pred: {final_preds.min()}, Max Pred: {final_preds.max()}")
        print(f"Unique Labels: {np.unique(final_labels)}")
        return 0.0


# --- 3. CHIẾN LƯỢC CHẠY ---
def get_edge_pairs(data):
    """
    Tự động bắt cặp cạnh thuận và cạnh nghịch.
    Quy tắc: Cạnh nghịch có thêm tiền tố 'rev_' hoặc là chiều ngược lại.
    """
    forward_edges = []
    reverse_edges = []

    for edge_type in data.edge_types:
        src, rel, dst = edge_type

        # 1. Bỏ qua nếu đây là cạnh 'rev_' (chúng ta sẽ xử lý nó khi gặp cạnh thuận)
        if rel.startswith('rev_'):
            continue

        # 2. Xây dựng tên cạnh ngược dự kiến
        rev_rel = f"rev_{rel}"
        rev_edge_type = (dst, rev_rel, src)

        # 3. Kiểm tra xem cạnh ngược này có tồn tại trong data không
        if rev_edge_type in data.edge_types:
            forward_edges.append(edge_type)
            reverse_edges.append(rev_edge_type)

    return forward_edges, reverse_edges


def prepare_data_splits(data, val_ratio=0.1, test_ratio=0.1):
    """
    Sử dụng RandomLinkSplit chuẩn của PyG.
    """
    print("--- PREPARING DATA SPLITS (RandomLinkSplit) ---")

    # 1. Tự động bắt cặp cạnh để xử lý Leakage
    target_edge_types, rev_edge_types = get_edge_pairs(data)
    for edge in target_edge_types:
        print(edge)
    print(f"-> Target Edges (Predicting): {len(target_edge_types)} types")

    # 2. Cấu hình Splitter
    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=False,  # Hetero Graph có hướng

        # [QUAN TRỌNG] Khai báo cặp cạnh để PyG tự xóa cạnh ngược trong message passing
        edge_types=target_edge_types,
        rev_edge_types=rev_edge_types,

        # Tách 30% cạnh train ra làm "Label" (Supervision), 70% giữ lại nối dây
        disjoint_train_ratio=0.3,

        add_negative_train_samples=False  # Để Loader tự sinh mẫu âm -> Tiết kiệm RAM
    )

    # 3. Thực hiện chia (Tạo ra 3 object Data riêng biệt)
    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data, target_edge_types


def call_back(model):
    repo = ModelRepository(MODEL_PATH)
    repo.save_model(model)


def train_one_config(split_data, config, device, final_mode=False, trial=None):
    print("Initializing Data for Training!", flush=True, end=" ")
    train_data, val_data, test_data, target_edge_types = split_data
    print("Completed!", flush=True)
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    dropout = config['dropout']
    print("Initializing Model!", flush=True, end=" ")
    model = LinkPredictionModel(hidden_dim, OUTPUT_DIM, train_data, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Completed!", flush=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,      # Reset LR mỗi 5 epoch
        T_mult=2,   # Lần sau dài gấp đôi (5 -> 10 -> 20)
        eta_min=1e-5 # LR thấp nhất
    )

    early_stop_patience = 10
    early_stop_counter = 0
    history = {"epoch": [], "loss": [], "val_auc": []}
    best_val_auc = 0.0
    final_test_auc = 0.0
    best_model_state = None
    best_epoch_found = 0

    print(f"\nHidden={hidden_dim}, LR={lr}, epochs={epochs}, dropout={dropout}, batch_size={batch_size}")
    scaler = torch.amp.GradScaler('cuda')  # 4. TRAINING LOOP
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_data, optimizer, device, target_edge_types, scaler, batch_size)

        history["epoch"].append(epoch)
        history["loss"].append(float(loss))
        log_msg = f"Epoch {epoch:03d} | Loss: {loss:.4f}"

        # Đánh giá trên tập Val để chọn Model tốt nhất
        if not final_mode:
            val_auc = evaluate(model, val_data, device, target_edge_types, batch_size)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history["val_auc"].append(float(val_auc))
            log_msg += f" | Val AUC: {val_auc:.4f}| LR: {current_lr :.5f}"
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()  # Lưu lại state tốt nhất
                early_stop_counter = 0
                call_back(model)
            else:
                if val_auc == best_val_auc:
                    early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(
                        f"Early Stopping tại Epoch {epoch} vì Val AUC không giao động trong {early_stop_patience} epochs.")
                    break
                # 1. Báo cáo kết quả hiện tại cho Optuna
        print(log_msg)

    # 5. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
    if not final_mode and best_model_state:
        # Load lại trọng số tốt nhất (đạt đỉnh ở Val) để test
        model.load_state_dict(best_model_state)
        final_test_auc = evaluate(model, test_data, device, target_edge_types, batch_size)
        history['test_auc'] = final_test_auc
        print(f"--> Test AUC: {final_test_auc:.4f}")

    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    # Nếu là final mode thì trả về 1.0 (hoặc train auc)
    best_model = model
    final_test_auc = 1.0
    best_epoch_found = epochs

    # Trả về Test AUC thay vì Val AUC để Grid Search in ra kết quả thực tế hơn
    # Hoặc bạn vẫn có thể trả về Val AUC để chọn tham số, nhưng in Test AUC để tham khảo

    torch.cuda.empty_cache()  # Xả VRAM
    gc.collect()
    return best_val_auc, final_test_auc, best_model


# Garbage Collection để dọn RAM thủ công
from optuna.pruners import MedianPruner
import torch

def is_edge_index_sorted(edge_index):
    """
    Kiểm tra xem edge_index có được sắp xếp theo thứ tự từ điển
    (theo hàng source trước, sau đó đến hàng target) hay không.

    Args:
        edge_index (torch.Tensor): Tensor có kích thước [2, E], kiểu dữ liệu long/int.
                                   Hàng 0 chứa source nodes, Hàng 1 chứa target nodes.

    Returns:
        bool: True nếu đã sắp xếp, False nếu chưa.
    """

    # Bước 1: Kiểm tra kích thước đầu vào để đảm bảo tính hợp lệ
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index phải có kích thước [2, E], hiện tại là {edge_index.shape}")

    num_edges = edge_index.shape[1]

    # Trường hợp cơ sở: Nếu có 0 hoặc 1 cạnh, mặc định là đã sắp xếp
    if num_edges <= 1:
        return True

    # Bước 2: Tách node nguồn (row 0) và node đích (row 1)
    src = edge_index[0, :]
    dst = edge_index[1, :]

    # Bước 3: Tính sự chênh lệch (diff) giữa các phần tử liên tiếp của source
    # diff_src[i] = src[i+1] - src[i]
    diff_src = src[1:] - src[:-1]

    # Điều kiện A: Source phải không giảm (non-decreasing)
    # Tức là src[i+1] >= src[i] => diff_src >= 0
    if torch.any(diff_src < 0):
        return False

    # Bước 4: Kiểm tra điều kiện phụ tại các vị trí mà source bằng nhau
    # Tìm các chỉ số (indices) mà tại đó src[i+1] == src[i]
    # mask là tensor boolean: True tại những nơi source không đổi
    mask = (diff_src == 0)

    # Nếu không có chỗ nào source bằng nhau, và source đã tăng dần (đã check ở trên),
    # thì coi như đã sắp xếp xong.
    if not torch.any(mask):
        return True

    # Lấy ra các phần tử của đích (dst) tương ứng với vị trí mask
    # dst[1:][mask] là dst[i+1] tại nơi src[i] == src[i+1]
    # dst[:-1][mask] là dst[i] tại nơi src[i] == src[i+1]

    relevant_dst_next = dst[1:][mask]
    relevant_dst_curr = dst[:-1][mask]

    # Điều kiện B: Tại những nơi source bằng nhau, target phải không giảm
    # dst[i+1] >= dst[i]
    if torch.any(relevant_dst_next < relevant_dst_curr):
        return False

    return True

# --- PHẦN SCRIPT TEST (TEST BENCH) ---

def run_test(data):
    l = torch.tensor([is_edge_index_sorted(data[edge_type].edge_index) for edge_type in data.edge_types])
    print(l.all())


def pre_process_data(data):
    # Giả sử data là HeteroData
    data.pin_memory()  # Tăng tốc transfer dữ liệu
    return data


def run_optimization(data = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # --- BƯỚC 1: CHUẨN BỊ DATA CHO GRID SEARCH ---
    print("\n>>> Loading & Splitting Data...")

    # 1. Load dữ liệu gốc
    #if data is None:
    data = get_or_prepare_data()


    run_test(data)
    # 2. Cắt dữ liệu (In-Place Modification)
    # Hàm này sẽ xóa cạnh Val/Test khỏi `data` và trả về indices rời
    # train_graph chính là biến `data` sau khi bị cắt
    split_data = prepare_data_splits(data, val_ratio=0.005, test_ratio=0.005)
    config = {
        'hidden_dim': 128,
        'batch_size': 128,
        'lr': 0.01,
        'dropout': 0.41372892081262375,
        'epochs': 50
    }
    best_val_auc, test_auc, best_model = train_one_config(split_data, config, device)

    repo = ModelRepository(MODEL_PATH)
    repo.save_model(best_model)


# Loading data
#data = ToUndirected(merge=False)(data)
#feature.save_data(data,mapping)
# Training
if __name__ == "__main__":
    run_optimization()
