import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from config.settings import PYG_DATA_PATH
from infrastructure.repositories import PyGDataRepository
feature = PyGDataRepository(PYG_DATA_PATH)
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
class LinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, data=None,metadata=None, dropout=0.5):
        super().__init__()
        if data is not None:
            self.metadata = data.metadata()
        elif metadata is not None:
            self.metadata = metadata
        else:
            raise ValueError("Phải cung cấp 'data' (lúc train) hoặc 'metadata' (lúc deploy)!")

        node_types, edge_types = self.metadata
        self.gnn = GraphSAGE(hidden_channels, out_channels, dropout)
        self.encoder = to_hetero(self.gnn, metadata, aggr='sum')

        self.decoders = torch.nn.ModuleDict()

        # Init Decoder cho từng loại cạnh
        for et in edge_types:
            _, rel, _ = et
            if rel.startswith('rev_'): continue

            key = f"__{rel}__"
            self.decoders[key] = InteractionMLP(out_channels, 64, 1, dropout)

    def forward(self, x, edge_index, target_edge_type, edge_label_index):
        z_dict = self.encoder(x, edge_index)

        src_type, rel, dst_type = target_edge_type

        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        key = f"__{rel}__"

        if key in self.decoders:
            return self.decoders[key](z_src, z_dst)
        else:
            return torch.zeros(z_src.size(0), device=z_src.device)