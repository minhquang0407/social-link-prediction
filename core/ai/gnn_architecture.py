import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
torch.set_float32_matmul_precision('medium')

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

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim *2 , hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
    def forward(self, z_src, z_dst):
        h = torch.cat([z_src, z_dst], dim = 1)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.lin2(h)
        return h.squeeze()

class LinkPredictionModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.5, data=None):
        super().__init__()
        self.gnn = GraphSAGE(hidden_channels, out_channels, dropout)
        self.encoder = to_hetero(self.gnn, data.metadata(), aggr='sum')

        self.decoders = torch.nn.ModuleDict()
        for et in data.edge_types:
            src, rel, dst = et
            if rel.startswith('rev_'):
                continue

            key = f"{src}__{rel}__{dst}"
            self.decoders[key] = MLP(out_channels, 64, 1, dropout)

    def forward(self, x, edge_index, target_edge_type, edge_label_index):
        z_dict = self.encoder(x, edge_index)
        src_type, rel, dst_type = target_edge_type

        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]

        key = f"{src_type}__{rel}__{dst_type}"

        if key in self.decoders:
            return self.decoders[key](z_src, z_dst)
        else:
            # Fallback: Trả về vector 1 chiều để khớp với MLP đã squeeze
            return torch.zeros(z_src.size(0), device=z_src.device)