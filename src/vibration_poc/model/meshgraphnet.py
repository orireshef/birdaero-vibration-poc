"""MeshGraphNet model for structural deformation prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Multi-layer perceptron with optional LayerNorm on output."""

    def __init__(self, dims: list[int], layer_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if layer_norm:
            layers.append(nn.LayerNorm(dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.net(x)
        return result


class EdgeBlock(nn.Module):
    """Update edge embeddings using sender, receiver, and edge features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = MLP([3 * hidden_dim, hidden_dim, hidden_dim])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        inp = torch.cat([edge_attr, x[src], x[dst]], dim=1)
        result: Tensor = self.mlp(inp)
        return result


class NodeBlock(nn.Module):
    """Update node embeddings using aggregated edge messages."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = MLP([2 * hidden_dim, hidden_dim, hidden_dim])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        dst = edge_index[1]
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_attr), edge_attr)
        delta: Tensor = self.mlp(torch.cat([x, agg], dim=1))
        return x + delta


class ProcessorLayer(nn.Module):
    """Single processor step: edge update then node update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_block = EdgeBlock(hidden_dim)
        self.node_block = NodeBlock(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        edge_attr = self.edge_block(x, edge_index, edge_attr)
        x = self.node_block(x, edge_index, edge_attr)
        return x, edge_attr


class MeshGraphNet(nn.Module):
    """Graph neural network for mesh-based simulation."""

    def __init__(
        self,
        input_dim_nodes: int = 4,
        input_dim_edges: int = 4,
        output_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 15,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP([input_dim_nodes, hidden_dim, hidden_dim])
        self.edge_encoder = MLP([input_dim_edges, hidden_dim, hidden_dim])
        self.processor = nn.ModuleList([ProcessorLayer(hidden_dim) for _ in range(num_layers)])
        self.decoder = MLP([hidden_dim, hidden_dim, output_dim], layer_norm=False)

    def forward(self, graph: dict[str, Tensor]) -> Tensor:
        x = graph["x"]
        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.processor:
            assert isinstance(layer, ProcessorLayer)
            x, edge_attr = layer(x, edge_index, edge_attr)

        decoded: Tensor = self.decoder(x)
        return decoded
