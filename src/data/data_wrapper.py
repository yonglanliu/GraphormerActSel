# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Yonglan Liu.
# Licensed under the MIT License.


from __future__ import annotations

import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx


@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    """
    Shift multi-field categorical features into a single shared embedding index space.
    Reserves 0 for padding by starting offsets at 1.
    """
    feature_num = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(
        0,
        feature_num * offset,
        offset,
        dtype=torch.long,
        device=x.device,
    )
    return x + feature_offset


def shortest_path(adj_np: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    All-pairs shortest path lengths and paths via networkx.

    Returns:
      - dist: [N,N] int64, unreachable=510
      - paths: dict-of-dicts, paths[i][j] = [i,...,j]
    """
    G = nx.from_numpy_array(adj_np)  # undirected
    dist_dict = dict(nx.all_pairs_shortest_path_length(G))
    N = adj_np.shape[0]

    dist = np.full((N, N), 510, dtype=np.int64)
    for i, dd in dist_dict.items():
        for j, d in dd.items():
            dist[i, j] = d

    paths = dict(nx.all_pairs_shortest_path(G))
    return dist, paths


def gen_edge_input(
    D: int,
    all_paths: dict,
    attn_edge_type_np: np.ndarray,
) -> np.ndarray:
    """
    Build multi-hop edge features along shortest paths with a FIXED distance dimension D.

    Args:
      D: fixed max hop distance dimension (e.g. multi_hop_max_dist)
      all_paths: dict-of-dicts from nx.all_pairs_shortest_path(G)
      attn_edge_type_np: [N, N, E] numpy array

    Returns:
      edge_input: [N, N, D, E] filled with -1 for unused
    """
    n = attn_edge_type_np.shape[0]
    edge_dim = attn_edge_type_np.shape[-1]
    edge_input = -1 * np.ones((n, n, D, edge_dim), dtype=np.int64)

    for i, targets in all_paths.items():
        for j, path_nodes in targets.items():
            if i == j:
                continue
            num_hops = len(path_nodes) - 1
            if num_hops <= 0:
                continue
            for k in range(min(num_hops, D)):
                u = path_nodes[k]
                v = path_nodes[k + 1]
                edge_input[i, j, k, :] = attn_edge_type_np[u, v, :]

    return edge_input


def make_preprocess_item(multi_hop_max_dist: int, offset: int = 512):
    """
    Returns a preprocess_item(item: Data) -> Data that clamps edge_input distance dimension to fixed D.
    """
    D_fixed = int(multi_hop_max_dist)

    def preprocess_item(item: Data) -> Data:
        # Expect: node_feat [N,F], edge_feat [E,F_e], edge_index [2,E], num_nodes
        edge_attr = getattr(item, "edge_feat")
        edge_index = getattr(item, "edge_index")
        node_feat = getattr(item, "node_feat")
        N = int(getattr(item, "num_nodes"))

        # Ensure tensors
        if not torch.is_tensor(node_feat):
            node_feat = torch.as_tensor(node_feat, dtype=torch.long)
        if not torch.is_tensor(edge_index):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if not torch.is_tensor(edge_attr):
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.long)

        # Ensure edge_index [2, E]
        if edge_index.dim() == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()

        # Node encoding -> x
        x = convert_to_single_emb(node_feat, offset=offset)

        # Symmetric adjacency
        adj = torch.zeros((N, N), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True
        adj[edge_index[1], edge_index[0]] = True

        # Edge features: ensure [E, F_e]
        if edge_attr.dim() == 1:
            edge_attr = edge_attr[:, None]

        # Edge-type encoding (store on [N,N,E])
        edge_enc = convert_to_single_emb(edge_attr, offset=offset)
        attn_edge_type = torch.zeros((N, N, edge_enc.size(-1)), dtype=torch.long)
        attn_edge_type[edge_index[0], edge_index[1]] = edge_enc
        attn_edge_type[edge_index[1], edge_index[0]] = edge_enc

        # Shortest paths
        dist_np, paths = shortest_path(adj.cpu().numpy())

        # Fixed-D edge_input (raw has -1 padding)
        edge_input_np = gen_edge_input(D_fixed, paths, attn_edge_type.cpu().numpy())
        edge_input_raw = torch.from_numpy(edge_input_np).long()

        # Make embedding-safe: -1 -> 0 (padding_idx), and shift others by +1
        edge_input = (edge_input_raw + 1).clamp(min=0)

        # Other tensors
        spatial_pos = torch.from_numpy(dist_np).long()
        attn_bias = torch.zeros((N + 1, N + 1), dtype=torch.float)

        # Attach
        item.x = x
        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree
        item.edge_input = edge_input
        item.edge_input_raw = edge_input_raw  # optional debug

        return item

    return preprocess_item



# -------------------------
# Collate for Graphormer
# -------------------------

def graphormer_collate(batch: List[Data]):
    """
    Pads graphormer tensors to maxN in batch.
    Assumes edge_input has FIXED D across dataset (due to make_preprocess_item).
    Returns a dict that your GraphNodeFeature/GraphAttnBias forward expects.
    """
    B = len(batch)
    maxN = max(d.x.size(0) for d in batch)

    F = batch[0].x.size(-1)
    E = batch[0].attn_edge_type.size(-1)
    D = batch[0].edge_input.size(-2)  # fixed

    device = batch[0].x.device

    x = torch.zeros((B, maxN, F), dtype=torch.long, device=device)
    in_degree = torch.zeros((B, maxN), dtype=torch.long, device=device)
    out_degree = torch.zeros((B, maxN), dtype=torch.long, device=device)

    spatial_pos = torch.zeros((B, maxN, maxN), dtype=torch.long, device=device)
    attn_edge_type = torch.zeros((B, maxN, maxN, E), dtype=torch.long, device=device)
    edge_input = torch.zeros((B, maxN, maxN, D, E), dtype=torch.long, device=device)

    attn_bias = torch.zeros((B, maxN + 1, maxN + 1), dtype=torch.float, device=device)

    ys = []
    for i, d in enumerate(batch):
        N = d.x.size(0)

        x[i, :N] = d.x
        in_degree[i, :N] = d.in_degree
        out_degree[i, :N] = d.out_degree

        spatial_pos[i, :N, :N] = d.spatial_pos
        attn_edge_type[i, :N, :N] = d.attn_edge_type
        edge_input[i, :N, :N] = d.edge_input  # fixed D

        attn_bias[i, :N+1, :N+1] = d.attn_bias

        # mask attention to padded nodes (recommended)
        if N < maxN:
            pad_start = N + 1  # +1 for graph token
            attn_bias[i, pad_start:, :] = float("-inf")
            attn_bias[i, :, pad_start:] = float("-inf")

        y = d.y
        if y.dim() == 2 and y.size(0) == 1:
            y = y.squeeze(0)
        ys.append(y)

    y = torch.stack(ys, dim=0)

    return {
        "x": x,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "spatial_pos": spatial_pos,
        "attn_edge_type": attn_edge_type,
        "edge_input": edge_input,
        "attn_bias": attn_bias,
        "y": y,
    }
