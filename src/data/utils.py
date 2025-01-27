# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Iterable

import torch
import numpy as np
import pyximport
from sklearn.model_selection import train_test_split
import json

pyximport.install(setup_args={"include_dirs": np.get_include()})
from src.data.algos import gen_edge_input, floyd_warshall
from torch_geometric.data import Data


class Scaler:
    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data, args=None):
        raise NotImplementedError

    def to_device(self, device):
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    z-score norm the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, args=None):
        data_shape = data.shape
        D = data_shape[-1]

        return (data * self.std) + self.mean

    def __str__(self):
        return f"StandardScaler(mean={self.mean}, std={self.std})"

    def to_device(self, device):
        for attr in ['mean', 'std']:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, np.ndarray):
                setattr(self, attr, torch.tensor(attr_val, dtype=torch.float, device=device))
            elif isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(device))
            elif isinstance(attr_val, float) or isinstance(attr_val, int):
                setattr(self, attr, torch.tensor(attr_val, device=device))
            else:
                raise NotImplementedError('scaler attributes should be torch.Tensor or np.ndarray or float/int')
        return self


def normalize(train_x, val_x, test_x, train_y, val_y, test_y):
    # all inputs have shape [num_points, num_time_points, n_nodes, node_dim]
    mean, std = train_x[..., 0].mean(), train_x[..., 0].std()
    # only normalize the sensor data, not the time_in_day data
    scaler = StandardScaler(np.array(mean), np.array(std))

    train_x[..., 0] = scaler.transform(train_x[..., 0])
    val_x[..., 0] = scaler.transform(val_x[..., 0])
    test_x[..., 0] = scaler.transform(test_x[..., 0])
    return train_x, val_x, test_x, train_y, val_y, test_y, scaler


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_xs(x: torch.Tensor, padlen):
    x = x + 1
    V, D, T = x.shape
    if V < padlen:
        new_x = x.new_zeros([V, D, padlen], dtype=x.dtype)
        new_x[:V, :, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def preprocess_item(item: Data, graph_token: bool):
    # see pyg data doc
    # x is node feature matrix with shape [n_nodes, node_feature_dim]
    # edge_attr same
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    assert len(x.shape) == 3
    t, N_nodes, feature_dim = x.shape

    # node adj matrix [N, N] bool
    adj = torch.zeros([N_nodes, N_nodes], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    weighted_adj = adj.detach().clone()

    # edge features
    if edge_attr is not None:
        # if the edge features are 1-hot encoding
        # (in the case of different edge types e.g. different molecular bonds)
        if torch.all(sum([edge_attr == i for i in [1, 0]]).bool()):
            edge_attr = edge_attr[:, None]
            attn_edge_type = torch.zeros([N_nodes, N_nodes, edge_attr.size(-1)], dtype=torch.long)
            # stop using convert_to_single, which converts to ints
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr.long()) + 1
        else:
            # in this case, edge_attr is edge distance
            assert torch.all(edge_attr < 15), 'Distances should be normalized'
            weighted_adj = weighted_adj.float()
            weighted_adj[edge_index[0, :], edge_index[1, :]] = edge_attr
            attn_edge_type = []
    else:
        attn_edge_type = []

    # path i,j entry stores the intermediate node used to reach i,j
    shortest_path_result, path = floyd_warshall(weighted_adj.numpy())
    max_dist = np.amax(shortest_path_result)

    if attn_edge_type:
        # collect edge attributes along the shortest paths
        edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
        item.edge_input = torch.from_numpy(edge_input).long()
    else:
        edge_input = []
        item.edge_input = edge_input

    # spatial pos is [n_node, n_node], the shortest path between nodes
    # used in spatial encoding attention bias where b_(vi, vj) is learnable scalar indexed by shortest path
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    if graph_token:
        attn_bias = torch.zeros([N_nodes + 1, N_nodes + 1], dtype=torch.float)
    else:
        attn_bias = torch.zeros([N_nodes, N_nodes], dtype=torch.float)

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)

    return item


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, graph_token=True, scaler=None):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_index,
            item.edge_attr,
            item.edge_input[:, :, :multi_hop_max_dist, :] if item.edge_input else item.edge_input,
            item.y,
            item.additional_features
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_indices,
        edge_attrs,
        edge_inputs,
        ys,
        add_features
    ) = zip(*items)

    if all(len(i) == 0 for i in edge_inputs):
        edge_inputs = None
    if all(len(i) == 0 for i in attn_edge_types):
        attn_edge_types = None

    for idx, _ in enumerate(attn_biases):
        if graph_token:
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        else:
            attn_biases[idx][:, :][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    # check if the graphs are static
    it = iter(xs)
    the_len = len(next(it))
    # if graphs are not static
    if not all(len(l) == the_len for l in it):
        # pad to max_node_num and max_dist
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        y = torch.stack(ys)
        # TODO: fix x padding for dynamic graphs
        x = torch.cat([pad_xs(i, max_node_num) for i in xs])

        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        ) if edge_inputs else None
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        ) if attn_edge_types else None

        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )

        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    else:
        # just stack
        y = torch.stack(ys)
        x = torch.stack(xs)
        edge_indices = torch.stack(edge_indices) if edge_indices else None
        edge_attrs = torch.stack(edge_attrs) if edge_attrs else None
        edge_input = torch.stack(edge_inputs) if edge_inputs else None
        attn_edge_type = torch.stack(attn_edge_types) if attn_edge_types else None
        attn_bias = torch.stack(attn_biases)
        spatial_pos = torch.stack(spatial_poses) + 1
        in_degree = torch.stack(in_degrees) + 1

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        x=x,
        edge_input=edge_input,
        y=y,
        add_features=add_features,
        scaler=scaler
    )


def generate_regression_task(
        data, n_hist, n_pred,
        add_time_in_day=True, add_day_in_week=False,
        replace_drops=True
):
    """
    Generate features and targets for regression tasks from a DataFrame or NumPy array.

    :param data: DataFrame (shape [T, V]) or NumPy array (shape [T, V, D]) of sensor data
    :param n_hist: Number of observed time points
    :param n_pred: Time points to be predicted
    :param add_time_in_day: Whether to add time-in-day information (only for DataFrame inputs)
    :param add_day_in_week: Whether to add day-in-week information (only for DataFrame inputs)
    :param replace_drops: Whether to replace sudden drops (values dropping to 0) with historical averages
    :return: Features and targets as NumPy arrays
    """
    features, targets = [], []

    # Check if input is a DataFrame or NumPy array
    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        df = data
        T, V = df.shape
        data_np = np.expand_dims(df.values, axis=-1)  # Convert DataFrame to NumPy array
        data_list = [data_np]

        # Handle time-based features if index is datetime
        if not df.index.values.dtype == np.dtype("<M8[ns]"):
            add_time_in_day = False
            add_day_in_week = False
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, V, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(T, V, 7))
            day_in_week[np.arange(T), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        # Combine all features into a single NumPy array
        data_np = np.concatenate(data_list, axis=-1)
    else:
        data_np = data

    T, V, D = data_np.shape

    # Replace sudden drops with historical averages if enabled
    if replace_drops:
        for v in range(V):  # Iterate over nodes
            for d in range(D):  # Iterate over features
                t = 1
                while t < T:
                    if data_np[t, v, d] == 0 and data_np[t - 1, v, d] != 0:  # Check for sudden drops
                        start_t = t
                        while t < T and data_np[t, v, d] == 0:
                            t += 1
                        data_np[start_t:t, v, d] = data_np[start_t - 1, v, d]
                    else:
                        t += 1

    # Create indices for slicing data into features and targets
    indices = [
        (i, i + (n_hist + n_pred))
        for i in range(T - (n_hist + n_pred) + 1)
    ]

    for i, j in indices:
        features.append(data_np[i: i + n_hist, ...])
        targets.append(data_np[i + n_hist: j, ...])

    # Convert features and targets to arrays
    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)

    return features, targets


def generate_split(X, y, split_ratio, norm):
    num_data = X.shape[0]
    assert num_data == y.shape[0]

    test_split, valid_split = split_ratio
    test_split, valid_split = test_split / 100, valid_split / 100
    print(
        f"creating train/valid/test datasets, ratio: "
        f"{1.0 - test_split - valid_split:.1f}/{valid_split:.1f}/{test_split:.1f}"
    )
    valid_split = valid_split / (1.0 - test_split)

    # no shuffle for traffic data to avoid train data leakage in test data since time slices overlap
    shuffle = False
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_data),
        test_size=test_split,
        shuffle=shuffle,
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_split,
        shuffle=shuffle,
    )
    train_x, val_x, test_x = X[train_idx], X[valid_idx], X[test_idx]
    train_y, val_y, test_y = y[train_idx], y[valid_idx], y[test_idx]
    if norm:
        return normalize(train_x, val_x, test_x, train_y, val_y, test_y), train_idx, valid_idx, test_idx
    else:
        return (train_x, val_x, test_x, train_y, val_y, test_y, None), train_idx, valid_idx, test_idx
