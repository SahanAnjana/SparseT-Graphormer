import numpy as np
from typing import List, Optional, Iterable
import os

from torch_geometric.data import Dataset
from src.data.StaticGraphTemporalSignal import StaticGraphTemporalSignal
from src.data.pyg_dataset import GraphTemporalDataset


def wrap_traffic_dataset(
        train_x,
        val_x,
        test_x,
        train_y,
        val_y,
        test_y,
        edge_indices,
        edge_values,
        scaler,
        **kwargs
) -> Optional[Dataset]:
    train_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=train_x,
        targets=train_y
    )
    val_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=val_x,
        targets=val_y
    )
    test_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=test_x,
        targets=test_y
    )
    return GraphTemporalDataset(
        train_set=train_dataset,
        valid_set=val_dataset,
        test_set=test_dataset,
        scaler=scaler,
        **kwargs
    )
