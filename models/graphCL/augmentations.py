import torch
from torch_geometric.utils import dropout_edge, subgraph

class TemporalAugmentor:
    def __init__(self, intra_step_drop_prob=0.5, inter_step_drop_prob=0.0, feature_mask_prob=0.2):
        self.intra_prob = intra_step_drop_prob
        self.inter_prob = inter_step_drop_prob
        self.feat_prob = feature_mask_prob

    def temporal_edge_drop(self, data):
        data = data.clone()
        edge_index = data.edge_index
        src, dst = edge_index[0], edge_index[1]
        t_src = data.time[src]
        t_dst = data.time[dst]

        # assumes backward edges are filtered out
        is_same_step = (t_src == t_dst)
        is_forward = (t_dst > t_src)

        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        intra_dropout_mask = torch.rand(is_same_step.sum(), device=edge_index.device) > self.intra_prob
        mask[is_same_step] = intra_dropout_mask

        if self.inter_prob > 0:
            inter_dropout_mask = torch.rand(is_forward.sum(), device=edge_index.device) > self.inter_prob
            mask[is_forward] = inter_dropout_mask

        data.edge_index = edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
            
        return data

    def temporal_subgraph_sampling(self, data):
        """
        Simulates 'Temporal Subgraph Sampling' on an already batched subgraph.
        It picks a central node and strictly enforces causal traversal within the batch,
        masking out nodes that are temporally disconnected.
        """
        data = data.clone()
        num_nodes = data.num_nodes
        
        # Pick a random anchor node in the batch
        anchor = torch.randint(0, num_nodes, (1,)).item()
        anchor_time = data.time[anchor]

        # In this view, we only want to keep nodes that represent a valid flow 
        # starting near the anchor or leading to the anchor within a window.
        # Simple heuristic: Keep nodes within [time, time + k] relative to anchor
        # to enforce local temporal consistency.
        # temporal window around the anchor
        # (e.g., -1 step (predecessors) to +2 steps (successors))
        valid_time_mask = (data.time >= anchor_time - 1) & (data.time <= anchor_time + 2)

        subset = torch.arange(num_nodes, device=data.x.device)[valid_time_mask]
        edge_index, edge_attr = subgraph(subset, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True)

        data.x = data.x[subset]
        data.time = data.time[subset]
        data.y = data.y[subset]
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        if hasattr(data, 'batch') and data.batch is not None:
            data.batch = data.batch[subset]
            
        return data

    def feature_masking(self, data):
        data = data.clone()
        x = data.x
        mask = torch.rand(x.size(), device=x.device) < self.feat_prob
        x[mask] = 0
        return data

    def get_view(self, data, mode='temporal_edge'):
        """
        Wrapper to generate a view based on selected mode.
        """
        if mode == 'temporal_edge':
            return self.temporal_edge_drop(data)
        elif mode == 'temporal_subgraph':
            return self.temporal_subgraph_sampling(data)
        elif mode == 'feature':
            return self.feature_masking(data)
        else:
            return data