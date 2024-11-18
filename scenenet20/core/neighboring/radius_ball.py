

import torch
import torch_cluster

import sys
sys.path.append('../')
sys.path.append('../../')

from core.neighboring.knn import keops_knn


class RadiusBall_Neighboring:

    def __init__(self, radius, k, pad_value=-1) -> None:
        self.radius = radius
        self.k = k
        self.pad_value = pad_value

    def __call__(self, q_points:torch.Tensor, support:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q_points - torch.Tensor
            query points of shape (B, Q, 3)

        support - torch.Tensor
            input tensor of shape (B, N, 3)

        Returns
        -------
        graph - torch.Tensor
            support points of shape (B, Q, k)
        """
        return keops_radius_search(q_points, support, self.radius, self.k, self.pad_value).contiguous()
        


def keops_radius_search(q_points: torch.Tensor, s_points: torch.Tensor, radius: float, neighbor_limit: int, pad_value=-1):
    """Radius search using kNN and radius filter.

    Args:
        q_points (Tensor): query points of shape (N, C).
        s_points (Tensor): support points of shape (M, C).
        radius (float): maximum distance to consider neighbors.
        neighbor_limit (int): max number of neighbors to consider per point.

    Returns:
        Tensor: indices of neighbors within the radius, shape (N, neighbor_limit).
    """
    # Find the k-nearest neighbors first
    knn_distances, knn_indices = keops_knn(q_points.contiguous(), s_points.contiguous(), neighbor_limit)  # (B, N, neighbor_limit)
    knn_indices[knn_distances > radius] = pad_value
    return knn_indices



if __name__ == "__main__":
    # disable warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # # Example usage with a larger point cloud
    # points = torch.tensor([
    #     [0.0, 0.0, 0.0],
    #     [0.1, 0.1, 0.1],
    #     [0.15, 0.1, 0.1],
    #     [0.2, 0.2, 0.2],
    #     [0.25, 0.25, 0.25],
    #     [0.3, 0.3, 0.3],
    #     [0.35, 0.35, 0.35],
    #     [0.4, 0.4, 0.4],
    #     [0.45, 0.45, 0.45],
    #     [0.5, 0.5, 0.5],
    #     [0.55, 0.55, 0.55],
    #     [0.6, 0.6, 0.6],
    #     [0.65, 0.65, 0.65],
    #     [0.7, 0.7, 0.7],
    #     [0.75, 0.75, 0.75],
    #     [0.8, 0.8, 0.8],
    #     [0.85, 0.85, 0.85],
    #     [0.9, 0.9, 0.9],
    #     [0.95, 0.95, 0.95],
    #     [1.0, 1.0, 1.0]
    # ], device='cuda')

    # query_points = torch.tensor([
    #     [0.1, 0.1, 0.1],
    #     [0.2, 0.2, 0.2],
    #     [0.3, 0.3, 0.3],
    #     [0.4, 0.4, 0.4],
    #     [0.5, 0.5, 0.5],
    #     [0.6, 0.6, 0.6],
    #     [0.7, 0.7, 0.7],
    #     [0.8, 0.8, 0.8],
    #     [0.9, 0.9, 0.9],
    #     [1.0, 1.0, 1.0]
    # ], device='cuda')

    # # points = points.reshape(2, 10, 3)
    # # query_points = query_points.reshape(2, 5, 3)
    # radius = 0.2
    # neighbor_limit = 5

    # k_graph = k_radius_ball(query_points, points, radius, neighbor_limit, loop=True)

    # print(k_graph.shape)

    # input("Press any key to continue...")

    # points, p_lengths = batch_to_pack(points)
    # query_points, q_lengths = batch_to_pack(query_points)

    # q_batch_vector = lengths_to_batchvector(q_lengths)
    # p_batch_vector = lengths_to_batchvector(p_lengths)
    # print(q_batch_vector)

   
    # # graph = radius_search(query_points, points, radius, neighbor_limit, q_batch_vector, p_batch_vector, loop=True)

    # # graph = pairs_to_tensor(graph)

    # print(k_graph.shape)

    # input("Press any key to continue...")


    ######
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from utils import constants
    from utils import pointcloud_processing as eda
    from core.datasets.TS40K import TS40K_FULL_Preprocessed, TS40K_FULL
    from core.sampling.FPS import Farthest_Point_Sampling


    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )

    num_points = 5000
    fps = Farthest_Point_Sampling(num_points)

    sample = ts40k[0]
    points, labels = sample[0], sample[1]

    concat = torch.cat([points, labels.reshape(-1, 1)], dim=1) # (N, 3 + C)

    print(concat.shape)

    q_concat = fps(concat)
    query_points, q_labels = q_concat[..., :-1], q_concat[..., -1]

    print(query_points.shape, q_labels.shape)
    

    radius = 0.1
    neighbor_limit = 100

   
    graph = keops_radius_search(query_points[:, :3], concat[:, :3], radius, neighbor_limit)
    # print(graph.shape)
    
    eda.plot_pointcloud(points.cpu().numpy(), labels.cpu().long().numpy(), window_name='Original Point Cloud', use_preset_colors=True)
    # plot a few clusters
    for i in range(1000):
        i = torch.randint(0, graph.shape[1], (1,)).item()
        cluster = graph[i].numpy()
        cluster = cluster[cluster != -1]
        if 5 not in labels[cluster].unique():
            continue
        print(f"number of dbscan neighbors: {len(cluster)}")
        # print(cluster)
        if len(cluster) == 0:
            continue
        eda.plot_pointcloud(points[cluster].numpy(), labels[cluster].numpy(), window_name=f'Cluster {i}', use_preset_colors=True)

    
    
    