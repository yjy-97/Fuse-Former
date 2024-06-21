import torch
import torch.nn.functional as F
import os
import numpy as np

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

def kmeans(x, k, max_iters=100):
    x = x.permute((0, 2, 1))
    x = torch.mean(x, dim=0).squeeze()
    centroids = x[torch.randperm(x.size(0))[:k]]
    cluster_indices = []

    for _ in range(max_iters):
        distances = pairwise_cos_sim(x, centroids)

        labels = torch.argmax(distances, dim=1)

        new_centroids = torch.stack([x[labels == i].mean(0) for i in range(k)])

        if torch.all(new_centroids == centroids):
            cluster_correlations = []
            for i in range(k):
                cluster_points = x[labels == i]
                if len(cluster_points) > 1:
                    cluster_points_np = cluster_points.cpu().detach().numpy()
                    cluster_points_np = np.transpose(cluster_points_np)
                    cluster_correlation = np.corrcoef(cluster_points_np, rowvar=False)
                    cluster_correlations.append(cluster_correlation)

            output_dir=r''
            os.makedirs(output_dir, exist_ok=True)
            for i, correlation_matrix in enumerate(cluster_correlations):
                file_path = os.path.join(output_dir, f"cluster_{i}__correlation_matrix.txt")
                np.savetxt(file_path, correlation_matrix, fmt='%f')

            max_value = torch.max(distances, dim=1)
            print(max_value[0])

            break

        centroids = new_centroids
        cluster_indices = [torch.where(labels == i)[0] for i in range(k)]

    return labels, centroids, cluster_indices