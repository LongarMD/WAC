import math
import torch

from typing import List, Tuple

# ===============================================
# Distance Methods
# ===============================================


def cosine_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> float:
    """Calculates the cosine similarity between two vectors
    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.
    Returns:
        cosine_similarity (float): The cosine similarity between the vectors.
    """
    return vector1.dot(vector2).item()


def jaccard_index(s1: set, s2: set) -> float:
    """Gets the Jaccard Index
    Calculates the Jaccard Index using the following equation:
        Jaccard(s1, s2) = \\frac{s1 \\cap s2}{s1 \\cup s2}
    Args:
        s1 (set): The first set.
        s2 (set): The second set.
    Returns:
        jaccard_index (float): The Jaccard Index of the two sets.
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return len(s1 & s2) / len(s1 | s2)


def avg_overlap(s1: set, s2: set) -> float:
    """Gets the average overlap
    Calculates the average overlap using the following equation:
        avg_overlap(s1, s2) = (\\frac{|s1 \\cap s2|}{|s1|} + \\frac{|s1 \\cap s2|}{|s2|}) / 2
    Args:
        s1 (set): The first set.
        s2 (set): The second set.
    Returns:
        avg_overlap (float): The average overlap of two sets.
    """
    if len(s1) == 0 or len(s2) == 0:
        return None
    return sum([len(s1 & s2) / len(s) for s in [s1, s2]]) / 2


# ===============================================
# Cluster Methods
# ===============================================


def get_centroid(embeds: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    """Calculates the centroid
    Calculates the centroid with the following equations:
        c = \\frac{sum_{i=1}^{k} a_{i}}{k}
        c = \\frac{c}{||c||}
    Args:
        embeds (List[torch.Tensor]): The embedding list of articles
            in the cluster.
    Returns:
        centroid (torch.Tensor): The normalized centroid.
        c_norm (float): The centroids norm before normalization.
    """
    X = torch.cat(tuple([embed.unsqueeze(0) for embed in embeds]), 0)
    centroid = torch.sum(X, 0) / X.shape[0]
    # calculate the centroid norm
    c_norm = torch.linalg.vector_norm(centroid, ord=2).item()
    # normalize the centroid
    centroid = centroid / c_norm
    return centroid, c_norm


def update_centroid(
    centroid: torch.Tensor, c_norm: float, n_articles: int, a_embed: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Updates the centroid
    Updates the centroid with the following equations:
        c_i = \\frac{n_{i-1} * ||c_{i-1}|| * c_{i-1} + a_{i}}{n_{i}}
        c_i = \\frac{c_i}{||c_i||}
    Args:
        centroid (torch.Tensor): The current centroids tensor. Corresponds to
            c_{i-1} in the equation.
        c_norm (float): The current centroids norm. Corresponds to ||c_{i-1}||
            in the equation.
        n_articles (int): The previous number of articles in the cluster.
            Corresponds to n_{i-1} in the equation.
        a_embed (torch.Tensor): The new articles tensor. Corresponds to a_{i}
            in the equation.
    Returns:
        centroid (torch.Tensor): The updated normalized centroid.
        c_norm (float): The updated centroids norm before normalization.
    """
    centroid *= n_articles * c_norm
    centroid += a_embed
    centroid /= n_articles + 1
    c_norm = torch.linalg.vector_norm(centroid, ord=2).item()
    centroid = centroid / c_norm
    return centroid, c_norm


# ===============================================
# Statistics Methods
# ===============================================


def get_max(vals: List[float]) -> float:
    return max(vals)


def get_min(vals: List[float]) -> float:
    return min(vals)


def get_avg(vals: List[float]) -> float:
    return sum(vals) / len(vals)
