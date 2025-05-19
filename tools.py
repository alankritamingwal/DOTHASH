import contextlib
import csv
import functools
import itertools
import math
import operator as op
import os
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Union

import scipy.sparse as ssp
import torch
import torchhd
from torch import LongTensor, Tensor

# The size of a hash value in number of bytes
hashvalue_byte_size = 8

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = torch.tensor((1 << 61) - 1, dtype=torch.long)
_max_hash = torch.tensor((1 << 32) - 1, dtype=torch.long)
_hash_range = 1 << 32

DenseOrSparse = Union[Tensor, ssp.csr_array]

def to_scipy_csr_array(edge_index, num_nodes, values):
    return ssp.csr_array((values, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))


def dot(input: DenseOrSparse, other: DenseOrSparse) -> Tensor:
    return torch.as_tensor((input * other).sum(-1), dtype=torch.float)

    
def dot_jaccard(
    set_a: DenseOrSparse,
    set_b: DenseOrSparse,
    size_a: Tensor,
    size_b: Tensor,
    eps: float = 1e-8,
):
    size_i = dot(set_a, set_b)
    return size_i / (size_a + size_b - size_i + eps)

####-------my added porton--------------######
def dot_salton(
    set_a: Tensor, set_b: Tensor, size_a: Tensor, size_b: Tensor, eps=1e-8
) -> Tensor:
    intersection = dot(set_a, set_b)  # ≈ |A ∩ B|
    denominator = torch.sqrt(size_a * size_b + eps)
    return intersection / denominator

def dot_hdi(
    set_a: Tensor, set_b: Tensor, size_a: Tensor, size_b: Tensor, eps: float = 1e-8
) -> Tensor:
    intersection = dot(set_a, set_b)
    max_size = torch.max(size_a, size_b)
    return intersection / (max_size + eps)

def dot_hpi(
    set_a: Tensor, set_b: Tensor, size_a: Tensor, size_b: Tensor, eps: float = 1e-8
) -> Tensor:
    intersection = dot(set_a, set_b)
    min_size = torch.min(size_a, size_b)
    return intersection / (min_size + eps)


def minhash_jaccard(set_a: Tensor, set_b: Tensor) -> Tensor:
    return (set_a == set_b).float().mean(dim=1)


def simhash_intersection(
    set_a: Tensor, set_b: Tensor, size_a: Tensor, size_b: Tensor
) -> Tensor:
    max_intersection_size = torch.minimum(size_a, size_b)
    cos = torch.sum(set_a == set_b, dim=1).div(set_a.size(1)).clamp(min=0)
    return cos * max_intersection_size


def simhash_jaccard(
    set_a: Tensor, set_b: Tensor, size_a: Tensor, size_b: Tensor, eps=1e-8
) -> Tensor:
    size_i = dot(set_a, set_b)
    return size_i / (size_a + size_b - size_i + eps)


def get_minhash_signatures(
    edge_index: LongTensor,
    num_nodes: int,
    dimensions: int,
    batch_size: int,
) -> LongTensor:
    device = edge_index.device
    node_ids = torch.arange(0, num_nodes).unsqueeze_(1)

    # Create parameters for a random bijective permutation function
    # http://en.wikipedia.org/wiki/Universal_hashing
    a, b = torch.randint(0, _mersenne_prime, (2, 1, dimensions))
    node_vectors = torch.bitwise_and((node_ids * a + b) % _mersenne_prime, _max_hash)

    to_nodes, from_nodes = edge_index
    # index_reduce_ is only implemented for long type on CPU
    to_batches = torch.split(to_nodes.cpu(), batch_size)
    from_batches = torch.split(from_nodes.cpu(), batch_size)

    size = (num_nodes, dimensions)
    signatures = torch.full(size, _max_hash, dtype=torch.long)
    for to_batch, from_batch in zip(to_batches, from_batches):
        from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
        signatures.index_reduce_(0, to_batch, from_node_vectors, "amin")

    return signatures.to(device)


def get_simhash_signatures(
    edge_index: LongTensor, node_vectors: Tensor, batch_size: int
) -> Tensor:
    to_nodes, from_nodes = edge_index

    to_batches = torch.split(to_nodes, batch_size)
    from_batches = torch.split(from_nodes, batch_size)

    signatures = torch.zeros_like(node_vectors)
    for to_batch, from_batch in zip(to_batches, from_batches):
        from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
        signatures.index_add_(0, to_batch, from_node_vectors)

    return signatures.greater(0)


def get_random_node_vectors(num_nodes: int, dimensions: int, device=None) -> Tensor:
    scale = math.sqrt(1 / dimensions)

    node_vectors = torchhd.random(num_nodes, dimensions, device=device)
    node_vectors.mul_(scale)  # make them unit vectors
    return node_vectors


def dot_signature_ordering(
        num_nodes: int,
        dimensions: int,
):
    difference = num_nodes - dimensions
    node_vectors = torch.eye(num_nodes, dimensions)
    node_ids = torch.arange(dimensions, num_nodes).unsqueeze_(1)

    a, b = torch.randint(0, _mersenne_prime, (2, 1, difference))
    c, d = torch.randint(0, _mersenne_prime, (2, 1, difference))

    hash_indices = torch.bitwise_and((node_ids * a + b) % _mersenne_prime, _max_hash) % dimensions
    hash_signs = torch.bitwise_and((node_ids * c + d) % _mersenne_prime, _max_hash) % 2
    hash_signs = torch.where(hash_signs == 0, -1, hash_signs).float()

    node_vectors[hash_indices] += hash_signs

    return node_vectors


def get_random_binary_node_vectors(
    num_nodes: int, dimensions: int, device=None
) -> Tensor:
    return torchhd.random(num_nodes, dimensions, device=device)


def get_num_neighbors(edge_index: LongTensor, num_nodes: int) -> LongTensor:
    to_nodes, _ = edge_index
    num_neighbors = to_nodes.bincount(minlength=num_nodes)
    return num_neighbors


def get_adamic_adar_node_scaling(edge_index: LongTensor, num_nodes: int) -> Tensor:
    num_neighbors = get_num_neighbors(edge_index, num_nodes).float()

    device = edge_index.device
    penalty = torch.zeros(num_nodes, device=device)
    # ensure numerical stability in log and sqrt with zero or one neighbors
    at_least_2 = num_neighbors >= 2.0
    penalty[at_least_2] = torch.sqrt(1 / torch.log(num_neighbors[at_least_2]))
    return penalty


def get_resource_allocation_node_scaling(
    edge_index: LongTensor, num_nodes: int
) -> Tensor:
    num_neighbors = get_num_neighbors(edge_index, num_nodes).float()

    device = edge_index.device
    penalty = torch.zeros(num_nodes, device=device)
    # ensure numerical stability in log and sqrt with zero or one neighbors
    at_least_1 = num_neighbors >= 1.0
    penalty[at_least_1] = torch.sqrt(1 / num_neighbors[at_least_1])
    return penalty


def get_node_signatures(
    edge_index: LongTensor, node_vectors: Tensor, batch_size: int
) -> Tensor:
    to_nodes, from_nodes = edge_index

    to_batches = torch.split(to_nodes, batch_size)
    from_batches = torch.split(from_nodes, batch_size)

    signatures = torch.zeros_like(node_vectors)
    for to_batch, from_batch in zip(to_batches, from_batches):
        from_node_vectors = torch.index_select(node_vectors, 0, from_batch)
        signatures.index_add_(0, to_batch, from_node_vectors)

    return signatures


def all_node_pairs(num_nodes: int):
    node_ids = range(num_nodes)
    return itertools.combinations(node_ids, r=2)


def chunks(iterable: Iterable, size: int):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))


def node_pairs_loader(
    num_nodes: int, batch_size: int, device=None
) -> Iterable[LongTensor]:
    node_pairs = all_node_pairs(num_nodes)
    for batch in chunks(node_pairs, batch_size):
        yield torch.tensor(list(batch), dtype=torch.long, device=device)


def ncr(n, r):
    r = min(r, n - r)
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


@contextlib.contextmanager
def open_metric_writer(filename: str, columns: List[str]):
    """starts a metric writer contexts that will write the specified columns to a csv file"""

    file = open(filename, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, columns)

    if os.path.getsize(filename) == 0:
        writer.writeheader()

    def write(metrics: Dict[str, Any]) -> None:
        writer.writerow(metrics)
        file.flush()  # make sure latest metrics are saved to disk

    yield write

    file.close()


def stopwatch() -> Callable[..., float]:
    start = perf_counter()

    def stop() -> float:
        return perf_counter() - start

    return stop

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

#------------------------------------------changes------------------------------#




def train_dothash_embedding(edge_index: torch.LongTensor, num_nodes: int, dims: int,
                            epochs: int = 20, lr: float = 0.01, margin: float = 0.1,
                            batch_size: int = 1024, device=None, early_stop_patience: int = 5) -> torch.Tensor:
    """
    Trains node embeddings φ(x) using ranking-based contrastive learning for DotHash,
    with support for early stopping.

    Parameters:
        edge_index (Tensor): Edge list in shape [2, num_edges] where each column is (u, v).
        num_nodes (int): Total number of nodes in the graph.
        dims (int): Dimension of each node's embedding vector.
        epochs (int): Maximum number of training iterations (epochs).
        lr (float): Learning rate for the Adam optimizer.
        margin (float): Margin value used in ranking loss between positive and negative pairs.
        batch_size (int): Number of edge updates processed per batch.
        device (torch.device): CPU or GPU device to use for training.
        early_stop_patience (int): Number of consecutive epochs to wait before early stopping
                                 if loss does not improve.

    Returns:
        Tensor: Learned DotHash signature vectors (sketches) for each node, shape [num_nodes, dims].
    """

    # Use given device or infer it from edge_index
    device = device or edge_index.device

    # Initialize an embedding matrix: φ(x) ∈ ℝ^dims for each node x
    embedding = nn.Embedding(num_nodes, dims).to(device)

    # Use Adam optimizer (adaptive learning rate) to update embedding vectors
    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr)

    # Extract positive edge pairs from the graph (edges in edge_index)
    pos_u, pos_v = edge_index

    # Generate negative edge pairs that do NOT exist in the graph
    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=pos_u.size(0))
    neg_u, neg_v = neg_edge_index

    # Initialize variables for early stopping
    best_loss = float('inf')  # Keep track of the lowest loss seen so far
    epochs_without_improvement = 0  # Counter for how many epochs had no improvement

    # Training loop
    for epoch in range(epochs):
        # Reset gradients before each epoch
        optimizer.zero_grad()

        # Step 1: Generate signature vector for each node by summing its neighbors' embeddings
        # These are the DotHash sketches used in link prediction
        signatures = get_node_signatures(edge_index, embedding.weight, batch_size)

        # Step 2: Compute dot product (similarity score) for positive edges
        pos_score = (signatures[pos_u] * signatures[pos_v]).sum(dim=1)

        # Step 3: Compute dot product for negative edges
        neg_score = (signatures[neg_u] * signatures[neg_v]).sum(dim=1)

        # Step 4: Ranking-based loss — encourage positive scores to be greater than negative by a margin
        # If this is already true, loss becomes zero (no update needed)
        loss = F.relu(margin - (pos_score - neg_score)).mean()

        # Step 5: Print the loss for this epoch for monitoring
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        # Step 6: Compute gradients and update embeddings
        loss.backward()
        optimizer.step()

        # Step 7: Check if the loss improved compared to the best seen so far
        # We use a small threshold (1e-5) to avoid false triggers from floating point noise
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()                     # Update best loss
            epochs_without_improvement = 0              # Reset counter
        else:
            epochs_without_improvement += 1             # Count stagnation epochs

        # Step 8: Trigger early stopping if no improvement for N consecutive epochs
        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs).")
            break  # Stop training early

    # Final step: Return the final sketch signatures (DotHash-style vectors)
    return get_node_signatures(edge_index, embedding.weight, batch_size)
