import datetime
import itertools
import os
import random
from typing import List, Literal, NamedTuple, Optional

import numpy
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from tap import Tap
from torch import LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import (FacebookPagePage, SNAPDataset,
                                      WikipediaNetwork)
from torch_geometric.utils import negative_sampling, to_undirected
from tqdm import tqdm

import tools


class Arguments(Tap):
    dataset: List[
        Literal[
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-collab",
            "soc-epinions1",
            "soc-livejournal1",
            "soc-pokec",
            "soc-slashdot0811",
            "soc-slashdot0922",
            "facebook",
            "wikipedia",
        ]
    ]
    # the dataset to run the experiment on
    dataset_dir: str = "data"  # directory containing the dataset files
    method: List[
        Literal[
            "jaccard",
            "adamic-adar",
            "common-neighbors",
            "dothash",
            "dothash-LE",
            "resource-allocation",
            "dothash-jaccard",
            "dothash-adamic-adar",
            "dothash-common-neighbors",
            "dothash-resource-allocation",
            "minhash",
            "simhash",
            "salton",
            "hdi",
            "hpi",
            "dothash-salton",
            "dothash-hdi",
            "dothash-hpi",
        ]
    ]
    # method to run the experiment with
    dimensions: List[int]
    # number of dimensions to use (does not affect the exact method)
    batch_size: int = 16384  # number of nodes to evaluate at once
    result_dir: str = "results"  # directory to write the results to
    device: List[str] = ["cpu"]  # which device to run the experiment on
    seed: List[int] = [1]  # random number generator seed


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed


class Result(NamedTuple):
    output_pos: torch.Tensor
    output_neg: torch.Tensor
    init_time: float
    calc_time: float
    dimensions: int


METRICS = [
    "method",
    "dataset",
    "dimensions",
    "hits@20",
    "hits@50",
    "hits@100",
    "init_time",
    "calc_time",
    "num_node_pairs",
    "device",
]


class Method:
    signatures: Optional[Tensor]
    num_neighbors: Optional[Tensor]

    def __init__(self, dimensions, batch_size, device):
        self.dimensions = dimensions
        self.device = device
        self.batch_size = batch_size

    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        NotImplemented

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        """
        Calculate the scores for the given node IDs and other node IDs.

        Parameters:
            node_ids (LongTensor): The tensor of node IDs.
            other_ids (LongTensor): The tensor of other node IDs.

        Returns:
            Tensor: The tensor containing the scores.
        """
        NotImplemented


class Jaccard(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )


class AdamicAdar(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        values = node_scaling[edge_index[1]]
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )

#--------------------dothash-common-neighbours----------
class CommonNeighbors(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class ResourceAllocation(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_scaling = tools.get_resource_allocation_node_scaling(edge_index, num_nodes)
        self.signatures = tools.to_scipy_csr_array(
            edge_index.cpu(), num_nodes, node_scaling[edge_index[1]]
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )
class DotHashAdamicAdar(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        node_vectors.mul_(node_scaling.unsqueeze(1))
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class DotHashCommonNeighbors(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class DotHashResourceAllocation(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        node_scaling = tools.get_resource_allocation_node_scaling(edge_index, num_nodes)
        node_vectors.mul_(node_scaling.unsqueeze(1))
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )

####---my portion of code edit for dothash-salton and others ---------#
class Salton(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))  # unweighted adjacency
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_salton(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )

class HDI(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))  # unweighted adjacency
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_hdi(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )

class HPI(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        values = torch.ones(edge_index.size(1))  # unweighted adjacency
        self.signatures = tools.to_scipy_csr_array(edge_index.cpu(), num_nodes, values)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_hpi(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )

class DotHashSalton(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(edge_index, node_vectors, self.batch_size)
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_salton(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )
#####------------------dothash-hdi-index------------------------------------#

class DotHashHDI(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_hdi(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )
#----------------------------dothash-hpi--------------------------------------#   
class DotHashHPI(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_hpi(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )

class DotHashJaccard(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )

class DotHashLE(Method):
    def __init__(self, dimensions, batch_size, device):
        super().__init__(dimensions, batch_size, device)

    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        self.signatures = tools.train_dothash_embedding(
            edge_index=edge_index,
            num_nodes=num_nodes,
            dims=self.dimensions,
            epochs=150,
            lr=0.01,
            margin=0.5,
            batch_size=self.batch_size,
            device=self.device,
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.dot(self.signatures[node_ids], self.signatures[other_ids])
    

class DotHash(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        """
        Initializes the signatures for the given edge index and number of nodes.

        Parameters:
            edge_index (LongTensor): The tensor of edge indices.
            num_nodes (int): The number of nodes.

        Returns:
            None
        """
        node_vectors = tools.get_random_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        node_scaling = tools.get_adamic_adar_node_scaling(edge_index, num_nodes)
        node_vectors.mul_(node_scaling.unsqueeze(1))
        self.signatures = tools.get_node_signatures(
            edge_index, node_vectors, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        """
        Calculate the dot product of the signatures of the given node IDs and other node IDs.

        Parameters:
            node_ids (LongTensor): The tensor of node IDs.
            other_ids (LongTensor): The tensor of other node IDs.

        Returns:
            Tensor: The tensor containing the dot product of the signatures.
        """
        return tools.dot(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class MinHash(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        self.signatures = tools.get_minhash_signatures(
            edge_index, num_nodes, self.dimensions, self.batch_size
        )

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.minhash_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
        )


class SimHash(Method):
    def init_signatures(self, edge_index: LongTensor, num_nodes: int):
        node_vectors = tools.get_random_binary_node_vectors(
            num_nodes, self.dimensions, device=self.device
        )
        self.signatures = tools.get_simhash_signatures(
            edge_index, node_vectors, self.batch_size
        )
        self.num_neighbors = tools.get_num_neighbors(edge_index, num_nodes).float()

    def calc_scores(self, node_ids: LongTensor, other_ids: LongTensor) -> Tensor:
        return tools.simhash_jaccard(
            self.signatures[node_ids],
            self.signatures[other_ids],
            self.num_neighbors[node_ids],
            self.num_neighbors[other_ids],
        )


class LinkPredDataset:
    def __init__(self, graph: Data):
        """
        Constructor for LinkPredDataset.

        Parameters
        ----------
        graph : Data
            The input graph.

        Notes
        -----
        The constructor performs the following tasks:

        1. Saves the input graph as an instance variable.
        2. Computes the negative edges using  sparse negative sampling, 
        which involves generating random edges that do not exist in the graph.
        3. Permutes the indices of the edges and
        splits them into training edges and test edges.
        4. Creates a new Data object with the 
        training edges and saves it as an instance variable.
        """
        #----------------------------------------------step 5----------------------------------------------
        self.graph = graph

        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        num_test_edges = num_edges // 20

        self.edge_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            method="sparse",
            force_undirected=True,
            num_neg_samples=num_test_edges,
        )

        permuted_indices = torch.randperm(num_edges)
        self.edge = edge_index[:, permuted_indices[:num_test_edges]]
        self.data = Data(
            edge_index=edge_index[:, permuted_indices[num_test_edges:]],
            num_nodes=num_nodes,
        )

    def get_edge_split(self):
        """
        Returns the edge split for this dataset.

        The edge split is a dictionary of dictionaries, with keys 'train', 'valid', and 'test'. 
        The 'test' key contains two keys: 'edge' and 'edge_neg', for the positive and negative test edges respectively.

        Returns
        -------
        dict
            A dictionary containing the edge split.
        """
        train = None
        valid = None

        test = {
            "edge": self.edge.T,
            "edge_neg": self.edge_neg.T,
        }

        return {"train": train, "valid": valid, "test": test}


def evaluate_hits_at(pred_pos: Tensor, pred_neg: Tensor, K: int) -> float:
    """
    Evaluate the hits@K for a given list of positive and negative edge scores.

    Parameters
    ----------
    pred_pos : Tensor
        The positive edge scores.
    pred_neg : Tensor
        The negative edge scores.
    K : int
        The number of negative edges to consider.

    Returns
    -------
    float
        The hits@K of the prediction.

    Notes
    -----
    If the number of negative edges is less than K, return 1.0.
    """
    #------------------------------------------------------------step 11---------------------------------------------------------------
    if len(pred_neg) < K:
        return 1.0

    kth_score_in_negative_edges = torch.topk(pred_neg, K)[0][-1]
    num_hits = torch.sum(pred_pos > kth_score_in_negative_edges).cpu()
    hitsK = float(num_hits) / len(pred_pos) ### main formula for calculation  of hit@k
    return hitsK


def executor(args: Arguments, method: Method, dataset, device=None):
    """
    Evaluate a link prediction method on a given dataset.

    Parameters
    ----------
    args : Arguments
        The arguments containing the dataset, method, and other parameters.
    method : Method
        The link prediction method to evaluate.
    dataset : LinkPredDataset
        The dataset to evaluate on.
    device : torch.device, optional
        The device to run the evaluation on. If None, use the CPU.

    Returns
    -------
    Result
        The result of the evaluation, 
        containing the positive and negative scores, 
        the time taken to initialize the method, 
        the time taken to calculate the scores, 
        and the number of dimensions used.
    """
    graph = dataset.data.to(device)
    split_edge = dataset.get_edge_split()
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    get_duration = tools.stopwatch()
    method.init_signatures(to_undirected(graph.edge_index), graph.num_nodes)
    init_time = get_duration()

    pos_scores = []
    neg_scores = []
    pos_test_edge_loader = pos_test_edge.split(args.batch_size)
    neg_test_edge_loader = neg_test_edge.split(args.batch_size)

    get_duration = tools.stopwatch()
    for edge_batch in tqdm(pos_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        pos_scores.append(scores.cpu())

    for edge_batch in tqdm(neg_test_edge_loader, leave=False):
        node_ids, other_ids = edge_batch[:, 0], edge_batch[:, 1]
        scores = method.calc_scores(node_ids, other_ids)
        neg_scores.append(scores.cpu())

    calc_time = get_duration()
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)

    dimensions = method.signatures.shape[1]

    return Result(pos_scores, neg_scores, init_time, calc_time, dimensions)


def get_dataset(name: str, root: str):
    """
    Gets a dataset by name from the given root directory.

    Args:
        name (str): Name of the dataset. Supported datasets are:
            - ogbl-*: OGB link prediction datasets.
            - soc-*: SNAP graph datasets.
            - facebook: Facebook Page-Page network dataset.
            - wikipedia: Wikipedia Network dataset.
        root (str): Root directory containing the dataset.

    Returns:
        LinkPredDataset: The dataset object.

    Raises:
        NotImplementedError: If the given name is not supported.
    """

    if name.startswith("ogbl-"):
        return PygLinkPropPredDataset(name, root)
    elif name.startswith("soc-"):
        dataset = SNAPDataset(root, name)
        return LinkPredDataset(dataset.data)
    elif name == "facebook":
        dataset = FacebookPagePage(os.path.join(root, "facebookpagepage"))
        #---------------------------------------Step 4-------------------------------------------
        return LinkPredDataset(dataset.data)
    elif name == "wikipedia":
        dataset = WikipediaNetwork("data", "crocodile", geom_gcn_preprocess=False)
        return LinkPredDataset(dataset.data)
    else:
        raise NotImplementedError()


def get_hits(result: Result):
    """
    Evaluates the hits@K for the given positive and negative edge scores.

    Parameters
    ----------
    result : Result
        The result object containing the positive and negative edge scores.

    Returns
    -------
    tuple
        A tuple containing the hits@20, hits@50, and hits@100 of the prediction.
    """
    
    output = []

    for K in [20, 50, 100]:
        #-------------------------step 10------------------------------------
        test_hits = evaluate_hits_at(result.output_pos, result.output_neg, K)
        output.append(test_hits)

    return tuple(output)


def get_metrics(conf: Config, args: Arguments, dataset, device=None):
    """
    Evaluates a link prediction method on a given dataset.

    Parameters
    ----------
    conf : Config
        The configuration containing the method, dimensions, and other parameters.
    args : Arguments
        The arguments containing the dataset, method, and other parameters.
    dataset : LinkPredDataset
        The dataset to evaluate on.
    device : torch.device, optional
        The device to run the evaluation on. If None, use the CPU.

    Returns
    -------
    dict
        A dictionary containing the results of the evaluation, 
        including the number of dimensions used, the hits at 20, 50, and 100, the time taken to initialize the method, 
        the time taken to calculate the scores, and the number of node pairs used.
    """
    #-------------------------------------step 7----------------------------------------------------------------------
    if conf.method == "jaccard":
        method_cls = Jaccard
    elif conf.method == "adamic-adar":
        method_cls = AdamicAdar
    elif conf.method == "dothash-jaccard":
        method_cls = DotHashJaccard
    elif conf.method == "dothash":
        method_cls = DotHash
    elif conf.method == "dothash-LE":
        method_cls = DotHashLE
    elif conf.method == "minhash":
        method_cls = MinHash
    elif conf.method == "simhash":
        method_cls = SimHash
    elif conf.method == "salton":
        method_cls = Salton
    elif conf.method == "dothash-salton":
        method_cls = DotHashSalton
    elif conf.method == "adamic-adar":
        method_cls = AdamicAdar
    elif conf.method == "dothash-adamic-adar":
        method_cls = DotHashAdamicAdar
    elif conf.method == "common-neighbors":
        method_cls = CommonNeighbors
    elif conf.method == "dothash-common-neighbors":
        method_cls = DotHashCommonNeighbors
    elif conf.method == "resource-allocation":
        method_cls = ResourceAllocation
    elif conf.method == "dothash-resource-allocation":
        method_cls = DotHashResourceAllocation
    elif conf.method == "hdi":
        method_cls = HDI
    elif conf.method == "dothash-hdi":
        method_cls = DotHashHDI
    elif conf.method == "hpi":
        method_cls = HPI
    elif conf.method == "dothash-hpi":
        method_cls = DotHashHPI
    else:
        raise NotImplementedError(f"requested method {conf.method} is not implemented")

    method = method_cls(conf.dimensions, args.batch_size, device)
    #------------------------------------------step 8-------------------------------------------------------------
    result = executor(args, method, dataset, device=device)
    total_time = result.init_time + result.calc_time
    num_node_pairs = result.output_pos.size(0) + result.output_neg.size(0)
    #--------------------------------------------step 9 -------------------------------------------------------
    hits = get_hits(result)
    print(
        f"{conf.method}: Hits: {hits[0]:.3g}@20, {hits[1]:.3g}@50, {hits[2]:.3g}@100; Time: {result.init_time:.3g}s + {result.calc_time:.3g}s = {total_time:.3g}s; Dims: {result.dimensions}"
    )

    return {
        "dimensions": result.dimensions,
        "hits@20": hits[0],
        "hits@50": hits[1],
        "hits@100": hits[2],
        "init_time": result.init_time,
        "calc_time": result.calc_time,
        "num_node_pairs": num_node_pairs,
    }


def main(conf: Config, args: Arguments, result_file: str):

    """
    Main entry point for the link prediction evaluation.

    This function evaluates the specified link prediction method on the specified dataset.

    Parameters
    ----------
    conf : Config
        Configuration containing the evaluation parameters.
    args : Arguments
        Program arguments.
    result_file : str
        File to write the results to.

    Returns
    -------
    None
    """
    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    random.seed(conf.seed)

    print("Device:", conf.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tools.open_metric_writer(result_file, METRICS) as write:

        print("Dataset:", conf.dataset)
        # -------------------------step 3----------------------------------------------------------------------------------
        dataset = get_dataset(conf.dataset, args.dataset_dir)

        try:
            # ----------------------------------------------step 6-------------------------------------------------
            metrics = get_metrics(conf, args, dataset, device=conf.device)
            metrics["method"] = conf.method
            metrics["dataset"] = conf.dataset
            metrics["device"] = conf.device.type
            write(metrics)
        except Exception as e:
            print(e)


def default_to_cpu(device: str) -> torch.device:
    """
    Returns a torch.device object based on CUDA availability.

    Parameters
    ----------
    device : str
        The device type as a string, typically 'cuda' or 'cpu'.

    Returns
    -------
    torch.device
        A torch.device object pointing to the specified device 
        if CUDA is available,
        otherwise defaults to the CPU.
    """

    if torch.cuda.is_available():
        return torch.device(device)
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # command-line argument to run pythone file is given below
    # python link_prediction.py --dataset wikipedia --method dothash --dimensions 10678 1000 8999  --device cpu
    #########-------------------step 1---------------------------------------------------------------
    #Parse command-line arguments, 
    args = Arguments(underscores_to_dashes=True).parse_args()

    result_filename = (
        "link_prediction-"
        + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        + ".csv"
    )

    # Generate the path to the result file
    result_file = os.path.join(args.result_dir, result_filename)
    os.makedirs(args.result_dir, exist_ok=True)

    devices = {default_to_cpu(d) for d in args.device}
    #Iterate over all possible combinations of, 
    #device, dimensions, dataset, and method.
    options = (args.seed, devices, args.dimensions, args.dataset, args.method)
    for seed, device, dimensions, dataset, method in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed)
        #--------------------------------------------step 2------------------------------------------
        #Entry point
        main(config, args, result_file)
