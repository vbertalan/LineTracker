from pathlib import Path
from typing import *  # type: ignore
import json
import re
import datetime
import gc
import os
from itertools import product
from textwrap import wrap
import argparse
import abc
import shutil
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import psutil
import multiprocessing as mp
import functools
import random
import fire
import torch
from torch import nn
import itertools as it
import sklearn.preprocessing as skPrepro
import sklearn.metrics as skMetrics
from sklearn.feature_extraction.text import TfidfVectorizer
import uuid
import pandas as pd
import contextlib
import hashlib
from sklearn.cluster import DBSCAN
import ctypes
import functools as ft
from sklearn.metrics import adjusted_rand_score, silhouette_score, jaccard_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.pairwise import pairwise_distances

try:
    import DrainMethod
    import drainmethod_functionnal as drain_func
except ImportError:
    import src.DrainMethod as DrainMethod
    import src.drainmethod_functionnal as drain_func

"""File of code: disclaimer functions comes from the repository https://github.com/AndressaStefany/severityPrediction"""
# tye hints
LlamaTokenizer = Union["trf.LlamaTokenizer", "trf.LlamaTokenizerFast"]
LlamaModel = "trf.LlamaForCausalLM"
PoolingOperationCode = Literal["mean", "sum"]
PoolingFn = Callable[["torch.Tensor"], "torch.Tensor"]
ModelName = Literal["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
DatasetName = Literal["eclipse_72k", "mozilla_200k"]
BugId: int
default_token: str = "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"
default_model: ModelName = "meta-llama/Llama-2-13b-chat-hf"
default_n_tokens_infered_max: int = 7364
default_input_field: str = "description"
if "USER" in os.environ:
    if os.environ["USER"] == "rmoine":
        default_folder_data: Path = Path(
            f"/scratch/{os.environ['USER']}/LineTracker/data"
        )
    elif os.environ["USER"] == "local_rmoine":
        default_folder_data: Path = Path(
            f"C:/Users/robin/Documents/projets/LineTracker/data/"
        )
    else:
        raise Exception
else:
    default_folder_data: Path = Path(f"/scratch/rmoine/LineTracker/data")
default_event_hdf5_path = (
    default_folder_data / "trat3_production_1650_1700_20231411_v1.hdf5"
)
default_path_embeddings_distances = default_folder_data / "distances.hdf5"
# typehint imports
if TYPE_CHECKING:
    import transformers as trf
    import torch
    import torch.nn as nn
    import torch.utils.data as dt
    import huggingface_hub
    import pandas as pd
    import numpy as np
    import peft
    import trl
    import matplotlib.pyplot as plt
    import matplotlib
    import sklearn.metrics as skMetr
    import sklearn.model_selection as skMsel
    import tqdm
    import datasets
    import h5py
    import bitsandbytes as bnb
    import evaluate  # type: ignore
    import optuna
    import accelerate


imports = [
    "import transformers as trf",
    "import torch",
    "import torch.nn as nn",
    "import torch.utils.data as dt",
    "import huggingface_hub",
    "import pandas as pd",
    "import numpy as np",
    "import peft",
    "import trl",
    "import matplotlib.pyplot as plt",
    "import matplotlib",
    "import sklearn.metrics as skMetr",
    "import sklearn.model_selection as skMsel",
    "import tqdm",
    "import datasets",
    "import h5py",
    "import bitsandbytes as bnb",
    "import evaluate",
    "import optuna",
    "import accelerate",
]
for i in imports:
    try:
        exec(i)
    except ImportError:
        print(f"Import of {i} failed")

try:
    from src.baseline.baseline_functions import *  # type: ignore
except Exception:
    pass


def existing_path(p: Union[str, Path], *, is_folder: bool) -> Path:
    p = Path(p)
    if not p.exists():
        raise Exception(f"{p.resolve()} does not exists")
    if p.is_dir() and not is_folder:
        raise Exception(f"{p.resolve()} is a folder not a file")
    if not p.is_dir() and is_folder:
        raise Exception(f"{p.resolve()} is a file not a folder")
    return p


def assert_valid_token(token: str):
    assert isinstance(token, str) and len(token) > 3 and token[:3] == "hf_"


def get_literal_value(model_name: str, literal: Any = ModelName) -> Any:
    assert isinstance(model_name, str) and model_name in get_args(literal)
    return model_name  # type: ignore


def get_dataset_choice(dataset_choice: str) -> DatasetName:
    assert isinstance(dataset_choice, str) and dataset_choice in get_args(DatasetName)
    return dataset_choice  # type: ignore


class CustomFormatter(logging.Formatter):
    def __init__(
        self, fmt=None, datefmt=None, style="%", validate: bool = True
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        try:
            self.total_ram_gpu = float(
                subprocess.check_output(
                    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0",
                    shell=True,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            self.total_ram_gpu = None
            pass

    def format(self, record):
        # Log CPU RAM usage
        ram = psutil.virtual_memory()
        cpu_ram_message = f"RAM {ram.used / (1024 ** 3):.3f}/{ram.total / (1024 ** 3):.3f}GB ({ram.used/ram.total:.2f}%)"

        # Log GPU VRAM usage (assuming a single GPU, adjust as needed)
        if self.total_ram_gpu is not None:
            used = float(
                subprocess.check_output(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0",
                    shell=True,
                )
                .decode("utf-8")
                .strip()
            )
            gpu_vram_message = f"GPU VRAM {used / (1024 ** 3):.3f}/{self.total_ram_gpu / (1024 ** 3):.3f}GB ({used/self.total_ram_gpu:.2f}%)"
        else:
            gpu_vram_message = "GPU VRAM <nan>"

        # Add the CPU RAM and GPU VRAM messages to the log record
        record.cpu_ram_message = cpu_ram_message
        record.gpu_vram_message = gpu_vram_message

        return super().format(record)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
custom_formatter = CustomFormatter(
    "%(asctime)s - %(levelname)s - %(message)s - %(cpu_ram_message)s - %(gpu_vram_message)s"
)
handler.setFormatter(custom_formatter)
logger.addHandler(handler)


def print_args(func):
    def inner(*args, **kwargs):
        print("Current time:", datetime.datetime.now())
        print("*" * 100)
        print("Start", func.__name__)
        print("With *args", args)
        print("With **kwargs", kwargs)
        print("-" * 100)
        return func(*args, **kwargs)

    return inner


class CienaLogData(TypedDict):
    """
    - dup_id: str, String that is the same for bugs that are the same
    - event_id: str, Unique string per bug_id
    - group_id: str, Group to put logs related together, 'true' class (found by CIENA) of algorithm
    - line_num: str, Plan id of the log: with log_name constitute the build_log
    - planid: str, Plan id of the log: with log_name constitute the build_log
    - log_name: str, Second part to make the build log
    - raw: str, Raw text of the bug
    - template: str, Template found by CIENA
    - variables: List[str], Variables found with the template of CIENA
    """

    dup_id: str
    event_id: str
    group_id: str
    line_num: str
    planid: str
    log_name: str
    raw: str
    template: str
    text: str
    variables: List[str]


class LogData(TypedDict):
    """
    - event_id: str, Unique string per bug_id
    - text: str, Text of the error
    - line_num: str, Plan id of the log: with log_name constitute the build_log
    """

    event_id: str
    text: str
    line_num: str


class TripletMatrix(TypedDict):
    """
    - variables_matrix: np.ndarray, matrix distances
    - embeddings_matrix: np.ndarray, embeddings distances
    - count_matrix: np.ndarray, count of line distances
    """

    variables_matrix: np.ndarray
    embeddings_matrix: np.ndarray
    count_matrix: np.ndarray


class TripletCoef(TypedDict):
    """
    - coef_variables_matrix: coefficient for the matrix distances
    - coef_embeddings_matrix: coefficient for the embeddings distances
    - coef_count_matrix: coefficient for the count of line distances
    """

    coef_variables_matrix: float
    coef_embeddings_matrix: float
    coef_count_matrix: float


class LineTrackerException(Exception):
    """Base class for the LineTracker exceptions"""


class EmptyLog(LineTrackerException):
    """Exception when no logs is found"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs


class NoVariable(LineTrackerException):
    """Exception when no variables are inside log lines"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs


class ParsingOutput(TypedDict):
    """
    - event_id: str, Unique string per bug_id
    - template: str, the template used in this event
    - variables: List[str], the variables in this event
    """

    event_id: str
    template: str
    variables: List[str]


class LogEmbeddingData(TypedDict):
    """
    - event_id: str, Unique string per bug_id
    - text: str, the source text provided to the model to make the embedding
    - embedding: np.ndarray, the embedding generated by the model
    """

    event_id: str
    text: str
    embedding: np.ndarray


class LogsEmbeddingData(TypedDict):
    """
    - event_id: str, Unique string per bug_id
    - text: str, the source text provided to the model to make the embedding
    - embedding: np.ndarray, the embedding generated by the model
    """

    event_ids: List[str]
    texts: List[str]
    embeddings: np.ndarray


class GroupDict(TypedDict):
    """
    event_id: str, Unique id for each event
    group_id: str, Group str common to all events that are in the same group. If 'None' there is no group assigned
    """

    event_id: str
    group_id: str


class TxtClustered(TypedDict):
    text: str
    cluster: int


class ClusteringOutput(TypedDict):
    hyperparameters_chosen: Dict[str, Any]
    clusters: List[int]
    texts: List[str]


def generate_combinations(params_dict):
    """
    Generate all combinations of hyperparameters from a dictionary of parameter names and possible values.

    # Parameters:
    - params_dict: Dictionary where keys are parameter names, and values are lists of possible values for each parameter.

    # Returns:
    A list of dictionaries representing all combinations of hyperparameters.
    """
    param_names = list(params_dict.keys())
    param_values = list(params_dict.values())

    hyperparameter_combinations = list(it.product(*param_values))

    hyperparameter_list = [
        {param_names[i]: combination[i] for i in range(len(param_names))}
        for combination in hyperparameter_combinations
    ]

    return hyperparameter_list


def get_build_log_name(data_sample: CienaLogData) -> str:
    return data_sample["planid"] + "--" + data_sample["log_name"]


def get_parser_cache_fn(parser_type: Literal["ciena", "drain"], depth: int = 5, similarity_threshold: float = 0.4, max_children: int = 3) -> Callable[[List[LogData]], List[ParsingOutput]]:  # type: ignore
    if parser_type == "ciena":
        return get_parsing_ciena  # type: ignore
    elif parser_type == "drain":
        return lambda events: get_parsing_drainparser(
            events,
            depth=depth,
            similarity_threshold=similarity_threshold,
            max_children=max_children,
        )


def get_default_parser_cache(
    parser_type: Literal["ciena", "drain"],
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
) -> Path:
    if parser_type == "ciena":
        return default_event_hdf5_path
    elif parser_type == "drain":
        return (
            default_folder_data
            / f"drain_parser__depth-{depth}__similarity_threshold-{str(similarity_threshold)[2:]}__max_children-{max_children}.hdf5"
        )
    else:
        raise Exception


def get_default_var_dist_matrix_cache(
    parser_type: Literal["ciena", "drain"],
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
) -> Path:
    if parser_type == "drain":
        return (
            default_folder_data
            / f"emb_var_distances__depth-{depth}__similarity_threshold-{str(similarity_threshold)[2:]}__max_children-{max_children}.hdf5"
        )
    elif parser_type == "ciena":
        return default_folder_data / f"emb_var_distances__{parser_type}.hdf5"


def get_embedder_fn(
    embedder_type: Literal["llama-13b", "tfidf"],
    pooling_code: PoolingOperationCode = "mean",
    model_name: ModelName = default_model,
    token: str = default_token,
    use_cpu: bool = False,
    limit_tokens: int = -1,
) -> Callable[[List[LogData]], Generator[LogEmbeddingData, None, None]]:
    if embedder_type == "llama-13b":
        pooling_fn = get_pooling_operation(pooling_code=pooling_code)
        return lambda events: generate_llama2_embeddings(
            events,
            pooling_fn=pooling_fn,
            model_name=model_name,
            token=token,
            use_cpu=use_cpu,
            limit_tokens=limit_tokens,
        )
    elif embedder_type == "tfidf":
        return generate_tfidf_embeddings
    else:
        raise Exception


def get_default_embedder_cache(
    embedder_type: Literal["llama-13b", "tfidf"],
    pooling_code: PoolingOperationCode = "mean",
    limit_tokens: int = -1,
) -> Path:
    if embedder_type == "llama-13b":
        return (
            default_folder_data
            / f"embeddings_{embedder_type}__{pooling_code}__{limit_tokens}.hdf5"
        )
    elif embedder_type == "tfidf":
        return default_folder_data / "embeddings_tfidf.hdf5"
    else:
        raise Exception


def get_emb_dist_matrix_fn(emb_dist_type: Literal["cosine", "euclidean"]) -> Callable:
    return lambda data: get_distance_matrix(data, metric=emb_dist_type)


def get_default_emb_dist_matrix_cache(
    embedder_type: Literal["llama-13b", "tfidf"],
    emb_dist_type: Literal["cosine", "euclidean"] = "cosine",
    pooling_code: PoolingOperationCode = "mean",
    limit_tokens: int = -1,
) -> Path:
    if embedder_type == "tfidf":
        return (
            default_folder_data
            / f"emb_distances__{embedder_type}__emb_dist_type-{emb_dist_type}.hdf5"
        )
    elif embedder_type == "llama-13b":
        return (
            default_folder_data
            / f"emb_distances__{embedder_type}__emb_dist_type-{emb_dist_type}__{pooling_code}__{limit_tokens}.hdf5"
        )


def get_default_split():
    return default_folder_data / "splitted_event_ids.json"


def get_tokenizer(token: str, model_name: str) -> "LlamaTokenizer":
    huggingface_hub.login(token=token)
    tokenizer: "LlamaTokenizer" = trf.AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir="/project/def-aloise/rmoine/cache_dir",
        token=token,
        trust_remote_code=True,
    )  # type: ignore
    return tokenizer


def initialize_model(
    model_name: str,
    token: str,
    hidden_states: bool = False,
    return_dict: bool = False,
    base_class: Any = trf.AutoModelForCausalLM,
    num_labels: int = 1,
    quant: bool = True,
    load_in_8bit: bool = True,
    use_cpu: bool = False,
    *args,
    **kwargs,
) -> "trf.LlamaForCausalLM":
    if hidden_states:
        last_state = True
    huggingface_hub.login(token=token)
    double_quant_config = None
    if quant and not use_cpu:
        double_quant_config = trf.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    model = base_class.from_pretrained(
        model_name,
        quantization_config=double_quant_config,
        return_dict=return_dict,
        output_hidden_states=hidden_states,
        cache_dir="/project/def-aloise/rmoine/cache_dir",
        token=token,
        num_labels=num_labels,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16 if use_cpu else None,
    )
    print(model)
    model.config.use_cache = False
    return model


def initialize_model_inference(
    model_name: str,
    token: str,
    return_model: bool = True,
    hidden_states: bool = False,
    return_dict: bool = False,
    num_labels: int = 1,
    quant: bool = True,
) -> Union[Tuple[LlamaTokenizer, "trf.LlamaForCausalLM"], LlamaTokenizer]:
    huggingface_hub.login(token=token)
    tokenizer = get_tokenizer(token=token, model_name=model_name)
    if return_model:
        model = initialize_model(
            model_name,
            token,
            hidden_states=hidden_states,
            return_dict=return_dict,
            base_class=trf.AutoModelForCausalLM,
            num_labels=num_labels,
            quant=quant,
        )
        return tokenizer, model
    else:
        return tokenizer


def get_pooling_operation(pooling_code: "PoolingOperationCode") -> "PoolingFn":
    if pooling_code == "mean":
        return lambda embedding: torch.mean(embedding, dim=0)
    elif pooling_code == "sum":
        return lambda embedding: torch.sum(embedding, dim=0)
    else:
        raise ValueError(f"{pooling_code=} is not possible")


def generate_seeds(
    n_data: Optional[int] = None,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
) -> Tuple[int, int]:
    assert (seed_start is None and seed_end is None) == (
        n_chunks is not None and interval_idx is not None and n_data is not None
    ), "Expecting either seed_start and seed_end or n_chunks and interval_idx"
    if seed_start is not None and seed_end is not None:
        if seed_end == -1:
            assert n_data is not None
            seed_end = n_data
        return (seed_start, seed_end)
    if n_chunks is not None and interval_idx is not None and n_data is not None:
        n_intervals = n_chunks
        intervals = [
            [i * (n_data // n_intervals), (i + 1) * (n_data // n_intervals)]
            for i in range(n_intervals)
        ]
        intervals[-1][1] = n_data
        [seed_start, seed_end] = intervals[interval_idx]
        return (seed_start, seed_end)
    raise Exception


def generate_llama2_embeddings(
    events: Iterable[LogData],
    pooling_fn: PoolingFn,
    model_name: ModelName = default_model,  # type: ignore
    token: str = default_token,
    use_cpu: bool = False,
    limit_tokens: int = -1,
) -> Generator[LogEmbeddingData, None, None]:
    tokenizer, model = initialize_model_inference(model_name, token, hidden_states=False, return_dict=True, quant=not use_cpu)  # type: ignore
    model.eval()
    for event in events:
        tokenized_full_text = tokenizer.encode(event["text"])
        limit_tokens_sample = limit_tokens
        if limit_tokens == -1:
            limit_tokens_sample = len(tokenized_full_text)
        tokenized_full_text = tokenized_full_text[:limit_tokens_sample]
        text = tokenizer.decode(tokenized_full_text)
        input_tensor = torch.tensor([tokenized_full_text], dtype=torch.int32)
        with torch.no_grad():
            try:
                embeddings = model(input_tensor)  # type: ignore
                embedding = embeddings.logits[0]
                embedding = np.array(pooling_fn(embedding).tolist(), dtype=np.float32)
                yield LogEmbeddingData(
                    event_id=event["event_id"], text=text, embedding=embedding
                )
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()  # type: ignore


"""End of inspired from https://github.com/AndressaStefany/severityPrediction"""


def generate_tfidf_embeddings(
    events: Iterable[LogData],
) -> Generator[LogEmbeddingData, None, None]:
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform([e["text"] for e in events])
    embeddings = np.asarray(embeddings.todense())  # type: ignore
    for embedding, event in zip(embeddings, events):
        yield LogEmbeddingData(
            event_id=event["event_id"], text=event["text"], embedding=embedding
        )


@print_args
def extract_logs_from_file(
    src_file: Optional[Path] = None, out_file: Optional[Path] = None
):
    """Extract the data from the source file for the embedding formatting part

    # Arguments:
    - src_file: Path, the path to the file containing the raw logs
    - out_file: Path, the path to the json file containing the data to process with text containing the input text to generate the embeddings from
    """
    if src_file is None:
        src_file = Path("./logs/trat3_production_1650_1700_20231411.json")
    if out_file is None:
        out_file = Path("./logs/trat3_production_1650_1700_20231411_v1.hdf5")
    with open(src_file) as fp:
        data_in = json.load(fp)
    with h5py.File(out_file, "w") as f:
        for planid, dico1 in tqdm.tqdm(data_in.items(), total=len(data_in)):
            for log_name, dico2 in dico1.items():
                for event in dico2["events"]:
                    event["log_name"] = log_name
                    event["planid"] = planid
                    event["text"] = event["raw"]
                    dataset = f.create_dataset(
                        name=str(event["event_id"]), shape=(0,), dtype="f"
                    )
                    for k, v in event.items():
                        dataset.attrs[k] = v


@print_args
def split_build_logs(path_in: Optional[Path] = None, path_out: Optional[Path] = None):
    """Split along the build logs"""
    if path_in is None:
        path_in = (
            default_folder_data
            / "input"
            / "trat3_production_1650_1700_20231411_v1.hdf5"
        )
    if path_out is None:
        path_out = default_folder_data / "splitted_event_ids.json"
    path_in = existing_path(path_in, is_folder=False)
    data = {}
    with h5py.File(path_in, "r") as fp:
        keys = list(fp)
        for k in tqdm.tqdm(keys):
            build_log_name = get_build_log_name({**fp[k].attrs})  # type: ignore
            if build_log_name not in data:
                data[build_log_name] = []
            data[build_log_name].append(k)
    with open(path_out, "w") as fp:
        json.dump(data, fp, indent=2)


def get_split_build_logs(path_split: Optional[Path] = None) -> Dict[str, List[str]]:
    if path_split is None:
        path_split = default_folder_data / "splitted_event_ids.json"
    with open(path_split) as fp:
        split = json.load(fp)
    return split


def log_json_to_dataframe(
    log_data: List[str], regex: re.Pattern[str], headers: List[str]
) -> pd.DataFrame:
    """Transforms log file to dataframe adding the line number of each line

    # Arguments
    - log_data: List[ParserInputDict], lines in order inside the same build log
    - regex: re.Pattern, the pattern to match all headers together
    - headers: List[str], the headers names in the regex named fields to extract: ex: ?P<fieldDate>.*?

    # Returns:
    - pd.DataFrame, dataframe with each header in a column plus the 'LineId'

    Disclaimer: heavily inspired from drain_func.log_to_dataframe
    """
    log_messages = []
    linecount = 0
    for i, (line) in enumerate(log_data):
        with contextlib.suppress(Exception):
            match = regex.search(line.strip())
            message = [match.group(header) for header in headers]  # type: ignore
            log_messages.append((i, *message))
            linecount += 1
    logdf = pd.DataFrame(log_messages, columns=("OriginalLine", *headers))
    logdf.insert(0, "LineId", None)
    logdf["LineId"] = [i + 1 for i in range(linecount)]
    return logdf


class JSONLogParser(DrainMethod.LogParser):
    """Parse JSON data provided in argument. Heavily inspired bu DrainMethod.LogParser

    # Arguments
        - depth: int = 4, depth of all leaf nodes
        - similarity_threshold: float = 0.4, similarity threshold
        - max_children: int = 3, max number of children of an internal node
        - regexes_preprocess: Optional[List] = None, regular expressions used in preprocessing (step1)
    """

    def __init__(
        self,
        *,
        depth: int = 4,
        similarity_threshold: float = 0.4,
        max_children: int = 3,
        regexes_preprocess: Optional[List] = None,
    ):
        if regexes_preprocess is None:
            regexes_preprocess = []
        # Note: as some attribute are used only in one function we pass them as arguments
        super().__init__(
            log_format=None,  # type: ignore
            indir=None,  # type: ignore
            outdir=None,  # type: ignore
            depth=depth,
            st=similarity_threshold,
            maxChild=max_children,
            rex=regexes_preprocess,
            keep_para=None,  # type: ignore
        )

    def outputResult(
        self,
        logClustL: List[DrainMethod.Logcluster],
        df_log: pd.DataFrame,
        keep_parameters: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Method to output the results

        # Arguments:
        - logClustL: List[DrainMethod.Logcluster], the log cluster returned by the parse method
        - df_log: pd.DataFrame, the initial df_log with "Content"
        - keep_parameters: bool = True, Choose wether if you want to keep the variables of the template (<*> values for each text)

        # Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]
            - df_log: pd.DataFrame, the initial df_log DataFrame with added columns 'EventId' (the template id (md5 hash) of the event) 'EventTemplate' (the template in itself with <*>), 'ParameterList' (if keep_parameters = True, the parameters in the template)
            - df_event: pd.DataFrame, events found in the logs with columns 'EventId' (str, hash of the template), 'EventTemplate' (str, the template), 'Occurrences' (for each template the number of time it is seen over the full log file)
        """
        log_templates: List[str] = [""] * df_log.shape[0]
        log_templateids: List[str] = [""] * df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = " ".join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(
            df_events, columns=["EventId", "EventTemplate", "Occurrences"]
        )
        df_log["EventId"] = log_templateids
        df_log["EventTemplate"] = log_templates

        if keep_parameters:
            df_log["ParameterList"] = df_log.apply(self.get_parameter_list, axis=1)

        occ_dict = dict(df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()
        df_event["EventTemplate"] = df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_log = df_log.rename(
            {"EventTemplate": "template", "ParameterList": "variables"}
        )
        return df_log, df_event

    def parse(self, log_df: pd.DataFrame) -> List[DrainMethod.Logcluster]:
        start_time = datetime.datetime.now()
        rootNode = DrainMethod.Node()
        logCluL = []

        for _, line in tqdm.tqdm(
            log_df.iterrows(), desc="Parsing Progress", total=len(log_df)
        ):
            logID = line["LineId"]

            ## Tokenization by splits
            logmessageL = self.preprocess(line["Content"]).strip().split()

            matchCluster = self.treeSearch(rootNode, logmessageL)

            ## Match no existing log cluster
            if matchCluster is None:
                newCluster = DrainMethod.Logcluster(
                    logTemplate=logmessageL, logIDL=[logID]
                )
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            ## Adds the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

        print(
            "Parsing done. [Time taken: {!s}]".format(
                datetime.datetime.now() - start_time
            )
        )
        return logCluL


def run_chunk_operation(
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
    split_file: str = "splitted_event_ids",
) -> Generator[Tuple[str, List[str]], None, None]:
    split = get_split_build_logs(default_folder_data / f"{split_file}.json")
    build_logs_names = list(split)
    seed_start, seed_end = generate_seeds(
        n_data=len(split), n_chunks=n_chunks, interval_idx=interval_idx
    )
    build_logs_names = build_logs_names[seed_start:seed_end]
    for build_log_name in tqdm.tqdm(build_logs_names):
        yield build_log_name, split[build_log_name]


@print_args
def run_parser(
    parser_type: Literal["ciena", "drain"],
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
):
    parser = get_parser_cache_fn(
        parser_type=parser_type,
        depth=depth,
        similarity_threshold=similarity_threshold,
        max_children=max_children,
    )
    parser_cache = get_default_parser_cache(
        parser_type, depth, similarity_threshold, max_children
    )
    with h5py.File(
        parser_cache.parent / f"{parser_cache.stem}_{interval_idx}.hdf5", "a"
    ) as fp_cache:
        event_ids_in_cache = set(list(fp_cache))
        for build_log_name, events_ids in run_chunk_operation(
            n_chunks=n_chunks, interval_idx=interval_idx
        ):
            all_seen = len(set(events_ids).difference(event_ids_in_cache)) == 0
            if all_seen:
                continue
            # If not already in cache get the texts
            events_list = []
            with h5py.File(default_event_hdf5_path, "r") as fp_in:
                for event_id in events_ids:
                    dico = {k: v for k, v in fp_in[event_id].attrs.items()}
                    dico["event_id"] = event_id
                    events_list.append(dico)
            parsing_output = parser(events_list)
            for event_parsed in parsing_output:
                dataset = fp_cache.create_dataset(
                    name=str(event_parsed["event_id"]), shape=(0,), dtype="f"
                )
                for k, v in event_parsed.items():
                    dataset.attrs[k] = v
                dataset.attrs["build_log_name"] = build_log_name


@print_args
def run_variable_matrix(
    parser_type: Literal["ciena", "drain"],
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
    variable_field: Literal["ParameterList", "variables"] = "ParameterList",
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
):
    var_dist_matrix_cache = get_default_var_dist_matrix_cache(
        parser_type, depth, similarity_threshold, max_children
    )
    with h5py.File(
        var_dist_matrix_cache.parent
        / f"{var_dist_matrix_cache.stem}_{interval_idx}.hdf5",
        "a",
    ) as fp_cache_out:
        with h5py.File(
            get_default_parser_cache(
                parser_type, depth, similarity_threshold, max_children
            ),
            "r",
        ) as fp_cache_in:
            event_ids_in_cache = set(list(fp_cache_in))
            for build_log_name, events_ids in run_chunk_operation(
                n_chunks=n_chunks, interval_idx=interval_idx
            ):
                all_seen = len(set(events_ids).difference(event_ids_in_cache)) == 0
                assert (
                    all_seen
                ), f"Expecting to have already computed all event templates and variables but is missing events {set(events_ids).difference(event_ids_in_cache)}"
                # Get the variables for each event
                if build_log_name in fp_cache_out:
                    continue
                try:
                    events: List[ParsingOutput] = [
                        {
                            k1: v1 if not isinstance(v1, np.ndarray) else v1.tolist()
                            for k1, v1 in fp_cache_in[k].attrs.items()
                        }
                        for k in events_ids
                    ]  # type: ignore
                    matrix = get_variable_matrix(events, variable_field=variable_field)
                    distance_matrix = get_distance_matrix(
                        LogsEmbeddingData(
                            texts=[], event_ids=events_ids, embeddings=matrix
                        ),
                        metric="jaccard",
                    )
                    fp_cache_out.create_dataset(
                        build_log_name, dtype="f", data=distance_matrix
                    )
                except NoVariable as e:
                    # Create empty matrix
                    fp_cache_out.create_dataset(build_log_name, shape=(0,), dtype="f")
                except KeyError as e:
                    fp_cache_out.create_dataset(build_log_name, shape=(0,), dtype="f")


@print_args
def run_embeddings(
    embedder_type: Literal["llama-13b", "tfidf"] = "llama-13b",
    pooling_code: PoolingOperationCode = "mean",
    token: str = default_token,
    use_cpu: bool = False,
    limit_tokens: int = -1,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
    split_file: str = "splitted_event_ids",
    id: str = "",
    input_event_file: Optional[Path] = None,
):  # TODO review in/out path with custom dataset
    if input_event_file is None:
        input_event_file = default_event_hdf5_path
    input_event_file = existing_path(input_event_file, is_folder=False)
    embedder_fn = get_embedder_fn(
        embedder_type=embedder_type,
        pooling_code=pooling_code,
        token=token,
        use_cpu=use_cpu,
        limit_tokens=limit_tokens,
    )

    embeddings_chunk_path = get_default_embedder_cache(
        embedder_type, pooling_code, limit_tokens
    )
    with h5py.File(
        embeddings_chunk_path.parent
        / f"{embeddings_chunk_path.stem}{id}_{interval_idx}.hdf5",
        "w",
    ) as fp_cache_out:
        with h5py.File(input_event_file, "r") as fp_cache_in:
            for build_log_name, events_ids in run_chunk_operation(
                n_chunks=n_chunks, interval_idx=interval_idx, split_file=split_file
            ):
                # Get the variables for each event
                events: List[LogData] = [{k: v for k, v in fp_cache_in[k].attrs.items()} for k in events_ids]  # type: ignore
                for event in embedder_fn(events):
                    if event["event_id"] in fp_cache_out:
                        continue
                    fp_cache_out.create_dataset(
                        event["event_id"], data=event["embedding"]
                    )


@print_args
def run_embeddings_distances(
    embedder_type: Literal["llama-13b", "tfidf"] = "llama-13b",
    pooling_code: PoolingOperationCode = "mean",
    limit_tokens: int = -1,
    emb_dist_type: Literal["cosine", "euclidean"] = "cosine",
    path_in: Optional[Path] = None,
    path_out: Optional[Path] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
    split_file: str = "splitted_event_ids",
):
    embedder_fn = get_emb_dist_matrix_fn(emb_dist_type)
    if path_out is None:
        path_out = get_default_emb_dist_matrix_cache(
            embedder_type=embedder_type,
            emb_dist_type=emb_dist_type,
            pooling_code=pooling_code,
            limit_tokens=limit_tokens,
        )
    path_out = Path(path_out)
    if path_in is None:
        path_in = get_default_embedder_cache(
            embedder_type, pooling_code=pooling_code, limit_tokens=limit_tokens
        )
    path_in = Path(path_in)
    logger.info(locals())
    with h5py.File(
        path_out.parent / f"{path_out.stem}_{interval_idx}.hdf5",
        "a",
    ) as fp_cache_out:
        with h5py.File(
            path_in,
            "r",
        ) as fp_cache_in:
            for build_log_name, events_ids in run_chunk_operation(
                n_chunks=n_chunks, interval_idx=interval_idx, split_file=split_file
            ):
                if build_log_name in fp_cache_out:
                    continue
                # Get the embeddings for each event
                embeddings = np.array([fp_cache_in[k] for k in events_ids])
                embeddings = LogsEmbeddingData(
                    event_ids=[k for k in events_ids],
                    texts=["" for _ in events_ids],
                    embeddings=embeddings,
                )
                matrix = embedder_fn(embeddings)
                fp_cache_out.create_dataset(build_log_name, data=matrix)


def get_groups(
    input_file: Optional[Path] = None,
    build_log_name: Optional[str] = None,
    split_file: Optional[Path] = None,
    event_ids: Optional[List[str]] = None,
) -> List[GroupDict]:
    """Get the groups for a given list of event_ids or for a given build_log_name

    # Parameters
    - input_file: Optional[Path] = None, the input hdf5 file where for each event id (the string key) you have an empty dataset with especially the attribute group_id
    And then either provide
    - build_log_name: Optional[str] = None, the build log name for which get the groups of each event_id inside
    - split_file: Optional[Path] = None, a json file in the format {"build_log_name1": [event_id_1,event_id_2,...], "build_log_name2":[...],...}
    Or provide
    - event_ids: Optional[List[str]] = None, a list of events ids to get the group of

    # Returns
    List[Dict[str,str]], For each event_id the group_id associated (str)
    [
        {"event_id":event_id, "group_id":group_id},
        ...
    ]
    """
    assert (event_ids is None) != (
        split_file is None and build_log_name is None
    ), "Expecting provided either event_ids to get the group from, xor the split_file and the build_log_name to get the events_ids to analyse from"
    if input_file is None:
        input_file = default_folder_data / "trat3_production_1650_1700_20231411_v1.hdf5"
    if event_ids is None:
        if split_file is None:
            split_file = default_folder_data / "splitted_event_ids.json"
        with open(split_file) as fp:
            split = json.load(fp)
        event_ids = split[build_log_name]
    assert event_ids is not None
    L = []
    with h5py.File(input_file, "r") as fp_in:
        for event_id in tqdm.tqdm(event_ids, total=len(event_ids)):
            group_id: Optional[str] = fp_in[event_id].attrs["group_id"]
            L.append({"event_id": event_id, "group_id": group_id})
    return L


def get_parsing_ciena(
    events: List[CienaLogData],
) -> List[ParsingOutput]:
    return events  # type: ignore


def get_parsing_drainparser(
    events: List[LogData],
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
) -> List[ParsingOutput]:
    log_format = "<Content>"  # Format of the file, if there are different fields
    # load the data
    headers, regex = drain_func.generate_logformat_regex(log_format)
    log_df = log_json_to_dataframe([e["text"] for e in events], regex=regex, headers=headers)  # type: ignore
    log_df["event_id"] = [e["event_id"] for e in events]
    regex = []  # Regex strings for Drain execution
    log_df = drain_func.preprocess_df(log_df, regex=regex)
    parser = JSONLogParser(
        depth=depth,
        similarity_threshold=similarity_threshold,
        max_children=max_children,
        regexes_preprocess=None,
    )
    logCluL = parser.parse(log_df=log_df)
    log_df, df_event = parser.outputResult(
        logClustL=logCluL, df_log=log_df, keep_parameters=True
    )
    templates_variables = (
        log_df[["event_id", "ParameterList", "EventTemplate"]]
        .rename({"ParameterList": "variables", "EventTemplate": "template"})
        .to_dict(orient="records")
    )
    return templates_variables  # type: ignore


def get_variable_matrix(
    parsed_events: List[ParsingOutput],
    variable_field: Literal["ParameterList", "variables"] = "ParameterList",
) -> np.ndarray:
    binarizer = skPrepro.MultiLabelBinarizer(sparse_output=False)
    matrix_variables = binarizer.fit_transform(
        (e[variable_field] for e in parsed_events)
    )
    if matrix_variables.shape[0] == 0:
        raise EmptyLog("No logs in the logs provided", logs=parsed_events)  # type: ignore
    if matrix_variables.shape[1] == 0:
        raise NoVariable("No variables in the logs provided", logs=parsed_events)  # type: ignore
    return matrix_variables.astype(bool)  # type: ignore


def get_distance_matrix(
    log_emeddings_data: LogsEmbeddingData,
    metric: Literal["jaccard", "cosine", "euclidean"],
) -> np.ndarray:
    return skMetrics.pairwise_distances(log_emeddings_data["embeddings"], metric=metric)


def get_count_distance_matrix(
    events: List[LogData],
    count_matrix_mode: Literal["absolute", "from_file"] = "absolute",
) -> np.ndarray:
    if count_matrix_mode == "absolute":
        n = len(events)
        matrix = np.array(
            [[abs(i - j) for j in range(n)] for i in range(n)], dtype=np.float32
        )
    elif count_matrix_mode == "from_file":
        line_numbers = np.array([int(e["line_num"]) for e in events])
        matrix = np.abs(line_numbers[:, np.newaxis] - line_numbers)
    else:
        raise KeyError(f"{count_matrix_mode=}")
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))  # type: ignore
    return matrix


def get_triplet_matrix(
    build_log_name: str,
    embedder_type: Literal["llama-13b", "tfidf"] = "llama-13b",
    pooling_code: PoolingOperationCode = "mean",
    limit_tokens: int = -1,
    emb_dist_type: Literal["cosine", "euclidean"] = "cosine",
    parser_type: Literal["ciena", "drain"] = "drain",
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
    count_matrix_mode: Literal["absolute", "from_file"] = "absolute",
) -> TripletMatrix:
    # get matrix variables
    path_matrix_variables = get_default_var_dist_matrix_cache(
        parser_type=parser_type,
        depth=depth,
        similarity_threshold=similarity_threshold,
        max_children=max_children,
    )
    with h5py.File(path_matrix_variables, "r") as fp:
        variables_matrix = np.array(fp[build_log_name])
    ## No need to normalize as the jaccard distance is already between 0 and 1
    # get matrix embeddings distances
    path_embedding = get_default_emb_dist_matrix_cache(
        embedder_type=embedder_type,
        pooling_code=pooling_code,
        limit_tokens=limit_tokens,
        emb_dist_type=emb_dist_type,
    )
    with h5py.File(path_embedding, "r") as fp:
        logger.info(f"Before embedding {path_embedding.stem}")
        embeddings_matrix = np.array(fp[build_log_name])
    logger.info(f"After embedding {path_embedding.stem}")
    ## Normalize
    if emb_dist_type == "cosine":
        # as cosine distance is between 0 and 2
        embeddings_matrix /= 2
    elif emb_dist_type == "euclidean":
        # apply a sigmoid
        embeddings_matrix = 1 / (1 + np.exp(-embeddings_matrix))
    else:
        embeddings_matrix = (embeddings_matrix - np.min(embeddings_matrix)) / (
            np.max(embeddings_matrix) - np.min(embeddings_matrix)
        )
    # get matrix count distance
    with open(get_default_split(), "r") as fp:
        split = json.load(fp)
    with h5py.File(
        default_folder_data / "trat3_production_1650_1700_20231411_v1.hdf5"
    ) as fp:
        events: List[LogData] = [{**fp[k].attrs} for k in split[build_log_name]]  # type: ignore
        count_matrix = get_count_distance_matrix(
            events=events, count_matrix_mode=count_matrix_mode
        )
    count_matrix = (count_matrix - np.min(count_matrix)) / (
        np.max(count_matrix) - np.min(count_matrix)
    )
    return {
        "variables_matrix": variables_matrix,
        "embeddings_matrix": embeddings_matrix,
        "count_matrix": count_matrix,
    }


def combine_matrices(
    triplet_matrix: TripletMatrix, triplet_coef: TripletCoef, **kwargs
) -> np.ndarray:
    if triplet_coef["coef_variables_matrix"] == 0:
        combined_distance_matrix = triplet_matrix["embeddings_matrix"] * triplet_coef["coef_embeddings_matrix"] + triplet_matrix["count_matrix"] * triplet_coef["coef_count_matrix"]  # type: ignore
    else:
        combined_distance_matrix = triplet_matrix["embeddings_matrix"] * triplet_coef["coef_embeddings_matrix"] + triplet_matrix["variables_matrix"] * triplet_coef["coef_variables_matrix"] + triplet_matrix["count_matrix"] * triplet_coef["coef_count_matrix"]  # type: ignore
    return combined_distance_matrix


def get_clustering_function(
    clustering_algorithm: Literal["dbscan", "kmedoids"],
    epsilon: Optional[float] = None,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    iteration_max: Optional[int] = None,
    number_of_clusters: Optional[int] = None,
):
    func = None
    if clustering_algorithm == "dbscan":
        assert epsilon is not None
        func = ft.partial(clustering_dbscan, epsilon=epsilon)
    elif clustering_algorithm == "kmedoids":
        assert (
            must_link is not None
            and cannot_link is not None
            and iteration_max is not None
            and number_of_clusters is not None
        )
        func = ft.partial(
            clustering_kmedoids,
            must_link=must_link,
            cannot_link=cannot_link,
            iteration_max=iteration_max,
            number_of_clusters=number_of_clusters,
        )
    else:
        raise Exception
    return func


def clustering_dbscan(
    combined_matrix: np.ndarray,
    *,
    epsilon: float,
    **kwargs,
) -> Dict[str, Dict[int, int]]:
    clusterer = DBSCAN(
        eps=epsilon, min_samples=2, metric="precomputed", algorithm="auto", n_jobs=-1
    )
    clusterer.fit(combined_matrix)
    labels = {i: v for i, v in enumerate(clusterer.labels_)}
    return {"type": "dbscan", "clustering": labels}


def best_clustering_kmedoid(
    combined_matrix: np.ndarray,
    *,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    iteration_max: Optional[int] = None,
    seed: int = 0,
    n_samples: int = 10,
    **kwargs,
):
    gc.collect()
    if len(combined_matrix) == 0:
        return {
            "type": "kmedoids",
            "score": 1,
            "clustering": {},
            "number_of_clusters": -1,
        }
    if len(combined_matrix) == 1:
        return {"score": 1, "clustering": {0: 0}, "number_of_clusters": -1}
    # done because silhouette score do not manage the case of 2 lines: in this case 1 custer per line is chosen
    if len(combined_matrix) == 2:
        return {"score": 1, "clustering": {0: 0, 1: 1}, "number_of_clusters": -1}
    if must_link is None:
        must_link = []
    if cannot_link is None:
        cannot_link = []
    if iteration_max is None:
        iteration_max = -1
    best = {"score": -float("inf"), "clustering": {}, "number_of_clusters": -1}
    n_samples = min(n_samples, len(combined_matrix) - 1 - 2 + 1)
    for k in set(
        np.round(np.linspace(2, len(combined_matrix) - 1, n_samples)).astype(int)
    ):
        clustering = clustering_kmedoids(
            combined_matrix,
            must_link=must_link,
            cannot_link=cannot_link,
            iteration_max=iteration_max,
            seed=seed,
            number_of_clusters=k,
        )
        clusters = np.unique(list(clustering.values()))
        n_clusters = len(clusters)
        assert n_clusters == k, f"expecting {k} clusters not {n_clusters}"
        if n_clusters < 2 or n_clusters > len(combined_matrix) - 1:
            continue
        assert (
            min(clusters) >= 0
        ), f"Expecting {k} positive clusters, found {len(np.unique(list(clustering.values())))}: {(np.unique(list(clustering.values())))}"
        assert max(clusters) < len(
            combined_matrix
        ), f"Expecting {k} clusters lower than {len(combined_matrix)}, found {len(np.unique(list(clustering.values())))}: {(np.unique(list(clustering.values())))}"

        score = skMetrics.silhouette_score(
            X=combined_matrix,
            labels=[clustering[i] for i in range(len(combined_matrix))],
            metric="precomputed",
        )
        if score > best["score"]:
            best = {"score": score, "clustering": clustering, "number_of_clusters": k}
    return best


def clustering_kmedoids(
    combined_matrix: np.ndarray,
    *,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    iteration_max: Optional[int] = None,
    number_of_clusters: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> Dict[int, int]:
    if must_link is None:
        must_link = []
    if cannot_link is None:
        cannot_link = []
    if iteration_max is None:
        iteration_max = -1
    if number_of_clusters == 1:
        return {i: 0 for i, c in enumerate(range(len(combined_matrix)))}
    if number_of_clusters == len(combined_matrix):
        return {i: c for i, c in enumerate(range(len(combined_matrix)))}
    assert (
        must_link is not None
        and cannot_link is not None
        and iteration_max is not None
        and number_of_clusters is not None
    )
    # Load the shared library
    n_points, n_dims = combined_matrix.shape
    combined_matrix = combined_matrix.flatten()
    # logger.info(f"Before load c++ lib")
    path_lib = Path("./clustering_lib.so").resolve().as_posix()
    dummy_cpp_library = ctypes.CDLL(path_lib)
    dummy_cpp_library.clusterize.restype = ctypes.POINTER(ctypes.c_int)
    combine_matrix_ptr = combined_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    must_link_array = np.array(must_link).flatten().astype(np.int32)
    must_link_array_ptr = must_link_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    n_must_link = len(must_link)
    cannot_link_array = np.array(cannot_link).flatten().astype(np.int32)
    cannot_link_array_ptr = cannot_link_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int)
    )
    n_cannot_link = len(cannot_link)
    logger.info(f"{combined_matrix.shape=}")
    data_ptr = dummy_cpp_library.clusterize(
        seed,
        combine_matrix_ptr,
        int(n_points),
        int(n_dims),
        int(number_of_clusters),
        must_link_array_ptr,
        int(n_must_link),
        cannot_link_array_ptr,
        n_cannot_link,
        iteration_max,
        -1,
        True,
    )
    data = np.copy(np.ctypeslib.as_array(data_ptr, shape=(n_points,)))
    with contextlib.suppress(Exception):
        dummy_cpp_library.free_array(data_ptr)

    return {i: c for i, c in enumerate(data)}


def test_running_kmedoids():
    print("Start")
    points = np.random.rand(10, 1000)
    combined_matrix = skMetr.pairwise_distances(points, metric="euclidean")
    number_of_clusters = 2
    clustering_kmedoids(
        combined_matrix,
        points=combined_matrix,
        must_link=[(0, 1)],
        cannot_link=[(3, 4)],
        iteration_max=10000,
        number_of_clusters=number_of_clusters,
    )


def labels_to_groups(
    build_log_name: str, group_mapping: Dict[int, int]
) -> Dict[str, int]:
    path_lines = get_default_split()
    with open(path_lines) as fp:
        mapping = json.load(fp)
    event_ids = [e["event_id"] for e in mapping[build_log_name]]
    event_id_group = {}
    for i, e in enumerate(event_ids):
        event_id_group[e] = group_mapping[i]
    return event_id_group


def run_main_clustering(
    build_log_name: str,
    clustering_algorithm: Literal["dbscan", "kmedoids"] = "dbscan",
    parser_type: Literal["ciena", "drain"] = "drain",
    embedder_type: Literal["llama-13b", "tfidf"] = "tfidf",
    emb_dist_type: Literal["cosine", "euclidean"] = "cosine",
    pooling_code: Literal["mean", "sum"] = "mean",
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
    limit_tokens: int = 1000,
    count_matrix_mode: Literal["absolute", "from_file"] = "absolute",
    n_elems_grid: int = 10,
    n_elems_grid_coef: int = 5,
) -> List[ClusteringOutput]:
    # get lines
    Llines = []
    split = get_split_build_logs()
    with h5py.File(default_event_hdf5_path, "r") as fp:
        for event_id in split[build_log_name]:
            Llines.append({**fp[event_id].attrs})
    if len(Llines) == 0:
        return []
    elif len(Llines) == 1:
        return [ClusteringOutput(hyperparameters_chosen={}, clusters=[0], texts=Llines)]
    # get triplet matrix

    triplet_matrix = get_triplet_matrix(
        build_log_name,
        embedder_type=embedder_type,
        pooling_code=pooling_code,
        limit_tokens=limit_tokens,
        emb_dist_type=emb_dist_type,
        parser_type=parser_type,
        depth=depth,
        similarity_threshold=similarity_threshold,
        max_children=max_children,
        count_matrix_mode=count_matrix_mode,
    )
    # prepare coefficients of the matrices
    hyperparameters: Dict[str, List[Any]] = {
        "triplet_coef": [],
    }
    coef_possibilities = np.linspace(0.0, 1.0, n_elems_grid_coef)

    if triplet_matrix["variables_matrix"].shape == (0,):
        logger.info(
            f"Warning: variable matrix shape is {triplet_matrix['variables_matrix'].shape}"
        )
        hyperparameters["triplet_coef"] = [
            TripletCoef(
                coef_variables_matrix=0,
                coef_embeddings_matrix=coef2,
                coef_count_matrix=coef3,
            )
            for coef1, coef2, coef3 in it.product(coef_possibilities, repeat=3)
            if coef2 + coef3 == 1
        ]
    else:
        hyperparameters["triplet_coef"] = [
            TripletCoef(
                coef_variables_matrix=coef1,
                coef_embeddings_matrix=coef2,
                coef_count_matrix=coef3,
            )
            for coef1, coef2, coef3 in it.product(coef_possibilities, repeat=3)
            if coef1 + coef2 + coef3 == 1
        ]
    # prepare clustering algorithm
    clustering_function = None
    if clustering_algorithm == "dbscan":
        clustering_function = clustering_dbscan
        hyperparameters["epsilon"] = np.logspace(
            start=-3, stop=1, num=n_elems_grid, base=10
        ).tolist()
    elif clustering_algorithm == "kmedoids":
        clustering_function = best_clustering_kmedoid
    else:
        raise NotImplemented(f"{clustering_algorithm} is not supported")
    Lclusters: List[ClusteringOutput] = []
    for hyperparameters_chosen in tqdm.tqdm(generate_combinations(hyperparameters)):
        combined_matrix = combine_matrices(triplet_matrix, **hyperparameters_chosen)
        assert not np.isnan(combined_matrix).any(), f"Nan matrix  for {build_log_name}"
        clusters_dict = clustering_function(combined_matrix, **hyperparameters_chosen)
        clusters = clusters_dict["clustering"]
        others = {k: v for k, v in clusters_dict.items() if k != "clustering"}
        assert len(clusters) == len(Llines)
        Lclusters.append(
            {  # type: ignore
                "hyperparameters_chosen": hyperparameters_chosen,
                "clusters": [clusters[i] for i in sorted(clusters)],
                "texts": Llines,
                **others,
            }
        )

    return Lclusters


def get_run_id(**kwargs) -> str:
    return "__".join([str(v) for k, v in sorted(kwargs.items(), key=lambda x: x[0])])


@print_args
def run_generate_clusters_all_build_logs(
    clustering_algorithm: Literal["dbscan", "kmedoids"] = "dbscan",
    parser_type: Literal["ciena", "drain"] = "drain",
    embedder_type: Literal["llama-13b", "tfidf"] = "tfidf",
    emb_dist_type: Literal["cosine", "euclidean"] = "cosine",
    pooling_code: Literal["mean", "sum"] = "mean",
    depth: int = 5,
    similarity_threshold: float = 0.4,
    max_children: int = 3,
    limit_tokens: int = 1000,
    count_matrix_mode: Literal["absolute", "from_file"] = "absolute",
    n_elems_grid: int = 3,
    n_elems_grid_coef: int = 5,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
):
    id = get_run_id(
        clustering_algorithm=clustering_algorithm,
        parser_type=parser_type,
        embedder_type=embedder_type,
        emb_dist_type=emb_dist_type,
        pooling_code=pooling_code,
        depth=depth,
        similarity_threshold=similarity_threshold,
        max_children=max_children,
        limit_tokens=limit_tokens,
        count_matrix_mode=count_matrix_mode,
        n_elems_grid=n_elems_grid,
        n_elems_grid_coef=n_elems_grid_coef,
    )
    with h5py.File(
        default_folder_data / f"clusters_build_logs_{id}_{interval_idx}.hdf5", "w"
    ) as f:
        chunck_data = list(
            run_chunk_operation(n_chunks=n_chunks, interval_idx=interval_idx)
        )
        for build_log_name, _ in tqdm.tqdm(chunck_data, position=0):
            Lclusters: List[ClusteringOutput] = run_main_clustering(
                build_log_name,
                clustering_algorithm,
                parser_type,
                embedder_type,
                emb_dist_type,
                pooling_code,
                depth,
                similarity_threshold,
                max_children,
                limit_tokens,
                count_matrix_mode,
                n_elems_grid,
                n_elems_grid_coef,
            )
            gpe = f.create_group(build_log_name)
            dataset_texts = gpe.create_dataset("texts", shape=(0,))
            dataset_texts.attrs["texts"] = [e["text"] for e in Lclusters[0]["texts"]]
            for i, clusters in enumerate(Lclusters):
                cluster = gpe.create_group(f"cluster_{i}")
                hyperparameters_chosen = cluster.create_dataset(
                    "hyperparameters_chosen", shape=(0,)
                )
                for k, v in clusters["hyperparameters_chosen"].items():
                    if isinstance(v, dict):
                        for k1, v1 in v.items():
                            hyperparameters_chosen.attrs[k1] = v1
                    else:
                        hyperparameters_chosen.attrs[k] = v
                cluster.create_dataset("clusters", data=clusters["clusters"])


@print_args
def run_merge_with_pattern(pattern: str):
    p = Path("./data/")
    elems = list(p.rglob(pattern))
    tot_written = 0
    path_out = elems[0].parent / f"{'_'.join(elems[0].stem.split('_')[:-1])}.hdf5"
    print(f"Writting in {path_out=}")
    with h5py.File(path_out, "a") as fp:
        for e in tqdm.tqdm(elems, position=0):
            with h5py.File(e, "r") as fp_in:
                for k, v in tqdm.tqdm(fp_in.items(), position=1, total=len(fp_in)):
                    if k in fp:
                        continue
                    try:
                        fp.create_dataset(k, data=np.copy(v))
                    except ValueError:
                        print(k, "already exists")

                    tot_written += 1
    print(f"{tot_written=}")


def copy_items(name, obj, dest_file):
    if isinstance(obj, h5py.Group):
        dest_group = create_or_override_group(dest_file, name)
        # You can add attributes to the new group if needed
        dest_group.attrs.update(obj.attrs)
    elif isinstance(obj, h5py.Dataset):
        create_or_override_dataset(dest_file, name, data=obj[:])
        # You can add attributes to the new dataset if needed
        dest_file[name].attrs.update(obj.attrs)


def create_or_override_group(h5file, group_name):
    """
    Create a new group or get a reference to an existing group in an HDF5 file.

    Parameters:
    - h5file: The h5py.File object representing the HDF5 file.
    - group_name: The name of the group.

    Returns:
    - group: The h5py.Group object.
    """
    if group_name in h5file:
        group = h5file[group_name]
    else:
        group = h5file.create_group(group_name)
    return group


def create_or_override_dataset(group, dataset_name, data):
    """
    Create a new dataset or get a reference to an existing dataset in a group.

    Parameters:
    - group: The h5py.Group object where the dataset should be created.
    - dataset_name: The name of the dataset.
    - data: The data to be stored in the dataset.

    Returns:
    - dataset: The h5py.Dataset object.
    """
    if dataset_name in group:
        dataset = group[dataset_name]
    else:
        dataset = group.create_dataset(dataset_name, data=data)
    return dataset


@print_args
def run_merge_nested(pattern: str):
    p = Path("./data/")
    elems = list(p.rglob(pattern))
    path_out = elems[0].parent / f"{'_'.join(elems[0].stem.split('_')[:-1])}.hdf5"
    print(f"Writting in {path_out=}")
    with h5py.File(path_out, "a") as fp_out:
        for e in tqdm.tqdm(elems, position=0):
            with h5py.File(e, "r") as fp_in:
                fp_in.visititems(lambda name, obj: copy_items(name, obj, fp_out))
            e.unlink(missing_ok=True)


def run_merge_json(pattern: str):
    p = Path("./data/")
    elems = list(p.rglob(pattern))
    path_out = elems[0].parent / f"{'_'.join(elems[0].stem.split('_')[:-1])}.hdf5"
    print(f"Writting in {path_out=}")
    with open(path_out, "w+") as fp_out:
        for e in tqdm.tqdm(elems, position=0):
            with open(e, "r") as fp_in:
                for l in fp_in.readlines():
                    fp_out.write(l.strip() + "\n")
            e.unlink(missing_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "i":
                return obj.tolist()
            elif obj.dtype.kind == "f":
                return obj.astype(float).tolist()
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def run_compare_to_ciena(
    path_hdf5: Path,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
):
    path_hdf5 = Path(path_hdf5)
    ciena_path = default_event_hdf5_path
    chunck_data = list(
        run_chunk_operation(n_chunks=n_chunks, interval_idx=interval_idx)
    )
    with open(
        default_folder_data / f"analysis_{path_hdf5.stem}_{interval_idx}.json", "w+"
    ) as fp_out:
        with h5py.File(path_hdf5) as fp:
            for build_log_name, _ in tqdm.tqdm(chunck_data, position=0):
                groups: List[GroupDict] = get_groups(
                    ciena_path,
                    build_log_name,
                )
                reference_clusters = []
                clusters_mapping = {}
                cluster_incr = 0
                for e in groups:
                    if e["group_id"] not in clusters_mapping:
                        clusters_mapping[e["group_id"]] = cluster_incr
                        cluster_incr += 1
                    reference_clusters.append(clusters_mapping[e["group_id"]])
                reference_clusters = np.array(reference_clusters)
                # get data from target file
                gpe = fp[build_log_name]
                for i, e in enumerate(filter(lambda x: "cluster_" in x, list(gpe))):  # type: ignore
                    predicted_clusters = np.array(gpe[e]["clusters"])  # type: ignore
                    data = {
                        "build_log_name": build_log_name,
                        "hyperparameters_chosen": {
                            k: v
                            for k, v in gpe[e]["hyperparameters_chosen"].attrs.items()
                        },
                        "predicted": [
                            int(e) for e in np.array(gpe[e]["clusters"]).tolist()
                        ],
                        "rand_index": float(
                            rand_score(reference_clusters, predicted_clusters)
                        ),
                        "adjusted_rand_index": float(
                            adjusted_rand_score(reference_clusters, predicted_clusters)
                        ),
                        "jaccard_index": float(
                            jaccard_score(
                                reference_clusters, predicted_clusters, average="micro"
                            )
                        ),
                    }
                    data = json.dumps(data, cls=NumpyEncoder)
                    fp_out.write(data + "\n")


def txt_embeddings_to_hdf5(folder: Path):
    folder = existing_path(folder, is_folder=True)
    split_file = {}
    split_unit = {}
    with h5py.File(folder.parent / f"datasets_benchmark.hdf5", "w") as fp_out:
        files = list(folder.rglob("*.txt"))
        for f in tqdm.tqdm(files, position=0):
            with open(f) as fp:
                lines = [e.strip() for e in fp.readlines()]
            for i, l in tqdm.tqdm(enumerate(lines), position=1, total=len(lines)):
                id = str(uuid.uuid4())
                dataset = fp_out.create_dataset(name=id, shape=(0,))
                dataset.attrs["event_id"] = id
                dataset.attrs["text"] = l.strip()
                dataset.attrs["file"] = f.stem
                dataset.attrs["line"] = i
                split_unit[id] = [id]
                if f.stem not in split_file:
                    split_file[f.stem] = []
                split_file[f.stem].append(id)

    with open(folder.parent / "split_unitaire_datasets_benchmark.json", "w") as fp:
        json.dump(split_unit, fp)
    with open(folder.parent / "split_file_datasets_benchmark.json", "w") as fp:
        json.dump(split_file, fp)


if __name__ == "__main__":
    print("start")
    fire.Fire(
        {
            "extract_logs_from_file": extract_logs_from_file,
            "run_parser": run_parser,
            "run_variable_matrix": run_variable_matrix,
            "run_embeddings": run_embeddings,
            "run_embeddings_distances": run_embeddings_distances,
            "test_running_kmedoids": test_running_kmedoids,
            "run_merge_with_pattern": run_merge_with_pattern,
            "run_generate_clusters_all_build_logs": run_generate_clusters_all_build_logs,
            "run_merge_nested": run_merge_nested,
            "run_merge_json": run_merge_json,
            "run_compare_to_ciena": run_compare_to_ciena,
            "txt_embeddings_to_hdf5": txt_embeddings_to_hdf5,
        }
    )
