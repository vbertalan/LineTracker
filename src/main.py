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
import DrainMethod

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
    default_folder_data: Path = Path(f"/scratch/{os.environ['USER']}/LineTracker/data")
else:
    default_folder_data: Path = Path(f"/scratch/rmoine/LineTracker/data")
default_datasetname = "eclipse_72k"

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
    import evaluate
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


def existing_path(p: Union[str, Path], is_folder: bool) -> Path:
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
    *args,
    **kwargs,
) -> "trf.LlamaForCausalLM":
    if hidden_states:
        last_state = True
    huggingface_hub.login(token=token)
    double_quant_config = None
    if quant:
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


@print_args
def get_llama2_embeddings(
    folder_out: str,  # type: ignore
    folder_data: str,  # type: ignore
    dataset_choice: DatasetName,
    pooling_op: PoolingOperationCode,
    layer_id: int = -1,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
    n_chunks: Optional[int] = None,
    interval_idx: Optional[int] = None,
    id_pred: str = "",
    model_name: ModelName = default_model,  # type: ignore
    token: str = default_token,
    use_cpu: bool = False,
    limit_tokens: int = -1,
    override: bool = False,
    field_bug_id: str = "event_id",
    field_data: str = "raw",
):
    """From a json file with the description use llama2 to generate the embeddings for each data sample. The intent of this function is to be called with multiple nodes on a slurm server to have faster results

    # Arguments
    - model_name: ModelName, the name of the model to use to generate the embeddings
    - path_data_preprocessed: Path, the path to the json file containing the data in the format [{'description': "...", 'bug_id': ...}, ...]
    - folder_out, Path the folder where to put the data, name automatically determined with start seed
    - pooling_fn: PoolingFn, function to do the aggregation
    - layers_ids: Optional[Tuple[int]] = (0, ), the layers embeddings to use
    - start: int = 0, the starting element in the data to process
    - end: int = -1, the ending element to process in the data
    - limit_tokens: int = -1, the limit number of tokens to use (all by default)
    - id_pred: str = "", the id to put in the filename to help for the aggregation of files after

    """
    id_pred = id_pred.replace("/", "--")
    folder_out: Path = (
        existing_path(folder_out, is_folder=True) / f"embeddings{id_pred}"
    )
    folder_out.mkdir(parents=True, exist_ok=True)
    folder_data: Path = existing_path(folder_data, is_folder=True)
    path_data_preprocessed: Path = existing_path(
        folder_data / f"{dataset_choice}.hdf5", is_folder=False
    )
    assert_valid_token(token)
    model_name: ModelName = get_literal_value(model_name)
    pooling_fn: PoolingFn = get_pooling_operation(
        get_literal_value(pooling_op, PoolingOperationCode)
    )
    tokenizer, model = initialize_model_inference(model_name, token, hidden_states=True, return_dict=True, quant=not use_cpu)  # type: ignore
    model.eval()
    with h5py.File(path_data_preprocessed, "r") as f:
        data = list(f)
        n_data = len(f)
    start, end = generate_seeds(
        n_data=n_data,
        seed_start=seed_start,
        seed_end=seed_end,
        n_chunks=n_chunks,
        interval_idx=interval_idx,
    )
    if end == -1:
        end = n_data
    data = data[start:end]
    get_file_path = (
        lambda layer_id: folder_out
        / f"embeddings_chunk{id_pred}_layer_{layer_id}_{start}.hdf5"
    )
    print(f"Running for {start=} {end=}")
    folder_predictions = folder_out
    folder_predictions.mkdir(exist_ok=True, parents=True)
    ids_already_there = set()
    if get_file_path(layer_id).exists():
        with h5py.File(get_file_path(layer_id), "r") as fp:
            ids_already_there = set(list(fp.keys()))
    with h5py.File(path_data_preprocessed, "r") as f:
        for data_key in tqdm.tqdm(data, total=len(data)):
            already_computed = str(f[data_key].attrs[field_bug_id]) in ids_already_there
            if already_computed and not override:
                continue
            tokenized_full_text = tokenizer.encode(f[data_key].attrs[field_data])
            limit_tokens_sample = limit_tokens
            if limit_tokens == -1:
                limit_tokens_sample = len(tokenized_full_text)
            tokenized_full_text = tokenized_full_text[:limit_tokens_sample]
            logger.info(f"{len(tokenized_full_text)=}")
            try:
                input_tensor = torch.tensor([tokenized_full_text], dtype=torch.int32)
                with torch.no_grad():
                    embeddings = model(input_tensor)  # type: ignore
                    embedding = embeddings.hidden_states[layer_id]
                    pooled_embedding = np.array(
                        pooling_fn(embedding).tolist()[0], dtype=np.float32
                    )
                with h5py.File(get_file_path(layer_id), "a") as fp:
                    id = str(f[data_key].attrs[field_bug_id])
                    if already_computed:
                        del fp[id]
                    fp.create_dataset(id, data=pooled_embedding, dtype="f")
                del embeddings
                del input_tensor
            except torch.cuda.OutOfMemoryError as e:
                logger.info(f"Error for {len(tokenized_full_text)} tokens: {e}")
            gc.collect()
            torch.cuda.empty_cache()  # type: ignore


"""End of extracted functions from repository https://github.com/AndressaStefany/severityPrediction"""


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
        out_file = Path("./logs/trat3_production_1650_1700_20231411_v1.json")
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
def gather_embeddings(folder_in: Path = default_folder_data, pattern: str = "*.hdf5", name_out: str = "embeddings.hdf5", folder_out: Path = default_folder_data):
    folder_in = existing_path(folder_in,is_folder=True)
    folder_out = existing_path(folder_out,is_folder=True)
    with h5py.File(folder_out / name_out, "w") as fp_out:
        files = list(folder_in.rglob(pattern=pattern))
        for p in tqdm.tqdm(files,total=len(files)):
            with h5py.File(p,"r") as fp_in:
                for id,v in tqdm.tqdm(fp_in.items(),total=len(fp_in)):
                    if id not in fp_out:
                        fp_out.create_dataset(id, data=np.copy(v), dtype="f")
                        fp_out.flush()

def get_distance_fn(distance_name: Literal['euc','cos']) -> Callable[[np.ndarray,np.ndarray],float]:
    if distance_name == "euc":
        return lambda x,y: np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    elif distance_name == "cos":
        return lambda x,y: 1-np.dot(x, y) / (np.dot(x, x)*np.dot(y, y))
    else:
        raise ValueError(f"{distance_name} is not a valid distance_name")

@print_args
def split_embeddings(path_in: Optional[Path] = None, path_out: Optional[Path] = None):
    """Split along the build logs"""
    if path_in is None:
        path_in = default_folder_data / "input" / "trat3_production_1650_1700_20231411_v1.hdf5"
    if path_out is None:
        path_out = default_folder_data / "splitted_event_ids.json"
    path_in = existing_path(path_in, is_folder=False)
    data = {}
    with h5py.File(path_in, "r") as fp:
        keys = list(fp)
        for k in tqdm.tqdm(keys):
            build_log_name = fp[k].attrs["planid"]+"--"+fp[k].attrs["log_name"]
            if build_log_name not in data:
                data[build_log_name] = []
            data[build_log_name].append(k)
    with open(path_out, "w") as fp:
        json.dump(data,fp, indent=2)
        
@print_args
def compute_distances(distance_name: Literal['euc','cos'], path_in: Optional[Path] = None, path_out: Optional[Path] = None, split_file: Optional[Path] = None):
    if path_in is None:
        path_in = default_folder_data / "embeddings.hdf5"
    path_in = Path(path_in)
    if path_out is None:
        path_out = default_folder_data / f"distances_{distance_name}.hdf5"
    path_out = Path(path_out)
    if split_file is None:
        split_file = default_folder_data / "splitted_event_ids.json"
    split_file = Path(split_file)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    distance = get_distance_fn(distance_name)
    with h5py.File(path_in, "r") as fp_in:
        keys = list(fp_in)
        mapping_keys = {k:i for i,k in enumerate(keys)}
        data = {}
        for build_log_name, keys_build_logs in splits.items():
            mapping_in_build_log = {k:i for i,k in enumerate(keys_build_logs)}
            data[build_log_name] = {"distances":[],"mapping_keys_src":mapping_keys, "mapping_ids":{}, "mapping_in_build_log": mapping_in_build_log}
            combinations = list(it.combinations(keys_build_logs, 2))
            for i, (k1, k2) in tqdm.tqdm(enumerate(combinations),total=len(combinations)):
                data[build_log_name]["mapping_ids"][i] = [k1,k2]
                embedding1 = np.copy(fp_in[k1]) #type: ignore
                embedding2 = np.copy(fp_in[k2]) #type: ignore
                data[build_log_name]["distances"].append(distance(embedding1, embedding2))
    with open(path_out, "w") as fp:
        json.dump(data, fp)

class JSONParserInput(TypedDict):
    text: str 
class ParserJSON(DrainMethod.LogParser):
    """Parse the variable from data provided as json
    
    # Arguments:
        - data: List[Dict]
        - log_format: str, Format of the file, if there are different fields
        - depth: int = 4, limit depth of which to search nodes
        - similarity_threshold: float = 0.4, similarity threshold between leaves/log clusters
        - maxChild: int = 3, max number of children of an internal node
        - rex: Optional[list] = None, regular expressions used in preprocessing (step1)
        - keep_para: bool = True, 
    """
    def __init__(self, data: List[JSONParserInput], log_format: str, depth: int = 4, similarity_threshold: float = 0.4, 
                 maxChild: int = 3, rex: Optional[list] = None, keep_para: bool = True):
        if rex is None:
            rex = []
        self.data = data
        super().__init__(
            log_format=log_format,
            indir=None,
            outdir=None,
            depth=depth,
            st=similarity_threshold,
            maxChild=maxChild,
            rex=rex,
            keep_para=keep_para
        )
    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(regex, headers)

    def log_to_dataframe(self, regex, headers):
        log_messages = []
        linecount = 0
        for line in self.data:
            try:
                match = regex.search(line["text"].strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf
        
def parse_logs(path_in: Optional[Path] = None):
    log_format = '<Content>' # Format of the file, if there are different fields
    regex = [] # Regex strings for Drain execution
if __name__ == "__main__":
    print("start")
    fire.Fire(
        {
            "get_llama2_embeddings": get_llama2_embeddings,
            "extract_logs_from_file": extract_logs_from_file,
            "gather_embeddings": gather_embeddings,
            "split_embeddings": split_embeddings,
            "compute_distances": compute_distances,
        }
    )
