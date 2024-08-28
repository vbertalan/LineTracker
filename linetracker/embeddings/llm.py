"""All the necessary functions to generates embeddings with a largelanguage model on huggingface. Main class is generate_embeddings_llm"""

import torch
import gc
import huggingface_hub
import numpy as np
import transformers as trf

from typing import *# type: ignore

PoolingOperationCode = Literal["mean", "sum"]
PoolingFn = Callable[["torch.Tensor"], "torch.Tensor"]
"""A function that reduces the dimensions of a tensor along the first dimension (seq_length of the large language model)"""
ModelName = Literal["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
"""The name of the model to use as described on huggingface hub"""

class OutOfMemoryError(Exception):
    """An exception that occurs when the inference of the model can not be done due to a lack of memorye"""

    def __init__(
        self,
        number_of_tokens: int,
        model_name: str,
        use_cpu: bool,
        type: Union[Literal["ram", "gpu"], Set[str]],
        text: str,
    ) -> None:
        self.number_of_tokens = number_of_tokens
        self.model_name = model_name
        self.use_cpu = use_cpu
        self.type = type
        self.text = text
        self.message = f"You are out of {self.type} memory for {number_of_tokens=} {model_name=} and {use_cpu=}\nText was {self.text}"
        print(self.message)
        super().__init__(self.message)

def get_pooling_function(pooling_code: "PoolingOperationCode" = "none") -> "PoolingFn":
    """Get the function if you want aggregate the large language model embedding over the sequence length

    # Arguments
    - pooling_code: PoolingOperationCode, the name of the operation to use

    # Returns
    - PoolingFn, a function that is able to ggregate the large language model embedding over the sequence length, first dimension of the tensor (see doc of PoolingFn)
    """
    if pooling_code == "mean":
        return lambda embedding: torch.mean(embedding, dim=0)
    elif pooling_code == "sum":
        return lambda embedding: torch.sum(embedding, dim=0)
    elif pooling_code == "none":
        return lambda embedding: embedding
    else:
        raise ValueError(f"{pooling_code=} is not possible")


def get_tokenizer(
    model_name: str, token: str, cache_dir: Optional[str] = None, *args, **kwargs
) -> trf.PreTrainedTokenizerFast:
    """Get the tokenizer linked with a model (here expected llama). Return the tokenizer

    # Arguments
    - model_name: str, the name of the model to use as described on huggingface hub
    - token: str, the token to access the model (read or write)
    - cache_dir: Optional[str] = None, providing this argument can avoid downloading repeatidly the tokenizer by storing it in the folder path specified in this argument

    # Returns
    - trf.PreTrainedTokenizerFast, the fast tokenizer version of the tokenizer (LlamaTokenizerFast if meta-llama/Llama-2-...b)
    """
    huggingface_hub.login(token=token)
    tokenizer: trf.PreTrainedTokenizerFast = trf.AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=True,
    )  # type: ignore
    return tokenizer


def get_model(
    model_name: str,
    token: str,
    optimization: bool = True,
    use_cpu: bool = False,
    cache_dir: Optional[str] = None,
    *args,
    **kwargs,
) -> "trf.LlamaForCausalLM":
    """Get the model specified as input using optimization if asked

    # Arguments
    - model_name: str, the name of the model to use as described on huggingface hub
    - token: str, the token to access the model (read or write)
    - optimization: bool = True, if True uses double quantization if on GPU or load in 8 bits if only cpu
    - use_cpu: bool = False, wether to use the cpu
    - cache_dir: Optional[str] = None, providing this argument can avoid downloading repeatidly the model by storing it in the folder path specified in this argument

    # Returns
    - trf.LlamaForCausalLM, the model ready to use for inference (embeddings here but can be also for simple inference). Additionnal steps if want to setup the model for finetuning
    """
    huggingface_hub.login(token=token)
    if not use_cpu:
        double_quant_config = None
        if optimization:
            double_quant_config = trf.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
        model = trf.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=double_quant_config,
            return_dict=True,
            output_hidden_states=False,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map="auto" if not use_cpu else "cpu",
            load_in_8bit=optimization,
            torch_dtype=None,
            **kwargs,
        )
    else:
        model = trf.AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            output_hidden_states=False,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **kwargs,
        )
    model.config.use_cache = False
    return model


class generate_embeddings_llm:
    def __init__(self, 
        model_name: str,
        token: str,
        cache_dir: Optional[str] = None,
        use_cpu: bool = False,
        **kwargs,
    ):
        self.tokenizer = get_tokenizer(model_name, token, cache_dir=cache_dir)
        self.model = get_model(model_name, token, optimization=True, use_cpu=use_cpu, cache_dir=cache_dir, **kwargs)  # type: ignore
        self.model.eval()
        self.model_name = model_name
        self.use_cpu = use_cpu
        self.preprompt = ""
    def __call__(self,
        events: List[str],
        pooling_fn: PoolingFn,
        limit_tokens: int = -1,
        precision = np.float32,
    ) -> Generator[np.ndarray, Any, Any]:
        """Generates large language model embeddings as a generator. Raises a OutOfMemoryError whenever there is a ram or gpu memory error (often due to too big size of the text and model)

        # Arguments
        - events: List[str], the events text to generate the embeddings from
        - pooling_fn: PoolingFn, a function that reduces the dimensions of a tensor along the first dimension (seq_length of the large language model)
        - model_name: ModelName, the name of the model to use as described on huggingface hub
        - token: str,  the huggingface token to access the model (read or write)
        - use_cpu: bool = False, if True will use the cpu and not the gpu, if false assumes that it can use a single gpu
        - cache_dir: Optional[str] = None, providing this argument can avoid downloading repeatidly the model and tokenizer by storing them in the folder path specified in this argument
        - limit_tokens: int = -1, limit the number of tokens of the input to avoid an out of memory error. -1 is for no limit
        - precision = np.float32, the precision to use after the conversion in the numpy array

        # Returns
        - Generator[np.ndarray, Any, Any], a generator that generates the embedding of dimension (voc_size=4096 for llama models (can be modified but requires retraining))
            
        # Example usage
        
        ```python
        >>> pooling_fn = get_pooling_function('mean')
        >>> token = os.getenv('LLAMA_TOKEN')
        >>> use_cpu = True
        >>> cache_dir = os.getenv('LLAMA_CACHE_DIR')
        >>> limit_tokens = 1000
        >>> for text, embedding in zip(events, generate_embeddings_llm(events, pooling_fn, 'meta-llama/Llama-2-13b', token, use_cpu, cache_dir, limit_tokens, np.float32)):
        ...    # either accumulate the embeddings or directly work on them here
        ...    print(text,embedding)
        >>> 
        ```
        """
        for i, event in enumerate(events):
            event = self.preprompt+event
            tokenized_full_text = self.tokenizer.encode(event, truncation=True)
            limit_tokens_sample = limit_tokens
            if limit_tokens == -1:
                limit_tokens_sample = len(tokenized_full_text)
            tokenized_full_text = tokenized_full_text[:limit_tokens_sample]
            text = self.tokenizer.decode(tokenized_full_text)
            input_tensor = torch.tensor([tokenized_full_text], dtype=torch.int32)
            with torch.no_grad():
                try:
                    embeddings = self.model(input_tensor)  # type: ignore
                    embedding = embeddings.logits[0]
                    embedding = np.array(pooling_fn(embedding).tolist(), dtype=precision)
                    yield embedding
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()  # type: ignore
                    error = OutOfMemoryError(
                        number_of_tokens=input_tensor.shape[1],
                        model_name=self.model_name,
                        use_cpu=self.use_cpu,
                        type="gpu",
                        text=text,
                    )
                    raise error
                except MemoryError:
                    gc.collect()
                    error = OutOfMemoryError(
                        number_of_tokens=input_tensor.shape[1],
                        model_name=self.model_name,
                        use_cpu=self.use_cpu,
                        type="ram",
                        text=text,
                    )
                    raise error
