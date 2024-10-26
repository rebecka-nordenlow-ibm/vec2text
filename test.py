import datasets 
import os
import torch
import shutil
from typing import Callable, Dict
import multiprocessing

def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1

def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size
    
def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:

    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
    except (RuntimeError, ValueError):
        # In non-distributed mode, just run regular map()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
    )
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset

def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex

def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)

    return dataset_dict["train"]

#print("STARTING LOAD")
#dataset = load_one_million_instructions()
#print("DATASET", type(dataset), dataset)

#world_size = 2 #1 #torch.distributed.get_world_size()
#cache_path = os.environ.get(
#    "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
#)
#ds_shard_filepaths = [
#    os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
#    for w in range(0, world_size)
#]
#print("ds_shard_filepaths", ds_shard_filepaths)
#dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)
#import time
#time.sleep(3600)






from transformers import AutoTokenizer
from vec2text.analyze_utils import load_experiment_and_trainer_from_pretrained

experiment, trainer = load_experiment_and_trainer_from_pretrained(
    "jxm/t5-base__llama-7b__one-million-instructions__emb",
    #"jxm/t5-base__llama-7b-chat__one-million-instructions__emb"
    #"/dccstor/rit_sva/rebecka/aligners-hackathon/cache/models--jxm--t5-base__llama-7b__one-million-instructions__emb/snapshots/c41cb5d2cab56fffb30b9f91187e99d3113d20e7"
)
print("FINISHED LOAD!!")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
#llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer_out = t5_tokenizer("Generate a Python code for crawling a website for a specific type of data.", return_tensors="pt")
#llama_out = llama_tokenizer("Generate a Python code for crawling a website for a specific type of data.", return_tensors="pt")
#outputs = llama_model(**tokenizer_out)
#tokenizer_out["frozen_embeddings"] = outputs.logits ## shape [1, seq_len, dims]
from datasets import Dataset
dataset = Dataset.from_dict({"labels": tokenizer_out["input_ids"], "input_ids": tokenizer_out["input_ids"]}) #, "frozen_embeddings": tokenizer_out["frozen_embeddings"]})
print(dataset)
#trainer.model.use_frozen_embeddings_as_input = False
#print(len(tokenizer_out["frozen_embeddings"].shape))
#tokenizer_out.to("cuda")
generated = trainer.predict(dataset)
#print(t5_tokenizer.batch_decode(generated))
#experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
#    "jxm/t5-base__llama-7b__one-million-instructions__emb",
#    #"jxm/t5-base__llama-7b-chat__one-million-instructions__emb"
#    #"/dccstor/rit_sva/rebecka/aligners-hackathon/cache/models--jxm--t5-base__llama-7b__one-million-instructions__emb/snapshots/c41cb5d2cab56fffb30b9f91187e99d3113d20e7"
#)
#print("done downloading exp and trainer!")
#trainer.model.use_frozen_embeddings_as_input = False
#trainer.args.per_device_eval_batch_size = 16
#trainer.evaluate(
#    eval_dataset=trainer.eval_dataset["python_code_alpaca"].remove_columns("frozen_embeddings").select(range(40))
#)

