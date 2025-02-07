# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datatrove.utils.dataset import DatatroveFolderDataset
from numba import jit
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from datasets import Dataset, load_dataset
from datasets import IterableDataset as HFIterableDataset
from datasets.distributed import split_dataset_by_node
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger


def _load_c4_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split="train", streaming=True)


def _process_c4_text(sample: Dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
# DATASETS = {
#     "c4": DatasetConfig(
#         path="allenai/c4",
#         loader=_load_c4_dataset,
#         text_processor=_process_c4_text,
#     ),
#     "c4_test": DatasetConfig(
#         path="tests/assets/c4_test",
#         loader=lambda path: load_dataset(path, split="train"),
#         text_processor=_process_c4_text,
#     ),
# }


# def _validate_dataset(dataset_name: str, dataset_path: str = None) -> tuple[str, Callable, Callable]:
#     """Validate dataset name and path."""
#     if dataset_name not in DATASETS:
#         raise ValueError(
#             f"Dataset {dataset_name} is not supported. " f"Supported datasets are: {list(DATASETS.keys())}"
#         )

#     config = DATASETS[dataset_name]
#     path = dataset_path or config.path
#     logger.info(f"Preparing {dataset_name} dataset from {path}")
#     return path, config.loader, config.text_processor


def load_nanoset(seq_len: int):
    nanoset_folder = [
        "tests/assets/smollm1_corpus/fineweb-edu-dedup",
        "tests/assets/smollm1_corpus/cosmopedia-v2",
        "tests/assets/smollm1_corpus/python-edu",
        "tests/assets/smollm1_corpus/open-web-math",
        "tests/assets/smollm1_corpus/stackoverflow",
    ]
    nanoset_weights = [0.7, 0.15, 0.08, 0.06, 0.01]
    nanoset = Nanoset(
        dataset_folders=nanoset_folder,
        dataset_weights=nanoset_weights,
        sequence_length=seq_len,
        token_size=2,  # smollm
        train_split_num_samples=100,
        random_seed=42,
    )
    return HFIterableDataset.from_generator(nanoset)


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        # dataset_name = dataset_name.lower()

        # path, dataset_loader, text_processor = _validate_dataset(dataset_name, dataset_path)
        # ds = dataset_loader(path)
        # self.dataset_name = dataset_name
        ds = load_nanoset(seq_len)
        self.dataset_name = "nanoset"
        self._data = split_dataset_by_node(ds, rank, world_size)
        # self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        # self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        # self._all_tokens: List[int] = []

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        return iter(self._data.skip(self._sample_idx))

    def __iter__(self):
        # max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                # sample_text = self._text_processor(sample)
                # sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                # self._all_tokens.extend(sample_tokens)
                # self._sample_idx += 1

                # while len(self._all_tokens) >= max_buffer_token_len:
                #     x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                #     # update tokens to the remaining tokens
                #     self._all_tokens = self._all_tokens[max_buffer_token_len:]
                #     input = x[:-1]
                #     label = x[1:]
                #     yield input, label
                yield sample[:-1], sample[1:]

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        # self._all_tokens = state_dict["token_buffer"]

    def state_dict(self):
        # return {"token_buffer": self._all_tokens, "sample_idx": self._sample_idx}
        return {"sample_idx": self._sample_idx}


# https://github.com/huggingface/nanotron/blob/2cde8f63519f15aa449803441ec70225644bd25d/src/nanotron/data/nanoset.py#L16
class Nanoset(IterableDataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        token_size (int): Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
    ) -> None:
        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.datatrove_datasets = []
        for dataset_folder in self.dataset_folders:
            self.datatrove_datasets.append(
                DatatroveFolderDataset(
                    folder_path=dataset_folder,
                    filename_pattern=os.path.join(dataset_folder, "*.ds"),
                    seq_len=sequence_length,
                    recursive=False,
                    token_size=token_size,
                    shuffle=True,
                )
            )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(
            self.dataset_weights
        ), f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()

        self.print_nanoset_info()

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        dataset = self.dataset_index[idx]
        dataset_sample = self.dataset_sample_index[idx]

        item = self.datatrove_datasets[dataset][dataset_sample]

        return item["input_ids"]  #!

    #!
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    # IterableDataset.from_generator requires __call__ to be implemented
    def __call__(self):
        for idx in range(len(self)):
            yield self[idx]

    #!

    def build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]

        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):
        logger.info(f"> Total number of samples: {len(self)}")
        logger.info(f"> Total number of tokens: {len(self) * self.sequence_length}")

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            logger.info(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
            )


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # Iterate over all samples
    for sample_idx in range(n_samples):
        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index


# https://github.com/huggingface/nanotron/blob/2cde8f63519f15aa449803441ec70225644bd25d/src/nanotron/data/utils.py#L6
def normalize(weights: Sequence[int | float]) -> np.ndarray:
    """
    Normalize elements of a list

    Args:
        weights (List[float]): The weights

    Returns:
        List[numpy.array]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = w / w_sum
    return w


def count_dataset_indexes(dataset_idx: np.ndarray, n_datasets: int):
    counts = []

    for dataset in range(n_datasets):
        counts.append(np.count_nonzero(dataset_idx == dataset))

    return counts


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, world_size: int):
        super().__init__(hf_ds, batch_size)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"
        # Data loader resharding is not yet supported, so we need to store the world size to compare during loading
        # raise error if dp_word_size does not match.
        self._world_size = world_size

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self._world_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}")
            return
        assert (
            self._world_size == state_dict["world_size"]
        ), "dp_degree is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    infinite: bool = True,
):
    """Build a data loader for HuggingFace datasets."""
    hf_ds = HuggingFaceDataset(dataset_name, dataset_path, tokenizer, seq_len, world_size, rank, infinite)
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size, world_size=world_size)
