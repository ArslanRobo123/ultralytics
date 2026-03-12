# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Yield batches from the underlying iterator.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite DataLoader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the DataLoader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration over a
    dataset without recreating the sampler.

    Attributes:
        sampler (torch.utils.data.Sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that assigns contiguous batch-aligned chunks of the dataset to each GPU.

    Unlike PyTorch's DistributedSampler which distributes samples in a round-robin fashion (GPU 0 gets indices
    [0,2,4,...], GPU 1 gets [1,3,5,...]), this sampler gives each GPU contiguous batches of the dataset (GPU 0 gets
    batches [0,1,2,...], GPU 1 gets batches [k,k+1,...], etc.). This preserves any ordering or grouping in the original
    dataset, which is critical when samples are organized by similarity (e.g., images sorted by size to enable efficient
    batching without padding when using rect=True).

    The sampler handles uneven batch counts by distributing remainder batches to the first few ranks, ensuring all
    samples are covered exactly once across all GPUs.

    Args:
        dataset (Dataset): Dataset to sample from. Must implement __len__.
        num_replicas (int, optional): Number of distributed processes. Defaults to world size.
        batch_size (int, optional): Batch size used by dataloader. Defaults to dataset.batch_size or 1.
        rank (int, optional): Rank of current process. Defaults to current rank.
        shuffle (bool, optional): Whether to shuffle indices within each rank's chunk. Defaults to False. When True,
            shuffling is deterministic and controlled by set_epoch() for reproducibility.

    Examples:
        >>> # For validation with size-grouped images
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # For training with shuffling
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize the sampler with dataset and distributed training parameters."""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        # ensure all ranks have a sample if batch size >= total size; degenerates to round-robin sampler
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """Calculate the start and end sample indices for this rank."""
        # Calculate which batches this rank handles
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas

        # This rank gets an extra batch if rank < remainder
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)

        # Calculate starting batch: base position + number of extra batches given to earlier ranks
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank

        # Convert batch indices to sample indices
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)

        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """Generate indices for this rank's contiguous chunk of the dataset."""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples in this rank's chunk."""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure different shuffling patterns across epochs.

        Args:
            epoch (int): Epoch number to use as the random seed for shuffling.
        """
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HarmonizedClassMap:
    """Harmonize class indices across multiple datasets with per-dataset class filtering.

    Accepts a list of per-dataset YAML paths, each containing a 'names' field and an
    optional 'classes_to_train' / 'classes_to_train_on' field.  Builds a unified global
    class list and per-dataset local→global remap dicts so that label class IDs from
    different datasets can be remapped to a single contiguous ID space at load time.

    Attributes:
        all_names (list[str]): All unique class names seen across every dataset YAML.
        name_to_global (dict[str, int]): Maps class name → global ID (first-seen order).
        dataset_maps (list[dict[int, int]]): Per-dataset {local_id: global_contiguous_id}.
        classes_to_train (set[str]): Union of all classes selected for training.
        per_dataset_train_classes (list[set[str]]): Selected classes per dataset.
        train_indices (list[int]): Global IDs that are active for training.
        train_names (list[str]): Class names corresponding to train_indices.
        global_to_train (dict[int, int]): Maps global_id → contiguous train_id.
        nc (int): Number of active training classes.
        source_names (list[str]): Stem of each YAML path (used for logging).
        yaml_paths (list[str]): Absolute YAML paths as strings.
    """

    def __init__(self, yaml_paths):
        """Build harmonized class map from a list of per-dataset YAML file paths."""
        if isinstance(yaml_paths, (str, Path)):
            yaml_paths = [str(yaml_paths)]
        if not yaml_paths:
            raise ValueError("harmonize_yaml_paths is empty")

        self.all_names = []
        self.name_to_global = {}
        self.dataset_maps = []
        self.classes_to_train = set()
        self.per_dataset_train_classes = []
        self.yaml_paths = [str(x) for x in yaml_paths]
        self.source_names = []

        def _sorted_names(names_obj):
            if isinstance(names_obj, dict):
                try:
                    keys = sorted(names_obj.keys(), key=lambda k: int(k))
                except Exception:
                    keys = sorted(names_obj.keys())
                return [str(names_obj[k]).strip() for k in keys]
            return [str(x).strip() for x in names_obj]

        def _resolve_train_classes(ydata, names):
            raw = ydata.get("classes_to_train", ydata.get("classes_to_train_on", None))
            if raw is None:
                return set(names)
            if isinstance(raw, (str, int)):
                raw = [raw]
            resolved = set()
            for item in raw:
                if isinstance(item, int):
                    if 0 <= item < len(names):
                        resolved.add(names[item])
                    continue
                s = str(item).strip()
                if s in names:
                    resolved.add(s)
                    continue
                if s.isdigit():
                    idx = int(s)
                    if 0 <= idx < len(names):
                        resolved.add(names[idx])
                        continue
                from ultralytics.utils import LOGGER as _L
                _L.warning(f"WARNING ⚠️ Ignoring unknown class '{item}' in classes_to_train")
            return resolved

        for ypath in yaml_paths:
            with open(ypath) as f:
                ydata = yaml.safe_load(f)
            if "names" not in ydata:
                raise KeyError(f"YAML missing 'names': {ypath}")
            names = _sorted_names(ydata["names"])
            train_classes = _resolve_train_classes(ydata, names)

            self.source_names.append(Path(ypath).stem)
            self.per_dataset_train_classes.append(train_classes)
            self.classes_to_train.update(train_classes)

            local_to_global = {}
            for i, cname in enumerate(names):
                if cname not in self.name_to_global:
                    self.name_to_global[cname] = len(self.all_names)
                    self.all_names.append(cname)
                if cname in train_classes:
                    local_to_global[i] = self.name_to_global[cname]
            self.dataset_maps.append(local_to_global)

        self.train_indices = [i for i, cname in enumerate(self.all_names) if cname in self.classes_to_train]
        if not self.train_indices:
            raise ValueError("No classes selected for training after harmonization. Check classes_to_train values.")
        self.nc = len(self.train_indices)
        self.train_names = [self.all_names[i] for i in self.train_indices]
        self.global_to_train = {g: t for t, g in enumerate(self.train_indices)}

        # Remap dataset_maps values from global IDs → contiguous train IDs
        self.dataset_maps = [
            {local: self.global_to_train[global_id]
             for local, global_id in dm.items()
             if global_id in self.global_to_train}
            for dm in self.dataset_maps
        ]


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
    harmonizer: HarmonizedClassMap | None = None,
) -> Dataset:
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        harmonizer=harmonizer,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> InfiniteDataLoader:
    """Create and return an InfiniteDataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[-1] in (
            IMG_FORMATS | VID_FORMATS
        )
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source for
            inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # DataLoader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
