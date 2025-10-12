import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Dict
import logging
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
logger = logging.getLogger(__name__)

class GLUEDataset(Dataset):
    def __init__(self, task_name: str='sst2', split: str='train', max_length: int=128, cache_dir: Optional[str]=None):
        if not HAS_DATASETS:
            raise ImportError('pip install datasets')
        self.task_name = task_name.lower()
        self.split = split
        self.max_length = max_length
        self.dataset = load_dataset('glue', self.task_name, split=split, cache_dir=cache_dir)
        self.num_labels = {'cola': 2, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'sts-b': 1, 'mnli': 3, 'qnli': 2, 'rte': 2, 'wnli': 2}[self.task_name]
        logger.info(f'Loaded GLUE {task_name} {split}: {len(self.dataset)} examples')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        if self.task_name == 'sst2':
            text = item['sentence']
            label = item['label']
        elif self.task_name in ['mrpc', 'qqp', 'rte', 'wnli']:
            text = item['sentence1'] + ' [SEP] ' + item['sentence2']
            label = item['label']
        elif self.task_name == 'cola':
            text = item['sentence']
            label = item['label']
        elif self.task_name == 'qnli':
            text = item['question'] + ' [SEP] ' + item['sentence']
            label = item['label']
        elif self.task_name == 'mnli':
            text = item['premise'] + ' [SEP] ' + item['hypothesis']
            label = item['label']
        elif self.task_name == 'sts-b':
            text = item['sentence1'] + ' [SEP] ' + item['sentence2']
            label = item['label']
        else:
            raise ValueError(f'Unknown GLUE task: {self.task_name}')
        return {'text': text, 'label': label}

class MNISTDataset(Dataset):
    def __init__(self, root: str='./data', train: bool=True, download: bool=True, flatten: bool=True):
        if not HAS_VISION:
            raise ImportError('pip install torchvision')
        self.flatten = flatten
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)
        logger.info(f"Loaded MNIST {('train' if train else 'test')}: {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.dataset[idx]
        if self.flatten:
            img = img.flatten()
        return (img, torch.tensor(label, dtype=torch.long))

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str='./data', train: bool=True, download: bool=True):
        if not HAS_VISION:
            raise ImportError('pip install torchvision')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))])
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        logger.info(f"Loaded CIFAR-10 {('train' if train else 'test')}: {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.dataset[idx]
        return (img, torch.tensor(label, dtype=torch.long))

class IMDBDataset(Dataset):
    def __init__(self, split: str='train', max_length: int=512, cache_dir: Optional[str]=None):
        if not HAS_DATASETS:
            raise ImportError('pip install datasets')
        self.split = split
        self.max_length = max_length
        self.dataset = load_dataset('imdb', split=split, cache_dir=cache_dir)
        logger.info(f'Loaded IMDB {split}: {len(self.dataset)} examples')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        return {'text': item['text'], 'label': item['label']}

class WikiTextDataset(Dataset):
    def __init__(self, version: str='wikitext-2-v1', split: str='train', seq_length: int=128, cache_dir: Optional[str]=None):
        if not HAS_DATASETS:
            raise ImportError('pip install datasets')
        self.split = split
        self.seq_length = seq_length
        self.dataset = load_dataset(version, split=split, cache_dir=cache_dir)
        logger.info(f'Loaded WikiText {version} {split}: {len(self.dataset)} examples')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        return {'text': item['text']}

def create_benchmark_dataset(dataset_name: str, split: str='train', **kwargs) -> Dataset:
    if dataset_name.startswith('glue_'):
        task = dataset_name.replace('glue_', '')
        return GLUEDataset(task_name=task, split=split, **kwargs)
    datasets = {
        'mnist': lambda: MNISTDataset(train=split == 'train', **kwargs),
        'cifar10': lambda: CIFAR10Dataset(train=split == 'train', **kwargs),
        'imdb': lambda: IMDBDataset(split=split, **kwargs),
        'wikitext': lambda: WikiTextDataset(split=split, **kwargs)
    }
    if dataset_name not in datasets:
        raise ValueError(f'Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}')
    return datasets[dataset_name]()
