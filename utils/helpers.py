import os
import csv
import pandas as pd
import numpy as np
from options import base_dir, anchors, versions
from typing import List, Union
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer

anchors = pd.read_csv(f'{base_dir}{anchors}', header=None)
sentence_transformer = SentenceTransformer('all-mpnet-base-v2')


def read_data_from_csv() -> List | None:
    """
    loads the data from the csv file and returns a dictonary
    """
    try:
        codes = []

        for version in versions:
            filepath = f'{base_dir}{version}.csv'

            if not os.path.exists(filepath):
                raise FileNotFoundError(f'couldn\'t find {filepath}.')

            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    code, code_description = row[0], row[1]
                    codes.append((versions.index(version), code,
                                 code_description))
        return codes

    except FileNotFoundError as e:
        print(e)

    return None


def read_data_from_csv_for(version: str) -> List | None:
    """
    loads the data from the csv file and returns a dictonary
    """
    try:
        codes = []
        filepath = f'{base_dir}{version}.csv'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f'couldn\'t find {filepath}.')

        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                code, code_description = row[0], row[1]
                codes.append((version, code, code_description))
        return codes

    except FileNotFoundError as e:
        print(e)

    return None


def get_num_anchors() -> int:
    return len(anchors)


def get_anchors() -> List:
    return anchors.iloc[:, 1].tolist()


# def get_one_hot_encoding(anchor: str) -> np.ndarray:
#     anchor_list = get_anchors()
#     if anchor not in anchor_list:
#         print(anchor)
#     vec = np.zeros(get_num_anchors())
#     idx = anchor_list.index(anchor)
#     vec[idx] = 1
#     return vec

def toTensors(idx: List[int]) -> Tensor:
    return torch.nn.functional.one_hot(
            torch.tensor(idx), num_classes=768)

def toTensorTensors(idx: Tensor) -> Tensor:
    return torch.nn.functional.one_hot(
            idx.detach().clone(), num_classes=768)


def get_max_num_codes() -> int:
    n_codes = list(map(lambda x: len(read_data_from_csv_for(
        x)) if read_data_from_csv_for(x) else 0, versions))

    return max(n_codes)

def get_code_idx(code: str, idx: int) -> Union[int, None]:
    codes = pd.read_csv(f'{base_dir}{versions[idx]}.csv', header=None).iloc[:, 0]
    codes = list(codes)

    if code not in codes:
        print(code)

    return codes.index(code) if code in codes  else None

def get_embeddings(codes: List[str]) -> Tensor:
    embeddings = list(map(lambda x: sentence_transformer.encode(x), codes))
    return torch.tensor(embeddings)

def get_code_embds_from_idx(idx: List[int], ver: List[int]) -> Tensor:
    code1 = pd.read_csv(f'{base_dir}{versions[0]}.csv', header=None).iloc[:, 1]
    code2 = pd.read_csv(f'{base_dir}{versions[1]}.csv', header=None).iloc[:, 1]

    codes = []
    for i in range(len(idx)):
        if ver[i] == 0:
            codes.append(code1[idx[i]])
            continue

        codes.append(code2[idx[i]])

    embeddings = list(map(lambda x: sentence_transformer.encode(x), codes))
    return torch.tensor(embeddings)

