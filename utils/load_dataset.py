import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from utils.helpers import get_anchors, get_max_num_codes, get_code_idx, read_data_from_csv


class LoadDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = read_data_from_csv()

        self.anchors_list = get_anchors()
        self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')

    def __getitem__(self, index):
        item = self.dataset[index]
        idx, code, text = item
        ## input will be <code> <code_descrition>;
        feature = torch.from_numpy(self.sentence_transformer.encode(f'{code} {text}'))
        version = idx
        ground_truth = torch.tensor(
            get_code_idx(code, idx))
        
        return feature, version, ground_truth

    def __len__(self) -> int:
        return len(self.dataset)
