import torch
import itertools
import numpy as np
import pandas as pd
from model.decoder import Decoder
from utils.helpers import get_max_num_codes, toTensors, get_embeddings, get_code_idx, get_code

if __name__ == "__main__":
    n_input = 768
    n_output = get_max_num_codes()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_path = './trained/state_new.pth'
    source_idx = 0
    target_idx = 1

    checkpoint = torch.load(trained_path)

    model = Decoder(n_input, n_output)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    gt = pd.read_csv('./data/mapping-icd10-10am.csv', header=None)
    source_code, source_text, target_code, target_text = gt.iloc[:, 0], gt.iloc[:, 1], gt.iloc[:, 2], gt.iloc[:, 3]
    
    source_code_gt = list(map(lambda x: get_code_idx(x, source_idx), source_code))

    target_code_gt = [];
    features = []
    for i in range(len(source_code_gt)):
        if source_code_gt[i] is None:
            features.append(None)
            target_code_gt.append(None)
            continue
        
        target_code_gt.append(target_code[i])
        features.append(source_text[i])

    target_code_gt = list(map(lambda x: get_code_idx(x, target_idx), target_code_gt))
    
    ## lets filter out None values

    target_code_gt = list(x for x in target_code_gt if x is not None)
    features = list(x for x in features if x is not None)

    embeddings = get_embeddings(features).to(device)
    target_idx = toTensors(list(itertools.repeat(target_idx, len(target_code_gt)))).to(device)

    output = model(embeddings, target_idx)

    _, predicted = torch.max(output, 1)

    predicted = predicted.cpu().data.numpy()


    df = pd.DataFrame({'source': get_code(source_code_gt, 0), 'gt': get_code(target_code_gt, 1), 'predicted': get_code(predicted, 1)})
    df.to_csv('results.csv', header=None, index=None)

    correct = 0
    for i in range(len(target_code_gt)):
        if target_code_gt[i] == predicted[i]:
            correct = correct + 1
    
    print(f'accuracy: {correct / len(target_code_gt)}')
