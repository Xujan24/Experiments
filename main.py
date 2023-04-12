import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# from sentence_transformers import SentenceTransformer
from utils.load_dataset import LoadDataset
from utils.helpers import get_max_num_codes, toTensors, toTensorTensors, get_code_embds_from_idx
from model.decoder import Decoder
from typing import Literal
import argparse


def train(start_epoch: int, epochs: int, model: Decoder, data_loader: DataLoader, device: Literal, loss_fn, optimizer, scheduler, path: str):
    lam = 1
    for t in range(start_epoch, epochs):
        for batch, (features, versions, labels) in enumerate(data_loader):
            _versions = toTensorTensors(versions)
            features, _versions, labels = features.to(device), _versions.to(device), labels.to(device)

            # denoising auto encoder part
            # the input feature hasn't been corrupted (so equivalent to model pre-training)
            model.train()
            pred = model(features, _versions)
            loss1 = loss_fn(pred, labels)

            # back-translation part
            model.eval()
            intermediate_versions = toTensors(list(map(lambda x: 1 if x==0 else 0, versions))).to(device)
            intermediate_pred = model(features, intermediate_versions)
            intermeditate_outs = np.argmax(intermediate_pred.cpu().data.numpy(), axis=-1)
            intermediate_features = get_code_embds_from_idx(intermeditate_outs, intermeditate_outs).to(device)

            model.train()
            final_pred = model(intermediate_features, _versions)
            loss2 = loss_fn(final_pred, labels)

            # make sure we alternate the training in every epoch
            # trying out some form of cyclical manner (roughly in every 'x' epoch the loss oscilates 
            # between loss1 and loss2 (two extremes), with both contributing in the intermediate steps)
            lam = 1 if t % 2 == 0 else 0

            # final loss
            loss = lam * loss1 + (1 - lam) * loss2

            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 2 == 0:
                loss = loss.item()
                print(f"[epoch: {t:>3d}] loss:{loss:>7f}")
        
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)

        scheduler.step()



if __name__ == "__main__":
    n_input = 768
    n_codes = get_max_num_codes()
    learning_rate = 1e-4
    batch_size = 64
    start_epoch = 0
    epochs = 1000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trained_path = './trained/state_new.pth'

    parser = argparse.ArgumentParser(description="options")
    parser.add_argument('-r', '--resume', type=bool)
    args = parser.parse_args()

    model = Decoder(n_input, n_codes)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.95)


    ## in case of resuming from previous checkpoint;
    if args.resume and os.path.exists(trained_path):
        checkpoint = torch.load(trained_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    data_loader = DataLoader(LoadDataset(), batch_size=120, shuffle=True)

    train(start_epoch, epochs, model, data_loader, device,
          loss_fn, optimizer, scheduler, trained_path)
