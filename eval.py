'''
Evaluate a model for synthetic data classification on non-linear manifold
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tools import get_default_device
from models import FFN
import numpy as np
from train import eval

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify model th file path')
    commandLineParser.add_argument('ARCH', type=str, help='ffn')
    commandLineParser.add_argument('TEST_DATA', type=str, help='path to test.npy file')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--num_hidden_layers', type=int, default=1, help="number of hidden layers")
    commandLineParser.add_argument('--hidden_layer_size', type=int, default=10, help="size of hidden layers")
    commandLineParser.add_argument('--force_cpu', type=str, default='no', help='do you want to use cpu only')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()
    if args.force_cpu == 'yes':
        device = torch.device('cpu')

    # Load the data as tensors
    with open(args.TEST_DATA, 'rb') as f:
        data = np.load(f)
    data = torch.from_numpy(data)
    x = data[:,:-1].type(torch.FloatTensor)
    y = data[:,-1].type(torch.FloatTensor)

    # Use dataloader to handle batches
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=args.B)

    # Initialise classifier
    model = FFN(num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size, inp_dim=x.size(1))
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.to(device)

    # Criterion
    criterion = nn.BCELoss().to(device)
    _ = eval(dl, model, criterion, device)