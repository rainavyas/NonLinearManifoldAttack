'''
Perform PGD attack on data points
Save the attacked data points to a file

delta (attack size) limits perturbation in l2-norm
'''

import torch
import torch.nn as nn
import sys
import os
import argparse
from models import FFN
import numpy as np

def pgd_attack(x, y, model, criterion, delta, steps=1):
    '''
    x: torch.Tensor [BxH]
    y: torch.Tensor [B]
    
    Perform Projected Gradient Descent attack
    '''
    model.eval()
    s = nn.Sigmoid()
    x_attacked = x+0
    for step in range(steps):
        model.zero_grad()
        x_attacked.requires_grad = True
        x_attacked.grad = 0
        x_attacked.retain_grad()
        pred = s(model(x_attacked))
        loss = criterion(pred, y)
        loss.backward()

        with torch.no_grad():
            new_pos = x_attacked + x_attacked.grad
            x_attacked = x + delta*(new_pos-x)/(torch.norm((new_pos-x), dim=-1).unsqueeze(1).repeat(1, x.size(-1)))
        import pdb; pdb.set_trace()
    return x_attacked

def fgsm_attack(x, y, model, criterion, delta):
    '''
    x: torch.Tensor [Bx3]
    y: torch.Tensor [B]
    
    Perform Finite Gradient Sign Method attack
    '''
    model.eval()
    s = nn.Sigmoid()
    x.requires_grad = True
    x.retain_grad()
    pred = s(model(x))
    loss = criterion(pred, y)
    loss.backward()

    x_attacked = x + delta*torch.sign(x.grad)
    # import pdb; pdb.set_trace()
    return 

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify model th file path')
    commandLineParser.add_argument('ARCH', type=str, help='ffn')
    commandLineParser.add_argument('ATTACK', type=str, help='pgd')
    commandLineParser.add_argument('ORIG_DATA', type=str, help='path to test.npy file')
    commandLineParser.add_argument('OUT_DATA', type=str, help='path to dir to save attacked test.npy file')
    commandLineParser.add_argument('--num_hidden_layers', type=int, default=1, help="number of hidden layers")
    commandLineParser.add_argument('--hidden_layer_size', type=int, default=10, help="size of hidden layers")
    commandLineParser.add_argument('--delta', type=float, default=0.1, help="attack size")
    commandLineParser.add_argument('--PGD_steps', type=int, default=3, help="PGD iterations")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the data as tensors
    with open(args.ORIG_DATA, 'rb') as f:
        data = np.load(f)
    data = torch.from_numpy(data)
    x = data[:,:-1].type(torch.FloatTensor)
    y = data[:,-1].type(torch.FloatTensor)

    # Initialise classifier
    model = FFN(num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size, inp_dim=x.size(1))
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))

    # Criterion
    criterion = nn.BCELoss()

    # Attack
    if args.ATTACK == 'pgd':
        x_attacked = pgd_attack(x, y, model, criterion, args.delta, args.PGD_steps)
    
    # Save the attacked data
    model_filebase = args.MODEL
    model_filebase = model_filebase.split('/')[-1][:-4]
    np.save(f'{args.OUT_DATA}/test_data_model_{model_filebase}_{args.ATTACK}_delta{args.delta}.npy', x_attacked.cpu().detach().numpy())