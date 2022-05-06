'''
- Takes in a test file and its equivalent attacked data
- Reports statistics
'''

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import FFN
from non_linear_functions import linear_exp

def get_fooling_rate(x, y, x_attacked, model):
    '''Fraction of correctly classified samples that are misclassified'''
    model.eval()
    with torch.no_grad():
        s = nn.Sigmoid()
        pred_orig = torch.round(s(model(x)))
        pred_attack = torch.round(s(model(x_attacked)))
        mask = torch.eq(pred_orig, y)

        correct_pred_orig = pred_orig[mask]
        correct_pred_attack = pred_attack[mask]
        fool_rate = 1 - (torch.eq(correct_pred_orig, correct_pred_attack).sum()/len(correct_pred_orig))
        return fool_rate
    

def dist_parallel(x, x_attacked, activation):
    '''
        Get perturbation distance along the manifold
    '''
    pert = activation(activation.inverse(x_attacked)) - activation(activation.inverse(x))
    dists = torch.norm(pert, dim=-1)
    
    return dists

def dist_perpendicular(x, x_attacked, activation):
    '''
        Get perturbation distance normal to the manifold
    '''
    parallel_dists = dist_parallel(x, x_attacked, activation)
    overall_dists = torch.norm((x_attacked-x), dim=-1)
    normal_dists = (torch.abs(overall_dists**2 - parallel_dists**2))**0.5
    import pdb; pdb.set_trace()
    return normal_dists

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify model th file path')
    commandLineParser.add_argument('ARCH', type=str, help='ffn')
    commandLineParser.add_argument('ORIG_DATA', type=str, help='path to test.npy file')
    commandLineParser.add_argument('ATTACK_DATA', type=str, help='path to attacked test.npy file')
    commandLineParser.add_argument('--activation', type=str, default='exp', help='non-linear activation')
    commandLineParser.add_argument('--H', type=int, default=6, help='observation space dimensions')
    commandLineParser.add_argument('--num_hidden_layers', type=int, default=1, help="number of hidden layers")
    commandLineParser.add_argument('--hidden_layer_size', type=int, default=10, help="size of hidden layers")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the data as tensors
    with open(args.ORIG_DATA, 'rb') as f:
        data = np.load(f)
    data = torch.from_numpy(data)
    x = data[:,:-1].type(torch.FloatTensor)
    y = data[:,-1].type(torch.FloatTensor)

    with open(args.ATTACK_DATA, 'rb') as f:
        data = np.load(f)
    data = torch.from_numpy(data)
    x_attacked = data.type(torch.FloatTensor)

    # Get the non-linear activation
    if args.activation == 'exp':
        activation = linear_exp(H=args.H)

    # Load classifier
    model = FFN(num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size, inp_dim=x.size(1))
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    
    # Calculate fooling rate
    print(f'Fooling Rate: {get_fooling_rate(x, y, x_attacked, model)}')

    # Calculate average distance to manifold and distance parallel to manifold
    parallel_dists = dist_parallel(x, x_attacked, activation)
    normal_dists = dist_perpendicular(x, x_attacked, activation)
    print(f'Parallel Distance: {torch.mean(parallel_dists)} +- {torch.std(parallel_dists)}')
    print(f'Perpendicular Distance: {torch.mean(normal_dists)} +- {torch.std(normal_dists)}')
