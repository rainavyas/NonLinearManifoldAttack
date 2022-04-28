'''
Train a model for synthetic data classification on non-linear manifold
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tools import AverageMeter, accuracy_binary, get_default_device
from models import FFN
import numpy as np

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(args.seed)

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    s = nn.Sigmoid()
    for i, (x, target) in enumerate(train_loader):

        x = x.to(device)
        target = target.to(device)

        # Forward pass
        pred = s(model(x))
        loss = criterion(pred, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy_binary(pred.data, target)
        accs.update(acc.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')
    return losses.avg

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        s = nn.Sigmoid()
        for i, (x, target) in enumerate(val_loader):

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            pred = s(model(x))
            loss = criterion(pred, target)

            # measure accuracy and record loss
            acc = accuracy_binary(pred.data, target)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

    print(f'Test\t Loss ({losses.avg:.4f})\tAccuracy ({accs.avg:.3f})\n')
    return losses.avg


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify directory to save output th file')
    commandLineParser.add_argument('ARCH', type=str, help='ffn')
    commandLineParser.add_argument('TRAIN_DATA', type=str, help='path to train.npy file')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=2, help="Specify maximum epochs")
    commandLineParser.add_argument('--dev_frac', type=float, default=0.2, help="Fraction of points to be used for dev")
    commandLineParser.add_argument('--lr', type=float, default=0.0001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed for reproducibility")
    commandLineParser.add_argument('--num_hidden_layers', type=int, default=1, help="number of hidden layers")
    commandLineParser.add_argument('--hidden_layer_size', type=int, default=10, help="size of hidden layers")
    commandLineParser.add_argument('--force_cpu', type=str, default='no', help='do you want to use cpu only')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()
    if args.force_cpu == 'yes':
        device = torch.device('cpu')

    # Load the data as tensors
    with open(args.TRAIN_DATA, 'rb') as f:
        data = np.load(f)
    rng = np.random.default_rng()
    rng.shuffle(data)
    num_dev = int(args.dev_frac*len(data))

    dev_data = torch.from_numpy(data[:num_dev])
    x_dev = dev_data[:,:-1].type(torch.FloatTensor)
    y_dev = dev_data[:,-1].type(torch.FloatTensor)

    train_data = torch.from_numpy(data[num_dev:])
    x_train = train_data[:,:-1].type(torch.FloatTensor)
    y_train = train_data[:,-1].type(torch.FloatTensor)

    # Use dataloader to handle batches
    train_ds = TensorDataset(x_train, y_train)
    dev_ds = TensorDataset(x_dev, y_dev)

    train_dl = DataLoader(train_ds, batch_size=args.B, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.B)

    # Initialise classifier
    model = FFN(num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size, inp_dim=x_train.size(1))
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.sch])

    # Criterion
    criterion = nn.BCELoss().to(device)

    best_loss = 1000 # save model after epoch if best
    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        _ = train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate on validation set
        loss_val = eval(dev_dl, model, criterion, device)

        # if best model, save it
        train_data_info = args.TRAIN_DATA
        train_data_info = train_data_info.split('/')[-1][:-4]
        model_name = f'{args.OUT}/{args.ARCH}_layers{args.num_hidden_layers}_hsize_{args.hidden_layer_size}_{train_data_info}.th'
        if loss_val < best_loss:
            state = model.state_dict()
            torch.save(state, model_name)
    