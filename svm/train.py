import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data import SVMDataset
from svm import LinearSVM


def str2bool(s):
    return s.lower() in ('true', '1', 'yes', 'y')

def main(args):

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print('Using cuda')
        else:
            print('Warning: cuda is available, but you have not choosed to use!')
            torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        if args.cuda:
            print('Error: cuda is unavailable, but you have choosed to use!')

    dataset = SVMDataset(1000, 1.0)
    dataloader = DataLoader(dataset, 32, True)

    model = LinearSVM()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()

    for epc in range(args.epoch):
        epc_loss = 0
        for batch, data in enumerate(dataloader):
            optimizer.zero_grad()
            X, y = data
            y = y.type(torch.FloatTensor)
            loss = torch.mean(torch.clamp(1-model(X).t()*y, min=0))
            loss += args.alpha * torch.mean(model.fc.weight**2)
            epc_loss += loss 
            loss.backward()
            optimizer.step()
        print('Epoch %d loss: %f'%(epc+1, epc_loss))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--alpha', type=float,default = 1.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    main(args)
