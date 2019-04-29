import argparse
import torch

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    main(args)
