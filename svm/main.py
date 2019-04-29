import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets.samples_generator import make_blobs

from model import LinearSVM


def get_data():
    X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    return X, Y


def to_np(x):
    return x.data.cpu().numpy()


def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]]
            y = Y[perm[i : i + args.batchsize]]

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            output = model(x)

            loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))  # hinge loss
            loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
            loss.backward()
            optimizer.step()

            sum_loss += to_np(loss)

        print("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N))


def visualize(X, Y, model):
    W = to_np(model.fc.weight[0])
    b = to_np(model.fc.bias[0])

    delta = 0.01
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.)] = 4
    z[np.where((z > 0.) & (z <= 1.))] = 3
    z[np.where((z > -1.) & (z <= 0.))] = 2
    z[np.where(z <= -1.)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()


def main(args):
    X, Y = get_data()
    X = (X - X.mean()) / X.std()
    Y[np.where(Y == 0)] = -1

    model = LinearSVM()
    if torch.cuda.is_available():
        model.cuda()

    train(X, Y, model, args)
    visualize(X, Y, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    main(args)
