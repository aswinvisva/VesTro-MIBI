import datetime
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import utils
import seaborn as sns


def vis_tensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


class VesselNet(nn.Module):
    def __init__(self, n=32):
        super(VesselNet, self).__init__()

        self.n = n
        self.criterion = nn.BCELoss()
        self.device = torch.device("cuda")
        self.writer = SummaryWriter()

        self.encoder = nn.Sequential(
            nn.Conv2d(34, 64, 5),
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(2304, 2048),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),  # b, 32, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=3, padding=1),  # b, 16, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=1),  # b, 8, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 34, 2, stride=2),  # b, 3, 56, 56
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(-1, 2304)

        encoder_output = self.fc(x)
        w = encoder_output.view(-1, 512, 2, 2)

        w = self.decoder(w)

        return w, encoder_output

    def fit(self, train_loader, epochs=1000, show_every=200):
        self.to(self.device)
        model = self.train()

        print(model)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader.iter:
                x = batch_input[0]
                y = batch_input[1]

                y_pred, encoded = model.forward(x)

                loss = self.criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            print("Epoch {}/{}, training loss: {:.5f}".format(epoch + 1, epochs, epoch_loss))

            if epoch % show_every == 0:
                y_pred_numpy = y_pred.cpu().data.numpy()
                y_true_numpy = y.cpu().data.numpy()

                row_i = np.random.choice(y_pred_numpy.shape[0], 1)
                random_pic = y_pred_numpy[row_i, :, :, :]
                random_pic = random_pic.reshape(34, self.n, self.n)

                true = y_true_numpy[row_i, :, :, :]
                true = true.reshape(34, self.n, self.n)

                for i in range(len(random_pic)):
                    cv.imshow("Predicted", random_pic[i] * 255)
                    cv.imshow("True", true[i] * 255)
                    cv.waitKey(0)

    def visualize_filters(self, channels=34):

        print("Visualizing encoder features")
        for layer in range(len(self.encoder)):
            for ch in range(channels):
                try:
                    f = self.encoder[layer].weight.data.cpu().clone()
                    vis_tensor(f, ch=ch, allkernels=False)

                    plt.axis('off')
                    plt.ioff()
                    plt.show()
                except torch.nn.modules.module.ModuleAttributeError:
                    pass

        print("Visualizing decoder features")
        for layer in range(len(self.decoder)):
            for ch in range(channels):
                try:
                    f = self.decoder[layer].weight.data.cpu().clone()
                    vis_tensor(f, ch=ch, allkernels=False)

                    plt.axis('off')
                    plt.ioff()
                    plt.show()
                except torch.nn.modules.module.ModuleAttributeError:
                    pass
