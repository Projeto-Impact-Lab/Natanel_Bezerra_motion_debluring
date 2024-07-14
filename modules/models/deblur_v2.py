import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(Downsample, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(Upsample, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        self.upsample = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat((x, skip), 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64, apply_batchnorm=False)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)
        self.down7 = Downsample(512, 512)
        self.down8 = Downsample(512, 512)

        self.up1 = Upsample(512, 512, apply_dropout=True)
        self.up2 = Upsample(1024, 512, apply_dropout=True)
        self.up3 = Upsample(1024, 512, apply_dropout=True)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.last = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        out = self.last(u7)
        return out


class Deblur():
    def __init__(self):
        self.generator = Generator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)

    def compile(self, optimizer_generator=None, loss_function=None):
        self.optimizer_generator = optimizer_generator or Adam(self.generator.parameters())
        self.loss_function = loss_function or nn.MSELoss()

    def fit(self, dataloader, epochs=1):
        self.generator.train()
        for epoch in range(epochs):
            for batch in dataloader:
                input_image, target = batch
                input_image = input_image.to(self.device)
                target = target.to(self.device)

                self.optimizer_generator.zero_grad()
                gen_output = self.generator(input_image)
                gen_loss = self.loss_function(gen_output, target)
                gen_loss.backward()
                self.optimizer_generator.step()

            print(f'Epoch: {epoch}, Gen Loss: {gen_loss.item()}')
