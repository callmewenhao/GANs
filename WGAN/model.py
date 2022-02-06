import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                in_channels, features_d, kernel_size=4, stride=2,padding=1,
            ),
            nn.LeakyReLU(0.2),
            # blocks
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid(), 不在使用sigmoid()函数
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, in_channels, img_channels, features_g):
        super().__init__()
        self.net = nn.Sequential(
            self._block(in_channels, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def main():
    N, in_channles, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn(N, in_channles, H, W)
    disc = Discriminator(in_channles, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed!"
    gen = Generator(noise_dim, in_channles, 8)
    z = torch.randn(N, noise_dim, 1, 1)
    assert gen(z).shape == (N, in_channles, H, W), "Generator test failed!"


if __name__ == "__main__":
    main()







