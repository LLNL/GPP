import torch.nn as nn
import torch.nn.functional as F

## models taken from PyTorch tutorial page https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu=1,nc=3,ngf=64,nz=100):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(nz,ngf*8*4*4)
        self.bn1 = nn.BatchNorm1d(ngf*8*4*4)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, 1, 4, 2, 1, bias=True),
            nn.Tanh()
        )


    def forward(self, input,nz=100,ngf=64):
        x = input.view(-1,nz)
        x = F.relu(self.bn1(self.fc1(x)))
        x = x.view(-1,ngf*8,4,4)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1,nc=3,ndf=64,nz=100):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x 32 x 32
        # state size. (ndf) x 32 x 32
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*3) x 4 x 4
        nn.Conv2d(ndf * 4, ndf*8, 4, 2, 1, bias=True),
        # nn.Sigmoid()
    )
        self.fcn = nn.Linear(ndf*8*2*2,1)


    def forward(self, input,ndf=64):
        x = self.main(input)
        x = x.view(-1,ndf*8*2*2)
        x = F.sigmoid(self.fcn(x))
        return x
