import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import torch.nn.functional as nnf

import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.transform import rescale, resize
from skimage import io

from models import *
# I_x = I_y = 1024
I_y = 768
# I_x = 1536
I_x = 1152
d_x = d_y = 32
dim_x = d_x*d_y
batch_size = (I_x*I_y)//(dim_x)
n_measure = 0.01
lr_factor = 1.0#*batch_size//64
nz = 100

dim_phi = int(n_measure*dim_x)
nIter = 5001
n_img_plot_x = I_x//d_x
n_img_plot_y = I_y//d_y
workers = 2
ngpu = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iters = np.array(np.geomspace(10,10,nIter),dtype=int)
fname = '../gpp-cs/test_images/{}.jpg'.format('color_leapord')

# x_test = Image.open(fname).convert(mode='L').resize((I_x,I_y))
image = io.imread(fname)
x_test = resize(image, (I_x, I_y),anti_aliasing=True,preserve_range=True,mode='reflect')
x_test_ = np.array(x_test)/np.max(x_test)
print(x_test_.shape)

# x_test_ = 2*x_test_-1
x_test = []
for i in range(n_img_plot_x):
    for j in range(n_img_plot_y):
        _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
        x_test.append(_x)

x_test = np.array(x_test)
# x_test = np.expand_dims(x_test,3)
print(x_test.shape)
test_images = torch.Tensor(np.transpose(x_test[:batch_size,:,:,:],[0,3,1,2]))
# vutils.imsave(test_images,[n_img_plot_x,n_img_plot_y],'cs_outs/gt_sample.png')
img_ = vutils.make_grid(test_images,nrow=n_img_plot_y, padding=0, normalize=True)
vutils.save_image(img_,'outs/gt_sample.png')

genPATH = './all_models/generator.model'
discPATH = './all_models/discriminator.model'
# Number of workers for dataloader



netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)
print(netG)

if os.path.isfile(genPATH):
    print('**** Loading Generator ****')
    netG.load_state_dict(torch.load(genPATH))
    netG.eval()

criterion = nn.MSELoss()
# z_prior = torch.rand(batch_size, nz,1,1,requires_grad=True, device=device)
z_prior = torch.zeros(batch_size,nz,1,1,requires_grad=True,device=device)

optimizerZ = optim.RMSprop([z_prior], lr=1e-3)

real_cpu = test_images.to(device)

for iters in range(nIter):
    fake = 0.5*netG(z_prior)+0.5
    fake = nnf.interpolate(fake, size=(d_x, d_y), mode='bilinear', align_corners=False)

    # print(torch.min(fake).item(),torch.min(real_cpu).item(),torch.max(fake).item(),torch.max(real_cpu).item())
    cost = criterion(real_cpu, fake)
    cost.backward()
    optimizerZ.step()
    if (iters % 250 == 0):
        print('Iter: {:d}, Projection Error : {:.3f}'.format(iters,cost.item()))
        with torch.no_grad():
            fake = 0.5*netG(z_prior).detach().cpu() + 0.5
        img_ = vutils.make_grid(fake,nrow=n_img_plot_y, padding=0)
        vutils.save_image(img_,'outs/iters_{}.png'.format(str(iters).zfill(4)))
        
