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
from skimage.measure import compare_psnr

from models import *

def cs_measure(gt,est,phi):
    n_dim = gt.shape[1]* gt.shape[2]

    y_gt = torch.matmul(gt.view(-1,n_dim),phi)
    y_est = torch.matmul(est.view(-1,n_dim),phi)
    return y_gt,y_est

# I_x = I_y = 1024
I_y = 1152
# I_x = 1536
I_x = 800
d_x = d_y = 32
dim_x = d_x*d_y
batch_size = (I_x*I_y)//(dim_x)
n_measure = 0.1
lr_factor = 1.0#*batch_size//64
nz = 100

dim_phi = int(n_measure*dim_x)
nIter = 5001
n_img_plot_x = I_x//d_x
n_img_plot_y = I_y//d_y
workers = 2
ngpu = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## measurement operator
phi_np = np.random.randn(dim_x,dim_phi)
phi_test = torch.Tensor(phi_np)

iters = np.array(np.geomspace(10,10,nIter),dtype=int)
fname = '../test_images/{}.jpg'.format('color_tiger')

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
img_gt = vutils.make_grid(test_images,nrow=n_img_plot_y, padding=0, normalize=True)
vutils.save_image(img_gt,'outs/gt_sample.png')

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
z_ = 0.*np.random.uniform(-1,1,size=[batch_size, nz,1,1])
z_prior = torch.Tensor(z_).requires_grad_()

# z_prior = torch.zeros(batch_size,nz,1,1,requires_grad=True,device=device)

optimizerZ = optim.RMSprop([z_prior], lr=1e-2)

real_cpu = test_images.to(device)

for iters in range(nIter):
    fake = 0.5*netG(torch.clamp(z_prior,-1.,1.))+0.5
    fake = nnf.interpolate(fake, size=(d_x, d_y), mode='bilinear', align_corners=False)
    cost = 0.

    for i in range(3):
        y_gt,y_est = cs_measure(real_cpu[:,i,:,:],fake[:,i,:,:],phi_test.to(device))
        cost += criterion(y_gt,y_est)

    cost.backward()
    optimizerZ.step()
    if (iters % 50 == 0):
        # print('Measurement dims',y_gt.shape,y_est.shape)

        with torch.no_grad():
            fake = 0.5*netG(torch.clamp(z_prior,-1.,1.)).detach().cpu() + 0.5
            fake2 = nnf.interpolate(fake, size=(d_x, d_y), mode='bilinear', align_corners=False)

        img_ = vutils.make_grid(fake2,nrow=n_img_plot_y, padding=0)
        imggt = np.transpose(img_gt.detach().cpu().numpy(),[1,2,0])
        imgest = np.transpose(img_.detach().cpu().numpy(),[1,2,0])
        psnr = compare_psnr(imggt,imgest,data_range=1.0)
        print('Iter: {:d}, Error: {:.3f}, PSNR: {:.3f}'.format(iters,cost.item(),psnr))

        vutils.save_image(img_,'outs/iters_{}.png'.format(str(iters).zfill(4)))
