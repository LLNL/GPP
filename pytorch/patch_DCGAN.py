import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
import os


from models import *

genPATH = './all_models/generator.model'
discPATH = './all_models/discriminator.model'

'''
This code uses snippets from the PyTorch DCGAN tutorial
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
workers = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Batch size during training
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 25
lr = 0.0002
beta1 = 0.5

ngpu = 2
####### DATASET LOADING ################
cifar = np.load('../mimicGAN/IMAGENET/dataset/cifar-c-intense2.npy')
cifar = np.transpose(cifar,[0,3,1,2])
tensor_x = torch.Tensor(cifar)
my_dataset = TensorDataset(tensor_x) # create your datset
dataloader = DataLoader(my_dataset,shuffle=True,batch_size=batch_size,num_workers=workers)

# Plot some training images
real_batch = next(iter(dataloader))
img_tensor = torch.cat(real_batch, dim=0)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
img_ = vutils.make_grid(img_tensor[:100],nrow=10, padding=1, normalize=True)

vutils.save_image(img_,'outs/gt_sample.png')

netG = Generator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
if os.path.isfile(genPATH):
    print('**** Loading Generator ****')
    netG.load_state_dict(torch.load(genPATH))
    netG.eval()

if os.path.isfile(discPATH):
    print('**** Loading Discriminator ****')
    netD.load_state_dict(torch.load(discPATH))
    netD.eval()

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = 1-2*torch.rand(100, nz,1,1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = 1-2*torch.rand(b_size, nz,1,1,device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_ = vutils.make_grid(fake,nrow=10, padding=1, normalize=True)
            vutils.save_image(img_,'outs/epoch_{}_iters_{}.png'.format(str(epoch).zfill(3),str(i).zfill(3)))
            torch.save(netG.state_dict(), genPATH)
            torch.save(netD.state_dict(), discPATH)


        iters += 1
