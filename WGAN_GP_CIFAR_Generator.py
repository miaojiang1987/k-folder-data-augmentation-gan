import utils, torch, time, os, pickle
import os
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from dataloader import dataloader
import torchvision.utils as tvutils

from imageio import imread
import torchvision.transforms as transforms

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


def get_image_from_path(path):

    img = imread(path)
    return img

def image_to_tensor(image):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans(image)

def get_tensor_image(path):
    img = get_image_from_path(path)
    return image_to_tensor(img)

def save_tensor_images(images,filename,nrow=None,normalize=True):
    if not nrow:
        tvutils.save_image(images, filename, normalize=normalize)
    else:
        tvutils.save_image(images, filename, normalize=normalize,nrow=nrow)

#desc = "Pytorch implementation of GAN collections"
parser = argparse.ArgumentParser()

parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN','WGAN_GP_PIER'],
                        help='The type of GAN')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar10_1','cifar10_2','cifar10_3','cifar10_4','cifar10_5','cifar100', 'svhn', 'stl10', 'lsun-bed','pier'],
                        help='The name of dataset')
parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
parser.add_argument("--netG_path", required=True,help="file path to G network to restore Generator for Sampling")
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--input_size', type=int, default=64, help='The size of input image')
parser.add_argument('--save_dir', type=str, default='models',help='Directory name to save the model')
parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
parser.add_argument('--lrG', type=float, default=0.0002)
parser.add_argument('--lrD', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--benchmark_mode', type=bool, default=True)
parser.add_argument("--output_dir", default=".", help="directory path to output intermediate images and impainted images")
parser.add_argument("--num_iters", type=int, default=1000, help="number of iterations form image impainting optimization")
parser.add_argument("--aligned_images", type=str, nargs="+",help="input source aligned images for mask and impainting")
parser.add_argument("--netD_path", required=True, help="file path to D network,to impainting for more real image")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate for adjusting gradient of z, default is 0.01")
args = parser.parse_args()
epoch = args.epoch
sample_num = 100
batch_size = args.batch_size
save_dir = args.save_dir
result_dir = args.result_dir
dataset = args.dataset
log_dir = args.log_dir
gpu_mode = args.gpu_mode
model_name = args.gan_type
input_size = args.input_size
z_dim = 62
lambda_ = 10
n_critic = 5

G = generator(input_dim=z_dim, output_dim=3, input_size=input_size)
D = discriminator(input_dim=3, output_dim=1, input_size=input_size)
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))       
        
G.cuda()
D.cuda()

G.load_state_dict(torch.load(args.netG_path))
#D.load_state_dict(torch.load(args.netD_path))

print('---------- Networks architecture -------------')
utils.print_network(G)
utils.print_network(D)
print('-----------------------------------------------')


G.load_state_dict(torch.load(args.netG_path))

z_ = torch.rand((batch_size, z_dim)).cuda()
fake_batch_images = generator(z_)
tvutils.save_image(fake_batch_images.detach(), "%s/sample_from_generator_with_seed_{%d}.png" % (args.output_dir, args.random_seed), normalize=True)
