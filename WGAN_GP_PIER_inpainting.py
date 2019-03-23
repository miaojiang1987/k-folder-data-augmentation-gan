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
parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
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
n_critic = 5               # the number of iterations of the critic per generator iteration

        # load dataset
        #self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        #data = self.data_loader.__iter__().__next__()[0]

        # networks init
G = generator(input_dim=z_dim, output_dim=3, input_size=input_size)
D = discriminator(input_dim=3, output_dim=1, input_size=input_size)
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))       
        
G.cuda()
D.cuda()

G.load_state_dict(torch.load(args.netG_path))
D.load_state_dict(torch.load(args.netD_path))

print('---------- Networks architecture -------------')
utils.print_network(G)
utils.print_network(D)
print('-----------------------------------------------')

     
#sample_z_ = torch.rand((batch_size, z_dim))
#sample_z_ = sample_z_.cuda()
#sample_batch_z = torch.rand((batch_size, z_dim)).cuda()
#print(sample_batch_z.size())
#fake_batch_images = G(sample_batch_z)
#tvutils.save_image(fake_batch_images.detach(), "%s/sample_from_generator_with_seed_{%d}.png" %(args.output_dir, 1), normalize=True)


output_dir = args.output_dir
source_imagedir = os.path.join(output_dir, "source_images")
masked_imagedir = os.path.join(output_dir, "masked_images")
impainted_imagedir = os.path.join(output_dir, "impainted_images")
os.makedirs(source_imagedir, exist_ok=True)
os.makedirs(masked_imagedir,exist_ok=True)
os.makedirs(impainted_imagedir,exist_ok=True)

    # 总共需要修复多少图片
num_images = len(args.aligned_images)
    # 总共可以分为多少的batch来进行处理
num_batches = int(np.ceil(num_images / batch_size))
image_shape = [3,64,64]
device = torch.device("cuda:0")

for idx in range(num_batches):
        # 对于每一个batch的图片进行如下处理
    lidx = idx * args.batch_size
    hidx = min(num_images, (idx + 1) * batch_size)
    realBatchSize = hidx - lidx

    batch_images = [get_tensor_image(imgpath) for imgpath in args.aligned_images[lidx:hidx]]
    batch_images = torch.stack(batch_images).cuda()
        # if realBatchSize < args.batch_size:
        #     print("number of batch images : ", realBatchSize)
        #     # 如果需要修补的图片没有一个batch那么多，用0来填充
        #     batch_images = np.pad(batch_images, [(0, args.batch_size - realBatchSize), (0, 0), (0, 0), (0, 0)], "constant")
        #     batch_images = batch_images.astype(np.float32)
        
        # 输入的原始图片已经准备好，开始准备mask
        # 暂时只提供中心mask
    mask = torch.ones(size=image_shape).to(device)
    imageCenterScale = 0.3
    lm = int(64 * imageCenterScale)
    hm = int(64 * (1 - imageCenterScale))
        # 将图像中心mask为0
    mask[:,lm:hm, lm:hm] = 0.0
    masked_batch_images = torch.mul(batch_images, mask).to(device)

        # 先保存一下原始图片和masked图片
    save_tensor_images(batch_images.detach(),
                   os.path.join(source_imagedir,"source_image_batch_{}.png".format(idx)))
    
    save_tensor_images(masked_batch_images.detach(), os.path.join(masked_imagedir, "masked_image_batch_{}.png".format(idx)))

       
    z_hat = torch.rand(size=[realBatchSize,z_dim],dtype=torch.float32,requires_grad=True,device=device)
    #z_hat = torch.rand((batch_size, z_dim)).cuda()
    #z_hat.requires_grad=True
    z_hat.data.mul_(2.0).sub_(1.0)
    #opt = optim.Adam([z_hat],lr=args.lr)       
    print("start impainting iteration for batch : {}".format(idx))
    v=torch.tensor(0,dtype=torch.float32,device=device)
    m=torch.tensor(0,dtype=torch.float32,device=device)
        
    for iteration in range(args.num_iters):
            # 对每一个batch的图像分别迭代impainting
        if z_hat.grad is not None:
            z_hat.grad.data.zero_()
        #G.zero_grad()
        #D.zero_grad()
        #print(z_hat.size())
        batch_images_g = G(z_hat)
        batch_images_g_masked = torch.mul(batch_images_g,mask) 
        impainting_images = torch.mul(batch_images_g,(1-mask))+masked_batch_images
        
        if iteration % 100==0:
                # 保存impainting 图片结果
            print("\nsaving impainted images for batch: {} , iteration:{}".format(idx,iteration))
            save_tensor_images(impainting_images.detach(), os.path.join(impainted_imagedir,"impainted_image_batch_{}_iteration_{}.png".format(idx,iteration)))
        
        #loss_context = torch.norm(
        #        (masked_batch_images-batch_images_g_masked),p=1)
        #dis_output = discriminator(impainting_images)
#             print(dis_output)
        #batch_labels = torch.full((realBatchSize,), 1, device=device)
        #loss_perceptual = discriminator(dis_output)
            
        #total_loss = loss_context + lamd*loss_perceptual
        #print("\r batch {} : iteration : {:4} , context_loss:{:.4f},percptual_loss:{:4f}".format(idx,iteration,loss_context,loss_perceptual),end="")
        #total_loss.backward()
        #opt.step()


#    if __name__ == "__main__":
#    	predict()
#        print("Sampling Done! Image saved at %s/sample_from_generator_with_seed_{%d}.png" % ((args.output_dir, args.random_seed)))
