import os 
import numpy as np
from PIL import Image
import sys
import shutil
#val=(int)(sys.argv[1])


count=1
names=os.listdir('cifar10_1/WGAN_GP/sliced/')
#length[i]=len(names)

for j in range(5):
    for name in names:
        shutil.copy2('cifar10_1/WGAN_GP/sliced/'+name, 'together/'+(str)(count)+'.png')
        count+=1

names=os.listdir('cifar10_2/WGAN_GP/sliced/')
for j in range(5):
    for name in names:
        shutil.copy2('cifar10_2/WGAN_GP/sliced/'+name, 'together/'+(str)(count)+'.png')
        count+=1

names=os.listdir('cifar10_3/WGAN_GP/sliced/')
for j in range(5):
    for name in names:
        shutil.copy2('cifar10_3/WGAN_GP/sliced/'+name, 'together/'+(str)(count)+'.png')
        count+=1

names=os.listdir('cifar10_4/WGAN_GP/sliced/')
for j in range(5):
    for name in names:
        shutil.copy2('cifar10_4/WGAN_GP/sliced/'+name, 'together/'+(str)(count)+'.png')
        count+=1

names=os.listdir('cifar10_5/WGAN_GP/sliced/')
for j in range(5):
    for name in names:
        shutil.copy2('cifar10_5/WGAN_GP/sliced/'+name, 'together/'+(str)(count)+'.png')
        count+=1


