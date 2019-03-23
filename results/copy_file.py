import os 
import numpy as np
from PIL import Image
import sys
import shutil
#val=(int)(sys.argv[1])


count=1
names=os.listdir('cifar10/WGAN_GP/sliced/')
#length[i]=len(names)

for j in range(25):
    for name in names:
        shutil.copy2('cifar10/WGAN_GP/sliced/'+name, 'original_cifar/'+(str)(count)+'.png')
        count+=1


