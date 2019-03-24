import os
from  PIL import Image
import numpy as np
import sys
import shutil

names=os.listdir('celeba/celeba')
length=len(names)
chunk=length//10

for i in range(10):
    for j in range(i*chunk,(i+1)*chunk):
        first_name=names[j]
        #shutil.copy2('celeba/celeba/'+first_name, 'original_celeba/'+(str)(count)+'.png')
        for k in range(10):
            if i==k:
                continue
            else:
                path='celeba_'+(str)(k+1)+'/celeba_'+(str)(k+1)
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copy2('celeba/celeba/'+first_name, path+'/'+(str)(j)+'.png')
    #im.save('cifar10_2/cifar10_2/'+(str)(i)+'.png')
    #im.save('cifar10_3/cifar10_3/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    #im.save('cifar10_1/cifar10_1/'+(str)(i)+'.png')
    #im.save('cifar10_2/cifar10_2/'+(str)(i)+'.png')
    #im.save('cifar10_3/cifar10_3/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    

