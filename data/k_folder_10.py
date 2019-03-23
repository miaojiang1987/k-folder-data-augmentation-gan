import os
from  PIL import Image
import numpy as np

names=os.listdir('cifar/cifar')
length=len(names)
chunk=length//10

for i in range(10):
    for j in range(i*chunk,(i+1)*chunk):
        first_name=names[j]
        im=Image.open('cifar/cifar/'+first_name)
        im2arr=np.array(im,dtype=np.float32)
        im = Image.fromarray(im2arr.astype('uint8'))
        for k in range(5):
            if i==k:
                continue
            else:
                path='cifar_'+(str)(k+1)+'/cifar_'+(str)(k+1)
                if not os.path.exists(path):
                    os.makedirs(path)
                im.save(path+'/'+(str)(j)+'.png')
    #im.save('cifar10_2/cifar10_2/'+(str)(i)+'.png')
    #im.save('cifar10_3/cifar10_3/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    #im.save('cifar10_1/cifar10_1/'+(str)(i)+'.png')
    #im.save('cifar10_2/cifar10_2/'+(str)(i)+'.png')
    #im.save('cifar10_3/cifar10_3/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    #im.save('cifar10_4/cifar10_4/'+(str)(i)+'.png')
    

