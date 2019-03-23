import os 
import numpy as np
from PIL import Image
import sys

val=(int)(sys.argv[1])

length=[0 for i in range(5)]
for i in range(5):
    names=os.listdir('../../../../data/celeba_'+(str)(i+1)+'/celeba_'+(str)(i+1))
    length[i]=len(names)
#names=os.listdir('../../../../data/cifar10_2/cifar10_2')
#length_2=len(names)
#names=os.listdir('../../../../data/cifar10_4/cifar10_4')
#length_3=len(names)
#names=os.listdir('../../../../data/cifar10_5/cifar10_5')
#length_5=len(names)

k=1
for i in range(8):
    if k==val:
       k+=1
    first_name='fake_samples__01_0'+str(i+1)+'.png'
    im=Image.open(first_name)
    im2arr=np.array(im,dtype=np.float32)
    r=im2arr[:,:,0]
    g=im2arr[:,:,1]
    b=im2arr[:,:,2]
    noisy_r = np.copy(r).astype(np.float)
    noisy_r += r.std()*0.5*np.random.standard_normal(r.shape)
    noisy_g = np.copy(g).astype(np.float)
    noisy_g += g.std()*0.5*np.random.standard_normal(g.shape)
    noisy_b = np.copy(b).astype(np.float)
    noisy_b += b.std()*0.5*np.random.standard_normal(b.shape)
    noisy_image = np.zeros(im2arr.shape)
    noisy_image[...,0]=noisy_r
    noisy_image[...,1]=noisy_g
    noisy_image[...,2]=noisy_b
    im = Image.fromarray(im2arr.astype('uint8'))
    im2 = im.resize((32, 32), Image.NEAREST) 
    im2.save('../../../../data/celeba_'+(str)(k)+'/celeba_'+(str)(k)+'/m_'+(str)(i+1+length[k-1])+'.png')
    if (i+1)%2==0:
        k+=1
    #im2.save('../../../../data/cifar10_2/cifar10_2/m_'+(str)(i+1+length_2)+'.png')
    #im2.save('../../../../data/cifar10_4/cifar10_4/m_'+(str)(i+1+length_3)+'.png')
    #im2.save('../../../../data/cifar10_5/cifar10_5/m_'+(str)(i+1+length_5)+'.png')


k=1
for i in range(8):
    if k==val:
        k+=1
    first_name='fake_samples__02_0'+str(i+1)+'.png'
    im=Image.open(first_name)
    im2arr=np.array(im,dtype=np.float32)
    r=im2arr[:,:,0]
    g=im2arr[:,:,1]
    b=im2arr[:,:,2]
    noisy_r = np.copy(r).astype(np.float)
    noisy_r += r.std()*0.5*np.random.standard_normal(r.shape)
    noisy_g = np.copy(g).astype(np.float)
    noisy_g += g.std()*0.5*np.random.standard_normal(g.shape)
    noisy_b = np.copy(b).astype(np.float)
    noisy_b += b.std()*0.5*np.random.standard_normal(b.shape)
    noisy_image = np.zeros(im2arr.shape)
    noisy_image[...,0]=noisy_r
    noisy_image[...,1]=noisy_g
    noisy_image[...,2]=noisy_b
    im = Image.fromarray(im2arr.astype('uint8'))
    im2 = im.resize((32, 32), Image.NEAREST) 
    im2.save('../../../../data/celeba_'+(str)(k)+'/celeba_'+(str)(k)+'/m_'+(str)(i+1+length[k-1])+'.png')
    if (i+1)%2==0:
        k+=1
