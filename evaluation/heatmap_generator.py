import os
import torch
import matplotlib.pyplot as plt

model = 'HINN'
filelist = sorted([file for file in os.listdir(f'prediction') if file.endswith('.pt')])
for idx in range(len(filelist)):
    name = filelist[idx].replace('.pt','')
    file = torch.load(f'prediction/{name}.pt')
    file = torch.tensor(file,dtype=torch.float)
    border = torch.abs(file).max().item()*0.4
    for i in range(100):
        for j in range(file.shape[1]):
            plt.figure(figsize=(4.8,3.6))
            plt.imshow(file[i,j,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'evaluation/SSIM/{model}/{name}_{int(2*i+j)}.png')
            plt.close()
