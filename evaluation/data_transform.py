import os
import torch

model = 'HINN'
filelist = sorted([file for file in os.listdir(f'prediction') if file.endswith('.pt')])
for idx in range(len(filelist)):
    name = filelist[idx].replace('.pt','')
    velocity = torch.load(f'prediction/{name}.pt')
    velocity = torch.tensor(velocity,dtype=torch.float)
    frame = velocity.shape[0]
    height = velocity.shape[2]
    width = velocity.shape[3]
    gap = torch.floor(torch.tensor(frame/4-1,dtype=torch.float)).long()
    velocity = velocity.index_select(0,torch.arange(0,frame,gap))
    velocity = velocity.view((-1,height,width))
    print(f'{name}:',velocity.shape)
    torch.save(velocity.detach().cpu().numpy(),f'evaluation/IS&FID/{model}/{name}.pt')
