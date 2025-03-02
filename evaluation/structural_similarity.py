# site-packages
import os
import cv2
import torch
import numpy as np
from math import exp

# heatmap dataset
class HeatMapList(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.filelist = sorted([f'{dataset}/{file}' for file in os.listdir(dataset) if file.endswith('.png')])
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        image = cv2.imread(self.filelist[idx])
        image = torch.from_numpy(np.rollaxis(image,2)).float()
        return image
    
def gaussian():
    gauss = torch.Tensor([exp(-(x-11//2)**2/float(2*1.5**2)) for x in range(11)])
    return gauss/gauss.sum()

def create_window():
    _1D_window = gaussian().unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(3,1,11,11).contiguous())
    return window

def structural_similarity(folder1,folder2):
    groundtruth_list = HeatMapList(dataset=f'evaluation/SSIM/{folder1}')
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_list,batch_size=32,shuffle=False,num_workers=4)
    generation_list = HeatMapList(dataset=f'evaluation/SSIM/{folder2}')
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    window = create_window()
    ssim_values = []
    for _,(image1,image2) in enumerate(zip(groundtruth_loader,generation_loader)):
        if torch.allclose(image2,255*torch.ones_like(image2)):
            penalty = torch.tensor(0.5)
        else:
            penalty = torch.tensor(0.0)
        mu1 = torch.nn.functional.conv2d(image1,window,padding=11//2,groups=3)
        mu2 = torch.nn.functional.conv2d(image2,window,padding=11//2,groups=3)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = torch.nn.functional.conv2d(image1*image1,window,padding=11//2,groups=3)-mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(image2*image2,window,padding=11//2,groups=3)-mu2_sq
        sigma12 = torch.nn.functional.conv2d(image1*image2,window,padding=11//2,groups=3)-mu1_mu2
        C1,C2 = 0.01**2,0.03**2
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))+penalty
        ssim_values.append(ssim_map.mean().item())
    ssim_mean = sum(ssim_values)/len(ssim_values)
    return ssim_mean

folder1,folder2 = 'GT','HINN'
ssim = structural_similarity(folder1,folder2)
print(f'{folder2} SSIM: {ssim}')
