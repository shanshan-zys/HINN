# site-packages
import os
import torch
import numpy as np
from scipy import linalg
from torchvision.models.inception import inception_v3,InceptionA,InceptionC,InceptionE

# cuda support
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
print('Device:',device)

# velocity dataset
class VelocityList(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.filelist = sorted([f'{dataset}/{file}' for file in os.listdir(dataset) if file.endswith('.pt')])
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        velocity = torch.load(self.filelist[idx])
        velocity = torch.tensor(velocity,dtype=torch.float)
        velocity = torch.clamp(velocity,-50,50)
        scaler = torch.abs(velocity).max()
        velocity = velocity*50/scaler
        return velocity
    
class FIDInceptionA(InceptionA):
    def __init__(self,in_channels,pool_features):
        super(FIDInceptionA,self).__init__(in_channels,pool_features)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = torch.nn.functional.avg_pool2d(x,kernel_size=3,stride=1,padding=1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch5x5,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class FIDInceptionC(InceptionC):
    def __init__(self,in_channels,channels_7x7):
        super(FIDInceptionC,self).__init__(in_channels,channels_7x7)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = torch.nn.functional.avg_pool2d(x,kernel_size=3,stride=1,padding=1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch7x7,branch7x7dbl,branch_pool]
        return torch.cat(outputs,1)

class FIDInceptionE_1(InceptionE):
    def __init__(self,in_channels):
        super(FIDInceptionE_1,self).__init__(in_channels)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3),self.branch3x3_2b(branch3x3),]
        branch3x3 = torch.cat(branch3x3,1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),self.branch3x3dbl_3b(branch3x3dbl),]
        branch3x3dbl = torch.cat(branch3x3dbl,1)
        branch_pool = torch.nn.functional.avg_pool2d(x,kernel_size=3,stride=1,padding=1,count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch3x3,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

class FIDInceptionE_2(InceptionE):
    def __init__(self,in_channels):
        super(FIDInceptionE_2,self).__init__(in_channels)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3),self.branch3x3_2b(branch3x3),]
        branch3x3 = torch.cat(branch3x3,1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),self.branch3x3dbl_3b(branch3x3dbl),]
        branch3x3dbl = torch.cat(branch3x3dbl,1)
        branch_pool = torch.nn.functional.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch3x3,branch3x3dbl,branch_pool]
        return torch.cat(outputs,1)

# modify fid_inception_v3
class VelocityFIDInceptionV3(torch.nn.Module):
    def __init__(self,cate_num):
        super(VelocityFIDInceptionV3,self).__init__()
        self.baseline = inception_v3(weights=None,init_weights=True)
        self.baseline.Conv2d_1a_3x3.conv = torch.nn.Conv2d(in_channels=10,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False)
        self.baseline.Mixed_5b = FIDInceptionA(192,pool_features=32)
        self.baseline.Mixed_5c = FIDInceptionA(256,pool_features=64)
        self.baseline.Mixed_5d = FIDInceptionA(288,pool_features=64)
        self.baseline.Mixed_6b = FIDInceptionC(768,channels_7x7=128)
        self.baseline.Mixed_6c = FIDInceptionC(768,channels_7x7=160)
        self.baseline.Mixed_6d = FIDInceptionC(768,channels_7x7=160)
        self.baseline.Mixed_6e = FIDInceptionC(768,channels_7x7=192)
        self.baseline.Mixed_7b = FIDInceptionE_1(1280)
        self.baseline.Mixed_7c = FIDInceptionE_2(2048)
        self.baseline.fc = torch.nn.Linear(in_features=2048,out_features=cate_num)

    def forward(self,velocity):
        classification = self.baseline(velocity)
        return classification
    
# frechet inception distance
def frechet_inception_distance(folder1,folder2,device):
    groundtruth_list = VelocityList(dataset=f'evaluation/IS&FID/{folder1}')
    groundtruth_loader = torch.utils.data.DataLoader(groundtruth_list,batch_size=32,shuffle=False,num_workers=4)
    generation_list = VelocityList(dataset=f'evaluation/IS&FID/{folder2}')
    generation_loader = torch.utils.data.DataLoader(generation_list,batch_size=32,shuffle=False,num_workers=4)
    model = VelocityFIDInceptionV3(6)
    checkpoint = f'evaluation/velocity_fid_inception_v3.pth'
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint)
        model.load_state_dict(parameter)
        print('Checkpoint loaded!')
    model.to(device)
    model.eval()
    prediction1 = torch.tensor([],dtype=torch.float,device=device)
    prediction2 = torch.tensor([],dtype=torch.float,device=device)
    for _,velocity in enumerate(groundtruth_loader):
        velocity = velocity.to(device)
        classification = model(velocity)
        classification = torch.nn.functional.softmax(classification,dim=1)
        prediction1 = torch.cat([prediction1,classification],dim=0)
    for _,velocity in enumerate(generation_loader):
        velocity = velocity.to(device)
        classification = model(velocity)
        classification = torch.nn.functional.softmax(classification,dim=1)
        prediction2 = torch.cat([prediction2,classification],dim=0)
    prediction1 = prediction1.detach().cpu().numpy()
    prediction2 = prediction2.detach().cpu().numpy()
    mu1,sigma1 = np.mean(prediction1,axis=0),np.cov(prediction1,rowvar=False)
    mu2,sigma2 = np.mean(prediction2,axis=0),np.cov(prediction2,rowvar=False)
    difference = mu1-mu2
    covmean,_ = linalg.sqrtm(sigma1.dot(sigma2),disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0])*(1e-6)
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag,0,atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = difference.dot(difference)+np.trace(sigma1)+np.trace(sigma2)-2*tr_covmean
    return fid

folder1,folder2 = 'GT','HINN'
fid = frechet_inception_distance(folder1,folder2,device)
print(f'{folder2} FID: {fid}')