# site-packages
import os
import torch
import numpy as np
from scipy.stats import entropy
from torchvision.models.inception import inception_v3

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

# modify inception_v3
class VelocityInceptionV3(torch.nn.Module):
    def __init__(self,cate_num):
        super(VelocityInceptionV3,self).__init__()
        self.baseline = inception_v3(weights=None,init_weights=True)
        self.baseline.Conv2d_1a_3x3.conv = torch.nn.Conv2d(in_channels=10,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False)
        self.baseline.fc = torch.nn.Linear(in_features=2048,out_features=cate_num)

    def forward(self,velocity):
        classification = self.baseline(velocity)
        return classification

# inception score
def inception_score(folder,device,splits):
    data_list = VelocityList(dataset=f'evaluation/IS&FID/{folder}')
    data_loader = torch.utils.data.DataLoader(data_list,batch_size=32,shuffle=False,num_workers=4)
    file_num = data_list.__len__()
    model = VelocityInceptionV3(6)
    checkpoint = f'evaluation/velocity_inception_v3.pth'
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint)
        model.load_state_dict(parameter)
        print('Checkpoint loaded!')
    model.to(device)
    model.eval()
    prediction = torch.tensor([],dtype=torch.float,device=device)
    for _,velocity in enumerate(data_loader):
        velocity = velocity.to(device)
        classification = model(velocity)
        classification = torch.nn.functional.softmax(classification,dim=1)
        prediction = torch.cat([prediction,classification],dim=0)
    prediction = prediction.detach().cpu().numpy()
    scores = []
    for i in range(splits):
        part = prediction[i*(file_num//splits):(i+1)*(file_num//splits),:]
        py = np.mean(part,axis=0)
        score = []
        for j in range(part.shape[0]):
            pyx = part[j,:]
            score.append(entropy(pyx,py))
        scores.append(np.exp(np.mean(score)))
    print('Inception Score:',scores)
    return np.mean(scores),np.std(scores)

folder = 'HINN'
is_mean,is_std = inception_score(folder,device,20)
print(f'{folder} Mean: {is_mean}  Std: {is_std}')
