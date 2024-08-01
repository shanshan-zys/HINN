# site-packeges
import os
import time
import torch
from hinn import HINN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

# cuda support
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('Device:',device)

# velocity dataset
class VelocityList(torch.utils.data.Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.filelist = sorted([file for file in os.listdir(folder) if file.endswith('.pt')])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,idx):
        file = os.path.join(self.folder,self.filelist[idx])
        velocity = torch.load(file)
        velocity = torch.tensor(velocity[0:frame,:,:,:],dtype=torch.float)
        velocity = torch.clamp(velocity,-50,50)
        scaler = torch.abs(velocity).max()
        velocity = velocity*50/scaler
        return self.filelist[idx],velocity

# train
def train(pattern,type,device,epochs,steps):
    global frame,height,width
    frame,height,width = 100,360,480
    # dataset
    data_list = VelocityList(folder=f'data/{pattern}/{type}')
    train_split,_ = train_test_split(data_list.filelist,train_size=0.7,test_size=0.3,random_state=1234)
    train_list = VelocityList(folder=f'data/{pattern}/{type}')
    train_list.filelist = train_split
    train_loader = torch.utils.data.DataLoader(train_list,batch_size=3,shuffle=True)
    print('Train List:',train_list.filelist)
    # model
    model = HINN(height=height,width=width,device=device)
    checkpoint = f'checkpoint/checkpoint_{pattern}_{type}_{epochs}.pth'
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device)
        model.load_state_dict(parameter)
        print('Checkpoint loaded!')
    model.to(device)
    loss_func = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 betas=(0.9,0.999),
                                 eps=1e-8,
                                 weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[500,1000,5000],gamma=0.1)
    # train
    start_time = time.time()
    for epoch in range(steps):
        model.train()
        print(f'Iter {epoch+epochs}:')
        with open(f'loss/loss_{pattern}_{type}.txt','a') as file:
            file.write(f'Iter {epoch+epochs}:\n')
        for batch,(_,groundtruth) in enumerate(train_loader):
            groundtruth = groundtruth.to(device)
            input = groundtruth[:,0,:,:,:]
            loss_total = 0
            for i in range(frame-1):
                optimizer.zero_grad()
                output = model(input)
                target = groundtruth[:,i+1,:,:,:]
                loss = loss_func(target,output)
                loss_total += loss.item()
                with torch.no_grad():
                    input = output.clone()
                loss.backward()
                optimizer.step()
            print(f'Batch {batch}, loss: {loss_total}')
            with open(f'loss/loss_{pattern}_{type}.txt','a') as file:
                file.write(f'Batch {batch}, loss: {loss_total}\n')
        print('')
        with open(f'loss/loss_{pattern}_{type}.txt','a') as file:
            file.write(f'\n')
        scheduler.step()
    elapsed = time.time()-start_time
    print(f'Training time: {elapsed}\n')
    with open(f'loss/loss_{pattern}_{type}.txt','a') as file:
        file.write(f'Training time: {elapsed}\n')
    torch.save(model.state_dict(),f'checkpoint/checkpoint_{pattern}_{type}_{epochs+steps}.pth')

# test
def test(pattern,type,device,epochs):
    global frame,height,width
    frame,height,width = 100,360,480
    # dataset
    data_list = VelocityList(folder=f'data/{pattern}/{type}')
    _,test_split = train_test_split(data_list.filelist,train_size=0.7,test_size=0.3,random_state=1234)
    test_list = VelocityList(folder=f'data/{pattern}/{type}')
    test_list.filelist = test_split
    test_loader = torch.utils.data.DataLoader(test_list,batch_size=1,shuffle=False)
    print('Test List:',test_list.filelist)
    # model
    model = HINN(height=height,width=width,device=device)
    checkpoint = f'checkpoint/checkpoint_{pattern}_{type}_{epochs}.pth'
    if os.path.exists(checkpoint):
        parameter = torch.load(checkpoint,map_location=device)
        model.load_state_dict(parameter)
        print('Checkpoint loaded!')
    model.to(device)
    loss_func = torch.nn.SmoothL1Loss()
    # test
    model.eval()
    with torch.no_grad():
        for _,(filename,groundtruth) in enumerate(test_loader):
            filename = filename[0].replace('.pt','')
            groundtruth = groundtruth.squeeze(0).to(device)
            input = groundtruth[0,:,:,:].unsqueeze(0)
            prediction = input
            for i in range(frame-1):
                output = model(input)
                prediction = torch.cat([prediction,output],dim=0)
                input = output.clone()
            loss = loss_func(groundtruth,prediction)
            print(f'Test {filename}, loss: {loss.item()}')
            with open(f'loss/loss.txt','a') as file:
                file.write(f'Test {filename}, loss: {loss.item()}\n')
            torch.save(prediction.detach().cpu().numpy(),f'prediction/{filename}.pt')
            # visualization
            gap = torch.floor(torch.tensor(frame/4-1,dtype=torch.float)).long()
            border = torch.abs(groundtruth).max().item()*0.4
            groundtruth = groundtruth.detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy()
            fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(18,12))
            for i in range(5):
                axes[0,i].imshow(groundtruth[gap*i,0,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[1,i].imshow(prediction[gap*i,0,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[2,i].imshow(groundtruth[gap*i,1,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[3,i].imshow(prediction[gap*i,1,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
            for ax in axes.flatten():
                ax.axis('on')
                ax.set_aspect('auto')
            plt.tight_layout()
            plt.savefig(f'prediction/{filename}.png')
            plt.close()
            """ fig,((r0c0,r0c1),(r1c0,r1c1)) = plt.subplots(nrows=2,ncols=2) 
            def update(t,data,prediction):
                u_star = data[[t],0,:,:].squeeze(0)
                v_star = data[[t],1,:,:].squeeze(0)
                u_pred = prediction[[t],0,:,:].squeeze(0)
                v_pred = prediction[[t],1,:,:].squeeze(0)
                # update
                r0c0.clear()
                r0c0.imshow(u_star,cmap='jet',vmin=-border,vmax=border,origin='upper')
                r1c0.clear()
                r1c0.imshow(u_pred,cmap='jet',vmin=-border,vmax=border,origin='upper')
                r0c1.clear()
                r0c1.imshow(v_star,cmap='jet',vmin=-border,vmax=border,origin='upper')
                r1c1.clear()
                r1c1.imshow(v_pred,cmap='jet',vmin=-border,vmax=border,origin='upper')
            gif = FuncAnimation(fig,update,frames=frame-1,fargs=(groundtruth,prediction),interval=41)
            gif.save(f'prediction/{filename}.gif',writer='pillow',fps=24)
            plt.close() """

epochs,steps = 0,1000
patternlist = sorted(os.listdir('data'))
# train
for pattern in patternlist:
    typelist = sorted(os.listdir(f'data/{pattern}'))
    for type in typelist:
        print('')
        print('Pattern:',pattern)
        print('Type:',type)
        train(pattern,type,device,epochs,steps)

epochs = epochs+steps
# test
for pattern in patternlist:
    typelist = sorted(os.listdir(f'data/{pattern}'))
    for type in typelist:
        checkpoint = f'checkpoint/checkpoint_{pattern}_{type}_{epochs}.pth'
        if os.path.exists(checkpoint):
            print('')
            test(pattern,type,device,epochs)
