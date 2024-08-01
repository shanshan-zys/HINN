import os
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

folderlist = sorted(os.listdir('data'))
for category in folderlist:
    typelist = sorted(os.listdir(f'data/{category}'))
    for type in typelist:
        filelist = sorted([file for file in os.listdir(f'data/{category}/{type}') if file.endswith('.mp4')])
        for idx in range(len(filelist)):
            name = filelist[idx].replace('.mp4','')
            if os.path.exists(f'data/{category}/{type}/{name}.pt'):
                continue
            # read video
            video = cv2.VideoCapture(f'data/{category}/{type}/{name}.mp4')
            frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(f'video: {name}')
            with open(f'data/{category}/{category}.txt','a') as file:
                file.write(f'video={name}, frame={frame}, height={height}, width={width}\n')
            # calculate velocity
            _,frame0 = video.read()
            frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
            opticalflow = torch.zeros((frame,2,height,width),dtype=torch.float)
            for i in range(frame):
                _,frame0 = video.read()
                frame2 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(frame1,frame2,None,0.5,5,15,5,5,1.1,0)
                opticalflow[i,0,:,:] = torch.from_numpy(flow[...,0])
                opticalflow[i,1,:,:] = torch.from_numpy(flow[...,1])
                frame1 = frame2
            video.release()
            # save velocity
            print(opticalflow.shape)
            torch.save(opticalflow.detach().cpu().numpy(),f'data/{category}/{type}/{name}.pt')
            """ # visualize velocity
            gap = torch.floor(torch.tensor(frame/9-1,dtype=torch.float)).long()
            border = torch.abs(opticalflow).max().item()*0.4
            fig,axes = plt.subplots(nrows=4,ncols=5,figsize=(18,12))
            for k in range(5):
                axes[0,k].imshow(opticalflow[gap*k,0,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[1,k].imshow(opticalflow[gap*(k+1),0,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[2,k].imshow(opticalflow[gap*k,1,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
                axes[3,k].imshow(opticalflow[gap*(k+1),1,:,:],cmap='jet',vmin=-border,vmax=border,origin='upper')
            for ax in axes.flatten():
                ax.axis('on')
                ax.set_aspect('auto')
            plt.tight_layout()
            plt.savefig(f'data/{category}/{type}/{name}.png')
            plt.close()
            fig,(r0c0,r0c1) = plt.subplots(nrows=1,ncols=2)
            def update(t,opticalflow):
                u_t = opticalflow[[t],0,:,:].squeeze(0).squeeze(0)
                v_t = opticalflow[[t],1,:,:].squeeze(0).squeeze(0)
                r0c0.clear()
                r0c0.imshow(u_t,cmap='jet',vmin=-border,vmax=border,origin='upper')
                r0c1.clear()
                r0c1.imshow(v_t,cmap='jet',vmin=-border,vmax=border,origin='upper')
                plt.tight_layout()
            gif = FuncAnimation(fig,update,frames=frame,fargs=(opticalflow,),interval=41)
            gif.save(f'data/{category}/{type}/{name}.gif',writer='pillow',fps=24)
            plt.close() """