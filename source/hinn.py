import cv2
import torch
import numpy as np

__all__ = ['HINN']

def NavierStokesEquation(velocity_ds,viscosity,k,device):
    # parameter
    new_height = velocity_ds.shape[2]
    new_width = velocity_ds.shape[3]
    border = torch.abs(velocity_ds).max().item()*0.4
    # operator
    dx = torch.tensor([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]],dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    dy = torch.tensor([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]],dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    dxx = torch.tensor([[0,1,0],
                        [1,-4,1],
                        [0,1,0]],dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    dyy = torch.tensor([[0,1,0],
                        [1,-4,1],
                        [0,1,0]],dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    avg = torch.tensor([[0.125,0.125,0.125],
                    [0.125,0,0.125],
                    [0.125,0.125,0.125]],dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    # derivation
    u = velocity_ds[:,0,:,:].unsqueeze(0)
    v = velocity_ds[:,1,:,:].unsqueeze(0)
    u_x = torch.nn.functional.conv2d(u,dx,padding=1)
    u_y = torch.nn.functional.conv2d(u,dy,padding=1)
    u_xx = torch.nn.functional.conv2d(u,dxx,padding=1)
    u_yy = torch.nn.functional.conv2d(u,dyy,padding=1)
    v_x = torch.nn.functional.conv2d(v,dx,padding=1)
    v_y = torch.nn.functional.conv2d(v,dy,padding=1)
    v_xx = torch.nn.functional.conv2d(v,dxx,padding=1)
    v_yy = torch.nn.functional.conv2d(v,dyy,padding=1)
    # alignment
    u_ali = torch.nn.functional.conv2d(u,avg,padding=1)-u
    v_ali = torch.nn.functional.conv2d(v,avg,padding=1)-v
    # navigation
    u_sign = torch.sign(u)
    v_sign = torch.sign(v)
    indices_x = torch.clamp(torch.arange(new_width,device=device).unsqueeze(0)+u_sign*10,0,new_width-1).long()
    indices_y = torch.clamp(torch.arange(new_height,device=device).unsqueeze(1)+v_sign*10,0,new_height-1).long()
    u_next = torch.zeros_like(u)
    v_next = torch.zeros_like(v)
    u_next[0] = u[0,0,indices_y,indices_x]
    v_next[0] = v[0,0,indices_y,indices_x]
    u_nav = u_next-u
    v_nav = v_next-v
    # cohesion
    u_sign = torch.sign(u)
    v_sign = torch.sign(v)
    velocity_mag,_ = cv2.cartToPolar(u.squeeze(0).squeeze(0).detach().cpu().numpy(),v.squeeze(0).squeeze(0).detach().cpu().numpy())
    _,crowd = cv2.threshold(velocity_mag,0.2,255,cv2.THRESH_BINARY)
    coh_canny = cv2.Canny(crowd.astype(np.uint8),100,200)
    contour,_ = cv2.findContours(coh_canny,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    coh_contour = [cnt for cnt in contour if cv2.contourArea(cnt)>5]
    cohesion = np.zeros_like(coh_canny)
    cv2.drawContours(cohesion,coh_contour,-1,1,thickness=5)
    u_sign = u_sign.squeeze(0).squeeze(0).detach().cpu().numpy()
    v_sign = v_sign.squeeze(0).squeeze(0).detach().cpu().numpy()
    indices_side1_x = np.clip(np.arange(new_width)+u_sign*10,0,new_width-1).astype(np.int32)
    indices_side1_y = np.clip(np.arange(new_height).reshape(-1,1),0,new_height-1).astype(np.int32)
    velocity_side1 = velocity_mag[indices_side1_y,indices_side1_x]
    indices_side2_x = np.clip(np.arange(new_width),0,new_width-1).astype(np.int32)
    indices_side2_y = np.clip(np.arange(new_height).reshape(-1,1)+v_sign*10,0,new_height-1).astype(np.int32)
    velocity_side2 = velocity_mag[indices_side2_y,indices_side2_x]
    coh_sign = velocity_side1>velocity_side2
    u_coh = torch.tensor(cohesion*coh_sign*u_sign*border,dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    v_coh = torch.tensor(-1*cohesion*coh_sign*v_sign*border,dtype=torch.float,device=device).unsqueeze(0).unsqueeze(0)
    # Navier-Stokes Equation
    u_pde = (u*u_x+v*u_y)-viscosity*(u_xx+u_yy)+k*u_ali+u_nav+u_coh
    v_pde = (u*v_x+v*v_y)-viscosity*(v_xx+v_yy)+k*v_ali+v_nav+v_coh
    velocity_pde = torch.cat([u_pde,v_pde],dim=1)
    return velocity_pde

def HIM(velocity,viscosity,k,device):
    batch = velocity.shape[0]
    height = velocity.shape[2]
    width = velocity.shape[3]
    new_height,new_width = int(height/2),int(width/2)
    grid_height,grid_width = torch.meshgrid(torch.linspace(-1,1,new_height),torch.linspace(-1,1,new_width),indexing='ij')
    grid = torch.stack((grid_width,grid_height),2).unsqueeze(0).clone().detach().requires_grad_(True).to(device)
    if batch==1:
        velocity_ds = torch.nn.functional.grid_sample(velocity,grid,mode='bilinear',padding_mode='border',align_corners=True)
        velocity_pde = NavierStokesEquation(velocity_ds,viscosity,k,device)
        velocity_us = torch.nn.functional.interpolate(velocity_pde,size=(height,width),mode='bicubic',align_corners=False)
        velocity_all = velocity_us
    else:
        velocity_all = torch.tensor([],dtype=torch.float,device=device)
        for i in range(batch):
            velocity_ds = torch.nn.functional.grid_sample(velocity[i,:,:,:].unsqueeze(0),grid,mode='bilinear',padding_mode='border',align_corners=True)
            velocity_pde = NavierStokesEquation(velocity_ds,viscosity,k,device)
            velocity_us = torch.nn.functional.interpolate(velocity_pde,size=(height,width),mode='bicubic',align_corners=False)
            velocity_all = torch.cat([velocity_all,velocity_us],dim=0)
    return velocity_all

class ConvResBlock(torch.nn.Module):
    def __init__(self,channels,conv_height,conv_width):
        super(ConvResBlock,self).__init__()
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.activation = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.layernorm = torch.nn.LayerNorm([channels,conv_height,conv_width])

    def forward(self,x):
        x_res = self.conv1(x)
        x_res = self.activation(x_res)
        x_res = self.conv2(x_res)
        x_res = self.layernorm(x_res)
        out = x+x_res
        return out

class ConvResNet(torch.nn.Module):
    def __init__(self,height,width,in_channels):
        super(ConvResNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.activation1 = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.activation2 = torch.nn.Tanh()
        self.convresblocks = torch.nn.Sequential(
            ConvResBlock(32,int(height/2),int(width/2)),
            ConvResBlock(32,int(height/2),int(width/2)),
            ConvResBlock(32,int(height/2),int(width/2)),
            ConvResBlock(32,int(height/2),int(width/2))
        )
        self.pixelshuffle = torch.nn.PixelShuffle(4)
        self.conv3 = torch.nn.Conv2d(in_channels=2,out_channels=2,kernel_size=3,stride=2,padding=1)

    def forward(self,input):
        output = self.conv1(input)
        output = self.activation1(output)
        output = self.conv2(output)
        output = self.activation2(output)
        output = self.convresblocks(output)
        output = self.pixelshuffle(output)
        output = self.conv3(output)
        return output

class HINN(torch.nn.Module):
    def __init__(self,height,width,device):
        super(HINN,self).__init__()
        self.device = device
        self.baseline = ConvResNet(height=height,width=width,in_channels=4)

    def forward(self,input):
        viscosity,k = 0.2,0.5
        input_nse = HIM(input,viscosity=viscosity,k=k,device=self.device)
        input = torch.cat([input,input_nse],dim=1)
        output = self.baseline(input)
        return output
