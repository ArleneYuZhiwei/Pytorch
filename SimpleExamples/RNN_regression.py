# -*- coding: utf-8 -*-
import torch 
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 10  #image height
INPUT_SIZE = 1 #image width
LR = 0.02


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                batch_first=True,#(bath,time_step,input)
                )
        self.out = nn.Linear(32,1)
    def forward(self,x,h_state): 
        #x (batch,time_step,input_size)
        #h_state (n_layers,batch,hidden_size)
        #r_out (batch,time_step,output_size=hidden_size)
        r_out,h_state = self.rnn(x,h_state) #x (batch,time_step,input_size) 
        outs =[]
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:,time_step,:]))
        #out = self.out(r_out[:,-1,:])  
        return torch.stack(outs,dim=1),h_state
    
rnn=RNN()
print (rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.MSELoss() #标签不是one_hot的形式
h_state =None #初始化hidden state
for step in range(60):
    start, end = step*np.pi, (step+1)*np.pi #time steps
    # use sin predicts cos
    steps = np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis]))
    #将只有一个维度的数据改成[batch,time_step,input_size]的形式
    y = Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))
    
    prediction,h_state = rnn(x,h_state)
    h_state = Variable(h_state.data) 
    #要把hidden state包装成一个variable再传入网络
    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    