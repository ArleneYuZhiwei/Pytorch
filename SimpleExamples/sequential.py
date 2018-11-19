# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)#x
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

x,y=Variable(x),Variable(y)
#搭图的过程，搭建神经网络
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
        
    
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return (x)
    
net1 = Net(2,10,2)
print (net)

#method2
net2=torch.nn.Sequential(
        torch.nn.Linear(2,10),
        torch.nn.ReLu(),
        torch.nn.Linear(10,2),)
print (net2)
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()
#训练次数
for t in range (100):
    out = net(x)   
    loss = loss_func(out,y)     
    optimizer.zero_grad()#清除上一批的gradient
    loss.backward()
    optimizer.step()
    prediction=torch.max(F.softmax(out),1)[1]
    print ("t",t)
    print ("out",out[0])
    print ("prediction",prediction[0])
    print ("loss",loss)