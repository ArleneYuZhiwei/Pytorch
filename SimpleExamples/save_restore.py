# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
torch.manual_seed(1)
#fake data
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x, requires_grad=False),Variable(y,requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
            )
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ("net_1",prediction[:5])
    torch.save(net1,'net.pkl')   #entire net
    torch.save(net1.state_dict(),'net_params.pkl')  #parameters
    

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    print ("net_2",prediction[:5])

def restore_params():
    net3 = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1))
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction=net3(x)
    print ("net_3",prediction[:5])

save()
restore_net()
restore_params()
