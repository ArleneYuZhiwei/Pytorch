# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data

BATCH_SIZE = 5
x=torch.linspace(1,10,10)
y=torch.linspace(10,1,10)

torch_dataset=Data.TensorDataset(data_tensor=x,target_tensor=y)
loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,#可以打乱数据排序进行训练
        num_workers=2,)
#epoch:整体数据的训练
#每个epoch中都会从头对每小批数据进行训练
#如果数据总量除以batch_size不整，最后一批返回剩的。
for epoch in range(3):
    for step, (batch_x,batch_y) in enumerate(loader):
        #training...
        print ('Epoch:',epoch,'|Step:',step,'|batch_x:',batch_x.numpy(),'|batch_y:',batch_y.numpy())
        

