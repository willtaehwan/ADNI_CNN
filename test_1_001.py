# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:11:49 2023

@author: kjath
"""

import ADNI_data as ad
import torch
import CNN_ADNI as ca
import torch.nn as nn


# ad.ADNI_data('csv파일 경로', 'gz파일 폴더 경로')
br = ad.ADNI_data('image70.csv','data/images_pytorch')

batch_size = 7
learning_rate = 0.0002
num_epoch = 50


#dataset load
b_c = br.brain_128().chunk(10)
l_c = br.label().chunk(10)

# train_test_split (7:3)
train_loader = [[b_c[i],l_c[i]] for i in range(7)]
test_loader = [[b_c[i], l_c[i]] for i in range(7, 10)]



# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CNN model load
model = ca.CNN(batch_size).to(device)



# loss_function
loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



loss_arr =[]
for i in range(num_epoch):
    for image,label in train_loader:
        x = image.to(device).type(torch.float32)
        y= label.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()
        
        
        print(loss)
        
        loss_arr.append(loss.cpu().detach().numpy())
        


correct = 0
total = 0

# evaluate model
model.eval()

with torch.no_grad():
    for image,label in test_loader:
        x = image.to(device).type(torch.float32)
        y= label.to(device)

        output = model.forward(x)
        _,y_max = torch.max(y,1)
        # torch.max함수는 (최댓값,index)를 반환 
        _,output_index = torch.max(output,1)
        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        
        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        correct += (output_index == y_max).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))