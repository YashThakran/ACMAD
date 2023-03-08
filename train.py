from numpy import argmax
from dataloader import *
from models import *
import torch.optim as optim
import torch
import torch.nn as nn



device = torch.device('cpu')
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
            
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.0)
    

model = DilatedNet()

model.apply(init_weights)
model.to(device)

softmax = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)

test_list=["test_hindi","test_bengali","test_bhojpuri","test_gujarati","test_haryanvi","test_kannada","test_malayalam","test_odia","test_punjabi","test_tamil"]
best_score_dict = {k: v for k, v in zip(test_list, [0]*len(test_list))}


for epoch in range(100):
    model.to(device)  
    model.train()

    # torch.cuda.empty_cache()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(train_loader, 0):

        audio, labels = data[0], data[1]
        
        audio=audio.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        output = model(audio)
        
        loss = criterion(output, labels.float())
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

        predict = torch.argmax(softmax(output),dim=1)
        
        true_label=torch.argmax(labels,dim=1)
        correct=torch.sum(torch.eq(predict,true_label))
        
        total = labels.size(0)
         
        correct += correct.item()
        total += total
        
    accuracy=100*correct/total
    
    print(f"Epoch - {epoch} :  Loss - {running_loss/total}, Accuracy - {accuracy}%")
