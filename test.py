from dataloader import *
from models import DilatedNet
import torch

model = DilatedNet()

model.load_state_dict(torch.load("../saved_ckpt/ckpt.pt",map_location=torch.device('cpu')))

softmax = nn.Softmax(dim=1)
criterion = nn.CrossEntropyLoss()

test_list=["test_all","test_hindi","test_bengali","test_bhojpuri","test_gujarati","test_haryanvi","test_kannada","test_malayalam","test_odia","test_punjabi","test_tamil"]
best_score_dict = {k: v for k, v in zip(test_list, [0]*len(test_list))}

with torch.no_grad():        
        model.to(torch.device("cpu"))
        model.eval()

        for i in test_list:
            test_data_lang = Adima(filepath,i)
            test_loader_lang = DataLoader(test_data_lang, batch_size=32, shuffle=True) 
            vsample, vcorrect = 0, 0
            for vdata, vlabel in test_loader_lang:
                
                vsample += len(vdata)
                
                vpreds = model(vdata.to(torch.device("cpu")))
                vloss = criterion(vpreds, vlabel)
                
                vcorrect += torch.eq(softmax(vpreds).argmax(dim=1), vlabel.argmax(dim=1)).sum().item()
            
            del(test_data_lang)
            del(test_loader_lang)   

            if best_score_dict[i] < round((vcorrect/vsample), 4):
                best_score_dict[i] = round((vcorrect/vsample), 4)
                        
        
print(best_score_dict)