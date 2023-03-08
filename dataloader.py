from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn

avg = nn.AvgPool1d(kernel_size=131,stride=3)

class StatsPool(nn.Module):
    def __init__(self) -> None:
        pass
    
    def forward(self, x, dim):
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        return torch.cat([mean, std], dim=-1)

a = "..\\extracted_features"
filepath="..\\Dataset\\data.txt"

def label(x):
    if x=="Yes":
        # return 1
        return torch.tensor([0,1],dtype=torch.float32)

    if x=="No":
        # return 0
        return torch.tensor([1,0],dtype=torch.float32)

class Adima(Dataset):

    def __init__(self,filepath,tag):
        
        self.tag=tag
        with open(filepath)  as F:
            self.lines = F.readlines()

        self.train_list = [i.replace("\n", "") for i in self.lines if "train" in i]
        self.train_list_ = [i.replace("\n", "").replace("train/", "").replace(".wav", "") for i in self.lines if "train" in i]

        self.test_list = [i.replace("\n", "") for i in self.lines if "train" not in i]

        self.Indicw2v_features = ["/indicw2v_base_features/"+i+".feats.npy" for i in self.train_list_]
        self.er_features = ["/wav2vec2-lg-xlsr-en-speech-emotion-recognition_features/"+i+".feats.npy" for i in self.train_list_]
        
        self.test_hindi = [i for i in self.test_list if "Hindi" in i]
        self.test_bengali = [i for i in self.test_list if "Bengali" in i]
        self.test_bhojpuri = [i for i in self.test_list if "Bhojpuri" in i]
        self.test_gujarati = [i for i in self.test_list if "Gujarati" in i]
        self.test_haryanvi = [i for i in self.test_list if "Haryanvi" in i]
        self.test_kannada = [i for i in self.test_list if "Kannada" in i]
        self.test_malayalam = [i for i in self.test_list if "Malayalam" in i]
        self.test_odia = [i for i in self.test_list if "Odia" in i]
        self.test_punjabi = [i for i in self.test_list if "Punjabi" in i]
        self.test_tamil = [i for i in self.test_list if "Tamil" in i]

    def __len__(self):
        
        if self.tag=="train":
            return len(self.train_list)
        if self.tag=="test_all":   
            return len(self.test_list)
        if self.tag=="test_hindi":
            return len(self.test_hindi) 
        if self.tag=="test_bengali":
            return len(self.test_bengali)
        if self.tag=="test_bhojpuri":
            return len(self.test_bhojpuri)
        if self.tag=="test_gujarati":
            return len(self.test_gujarati)
        if self.tag=="test_haryanvi":
            return len(self.test_haryanvi)
        if self.tag=="test_kannada":
            return len(self.test_kannada)
        if self.tag=="test_malayalam":
            return len(self.test_malayalam)
        if self.tag=="test_odia":
            return len(self.test_odia)
        if self.tag=="test_punjabi":
            return len(self.test_punjabi)
        if self.tag=="test_tamil":
            return len(self.test_tamil)

    def __getitem__(self, idx):

        if self.tag == "train":
            d = self.train_list
        if self.tag == "test_all":
            d = self.test_list
        if self.tag=="test_hindi":
            d = self.test_hindi
        if self.tag=="test_bengali":
            d = self.test_bengali
        if self.tag=="test_bhojpuri":
            d = self.test_bhojpuri
        if self.tag=="test_gujarati":
            d = self.test_gujarati
        if self.tag=="test_haryanvi": 
            d = self.test_haryanvi
        if self.tag=="test_kannada":
            d = self.test_kannada
        if self.tag=="test_malayalam":
            d = self.test_malayalam
        if self.tag=="test_odia":
            d = self.test_odia
        if self.tag=="test_punjabi":
            d = self.test_punjabi
        if self.tag=="test_tamil":
            d = self.test_tamil
        
        
        m = np.load(a+"\\"+self.Indicw2v_features[idx].split("/")[1]+"\\"+self.Indicw2v_features[idx].split("/")[2]+"\\"+self.Indicw2v_features[idx].split("/")[4])
        n = np.load(a+"\\"+self.er_features[idx].split("/")[1]+"\\"+self.er_features[idx].split("/")[2]+"\\"+self.er_features[idx].split("/")[4])  

        features = torch.stack([StatsPool().forward((avg(torch.tensor(m).reshape(1,torch.tensor(m).shape[2],torch.tensor(m).shape[1]))),dim=1),torch.tensor(n)],dim=1)

        labels = label(self.CLSRIL_features[idx].split("/")[3])
        
        features=features.reshape(2,256)
                        
        return features, labels


train_data = Adima(filepath, "train")
test_data = Adima(filepath, "test_all")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
