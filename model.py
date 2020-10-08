
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
    
    
        self.embedding = nn.Embedding(vocab_size, embed_size) #https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    
        self.lstm = nn.LSTM(input_size =  embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bias = True,
                            batch_first = True,
                            dropout = 0.2) #https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        self.linear = nn.Linear(hidden_size, vocab_size) #https://pytorch.org/docs/stable/nn.html
        
    def forward(self, features, captions):
        
        captions_2 = torch.stack([i[:-1] for i in captions]) # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/2
        
        # embedding > lstm > linear
        x = self.embedding(captions_2)
        x = torch.cat((features.unsqueeze(1), x),1) # https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
        x, h = self.lstm(x) 
        
        return(self.linear(x))
        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
      
        pred_token = 0
        sample_out = []     
        
        for i in range(max_len+1):
            if pred_token == 1 | i == max_len:
                pass
            if pred_token != 1 & i != max_len:
                out, states = self.lstm(inputs, states)
                word = self.linear(out).to('cpu')
                max_p, pred_word = word.max(2)
                sample_out.append(pred_word.item())
                pred_token = pred_word.item()

                inputs = self.embedding(pred_word.to(inputs.device))

                
        return [i for i in sample_out if i > 1]



