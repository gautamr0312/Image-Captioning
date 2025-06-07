import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
        # TODO: Complete this function
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        # TODO: Complete this function
        inp = features.unsqueeze(1)
        inputs = torch.cat((inp,embeddings),dim=1)
        out,(hn,cn) = self.lstm(inputs)
        outputs = self.linear(out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        states = (torch.randn(1,1,self.hidden_size).to(inputs.device),
                  torch.randn(1,1,self.hidden_size).to(inputs.device))
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            if predicted == 1:
                break
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence
