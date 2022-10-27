import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 5:
        (n_out, n_in, length, height, width) = layer.weight.size()
        n = n_in * length * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)




#########################################
class EmbeddingLayerCNN(nn.Module):
    def __init__(self, classes_num):
        super(EmbeddingLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn1 = nn.BatchNorm2d(1)

        self.init_weights()
    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, reduce_dim=True):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)

        x = F.relu(self.bn1(self.conv1(x)))

        if reduce_dim == True:
            x = torch.transpose(x, 1, 3)
            x = F.avg_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])

        return x

class EmbeddingLayerRNN(nn.Module):
    def __init__(self, model_type, feature_num, classes_num):
        super(EmbeddingLayerRNN, self).__init__()

        self.model_type = model_type
        if self.model_type == 'LSTM':
            self.rnn1 = nn.LSTM(feature_num, feature_num, batch_first=True, num_layers=1)
        elif self.model_type == 'GRU':
            self.rnn1 = nn.GRU(feature_num, feature_num, batch_first=True, num_layers=1)

        self.ln1 = nn.LayerNorm(feature_num)

    def init_hidden(self, batch, hidden_dim):
        if self.model_type == 'LSTM':
            return (Variable(torch.rand(1, batch, hidden_dim).cuda()), Variable(torch.rand(1, batch, hidden_dim).cuda()))
        elif self.model_type == 'GRU':
            return Variable(torch.rand(1, batch, hidden_dim).cuda())

    def forward(self, input, reduce_dim=True):

        h0 = self.init_hidden(input.shape[0], 768)
        x, _ = self.rnn1(input, h0)
        x = F.selu(self.ln1(x))

        if reduce_dim == True:
            x = torch.transpose(x, 1, 2)
            x = F.avg_pool1d(x, kernel_size=x.shape[2])
            x = x.view(x.shape[0:2])

        return x

##########Self Distillation########### 

class Wav2vec2DistillRNNnew(nn.Module):
    def __init__(self, model_deep, classes_num, distill_layer, feature_level, distill_modeltype = 'LSTM'):
        """
        model_deep: the deepest model
        distill_layer: 1-11 (output of the embeddings + the output of each layer) of shape (batch_size, sequence_length, hidden_size)) (12 transformer layers in wav2vec2-base)
        feature_level: "embedding" OR "linear"
        distill_modeltype: 'LSTM' OR 'GRU'
        """

        super(Wav2vec2DistillRNNnew, self).__init__()

        self.model_deep = model_deep
        self.distill_layer = distill_layer
        self.feature_level = feature_level

        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        

        # distill layers
        self.distrnn = nn.ModuleList()
        self.distlinear = nn.ModuleList()
        self.distclassifier = nn.ModuleList()
        for i in range(0, len(self.distill_layer)):
            self.distrnn.append(EmbeddingLayerRNN(distill_modeltype, 768, classes_num))
            self.distlinear.append(nn.Linear(in_features=768, out_features=256, bias=True))
            self.distclassifier.append(nn.Linear(256, classes_num, bias=True))      

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)
        for i in range(0, len(self.distill_layer)):
            init_layer(self.distlinear[i])
            init_layer(self.distclassifier[i])

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states             

        x_feature = []
        # for the deepest model
        x = torch.transpose(hidden_states[-1], 1, 2)
        x = F.avg_pool1d(x, kernel_size=x.shape[2]).view(x.shape[0:2])
        if self.feature_level == "embedding": 
            x_feature.append(x)

        x = self.linear(x)
        if self.feature_level == "linear": 
            x_feature.append(x)

        x = self.classifier(x)
        #x = x.logits

        # for the distilled models
        x_distill = []
        for i in range(0, len(self.distill_layer)):
            x_d = self.distrnn[i](hidden_states[self.distill_layer[i]])
            if self.feature_level == "embedding": 
                x_feature.append(x_d)  
                          
            x_d = self.distlinear[i](x_d)
            if self.feature_level == "linear": 
                x_feature.append(x_d)

            x_d = self.distclassifier[i](x_d)
            x_distill.append(x_d)


        return x, x_distill, x_feature



class Wav2vec2DistillCNN(nn.Module):
    def __init__(self, model_deep, classes_num, distill_layer, feature_level, distill_modeltype=None):
        """
        model_deep: the deepest model
        distill_layer: 1-10 (output of the embeddings + the output of each layer) of shape (batch_size, sequence_length, hidden_size)) (12 transformer layers in wav2vec2-base)
        feature_level: "embedding" OR "linear"
        distill_modeltype: 
        """

        super(Wav2vec2DistillCNN, self).__init__()

        self.model_deep = model_deep
        self.distill_layer = distill_layer
        self.feature_level = feature_level

        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        

        # distill layers
        self.distcnn = nn.ModuleList()
        self.distlinear = nn.ModuleList()
        self.distclassifier = nn.ModuleList()
        for i in range(0, len(self.distill_layer)):
            self.distcnn.append(EmbeddingLayerCNN(classes_num))
            self.distlinear.append(nn.Linear(in_features=768, out_features=256, bias=True))
            self.distclassifier.append(nn.Linear(256, classes_num, bias=True))      

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)
        for i in range(0, len(self.distill_layer)):
            init_layer(self.distlinear[i])
            init_layer(self.distclassifier[i])

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states             

        x_feature = []
        # for the deepest model
        x = torch.transpose(hidden_states[-1], 1, 2)
        x = F.avg_pool1d(x, kernel_size=x.shape[2]).view(x.shape[0:2])
        if self.feature_level == "embedding": 
            x_feature.append(x)

        x = self.linear(x)
        if self.feature_level == "linear": 
            x_feature.append(x)

        x = self.classifier(x)
        #x = x.logits

        # for the distilled models
        x_distill = []
        for i in range(0, len(self.distill_layer)):
            x_d = self.distcnn[i](hidden_states[self.distill_layer[i]])
            if self.feature_level == "embedding": 
                x_feature.append(x_d)  
                          
            x_d = self.distlinear[i](x_d)
            if self.feature_level == "linear": 
                x_feature.append(x_d)

            x_d = self.distclassifier[i](x_d)
            x_distill.append(x_d)


        return x, x_distill, x_feature



##########Teacher-Student Distillation###########      


class Wav2vec2DistillRNNTeacherStudent(nn.Module):
    def __init__(self, model_deep, classes_num, distill_layer, distill_modeltype = 'LSTM'):
        """
        model_deep: the deepest model
        distill_layer: 1-10 (output of the embeddings + the output of each layer) of shape (batch_size, sequence_length, hidden_size)) (12 transformer layers in wav2vec2-base)
        feature_level: "embedding" 
        distill_modeltype: 'LSTM' OR 'GRU'
        """

        super(Wav2vec2DistillRNNTeacherStudent, self).__init__()

        self.model_deep = model_deep
        self.distill_layer = distill_layer

        # predict distill layers
        self.distrnn = nn.ModuleList()
        for i in range(0, len(self.distill_layer)):
            self.distrnn.append(EmbeddingLayerRNN(distill_modeltype, 768, classes_num))

        # predict classes
        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states             
        
        x_output = torch.transpose(hidden_states[2], 1, 2)
        x_output = F.avg_pool1d(x_output, kernel_size=x_output.shape[2]).view(x_output.shape[0:2])
        x_output = self.linear(x_output)
        x_output = self.classifier(x_output)

        x_feature = []
        x = hidden_states[2]
        for i in range(0, len(self.distill_layer)):
            if i < len(self.distill_layer)-1:
                x = self.distrnn[i](x, reduce_dim=False)

                x_temp = torch.transpose(x, 1, 2)
                x_temp = F.avg_pool1d(x_temp, kernel_size=x_temp.shape[2])
                x_temp = x_temp.view(x_temp.shape[0:2])
                x_feature.append(x_temp)
            else:
                x = self.distrnn[i](x)
                x_feature.append(x)

        return x_output, x_feature   



class Wav2vec2DistillCNNTeacherStudent(nn.Module):
    def __init__(self, model_deep, classes_num, distill_layer, distill_modeltype = None):
        """
        model_deep: the deepest model
        distill_layer: 1-10 (output of the embeddings + the output of each layer) of shape (batch_size, sequence_length, hidden_size)) (12 transformer layers in wav2vec2-base)
        feature_level: "embedding" 
        distill_modeltype: 'LSTM' OR 'GRU'
        """

        super(Wav2vec2DistillCNNTeacherStudent, self).__init__()

        self.model_deep = model_deep
        self.distill_layer = distill_layer

        # predict distill layers
        self.distcnn = nn.ModuleList()
        for i in range(0, len(self.distill_layer)):
            self.distcnn.append(EmbeddingLayerCNN(classes_num))

        # predict classes
        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states   

        x_output = torch.transpose(hidden_states[2], 1, 2)
        x_output = F.avg_pool1d(x_output, kernel_size=x_output.shape[2]).view(x_output.shape[0:2])
        x_output = self.linear(x_output)
        x_output = self.classifier(x_output)

        x_feature = []
        x = hidden_states[2]
        for i in range(0, len(self.distill_layer)):
            if i < len(self.distill_layer)-1:
                x = self.distcnn[i](x, reduce_dim=False)

                x_temp = torch.transpose(x, 1, 3)
                x_temp = F.avg_pool2d(x_temp, kernel_size=x_temp.shape[2:])
                x_temp = x_temp.view(x_temp.shape[0:2])
                x_feature.append(x_temp)

                x = torch.squeeze(x, 1)
            else:
                x = self.distcnn[i](x)
                x_feature.append(x)        

        return x_output, x_feature






########### Fine Tuning ###################
class Wav2vec2FineTune(nn.Module):
    def __init__(self, model_deep, classes_num, layer):
        super(Wav2vec2FineTune, self).__init__()

        self.model_deep = model_deep
        self.layer = layer

        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states             

        x = torch.transpose(hidden_states[self.layer], 1, 2)
        x = F.avg_pool1d(x, kernel_size=x.shape[2]).view(x.shape[0:2])
        x = self.linear(x)
        x = self.classifier(x)
        
        return x






##################### Old model ######

class Wav2vec2DistillRNN(nn.Module):
    def __init__(self, model_deep, classes_num, distill_layer, feature_level, distill_modeltype = 'LSTM'):
        """
        model_deep: the deepest model
        distill_layer: 1-10 (output of the embeddings + the output of each layer) of shape (batch_size, sequence_length, hidden_size)) (12 transformer layers in wav2vec2-base)
        feature_level: "embedding" OR "linear"
        """

        super(Wav2vec2DistillRNN, self).__init__()

        self.model_deep = model_deep
        self.distill_layer = distill_layer
        self.feature_level = feature_level

        self.linear = nn.Linear(in_features=768, out_features=256, bias=True)
        self.classifier = nn.Linear(256, classes_num, bias=True)        


        if len(self.distill_layer) >= 1: 
            self.embrnn1 = EmbeddingLayerRNN(distill_modeltype, 768, classes_num)
            self.linear1 = nn.Linear(in_features=768, out_features=256, bias=True)
            self.classifier1 = nn.Linear(256, classes_num, bias=True)

        if len(self.distill_layer) >=2:
            self.embrnn2 = EmbeddingLayerRNN(distill_modeltype, 768, classes_num)
            self.linear2 = nn.Linear(in_features=768, out_features=256, bias=True)
            self.classifier2 = nn.Linear(256, classes_num, bias=True)           

        if len(self.distill_layer) >=3:
            self.embrnn3 = EmbeddingLayerRNN(distill_modeltype, 768, classes_num)
            self.linear3 = nn.Linear(in_features=768, out_features=256, bias=True)
            self.classifier3 = nn.Linear(256, classes_num, bias=True)           

        self.init_weights()

    def init_weights(self):
        init_layer(self.linear)
        init_layer(self.classifier)
        if  len(self.distill_layer) >= 1: 
            init_layer(self.linear1)
            init_layer(self.classifier1)
        if  len(self.distill_layer) >= 2: 
            init_layer(self.linear2)
            init_layer(self.classifier2)
        if  len(self.distill_layer) >= 3: 
            init_layer(self.linear3)
            init_layer(self.classifier3)

    def forward(self, input):
        x = self.model_deep(input)
        hidden_states = x.hidden_states             

        x_feature = []
        # for the deepest model
        x = torch.transpose(hidden_states[-1], 1, 2)
        x = F.avg_pool1d(x, kernel_size=x.shape[2]).view(x.shape[0:2])
        if self.feature_level == "embedding": 
            x_feature.append(x)

        x = self.linear(x)
        if self.feature_level == "linear": 
            x_feature.append(x)

        x = self.classifier(x)
        #x = x.logits

        # for the distilled models
        x_distill = []
        if len(self.distill_layer) >= 1:
            x_d = self.embrnn1(hidden_states[self.distill_layer[0]])
            if self.feature_level == "embedding": 
                x_feature.append(x_d)
            
            x_d = self.linear1(x_d)
            if self.feature_level == "linear": 
                x_feature.append(x_d)

            x_d = self.classifier1(x_d)
            x_distill.append(x_d)

        if len(self.distill_layer) >= 2:
            x_d = self.embrnn2(hidden_states[self.distill_layer[1]])
            if self.feature_level == "embedding": 
                x_feature.append(x_d)
            
            x_d = self.linear2(x_d)
            if self.feature_level == "linear": 
                x_feature.append(x_d)

            x_d = self.classifier2(x_d)
            x_distill.append(x_d)
        if len(self.distill_layer) >= 3:
            x_d = self.embrnn3(hidden_states[self.distill_layer[2]])
            if self.feature_level == "embedding": 
                x_feature.append(x_d)

            x_d = self.linear3(x_d)
            if self.feature_level == "linear": 
                x_feature.append(x_d)

            x_d = self.classifier3(x_d)
            x_distill.append(x_d)

        return x, x_distill, x_feature









