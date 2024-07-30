import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models



class mymodel(nn.Module):
    def __init__(self, window_len, axis = 6):
        super(mymodel, self).__init__()
        self.encoder = temporal_att_embedding(input_size = axis)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # nn.init.normal_(self.alpha, mean=0, std=1) #original yes
        self.att_T = incept_triplet()
        self.att_F = incept_triplet()
        self.decoder = co_attention(window_len=window_len)
        self.MLP_t = MLP(window_len=window_len)
        self.MLP_f = MLP(window_len=window_len)
    
    def forward(self, t, f):
        t,f = self.encoder(t,f)
        t = self.att_T(t.transpose(2,1))
        f = self.att_F(f.transpose(2,1))
        t,f = self.decoder(t.transpose(2,1),f.transpose(2,1))
        t_out = self.MLP_t(t)
        f_out = self.MLP_f(f)
        

        return t_out, f_out, self.alpha


class temporal_att_embedding(nn.Module):
    def __init__(self, input_size):
        super(temporal_att_embedding, self).__init__()
        self.gru_t = nn.GRU(input_size = input_size, hidden_size = 64, batch_first=True)
        self.gru_f = nn.GRU(input_size = input_size, hidden_size = 64, batch_first=True)
        
    def forward(self, t, f):
        
        t,_ = self.gru_t(t)
        t = nn.Dropout(0.4)(t)

        f,_ = self.gru_f(f)
        f = nn.Dropout(0.4)(f)
        

        return t,f
    
    

class co_attention(nn.Module):
    def __init__(self, window_len):
        super(co_attention, self).__init__()
        # self.weights = torch.zeros(64,64)
        self.weights = nn.Parameter(torch.empty(window_len, window_len))
        nn.init.normal_(self.weights, mean=0, std=1)

    def forward(self, t, f):
        f_T = f.transpose(2,1)
        s_weight = torch.matmul(torch.matmul(t, self.weights.to('cuda')), f_T)

        
        t_weight = nn.Softmax(dim = 1)(s_weight)
        f_weight = nn.Softmax(dim = 2)(s_weight)

        t_out = torch.matmul(t.transpose(2,1), t_weight)
        f_out = torch.matmul(f_T, f_weight)
        t_out = nn.Dropout(0.3)(t_out)
        f_out = nn.Dropout(0.3)(f_out)
        return t_out, f_out 

class incept_triplet(nn.Module):
    def __init__(self):
        super(incept_triplet, self).__init__()
        self.triplet_3 = TripletAttention(kernel_size = 3)
        self.triplet_5 = TripletAttention(kernel_size = 5)
        self.triplet_7 = TripletAttention(kernel_size = 7)
    
    def forward(self, x):
        x_3 = self.triplet_3.forward(x)
        x_5 = self.triplet_5.forward(x)
        x_7 = self.triplet_7.forward(x)
        total_output = torch.cat([x_3, x_5, x_7], dim=2)
        return total_output

class MLP(nn.Module):
    def __init__(self, window_len):
        super(MLP, self).__init__()
        
        self.out = nn.Sequential(
            nn.Linear(window_len, window_len),
            nn.ReLU(),
            nn.BatchNorm1d(window_len),
            nn.Dropout(0.7),
            nn.Linear(window_len, 6),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = torch.mean(x, dim = 2)
        out = self.out(x)

        return out

class TripletAttention(nn.Module):
    def __init__(self, kernel_size):
        super(TripletAttention, self).__init__()
        self.T_block = Att_block(kernel_size)
        self.C_block = Att_block(kernel_size)
        
        # self.alpha1 = nn.Parameter(torch.tensor(1/3))
        # self.alpha2 = nn.Parameter(torch.tensor(1/3))
    def forward(self, x):
        T_out = self.T_block(x)
        C_out = self.C_block(x.transpose(2,1))

        return (T_out + C_out.transpose(2,1))/2


class Att_block(nn.Module):
    def __init__(self, kernel_size):
        super(Att_block, self).__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=kernel_size, stride=1, padding = kernel_size // 2)
    
    def forward(self, x):
        b, _, out_shape = x.shape
        pool = self.four_pool(x)
        pool = self.conv(pool)
        AttWeight = nn.ReLU()(pool)
        AttWeight = AttWeight.repeat(1, out_shape, 1)
        att_out = torch.matmul(AttWeight, x.transpose(2,1))
        return att_out

    def four_pool(self, x):
        mean = torch.mean(x, 1).unsqueeze(1)
        std = torch.std(x, 1).unsqueeze(1)

        max_array,_ = torch.max(x, 1)
        max_array = max_array.unsqueeze(1)

        min_array,_ = torch.min(x, 1)
        min_array = min_array.unsqueeze(1)

        final_out = torch.cat([mean, std, max_array, min_array], dim = 1)
        
        return final_out



