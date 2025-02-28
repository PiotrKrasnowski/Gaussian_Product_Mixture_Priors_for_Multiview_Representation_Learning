import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import cuda
from numbers import Number
import sys 
import torchvision.models as models

class IBNet_multiview(nn.Module):

    def __init__(self, K=256, class_size=10, n_views=2, model_name='CNN4', model_mode='stochastic', 
                 mean_normalization_flag=False, std_normalization_flag=False):  # the mode is either 'stochastic' or 'deterministic'
        if model_mode not in ['stochastic','deterministic']:
            print("The model mode can be either stochastic or deterministic!")
            sys.exit()
        
        if model_name not in ['CNN4','Resnet']:
            print("The model can be choosen only as CNN4 or Resnet!")
            sys.exit()

        super(IBNet_multiview, self).__init__()
        self.K = K
        self.n_views    = n_views
        self.model_mode = model_mode
        self.model_name = model_name
        self.mean_normalization_flag = mean_normalization_flag
        self.std_normalization_flag = std_normalization_flag
        self.class_size = class_size

        # encoders
        self.encoders = []
        self.layer_norm1 = []
        self.layer_norm2 = []
        for _ in range(self.n_views):
            if model_name == 'CNN4':
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(3, 8, 5, 1, 2),
                    nn.Conv2d(8, 8, 5, 1, 2),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(8, 16, 3, 1, 1),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten(),
                    nn.Linear(1024, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 2*self.K),
                    )) 
            elif model_name == "Resnet":
                print("Resnet Model is chosen!")
                encoder = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1") 
                input_last = encoder.fc.in_features
                encoder.fc = nn.Linear(input_last, 2*self.K) 
                self.encoders.append(encoder)

            self.layer_norm1.append(nn.Sequential(torch.nn.LayerNorm([self.K],elementwise_affine=True)).cuda())
            self.layer_norm2.append(nn.Sequential(torch.nn.LayerNorm([self.K],elementwise_affine=True)).cuda())

        # decoders    
        self.decoder_joint = nn.Sequential(
                    nn.Linear(self.n_views * self.K, self.class_size)
                    )

    def forward(self, x, num_sample = 1):

        if (not self.mean_normalization_flag) and (not self.std_normalization_flag):
            statistics = torch.stack([self.encoders[k](x[k]) for k in range(self.n_views)],dim=1) #[B,V,2K]
            mu  = statistics[:,:,:self.K]  #[B,V,K]
            std =  F.softplus(statistics[:,:,self.K:]-5,beta=1)         #[B,V,K]
        elif not self.mean_normalization_flag:
            statistics = torch.stack([self.encoders[k](x[k]) for k in range(self.n_views)],dim=1) #[B,V,2K]
            mu  = statistics[:,:,:self.K]  #[B,V,K]
            std = torch.stack([ 0.5*self.layer_norm2[k](F.softplus(self.encoders[k](x[k])[:,self.K:]-5,beta=1)) for k in range(self.n_views)],dim=1) #[B,V,K] 
        elif not self.std_normalization_flag:
            statistics = torch.stack([self.encoders[k](x[k]) for k in range(self.n_views)],dim=1) #[B,V,2K]
            std =  F.softplus(statistics[:,:,self.K:]-5,beta=1)         #[B,V,K]
            mu = torch.stack([self.layer_norm1[k](self.encoders[k](x[k])[:,:self.K]) for k in range(self.n_views)],dim=1) #[B,V,K]
        else:
            mu = torch.stack([self.layer_norm1[k](self.encoders[k](x[k])[:,:self.K]) for k in range(self.n_views)],dim=1) #[B,V,K]
            std = torch.stack([ 0.5*self.layer_norm2[k](F.softplus(self.encoders[k](x[k])[:,self.K:]-5,beta=1)) for k in range(self.n_views)],dim=1) #[B,V,K] 
            
        if self.model_mode == 'stochastic':
            encoding = self.reparametrize_n(mu,std,num_sample) 
        else:
            encoding = mu           

        if num_sample > 1: 
            logit = self.decoder_joint( encoding.view(num_sample,x.size(1),self.n_views*self.K) ) 
        else:
            logit = self.decoder_joint( encoding.view(x.size(1),self.n_views*self.K) ) 

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        if n != 1 :
            mu - self.expand(mu,n)
            std = self.expand(std,n)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + eps * std

    def expand(self, v, n):
            if isinstance(v, Number): return torch.Tensor([v]).expand(n, 1)
            else: return v.expand(n, *v.size())

    def weight_init(self):
        if self.model_name == "CNN4":
            for k in range(self.n_views): xavier_init(self.encoders[k])
        xavier_init(self.decoder_joint)


def xavier_init(ms):
    torch.manual_seed(0)
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()

