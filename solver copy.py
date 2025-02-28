import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from dataset import return_data_x_y
from model import IBNet_multiview
import math
import scipy.stats as st
import random, sys
import matplotlib.pyplot as plt
import pandas as pd

from utils import cuda, set_constants_utils
from utils import KL_DG_DGM_prod, KL_lossy_DG_DGM_var
from utils import contribution_Gaussian_to_GM,contribution_lossy_Gaussian_to_GM, compute_marginals

class Solver(object):

    def __init__(self, args):
        self.args = args

        # Training parameters
        self.cuda        = (args["cuda"] and torch.cuda.is_available())
        self.seed        = args["seed"]
        self.epoch_1     = args["epoch_1"]
        self.epoch_2     = args["epoch_2"]
        self.batch_size  = args["batch_size"]
        self.lr          = args["lr"]
        self.eps         = 1e-20
        self.global_iter = 0
        self.global_epoch = 0
        self.counter = 0

        # Model parameters
        self.K      = args["K"]
        self.alpha  = args["alpha"]
        self.n_views = args["n_views"]
        self.num_avg = args["num_avg"]
        self.loss_id = args["loss_id"] 
        self.model_mode = args["model_mode"]
        self.model_name = args["model_name"]
        self.beta_joint = args["beta_joint"]

        if self.model_mode == 'deterministic' and self.loss_id not in [3,5,7,8,9,10]:
            print("In deterministic mode, only lossy regularizers can be used!")
            sys.exit()
    
        if self.num_avg > 1 and self.model_mode == 'deterministic':
            self.num_avg = 1 

        # CDVIB parameters
        self.M = args["centers_num"] 
        self.moving_coefficient_multiple = args["mov_coeff_mul"]
        self.moving_coefficient_var = args["mov_coeff_var"]
        self.temp_coeff = args["temp_coeff"]

        # Dataset size and name
        if args["dataset"] == 'MNIST':   
            self.train_size = 50000
            self.class_size = 10
        elif args["dataset"] == 'CIFAR10': 
            self.train_size = 50000
            self.class_size = 10
        elif args["dataset"] == 'CIFAR100': 
            self.train_size = 50000
            self.class_size = 100
        elif args["dataset"] == 'USPS': 
            self.train_size = 7291
            self.class_size = 10
        elif args["dataset"] == 'Caltech101': 
            self.train_size = 9145
            self.class_size = 102
        elif args["dataset"] == 'INTEL':
            self.train_size = 13986
            self.class_size = 6
            self.HY = math.log(self.class_size,2)
        
        self.HY = math.log(self.class_size,2)
            
        # Network
        self.IBnet = cuda(IBNet_multiview(K=self.K,class_size = self.class_size,n_views=self.n_views,model_name=self.model_name,model_mode=self.model_mode,mean_normalization_flag = args["mean_normalization_flag"],std_normalization_flag=args["std_normalization_flag"]), self.cuda)
        cuda(self.IBnet.decoder_joint, self.cuda)

        for k in range(self.n_views): 
            cuda(self.IBnet.encoders[k], self.cuda)
            if args["mean_normalization_flag"]:
                cuda(self.IBnet.layer_norm1[k],self.cuda)
            if args["std_normalization_flag"]:
                cuda(self.IBnet.layer_norm2[k],self.cuda)

        self.IBnet.weight_init()

        # Dataset Loading  
        self.num_datasets = args["num_datasets"]
        self.degrees_coeff = args["degrees_coeff"]
        self.translate_coeff= args["translate_coeff"]
        self.scale_coeff=args["scale_coeff"]
        self.PixelCorruption_coeff=args["PixelCorruption_coeff"]

        if (len(self.degrees_coeff) != self.n_views) or (len(self.translate_coeff) != self.n_views) or (len(self.scale_coeff) != self.n_views) or (len(self.PixelCorruption_coeff) != self.n_views):
            print("The transformation parameters are not consistent with the number of views!")
            sys.exit()

        train_x, train_y = [], []
        test_x, test_y = [], []
        self.dataset_x, self.dataset_y = {}, {}
        for i in range(self.num_datasets):
            train_x_new,train_y_new,test_x_new,test_y_new = return_data_x_y(args["dataset"], args["dset_dir"], self.n_views, self.degrees_coeff,self.translate_coeff,self.scale_coeff,self.PixelCorruption_coeff,args["occlusion"])
            train_x.append(train_x_new.numpy())
            train_y.append(train_y_new.numpy())
            test_x.append(test_x_new.numpy())
            test_y.append(test_y_new.numpy())
        self.dataset_x["train"] = train_x
        self.dataset_x["test"] = test_x
        self.dataset_y["train"] = train_y
        self.dataset_y["test"] = test_y

        # CDVIB and GMVIB initializations
        self.lossy_variance = args["lossy_variance"]   
        self.lossy_variance_var = args["lossy_variance_var"]   
        self.average_distance_centers_means = cuda(torch.tensor([0] * self.n_views), self.cuda)
        self.initialize_centers(args["center_initialization"],args["dataset"], args["dset_dir"])

        # Optimizer
        parameters =  [{'params': self.IBnet.encoders[k].parameters()} for k in range(self.n_views)]
        parameters += [{'params': self.IBnet.decoder_joint.parameters()}]
        if args["mean_normalization_flag"]:
            parameters +=  [{'params': self.IBnet.layer_norm1[k].parameters()} for k in range(self.n_views)]
        if args["std_normalization_flag"]:
            parameters +=  [{'params': self.IBnet.layer_norm2[k].parameters()} for k in range(self.n_views)]
        self.optim     = optim.Adam(parameters,lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)
        
        # Other
        self.per_epoch_stat = args["per_epoch_stat"]
        self.sqrt2pi = cuda(torch.sqrt(torch.tensor(2*torch.pi)), self.cuda) 
        self.matrix_A = cuda(torch.normal(0, 1, size=(2,self.K)),self.cuda) #Projection Matrix
        self.timestamp = args["timestamp"]
        self.figures_path = args["figures_path"]
        self.perturbation_flag =args["perturbation_flag"]
        

        self.loss_choice = args["loss_choice"]                      # This choice can be either 'prod_var', or 'prod', or 'var'
        self.update_centers_rule = args["update_centers_rule"]      # This choice can be either 'expectation', or 'D_KL'
        self.update_centers_coeff_mode = args["update_centers_coeff_mode"]

        set_constants_utils(self.K,self.M,self.eps,self.lossy_variance,self.n_views,self.cuda)
      
        self.index_matrix = cuda(torch.zeros(self.n_views,self.M**self.n_views,self.K,dtype=torch.long),self.cuda)
        for id_M_V in range(self.M**self.n_views):
            self.give_next_index(id_M_V,id_M_V,self.n_views,self.index_matrix)

    
    def give_next_index(self,id_M_V,remain_V,running_index,mat_matrix):
        if running_index == 1:
            mat_matrix[0,id_M_V,:]=remain_V
        else:
            mat_matrix[-1,id_M_V,:] = remain_V % self.M
            self.give_next_index(id_M_V,(remain_V-mat_matrix[-1,id_M_V,:])//self.M,running_index-1,mat_matrix[:-1,:,:])


    def train_full(self):
        print('Loss ID:{}, Beta joint:{:.0e}, Beta joint:{:.0e}, NumCent:{}, MovCoeff:{}, Mode:{}, Loss Choice:{}, update_center_rule:{}'.format(self.loss_id,self.beta_joint,self.beta_joint,self.M,self.moving_coefficient_multiple,self.model_mode,self.loss_choice,self.update_centers_rule))

        if self.model_mode == 'deterministic':
            print("Deterministic mode activated!")

        ##################
        # First training #
        ##################
        # reinitialize seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.train_accuracy_list = []
        self.test_accuracy_list  = []
        
        # train
        self.train_step1()
        # Computing some statistics after the fist training
        # a) accuracy over the training/testing datasets OK
        # b) average log-likelihood over the training/testing datasets OK
        # c) relevance I(Z;Y) and complexity I(Z;X) and regularization complexity OK
        # d) confidence intervals for accuracy/log-likelihood over the training/testing datasets OK
        # e) relevance I(Z;Y) and complexity I(Z;X) and regularization complexity per epoch OK
        print("Training 1 is finished")
        self.train1_epochs = {} 
        if self.per_epoch_stat == True:
            self.train1_epochs["izy_relevance_epochs"] = self.izy_relevance_epochs.cpu().numpy()
            self.train1_epochs["izx_bound"] = self.izx_complexity_epochs.cpu().numpy()
            self.train1_epochs["reg_complexity"] = self.reg_complexity_epochs.cpu().numpy()
        
        # self.plot_centers(0)
        self.test('train')

        self.train_accuracy_list.append(self.accuracy.detach())
        
        print('Loss ID:{}, Beta joint:{:.0e}, NumCent:{}, MovCoeff:{}'.format(self.loss_id,self.beta_joint,self.M,self.moving_coefficient_multiple))
        print('Final training accuracy: {:.3f}'.format(self.accuracy))
                  
        self.train1_train_dataset = {}
        self.train1_train_dataset["accuracy"] = self.accuracy.cpu().numpy()
        # self.train1_train_dataset["accuracy_confidence_low"] = self.accuracy_confidence_intervals[0]
        # self.train1_train_dataset["accuracy_confidence_high"] = self.accuracy_confidence_intervals[1]

        self.train1_train_dataset["log_likelihood"] = self.log_likelihood.cpu().numpy()
        # self.train1_train_dataset["log_likelihood_confidence_low"] = self.log_likelihood_confidence_intervals[0]
        # self.train1_train_dataset["log_likelihood_confidence_high"] = self.log_likelihood_confidence_intervals[1]

        # self.train1_train_dataset["izy_bound"] = self.izy_bound.cpu().numpy()
        # self.train1_train_dataset["izx_bound"] = self.izx_bound.cpu().numpy()
        # self.train1_train_dataset["reg_complexity"] = self.reg_complexity.cpu().numpy()

        self.test('test')
        self.test_accuracy_list.append(self.accuracy.detach())
        print('Final test accuracy: {:.3f}'.format(self.accuracy))
        self.train1_test_dataset = {}
        self.train1_test_dataset["accuracy"] = self.accuracy.cpu().numpy()

        print("Testing 1 is finished")

        
    def train_step1(self):
        self.set_mode('train')
        self.izy_relevance_epochs  = [] 
        self.izx_complexity_epochs = [] 
        self.reg_complexity_epochs = []

        if self.loss_id in [13,14]:
            optimization_mode = 'joint'
        else:
            optimization_mode = 'no_centers'

        for epoch in range(self.epoch_1):
            self.global_epoch += 1
            print('epoch:{}'.format(epoch))
            
             #### Perturbing the centers
            if self.loss_id != 0 and self.perturbation_flag:
                avg_diff_centers = (self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,self.M,1,self.K)-self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,1,self.M,self.K)).pow(2).mean((1,2,3,4)).sqrt()
                for view in range(self.n_views):
                    if avg_diff_centers[view] < self.average_distance_centers_means[view]/2:
                        eps = cuda(torch.randn(self.moving_mean_multiple_tensor[view].size()), self.cuda)
                        self.moving_mean_multiple_tensor[view] += eps* self.average_distance_centers_means[view]
            #### Perturbing the centers
            
            x_d = self.dataset_x['train'][self.global_epoch % self.num_datasets]
            y_d = self.dataset_y['train'][self.global_epoch % self.num_datasets]

            for idx in range(len(y_d)//self.batch_size):
                self.global_iter += 1
                
                x = cuda(torch.from_numpy(x_d[:,idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]).float(), self.cuda)
                y = cuda(torch.from_numpy(y_d[idx*self.batch_size:(idx+1)*self.batch_size]).type(torch.LongTensor), self.cuda)

                (mu, std), logit = self.IBnet(x)
                
                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss_joint = self.regularization_joint(mu, std,y,self.loss_id)
                total_loss = class_loss + self.beta_joint*info_loss_joint 

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                # to make centers suitable 
                if optimization_mode != 'no_centers' and self.loss_id in [13,14]:
                    center_means_expanded = cuda(torch.zeros(self.class_size,self.M**self.n_views,self.n_views,self.K),self.cuda) #[C,M^V,V,K]
                    center_vars_expanded = cuda(torch.zeros(center_means_expanded.shape),self.cuda)

                    for id_view in range(self.n_views):
                        center_means_expanded[:,:,id_view,:] = torch.gather(self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[id_view,:,:,:],1,self.index_matrix[id_view,:,:].unsqueeze(0).repeat(self.class_size,1,1)) 
                        center_vars_expanded[:,:,id_view,:] = torch.gather(self.moving_variance_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[id_view,:,:,:],1,self.index_matrix[id_view,:,:].unsqueeze(0).repeat(self.class_size,1,1)) 
                        
                    eps = Variable(cuda(center_vars_expanded.data.new(center_vars_expanded.size()).normal_(), cuda))
                    enc = (center_means_expanded+eps*center_vars_expanded.sqrt()).view(self.class_size,self.M**self.n_views,self.n_views*self.K) #[C,M^V,V*d]
                    logit_centers = self.IBnet.decoder_joint(enc)
                    
                    y_centers = cuda(torch.arange(self.class_size).unsqueeze(-1).repeat(1,self.M**self.n_views), self.cuda)

                    centers_loss = (self.moving_alpha_joint_multiple_tensor * F.cross_entropy(logit_centers.view(self.class_size*self.M**self.n_views,self.class_size),y_centers.view(self.class_size*self.M**self.n_views),reduction='none').view(self.class_size,self.M**self.n_views)).sum().div(self.class_size*math.log(2))
                    total_loss = centers_loss

                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()
            
            self.scheduler.step()
 
    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.IBnet.train()
            self.mode = 'train'
        elif mode == 'eval' :
            self.IBnet.eval()
            self.mode='eval'
        else : raise('mode error. It should be either train or eval')

    def update_centers_loss_joint(self,mu,var,y,gamma_values_KL,view,gamma_values_expectation = 0):
        
        gamma_indices = cuda(torch.arange(self.M).reshape((1,1,self.M)), self.cuda).repeat(y.shape[0],self.n_views,1) + self.M * y[:,None,None] #[B,V,M]
        
        moving_centers_multiple_gamma_KL = cuda(torch.zeros(y.shape[0],self.n_views, self.class_size * self.M), self.cuda).scatter_(-1, gamma_indices, gamma_values_KL) # [B,V,C*M]
        moving_centers_multiple_gamma_expectation = cuda(torch.zeros(y.shape[0],self.n_views, self.class_size * self.M), self.cuda).scatter_(-1, gamma_indices, gamma_values_expectation) # [B,V,C*M]
        
        beta_values_KL = moving_centers_multiple_gamma_KL.sum(dim=0)
        beta_values_expectation = moving_centers_multiple_gamma_expectation.sum(dim=0)

        if self.loss_id == 13:
            moving_centers_multiple_gamma = (moving_centers_multiple_gamma_KL+moving_centers_multiple_gamma_expectation)/2#.detach()
            beta_values = (beta_values_KL+beta_values_expectation)/2

            moving_centers_multiple_gamma_alt =  (2*moving_centers_multiple_gamma_KL+moving_centers_multiple_gamma_expectation)/3
            beta_values_alt = (2*beta_values_KL+beta_values_expectation)/3
        elif self.loss_id == 14:
            moving_centers_multiple_gamma = moving_centers_multiple_gamma_KL
            beta_values = beta_values_KL
            
            moving_centers_multiple_gamma_alt =  moving_centers_multiple_gamma_KL
            beta_values_alt = beta_values_KL

        center_weighted_mean_batch_normalized = torch.matmul(moving_centers_multiple_gamma_alt.permute(1,2,0).unsqueeze(-2), mu.unsqueeze(-2).permute(1,2,0,3)).squeeze(-2) / (beta_values_alt.unsqueeze(-1).repeat(1,1,self.K)+self.eps) #size [V,C*M,d] 
        
        if self.update_centers_coeff_mode == 'prop_M_b':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M * beta_values.unsqueeze(-1).repeat(1,1,self.K)
            coeff_new = 1 - coeff_old   

        elif self.update_centers_coeff_mode == 'prop_b':
            coeff_old = 1 - self.moving_coefficient_multiple * beta_values.unsqueeze(-1).repeat(1,1,self.K)
            coeff_new = 1 - coeff_old 

        elif self.update_centers_coeff_mode == 'prop_M':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M 
            coeff_new = 1 - coeff_old  

        elif self.update_centers_coeff_mode == 'not_prop':
            coeff_old = 1 - self.moving_coefficient_multiple *(beta_values.view(self.n_views,self.class_size,self.M).sum(-1)>4).int().view(self.n_views,self.class_size,1,1).repeat(1,1,self.M,self.K).view(self.n_views,self.class_size*self.M,self.K) # [V,C*M,d]
            coeff_new = 1 - coeff_old   

            if (beta_values.view(self.n_views,self.class_size,self.M).sum(-1)==0).int().sum()>0:
                print("At least one center has not been updated due to absence of any sample!")

        else:
            print("Invaid center coefficient update choice!")
            sys.exit()
              
        self.moving_mean_multiple_tensor *= coeff_old
        self.moving_mean_multiple_tensor += coeff_new * center_weighted_mean_batch_normalized
         
        if self.model_mode == 'stochastic':
            center_weighted_var = torch.matmul(moving_centers_multiple_gamma_KL.permute(1,2,0).unsqueeze(-2), var.unsqueeze(-2).permute(1,2,0,3)).squeeze(-2) #size [V,C*M,d] 
            center_weighted_var_batch_normalized_alternative = center_weighted_var / (beta_values_KL.unsqueeze(-1).repeat(1,1,self.K)+self.eps) #size [V,C*M,d]

            # Updating the variance
            coeff_var = self.moving_coefficient_var*(beta_values.view(self.n_views,self.class_size,self.M).sum(-1)>4).int().view(self.n_views,self.class_size,1,1).repeat(1,1,self.M,self.K).view(self.n_views,self.class_size*self.M,self.K)
            self.moving_variance_multiple_tensor *= (1-coeff_var)
            self.moving_variance_multiple_tensor += coeff_var * center_weighted_var_batch_normalized_alternative
            
        return

    def update_centers_loss_marginals(self,mu,var,y,gamma_values_KL,view,gamma_values_expectation = 0):
        
        gamma_indices = cuda(torch.arange(self.M).reshape((1,1,self.M)), self.cuda).repeat(y.shape[0],self.n_views,1) + self.M * y[:,None,None] #[B,V,M]
        
        moving_centers_multiple_gamma_KL = cuda(torch.zeros(y.shape[0],self.n_views, self.class_size * self.M), self.cuda).scatter_(-1, gamma_indices, gamma_values_KL) # [B,V,C*M]
        moving_centers_multiple_gamma_expectation = cuda(torch.zeros(y.shape[0],self.n_views, self.class_size * self.M), self.cuda).scatter_(-1, gamma_indices, gamma_values_expectation) # [B,V,C*M]
        
        beta_values_KL = moving_centers_multiple_gamma_KL.sum(dim=0)
        beta_values_expectation = moving_centers_multiple_gamma_expectation.sum(dim=0)

        if self.loss_id == 11:
            moving_centers_multiple_gamma = (moving_centers_multiple_gamma_KL+moving_centers_multiple_gamma_expectation)/2#.detach()
            beta_values = (beta_values_KL+beta_values_expectation)/2

            moving_centers_multiple_gamma_alt = (2*moving_centers_multiple_gamma_KL+moving_centers_multiple_gamma_expectation)/3
            beta_values_alt = (2*beta_values_KL+beta_values_expectation)/3
        elif self.loss_id == 12:
            moving_centers_multiple_gamma = moving_centers_multiple_gamma_KL
            beta_values = beta_values_KL
            
            moving_centers_multiple_gamma_alt = moving_centers_multiple_gamma_KL
            beta_values_alt = beta_values_KL

        center_weighted_mean_batch_normalized = torch.matmul(moving_centers_multiple_gamma_alt.permute(1,2,0).unsqueeze(-2), mu.unsqueeze(-2).permute(1,2,0,3)).squeeze(-2) / (beta_values_alt.unsqueeze(-1).repeat(1,1,self.K)+self.eps) #size [V,C*M,d] 
        

        if self.update_centers_coeff_mode == 'prop_M_b':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M * beta_values.unsqueeze(-1).repeat(1,1,self.K)
            coeff_new = 1 - coeff_old 

            coeff_old_alpha = coeff_old[:,:,0]
            coeff_new_alpha = coeff_new[:,:,0]

        elif self.update_centers_coeff_mode == 'prop_b':
            coeff_old = 1 - self.moving_coefficient_multiple * beta_values.unsqueeze(-1).repeat(1,1,self.K)
            coeff_new = 1 - coeff_old  

            coeff_old_alpha = coeff_old[:,:,0]
            coeff_new_alpha = coeff_new[:,:,0]

        elif self.update_centers_coeff_mode == 'prop_M':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M 
            coeff_new = 1 - coeff_old 

            coeff_old_alpha = coeff_old
            coeff_new_alpha = coeff_new

        elif self.update_centers_coeff_mode == 'not_prop':
            coeff_old = 1 - self.moving_coefficient_multiple *(beta_values.view(self.n_views,self.class_size,self.M).sum(-1)>4).int().view(self.n_views,self.class_size,1,1).repeat(1,1,self.M,self.K).view(self.n_views,self.class_size*self.M,self.K) # [V,C*M,d]
            coeff_new = 1 - coeff_old   

            if (beta_values.view(self.n_views,self.class_size,self.M).sum(-1)==0).int().sum()>0:
                print("At least one center has not been updated due to absence of any sample!")

            coeff_old_alpha = coeff_old[:,:,0]
            coeff_new_alpha = coeff_new[:,:,0]

        else:
            print("Invaid center coefficient update choice!")
            sys.exit()
              
        self.moving_mean_multiple_tensor *= coeff_old
        self.moving_mean_multiple_tensor += coeff_new * center_weighted_mean_batch_normalized
         
        if self.model_mode == 'stochastic':
            center_weighted_var = torch.matmul(moving_centers_multiple_gamma_KL.permute(1,2,0).unsqueeze(-2), var.unsqueeze(-2).permute(1,2,0,3)).squeeze(-2) #size [V,C*M,d] 
            center_weighted_var_batch_normalized_alternative = center_weighted_var / (beta_values_KL.unsqueeze(-1).repeat(1,1,self.K)+self.eps) #size [V,C*M,d]

            # Updating the variance
            coeff_var = self.moving_coefficient_var*(beta_values.view(self.n_views,self.class_size,self.M).sum(-1)>4).int().view(self.n_views,self.class_size,1,1).repeat(1,1,self.M,self.K).view(self.n_views,self.class_size*self.M,self.K)
            self.moving_variance_multiple_tensor *= (1-coeff_var)
            self.moving_variance_multiple_tensor += coeff_var * center_weighted_var_batch_normalized_alternative
        
       
        self.moving_alpha_marginals_multiple_tensor *= coeff_old_alpha
        self.moving_alpha_marginals_multiple_tensor += coeff_new_alpha* (beta_values.view(self.n_views,self.class_size,self.M) / (beta_values.view(self.n_views,self.class_size,self.M).sum(-1).unsqueeze(-1)+self.eps)).view(self.n_views,self.class_size*self.M)
        self.moving_alpha_marginals_multiple_tensor = (self.moving_alpha_marginals_multiple_tensor.view(self.n_views,self.class_size,self.M) / self.moving_alpha_marginals_multiple_tensor.view(self.n_views,self.class_size,self.M).sum(-1).unsqueeze(-1)).view(self.n_views,self.class_size*self.M)

        return
    
    def regularization_joint(self,mu, std,y,idx,reduction='mean'):
        if idx == 0:   # no regularization
            info_loss = cuda(torch.tensor(0.0),self.cuda)
        elif idx == 1:
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2)) 
        elif idx == 11 or idx == 12: # lossy marginals
            # select corresponding components
            centers_mean_label = self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[:,y,:,:].transpose(0,1)     # size [B,V, M, K]
            centers_var_label = self.moving_variance_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[:,y,:,:].transpose(0,1) # size [B,V, M, K]
            centers_alpha_marginals = self.moving_alpha_marginals_multiple_tensor.view(self.n_views,self.class_size,self.M)[:,y,:].transpose(0,1) # [B,V,M]
                        
            var_lossy = self.lossy_variance * cuda(torch.ones_like(mu),self.cuda)
            centers_var_label_lossy = self.lossy_variance * cuda(torch.ones_like(centers_var_label),self.cuda)
            var = std.pow(2)

            # compute the regularizer term
            if idx == 11:
                info_prod = KL_DG_DGM_prod(mu, var_lossy, centers_alpha_marginals, centers_mean_label, centers_var_label_lossy,multi_view_flag=False).sum()
                info_lossy_var = KL_lossy_DG_DGM_var(mu, var, centers_alpha_marginals, centers_mean_label, centers_var_label,multi_view_flag=False).sum()
                info_loss = (info_prod+info_lossy_var)/2
            elif idx == 12:
                info_loss = KL_lossy_DG_DGM_var(mu, var, centers_alpha_marginals, centers_mean_label, centers_var_label,multi_view_flag=False).sum()

            #update centers
            if self.mode == 'train':
                # compute weights gamma
                gamma_values_KL = contribution_lossy_Gaussian_to_GM(mu,var,centers_alpha_marginals,centers_mean_label,centers_var_label,'D_KL',multi_view_flag=False) # [B,V,M]
                gamma_values_expectation = contribution_Gaussian_to_GM(mu,var_lossy,centers_alpha_marginals,centers_mean_label,centers_var_label_lossy,'expectation',multi_view_flag=False)  # [B,V,M]

                self.update_centers_loss_marginals(mu.detach(),var.detach(),y.detach(),gamma_values_KL.detach(),gamma_values_expectation.detach())
              
        elif idx == 13 or idx == 14: # lossy Joints
            # select corresponding components
            centers_mean_label = self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[:,y,:,:].transpose(0,1)     # size [B,V, M, K]
            centers_var_label = self.moving_variance_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[:,y,:,:].transpose(0,1) # size [B,V, M, K]
            centers_alpha_joint = self.moving_alpha_joint_multiple_tensor.view(self.class_size,self.M**self.n_views)[y,:] # [B,M^V]
                        
            var_lossy = self.lossy_variance * cuda(torch.ones_like(mu),self.cuda)
            var = std.pow(2)

            center_means_expanded = cuda(torch.zeros(mu.size(0),self.n_views,self.M**self.n_views,self.K),self.cuda) #[B,V,M^V,K]
            center_vars_expanded = cuda(torch.zeros(center_means_expanded.shape),self.cuda) #[B,V,M^V,K]
            centers_var_label_lossy_expanded = self.lossy_variance * cuda(torch.ones_like(center_vars_expanded),self.cuda) #[B,V,M^V,K]

            ######### Compute distances
            mu_expanded = mu.unsqueeze(2).repeat(1,1,self.M,1)
            self.average_distance_centers_means = 0.05 * ((mu_expanded-centers_mean_label).pow(2).sum(-1).min(dim=-1)[0].mean(0).div(self.K).sqrt()).detach()/5 + 0.95 * self.average_distance_centers_means
            #########         

            for id_view in range(self.n_views):
                center_means_expanded[:,id_view,:,:] = torch.gather(centers_mean_label[:,id_view,:,:],1,self.index_matrix[id_view,:,:].unsqueeze(0).repeat(mu.size(0),1,1)) 
                center_vars_expanded[:,id_view,:,:] = torch.gather(centers_var_label[:,id_view,:,:],1,self.index_matrix[id_view,:,:].unsqueeze(0).repeat(mu.size(0),1,1)) 

            # compute the regularizer term   
            if idx == 13:
                info_prod = KL_DG_DGM_prod(mu, var_lossy, centers_alpha_joint, center_means_expanded, centers_var_label_lossy_expanded,multi_view_flag=True).sum()
                info_lossy_var = KL_lossy_DG_DGM_var(mu, var+self.lossy_variance_var, centers_alpha_joint, center_means_expanded, center_vars_expanded+self.lossy_variance_var,multi_view_flag=True).sum()
                info_loss = (info_prod+info_lossy_var)/2

            elif idx == 14:
                info_loss = KL_lossy_DG_DGM_var(mu, var+self.lossy_variance_var, centers_alpha_joint, center_means_expanded, center_vars_expanded+self.lossy_variance_var,multi_view_flag=True).sum()
                   
            #update centers
            if self.mode == 'train':
                # compute weights gamma
                gamma_values_KL_joint = contribution_lossy_Gaussian_to_GM(mu,var+self.lossy_variance_var,centers_alpha_joint,center_means_expanded,center_vars_expanded+self.lossy_variance_var,'D_KL',multi_view_flag=True) # [B,M^V]
                gamma_values_expectation_joint = contribution_Gaussian_to_GM(mu,var_lossy,centers_alpha_joint,center_means_expanded,centers_var_label_lossy_expanded,'expectation',multi_view_flag=True)  # [B,M^V]

                gamma_values_KL = compute_marginals(gamma_values_KL_joint) # [B,V,M]
                gamma_values_expectation= compute_marginals(gamma_values_expectation_joint) # [B,V,M]

                self.update_centers_loss_joint(mu.detach(),var.detach(),y.detach(),gamma_values_KL.detach(),gamma_values_expectation.detach())
                self.update_joint_alpha(y,gamma_values_KL_joint.detach(),gamma_values_expectation_joint.detach())

        if reduction == 'sum':
            return info_loss
        elif reduction == 'mean':
            return info_loss.div(y.size(0))
    
    def update_joint_alpha(self,y,gamma_values_KL_joint,gamma_values_expectation_joint):
        
        gamma_values_KL_joint_expanded = cuda(torch.zeros(y.shape[0],self.class_size, self.M**self.n_views), self.cuda) #[B,C,M^V]
        gamma_values_KL_joint_expanded.scatter_(1,y[:,None,None].repeat(1,self.class_size, self.M**self.n_views),gamma_values_KL_joint.unsqueeze(1).repeat(1,self.class_size,1))
        beta_values_KL = gamma_values_KL_joint_expanded.sum(dim=0) #[C,M^V]
        
        gamma_values_expectation_joint_expanded = cuda(torch.zeros(y.shape[0],self.class_size, self.M**self.n_views), self.cuda)
        gamma_values_expectation_joint_expanded.scatter_(1,y[:,None,None].repeat(1,self.class_size, self.M**self.n_views),gamma_values_expectation_joint.unsqueeze(1).repeat(1,self.class_size,1)) #[B,C,M^V]
        beta_values_expectation = gamma_values_expectation_joint_expanded.sum(dim=0) #[C,M^V]
        
        if self.loss_id == 13:
            beta_values = (beta_values_KL+beta_values_expectation)/2
        elif self.loss_id == 14:
            beta_values = beta_values_KL

        if self.update_centers_coeff_mode == 'prop_M_b':
            coeff_old_alpha = self.moving_coefficient_multiple * self.M * beta_values
            coeff_new_alpha = 1 - coeff_old_alpha

        elif self.update_centers_coeff_mode == 'prop_b':
            coeff_old_alpha = self.moving_coefficient_multiple * beta_values
            coeff_new_alpha = 1 - coeff_old_alpha

        elif self.update_centers_coeff_mode == 'prop_M':
            coeff_old_alpha = self.moving_coefficient_multiple * self.M 
            coeff_new_alpha = 1 - coeff_old_alpha

        elif self.update_centers_coeff_mode == 'not_prop':
            coeff_old_alpha = 1 - self.moving_coefficient_multiple * (beta_values.sum(-1)>4).int().view(self.class_size,1).repeat(1,self.M**self.n_views) #* (beta_values>1e-15).int()
            coeff_new_alpha = 1 -coeff_old_alpha

        else:
            print("Invaid center coefficient update choice!")
            sys.exit()
			
        self.moving_alpha_joint_multiple_tensor  *= coeff_old_alpha
        self.moving_alpha_joint_multiple_tensor  += coeff_new_alpha* beta_values / (beta_values.sum(-1).unsqueeze(-1)+self.eps)
        # We normalize the updated alpha_r
        self.moving_alpha_joint_multiple_tensor = (self.moving_alpha_joint_multiple_tensor / self.moving_alpha_joint_multiple_tensor.sum(-1).unsqueeze(-1))

        return
    
    def test(self, dataloader_type, bootstrap = False, save_ckpt=True):
        self.set_mode('eval')
        
        x_d = self.dataset_x[dataloader_type][self.counter % self.num_datasets]
        y_d = self.dataset_y[dataloader_type][self.counter % self.num_datasets]
        
        total_num, correct, cross_entropy, zx_complexity, reg_complexity = cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)
        
        for idx in range(len(y_d)//self.batch_size):
            x = cuda(torch.from_numpy(x_d[:,idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]).float(), self.cuda)
            y = cuda(torch.from_numpy(y_d[idx*self.batch_size:(idx+1)*self.batch_size]).type(torch.LongTensor), self.cuda)

            total_num += y.size(0)

            (mu, std), soft_logit = self.IBnet(x,self.num_avg)
            
            if self.num_avg > 1 and self.model_mode == 'stochastic':
                # cross entropy
                cross_entropy += sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                # accuracy
                predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                correct    += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
            else:
                # cross entropy
                cross_entropy += F.cross_entropy(soft_logit, y, reduction='sum').detach()
                # accuracy
                prediction = soft_logit.max(1)[1]
                correct   += torch.eq(prediction,y).float().sum().detach()
            # complexity
            # zx_complexity  += sum([self.regularization_marginal(mu[k], std[k], y, 1, k, reduction = 'sum').detach() for k in range(self.n_views)])
            # reg_complexity += sum([self.regularization_marginal(mu[k], std[k], y, self.loss_id, k, reduction = 'sum').detach() for k in range(self.n_views)])
        # some statistics at the end of testing
        if self.model_mode == 'stochastic':
            const_den = self.num_avg
        else:
            const_den = 1

        self.accuracy       = correct/total_num/const_den
        self.log_likelihood = -cross_entropy/total_num/const_den
        self.izy_bound      = self.HY - cross_entropy/total_num/const_den/math.log(2)
        self.izx_bound      = zx_complexity/total_num
        self.reg_complexity = reg_complexity/total_num
        if bootstrap:
            self.bootstrap_confidence_intervals(dataloader_type = dataloader_type + '_bootstrap', confidence = 0.95, sample_size=1000, repetitions=100)

    def initialize_centers(self, initialization_mode, dataset_name, dset_dir):
        if self.loss_id in [11,12,13,14]:
            if initialization_mode == 'fixed':
                self.moving_mean_multiple_tensor     = cuda(torch.zeros(self.n_views,self.class_size*self.M,self.K),self.cuda) #[V,C * M, d]
                self.moving_variance_multiple_tensor = 0.1*cuda(torch.ones(self.n_views,self.class_size*self.M,self.K),self.cuda)  #[V,C * M, d]
            elif initialization_mode == 'random':
                self.moving_mean_multiple_tensor     = 0.1*cuda(torch.randn(self.n_views,self.class_size*self.M,self.K),self.cuda) #[V,C * M, d]
                self.moving_variance_multiple_tensor = 0.08*cuda(torch.ones(self.n_views,self.class_size*self.M,self.K),self.cuda)+0.02*cuda(torch.randn(self.n_views,self.class_size*self.M,self.K),self.cuda)  #[V,C * M, d]
            elif initialization_mode == 'scattered':
                self.moving_mean_multiple_tensor     = cuda(torch.zeros(self.n_views,self.class_size*self.M,self.K),self.cuda) #[V,C * M, d]
                self.moving_variance_multiple_tensor = 0.1*cuda(torch.ones(self.n_views,self.class_size*self.M,self.K),self.cuda)  #[V,C * M, d]

                x_d = self.dataset_x['train'][0]
                y_d = self.dataset_y['train'][0]
                batch_size_ini = min(self.class_size*200,int(self.train_size/self.M/2),1000)

                counter_chosen_centers = 0

                temp_flag = True 
                classes_labels = cuda(torch.arange(self.class_size), self.cuda)

                while (counter_chosen_centers < self.M) and temp_flag == True:

                    for idx in range(len(y_d)//batch_size_ini):

                        x = cuda(torch.from_numpy(x_d[:,idx*batch_size_ini:(idx+1)*batch_size_ini,:,:,:]).float(), self.cuda)
                        y = cuda(torch.from_numpy(y_d[idx*batch_size_ini:(idx+1)*batch_size_ini]).type(torch.LongTensor), self.cuda)

                        # We check if the batch contains at least one sample from each class
                        if y.unique().size(0) != self.class_size:
                            break 

                        if counter_chosen_centers == 0:
                            running_M = 1
                        else:
                            running_M = counter_chosen_centers 

                        (mu, _), _ = self.IBnet(x) 

                        centers_mean_label   = self.moving_mean_multiple_tensor.view(self.n_views,self.class_size,self.M,self.K)[:,y,:running_M,:].transpose(0,1) # size [B1, V, running_M, d]
                        centers_selected_ind = ((centers_mean_label-mu.unsqueeze(2).repeat(1,1,running_M,1)).pow(2)).sum(-1).argmin(dim=-1) +( y.unsqueeze(-1).repeat(1,self.n_views)*self.M) # size [B1,V]        
                        
                        center_mean_selected = cuda(torch.zeros(mu.size(0),self.n_views,self.K),self.cuda) # size [B1,V, K]

                        for id_V in torch.arange(self.n_views):
                            center_mean_selected[:,id_V,:] = self.moving_mean_multiple_tensor[id_V,centers_selected_ind[:,id_V],:]
                            
                        min_distances = (center_mean_selected-mu).pow(2).sum(-1).sum(-1)  # size[B1]
                    
                        for ilabels in classes_labels:
                            index_i = (y == ilabels).nonzero(as_tuple=True)[0]
                            for id_V in range(self.n_views):
                                self.moving_mean_multiple_tensor[id_V,:,:].view(self.class_size,self.M,self.K)[ilabels,counter_chosen_centers,:] = mu[index_i[torch.multinomial(min_distances[index_i],1)],id_V,:].detach()

                        counter_chosen_centers += 1
                        if counter_chosen_centers > self.M-1:
                            temp_flag = False
                            break
        
        elif self.loss_id not in [0,1]:
            print("Wrong center initializationg!")
            sys.exit()

        # Gaussian Mixture parameters
        if self.loss_id in [13,14]:
            self.moving_alpha_joint_multiple_tensor = cuda(torch.ones(self.class_size,self.M**self.n_views),self.cuda)/self.M**self.n_views  #[C,M^V]
        elif self.loss_id in [11,12]:
            self.moving_alpha_marginals_multiple_tensor = cuda(torch.ones(self.n_views,self.class_size*self.M),self.cuda)/(self.M)  #[V,C*M]
       
        return

    def update_centers(self,mu,var,y,gamma_values,view):
        gamma_indices = cuda(torch.arange(self.M).reshape((1,self.M)), self.cuda).repeat(y.shape[0],1) + self.M * y[:,None]#.detach()
        moving_centers_multiple_gamma = cuda(torch.zeros(y.shape[0], self.class_size * self.M), self.cuda).scatter_(1, gamma_indices, gamma_values)#.detach()
        beta_values = moving_centers_multiple_gamma.sum(dim=0)
        center_weighted_mean_batch_normalized = torch.matmul(moving_centers_multiple_gamma.transpose(0,1), mu) / (beta_values.unsqueeze(-1).repeat(1,self.K)+self.eps)#.detach()                            # size [C*M,K] 
                   
        if self.update_centers_coeff_mode == 'prop_M_b':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M * beta_values.unsqueeze(1).repeat(1,self.K)
            coeff_new = 1 - coeff_old  

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]
        elif self.update_centers_coeff_mode == 'prop_b':
            coeff_old = 1 - self.moving_coefficient_multiple * beta_values.unsqueeze(1).repeat(1,self.K)
            coeff_new = 1 - coeff_old 

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]
        elif self.update_centers_coeff_mode == 'prop_M':
            coeff_old = 1 - self.moving_coefficient_multiple * self.M 
            coeff_new = 1 - coeff_old  

            coeff_old_alpha = coeff_old
            coeff_new_alpha = coeff_new
        elif self.update_centers_coeff_mode == 'not_prop':
            coeff_old = 1 - self.moving_coefficient_multiple *(beta_values.view(self.class_size,self.M).sum(-1)>4).int().view(self.class_size,1,1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)
            coeff_new = 1-coeff_old

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]
                
        elif self.update_centers_coeff_mode == 'prop_b_logistic':
            coeff_logistic = 1/(1+(-self.temp_coeff*(beta_values.view(self.class_size,self.M).sum(-1)-self.batch_size/(2*self.class_size))).exp()) #1/(1+exp(-c*(x-x_0)))

            coeff_old = 1 - coeff_logistic.unsqueeze(-1).unsqueeze(-1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)
            coeff_new = 1 - coeff_old 

            coeff_old_alpha = coeff_old[:,0]
            coeff_new_alpha = coeff_new[:,0]

        else:
            print("Invaid center coefficient update choice!")
            sys.exit()
 
        self.moving_mean_multiple_tensor[view] *= coeff_old
        self.moving_mean_multiple_tensor[view] += coeff_new * center_weighted_mean_batch_normalized
       
        if self.model_mode == 'stochastic':
            center_weighted_mean_diff_batch = (moving_centers_multiple_gamma.unsqueeze(-1)*(self.moving_mean_multiple_tensor.unsqueeze(0).detach()-mu.unsqueeze(1)).pow(2)).sum(0) #.detach()
            center_weighted_var = torch.matmul(moving_centers_multiple_gamma.transpose(0,1), var)
                
            if self.update_centers_rule == 'expectation':
                center_weighted_var_batch = center_weighted_mean_diff_batch - center_weighted_var
            elif self.update_centers_rule == 'D_KL':
                center_weighted_var_batch = center_weighted_mean_diff_batch + center_weighted_var
            else:
                print("Invaid update choice!")
                sys.exit()

            center_weighted_var_batch_normalized = torch.clamp(center_weighted_var_batch, min=0) / (beta_values.unsqueeze(-1).repeat(1,self.K)+self.eps)
            center_weighted_var_batch_normalized_alternative = center_weighted_var / (beta_values.unsqueeze(-1).repeat(1,self.K)+self.eps)

            # Updating the variance
            ceeff_var = self.moving_coefficient_var*(beta_values.view(self.class_size,self.M).sum(-1)>4).int().view(self.class_size,1,1).repeat(1,self.M,self.K).view(self.class_size*self.M,self.K)
            self.moving_variance_multiple_tensor[view] *= (1-ceeff_var)

            self.moving_variance_multiple_tensor[view] += ceeff_var * center_weighted_var_batch_normalized_alternative
          
        self.moving_alpha_multiple_tensor[view] *= coeff_old_alpha
        
        self.moving_alpha_multiple_tensor[view] += coeff_new_alpha* (beta_values.view(self.class_size,self.M) / (beta_values.view(self.class_size,self.M).sum(-1).unsqueeze(-1)+self.eps)).view(self.class_size*self.M)
        self.moving_alpha_multiple_tensor[view] = (self.moving_alpha_multiple_tensor.view(self.class_size,self.M) / self.moving_alpha_multiple_tensor.view(self.class_size,self.M).sum(1).unsqueeze(-1)).view(self.class_size*self.M)
        
        return
    
    def bootstrap_confidence_intervals(self, dataloader_type, confidence = 0.95, sample_size=1000, repetitions=100):
        accuracies, log_likelihoods_av = [], []

        if dataloader_type == 'train_bootstrap': dataset_type = 'train'
        else: dataset_type = 'test' 
        x_d = self.dataset_x[dataset_type][self.counter % self.num_datasets]
        y_d = self.dataset_y[dataset_type][self.counter % self.num_datasets]

        # repeat repetitions time
        for rep in range(repetitions):
            total_num, correct, log_likelihood = cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)     
            # take randomly samples from the dataset
            for idx in range(len(y_d)//self.batch_size):
                x = cuda(torch.from_numpy(x_d[:,idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]).float(), self.cuda)
                y = cuda(torch.from_numpy(y_d[idx*self.batch_size:(idx+1)*self.batch_size]).type(torch.LongTensor), self.cuda)

                total_num += y.size(0)
                _, soft_logit = self.IBnet(x, self.num_avg)
                if self.num_avg > 1 and self.model_mode == 'stochastic':
                    # log_likelihood
                    log_likelihood -= sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                    # accuracy
                    predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                    correct += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
                else:
                    # log_likelihood
                    log_likelihood -= F.cross_entropy(soft_logit, y, reduction='sum').detach()
                    # accuracy
                    prediction = soft_logit.max(1)[1]
                    correct += torch.eq(prediction,y).float().sum().detach()
          
                # terminate if processed more than sample_size
                if idx*self.batch_size + 1 > sample_size: break
            # compute accuracy
            accuracy = correct/total_num/self.num_avg
            accuracies.append(accuracy)
            # compute average log_likelihood
            log_likelihood_av = log_likelihood/total_num/self.num_avg
            log_likelihoods_av.append(log_likelihood_av)

        # compute confidence intervals
        accuracy_confidence_intervals = st.norm.interval(confidence=confidence, loc=np.mean(torch.asarray(accuracies).cpu().numpy()), scale=st.sem(torch.asarray(accuracies).cpu().numpy()))
        log_likelihood_confidence_intervals = st.norm.interval(confidence=confidence, loc=np.mean(torch.asarray(log_likelihoods_av).cpu().numpy()), scale=st.sem(torch.asarray(log_likelihoods_av).cpu().numpy()))
        
        # output  
        self.accuracies_bootstrap = torch.asarray(accuracies)
        self.accuracy_confidence_intervals = accuracy_confidence_intervals
        self.log_likelihoods_bootstrap = torch.asarray(log_likelihoods_av)
        self.log_likelihood_confidence_intervals = log_likelihood_confidence_intervals

    def plot_centers(self, FigLabel, save_ckpt=True):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))  

        for k, ax in zip(range(self.n_views),axs.flat):
            cCenters = torch.transpose(torch.matmul(self.matrix_A,torch.transpose(self.moving_mean_multiple_tensor[k][torch.tensor(self.M*FigLabel+np.arange(self.M), dtype=torch.int64),:],0,1)),0,1)
            dfCenter = pd.DataFrame({'x': cCenters[:,0].detach().cpu().numpy(), 'y': cCenters[:,1].detach().cpu().numpy()})
            ax.scatter(dfCenter.x,dfCenter.y,c='k',marker='X', s=20)
        
        fig.savefig(self.figures_path+'/Fig_center{:,.0e}M{:,.0f}Label{:,.0e}K{:,.0f}epoch{:}_timestamp{}.png'.format(self.beta_joint,self.M,FigLabel,self.K,self.global_epoch,self.timestamp))
        del fig
        plt.close()
        

