import torch
import math
import sys 
import numpy as np

M_value = 1
eps_value = 1e-15
lossy_variance_value = 1
cuda_value = False

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


def give_next_index(id_M_V,remain_V,running_index,mat_matrix):
    if running_index == 1:
        mat_matrix[0,id_M_V,:]=remain_V
    else:
        mat_matrix[-1,id_M_V,:] = remain_V % M_value
        give_next_index(id_M_V,(remain_V-mat_matrix[-1,id_M_V,:])//M_value,running_index-1,mat_matrix[:-1,:,:])


def set_constants_utils(value_K,value_M,value_eps,value_lossy_variance,value_n_views,value_cuda):
    global M_value
    global eps_value
    global lossy_variance_value
    global cuda_value
    global n_views
    global index_matrix
    global index_list_inverse

    M_value = value_M
    eps_value = value_eps
    lossy_variance_value = value_lossy_variance
    cuda_value = value_cuda
    n_views = value_n_views


    index_matrix = cuda(torch.zeros(n_views,M_value**n_views,value_K,dtype=torch.long),cuda_value)
    index_list_inverse = []

    for id_M_V in range(M_value**n_views):
        give_next_index(id_M_V,id_M_V,n_views,index_matrix)

    
    for id_V in range(n_views):
        per_view_tensor = cuda(torch.zeros(M_value,M_value**(n_views-1),dtype=torch.long),cuda_value)
        for id_M in range(M_value):
            per_view_tensor[id_M,:] = (index_matrix[id_V,:,0]==id_M).nonzero(as_tuple=True)[0]
        
        index_list_inverse.append(per_view_tensor)


def KL_DG_DGM_prod(mu, var, alpha, center_means, center_vars,multi_view_flag=False):  # Product-approximation of KL divergence of adigaonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    
    L_P_P = -0.5 * (2*torch.pi*torch.e*var).log().sum(-1).sum(-1) # size [B]: This is -h(P) 
 
    if not multi_view_flag: #len(mu.size()) == 2:
        if len(mu.size())==2:
            mu_expanded = mu.unsqueeze(-2).repeat(1,M_value,1)
            var_expanded = var.unsqueeze(1).repeat(1,M_value,1)
        elif len(mu.size())==3:
            L_P_P = -0.5 * (2*torch.pi*torch.e*var).log().sum(-1) # size [B,V]: This is -h(P) 
            mu_expanded = mu.unsqueeze(-2).repeat(1,1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,1,M_value,1)
        t_values_log = alpha.log() + log_expect_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M] or [B,V,M]
    else: # len(mu.size()) == 3:
        mu_expanded = mu.unsqueeze(2).repeat(1,1,alpha.size(1),1)
        var_expanded = var.unsqueeze(2).repeat(1,1,alpha.size(1),1)

        t_values_log = alpha.log() + log_expect_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1) # size [B, M^V]
    
    L_P_Q = torch.logsumexp(t_values_log,-1) # size [B]: This is a bound on E_p[log(Q)]
    return (L_P_P-L_P_Q)


def KL_lossy_DG_DGM_prod(mu, var, alpha, center_means, center_vars):  # Product-approximation of "Lossy" KL divergence of a diagonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    all_one_tensor = lossy_variance_value * cuda(torch.ones_like(var),cuda_value)
    
    L_P_P_1 = -0.5 * (2*torch.pi*torch.e*all_one_tensor).log().sum(-1) # size [B]: This is -h(P_1) 
    L_P_P_2 = -0.5 * (2*torch.pi*torch.e*var).log().sum(-1) # size [B]: This is -h(P_2) 

    #This is calculated two times (in contribution_Gaussian_to_GM also); maybe we can optimize it
    if len(mu.size()) == 2:
        mu_expanded = mu.unsqueeze(1).repeat(1,M_value,1)
        var_expanded = var.unsqueeze(1).repeat(1,M_value,1)
    elif len(mu.size()) == 3:
        mu_expanded = mu.unsqueeze(2).repeat(1,1,M_value,1)
        var_expanded = var.unsqueeze(2).repeat(1,1,M_value,1)

    t_values_log = alpha.log() + log_expect_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M] or [B, H, M]
    
    L_P_Q = torch.logsumexp(t_values_log,-1) # size [B]: This is a bound on E_p[log(Q)]
    return (L_P_P_1+L_P_P_2-L_P_Q)


def KL_DG_DGM_var(mean, var, alpha, center_means, center_vars): # Variation-approximation of KL divergence of a diagonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    if len(mean.size()) == 2:
        mu_expanded = mean.unsqueeze(1).repeat(1,M_value,1)
        var_expanded = var.unsqueeze(1).repeat(1,M_value,1)
    elif len(mean.size()) == 3:
        mu_expanded = mean.unsqueeze(2).repeat(1,1,M_value,1)
        var_expanded = var.unsqueeze(2).repeat(1,1,M_value,1)
    
    D_var = -torch.logsumexp(alpha.log()-KL_DG_DG(mu_expanded,var_expanded,center_means,center_vars),-1) # size [B] or [B , H] in multi-head
    return D_var


def KL_lossy_DG_DGM_var(mean, var, alpha, center_means, center_vars,multi_view_flag=False): # Variation-approximation of "Lossy" KL divergence of a diagonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    if not multi_view_flag: #len(mu.size()) == 2:
        if len(mean.size())==2:
            mu_expanded = mean.unsqueeze(-2).repeat(1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,M_value,1)
        elif len(mean.size())==3:
            mu_expanded = mean.unsqueeze(-2).repeat(1,1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,1,M_value,1)
       
        D_lossy_var = -torch.logsumexp(alpha.log()-KL_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars),-1) # size [B] or [B,V]
    else: # len(mu.size()) == 3:
        mu_expanded = mean.unsqueeze(2).repeat(1,1,alpha.size(1),1)
        var_expanded = var.unsqueeze(2).repeat(1,1,alpha.size(1),1)

        D_lossy_var = -torch.logsumexp(alpha.log()-KL_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1),-1) # size [B]

    return D_lossy_var


def KL_lossy_G_GM_prod(mean, var, alpha, center_means, center_vars, model_mode): # Variation-approximation of "Lossy" KL divergence of a diagonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    D_lossy_prod = 0 # size [B , H]
    return D_lossy_prod


def KL_lossy_G_GM_var(mean, var, alpha, center_means, center_vars, model_mode): # Variation-approximation of "Lossy" KL divergence of a diagonal Multivariate Gaussian with a diagonal Gaussian Mixture distributions 
    D_lossy_var = 0 # size [B , H]
    return D_lossy_var


def log_expect_DG_DG(mu1,var1,mu2,var2):  # Compute log(E_P[Q]), where P=N(mu1,diag(var1)) and Q=N(mu2,diag(var2))
    log_expect = ((-0.5 * (mu2-mu1).pow(2) / (var2+var1+eps_value))-0.5*(2*torch.pi*(var2+var1+eps_value)).log()).sum(-1)
    return log_expect

def log_expect_lossy_DG_DG(mu1,var1,mu2,var2):  # Compute log(E_P[Q]) in a "lossy manner", where P=N(mu1,diag(var1)) and Q=N(mu2,diag(var2))
    all_zero_tensor = cuda(torch.zeros_like(mu1),cuda_value)
    all_one_tensor = lossy_variance_value * cuda(torch.ones_like(var1),cuda_value)
        
    log_expect_lossy = (log_expect_DG_DG(mu1,all_one_tensor,mu2,all_one_tensor)\
                        +log_expect_DG_DG(all_zero_tensor,var1,all_zero_tensor,var2))
    return log_expect_lossy


def KL_DGM_DGM_prod(alpha_1, centers_means_1, centers_vars_1, alpha_2, centers_means_2, centers_vars_2): # Product-approximation of two diagonal Gaussian Mixture distributions, first one with B centers, second one with M centers
    t_values_log_a_alpha = alpha_1.log().unsqueeze(0).repeat(alpha_1.size(0),1) + log_expect_DG_DG(centers_means_1.unsqueeze(0),centers_vars_1.unsqueeze(0),centers_means_1.unsqueeze(1),centers_vars_1.unsqueeze(1)) # size [B,B]
    t_values_log_a = alpha_1 * torch.logsumexp(t_values_log_a_alpha,-1) # size [B]:
    L_P_P =  t_values_log_a.sum() # size [1]: This is a bound on L_P_P
        
    mu_expanded=centers_means_1.unsqueeze(1).repeat(1,M_value,1)
    var_expanded=centers_vars_1.unsqueeze(1).repeat(1,M_value,1)
    
    p_values_log_a_b = alpha_2.log() + log_expect_DG_DG(mu_expanded,var_expanded,centers_means_2,centers_vars_2) # size [B, M]
    p_values_log_a = alpha_1 * torch.logsumexp(p_values_log_a_b,-1) # size [B]:
    L_P_Q = p_values_log_a.sum() # size [1]: This is a bound on L_P[Q]
    
    return (L_P_P-L_P_Q)

def KL_DGM_DGM_var(alpha_1, centers_means_1, centers_vars_1, alpha_2, centers_means_2, centers_vars_2): # Variation-approximation of KL divergence of two Diagonal Gaussian Mixture distributions
    t_values_log_a_alpha = alpha_1.log().unsqueeze(0).repeat(alpha_1.size(0),1) - KL_DG_DG(centers_means_1.unsqueeze(0),centers_vars_1.unsqueeze(0),centers_means_1.unsqueeze(1),centers_vars_1.unsqueeze(1)) # size [B,B]
    t_values_log_a = alpha_1 * torch.logsumexp(t_values_log_a_alpha,-1) # size [B]:
    L_P_P =  t_values_log_a.sum() # size [1]: This is a bound on L_P_P
   
    mu_expanded=centers_means_1.unsqueeze(1).repeat(1,M_value,1)
    var_expanded=centers_vars_1.unsqueeze(1).repeat(1,M_value,1)
    
    p_values_log_a_b = alpha_2.log() - KL_DG_DG(mu_expanded,var_expanded,centers_means_2,centers_vars_2) # size [B, M]
    p_values_log_a = alpha_1 * torch.logsumexp(p_values_log_a_b,-1) # size [B]:
    L_P_Q = p_values_log_a.sum() # size [1]: This is a bound on L_p[Q]
    
    return (L_P_P-L_P_Q)

def KL_lossy_DGM_DGM_prod(alpha_1, centers_means_1, centers_vars_1, alpha_2, centers_means_2, centers_vars_2): # Product-approximation of two Diagonal Gaussian Mixture distributions 
    #FIXME: to check
    t_values_log_a_alpha = alpha_1.log().unsqueeze(0).repeat(alpha_1.size(0),1) + log_expect_lossy_DG_DG(centers_means_1.unsqueeze(0),centers_vars_1.unsqueeze(0),centers_means_1.unsqueeze(1),centers_vars_1.unsqueeze(1)) # size [B,B]
    t_values_log_a = alpha_1 * torch.logsumexp(t_values_log_a_alpha,-1) # size [B]:
    L_P_P =  t_values_log_a.sum() # size [1]: This is a bound on L_P_P

    mu_expanded=centers_means_1.unsqueeze(1).repeat(1,M_value,1)
    var_expanded=centers_vars_1.unsqueeze(1).repeat(1,M_value,1)
    
    p_values_log_a_b = alpha_2.log() + log_expect_lossy_DG_DG(mu_expanded,var_expanded,centers_means_2,centers_vars_2) # size [B, M]
    p_values_log_a = alpha_1 * torch.logsumexp(p_values_log_a_b,-1) # size [B]:
    L_P_Q = p_values_log_a.sum() # size [1]: This is a bound on L_P[Q]
    
    return (L_P_P-L_P_Q)   
    

def KL_lossy_DGM_DGM_var(alpha_1, centers_means_1, centers_vars_1, alpha_2, centers_means_2, centers_vars_2): # Variation-approximation of KL divergence of two Diagonal Gaussian Mixture distributions
    #FIXME: to check
    t_values_log_a_alpha = alpha_1.log().unsqueeze(0).repeat(alpha_1.size(0),1) - KL_lossy_DG_DG(centers_means_1.unsqueeze(0),centers_vars_1.unsqueeze(0),centers_means_1.unsqueeze(1),centers_vars_1.unsqueeze(1)) # size [B,B]
    t_values_log_a = alpha_1 * torch.logsumexp(t_values_log_a_alpha,-1) # size [B]:
    L_P_P =  t_values_log_a.sum() # size [1]: This is a bound on L_P_P
        
    mu_expanded=centers_means_1.unsqueeze(1).repeat(1,M_value,1)
    var_expanded=centers_vars_1.unsqueeze(1).repeat(1,M_value,1)
    
    p_values_log_a_b = alpha_2.log() - KL_lossy_DG_DG(mu_expanded,var_expanded,centers_means_2,centers_vars_2) # size [B, M]
    p_values_log_a = alpha_1 * torch.logsumexp(p_values_log_a_b,-1) # size [B]:
    L_P_Q = p_values_log_a.sum() # size [1]: This is a bound on L_p[Q]
    
    return (L_P_P-L_P_Q)


def KL_DG_DG(mean1,var1,mean2,var2):  # KL divergence of two Multivariate Gaussian distributions with diagonal covariance matrices
    D_KL = (- 0.5*(1+var1.log()) \
                    + 0.5 * var2.log() \
                    + 0.5 * (mean2-mean1).pow(2)/(var2+eps_value) \
                    + 0.5 * var1/(var2+eps_value)).sum(-1).div(math.log(2)) 
    return D_KL


def KL_lossy_DG_DG(mean1,var1,mean2,var2):  # "Lossy" KL divergence of two Multivariate Gaussian distributions with diagonal covariance matrices
    all_zero_tensor = cuda(torch.zeros_like(mean1),cuda_value)
    all_one_tensor = lossy_variance_value * cuda(torch.ones_like(var1),cuda_value)
    
    D_KL_lossy = KL_DG_DG(mean1,all_one_tensor,mean2,all_one_tensor) \
                    +KL_DG_DG(all_zero_tensor,var1,all_zero_tensor,var2)
    
    return D_KL_lossy


def contribution_Gaussian_to_GM(mu,var,alpha,center_means,center_vars,update_centers_rule,multi_view_flag=False):
    if not multi_view_flag: #len(mu.size()) == 2:
        if len(mu.size()) == 2:
            mu_expanded = mu.unsqueeze(-2).repeat(1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,M_value,1)
        elif len(mu.size()) == 3:
            mu_expanded = mu.unsqueeze(-2).repeat(1,1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,1,M_value,1)

        if update_centers_rule == 'expectation':
            gamma_values_log = alpha.log() + log_expect_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M]
        elif update_centers_rule == 'D_KL':
            gamma_values_log = alpha.log() - KL_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M]
        else:
            print("Invaid update choice!")
            sys.exit()
        
        gamma_values = (gamma_values_log-torch.logsumexp(gamma_values_log,-1).unsqueeze(-1)).exp() # size [B, M] 
   
    else: # len(mu.size()) == 3:
        mu_expanded = mu.unsqueeze(2).repeat(1,1,alpha.size(1),1)
        var_expanded = var.unsqueeze(2).repeat(1,1,alpha.size(1),1)

        if update_centers_rule == 'expectation':
            gamma_values_log = alpha.log() + log_expect_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1) # size [B, M^V]
        elif update_centers_rule == 'D_KL':
            gamma_values_log = alpha.log() - KL_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1) # size [B, M^V]
        else:
            print("Invaid update choice!")
            sys.exit()
        
        gamma_values = (gamma_values_log-torch.logsumexp(gamma_values_log,-1).unsqueeze(-1)).exp() # size [B, M^V]

    return gamma_values


def contribution_lossy_Gaussian_to_GM(mu,var,alpha,center_means,center_vars,update_centers_rule,multi_view_flag=False):
    
    if not multi_view_flag: #len(mu.size()) == 2:
        if len(mu.size()) == 2:
            mu_expanded = mu.unsqueeze(-2).repeat(1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,M_value,1)
        elif len(mu.size()) == 3:
            mu_expanded = mu.unsqueeze(-2).repeat(1,1,M_value,1)
            var_expanded = var.unsqueeze(-2).repeat(1,1,M_value,1)

        if update_centers_rule == 'expectation':
            gamma_values_log = alpha.log() + log_expect_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M] or [B,V,M] for marginal regularizer
        elif update_centers_rule == 'D_KL':
            gamma_values_log = alpha.log() - KL_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars) # size [B, M] or [B,V,M] for marginal regularizer
        else:
            print("Invaid update choice!")
            sys.exit()
        
        gamma_values = (gamma_values_log-torch.logsumexp(gamma_values_log,-1).unsqueeze(-1)).exp() # size [B, M] or [B,V,M] for marginal regularizer
  
    else: # len(mu.size()) == 3:
        mu_expanded = mu.unsqueeze(2).repeat(1,1,alpha.size(1),1)
        var_expanded = var.unsqueeze(2).repeat(1,1,alpha.size(1),1)

        if update_centers_rule == 'expectation':
            gamma_values_log = alpha.log() + log_expect_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1) # size [B, M^V]
        elif update_centers_rule == 'D_KL':
            gamma_values_log = alpha.log() - KL_lossy_DG_DG(mu_expanded,var_expanded,center_means,center_vars).sum(1) # size [B, M^V]
        else:
            print("Invaid update choice!")
            sys.exit()
        
        gamma_values = (gamma_values_log-torch.logsumexp(gamma_values_log,-1).unsqueeze(-1)).exp() # size [B, M^V]

    return gamma_values


def weighted_attention(mu,centers_alpha,centers_mean_label,heads_K,heads_Q,d_K,n_heads): # [B,H,M]
    batch_size = mu.size(0)
    M = centers_mean_label.size(-2)

    mu_heads_query = heads_Q(mu).view(batch_size,n_heads,d_K) # [B,H,d_K]
    centers_mean_label_keys_all = heads_K(centers_mean_label).view(batch_size,n_heads,M,n_heads,d_K).transpose(1,-1) # [B,d_Q,M.H,H]
    centers_mean_label_keys = centers_mean_label_keys_all.diagonal(dim1=-1,dim2=-2).transpose(1,-1)  # [B,H,M,d_Q]

    mu_query_expanded = mu_heads_query.unsqueeze(2).repeat(1,1,M,1) # [B,H,M,d_K]
    inner_product = (mu_query_expanded*centers_mean_label_keys).sum(-1)/np.sqrt(d_K)

    att =  (centers_alpha.log() + inner_product -torch.logsumexp(centers_alpha.log() + inner_product,-1).unsqueeze(-1).detach()).exp() 

    return att


def weighted_attention_same_space(mu,centers_alpha,centers_mean_label,heads_K,heads_Q,d_K,n_heads): # [B,H,M]
    batch_size = mu.size(0)
    M = centers_mean_label.size(-2)

    mu_heads_query = heads_Q(mu).view(batch_size,n_heads,d_K) # [B,H,d_K]
    centers_mean_label_keys =  heads_K(centers_mean_label).view(batch_size,M,n_heads,d_K).transpose(1,2) # [B,H,M,d_Q]
    
    mu_query_expanded = mu_heads_query.unsqueeze(2).repeat(1,1,M,1) # [B,H,M,d_K]
    inner_product = (mu_query_expanded*centers_mean_label_keys).sum(-1)/np.sqrt(d_K)

    att =  (centers_alpha.log() + inner_product -torch.logsumexp(centers_alpha.log() + inner_product,-1).unsqueeze(-1).detach()).exp()

    return att


def projection_before_multihead(centers_mean_label_heads,heads_V): # Project the means of each head by multiplying by  heads_V^-1
    n_heads = heads_V.size(0)
    batch_size = centers_mean_label_heads.size(0)
    M = centers_mean_label_heads.size(-2)
    d_K = heads_V.size(-2)

    heads_V_inverse = torch.linalg.pinv(heads_V) #[H,self.K,d_K]

    centers_mean_label = torch.matmul(heads_V_inverse.unsqueeze(0).unsqueeze(-3).repeat(batch_size,1,M,1,1),centers_mean_label_heads.unsqueeze(-1)).squeeze(-1)  # [B,H,M,K]
      
    return centers_mean_label


def compute_marginals(gamma_values_joint):
    gamma_values = cuda(torch.zeros(gamma_values_joint.size(0),n_views,M_value),cuda_value)
    for id_view in range(n_views):
        gamma_values[:,id_view,:] = gamma_values_joint[:,index_list_inverse[id_view]].sum(-1)

    return gamma_values
                

# Transform which randomly corrupts pixels with a given probabiliy
class PixelCorruption(object):
    MODALITIES = ['flip', 'drop']

    def __init__(self, p, min=0, max=1, mode='drop'):
        super(PixelCorruption, self).__init__()

        assert mode in self.MODALITIES

        self.p = p
        self.min = min
        self.max = max
        self.mode = mode


    def __call__(self, im):

        if self.p < 1.0:
            mask = torch.bernoulli(torch.zeros(im.size(0), 1, im.size(2), im.size(3)) + 1. - self.p).bool().repeat(1,im.size(1),1,1)
        else:
            mask = torch.zeros(im.size(0), 1, im.size(2), im.size(3)).bool().repeat(1,im.size(1),1,1)

        if self.mode == 'flip':
            im[mask] = self.max - im[mask]
        elif self.mode == 'drop':
            im[mask] = self.min

        return im

