import numpy as np
import torch
import os, time
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

global_parameters = {
    "repetitions":   5,                            # total number of repetitions
    "beta_joint":    list(np.logspace(-5, 0, 11)), # joint regularization parameter
    "loss_id":       [0,1,2],                      # 0 - no regularizarion, 
                                                   # 1 - marginal regularizer (VIB) 
                                                   # 2 - joint regularizer (GM-MDL)
    "centers_num":   [5],                          # number of centers in loss 2 and loss 3
    "mov_coeff_mul": [0.9],                        # coefficient for smooth change of prior centers in loss 2 and loss 3
    "mov_coeff_var": [5e-4],                       # coefficient for smooth change of var, default: 5e-4
    "save_results":  True,                         # save the results
    ######## Others that are fixed ########
    "results_dir":   'results',                    # dir to results
    "save_model":    False ,                       # saves the model after training
}

solver_parameters = {
    #### Optimization Parameters #########
    "batch_size":    128,                 # batch size
    "epoch_1":       200,                 # first training phase
    #### Model Parameters ############
    "model_name": 'CNN4',                 # 'CNN4' or 'Resnet'
    "mean_normalization_flag": False,
    "std_normalization_flag": False,
    "perturbation_flag": False,
    #### Lossy Regularizer Params ######### 
    "lossy_variance": 8*np.sqrt(2),       # The variance used for lossy computations
    "lossy_variance_var": 8*np.sqrt(2),   # The offset variance used for lossy computations of the term related to variances
    ##### Dataset Parameters ########
    "dataset": 'CIFAR10',                 # dataset; 'CIFAR10', 'CIFAR100', 'INTEL', 'USPS', 'Caltech101'
    "num_datasets": 5,
    ###### Views: numbers and distortions #######
    "n_views": 2,                         # number of views
    ## The length of below values should be equal to n_views ##
    "occlusion": False,
    "degrees_coeff": [0.,0.],             # light: 5, medium: 7.5, heavy: 10, ultimate: 20 
    "translate_coeff": [0.,0.],           # light: 0, medium: 0,  heavy: 0.2, ultimate: 0.4 
    "scale_coeff": [0.,0.],               # light: 0.1, medium: 0.2, heavy: 0.4, ultimate: 0.5 
    "PixelCorruption_coeff": [1.,1.],     # light: 0.95, medium: 0.9, heavy: 0.8, ultimate: 0.6 
    "occlusion_coeff": ["None", "None"],  # "None", "L", "R", "U", "B", "LU", "RU", "LB", "RB"
    ####### Others that are fixed ###########3
    "cuda":          True,
    "seed":          0,                   # used to re-initialize dataloaders 
    "epoch_2":       0,                   # second training phase                                          
    "lr":            1e-4,                # learning rate
    "alpha":         [],                  # relaxation coefficient
    "K":             64,                  # dimension of encoding Z
    "num_avg":       5,                   # number of samplings Z
    "dset_dir":      'datasets',          # dir with datasets
    "center_initialization": 'scattered', # The way the centers are generated (random or fixed or scattered)
    "per_epoch_stat": False,              # true if some statistics computed every epoch (slow) 
    "model_mode": 'stochastic',           # the model mode is either 'stochastic' or 'deterministic'
    "loss_choice": 'prod_var' ,           # This choice can be either 'prod_var', or 'prod', or 'var'
    "update_centers_rule": 'D_KL',        # This choice can be either 'expectation', or 'D_KL'
    "update_centers_coeff_mode": 'not_prop',  # This will make the updates proportional to either only self.M ('prop_M'), or only number of importance ('prop_b') or both ('prop_M_b') or none of them ('not_prop') or ('prop_b_logistic')
    "temp_coeff": 2,                      # coefficient value for decaying the dependence of the upadate of the centers on the current batch in not_prop_logistic
    "decaying_lossy_constants": 1,        # If it equals 1, the lossy constants do not change over the time
}

beta_list = []
training_accuracy_list = []
test_accuracy_list = []
def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    # create a new folder to store our results and figures
    timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    if global_parameters["save_results"]:
        results_path = global_parameters["results_dir"] + '/results_' +  timestamp
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # save global and initial solver parameters to a file
        if not os.path.isfile(results_path + '/global_parameters.npz'): 
            np.savez(results_path + '/global_parameters', global_parameters = global_parameters)          

    # training of models with various parameters
    for rep in range(global_parameters["repetitions"]):
        for loss_id in global_parameters["loss_id"]:
            for beta_joint in global_parameters["beta_joint"]:
                for centers_num in global_parameters["centers_num"]:
                    for mov_coeff_mul in global_parameters["mov_coeff_mul"]:  
                        for mov_coeff_var in global_parameters["mov_coeff_var"]:                    
                            # update selected solver parameters
                            solver_parameters["seed"]          = rep 
                            solver_parameters["beta_joint"]    = beta_joint
                            solver_parameters["loss_id"]       = loss_id
                            solver_parameters["mov_coeff_mul"] = mov_coeff_mul
                            solver_parameters["centers_num"]   = centers_num
                            solver_parameters["timestamp"]     = timestamp
                            solver_parameters["mov_coeff_var"] = mov_coeff_var
                            
                            filename = '/_results_LossID_{}_Beta_Joint{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.npz'.format(loss_id,beta_joint,centers_num,mov_coeff_mul,rep)

                            # create a model and train
                            net = Solver(solver_parameters)
                            net.train_full()

                            beta_list.append(beta_joint)
                            training_accuracy_list.append(net.train1_train_dataset["accuracy"] )
                            test_accuracy_list.append(net.train1_test_dataset["accuracy"] )

                            print('Loss_ID: {}'.format(solver_parameters["loss_id"])) 
                            print('N_views: {:}'.format(solver_parameters["n_views"])) 
                            print('Dataset: {}'.format(solver_parameters["dataset"]))
                            print('Model: {}'.format(solver_parameters["model_name"]))
                            print('Beta_joints: {:}'.format(beta_list))
                            print('Training_accuracies: {:}'.format(training_accuracy_list))
                            print('Test_accuracies: {:}'.format(test_accuracy_list))
                            
                            if global_parameters["save_results"]:
                                # extract interesting statistics and save to a .npz file
                                np.savez(results_path+filename, 
                                            solver_parameters    = solver_parameters,
                                            train1_train_dataset = net.train1_train_dataset, 
                                            train1_test_dataset  = net.train1_test_dataset,
                                            train1_epochs        = net.train1_epochs,
                                            counter              = rep
                                    )
                                
                                if global_parameters["save_model"]: torch.save(net.IBnet, results_path+'/_trained_model_LossID_{}_Beta_joint_{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.pth'.format(loss_id,beta_joint,centers_num,mov_coeff_mul,rep))
                            del net
    
    print('Loss_ID: {}'.format(solver_parameters["loss_id"])) 
    print('N_views: {:}'.format(solver_parameters["n_views"])) 
    print('Dataset: {}'.format(solver_parameters["dataset"]))
    print('Model: {}'.format(solver_parameters["model_name"]))
    print('Beta_joints: {:}'.format(beta_list))
    print('Training_accuracies: {:}'.format(training_accuracy_list))
    print('Test_accuracies: {:}'.format(test_accuracy_list))

if __name__ == "__main__":
    main()
