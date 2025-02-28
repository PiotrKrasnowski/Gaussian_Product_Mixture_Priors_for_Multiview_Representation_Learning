########################################################################################
#                                                                                      #
#                            Code for JMLR submission                                  #
#         Generalization Guarantees for Multi-View Representation  Learning            #
#        and Application to Regularization via Gaussian Product Mixture Prior          #
#                                                                                      #
########################################################################################

This archive contains code and instructions for experiments in the submitted paper titled:
Generalization Guarantees for Multi-View Representation Learning 
and Application to Regularization via Gaussian Product Mixture Prior


#################
# Requirements  #
#################
The code is tested with Python 3.10.0 on Linux. Please see requirements.txt.


##################
# Code structure #
##################
 main.py            # File to run to reproduce presented experiments in the paper

 datasets.py        # File which specifies and creates data loaders 
 model.py           # File which specifies the prediction model
 solver.py          # File which specifies the training and testing algorithms
 utils.py           # File which defines some auxilary functions

 datasets/          # Directory with datasets
 results/           # Directory with training results


#######################
# Running experiments #
#######################  
 Download the datasets
   - CIFAR10/CIFAR100 data from https://www.cs.toronto.edu/~kriz/cifar.html 
   - INTEL data from https://www.kaggle.com/datasets/puneet6060/intel-image-classification
   - USPS data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps .
 Extract the files and place under the right folder in the directory datasets/ .
 To reproduce results in the paper: 
   - Launch the script main.py . The selected training parameters and training results will appear in the right folder in the directory results/ . 


##############
# References #
##############
This code is based on the Variation Information Bottleneck implementation (VIB) of https://github.com/1Konny/VIB-pytorch.git. The code, however, has been modified substantially both for VIB and for our proposed objective function.