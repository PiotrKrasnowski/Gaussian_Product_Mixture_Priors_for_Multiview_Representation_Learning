import torch, os
from utils import PixelCorruption
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, USPS, Caltech101
import numpy as np 
from PIL import Image

     
class Intel_Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, labels, transform):
        self.num_samples  = labels.size(0)
        self.labels       = labels
        self.dataset      = samples
        self.transform    = transform

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]), self.labels[idx]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(dataset_dir, dataset_name):
    
    if dataset_name == "CIFAR10":
        dir = dataset_dir + "/CIFAR10/cifar-10-batches-py/"
        filelist = sorted(os.listdir(dir))
        # train data
        train_data, train_labels = [], []
        for i in range(1,6):
            z = unpickle(dir+filelist[i])
            z_data = z[b"data"]
            train_data.append(z_data.reshape(z_data.shape[0],3,32,32)/255.)
            train_labels = train_labels + z[b"labels"]

        train_data = np.concatenate(train_data)
        train_labels = np.array(train_labels, dtype = "int64")
        z = unpickle(dir+filelist[-1])
        z_data = z[b"data"]
        test_data = z_data.reshape(z_data.shape[0],3,32,32)/255.
        test_labels = np.array(z[b"labels"], dtype = "int64")
    
    elif dataset_name == "CIFAR100":
        dir = dataset_dir + "/CIFAR100/cifar-100-python/"
        train_read = unpickle(dir+'train')
        train_data = train_read[b"data"].reshape(train_read[b"data"].shape[0],3,32,32)/255.
        train_labels = np.array(train_read[b"fine_labels"], dtype = "int64")

        test_read = unpickle(dir+'test')
        test_data = test_read[b"data"].reshape(test_read[b"data"].shape[0],3,32,32)/255.
        test_labels = np.array(test_read[b"fine_labels"], dtype = "int64")
    
    elif dataset_name == "USPS":
        root = os.path.join(dataset_dir,'USPS')
        train_kwargs = {'root':root,'train':True,'transform':None,'download':True}
        test_kwargs = {'root':root,'train':False,'transform':None,'download':True}
        dset = USPS
        train_dataset = dset(**train_kwargs)
        test_dataset = dset(**test_kwargs)
        
        length_train = train_dataset.__len__()
        train_data, train_labels = [], []
        for ind in range(length_train): 
            (image, label) = train_dataset.__getitem__(ind)
            image = np.tile(np.array(image.getdata()).reshape(1, image.size[0], image.size[1])/255.,(3,1,1))
            train_data.append(image)
            train_labels.append(label)
        train_data = np.stack(train_data)
        train_labels = np.stack(train_labels)

        length_test = test_dataset.__len__()
        test_data, test_labels = [], []
        for ind in range(length_test): 
            (image, label) = test_dataset.__getitem__(ind)
            image = np.tile(np.array(image.getdata()).reshape(1, image.size[0], image.size[1])/255.,(3,1,1))
            test_data.append(image)
            test_labels.append(label)
        test_data = np.stack(test_data)
        test_labels = np.stack(test_labels)

    elif dataset_name == "INTEL":

        dir = dataset_dir + "/INTEL/"
        # train data
        train_data, train_labels = [], []
        list_category = sorted(os.listdir(dir + "seg_train/seg_train/"))       
        for category_ind, category in enumerate(list_category):
            filelist = sorted(os.listdir(dir + "seg_train/seg_train/" + category + '/'))
            for file in filelist:
                image = np.array(Image.open(dir + "seg_train/seg_train/" + category + '/' + file))/255.
                if image.shape[0] != 150 or image.shape[1] != 150: 
                    continue
                train_data.append(np.moveaxis(image, -1, 0))
                train_labels.append(category_ind)

        train_data = np.stack(train_data)
        train_labels = np.array(train_labels, dtype = "int64")

        # test data
        test_data, test_labels = [], []
        list_category = sorted(os.listdir(dir + "seg_test/seg_test/"))       
        for category_ind, category in enumerate(list_category):
            filelist = sorted(os.listdir(dir + "seg_test/seg_test/" + category + '/'))
            for file in filelist:
                image = np.array(Image.open(dir + "seg_test/seg_test/" + category + '/' + file))/255.
                if image.shape[0] != 150 or image.shape[1] != 150: 
                    continue
                test_data.append(np.moveaxis(image, -1, 0))
                test_labels.append(category_ind)

        test_data = np.stack(test_data)
        test_labels = np.array(test_labels, dtype = "int64")

    elif dataset_name == "Caltech101":
        
        root = os.path.join(dataset_dir,'Caltech101')
        data = np.load(root +'/images.npy')
        labels = np.load(root +'/labels.npy')

        train_data = data[:8000]
        train_labels = labels[:8000]

        test_data = data[8000:]
        test_labels = labels[8000:]

    return (train_data, train_labels), (test_data, test_labels)


class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"


def generate_multiview_data(samples_init,view_number,degrees_coeff=[],translate_coeff=[],scale_coeff=[],PixelCorruption_coeff=[],occlusion_coeff=[],occlusion = False, model_name="CNN4"):
    # transform = []
    samples_per_view = []
    for view in range(view_number):
        samples_view = samples_init.clone()
        
        if model_name == "CNN4":
            transform = transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.RandomAffine(degrees=degrees_coeff[view],
                        translate=[translate_coeff[view], translate_coeff[view]],
                        scale=[1-scale_coeff[view], 1+scale_coeff[view]],
                        shear=None),  # Small affine transformations
                        PixelCorruption(PixelCorruption_coeff[view]),
                        ])
        elif model_name == "Resnet":
            transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomAffine(degrees=degrees_coeff[view],
                        translate=[translate_coeff[view], translate_coeff[view]],
                        scale=[1-scale_coeff[view], 1+scale_coeff[view]],
                        shear=None),  # Small affine transformations
                        PixelCorruption(PixelCorruption_coeff[view]),
                        ])
       
        if occlusion:
            shape = samples_view.shape
            if "R" in occlusion_coeff[view]: samples_view = samples_view[:,:,:shape[2]//2,:]
            if "L" in occlusion_coeff[view]: samples_view = samples_view[:,:,shape[2]//2:,:]
            if "B" in occlusion_coeff[view]: samples_view = samples_view[:,:,:,:shape[3]//2]
            if "U" in occlusion_coeff[view]: samples_view = samples_view[:,:,:,shape[3]//2:]

        samples_view = transform(samples_view)
        samples_per_view.append(samples_view)

    return torch.stack(samples_per_view,dim=0)


def return_data_x_y(name, dset_dir, view_number, degrees_coeff=[],translate_coeff=[],scale_coeff=[],PixelCorruption_coeff=[],occlusion_coeff=[],occlusion = False):

    (training_images, training_labels), (testing_images, testing_labels) = load_dataset(dset_dir, name)

    training_images = torch.tensor(training_images, dtype = torch.float32)
    training_labels = torch.tensor(training_labels, dtype = torch.int64) 

    testing_images = torch.tensor(testing_images, dtype = torch.float32)
    testing_labels = torch.tensor(testing_labels, dtype = torch.int64)

    training_images_per_views = generate_multiview_data(training_images, view_number, degrees_coeff, translate_coeff, scale_coeff, PixelCorruption_coeff, occlusion_coeff, occlusion)
    testing_images_per_views = generate_multiview_data(testing_images, view_number, degrees_coeff, translate_coeff, scale_coeff, PixelCorruption_coeff, occlusion_coeff, occlusion)

    return training_images_per_views, training_labels, testing_images_per_views, testing_labels


if __name__ == '__main__' :
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dset_dir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

