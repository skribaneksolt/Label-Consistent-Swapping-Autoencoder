import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

from glob import glob
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch

  
class UnalignedLabeledDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
                and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "testA")
            self.dir_B = os.path.join(opt.dataroot, "testB")
            
        if self.dir_A.endswith('trainA'):
            self.dir_Asem = os.path.join(opt.dataroot, "trainAsem")
            self.dir_Bsem = os.path.join(opt.dataroot, "trainBsem")
            self.A_ids = [os.path.split(path)[1][:-11] for path in sorted(glob(self.dir_A + "/*_rgb512.png"))]
            random.Random(0).shuffle(self.A_ids)
            self.A_size = len(self.A_ids)
            self.B_ids = [os.path.split(path)[1][:-11] for path in sorted(glob(self.dir_B + "/*_rgb512.png"))]
            random.Random(0).shuffle(self.B_ids)
            self.B_size = len(self.B_ids)
            self.B_indices = list(range(self.B_size))

            self.random_flip = opt.isTrain and (not opt.no_flip)
            self.crop = 'crop' in opt.preprocess
            if self.crop:
                self.crop_size = opt.crop_size
        else:   # test case
            self.A_paths = sorted(make_dataset(self.dir_A))  # load images from '/path/to/data/trainA'
            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.transform_A = get_transform(self.opt, grayscale=False)
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if not self.dir_A.endswith('trainA'):   # test case
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            A_rgb = Image.open(A_path).convert('RGB')
            A = self.transform_A(A_rgb)
            return {'real_A': A, 'path_A': A_path}
        
        else:   # train case :)
            # variables : dirA, dirB, dirAsem, dirBsem, A_ids, B_ids, A_size, B_size
            A_id = self.A_ids[index % self.A_size]
            
            if index == 0 and self.opt.isTrain:
                random.shuffle(self.B_indices)
            index_B = self.B_indices[index % self.B_size]
            B_id = self.B_ids[index_B]
            
            A_path = os.path.join(self.dir_A, A_id + '_rgb512.png')
            Asem_path = os.path.join(self.dir_Asem, A_id + '_sem512.png')
            B_path = os.path.join(self.dir_B, B_id + '_rgb512.png')
            Bsem_path = os.path.join(self.dir_Bsem, B_id + '_sem512.png')

            A_rgb = Image.open(A_path).convert('RGB')
            B_rgb = Image.open(B_path).convert('RGB')
            A_sem = Image.open(Asem_path).convert('I')
            B_sem = Image.open(Bsem_path).convert('I')
            
            # augmentations: crop, randomFlip, convert: {toTensor, Normalize
            if self.crop:
                cropA_params = transforms.RandomCrop.get_params(A_rgb, output_size=(self.crop_size, self.crop_size))
                cropB_params = transforms.RandomCrop.get_params(B_rgb, output_size=(self.crop_size, self.crop_size))
                A_rgb = TF.crop(A_rgb, *cropA_params)
                A_sem = TF.crop(A_sem, *cropA_params)
                B_rgb = TF.crop(B_rgb, *cropB_params)
                B_sem = TF.crop(B_sem, *cropB_params)
            
            if self.random_flip:
                if random.random() > 0.5:
                    A_rgb = TF.hflip(A_rgb)
                    A_sem = TF.hflip(A_sem)
                if random.random() > 0.5:
                    B_rgb = TF.hflip(B_rgb)
                    B_sem = TF.hflip(B_sem)
                    
            A_rgb = TF.to_tensor(A_rgb)
            A_rgb = TF.normalize(A_rgb, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            B_rgb = TF.to_tensor(B_rgb)
            B_rgb = TF.normalize(B_rgb, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            
            ow, oh = A_sem.size
            new_w, new_h = ow // 4, oh // 4
            A_sem_small = TF.resize(A_sem, size=[new_w, new_h], interpolation=InterpolationMode.NEAREST)
            B_sem_small = TF.resize(B_sem, size=[new_w, new_h], interpolation=InterpolationMode.NEAREST)
            
            A_sem = self.ToLabel(TF.to_tensor(A_sem))
            B_sem = self.ToLabel(TF.to_tensor(B_sem))
            A_sem_small = self.ToLabel(TF.to_tensor(A_sem_small))
            B_sem_small = self.ToLabel(TF.to_tensor(B_sem_small))
            
            return {'real_A': A_rgb, 'sem_A': A_sem, 'sem_A_small': A_sem_small,
                    'real_B': B_rgb, 'sem_B': B_sem, 'sem_B_small': B_sem_small}
        
    @staticmethod
    def ToLabel(tensor):
        return torch.squeeze(tensor.long())

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.dir_A.endswith('trainA'):   # train
            return max(self.A_size, self.B_size)
        else:   # test: no B_size
            return self.A_size
