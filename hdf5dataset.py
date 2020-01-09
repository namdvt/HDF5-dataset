import h5py
import torch.utils.data as dset
import numpy as np


class Hdf5Dataset(dset.Dataset):
    def __init__(self, file_address, subset):
        super().__init__()
        assert subset in {'train', 'validation', 'test'}, "sai subset name roi nhe!"
        self.file = h5py.File(file_address, 'r')
        self.subset = subset

    def __getitem__(self, index):
        return self.file[self.subset]['data'][index], self.file[self.subset]['label'][index]

    def __len__(self) -> int:
        return len(self.file[self.subset]['label'])

    def get_num_classes(self):
        return np.unique(self.file[self.subset]['label']).size
