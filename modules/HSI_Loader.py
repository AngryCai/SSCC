import torch
from torch.utils.data import Dataset, DataLoader
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import scale, minmax_scale, normalize
from sklearn.decomposition import PCA
import numpy as np


class HSI_Data(Dataset):

    def __init__(self, path_to_data, path_to_gt, patch_size=(11, 11), transform=None, pca=True, pca_dim=8, is_labeled=True):
        # super(HSI_Data, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        p = Processor()
        img, gt = p.prepare_data(path_to_data, path_to_gt)
        n_row, n_column, n_band = img.shape
        if pca:
            img = scale(img.reshape(n_row * n_column, -1))  # .reshape((n_row, n_column, -1))
            pca = PCA(n_components=pca_dim)
            img = pca.fit_transform(img).reshape((n_row, n_column, pca_dim))
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
        # classes = np.unique(y_)
        # for c in classes:
        #     if np.nonzero(y_ == c)[0].shape[0] < 30:
        #         x_patches = np.delete(x_patches, np.nonzero(y_ == c), axis=0)
        #         y_ = np.delete(y_, np.nonzero(y_ == c))
        y = p.standardize_label(y_)
        if not is_labeled:
            self.n_classes = np.unique(y).shape[0] - 1
        else:
            self.n_classes = np.unique(y).shape[0]
        # n_class = np.unique(y).shape[0]
        n_samples, n_row, n_col, n_channel = x_patches.shape
        self.data_size = n_samples
        x_patches = scale(x_patches.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        # x_patches = x_patches.reshape((n_samples, -1)).reshape((n_samples, n_row, n_col, -1))
        x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        self.x_tensor, self.y_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor), \
                                       torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, idx):
        x, y = self.x_tensor[idx], self.y_tensor[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.data_size
