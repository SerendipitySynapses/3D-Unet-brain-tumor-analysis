import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import cv2
from tensorflow.keras.utils import Sequence
from tqdm import tqdm


class DataGenerator(Sequence):
    def __init__(self, list_IDs, input_dir, batch_size=9, dim=(128, 128), n_channels=4, n_classes=4, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.input_dir = input_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __load_nifti_file(self, filepath):
        scan = nib.load(filepath)
        return scan.get_fdata()

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size * 155, *self.dim, self.n_channels))
        y = np.empty((self.batch_size * 155, *self.dim, self.n_classes), dtype=int)
        with tqdm(total=self.batch_size * 155, desc='Generating batch data', unit='slice') as pbar:
            for i, ID in enumerate(list_IDs_temp):
                # id_str = f"{int(ID):03d}"
                folder_name = os.path.join(self.input_dir, f'BraTS20_Training_{ID}')
                mask_name = f'BraTS20_Training_{ID}_seg'
                npy_mask = self.__load_nifti_file(os.path.join(folder_name, mask_name + '.nii'))
                for h in range(155):
                    mask = tf.one_hot(npy_mask[:, :, h], self.n_classes)
                    y[i * 155 + h] = tf.image.resize(mask, self.dim).numpy()
                    for k, modality in enumerate(['flair', 't1', 't1ce', 't2']):
                        channel_name = f'BraTS20_Training_{ID}_{modality}'
                        nmpy_channel = self.__load_nifti_file(os.path.join(folder_name, channel_name + '.nii'))
                        X[i * 155 + h, :, :, k] = cv2.resize(nmpy_channel[:, :, h], self.dim)
                    pbar.update(1)

        y[y == self.n_classes] = self.n_classes - 1
        return X / np.max(X), y


