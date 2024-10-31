import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
import os
from utils import draw_skeleton_2d, draw_skeleton_3d
from norm_function import normalize_head


class H36MDataset(Dataset):
    def __init__(self, fname, subjects, normalize_2d=False,
                 get_2dgt=True, get_PCA=False, normalize_func=None, h36m=True, occ=True):
        self.data = {}
        joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]
        parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        subjects = ['S' + str(i) for i in subjects]
        data_2d = np.load(os.path.join(fname, 'new_d2.npy'), allow_pickle=True).item()
        data_3d = np.load(os.path.join(fname, 'new_d3.npy'), allow_pickle=True).item()
        if occ:
            img_path = np.load(os.path.join(fname, 'img_path.npy'), allow_pickle=True).item()
            mask = np.load(os.path.join(fname, 'mask.npy'), allow_pickle=True).item()
            temp_mask = []
            temp_img_path = []
            for s in subjects:
                for p in data_2d[s].keys():
                    for c in data_2d[s][p].keys():
                        temp_mask.append(mask[s][p][c])
                        temp_img_path.append(img_path[s][p][c])
            self.data['mask'] = np.concatenate(temp_mask, axis=0)
            self.data['img_path'] = np.concatenate(temp_img_path, axis=0)

        temp_3d = []
        temp_2d = []
        for s in subjects:
            for p in data_2d[s].keys():
                for c in data_2d[s][p].keys():
                    temp_3d.append(data_3d[s][p][c])
                    temp_2d.append(data_2d[s][p][c])

        self.data['poses_3d'] = np.concatenate(temp_3d, axis=0)
        self.data['poses_2d'] = np.concatenate(temp_2d, axis=0)
        if h36m:
            self.data['poses_3d'] = self.data['poses_3d'][:, joints, :]
            self.data['poses_2d'] = self.data['poses_2d'][:, joints, :]
        self.data['poses_2d'] = self.data['poses_2d'].transpose(0, 2, 1).reshape(-1, 2 * len(joints))
        self.data['poses_3d'] = self.data['poses_3d'].transpose(0, 2, 1)
        self.normalize_2d = normalize_2d
        self.get_2dgt = get_2dgt

        if normalize_func:
            self.data['org_2d'] = self.data['poses_2d'].copy()
            self.data['poses_2d'] = normalize_func(self.data['poses_2d'])
        else:
            if self.normalize_2d:
                save = []
                self.data['poses_2d'] = (
                        self.data['poses_2d'].reshape(-1, 2, len(joints)) -
                        self.data['poses_2d'].reshape(-1, 2, len(joints)).mean(axis=2, keepdims=True)).reshape(-1, 2*len(joints))
                temp = np.linalg.norm(self.data['poses_2d'], ord=2, axis=1, keepdims=True)
                for i, t in enumerate(temp):
                    if t[0] != np.inf:
                        save.append(i)
                self.data['poses_2d'] = self.data['poses_2d'][save, :]
                self.data['poses_3d'] = self.data['poses_3d'][save, :]
                self.data['poses_2d'] /= np.linalg.norm(self.data['poses_2d'], ord=2, axis=1, keepdims=True)

        self.data['poses_3d'] = torch.tensor(self.data['poses_3d'], dtype=torch.float).reshape(-1, 3*len(joints))
        self.data['poses_2d'] = torch.tensor(self.data['poses_2d'], dtype=torch.float).reshape(-1, 2*len(joints))

        #bl = (self.data['poses_3d'][:, :, 1:] - self.data['poses_3d'][:, :, parent]).norm(dim=1)
        #bl = bl.mean(dim=0) / bl.mean()
        if get_PCA:
            self.pca = PCA()
            self.pca.fit(self.data['poses_2d'])

    def __len__(self):
        return self.data['poses_3d'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()
        if 'mask' in self.data.keys():
            sample['mask'] = self.data['mask'][idx]
            sample['img_path'] = self.data['img_path'][idx]
            sample['org_2d'] = self.data['org_2d'][idx]
        if self.get_2dgt:
            sample['p2d_gt'] = self.data['poses_2d'][idx]
        else:
            sample['p2d_pred'] = self.data['poses_2d'][idx]

        sample['poses_3d'] = self.data['poses_3d'][idx]

        return sample

    def draw_skeleton_3d(self, idx):
        data = self.data['poses_3d'][idx].reshape(3, -1).transpose(1, 0)
        draw_skeleton_3d(data)

    def draw_skeleton_2d(self, idx):
        data = self.data['poses_2d'][idx].reshape(2, -1).transpose(1, 0)
        draw_skeleton_2d(data)


if __name__ == '__main__':
    path = 'paper_dataset'
    dataset = H36MDataset(path, [1], normalize_func=normalize_head, h36m=True, occ=False)
    dataset.draw_skeleton_2d(idx=0)
