import random
import FrEIA.framework as Ff
from norm_function import *
from metrics import Metrics
from metrics_batch import Metrics as mb
from H36Dataset import H36MDataset
import pytorch_lightning as pl
from utils import euler_angles_to_matrix, perspective_projection, subnet_fc, create_new_pose
from models import DepthAngleEstimator
from torch.utils.data import DataLoader
import FrEIA.modules as Fm


class Config:
    batch_size = 256
    p_hint = 0.9
    train_file = 'paper_dataset'
    test_file = 'paper_dataset'
    dataset = {
        'train': [1, 5, 6, 7, 8],
        'test': [9, 11]
    }
    g_lr = 1e-2
    d_lr = 1e-2
    l_lr = 2e-4
    n_miss = 3
    test_max_n_miss = 3
    decay_rate = 0.95
    depth = 10
    occ = False
    h36m = True if not (train_file == 'oc_dataset' or train_file == 'mpii') else False


def set_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    torch.cuda.manual_seed(3)


class LiftingNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.inn_2d = inn_2d_1
        self.depth_estimator = depth_estimator
        self.opts = {}
        self.pca = None
        self.bl_prior = torch.tensor([[0.4779, 1.8354, 1.5134, 0.4779, 1.8333, 1.5134, 0.9031, 0.9810, 0.3450, 0.6903,
                                       0.5711, 1.2172, 0.9257, 0.5723, 1.2172, 0.9257]],
                                     device='cuda:0')
        self.parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.metrics = Metrics()

    def train_dataloader(self):
        train_set = H36MDataset(Config.train_file, Config.dataset['train'], normalize_func=normalize_head, get_PCA=True,
                                h36m=Config.h36m, occ=Config.occ)
        self.pca = train_set.pca
        train_loader = DataLoader(train_set, Config.batch_size, True)

        return train_loader

    def val_dataloader(self):
        test_set = H36MDataset(Config.test_file, Config.dataset['test'], normalize_func=normalize_head_test,
                               h36m=Config.h36m, occ=Config.occ)
        test_loader = DataLoader(test_set, 1000, False)

        return test_loader

    def configure_optimizers(self):
        l_opt = torch.optim.Adam(self.depth_estimator.parameters(), lr=Config.l_lr)
        self.opts['l_opt'] = l_opt

        return l_opt

    def training_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']
        batch_size = input_2d.shape[0]
        d2_pose = input_2d.clone()
        d2_pose = d2_pose.reshape(-1, 2, 17).transpose(2, 1)
        new_pose = create_new_pose(d2_pose,)
        d2_pose = torch.cat([d2_pose, new_pose], dim=0)

        d3_pose, R = self.L(d2_pose)
        d3_pose -= d3_pose[:, [0]]
        d3_rot = d3_pose @ R
        d3_rot[:, :, 2] += Config.depth
        d2_rot = perspective_projection(d3_rot)
        norm_pose = d2_rot.transpose(2, 1).reshape(-1, 34) - \
                    torch.tensor(self.pca.mean_.reshape(1, 34), device='cuda:0', dtype=torch.float32)
        latent = norm_pose @ torch.tensor(self.pca.components_.T, device='cuda:0', dtype=torch.float32)
        z, log_jac_det = self.inn_2d(latent[:, :26])

        d3_rot_res, _ = self.L(d2_rot)
        temp = d3_rot_res - d3_rot_res[:, [0]]
        d3_pose_res = temp @ R.transpose(2, 1)
        d3_pose_res[:, :, 2] += Config.depth
        d2_pose_res = perspective_projection(d3_pose_res.clone())
        d3_pose_res[:, :, 2] -= Config.depth

        loss_d = (0.5 * torch.sum(z ** 2, 1) - log_jac_det).mean()
        bl = (d3_pose[:, 1:] - d3_pose[:, self.parent]).norm(dim=-1)
        bl /= bl.mean(dim=-1, keepdim=True)
        loss_bl = (bl - self.bl_prior).square().sum(dim=-1).mean()
        loss_3d = (d3_rot.reshape(batch_size, -1) - d3_rot_res.reshape(batch_size, -1)).norm(dim=-1).mean()
        loss_2d = (d2_pose_res.reshape(batch_size, -1) - d2_pose.reshape(batch_size, -1)).abs().sum(dim=-1).mean()
        pairs_0 = [2 * i for i in range(batch_size // 2)]
        pairs_1 = [2 * i + 1 for i in range(batch_size // 2)]
        loss_def = ((d3_pose[pairs_0].reshape(-1, 51) - d3_pose[pairs_1].reshape(-1, 51)) -
                    (d3_pose_res[pairs_0].reshape(-1, 51) - d3_pose_res[pairs_1].reshape(-1, 51))).norm(dim=-1).mean()

        loss_l = loss_3d + loss_2d + loss_def + 50 * loss_bl + loss_d

        self.log_dict({'l_loss': loss_l, 'loss_3d': loss_3d}, prog_bar=True)

        return loss_l

    def validation_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']
        input_3d = batch['poses_3d']

        batch_size = input_2d.shape[0]
        d2_pose = input_2d.clone()
        d2_pose = d2_pose.reshape(-1, 2, 17).transpose(2, 1)

        d3_hat, _ = self.L(d2_pose)
        d3_hat = d3_hat.transpose(2, 1)

        pa = 0
        pk = 0
        for i in range(batch_size):
            pck, err = self.metrics.pa_PCK(input_3d[[i]].cpu().numpy(),
                                           d3_hat[i].detach().cpu().numpy().reshape(-1, 51),
                                           reflection='best')
            pk += pck
            pa += err
        pa /= batch_size
        pk /= batch_size

        mpjpe_scaled = mb().mpjpe(input_3d, d3_hat, root_joint=0, num_joints=17).mean()
        n_pck = mb().PCK(input_3d, d3_hat, root_joint=0, num_joints=17)
        auc = mb().AUC(input_3d, d3_hat, root_joint=0, num_joints=17)

        self.log_dict({'mpjpe_scale': mpjpe_scaled, 'pa': pa, 'pck': pk, 'n-pck': n_pck, 'auc': auc}, prog_bar=True)

    def L(self, x):
        # b, 17, 2
        batch_size = x.shape[0]
        xd, xa = self.depth_estimator(x)

        xd[:, 0] = 0
        xd = xd[:, :, None]
        depth = xd + Config.depth
        depth[depth < 1.0] = 1.0
        out_3d = torch.cat([x.reshape(batch_size, 17, 2) * depth, depth], dim=-1)

        zeros = torch.zeros(size=(batch_size, 1), device='cuda:0')
        comp_angle_x = torch.ones(size=(batch_size, 1), device='cuda:0') * xa
        comp_angle = torch.cat([comp_angle_x, zeros, zeros], dim=1)
        R_comp = euler_angles_to_matrix(comp_angle, 'XYZ')

        elev_angle_x = -xa.mean() + xa.std() * torch.normal(torch.zeros(batch_size, 1, device=self.device),
                                                            torch.ones(batch_size, 1, device=self.device))
        elev_angle = torch.cat([elev_angle_x, zeros, zeros], dim=1)
        R_elev = euler_angles_to_matrix(elev_angle, 'XYZ')

        azim_angle_y = (torch.rand(size=(batch_size, 1), device='cuda:0') - 0.5) * 2 * np.pi
        azim_angle = torch.cat([zeros, azim_angle_y, zeros], dim=1)
        R_azim = euler_angles_to_matrix(azim_angle, 'XYZ')

        R = R_elev @ R_azim @ R_comp

        return out_3d, R.transpose(2, 1)

    def on_train_epoch_end(self):
        for p in self.opts['l_opt'].param_groups:
            p['lr'] *= Config.decay_rate


depth_estimator = DepthAngleEstimator()
inn_2d_1 = Ff.SequenceINN(26)
for k in range(8):
    inn_2d_1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

inn_2d_1.load_state_dict(torch.load('models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_26_headnorm.pt'))
for param in inn_2d_1.parameters():
    param.requires_grad = False


if __name__ == '__main__':
    callback1 = pl.callbacks.ModelCheckpoint('liftnet', Config.test_file+'_{epoch}-{pa:.4f}', monitor='pa', mode='min', save_top_k=1)
    callback2 = pl.callbacks.ModelCheckpoint('liftnet', Config.test_file+'_{epoch}-{n-pck:.4f}', monitor='n-pck', mode='max', save_top_k=1)
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=10, callbacks=[callback1])
    l_net = LiftingNetwork()
    trainer.fit(l_net)
