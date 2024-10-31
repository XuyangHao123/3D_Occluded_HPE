import random
from losses import *
from norm_function import *
from metrics import Metrics
from metrics_batch import Metrics as mb
from H36Dataset import H36MDataset
import pytorch_lightning as pl
from utils import euler_angles_to_matrix, perspective_projection, subnet_fc
from models import Discriminator, Generator, DepthAngleEstimator_wight
from torch.utils.data import DataLoader
from pre_train_lifting_network import LiftingNetwork
import FrEIA.modules as Fm
import FrEIA.framework as Ff


class Config:
    batch_size = 256
    train_file = 'paper_dataset'
    test_file = 'paper_dataset'
    dataset = {
        'train': [1, 5, 6, 7, 8],
        'test': [9, 11]
    }
    g_lr = 2e-3
    s_lr = 1e-2
    l_lr = 2e-4
    n_miss = 3
    test_max_n_miss = 3
    decay_rate = 0.95
    depth = 10
    g_end = 200
    l_begin = 60
    max_epoch = 200


def set_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    torch.cuda.manual_seed(3)


class Gain(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.inn_2d = inn_2d_1
        self.pca = None
        self.teacher = teacher_model
        self.generator = generator
        self.discriminator = discriminator
        self.depth_estimator = depth_estimator
        self.opts = {}
        self.bl_prior = torch.tensor([[0.5246, 1.7387, 1.7132, 0.5246, 1.7387, 1.7132, 0.9316, 0.9793,
                                       0.4482, 0.4407, 0.5795, 1.0863, 0.9578, 0.5795, 1.0863, 0.9578]],
                                     device='cuda:0')
        self.parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.automatic_optimization = False
        self.metrics = Metrics()

    def train_dataloader(self):
        train_set = H36MDataset(Config.train_file, Config.dataset['train'], normalize_func=normalize_head, get_PCA=True)
        self.pca = train_set.pca
        train_loader = DataLoader(train_set, Config.batch_size, True)

        return train_loader

    def val_dataloader(self):
        test_set = H36MDataset(Config.test_file, Config.dataset['test'], normalize_func=normalize_head_test)
        test_loader = DataLoader(test_set, 100, False)

        return test_loader

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=Config.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=Config.s_lr)
        l_opt = torch.optim.Adam(self.depth_estimator.parameters(), lr=Config.l_lr)

        self.opts['g_opt'] = g_opt
        self.opts['d_opt'] = d_opt
        self.opts['l_opt'] = l_opt

        return g_opt, d_opt, l_opt

    def training_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']
        batch_size = input_2d.shape[0]
        d2_pose = input_2d.clone()
        d2_pose = d2_pose.reshape(-1, 2, 17).transpose(2, 1)

        d_opt = self.opts['d_opt']
        f_opt = self.opts['g_opt']
        l_opt = self.opts['l_opt']

        m = [np.random.permutation(17) for i in range(batch_size)]
        m = np.array(m)[:, :Config.n_miss]
        m = np.sort(m)
        m_bool = torch.tensor([[True] * 17] * batch_size, device='cuda:0')
        for i in range(batch_size):
            m_bool[i, m[i]] = False

        x_hat = self.F(d2_pose, m_bool, m)
        f_loss = generator_loss(d2_pose, x_hat)
        self.log_dict({'mse_loss': f_loss}, prog_bar=True)
        if self.current_epoch < Config.g_end:
            f_opt.zero_grad()
            f_loss.backward()
            f_opt.step()

        x_hat = x_hat.detach()
        d_prob = self.discriminator(x_hat)
        d_loss = discriminator_loss(d_prob, x_hat, d2_pose)
        self.log_dict({'d_loss': d_loss}, prog_bar=True)
        if self.current_epoch < Config.g_end:
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

        if self.current_epoch < Config.l_begin:
            return

        d_prob = d_prob.detach()
        x_hat = x_hat.detach()
        x_hat[:, 0] = 0

        d3_pose, R = self.L(self.depth_estimator, d_prob, x_hat)
        d3_pose -= d3_pose[:, [0]]
        d3_rot = d3_pose @ R
        d3_rot[:, :, 2] += Config.depth
        d2_rot = perspective_projection(d3_rot)
        norm_pose = d2_rot.transpose(2, 1).reshape(-1, 34) - \
                    torch.tensor(self.pca.mean_.reshape(1, 34), device='cuda:0', dtype=torch.float32)
        latent = norm_pose @ torch.tensor(self.pca.components_.T, device='cuda:0', dtype=torch.float32)
        z, log_jac_det = self.inn_2d(latent[:, :26])
        prob = self.D(d2_rot)

        d3_rot_res, _ = self.L(self.depth_estimator, prob, d2_rot)
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
        loss_2d = (d2_pose_res.reshape(batch_size, -1) - x_hat.reshape(batch_size, -1)).abs().sum(dim=-1).mean()
        pairs_0 = [2 * i for i in range(batch_size // 2)]
        pairs_1 = [2 * i + 1 for i in range(batch_size // 2)]
        loss_def = ((d3_pose[pairs_0].reshape(-1, 51) - d3_pose[pairs_1].reshape(-1, 51)) -
                    (d3_pose_res[pairs_0].reshape(-1, 51) - d3_pose_res[pairs_1].reshape(-1, 51))).norm(dim=-1).mean()

        loss_l = loss_3d + loss_2d + loss_def + 50 * loss_bl + 15*loss_d

        l_opt.zero_grad()
        loss_l.backward()
        l_opt.step()

        self.log_dict({'loss_d': loss_d, 'l_loss': loss_l, '3d_loss': loss_3d}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']
        input_3d = batch['poses_3d']

        batch_size = input_2d.shape[0]
        d2_pose = input_2d.clone()
        d2_pose = d2_pose.reshape(-1, 2, 17).transpose(2, 1)

        m = [np.random.permutation(17) for i in range(batch_size)]
        n = np.random.randint(0, Config.test_max_n_miss + 1)
        m = np.array(m)[:, :n]
        m = np.sort(m)
        m_bool = torch.tensor([[True] * 17] * batch_size, device='cuda:0')
        for i in range(batch_size):
            m_bool[i, m[i]] = False

        # x_hat = d2_pose
        x_hat = self.F(d2_pose, m_bool, m)
        prob = self.D(x_hat)
        val_f_loss = generator_loss(d2_pose, x_hat)
        d3_hat, _ = self.L(self.depth_estimator, prob, x_hat)
        d3_hat = d3_hat.transpose(2, 1)

        pa = 0
        for i in range(batch_size):
            err = self.metrics.pmpjpe(input_3d[[i]].cpu().numpy(),
                                      d3_hat[i].detach().cpu().numpy().reshape(-1, 51),
                                      reflection='best')
            pa += err
        pa /= batch_size

        mpjpe_scaled = mb().mpjpe(input_3d, d3_hat, root_joint=0, num_joints=17).mean()

        self.log_dict({'val_f_loss': val_f_loss, 'mpjpe_scale': mpjpe_scaled, 'pa': pa}, prog_bar=True)

    def F(self, x, m_bool, m):
        batch_size = x.shape[0]

        g_x = x[m_bool].reshape(batch_size, -1, 2)
        x_hat = self.generator(g_x, m_bool)
        g_result = x.clone()
        for i in range(batch_size):
            g_result[i, m[i]] = x_hat[i]
        g_score = self

        return g_result

    def D(self, x):
        d_prob = self.discriminator(x)

        return d_prob

    def L(self, model, score, x):
        # b, 17, 2
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        xd, xa = model(score, x)

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
        if self.current_epoch < Config.g_end:
            for p in self.opts['g_opt'].param_groups:
                p['lr'] *= Config.decay_rate
        if self.current_epoch > Config.l_begin:
            for p in self.opts['l_opt'].param_groups:
                p['lr'] *= Config.decay_rate

    '''def nf_loss(self, d2_pose):
        norm_pose = d2_pose.transpose(2, 1).reshape(-1, 34) - \
                    torch.tensor(self.pca.mean_.reshape(1, 34), device='cuda:0', dtype=torch.float32)
        latent = norm_pose @ torch.tensor(self.pca.components_.T, device='cuda:0', dtype=torch.float32)
        z, log_jac_det = self.inn_2d(latent[:, :26])
        loss_d = (0.5 * torch.sum(z ** 2, 1) - log_jac_det).mean()

        return loss_d'''


set_seed()
generator = Generator()
discriminator = Discriminator()
depth_estimator = DepthAngleEstimator_wight(num_joints=17)

########################################################################################################################
lifter = LiftingNetwork().load_from_checkpoint('checkpoint/epoch=59-pa=33.869.ckpt')
teacher_model = lifter.depth_estimator
for param in teacher_model.parameters():
    param.requires_grad = False
########################################################################################################################
########################################################################################################################
inn_2d_1 = Ff.SequenceINN(26)
for k in range(8):
    inn_2d_1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
inn_2d_1.load_state_dict(torch.load('models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % 26))
for param in inn_2d_1.parameters():
    param.requires_grad = False
########################################################################################################################

trainer = pl.Trainer(max_epochs=Config.max_epoch, check_val_every_n_epoch=10)
gain = Gain()
trainer.fit(gain)
