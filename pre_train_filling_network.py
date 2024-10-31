import random
from losses import *
from norm_function import *
from H36Dataset import H36MDataset
import pytorch_lightning as pl
from models import Generator
from torch.utils.data import DataLoader


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
    l_begin = 50
    max_epoch = 100
    occ = True if test_file == 'oc_dataset' else False


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)


class FillingNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = generator
        self.opts = {}

    def train_dataloader(self):
        train_set = H36MDataset(Config.train_file, Config.dataset['train'], normalize_func=normalize_head, get_PCA=False,
                                h36m=True, occ=False)
        train_loader = DataLoader(train_set, Config.batch_size, True)

        return train_loader

    def val_dataloader(self):
        test_set = H36MDataset(Config.test_file, Config.dataset['test'], normalize_func=normalize_head_test,
                               h36m=True, occ=False)
        test_loader = DataLoader(test_set, 100, True)

        return test_loader

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=Config.g_lr)

        self.opts['g_opt'] = g_opt

        return g_opt

    def training_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']
        batch_size = input_2d.shape[0]
        d2_pose = input_2d.clone()
        d2_pose = d2_pose.reshape(-1, 2, 17).transpose(2, 1)

        m = [np.random.permutation(17) for i in range(batch_size)]
        m = np.array(m)[:, :Config.n_miss]
        m = np.sort(m)
        m_bool = torch.tensor([[True] * 17] * batch_size, device='cuda:0')
        for i in range(batch_size):
            m_bool[i, m[i]] = False

        x_hat = self.F(d2_pose, m_bool, m)
        mse_loss = generator_loss(d2_pose, x_hat)
        f_loss = mse_loss
        self.log_dict({'mse_loss': mse_loss}, prog_bar=True)

        return f_loss

    def validation_step(self, batch, batch_idx):
        input_2d = batch['p2d_gt']

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

        x_hat = self.F(d2_pose, m_bool, m)
        val_f_loss = generator_loss(d2_pose, x_hat)

        self.log_dict({'val_f_loss': val_f_loss}, prog_bar=True)

    def F(self, x, m_bool, m):
        batch_size = x.shape[0]

        g_x = x[m_bool].reshape(batch_size, -1, 2)
        x_hat = self.generator(g_x, m_bool)
        g_result = x.clone()
        for i in range(batch_size):
            g_result[i, m[i]] = x_hat[i]

        return g_result

    def on_train_epoch_end(self):
        for p in self.opts['g_opt'].param_groups:
            p['lr'] *= Config.decay_rate


generator = Generator(h_dim=96)
if __name__ == '__main__':
    set_seed()
    callback = pl.callbacks.ModelCheckpoint('fillnet', 'human36_fill_net-{epoch}-{val_f_loss}', save_top_k=1, mode='min',
                                            monitor='val_f_loss')
    trainer = pl.Trainer(max_epochs=Config.max_epoch, check_val_every_n_epoch=5, callbacks=callback)
    fill_net = FillingNetwork()
    trainer.fit(fill_net)
