from H36Dataset import H36MDataset
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from utils import subnet_fc
from norm_function import *
from torch.utils.data import DataLoader
from torch.optim import Adam


inn_2d = Ff.SequenceINN(26)
for k in range(8):
    inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

train_dataset = H36MDataset('mpii', list(range(1, 9)), get_PCA=True, normalize_func=normalize_head, h36m=False, occ=False)
pca = train_dataset.pca
loader = DataLoader(train_dataset, 256, True)
test_dataset = H36MDataset('mpii', list((range(9, 15))), normalize_func=normalize_head_test, get_PCA=False, h36m=False, occ=False)
test_loader = DataLoader(test_dataset, 1, False)
lr = 0.0001
inn_2d.cuda()
opt = Adam(inn_2d.parameters(), lr=lr)
for i in range(25):
    for data in loader:
        d2_pose = data['p2d_gt'].cuda()
        norm_pose = d2_pose.reshape(-1, 34) - \
                    torch.tensor(pca.mean_.reshape(1, 34)).to(d2_pose)
        latent = norm_pose @ torch.tensor(pca.components_.T).to(d2_pose)
        z, log_jac_det = inn_2d(latent[:, :26])
        loss_d = (0.5 * torch.sum(z ** 2, 1) - log_jac_det).mean()
        opt.zero_grad()
        loss_d.backward()
        opt.step()
        print(f'epoch:{i}', ' loss:', loss_d.item())
for data in test_loader:
    with torch.no_grad():
        d2_pose = data['p2d_gt'].cuda()
        norm_pose = d2_pose.reshape(-1, 34) - \
                    torch.tensor(pca.mean_.reshape(1, 34)).to(d2_pose)
        latent = norm_pose @ torch.tensor(pca.components_.T).to(d2_pose)
        z, log_jac_det = inn_2d(latent[:, :26])
        loss_d = (0.5 * torch.sum(z ** 2, 1) - log_jac_det).mean()
        print(f'epoch:test, loss_d:{loss_d}')
torch.save(inn_2d.state_dict(), 'oc_inn2d_mpii.pt')