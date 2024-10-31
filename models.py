import torch
import torch.nn as nn
from human_graph import get_graph
import numpy as np


class res_block(nn.Module):
    def __init__(self, num_neurons: int = 1024, use_batchnorm: bool = False):
        super(res_block, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.l1 = nn.Linear(num_neurons, num_neurons)
        self.bn1 = nn.BatchNorm1d(num_neurons)
        self.l2 = nn.Linear(num_neurons, num_neurons)
        self.bn2 = nn.BatchNorm1d(num_neurons)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = nn.LeakyReLU()(self.l2(x))
        if self.use_batchnorm:
            x = self.bn2(x)
        x += inp

        return x


class DepthAngleEstimator(nn.Module):
    def __init__(self, use_batchnorm=False, in_dim=2, num_joints=17, h_dim=32):
        super(DepthAngleEstimator, self).__init__()
        self.graph = graph_conv(in_dim, h_dim)
        self.upscale = nn.Linear(h_dim*num_joints, 1024)
        self.res_common = res_block(use_batchnorm=use_batchnorm)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm)
        # self.res_pose3 = res_block(use_batchnorm=use_batchnorm)
        # self.res_pose4 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm)
        # self.res_angle3 = res_block(use_batchnorm=use_batchnorm)
        # self.res_angle4 = res_block(use_batchnorm=use_batchnorm)
        self.depth = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        batch_size = x.shape[0]
        x = self.graph(x)
        x = x.reshape(batch_size, -1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        # xd = nn.LeakyReLU()(self.res_pose3(xd))
        # xd = nn.LeakyReLU()(self.res_pose4(xd))
        xd = self.depth(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = self.angles(xa)
        xa = xa

        return xd, xa


'''
class Generator(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super().__init__()
        self.positions = nn.Parameter(torch.randn((1, 17, h_dim)), requires_grad=True)
        self.up_dim = nn.Linear(dim+h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.down_dim = nn.Linear(h_dim, dim)

    def forward(self, x, m):
        batch = x.shape[0]
        dim = self.positions.shape[-1]

        p = self.positions.repeat(batch, 1, 1)
        k = p[m].reshape(batch, -1, dim)
        q = p[m == False].reshape(batch, -1, dim)
        x = torch.cat([x, k], dim=-1)
        up_x = nn.LeakyReLU()(self.up_dim(x))
        v = self.V(up_x)
        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / dim**0.5)
        out1 = nn.LeakyReLU()(att @ v)

        out = self.down_dim(out1)

        return out'''


class Discriminator(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super().__init__()
        self.positions = nn.Parameter(torch.randn((17, h_dim)), requires_grad=True)
        self.up_dim = nn.Linear(dim+h_dim, h_dim)
        self.Q = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.down_dim = nn.Linear(h_dim, 1)

    def forward(self, x):
        batch = x.shape[0]
        dim = self.positions.shape[1]

        p = self.positions[None].repeat(batch, 1, 1)

        up_x = torch.cat([x, p], dim=-1)
        up_x = nn.LeakyReLU()(self.up_dim(up_x))

        q = self.Q(up_x)
        k = self.K(up_x)
        v = self.V(up_x)

        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / dim ** 0.5)
        out1 = nn.LeakyReLU()(att @ v)

        out = nn.Sigmoid()(self.down_dim(out1).squeeze(-1))

        return out


class Transformer_da_Estimation(nn.Module):
    def __init__(self, dim=3, h_dim=32):
        super().__init__()
        self.positions = nn.Parameter(torch.randn((17, h_dim)), requires_grad=True)
        self.up_dim = nn.Linear(dim + h_dim, h_dim)
        self.Q = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.down_dim = nn.Linear(h_dim, 1)

    def forward(self, x):
        batch = x.shape[0]
        dim = self.positions.shape[1]

        p = self.positions[None].repeat(batch, 1, 1)

        up_x = torch.cat([x, p], dim=-1)
        up_x = nn.LeakyReLU()(self.up_dim(up_x))

        q = self.Q(up_x)
        k = self.K(up_x)
        v = self.V(up_x)

        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / dim ** 0.5)
        out1 = nn.LeakyReLU()(att @ v)

        out = self.down_dim(out1).squeeze(-1)

        return out


class WeightLinear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(WeightLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(size=(1, in_feature, out_feature)))
        self.bias = nn.Parameter(torch.randn(size=(1, out_feature)))

    def forward(self, score, x):
        """

        :param score: b, 34
        :param x: b, 34
        :return:
        """
        weight = self.weight * score[:, :, None]
        x = x[:, None] @ weight + self.bias

        return x.reshape(x.shape[0], -1)


class WeightAttention(nn.Module):
    def __init__(self, in_dim=2, dim=32, num_joints=17):
        super(WeightAttention, self).__init__()
        positions = torch.randn(size=(1, num_joints, dim))
        self.positions = nn.Parameter(positions)
        self.up_dim = nn.Linear(in_dim, dim)


class DepthAngleEstimator_wight(nn.Module):
    def __init__(self, use_batchnorm=False, num_joints=16):
        super(DepthAngleEstimator_wight, self).__init__()

        self.upscale = WeightLinear(2*num_joints, 1024)
        self.score_machine = nn.Sequential(
            nn.Linear(num_joints, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_joints)
        )
        self.res_common = res_block(use_batchnorm=use_batchnorm)
        self.res_pose1 = res_block(use_batchnorm=use_batchnorm)
        self.res_pose2 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle1 = res_block(use_batchnorm=use_batchnorm)
        self.res_angle2 = res_block(use_batchnorm=use_batchnorm)
        self.depth = nn.Linear(1024, num_joints)
        self.angles = nn.Linear(1024, 1)

    def forward(self, x_score):
        x = x_score[0]
        score = x_score[1]
        batch_size = score.shape[0]
        score = self.score_machine(score)
        score = nn.Softmax(dim=-1)(score)
        score = torch.cat([score, score], dim=1).reshape(batch_size, 2, -1).transpose(2, 1).reshape(batch_size, -1)
        x = self.upscale(score, x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xd = nn.LeakyReLU()(self.res_pose1(x))
        xd = nn.LeakyReLU()(self.res_pose2(xd))
        xd = self.depth(xd)

        # depth path
        xa = nn.LeakyReLU()(self.res_angle1(x))
        xa = nn.LeakyReLU()(self.res_angle2(xa))
        xa = self.angles(xa)

        return xd, xa


class ScoreMechain(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super(ScoreMechain, self).__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.up_dim = nn.Linear(dim*17, h_dim*17)
        self.self_att = SelfAtt(h_dim)
        self.down_dim = nn.Linear(h_dim, 1)
        self.batch_norm = nn.BatchNorm1d(17)
        self.layer_norm = nn.LayerNorm(17)

    def forward(self, x, m):
        batch = x.shape[0]
        x = x.reshape(batch, -1)

        up_x = nn.LeakyReLU()(self.up_dim(x)).reshape(batch, -1, self.h_dim)
        up_x1 = up_x[m].reshape(batch, -1, self.h_dim)
        up_x2 = up_x[m == False].reshape(batch, -1, self.h_dim)

        out1 = self.self_att(up_x1, up_x2, m)

        out = nn.Sigmoid()(self.down_dim(out1)).squeeze(-1)

        return out


'''class Generator(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super().__init__()
        self.positions = nn.Parameter(torch.randn((1, 17, h_dim)), requires_grad=True)

        self.att_up_dim = nn.Linear(dim+h_dim, h_dim)
        self.att_V = nn.Linear(h_dim, h_dim)
        self.att_down_dim = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, dim),
        )

        self.self_att_up_dim = nn.Linear(dim+h_dim, h_dim)
        self.self_att_Q = nn.Linear(h_dim, h_dim)
        self.self_att_K = nn.Linear(h_dim, h_dim)
        self.self_att_V = nn.Linear(h_dim, h_dim)
        self.self_att_down_dim = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, dim),
        )

    def forward(self, x, m):
        batch = x.shape[0]
        dim = self.positions.shape[-1]

        p = self.positions.repeat(batch, 1, 1)
        k = p[m].reshape(batch, -1, dim)
        q = p[m == False].reshape(batch, -1, dim)
        x = torch.cat([x, k], dim=-1)

        up_x = nn.LeakyReLU()(self.att_up_dim(x))
        v = self.att_V(up_x)
        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / dim**0.5)
        att_out1 = nn.LeakyReLU()(att @ v)

        att_out = self.att_down_dim(att_out1)

        new_x2 = torch.cat([att_out, q], dim=-1)
        new_x = torch.cat([x, new_x2], dim=1)
        new_up_x = nn.LeakyReLU()(self.self_att_up_dim(new_x))
        new_up_x2 = nn.LeakyReLU()(self.self_att_up_dim(new_x2))
        Q = self.self_att_Q(new_up_x2)
        K = self.self_att_K(new_up_x)
        V = self.self_att_V(new_up_x)
        self_att = nn.Softmax(dim=-1)((Q @ K.transpose(2, 1)) / dim**0.5)
        self_att_out1 = nn.LeakyReLU()(self_att @ V)
        out = self.self_att_down_dim(self_att_out1) + att_out

        return out
'''


class Att(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super(Att, self).__init__()
        self.dim = dim
        self.h_dim = h_dim

        self.positions = nn.Parameter(
            torch.randn(size=(1, 17, h_dim)),
            requires_grad=True
        )

        self.up_dim = nn.Linear(dim+h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.down_dim = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, x, m_bool):
        batch = x.shape[0]

        p = self.positions.repeat(batch, 1, 1)
        k = p[m_bool].reshape(batch, -1, self.h_dim)
        q = p[m_bool == False].reshape(batch, -1, self.h_dim)
        x = torch.cat([x, k], dim=-1)

        up_x = nn.LeakyReLU()(self.up_dim(x))
        k = self.K(up_x)
        v = self.V(up_x)
        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / self.dim**0.5)
        out1 = att @ v

        out = self.down_dim(out1)

        return out


class SelfAtt(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super(SelfAtt, self).__init__()
        self.dim = dim
        self.h_dim = h_dim

        self.positions = nn.Parameter(
            torch.randn(size=(1, 17, h_dim)),
            requires_grad=True
        )
        self.up_dim = nn.Linear(dim+h_dim, h_dim)
        self.Q = nn.Linear(h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.down_dim = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, x1, x2, m_bool):
        batch = x1.shape[0]

        p = self.positions.repeat(batch, 1, 1)
        k = p[m_bool].reshape(batch, -1, self.h_dim)
        q = p[m_bool == False].reshape(batch, -1, self.h_dim)
        x1 = torch.cat([x1, k], dim=-1)
        x2 = torch.cat([x2, q], dim=-1)

        up_x1 = nn.LeakyReLU()(self.up_dim(x1))
        up_x2 = nn.LeakyReLU()(self.up_dim(x2))
        up_x = torch.cat([up_x1, up_x2], dim=1)

        q = self.Q(up_x2)
        k = self.K(up_x)
        v = self.V(up_x)

        att = nn.Softmax(dim=-1)((q @ k.transpose(2, 1)) / self.dim**0.5)
        out1 = att @ v

        out = self.down_dim(out1) + x2[:, :, :self.h_dim]
        return out

    def get_att_matrix(self, x):
        batch = x.shape[0]
        nag_inf = -1e50*torch.ones(17).to(x)
        nag_inf = torch.diag(nag_inf).reshape(1, 17, 17)
        nag_inf = nag_inf.repeat(batch, 1, 1)

        p = self.positions.repeat(batch, 1, 1)
        x = torch.cat([x, p], dim=-1)
        up_x = self.up_dim(x)

        q = self.Q(up_x)
        k = self.K(up_x)
        att = (q @ k.transpose(2, 1)) / self.dim ** 0.5
        att += nag_inf
        att = nn.Softmax(dim=-1)(att)

        return att


class Generator(nn.Module):
    def __init__(self, dim=2, h_dim=32):
        super(Generator, self).__init__()
        self.dim = dim
        self.h_dim = h_dim
        self.att = Att(h_dim, h_dim)
        self.self_att1 = SelfAtt(h_dim, h_dim)
        self.self_att2 = SelfAtt(h_dim, h_dim)
        #self.self_att3 = SelfAtt(h_dim, h_dim)
        self.up_dim = nn.Linear(dim, h_dim)
        self.down_dim = nn.Linear(h_dim, dim)

    def forward(self, x, m_bool):
        x = self.up_dim(x)
        att_out = nn.LeakyReLU()(self.att(x, m_bool))
        self_att_out1 = nn.LeakyReLU()(self.self_att1(x, att_out, m_bool))
        self_att_out2 = nn.LeakyReLU()(self.self_att2(x, self_att_out1, m_bool))
        #self_att_out3 = nn.LeakyReLU()(self.self_att3(x, self_att_out2, m_bool))
        out = self.down_dim(self_att_out2)

        return out

    def get_att_matrix(self, x):
        x = self.up_dim(x)
        self_att = self.self_att2.get_att_matrix(x)

        return self_att


class create_new_pose_net(nn.Module):
    def __init__(self, in_dim=2*2*17, h_dim=1024):
        super(create_new_pose_net, self).__init__()
        up_dim = nn.Linear(in_dim, h_dim)
        l1 = res_block(h_dim)
        l2 = res_block(h_dim)
        l3 = res_block(h_dim)
        down_dim = nn.Linear(h_dim, 1)
        self.seq = nn.Sequential(
            up_dim,
            nn.LeakyReLU(),
            l1,
            nn.LeakyReLU(),
            l2,
            nn.LeakyReLU(),
            l3,
            nn.LeakyReLU(),
            down_dim,
        )

    def forward(self, x):
        x = self.seq(x).squeeze()
        x = nn.Sigmoid()(x)
        return x


class generator_feature(nn.Module):
    def __init__(self, in_dim=2, h_dim=32):
        super(generator_feature, self).__init__()
        self.up_dim = nn.Linear(in_dim, h_dim)
        self.self_att1 = SelfAtt(h_dim, h_dim)
        self.self_att2 = SelfAtt(h_dim, h_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        m_bool = torch.tensor(
            [[False] * 17] * batch_size, device='cuda:0'
        )
        x = self.up_dim(x)
        _ = torch.zeros((x.shape[0], 0, x.shape[2])).to(x)
        x = self.self_att1(_, x, m_bool)
        x = self.self_att2(_, x, m_bool)

        return x


class graph_conv(nn.Module):
    def __init__(self, in_dim=2, h_dim=32):
        super(graph_conv, self).__init__()
        self.weight_1 = nn.Parameter(
            torch.randn(size=(1, in_dim, h_dim), requires_grad=True),
            requires_grad=True
        )
        graph = get_graph()
        du = graph.sum(axis=-1)
        du_matrix = np.diag(du)
        r_du_matrix = np.diag(du**(-0.5))
        laplace = r_du_matrix @ (du_matrix - graph) @ r_du_matrix
        self.laplace = nn.Parameter(
            torch.tensor(laplace[None, ], requires_grad=False).to(self.weight_1), requires_grad=False
        )

    def forward(self, x):
        x = self.laplace @ x @ self.weight_1
        return x
