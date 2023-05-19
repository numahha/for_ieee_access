import numpy as np
import torch
from config import cfg_dec_hidden

clamp_logvarmin=-8 #-8 # -10
clamp_ratiomin=1e-8
clamp_ratiomax=1e8
clamp_flag = True


class Encoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim):
        super(Encoder, self).__init__()
        h_dim=32
        self.activate_fn=torch.nn.ReLU
        self.z_dim = z_dim
        self.zz_dim = 2*z_dim
        self.net1 = torch.nn.Sequential(
                            torch.nn.Linear(2*s_dim+a_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, self.zz_dim)
                            )
        self.net2 = torch.nn.Sequential(
                            torch.nn.Linear(self.zz_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            self.activate_fn(),
                            torch.nn.Linear(h_dim, 2*self.z_dim)
                            )

    def forward(self, data):
        x = self.net1(data)
        mu_logvar = self.net2(x.sum(0))
        logvar = torch.clamp(mu_logvar[self.z_dim:], min=clamp_logvarmin)
        return torch.hstack([mu_logvar[:self.z_dim], logvar])


class Decoder(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim):
        super(Decoder, self).__init__()
        h_dim=cfg_dec_hidden
        print("dec: h_dim",h_dim)
        self.s_dim=s_dim
        self.saz_dim = s_dim+a_dim+z_dim
        self.activate_fn=torch.nn.ReLU
        self.net_phat = torch.nn.Sequential(
                            torch.nn.Linear(self.saz_dim, h_dim),
                            self.activate_fn(),
#                             torch.nn.Linear(h_dim, h_dim),
#                             self.activate_fn(),
                            # torch.nn.Linear(h_dim, h_dim),
                            # self.activate_fn(),
                            torch.nn.Linear(h_dim, 2*s_dim)
                            )

    def forward(self, saz):
        if clamp_flag:
            mu_logvar = self.net_phat(saz)
            logvar = mu_logvar[:, self.s_dim:]
            logvar = torch.clamp(logvar, min=clamp_logvarmin)
            return  torch.hstack([mu_logvar[:, :self.s_dim], logvar])
        else:
            return self.net_phat(saz)

    def my_np_compile(self):
        self.my_np_layer=[]
        with torch.no_grad():
            for i in range(len(self.net_phat)):
                l = self.net_phat[i]
                if type(l)==torch.nn.modules.linear.Linear:
                    self.my_np_layer.append(["linear", l.weight.numpy(), l.bias.numpy()])
                elif type(l)==torch.nn.modules.activation.ReLU:
                    self.my_np_layer.append(["relu"])
                else:
                    print(type(l))
                    raise Exception

    def my_np_forward(self, x):
        x = x.reshape(-1,self.saz_dim).T
        for i in range(len(self.net_phat)):
            if self.my_np_layer[i][0] == "linear":
                # print(self.my_np_layer[i][1].shape, x.shape)
                x = self.my_np_layer[i][1] @ x + self.my_np_layer[i][2].reshape(-1,1)
            if self.my_np_layer[i][0] == "relu":
                x = np.clip(x, 0, None)
        x = x.T
        x = np.hstack([x[:,:self.s_dim], np.clip(x[:,self.s_dim:],clamp_logvarmin,None)])
        return x



class RatioModel(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim ):
        super(RatioModel, self).__init__()
        # h_dim=16
        h_dim=32
        activate_fn=torch.nn.Tanh
        self.net = torch.nn.Sequential(
                            torch.nn.Linear(s_dim+a_dim+2*z_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            # torch.nn.Linear(h_dim, h_dim),
                            # activate_fn(),
                            torch.nn.Linear(h_dim,1),
                            torch.nn.Softplus()
                            )

    def forward(self, sazg):
        return torch.clamp(self.net(sazg), min=clamp_ratiomin, max=clamp_ratiomax)



class PenaltyModel(torch.nn.Module):
    def __init__(self, s_dim, a_dim, z_dim ):
        super(PenaltyModel, self).__init__()
        h_dim=16
        activate_fn=torch.nn.Tanh
        self.net = torch.nn.Sequential(
                            torch.nn.Linear(s_dim+a_dim+3*z_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim, h_dim),
                            activate_fn(),
                            torch.nn.Linear(h_dim,1)
                            )

    def forward(self, sazg):
        return self.net(sazg)