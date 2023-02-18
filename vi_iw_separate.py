import numpy as np
import torch
import random
import copy
from vi_base import baseVI
from utils import log_gaussian, kld, logsumexp, torch_from_numpy, logireg_loss
from model_bamdp import Encoder, Decoder, RatioModel2
import matplotlib.pyplot as plt

device = torch.device('cpu')

class iwVI(baseVI):
    def __init__(self, args_init_dict):
        super().__init__(args_init_dict)
        self.ratio_model_list = [RatioModel2(self.s_dim, self.a_dim, self.z_dim) for _ in range(len(self.offline_data))]

    def save(self):
        super().save(ckpt_name="vi_iw_ckpt_basepart")
        # torch.save({'ratio_model_state_dict': self.ratio_model.state_dict()
        #            },"vi_iw_ckpt")

    def load(self):
        try:
            super().load(ckpt_name="vi_iw_ckpt_basepart")
            print("success load vi_iw_ckpt_basepart")
        except:
            print("fail load vi_iw_ckpt_basepart")
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
        checkpoint = torch.load("vi_iw_ckpt")
        # self.ratio_model.load_state_dict(checkpoint['ratio_model_state_dict'])

    def load_base(self):
        super().load()

    # def _loss_train_ratio_m(self, valid_flag=False):
    #     # m = 35
    #     with torch.no_grad():
    #         z_mulogvar_offline = self.mulogvar_list_for_mixture_of_gaussian_belief[m]
    #     simulation_data_saz = torch_from_numpy(self.simenv_rolloutdata[m])
    #     z = simulation_data_saz[:,self.sa_dim:]
    #     g = z_mulogvar_offline*torch.ones(len_data, 2*self.z_dim)
    #     de_input_data = torch.cat([temp_offline_sa, z, g], axis=1)
    #     nu_input_data = torch.cat([simulation_data_saz, g], axis=1)
    #     de_output_data = self.ratio_model(de_input_data)
    #     nu_output_data = self.ratio_model(nu_input_data)
    #     loss = logireg_loss(de_output_data, nu_output_data)
    #     self.offlinedata_weight[m] = de_output_data.detach().clone()
    #     self.offlinedata_weight[m] *= len_data/self.offlinedata_weight[m].sum()
    #     return loss



    def train_ratio(self, num_iter, lr, early_stop_step, policy):

        self.policy = policy
        self.offlinedata_weight = [None]*len(self.offline_data)

        print("train ratio list")
        for m in range(len(self.offline_data)):
            optimizer = torch.optim.Adam(self.ratio_model_list[m].parameters(), lr=lr)
            temp_offline_sa = self.offline_data[m][:,:(self.sa_dim)]
            simulation_data_saz = torch_from_numpy(self.simenv_rolloutdata[m])
            z_mulogvar = self.mulogvar_list_for_mixture_of_gaussian_belief[m]
            nu_input_data = simulation_data_saz
            best_loss = 1e10
            best_iter = 0
            train_idx = np.array( range(len(self.offline_data)) )
            random.shuffle(train_idx)
            # valid_idx = train_idx[int(len(self.offline_data)*0.5):].copy()
            # train_idx = train_idx[:int(len(self.offline_data)*0.5)].copy()
            valid_idx = train_idx[:]
            train_idx = train_idx[:]
            for i in range(num_iter):
                z = self.sample_z(z_mulogvar, len(temp_offline_sa))
                de_input_data = torch.cat([temp_offline_sa, z], axis=1)
                de_output_data = self.ratio_model_list[m](de_input_data[train_idx])
                nu_output_data = self.ratio_model_list[m](nu_input_data[train_idx])
                loss = logireg_loss(de_output_data, nu_output_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    de_output_data = self.ratio_model_list[m](de_input_data[valid_idx])
                    nu_output_data = self.ratio_model_list[m](nu_input_data[valid_idx])
                    loss = logireg_loss(de_output_data, nu_output_data)
                if best_loss>loss.item():
                    best_loss = loss.item()
                    best_iter=i*1
                    temp_ratio = copy.deepcopy(self.ratio_model_list[m])
                if i%2000==0:
                    print("task:", m, ", iter:", i,", best_loss:", best_loss)
                if (i-best_iter)>early_stop_step:
                    break

            with torch.no_grad():
                z = self.sample_z(z_mulogvar, len(temp_offline_sa))
                de_input_data = torch.cat([temp_offline_sa, z], axis=1)
                self.offlinedata_weight[m] = self.ratio_model_list[m](de_input_data).detach()
            print("task:", m, ", iter:", i,", best_loss:", best_loss)
            fig = plt.figure(figsize=(18,12), dpi=200)
            fig.patch.set_facecolor('white')
            plt.plot(nu_input_data[:,0],nu_input_data[:,1],"kx")
            plt.scatter(self.offline_data[m][:,0], self.offline_data[m][:,1], c=np.log10(self.offlinedata_weight[m].numpy()))
            plt.colorbar()
            plt.savefig("ratio"+str(m)+".png")
            plt.close()

            self.ratio_model_list[m] = copy.deepcopy(temp_ratio)
        return None

    def train_weighted_vae(self, num_iter, lr, early_stop_step, weight_alpha):

        self.weight_alpha = weight_alpha
        param_list = list(self.enc.parameters())+list(self.dec.parameters())
        loss_fn = self._loss_train_weighted_vae
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
        return ret


    def _loss_train_weighted_vae(self, m, valid_flag=False):
        temp_data_m = self.offline_data[m]
        z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
        if not valid_flag:
            z = self.sample_z(z_mulogvar, len(temp_data_m))
        else:
            z = z_mulogvar[:self.z_dim] * torch.ones(len(temp_data_m), self.z_dim)
        saz = torch.cat([temp_data_m[:, :(self.sa_dim)], z], dim=1)
        ds_mulogvar = self.dec(saz)
        ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]

        with torch.no_grad():
            w = self.ratio_model_list[m](torch.cat([temp_data_m[:, :(self.sa_dim)], z], axis=1)).flatten()
            # w = self.offlinedata_weight[m]
        w = len(temp_data_m) * w / w.sum()

        loss = 0

        # Approximate of E_{z~q}[ - log p(y|x,z) ]
        if not valid_flag:
            temp_alpha = self.weight_alpha
        else:
            temp_alpha = 1

        loss += - (log_gaussian(ds_m, # y
                               ds_mulogvar[:, :self.s_dim], # mu
                               ds_mulogvar[:, self.s_dim:] # logvar
                               ) * (w**temp_alpha)).sum()

        # nu * E_{z~q}[ log q(z) - log p(z) ]
        loss += self.nu * kld(z_mulogvar[:self.z_dim],
                              z_mulogvar[self.z_dim:],
                              self.prior[:self.z_dim],
                              self.prior[self.z_dim:])

        return loss
