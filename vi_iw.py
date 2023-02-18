import numpy as np
import torch
import random
import copy
from vi_base import baseVI
from utils import log_gaussian, kld, torch_from_numpy, logireg_loss
from model_bamdp import RatioModel

device = torch.device('cpu')

weight_normalize_flag = False

class iwVI(baseVI):
    def __init__(self, args_init_dict):
        super().__init__(args_init_dict)
        self.ratio_model = RatioModel(self.s_dim, self.a_dim, self.z_dim)

    def save(self):
        super().save(ckpt_name="vi_iw_ckpt_basepart")
        torch.save({'ratio_model_state_dict': self.ratio_model.state_dict()
                   },"vi_iw_ckpt")

    def load(self):
        try:
            super().load(ckpt_name="vi_iw_ckpt_basepart")
            print("success load vi_iw_ckpt_basepart")
        except:
            print("fail load vi_iw_ckpt_basepart")
        checkpoint = torch.load("vi_iw_ckpt")
        self.ratio_model.load_state_dict(checkpoint['ratio_model_state_dict'])
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()

    def load_base(self):
        super().load()


    def train_ratio(self, num_iter, lr, early_stop_step, policy):

        self.policy = policy
        param_list = list(self.ratio_model.parameters())
        loss_fn = self._loss_train_ratio
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)

        self.offlinedata_weight = [None]*len(self.offline_data)
        self.offlinedata_weight_sum = [None]*len(self.offline_data)
        with torch.no_grad():
            for m in range(len(self.offline_data)):
                temp_offline_sa = self.offline_data[m][:,:(self.sa_dim)]
                len_data = len(temp_offline_sa)
                z_mulogvar_offline = self.mulogvar_list_for_mixture_of_gaussian_belief[m]
                z = z_mulogvar_offline[:self.z_dim]*torch.ones(len_data, self.z_dim)
                g = z_mulogvar_offline*torch.ones(len_data, 2*self.z_dim)
                de_input_data = torch.cat([temp_offline_sa, z, g], axis=1)
                de_output_data = self.ratio_model(de_input_data)
                self.offlinedata_weight[m] = de_output_data.clone()
                self.offlinedata_weight_sum[m] = self.offlinedata_weight[m].sum().numpy()
        return ret

    def train_weighted_vae(self, num_iter, lr, early_stop_step, weight_alpha, flag=1):

        self.weight_alpha = weight_alpha
        if flag==1:
            print("train_weighted_vae: enc_dec")
            param_list = list(self.enc.parameters())+list(self.dec.parameters())
        elif flag==2:
            print("train_weighted_vae: enc")
            param_list = list(self.enc.parameters())
        elif flag==3:
            print("train_weighted_vae: dec")
            param_list = list(self.dec.parameters())
        else:
            return [], []
        loss_fn = self._loss_train_weighted_vae
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
        return ret

    def eval_loss(self, weight_alpha):
        self.weight_alpha = weight_alpha
        total_idx_list = np.array( range(len(self.offline_data)) )
        train_idx_list = copy.deepcopy(total_idx_list)[self.validdata_num:]
        valid_idx_list = copy.deepcopy(total_idx_list)[:self.validdata_num]
        loss_fn = self._loss_train_weighted_vae
        ave_num = 10        
        with torch.no_grad():

            train_loss_list = []
            for _ in range(ave_num):
                temp_train_loss = 0
                for m in train_idx_list:
                    temp_train_loss += loss_fn(m, flag=True).item()
                temp_train_loss /= len(train_idx_list)
                train_loss_list.append(temp_train_loss)
            train_loss_list = np.array(train_loss_list)
            train_loss = train_loss_list.mean()

            valid_loss_list = []
            for _ in range(ave_num):
                temp_valid_loss = 0
                for m in valid_idx_list:
                    temp_valid_loss += loss_fn(m, flag=True).item()
                temp_valid_loss /= len(valid_idx_list)
                valid_loss_list.append(temp_valid_loss)
            valid_loss_list = np.array(valid_loss_list)
            valid_loss = valid_loss_list.mean()

        print("train_loss: ",train_loss)
        print("valid_loss: ",valid_loss)


    def _loss_train_ratio(self, m, flag=False):

        temp_offline_sa = self.offline_data[m][:,:(self.sa_dim)]

        len_data = len(temp_offline_sa)
        with torch.no_grad():
            z_mulogvar_offline = self.mulogvar_list_for_mixture_of_gaussian_belief[m]
        simulation_data_saz = torch_from_numpy(self.simenv_rolloutdata[m])
        z = self.sample_z(z_mulogvar_offline, len_data)
        g = z_mulogvar_offline*torch.ones(len_data, 2*self.z_dim)
        de_input_data = torch.cat([temp_offline_sa, z, g], axis=1)
        nu_input_data = torch.cat([simulation_data_saz, g], axis=1)
        de_output_data = self.ratio_model(de_input_data)
        nu_output_data = self.ratio_model(nu_input_data)
        loss = logireg_loss(de_output_data, nu_output_data)
        return loss


    def _loss_train_weighted_vae(self, m, flag=False):
        temp_data_m = self.offline_data[m]
        z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])

        z = self.sample_z(z_mulogvar, 1).flatten() * torch.ones(len(temp_data_m), self.z_dim)

        saz = torch.cat([temp_data_m[:, :(self.sa_dim)], z], dim=1)
        ds_mulogvar = self.dec(saz)
        ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]

        with torch.no_grad():
            zg = torch.cat([z*torch.ones(len(temp_data_m),self.z_dim), z_mulogvar*torch.ones(len(temp_data_m), 2*self.z_dim)], axis=1)
            w = self.ratio_model(torch.cat([temp_data_m[:, :(self.sa_dim)], zg], axis=1)).flatten()
            if weight_normalize_flag:
                w *= len(temp_data_m) / np.mean(self.offlinedata_weight_sum)


        loss = 0

        # Approximate of E_{z~q}[ - log p(y|x,z) ]
        if flag:
            temp_alpha = 1
        else:
            temp_alpha = self.weight_alpha
            # temp_alpha = self.weight_alpha

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
