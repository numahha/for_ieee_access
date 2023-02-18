import numpy as np
import torch
import random

from utils import log_gaussian, kld, logsumexp, torch_from_numpy
from model_bamdp import Encoder, Decoder#, PenaltyModel

device = torch.device('cpu')


class baseVI(torch.nn.Module):
    def __init__(self, args_init_dict):
        super(baseVI, self).__init__()

        offline_data = args_init_dict["offline_data"]
        s_dim = args_init_dict["s_dim"]
        a_dim = args_init_dict["a_dim"]
        z_dim = args_init_dict["z_dim"]
        env = args_init_dict["env"]

        train_valid_ratio = 0.2
        validdata_num = int(train_valid_ratio*len(offline_data))
        self.early_stopping_num = 50
        self.valid_ave_num=1 # validlossを計算するためのサンプル数

        self.gamma = 0.99
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.sa_dim = s_dim + a_dim
        self.sas_dim = 2*s_dim + a_dim
        self.z_dim = z_dim
        self.init_state_fn     = env.reset
        self.rew_fn            = env.env.env.rew_fn
        self._max_episode_steps = env.spec.max_episode_steps
        self.action_space = env.action_space
        self.observation_space = np.ones((s_dim+2*z_dim,2))

        self.train_data = offline_data[validdata_num:] # [M, N , |SAS'R|] : M ... num of MDPs, N ... trajectory length, |SAS'R| ... dim of (s,a,s',r)
        self.valid_data = offline_data[:validdata_num]
        for m in range(len(self.train_data)):
            self.train_data[m][:, (self.sa_dim):(self.sas_dim)] -= self.train_data[m][:, :(self.s_dim)] # ds = s'-s
        for m in range(len(self.valid_data)):
            self.valid_data[m][:, (self.sa_dim):(self.sas_dim)] -= self.valid_data[m][:, :(self.s_dim)]


        self.enc = Encoder(s_dim, a_dim, z_dim)         # q(z|D^train_m)
        self.enc_belief = Encoder(s_dim, a_dim, z_dim)  # beta^t(z)
        self.dec = Decoder(s_dim, a_dim, z_dim)         # p(ds|s,a,z)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=False)  # [mean, logvar] for VAE
        self.nu = 1e0 # KLDによる正則化の重み

        self.train_g_m_list=None
        self.train_mulogvar_list_for_mixture_of_gaussian_belief=None
        self.valid_g_m_list=None
        self.valid_mulogvar_list_for_mixture_of_gaussian_belief=None
        self.mulogvar_list_for_mixture_of_gaussian_belief=None
        # self.penalty_model = PenaltyModel(s_dim, a_dim, z_dim) # ibisには要らない
        # self.bamdapdada_for_sac = None

        self.lam=1e-4 # ペナルティの係数？

        # self.train_ds_m_list = []
        # self.valid_ds_m_list = []
        # for m in range(len(self.train_data)):
        #     temp_data_m = self.train_data[m]
        #     self.train_ds_m_list.append(temp_data_m[:, (self.sa_dim):(self.sas_dim)] - temp_data_m[:, :self.s_dim])
        # for m in range(len(self.valid_data)):
        #     temp_data_m = self.valid_data[m]
        #     self.valid_ds_m_list.append(temp_data_m[:, (self.sa_dim):(self.sas_dim)] - temp_data_m[:, :self.s_dim])

        self.initial_belief = torch.nn.Parameter(torch.zeros(2*z_dim))  # [mean, logvar]

            # self.mulogvar_list_for_mixture_of_gaussian_belief = []
            # for m in range(len(self.train_data)):
            #     temp_data_m = self.train_data[m]
            #     z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
            #     self.train_mulogvar_list_for_mixture_of_gaussian_belief.append(z_mulogvar)


    def save(self):
        torch.save({'enc_state_dict': self.enc.state_dict(),
                    'dec_state_dict': self.dec.state_dict(),
                    'prior': self.prior
                   },"vi_base_ckpt")

    def load(self):
        checkpoint = torch.load("vi_base_ckpt")
        self.enc.load_state_dict(checkpoint['enc_state_dict'])
        self.dec.load_state_dict(checkpoint['dec_state_dict'])
        self.prior = checkpoint['prior']

    def reset(self, z=None, fix_init=False):
        self.sim_timestep=0
        if z is None:
            std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
            eps = torch.randn_like(std)
            self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
        else:
            self.sim_z = torch_from_numpy(z.flatten())
        self.sim_s =  torch_from_numpy(self.init_state_fn(fix_init=fix_init).flatten())
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().flatten()
        sb = torch.hstack([self.sim_s, self.sim_b])
        return sb



    def step(self, a, update_belief=True, penalty_flag=False):
        a= torch_from_numpy(a.flatten())
        saz = torch.hstack([self.sim_s, a, self.sim_z]).reshape(1,-1)
        with torch.no_grad():
            ds_mulogvar = self.dec(saz).flatten()
            ds_mu = ds_mulogvar[:self.s_dim]
            eps = torch.randn_like(ds_mu)
            std = torch.exp(0.5 * ds_mulogvar[self.s_dim:])
            ds = (eps*std+ds_mu).flatten()
        rew = self.rew_fn(self.sim_s.numpy(), a.numpy())
        if penalty_flag:
            with torch.no_grad():
                penalty = self.penalty_model(torch.hstack([saz, self.train_g_m_list[self.sim_m]]))
            rew -= self.lam * penalty.flatten()[0]

        # current_data = torch.hstack([self.sim_s, a, self.sim_s+ds, torch.Tensor([rew])])
        # self.online_data = torch.vstack([self.online_data, current_data])
        self.sim_s = self.sim_s + ds
        done = False
        # if np.random.rand()>self.gamma:
        if self.sim_timestep>=(self._max_episode_steps-1):
            done=True
        self.sim_timestep+=1
        if update_belief:
            self.sim_b = self.get_belief(self.online_data[:, :(self.sas_dim)]).detach().flatten()
        sb = torch.hstack([self.sim_s, self.sim_b])
        return sb, rew, done, {}

    def episode_rollout(self, task_index=None, z=None, fix_init=False):
        self.sim_timestep=0
        if task_index is None:
            self.sim_m = np.random.randint(len(self.train_data))
        else:
            self.sim_m = task_index
        if z is None:
            # eps = torch.randn_like(self.train_belief_mu_list[self.sim_m])
            # std = torch.exp(0.5 * self.train_belief_logvar_list[self.sim_m])
            # self.sim_z = (eps*std+self.train_belief_mu_list[self.sim_m]).detach().flatten()
            std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
            eps = torch.randn_like(std)
            self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
        else:
            self.sim_z = torch_from_numpy(z.flatten())
        self.sim_s =  torch_from_numpy(self.init_state_fn(fix_init=fix_init).flatten())
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().flatten()
        sb = torch.hstack([self.sim_s, self.sim_b])
        return sb



    def get_belief(self, sads_array=None):
        with torch.no_grad():
            if sads_array is None or len(sads_array)==0:
                return self.initial_belief.detach()
            else:
                return self.enc(sads_array[:, :(self.sas_dim)])


    def train_unweighted_vae(self, num_iter, lr, early_stop_step):

        for p in self.enc.parameters():
            p.requires_grad = True
        for p in self.dec.parameters():
            p.requires_grad = True

#         param_list = list(self.enc.parameters())+list(self.dec.parameters())+[self.prior]
        param_list = list(self.enc.parameters())+list(self.dec.parameters())
        optimizer = torch.optim.Adam(param_list, lr=lr)

        idx_list = np.array( range(len(self.train_data)) )
        best_valid_loss = 1e10
        best_valid_iter = 0

        train_curve = []
        valid_curve = []
        for i in range(num_iter):

            random.shuffle(idx_list)
            train_loss = 0
            loss = 0.
            for m in range(len(idx_list)):
                loss += self._loss_train_unweighted_vae(self.train_data[idx_list[m]])
                # if m%5==4:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loss = 0.
            train_loss /= len(idx_list)

            if 0==i%1:
                with torch.no_grad():
                    valid_loss_list = []
                    for _ in range(self.valid_ave_num):
                        temp_valid_loss = 0
                        for m in range(len(self.valid_data)):
                            temp_valid_loss += self._loss_train_unweighted_vae(self.valid_data[m]).item()
                        temp_valid_loss /= len(self.valid_data)
                        valid_loss_list.append(temp_valid_loss)
                    valid_loss_list = np.array(valid_loss_list)
                    valid_loss = valid_loss_list.mean()

                if best_valid_loss>=valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_iter = i

                print("train_vae: iter",i,
                      " trainloss {:.5f}".format(train_loss),
                      " validloss {:.5f}".format(valid_loss)+"±{:.5f}".format(valid_loss_list.std()),
                      " bestvalidloss {:.5f}".format(best_valid_loss),
                      " last_update", i-best_valid_iter)
            train_curve.append(train_loss)
            valid_curve.append(valid_loss)
            if (i-best_valid_iter)>early_stop_step:
                break
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
        print("train_vae: fin")
        return train_curve, valid_curve


    def sample_z(self, z_mulogvar, datanum):
        # # reparametrization trick type A
        # std = torch.exp(0.5 * z_mulogvar[self.z_dim:])
        # eps = torch.randn(self.z_dim)
        # z = (eps*std+z_mulogvar[:self.z_dim]) * torch.ones(datanum, self.z_dim)

        # reparametrization trick type B
        std = torch.exp(0.5 * z_mulogvar[self.z_dim:]) * torch.ones(datanum, self.z_dim)
        eps = torch.randn(datanum, self.z_dim)
        z = eps * std + z_mulogvar[:self.z_dim]
        return z


    def _loss_train_unweighted_vae(self, temp_data_m):
        z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
        z = self.sample_z(z_mulogvar, len(temp_data_m))
        saz = torch.cat([temp_data_m[:, :(self.sa_dim)], z], dim=1)
        ds_mulogvar = self.dec(saz)
        ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]

        loss = 0
        # Approximate of E_{z~q}[ - log p(y|x,z) ]
        loss += - log_gaussian(ds_m, # y
                               ds_mulogvar[:, :self.s_dim], # mu
                               ds_mulogvar[:, self.s_dim:] # logvar
                               ).sum()

        # nu * E_{z~q}[ log q(z) - log p(z) ]
        loss += self.nu * kld(z_mulogvar[:self.z_dim],
                              z_mulogvar[self.z_dim:],
                              self.prior[:self.z_dim],
                              self.prior[self.z_dim:])

        # # Approximate of E_{z~q}[ log q(z)]
        # log_q_z = log_gaussian(z, # y
        #                        z_mulogvar[:self.z_dim], # mu
        #                        z_mulogvar[self.z_dim:] # logvar
        #                        ).sum()
        # # Approximate of E_{z~q}[ log p(z)]
        # log_p_z = log_gaussian(z, # y
        #                        self.prior[:self.z_dim], # mu
        #                        self.prior[self.z_dim:] # logvar
        #                        ).sum()
        # # nu * E_{z~q}[ log q(z) - log p(z) ]
        # loss += self.nu * (log_q_z-log_p_z)

        return loss


    def update_mulogvar_list_for_mixture_of_gaussian_belief(self):
        with torch.no_grad():
            self.train_mulogvar_list_for_mixture_of_gaussian_belief = []
            for m in range(len(self.train_data)):
                temp_data_m = self.train_data[m]
                z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
                self.train_mulogvar_list_for_mixture_of_gaussian_belief.append(z_mulogvar)
            self.valid_mulogvar_list_for_mixture_of_gaussian_belief = []
            for m in range(len(self.valid_data)):
                temp_data_m = self.valid_data[m]
                z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
                self.valid_mulogvar_list_for_mixture_of_gaussian_belief.append(z_mulogvar)

            self.mulogvar_list_for_mixture_of_gaussian_belief = self.valid_mulogvar_list_for_mixture_of_gaussian_belief + self.train_mulogvar_list_for_mixture_of_gaussian_belief



    # def train_enc_belief(self, num_iter, lr, early_stop_step):
    #
    #     for p in self.dec.parameters():
    #         p.requires_grad = False
    #
    #     optimizer = torch.optim.Adam(self.enc_belief.parameters(),lr=lr)
    #
    #     idx_list = [ m for m in range(len(self.train_data))]
    #     best_valid_loss = 1e10
    #     best_valid_iter = 0
    #     for i in range(num_iter):
    #         random.shuffle(idx_list)
    #         train_loss = 0
    #         for m in idx_list:
    #             optimizer.zero_grad()
    #             loss = self._loss_train_enc_belief(self.train_data[m])
    #             loss.backward()
    #             optimizer.step()
    #
    #         if 0==i%10:
    #             valid_loss = 0
    #             with torch.no_grad():
    #                 for _ in range(self.valid_ave_num):
    #                     for m2 in range(len(self.valid_data)):
    #                         valid_loss += self._loss_train_enc_belief(self.valid_data[m2]).item() / (len(self.valid_data)*self.valid_ave_num)
    #
    #             if best_valid_loss>valid_loss:
    #                 best_valid_loss = valid_loss
    #                 best_valid_iter = i
    #
    #             print("train_enc_belief: iter",i," train_loss",train_loss," valid_loss",valid_loss," best_valid_loss",best_valid_loss)
    #
    #         if (i-best_valid_iter)>=self.early_stopping_num:
    #             break

    #
    # def approx_log_mixture_of_gaussian_belief(self, z):
    #     idx_list = [m for m in range(len(self.train_mulogvar_list_for_mixture_of_gaussian_belief))]
    #     random.shuffle(idx_list)
    #     # idx_list = idx_list[:20]
    #     log_p_z_by_M_array = []
    #     for m in idx_list:
    #          log_p_z_by_M_array.append(log_gaussian(z.reshape(-1,self.z_dim),
    #                                                 self.train_mulogvar_list_for_mixture_of_gaussian_belief[m][:self.z_dim].reshape(-1,self.z_dim),
    #                                                 self.train_mulogvar_list_for_mixture_of_gaussian_belief[m][self.z_dim:].reshape(-1,self.z_dim)
    #                                                 ).reshape(1,-1) / len(idx_list)
    #                                   )
    #     log_p_z_by_M_array = torch.cat(log_p_z_by_M_array)
    #     return logsumexp(log_p_z_by_M_array, dim=0)



    # def _loss_train_enc_belief(self, temp_data_m):
    #     z_mulogvar = self.enc_belief(temp_data_m[:, :(self.sas_dim)])
    #     z = self.sample_z(z_mulogvar, len(temp_data_m))
    #     saz = torch.cat([temp_data_m[:, :(self.sa_dim)], z], dim=1)
    #     ds_mulogvar = self.dec(saz)
    #     ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]
    #
    #     loss = 0
    #     # Approximate of E_{z~q}[ - log p(y|x,z) ]
    #     loss += - log_gaussian(ds_m, # y
    #                            ds_mulogvar[:, :self.s_dim], # mu
    #                            ds_mulogvar[:, self.s_dim:] # logvar
    #                            ).sum()
    #
    #     # Approximate of E_{z~q}[ log q(z)]
    #     log_q_z = log_gaussian(z, # y
    #                            z_mulogvar[:self.z_dim], # mu
    #                            z_mulogvar[self.z_dim:] # logvar
    #                            ).sum()
    #     # Approximate of E_{z~q}[ log p(z)]
    #     log_p_z = self.approx_log_mixture_of_gaussian_belief(z).sum()
    #
    #     # nu * E_{z~q}[ log q(z) - log p(z) ]
    #     loss += self.nu * (log_q_z-log_p_z)
    #
    #     return loss

    #
    # def train_initial_belief(self, num_iter, lr):
    #
    #     self.initial_belief = torch.nn.Parameter(torch.hstack([self.train_belief_mu_list[0],
    #                                                            self.train_belief_logvar_list[0]]).flatten())
    #
    #
    #     optimizer = torch.optim.Adam([self.initial_belief],lr=lr)
    #
    #     idx_list = [ m for m in range(len(self.train_data))]
    #     best_valid_loss = 1e10
    #     best_valid_iter = 0
    #     for i in range(num_iter):
    #         random.shuffle(idx_list)
    #         print("self.valid_belief_mu_list",self.valid_belief_mu_list)
    #         print("self.initial_belief",self.initial_belief)
    #
    #         optimizer.zero_grad()
    #         z = self.sample_z(self.initial_belief, 256))
    #         # Approximate of E_{z~q}[ log q(z)]
    #         log_q_z = log_gaussian(eps*std+z_mulogvar[:self.z_dim], # y
    #                                z_mulogvar[:self.z_dim], # mu
    #                                z_mulogvar[self.z_dim:] # logvar
    #                                ).sum()
    #         # Approximate of E_{z~q}[ log p(z)]
    #         log_p_z = self.approx_log_mixture_of_gaussian_belief(eps*std+z_mulogvar[:self.z_dim])
    #         loss =
    #
    #
    #         for m in idx_list:
    #             z_mulogvar = self.enc_belief(self.train_data[m][:, :(self.sas_dim)])
    #
    #             # reparametrization trick
    #             eps = torch.randn_like(z_mulogvar[:self.z_dim])
    #             std = torch.exp(0.5 * z_mulogvar[self.z_dim:])
    #             loss = kdl_var_approx(self.initial_belief[:self.z_dim],
    #                                   self.initial_belief[self.z_dim:],
    #                                   self.train_belief_mu_list,
    #                                   self.train_belief_logvar_list)
    #             # loss = kld(self.initial_belief[:self.z_dim],
    #             #            self.initial_belief[self.z_dim:],
    #             #            self.prior[:self.z_dim],
    #             #            self.prior[self.z_dim:])
    #             loss.backward()
    #             optimizer.step()
    #
    #         if 0==i%10:
    #             valid_loss = 0
    #             with torch.no_grad():
    #                 for _ in range(self.valid_ave_num):
    #                     for m in range(len(self.valid_data)):
    #                         valid_loss += kdl_var_approx(self.initial_belief[:self.z_dim],
    #                                               self.initial_belief[self.z_dim:],
    #                                               self.valid_belief_mu_list,
    #                                               self.valid_belief_logvar_list).item() / (len(self.valid_data)*self.valid_ave_num)
    #                         # valid_loss += kld(self.initial_belief[:self.z_dim],
    #                         #            self.initial_belief[self.z_dim:],
    #                         #            self.prior[:self.z_dim],
    #                         #            self.prior[self.z_dim:]).item() / (len(self.valid_data)*self.valid_ave_num)
    #
    #             if best_valid_loss>valid_loss:
    #                 best_valid_loss = valid_loss
    #                 best_valid_iter = i
    #
    #             print("train_initial_belief: iter",i," train_loss",train_loss," valid_loss",valid_loss," best_valid_loss",best_valid_loss)
    #
    #         if (i-best_valid_iter)>=self.early_stopping_num:
    #             break

    # def reset(self, task_index=None, z=None, fix_init=False):
    #     self.sim_timestep=0
    #     if task_index is None:
    #         self.sim_m = np.random.randint(len(self.train_data))
    #     else:
    #         self.sim_m = task_index
    #     if z is None:
    #         # eps = torch.randn_like(self.train_belief_mu_list[self.sim_m])
    #         # std = torch.exp(0.5 * self.train_belief_logvar_list[self.sim_m])
    #         # self.sim_z = (eps*std+self.train_belief_mu_list[self.sim_m]).detach().flatten()
    #         std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
    #         eps = torch.randn_like(std)
    #         self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
    #     else:
    #         self.sim_z = torch_from_numpy(z.flatten())
    #     self.sim_s =  torch_from_numpy(self.init_state_fn(fix_init=fix_init).flatten())
    #     self.online_data = torch.empty((0,self.sas_dim+1))
    #     self.sim_b = self.get_belief(sads_array=None).detach().flatten()
    #     sb = torch.hstack([self.sim_s, self.sim_b])
    #     return sb


"""


    def compute_bamdpdata_for_sac(self):

        self.bamdapdada_for_sac = []
        for m in range(len(self.train_data)):
            temp_data_m = self.train_data[m]
            b_next_data = []
            for n in range(len(temp_data_m)):
                b_next_data.append(self.get_belief(temp_data_m[:n+1, :(self.sas_dim)]))
            b_next_data = torch.vstack(b_next_data)
            b_data = torch.vstack([self.get_belief(), b_next_data[:-1]])

            # s_data = temp_data_m[:, :self.s_dim]
            # a_data = temp_data_m[:, self.s_dim : (self.sa_dim)]
            # s_next_data = temp_data_m[:, (self.sa_dim) : (self.sas_dim)]
            bamdpdata_m = torch.hstack([temp_data_m[:, :self.s_dim], # a
                                        b_data,
                                        temp_data_m[:, self.s_dim : (self.sa_dim)], # a
                                        temp_data_m[:, (self.sa_dim) : (self.sas_dim)], # s_next
                                        b_next_data,
                                        temp_data_m[:, (self.sas_dim):] # r, done
                                        ])
            self.bamdapdada_for_sac.append(bamdpdata_m)
        self.bamdapdada_for_sac = torch.cat(self.bamdapdada_for_sac).numpy()


    def train_penalty(self, num_iter, lr):

        optimizer = torch.optim.Adam(self.penalty_model.parameters(),lr=lr)

        idx_list = [ m for m in range(len(self.train_data))]
        best_valid_loss = 1e10
        best_valid_iter = 0
        for i in range(num_iter):
            random.shuffle(idx_list)
            train_loss = 0
            for m in idx_list:
                temp_data_m = self.train_data[m]
                ds_m = self.train_ds_m_list[m]
                eps = torch.randn(temp_data_m.shape[0], self.z_dim)
                std = torch.exp(0.5*self.train_belief_logvar_list[m])
                z = eps*std.reshape(1,-1) + self.train_belief_mu_list[m]
                # sazg = torch.hstack([temp_data_m[:, :(self.sa_dim)], z, g_m_list[m]*torch.ones(temp_data_m.shape[0],g_m_list[m].shape[1])])
                saz = torch.hstack([temp_data_m[:, :(self.sa_dim)], z])
                with torch.no_grad():
                    ds_mulogvar = self.dec(saz)
                    pointwise_loss = gaussian_likelihood_pointwiseloss(ds_m, # y
                                                                       ds_mulogvar[:, :self.s_dim], # mu
                                                                       ds_mulogvar[:, self.s_dim:]) # logvar

                optimizer.zero_grad()
                pred = self.penalty_model(torch.hstack([saz, self.train_g_m_list[m].reshape(1,self.g_dim)*torch.ones(saz.shape[0], self.g_dim)]))
                loss = ((pred.flatten()-pointwise_loss)**2).sum()
                loss.backward()
                optimizer.step()

            if 0==i%10:
                valid_loss = 0
                with torch.no_grad():
                    for _ in range(self.valid_ave_num):
                        for m in range(len(self.valid_data)):

                            temp_data_m = self.valid_data[m]
                            ds_m = self.valid_ds_m_list[m]
                            eps = torch.randn(temp_data_m.shape[0], self.z_dim)
                            std = torch.exp(0.5*self.valid_belief_logvar_list[m])
                            z = eps*std.reshape(1,-1) + self.valid_belief_mu_list[m]
                            # sazg = torch.hstack([temp_data_m[:, :(self.sa_dim)], z, g_m_list[m]*torch.ones(temp_data_m.shape[0],g_m_list[m].shape[1])])
                            saz = torch.hstack([temp_data_m[:, :(self.sa_dim)], z])
                            ds_mulogvar = self.dec(saz)
                            pointwise_loss = gaussian_likelihood_pointwiseloss(ds_m, # y
                                                                               ds_mulogvar[:, :self.s_dim], # mu
                                                                               ds_mulogvar[:, self.s_dim:]) # logvar

                            pred = self.penalty_model(torch.hstack([saz, self.valid_g_m_list[m].reshape(1,self.g_dim)*torch.ones(saz.shape[0], self.g_dim)]))
                            valid_loss += ((pred.flatten()-pointwise_loss)**2).sum().item() / (len(self.valid_data)*self.valid_ave_num)

                if best_valid_loss>valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_iter = i

                print("train_penalty: iter",i," train_loss",train_loss," valid_loss",valid_loss," best_valid_loss",best_valid_loss)

            if (i-best_valid_iter)>=self.early_stopping_num:
                break






    def save_model(self, model_path="data_model_uwvi"):
        torch.save(self.state_dict(), model_path)


    def load_model(self, model_path="data_model_uwvi"):
        # for key, val in self.state_dict().items():
        #     print("key:",key,"\nval:\n", val)
        self.load_state_dict(torch.load(model_path))
        self._update_g_m_list()
        self._update_belief_list()
        self.compute_bamdpdata_for_sac()
        print("uwvi: load",model_path)




    def close(self):
        pass
"""
