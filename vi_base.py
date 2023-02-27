import numpy as np
import torch
import random
import copy
import time

from utils import log_gaussian, kld, kdl_var_approx, torch_from_numpy
from model_bamdp import Encoder, Decoder#, PenaltyModel

device = torch.device('cpu')

class baseVI:
    def __init__(self, args_init_dict):

        self.offline_data = copy.deepcopy(args_init_dict["offline_data"]) # [M, N , |SAS'R|] : M ... num of MDPs, N ... trajectory length, |SAS'R| ... dim of (s,a,s',r)
        s_dim = args_init_dict["s_dim"]
        a_dim = args_init_dict["a_dim"]
        z_dim = args_init_dict["z_dim"]
        env = args_init_dict["env"]
        self.mdp_policy = args_init_dict["policy"]
        debug_info = args_init_dict["debug_info"]

        train_valid_ratio = 0.2
        self.valid_ave_num=1 # validlossを計算するためのサンプル数

        self.nu = 1e0 # KLDによる正則化の重み

        self.gamma = 0.98
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.sa_dim = s_dim + a_dim
        self.sas_dim = 2*s_dim + a_dim
        self.z_dim = z_dim
        self.init_state_fn      = env.reset
        self.rew_fn             = env.env.env.rew_fn
        self._max_episode_steps = env.spec.max_episode_steps
        self.action_space       = env.action_space
        self.observation_space       = env.observation_space


        self.enc = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)
        self.dec = Decoder(self.s_dim, self.a_dim, self.z_dim)         # p(ds|s,a,z)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=False)  # [mean, logvar] for VAE training
        # self.enc_belief = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)

        # self.lam=1e-4 # ペナルティの係数？
        # self.penalty_model = PenaltyModel(s_dim, a_dim, z_dim) # ibisには要らない
        # self.train_g_m_list=None
        # self.valid_g_m_list=None
        self.initial_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)
        self.temp_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)


        self.validdata_num = int(train_valid_ratio*len(self.offline_data))
        for m in range(len(self.offline_data)):
            self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] = self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] - self.offline_data[m][:, :(self.s_dim)] # ds = s'-s


        self.mulogvar_offlinedata=None


        # only used for debug
        if debug_info is not None:
            self.debug_realenv = env
            self.debug_c_list = args_init_dict["debug_info"][:,1]
            self.debug_realenv_rolloutdata = [None]*len(self.offline_data)

    # def reset_encdec(self):
    #     self.enc = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)
    #     self.dec = Decoder(self.s_dim, self.a_dim, self.z_dim)         # p(ds|s,a,z)
    #     self.enc_belief = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)


    def store_encdec(self):
        self.enc_store = copy.deepcopy(self.enc)         # q(z|D^train_m)
        self.dec_store = copy.deepcopy(self.dec)         # p(ds|s,a,z)
        # self.enc_belief_store = copy.deepcopy(self.enc_belief)         # q(z|D^train_m)


    def restore_encdec(self):
        self.enc = copy.deepcopy(self.enc_store)         # q(z|D^train_m)
        self.dec = copy.deepcopy(self.dec_store)         # p(ds|s,a,z)
        self.dec.my_np_compile()
        # self.enc_belief = copy.deepcopy(self.enc_belief_store)         # q(z|D^train_m)


    def save(self, ckpt_name="vi_base_ckpt"):
        torch.save({'enc_state_dict': self.enc.state_dict(),
                    'dec_state_dict': self.dec.state_dict(),
                    'prior': self.prior,
                    'initial_belief': self.initial_belief
                    # 'enc_belief_state_dict': self.enc_belief.state_dict()
                   },ckpt_name)

    def load(self, ckpt_name="vi_base_ckpt"):
        checkpoint = torch.load(ckpt_name)
        self.enc.load_state_dict(checkpoint['enc_state_dict'])
        self.dec.load_state_dict(checkpoint['dec_state_dict'])
        self.prior = checkpoint['prior']
        # self.enc_belief.load_state_dict(checkpoint['enc_belief_state_dict'])
        self.initial_belief = checkpoint['initial_belief']

        if self.offline_data is not None:
            self.update_mulogvar_offlinedata()
        print("load", ckpt_name)
        self.dec.my_np_compile()


    def reset(self, z=None, fix_init=False):
        self.sim_timestep=0
        if z is None:
            std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
            eps = torch.randn_like(std)
            self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
            print("self.initial_belief",self.initial_belief.data, "self.sim_z",self.sim_z)
        else:
            self.sim_z = z.flatten()
        self.sim_s = self.init_state_fn(fix_init=fix_init).flatten()
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().numpy().flatten()
        self.temp_belief = copy.deepcopy(self.initial_belief)
        # self.temp_belief = copy.deepcopy(self.initial_belief)
        sb =np.hstack([self.sim_s, self.sim_b])
        return sb



    def step(self, a, update_belief=True, penalty_flag=True):
        a= a.flatten()
        saz = np.hstack([self.sim_s, a, self.sim_z]).reshape(1,-1)
        ds_mulogvar = self.dec.my_np_forward(saz).flatten()
        ds_mu = ds_mulogvar[:self.s_dim]
        eps = np.random.randn(len(ds_mu)) #* 0. # デバッグ：確定的システムにするなら0をかける
        std = np.exp(0.5 * ds_mulogvar[self.s_dim:])
        ds = (eps*std+ds_mu)
        rew = self.rew_fn(self.sim_s, a)

        self.sim_s = self.sim_s + ds
        done = False
        if self.sim_timestep>=(self._max_episode_steps-1):
            done=True
        if np.abs(self.sim_s).max()>100:
            print("predict diverge",self.sim_s)
            done = True
        self.sim_timestep+=1

        # if penalty_flag:
        #     with torch.no_grad():
        #         penalty = self.penalty_model(torch.hstack([saz, self.train_g_m_list[self.sim_m]]))
        #     rew -= self.lam * penalty.flatten()[0]
        current_data = torch_from_numpy(np.hstack([self.sim_s, a, ds, rew]))
        self.online_data = torch.vstack([self.online_data, current_data])
        if update_belief:
            self.sim_b = self.get_belief(self.online_data[:, :(self.sas_dim)]).detach().flatten()
        sb = np.hstack([self.sim_s, self.sim_b])
        return sb, rew, done, {}

    def get_belief(self, sads_array=None):
        if sads_array is None or len(sads_array)==0:
            return 1. * self.initial_belief.detach()
        else:
            sads_array = torch_from_numpy(sads_array)
            # self.temp_belief = torch.nn.Parameter(torch.hstack([self.temp_belief[:self.z_dim], self.mulogvar_offlinedata.mean(axis=0)[self.z_dim:]]), requires_grad=True)
            # self.temp_belief = torch.nn.Parameter(torch.hstack([self.initial_belief[:self.z_dim], self.mulogvar_offlinedata.mean(axis=0)[self.z_dim:]]), requires_grad=True)
            self.temp_belief = torch.nn.Parameter(torch.hstack([self.sim_z, self.mulogvar_offlinedata.mean(axis=0)[self.z_dim:]]), requires_grad=True)
            
            for _ in range(1):
                optimizer = torch.optim.Adam([self.temp_belief], lr=1e-3)
                # optimizer = torch.optim.SGD([self.temp_belief],lr=0.01)
                best_loss=1e10
                best_iter = 0
                start_time = time.time()
                for i in range(10000):

                    optimizer.zero_grad()

                    # maximum likelihood
                    tmp_sads_array = 1. * sads_array #[np.random.randint(0,len(sads_array),int(0.8*len(sads_array))),:]
                    z = self.temp_belief[:self.z_dim] * torch.ones(len(tmp_sads_array), self.z_dim)
                    saz = torch.cat([tmp_sads_array[:, :(self.sa_dim)], z], dim=1)
                    ds_mulogvar = self.dec(saz)
                    ds = tmp_sads_array[:, (self.sa_dim):(self.sas_dim)]
                    loss = - log_gaussian(ds, # y
                               ds_mulogvar[:, :self.s_dim], # mu
                               ds_mulogvar[:, self.s_dim:] # logvar
                               ).sum() 

                    # variational bayes
                    # z = self.sample_z(self.temp_belief, 1).flatten() * torch.ones(len(sads_array), self.z_dim)
                    # saz = torch.cat([sads_array[:, :(self.sa_dim)], z], dim=1)
                    # ds_mulogvar = self.dec(saz)
                    # ds = sads_array[:, (self.sa_dim):(self.sas_dim)]
                    # loss = - log_gaussian(ds, # y
                    #         ds_mulogvar[:, :self.s_dim], # mu
                    #         ds_mulogvar[:, self.s_dim:] # logvar
                    #         ).sum() 
                    # loss +=  kld(self.temp_belief[:self.z_dim],
                    #             self.temp_belief[self.z_dim:],
                    #             self.initial_belief.detach()[:self.z_dim],
                    #             self.initial_belief.detach()[self.z_dim:])

                    if loss.item()<best_loss:
                        best_loss = loss.item()
                        best_iter = 1*i
                        best_temp_belief = copy.deepcopy(self.temp_belief)
                    #     print("a", i, self.temp_belief.data)
                    # else:
                    #     print("b", i, self.temp_belief.data)
                    if (i-best_iter)>100:
                        break
                    loss.backward()
                    # self.temp_belief.grad += torch.randn_like(self.temp_belief.grad) * 0.1 * (torch.max(self.mulogvar_offlinedata, axis=0)[0] - torch.min(self.mulogvar_offlinedata, axis=0)[0])
                    optimizer.step()
                self.temp_belief = copy.deepcopy(best_temp_belief)
                print("get_belief: ", self.temp_belief.data.numpy(),"iter",i,"len",len(sads_array),"compute_time {:.3g}".format(time.time()-start_time),"best_loss {:.3g}".format(best_loss),"loss.item() {:.3g}".format(loss.item()))
            return 1*self.temp_belief.detach()            


    
    # def rollout_oneepisode_realenv(self, temp_c):
    def rollout_mdppolicy_oneepisode_realenv(self, temp_c):
        state = self.debug_realenv.reset(fix_init=True)
        done = False
        stateaction_history = []
        self.debug_realenv.env.env.set_params(c=temp_c)
        while not done:
            with torch.no_grad():
                action = self.mdp_policy(state, evaluate=self.mdp_policy_evaluate)  # Sample action from policy
            stateaction_history.append(np.hstack([state.flatten(), action.flatten()]))
            next_state, reward, done, _ = self.debug_realenv.step(action) # Step
            state = 1 * next_state
        return np.array(stateaction_history)

    
    # def rollout_oneepisode_simenv(self, z=None, random_stop=True, update_belief=False):
    def rollout_mdppolicy_oneepisode_simenv(self, z=None, random_stop=True, update_belief=False):
        sb = self.reset(fix_init=True, z=z)
        state = sb[:self.s_dim]
        done = False
        stateaction_history = []
        while True:
            if np.abs(state).max()>1e3:
                break
            with torch.no_grad():
                action = self.mdp_policy(state, evaluate=self.mdp_policy_evaluate)
            stateaction_history.append(np.hstack([state.flatten(), action.flatten(), z]))
            next_sb, reward, done, _ = self.step(action, update_belief=update_belief)
            state = next_sb[:self.s_dim]
            if random_stop:
                if np.random.rand()>self.gamma:
                    break
            else:
                if done:
                    break
        return np.array(stateaction_history)

    def get_sim_rollout_mdppolicy_data_fixlen(self, update_belief=False):
        self.dec.my_np_compile()
        self.mdp_policy_evaluate=True
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            z = 1. * self.mulogvar_offlinedata[m][:self.z_dim]
            # print("debug print",m,z)
            self.simenv_rolloutdata[m] = self.rollout_mdppolicy_oneepisode_simenv(z=z, random_stop=False, update_belief=update_belief)
        print(" ")


    # def get_sim_rollout_data_randomlen(self, update_belief=False):
    #     self.dec.my_np_compile()
    #     self.mdp_policy_evaluate=False
    #     self.simenv_rolloutdata = [None]*len(self.offline_data)
    #     for m in range(len(self.offline_data)):
    #         print(m," ", end="")
    #         self.simenv_rolloutdata[m] = self.rollout_episode_simenv(self.mulogvar_offlinedata[m], len_data=200, random_stop=True, zmean=False, update_belief=False)
    #     print(" ")

        

    def get_real_rollout_mdppolicy_data(self):
        self.mdp_policy_evaluate= True
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            self.debug_realenv_rolloutdata[m] = self.rollout_mdppolicy_oneepisode_realenv(self.debug_c_list[m])
        print(" ")


    def train_unweighted_vae(self, num_iter, lr, early_stop_step, flag=1):
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
        loss_fn = self._loss_train_unweighted_vae
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)
        self.restore_encdec()
        self.update_mulogvar_offlinedata()
        return ret


    def _train(self, num_iter, lr, early_stop_step, loss_fn, param_list):

        optimizer = torch.optim.Adam(param_list, lr=lr)

        total_idx_list = np.array( range(len(self.offline_data)) )
        train_idx_list = copy.deepcopy(total_idx_list)[self.validdata_num:]
        valid_idx_list = copy.deepcopy(total_idx_list)[:self.validdata_num]
        best_valid_loss = 1e10
        best_valid_iter = 0

        train_curve = []
        valid_curve = []
        for i in range(num_iter):

            with torch.no_grad():
                valid_loss_list = []
                for _ in range(self.valid_ave_num):
                    temp_valid_loss = 0
                    for m in valid_idx_list:
                        temp_valid_loss += loss_fn(m).item()
                    temp_valid_loss /= len(valid_idx_list)
                    valid_loss_list.append(temp_valid_loss)
                valid_loss_list = np.array(valid_loss_list)
                valid_loss = valid_loss_list.mean()

            if best_valid_loss>=valid_loss:
                best_valid_loss = valid_loss
                best_valid_iter = i
                self.store_encdec()

            if (i-best_valid_iter)>early_stop_step:
                break


            random.shuffle(train_idx_list)
            train_loss = 0
            for m in train_idx_list:
                optimizer.zero_grad()
                loss = loss_fn(m)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_idx_list)

            print("train: iter",i,
                  " trainloss {:.5f}".format(train_loss),
                  " validloss {:.5f}".format(valid_loss)+"±{:.5f}".format(valid_loss_list.std()),
                  " bestvalidloss {:.5f}".format(best_valid_loss),
                  " last_update", i-best_valid_iter)
            train_curve.append(train_loss)
            valid_curve.append(valid_loss)
        

        print("train: fin")
        return train_curve, valid_curve


    def sample_z(self, z_mulogvar, datanum):
        # # reparametrization trick type A
        std = torch.exp(0.5 * z_mulogvar[self.z_dim:])
        eps = torch.randn(self.z_dim)
        z = (eps*std+z_mulogvar[:self.z_dim]) * torch.ones(datanum, self.z_dim)

        # reparametrization trick type B
        # std = torch.exp(0.5 * z_mulogvar[self.z_dim:]) * torch.ones(datanum, self.z_dim)
        # eps = torch.randn(datanum, self.z_dim)
        # z = eps * std + z_mulogvar[:self.z_dim]
        return z


    def _loss_train_unweighted_vae(self, m, flag=False):
        temp_data_m = self.offline_data[m]
        z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
        z = self.sample_z(z_mulogvar, 1).flatten() * torch.ones(len(temp_data_m), self.z_dim)

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

        return loss


    def update_mulogvar_offlinedata(self):
        with torch.no_grad():
            self.mulogvar_offlinedata = []
            for m in range(len(self.offline_data)):
                temp_data_m = self.offline_data[m]
                z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
                self.mulogvar_offlinedata.append(z_mulogvar)
            self.mulogvar_offlinedata = torch.vstack(self.mulogvar_offlinedata)
            



    def _loss_train_initial_belief(self, m, flag=False):
        tmp_z= self.sample_z(self.mulogvar_offlinedata[m], 1)
        return - log_gaussian(tmp_z, # y
                                self.initial_belief[:self.z_dim], # mu
                                self.initial_belief[self.z_dim:] # logvar
                                ).sum()

    def train_initial_belief(self, num_iter, lr, early_stop_step):
        
        param_list = [self.initial_belief]
        loss_fn = self._loss_train_initial_belief
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)
        # self.enc_belief = copy.deepcopy(self.enc)
        return ret
