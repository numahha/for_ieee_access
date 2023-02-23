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

        offline_data = args_init_dict["offline_data"]
        s_dim = args_init_dict["s_dim"]
        a_dim = args_init_dict["a_dim"]
        z_dim = args_init_dict["z_dim"]
        env = args_init_dict["env"]
        self.policy = args_init_dict["policy"]

        train_valid_ratio = 0.2
        self.validdata_num = int(train_valid_ratio*len(offline_data))
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

        self.offline_data = copy.deepcopy(offline_data) # [M, N , |SAS'R|] : M ... num of MDPs, N ... trajectory length, |SAS'R| ... dim of (s,a,s',r)

        for m in range(len(self.offline_data)):
            self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] = self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] - self.offline_data[m][:, :(self.s_dim)] # ds = s'-s

        self.enc = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)
        self.dec = Decoder(self.s_dim, self.a_dim, self.z_dim)         # p(ds|s,a,z)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=False)  # [mean, logvar] for VAE training
        self.enc_belief = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)

        self.mulogvar_list_for_mixture_of_gaussian_belief=None

        # self.lam=1e-4 # ペナルティの係数？
        # self.penalty_model = PenaltyModel(s_dim, a_dim, z_dim) # ibisには要らない
        # self.train_g_m_list=None
        # self.valid_g_m_list=None
        self.initial_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)
        self.temp_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)



        # only used for debug
        self.debug_realenv = env
        self.debug_c_list = args_init_dict["debug_info"][:,1]
        self.debug_realenv_rolloutdata = [None]*len(offline_data)

    def reset_encdec(self):
        self.enc = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)
        self.dec = Decoder(self.s_dim, self.a_dim, self.z_dim)         # p(ds|s,a,z)
        self.enc_belief = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)


    def store_encdec(self):
        self.enc_store = copy.deepcopy(self.enc)         # q(z|D^train_m)
        self.dec_store = copy.deepcopy(self.dec)         # p(ds|s,a,z)
        self.enc_belief_store = copy.deepcopy(self.enc_belief)         # q(z|D^train_m)


    def restore_encdec(self):
        self.enc = copy.deepcopy(self.enc_store)         # q(z|D^train_m)
        self.dec = copy.deepcopy(self.dec_store)         # p(ds|s,a,z)
        self.enc_belief = copy.deepcopy(self.enc_belief_store)         # q(z|D^train_m)


    def save(self, ckpt_name="vi_base_ckpt"):
        torch.save({'enc_state_dict': self.enc.state_dict(),
                    'dec_state_dict': self.dec.state_dict(),
                    'prior': self.prior,
                    'enc_belief_state_dict': self.enc_belief.state_dict()
                   },ckpt_name)

    def load(self, ckpt_name="vi_base_ckpt"):
        checkpoint = torch.load(ckpt_name)
        self.enc.load_state_dict(checkpoint['enc_state_dict'])
        self.dec.load_state_dict(checkpoint['dec_state_dict'])
        self.prior = checkpoint['prior']
        self.enc_belief.load_state_dict(checkpoint['enc_belief_state_dict'])
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
        print("load", ckpt_name)


    def reset(self, z=None, fix_init=False):
        self.sim_timestep=0
        if z is None:
            std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
            eps = torch.randn_like(std)
            self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
        else:
            self.sim_z = z.flatten()
        self.sim_s = self.init_state_fn(fix_init=fix_init).flatten()
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().numpy().flatten()
        self.temp_belief = copy.deepcopy(self.initial_belief)
        sb =np.hstack([self.sim_s, self.sim_b])
        return sb



    def step(self, a, update_belief=False, penalty_flag=False):
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
        if np.abs(self.sim_s).max()>20:
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

    
    def rollout_oneepisode_realenv(self, temp_c):
        state = self.debug_realenv.reset(fix_init=True)
        done = False
        stateaction_history = []
        self.debug_realenv.env.env.set_params(c=temp_c)
        while not done:
            with torch.no_grad():
                action = self.policy(state, evaluate=self.policy_evaluate)  # Sample action from policy
            stateaction_history.append(np.hstack([state.flatten(), action.flatten()]))
            next_state, reward, done, _ = self.debug_realenv.step(action) # Step
            state = 1 * next_state
        return np.array(stateaction_history)

    
    def rollout_oneepisode_simenv(self, z=None, random_stop=True, update_belief=False):
        sb = self.reset(fix_init=True, z=z)
        state = sb[:self.s_dim]
        done = False
        stateaction_history = []
        while True:
            if np.abs(state).max()>1e3:
                break
            with torch.no_grad():
                action = self.policy(state, evaluate=self.policy_evaluate)
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

    def get_sim_rollout_data_fixlen(self, update_belief=False):
        self.dec.my_np_compile()
        self.policy_evaluate=True
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            z = 1. * self.mulogvar_list_for_mixture_of_gaussian_belief[m][:self.z_dim]
            # print("debug print",m,z)
            self.simenv_rolloutdata[m] = self.rollout_oneepisode_simenv(z=z, random_stop=False, update_belief=update_belief)
        print(" ")


    # def get_sim_rollout_data_randomlen(self, update_belief=False):
    #     self.dec.my_np_compile()
    #     self.policy_evaluate=False
    #     self.simenv_rolloutdata = [None]*len(self.offline_data)
    #     for m in range(len(self.offline_data)):
    #         print(m," ", end="")
    #         self.simenv_rolloutdata[m] = self.rollout_episode_simenv(self.mulogvar_list_for_mixture_of_gaussian_belief[m], len_data=200, random_stop=True, zmean=False, update_belief=False)
    #     print(" ")

        

    def get_real_rollout_data(self):
        self.policy_evaluate= True
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            self.debug_realenv_rolloutdata[m] = self.rollout_oneepisode_realenv(self.debug_c_list[m])
        print(" ")



    def get_belief(self, sads_array=None):
        if sads_array is None or len(sads_array)==0:
            return 1. * self.initial_belief.detach()
        else:
            sads_array = torch_from_numpy(sads_array)
#             with torch.no_grad():
#                 return 1. * self.enc_belief(sads_array).detach()
            optimizer = torch.optim.Adam([self.temp_belief], lr=5e-4)
            best_loss=1e10
            best_iter = 0
            start_time = time.time()
            for i in range(1000):
                optimizer.zero_grad()
                z = self.sample_z(self.temp_belief, 1).flatten() * torch.ones(len(sads_array), self.z_dim)
                saz = torch.cat([sads_array[:, :(self.sa_dim)], z], dim=1)
                ds_mulogvar = self.dec(saz)
                ds = sads_array[:, (self.sa_dim):(self.sas_dim)]
#                 print("hishi",ds[0], ds_mulogvar[0])
                loss = - log_gaussian(ds, # y
                           ds_mulogvar[:, :self.s_dim], # mu
                           ds_mulogvar[:, self.s_dim:] # logvar
                           ).sum() 
                loss +=  kld(self.temp_belief[:self.z_dim],
                             self.temp_belief[self.z_dim:],
                             self.initial_belief.detach()[:self.z_dim],
                             self.initial_belief.detach()[self.z_dim:])
                loss.backward()
                optimizer.step()
                if loss.item()<best_loss:
                    best_loss = loss.item()
                    best_iter = 1*i
                if (i-best_iter)>100:
                    break
            print("get_belief",i,"compute_time",time.time()-start_time)
            return 1*self.temp_belief.detach()            

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
        self.update_mulogvar_list_for_mixture_of_gaussian_belief()
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


    def update_mulogvar_list_for_mixture_of_gaussian_belief(self):
        with torch.no_grad():
            self.mulogvar_list_for_mixture_of_gaussian_belief = []
            for m in range(len(self.offline_data)):
                temp_data_m = self.offline_data[m]
                z_mulogvar = self.enc(temp_data_m[:, :(self.sas_dim)])
                self.mulogvar_list_for_mixture_of_gaussian_belief.append(z_mulogvar)



    def _loss_train_initial_belief(self, m, flag=False):
        tmp_z= self.sample_z(self.mulogvar_list_for_mixture_of_gaussian_belief[m], 1)
        return - log_gaussian(tmp_z, # y
                                self.initial_belief[:self.z_dim], # mu
                                self.initial_belief[self.z_dim:] # logvar
                                ).sum()

    def train_initial_belief(self, num_iter, lr, early_stop_step):
        
        param_list = [self.initial_belief]
        loss_fn = self._loss_train_initial_belief
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)
        self.enc_belief = copy.deepcopy(self.enc)
        return ret

#     def train_initial_belief(self, sads):
        
#         param_list = [self.initial_belief]
#         loss_fn = self._loss_train_initial_belief
#         ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list)     
#         return ret