import numpy as np
import torch
import random
import copy
import time

from utils import log_gaussian, kld, torch_from_numpy
from model_bamdp import Encoder, Decoder, PenaltyModel

device = torch.device('cpu')


class baseVI:
    def __init__(self, args_init_dict):

        self.offline_data = copy.deepcopy(args_init_dict["offline_data"]) # [M, N , |SAS'R|] : M ... num of MDPs, N ... trajectory length, |SAS'R| ... dim of (s,a,s',r)
        s_dim = args_init_dict["s_dim"]
        a_dim = args_init_dict["a_dim"]
        z_dim = args_init_dict["z_dim"]
        env = args_init_dict["env"]
        self.mdp_policy = args_init_dict["mdp_policy"]
        self.bamdp_policy = args_init_dict["bamdp_policy"]
        debug_info = args_init_dict["debug_info"]
        self.ckpt_suffix = args_init_dict["ckpt_suffix"]

        train_valid_ratio = 0.2
        self.valid_ave_num=1 # validlossを計算するためのサンプル数

        self.nu = 1e0 # KLDによる正則化の重み

        self.gamma = 0.99
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
        self.h_min_tilde=None
        self.kappa_tilde= 1
        self.c_coeff = 0.1
        self.update_belief=True
        self.penalty_flag=True

        self.abs_s_max = torch.zeros(self.s_dim)
        for m in range(len(self.offline_data)):
            for i in range(self.s_dim):
                self.abs_s_max[i] = torch.max(self.abs_s_max[i],torch.abs(self.offline_data[m][:,i]).max())
        self.abs_s_max = self.abs_s_max.numpy()

        self.enc = Encoder(self.s_dim, self.a_dim, self.z_dim)         # q(z|D^train_m)
        self.dec = Decoder(self.s_dim, self.a_dim, self.z_dim)         # p(ds|s,a,z)
        self.prior = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=False)  # [mean, logvar] for VAE training

        self.penalty_model = PenaltyModel(s_dim, a_dim, z_dim)
        self.initial_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)
        self.temp_belief = torch.nn.Parameter(torch.zeros(2*z_dim), requires_grad=True)  # [mean, logvar] for planning, a gaussian approximate of 1/M * sum_{m} q(z|D^train_m)


        self.validdata_num = int(train_valid_ratio*len(self.offline_data))
        for m in range(len(self.offline_data)):
            self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] = self.offline_data[m][:, (self.sa_dim):(self.sas_dim)] - self.offline_data[m][:, :(self.s_dim)] # ds = s'-s


        self.mulogvar_offlinedata=None
        self.mulogvar_startpoints=None        


        # only used for debug
        if debug_info is not None:
            self.debug_realenv = env
            self.debug_p_list = args_init_dict["debug_info"][:,:]
            self.debug_realenv_rolloutdata = [None]*len(self.offline_data)


    def tmp_store_encdec(self):
        self.tmp_enc_store = copy.deepcopy(self.enc)         # q(z|D^train_m)
        self.tmp_dec_store = copy.deepcopy(self.dec)         # p(ds|s,a,z)


    def tmp_restore_encdec(self):
        self.enc = copy.deepcopy(self.tmp_enc_store)         # q(z|D^train_m)
        self.dec = copy.deepcopy(self.tmp_dec_store)         # p(ds|s,a,z)
        self.dec.my_np_compile()


    def tmp2_store_encdec(self):
        self.tmp2_enc_store = copy.deepcopy(self.enc)         # q(z|D^train_m)
        self.tmp2_dec_store = copy.deepcopy(self.dec)         # p(ds|s,a,z)


    def tmp2_restore_encdec(self):
        self.enc = copy.deepcopy(self.tmp2_enc_store)         # q(z|D^train_m)
        self.dec = copy.deepcopy(self.tmp2_dec_store)         # p(ds|s,a,z)
        self.dec.my_np_compile()


    def tmp_store_penalty(self):
        self.tmp_penalty_store = copy.deepcopy(self.penalty_model)

    def tmp_restore_penalty(self):
        self.penalty_model = copy.deepcopy(self.tmp_penalty_store)


    def tmp_store_intialbelief(self):
        self.tmp_intialbelief_store = copy.deepcopy(self.initial_belief)

    def tmp_restore_intialbelief(self):
        self.initial_belief = copy.deepcopy(self.tmp_intialbelief_store)



    def save(self, ckpt_key="unweighted"):
        ckpt_name = "ckpt_basevi_"+self.ckpt_suffix+"_"+ckpt_key
        print("base save ckpt", ckpt_name)
        torch.save({'enc_state_dict': self.enc.state_dict(),
                    'dec_state_dict': self.dec.state_dict(),
                    'prior': self.prior,
                    'initial_belief': self.initial_belief,
                    'penalty_model_dict': self.penalty_model.state_dict(),
                   }, ckpt_name)
        print("base load self.initial_belief.data.sum()", self.initial_belief.data.sum())
        print("base load dec.state_dict()['net_phat.0.weight'].sum()",self.dec.state_dict()['net_phat.0.weight'].sum())


    def load(self, ckpt_key="unweighted"):
        ckpt_name = "ckpt_basevi_"+self.ckpt_suffix+"_"+ckpt_key
        print("base load ckpt", ckpt_name)
        checkpoint = torch.load(ckpt_name)
        self.enc.load_state_dict(checkpoint['enc_state_dict'])
        self.dec.load_state_dict(checkpoint['dec_state_dict'])
        self.prior = checkpoint['prior']
        self.initial_belief = checkpoint['initial_belief']
        self.penalty_model.load_state_dict(checkpoint['penalty_model_dict'])

        print("base load self.initial_belief.data.sum()", self.initial_belief.data.sum())
        print("base load dec.state_dict()['net_phat.0.weight'].sum()",self.dec.state_dict()['net_phat.0.weight'].sum())
        self.update_mulogvar_offlinedata()
        self.set_h_min_tilde()
        self.eval_loss_unweighted()
        self.dec.my_np_compile()


    def reset(self, z=None, fix_init=False):
        self.sim_timestep=0
        if z is None:
            std = torch.exp(0.5 * self.initial_belief[self.z_dim:])
            eps = torch.randn_like(std)
            self.sim_z = (eps*std+self.initial_belief[:self.z_dim]).detach().flatten()
            # print("self.initial_belief",self.initial_belief.data, "self.sim_z",self.sim_z)
        else:
            self.sim_z = z.flatten()
        self.sim_s = self.init_state_fn(fix_init=fix_init).flatten()
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().numpy().flatten()
        self.temp_belief = copy.deepcopy(self.initial_belief)
        # self.temp_belief = copy.deepcopy(self.initial_belief)
        sb =np.hstack([self.sim_s, self.sim_b])
        return sb


    def reset_po(self):
        self.sim_timestep=0
        m = np.random.randint(len(self.mulogvar_offlinedata))
        mulogvar = self.mulogvar_offlinedata[m]
        std = torch.exp(0.5 * mulogvar[self.z_dim:])
        eps = torch.randn_like(std)
        print(eps, std, mulogvar[:self.z_dim])
        self.sim_z = (eps*std+mulogvar[:self.z_dim]).detach().flatten()
        # print("mulogvar",mulogvar, "self.sim_z",self.sim_z)
        self.sim_s = self.init_state_fn(fix_init=False).flatten()
        self.online_data = torch.empty((0,self.sas_dim+1))
        self.sim_b = self.get_belief(sads_array=None).detach().numpy().flatten()
        self.temp_belief = copy.deepcopy(self.initial_belief)
        # self.temp_belief = copy.deepcopy(self.initial_belief)
        sb =np.hstack([self.sim_s, self.sim_b])
        return sb


    def step(self, a):
        a= a.flatten()
        saz = np.hstack([self.sim_s, a, self.sim_z]).reshape(1,-1)
        ds_mulogvar = self.dec.my_np_forward(saz).flatten()
        ds_mu = ds_mulogvar[:self.s_dim]
        std = np.exp(0.5 * ds_mulogvar[self.s_dim:])
        eps = np.random.randn(len(ds_mu)) #* 0. # デバッグ：確定的システムにするなら0をかける
        ds = (eps*std+ds_mu)
        # print("ds",ds,"ds_mu",ds_mu,"std*eps",std*eps)
        rew = self.rew_fn(self.sim_s, a)

        current_data = torch_from_numpy(np.hstack([1*self.sim_s, a, ds, rew]))
        self.online_data = torch.vstack([self.online_data, current_data])

        self.sim_s = self.sim_s + ds
        done = False
        if self.sim_timestep>=(self._max_episode_steps-1):
            done=True

        s_limit = 2*self.abs_s_max
        if np.count_nonzero(np.abs(self.sim_s)>(2*self.abs_s_max))>0:
            print("predict diverge", self.sim_s, ds, s_limit, self.abs_s_max, "sim_timestep", self.sim_timestep)
            self.sim_s = np.clip(self.sim_s, -s_limit, s_limit)
            # print( self.sim_s)
            done = True
            rew -= self._max_episode_steps
        self.sim_timestep+=1

        if self.penalty_flag and (self.kappa_tilde is not None):
            with torch.no_grad():
                sazmulogvar = torch.cat([torch_from_numpy(saz), torch_from_numpy(self.sim_b)*torch.ones((len(saz), 2*self.z_dim))], dim=1)
                tmp_penalty = self.penalty_model(sazmulogvar).numpy().flatten()[0]
            # print("self.kappa_tilde", self.kappa_tilde)
            rew -= self.kappa_tilde * (tmp_penalty - self.h_min_tilde)
        if self.update_belief:
            self.sim_b = self.get_belief(self.online_data[:, :(self.sas_dim)]).detach().flatten()
        sb = np.hstack([self.sim_s, self.sim_b])
        return sb, rew, done, {}


    def get_belief(self, sads_array=None):
        if sads_array is None or len(sads_array)==0:
            return 1. * self.initial_belief.detach()
        else:
            sads_array = torch_from_numpy(sads_array)

            # find good start point
            self.mulogvar_startpoints[-2] = 1. * self.initial_belief.detach()
            best_initial_loss = np.inf
            best_initial_index = 0

            # for m in range(len(self.mulogvar_startpoints)-2):
            for m in range(len(self.mulogvar_startpoints)):
                tmp_mulogvar = self.mulogvar_startpoints[m]
                with torch.no_grad():
                    z = tmp_mulogvar[:self.z_dim]* torch.ones(len(sads_array), self.z_dim)
                    saz = torch.cat([sads_array[:, :(self.sa_dim)], z], dim=1)
                    ds_mulogvar = self.dec(saz)
                    ds = sads_array[:, (self.sa_dim):(self.sas_dim)]
                    loss = - log_gaussian(ds, # y
                            ds_mulogvar[:, :self.s_dim], # mu
                            ds_mulogvar[:, self.s_dim:] # logvar
                            ).sum() 
                    loss +=  kld(tmp_mulogvar[:self.z_dim],
                                 tmp_mulogvar[self.z_dim:],
                                 self.initial_belief.detach()[:self.z_dim],
                                 self.initial_belief.detach()[self.z_dim:])
                if (best_initial_loss>loss.item()):
                    best_initial_loss = 1 * loss.item()
                    best_initial_index = 1 * m

            self.temp_belief = torch.nn.Parameter(self.mulogvar_startpoints[best_initial_index], requires_grad=True)
            best_temp_belief = copy.deepcopy(self.temp_belief)
            
            for _ in range(1):
                optimizer = torch.optim.Adam([self.temp_belief], lr=2e-3)
                best_loss=np.inf
                best_iter = 0
                start_time = time.time()
                for i in range(5000):

                    optimizer.zero_grad()

                    z = self.sample_z(self.temp_belief, 1).flatten() * torch.ones(len(sads_array), self.z_dim)
                    saz = torch.cat([sads_array[:, :(self.sa_dim)], z], dim=1)
                    ds_mulogvar = self.dec(saz)
                    ds = sads_array[:, (self.sa_dim):(self.sas_dim)]
                    loss = - log_gaussian(ds, # y
                            ds_mulogvar[:, :self.s_dim], # mu
                            ds_mulogvar[:, self.s_dim:] # logvar
                            ).sum() 
                    loss +=  kld(self.temp_belief[:self.z_dim],
                                self.temp_belief[self.z_dim:],
                                self.initial_belief.detach()[:self.z_dim],
                                self.initial_belief.detach()[self.z_dim:])

                    if loss.item()<best_loss:
                        best_loss = loss.item()
                        best_iter = 1*i
                        best_temp_belief = copy.deepcopy(self.temp_belief)
                    # if (i-best_iter)>50:
                    if (i-best_iter)>5:
                        break
                    loss.backward()
                    optimizer.step()
                self.temp_belief = copy.deepcopy(best_temp_belief)
                # print("get_belief: ", self.temp_belief.data.numpy(),"iter",i,"len",len(sads_array),"compute_time {:.3g}".format(time.time()-start_time),"best_loss {:.3g}".format(best_loss),"loss.item() {:.3g}".format(loss.item()),end="       \r")
            self.mulogvar_startpoints[-1] = 1*self.temp_belief.detach()
            return 1*self.temp_belief.detach()


    def get_nll(self, sads_array, z):
        with torch.no_grad():
            sads_array = torch_from_numpy(sads_array)
            saz = torch.cat([sads_array[:, :(self.sa_dim)], z*torch.ones(len(sads_array), self.z_dim)], dim=1)
            ds_mulogvar = self.dec(saz)
            ds = sads_array[:, (self.sa_dim):(self.sas_dim)]
            loss = - log_gaussian(ds, # y
                    ds_mulogvar[:, :self.s_dim], # mu
                    ds_mulogvar[:, self.s_dim:] # logvar
                    ).sum() 
        return loss.item()

    
    def rollout_mdppolicy_oneepisode_realenv(self, temp_p):
        state = self.debug_realenv.reset(fix_init=True)
        done = False
        stateaction_history = []
        self.debug_realenv.env.env.set_params(temp_p)
        self.update_belief=False
        self.penalty_flag=False
        while not done:
            with torch.no_grad():
                action = self.mdp_policy(state, evaluate=self.policy_evaluate)  # Sample action from policy
            stateaction_history.append(np.hstack([state.flatten(), action.flatten()]))
            next_state, reward, done, _ = self.debug_realenv.step(action) # Step
            state = 1 * next_state
        self.update_belief=True
        self.penalty_flag=True
        return np.array(stateaction_history), None


    def rollout_bamdppolicy_oneepisode_realenv(self, temp_p):
        state = self.debug_realenv.reset(fix_init=True)
        done = False
        sads_array=np.empty((0,self.s_dim*2+self.a_dim))
        belief = self.get_belief()
        stateaction_history = []
        self.debug_realenv.env.env.set_params(temp_p)
        self.update_belief=True
        self.penalty_flag=False
        while not done:
            aug_state = np.hstack([state, belief.numpy()])
            with torch.no_grad():
                action = self.bamdp_policy(aug_state, evaluate=self.policy_evaluate)  # Sample action from policy
            stateaction_history.append(np.hstack([state.flatten(), action.flatten()]))
            next_state, reward, done, _ = self.debug_realenv.step(action) # Step
            sads_array = np.vstack([sads_array, 
                                    np.hstack([state, action, next_state-state])])
            belief = self.get_belief(sads_array=sads_array)
            state = 1 * next_state
        self.update_belief=True
        self.penalty_flag=True
        return np.array(stateaction_history), None

    
    def rollout_mdppolicy_oneepisode_simenv(self, z=None, random_stop=True, fix_init=True):
        sb = self.reset(fix_init=fix_init, z=z)
        state = sb[:self.s_dim]
        done = False
        stateaction_history = []
        self.update_belief=False
        self.penalty_flag=False
        while True:
            with torch.no_grad():
                action = self.mdp_policy(state, evaluate=self.policy_evaluate)
            stateaction_history.append(np.hstack([state.flatten(), action.flatten(), z]))
            next_sb, reward, done, _ = self.step(action)
            state = next_sb[:self.s_dim]
            if done:
                break
            if random_stop:
                if np.random.rand()>self.gamma:
                    break
        self.update_belief=True
        self.penalty_flag=True
        return np.array(stateaction_history)
    

    def rollout_bamdppolicy_oneepisode_simenv(self, z=None, fix_init=True, random_stop=True):
        aug_state = self.reset(fix_init=fix_init, z=z)
        done = False
        stateaction_history = []
        self.update_belief=True
        self.penalty_flag=False
        while True:
            with torch.no_grad():
                action = self.bamdp_policy(aug_state, evaluate=self.policy_evaluate)
            stateaction_history.append(np.hstack([aug_state.flatten()[:self.s_dim], action.flatten(), z]))
            next_aug_state, reward, done, _ = self.step(action)
            aug_state = next_aug_state
            if done:
                break
            if random_stop:
                if np.random.rand()>self.gamma:
                    break
        self.update_belief=True
        self.penalty_flag=True
        return np.array(stateaction_history)


    def get_sim_rollout_mdppolicy_data_fixlen(self):
        self.dec.my_np_compile()
        self.policy_evaluate=True
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            z = 1. * self.mulogvar_offlinedata[m][:self.z_dim]
            # print("debug print",m,z)
            self.simenv_rolloutdata[m] = self.rollout_mdppolicy_oneepisode_simenv(z=z, random_stop=False, fix_init=True)
        print(" ")
        self.update_belief=True


    def get_sim_rollout_mdppolicy_data_randomstop(self):
        self.dec.my_np_compile()
        self.policy_evaluate=False
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            tmp_rolloutdata = None
            while 1:
                z = 1. * self.mulogvar_offlinedata[m][:self.z_dim]
                if tmp_rolloutdata is None:
                    tmp_rolloutdata = self.rollout_mdppolicy_oneepisode_simenv(z=z, random_stop=True)
                else:
                    tmp_rolloutdata = np.vstack([tmp_rolloutdata, self.rollout_mdppolicy_oneepisode_simenv(z=z, random_stop=True, fix_init=False)])
                if len(tmp_rolloutdata)>(len(self.offline_data[m])*2):
                    break
            idx = np.array(range(len(tmp_rolloutdata)))
            np.random.shuffle(idx)
            self.simenv_rolloutdata[m] = tmp_rolloutdata[ idx[:len(self.offline_data[m])] ]
        print(" ")


    def get_sim_rollout_bamdppolicy_data_fixlen(self):
        self.dec.my_np_compile()
        self.policy_evaluate=True
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        tmp_clock = time.time()
        for m in range(len(self.offline_data)):
            print("\n",m,time.time()-tmp_clock)
            tmp_clock = time.time()
            z = 1. * self.mulogvar_offlinedata[m][:self.z_dim]
            # print("debug print",m,z)
            self.simenv_rolloutdata[m] = self.rollout_bamdppolicy_oneepisode_simenv(z=z, random_stop=False, fix_init=True)
        print(" ")


    def get_sim_rollout_bamdppolicy_data_randomstop(self):
        self.dec.my_np_compile()
        self.policy_evaluate=False
        # self.policy_evaluate=True
        self.simenv_rolloutdata = [None]*len(self.offline_data)
        for m in range(len(self.offline_data)):
            print("\n",m)
            tmp_rolloutdata = None
            while 1:
                z = 1. * self.mulogvar_offlinedata[m][:self.z_dim]
                if tmp_rolloutdata is None:
                    tmp_rolloutdata = self.rollout_bamdppolicy_oneepisode_simenv(z=z, random_stop=True)
                else:
                    tmp_rolloutdata = np.vstack([tmp_rolloutdata, self.rollout_bamdppolicy_oneepisode_simenv(z=z, random_stop=True, fix_init=False)])
                if len(tmp_rolloutdata)>(len(self.offline_data[m])*3):
                    break
            idx = np.array(range(len(tmp_rolloutdata)))
            np.random.shuffle(idx)
            self.simenv_rolloutdata[m] = tmp_rolloutdata[ idx[:len(self.offline_data[m])] ]
        print(" ")


    def get_real_rollout_mdppolicy_data(self):
        self.policy_evaluate= True
        for m in range(len(self.offline_data)):
            print(m," ", end="")
            self.debug_realenv_rolloutdata[m], _ = self.rollout_mdppolicy_oneepisode_realenv(self.debug_p_list[m])
        print(" ")


    def get_real_rollout_bamdppolicy_data(self):
        self.policy_evaluate= True
        tmp_clock = time.time()
        for m in range(len(self.offline_data)):
            print("\n",m,time.time()-tmp_clock)
            tmp_clock = time.time()
            self.debug_realenv_rolloutdata[m], _ = self.rollout_bamdppolicy_oneepisode_realenv(self.debug_p_list[m])
        print(" ")


    def train_unweighted_vae(self, num_iter, lr, early_stop_step, flag=1):
        if flag==1:
            print("train_vae: enc_dec")
            param_list = list(self.enc.parameters())+list(self.dec.parameters())
        # elif flag==2:
        #     print("train_vae: enc")
        #     param_list = list(self.enc.parameters())
        # elif flag==3:
        #     print("train_vae: dec")
        #     param_list = list(self.dec.parameters())
        else:
            return [], []
        loss_fn = self._loss_train_unweighted_vae
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list, self.tmp_store_encdec)
        self.tmp_restore_encdec()
        self.update_mulogvar_offlinedata()
        self.set_h_min_tilde()
        return ret


    def _train(self, num_iter, lr, early_stop_step, loss_fn, param_list, tmp_store_model=None):

        optimizer = torch.optim.Adam(param_list, lr=lr)

        total_idx_list = np.array( range(len(self.offline_data)) )
        train_idx_list = copy.deepcopy(total_idx_list)[self.validdata_num:]
        valid_idx_list = copy.deepcopy(total_idx_list)[:self.validdata_num]
        best_valid_loss = np.inf
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
                if tmp_store_model is not None:
                    tmp_store_model()


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
        # eps = torch.randn(self.z_dim)
        eps = torch.Tensor(np.random.randn(self.z_dim))
        z = (eps*std+z_mulogvar[:self.z_dim]) * torch.ones(datanum, self.z_dim)

        # reparametrization trick type B
        # std = torch.exp(0.5 * z_mulogvar[self.z_dim:]) * torch.ones(datanum, self.z_dim)
        # eps = torch.randn(datanum, self.z_dim)
        # z = eps * std + z_mulogvar[:self.z_dim]
        return z


    def _loss_train_unweighted_vae(self, m):
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
            self.mulogvar_startpoints = torch.vstack([self.mulogvar_offlinedata[::4], self.mulogvar_offlinedata[:2]])


    def _loss_train_initial_belief(self, m):
        tmp_z= self.sample_z(self.mulogvar_offlinedata[m], 1)
        return - log_gaussian(tmp_z, # y
                                self.initial_belief[:self.z_dim], # mu
                                self.initial_belief[self.z_dim:] # logvar
                                ).sum()


    def train_initial_belief(self, num_iter, lr, early_stop_step):
        
        param_list = [self.initial_belief]
        loss_fn = self._loss_train_initial_belief
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list, self.tmp_store_intialbelief)
        self.tmp_restore_intialbelief()
        return ret


    def set_h_min_tilde(self):
        self.h_min_tilde=np.inf
        target_max = -np.inf
        with torch.no_grad():
            for m in range(len(self.offline_data)):
                temp_data_m = self.offline_data[m]
                tmp_z= self.sample_z(self.mulogvar_offlinedata[m], len(temp_data_m))

                saz = torch.cat([temp_data_m[:, :(self.sa_dim)], tmp_z], dim=1)
                ds_mulogvar = self.dec(saz)
                ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]
                penalty_target = - log_gaussian(ds_m, # y
                                                ds_mulogvar[:, :self.s_dim], # mu
                                                ds_mulogvar[:, self.s_dim:] # logvar
                                                )
                penalty_target_min = penalty_target.min()
                if self.h_min_tilde>penalty_target_min:
                    self.h_min_tilde = penalty_target_min
                if target_max<penalty_target.max():
                    target_max = penalty_target.max()
            print("penalty_target_min", self.h_min_tilde, "penalty_target_max", target_max)
        self.h_min_tilde = self.h_min_tilde.numpy()
        


    def _loss_train_penalty(self, m):
        temp_data_m = self.offline_data[m]
        tmp_z= self.sample_z(self.mulogvar_offlinedata[m], len(temp_data_m))

        saz = torch.cat([temp_data_m[:, :(self.sa_dim)], tmp_z], dim=1)
        with torch.no_grad():
            ds_mulogvar = self.dec(saz)
        ds_m = temp_data_m[:, (self.sa_dim):(self.sas_dim)]
        penalty_target = - log_gaussian(ds_m, # y
                                        ds_mulogvar[:, :self.s_dim], # mu
                                        ds_mulogvar[:, self.s_dim:] # logvar
                                        )
        sazmulogvar = torch.cat([saz, self.mulogvar_offlinedata[m]*torch.ones((len(saz), 2*self.z_dim))], dim=1)
        penalty_pred = self.penalty_model(sazmulogvar)

        return (( penalty_pred - penalty_target )**2).mean()
    

    def train_penalty(self, num_iter, lr, early_stop_step):
        
        param_list = list(self.penalty_model.parameters())
        loss_fn = self._loss_train_penalty
        ret = self._train(num_iter, lr, early_stop_step, loss_fn, param_list, self.tmp_store_penalty)
        self.tmp_restore_penalty()
        return ret
    

    def eval_loss_unweighted(self):
        total_idx_list = np.array( range(len(self.offline_data)) )
        train_idx_list = copy.deepcopy(total_idx_list)[self.validdata_num:]
        valid_idx_list = copy.deepcopy(total_idx_list)[:self.validdata_num]
        loss_fn = self._loss_train_unweighted_vae
        ave_num = 10        
        with torch.no_grad():

            train_loss_list = []
            for _ in range(ave_num):
                temp_train_loss = 0
                for m in train_idx_list:
                    temp_train_loss += loss_fn(m).item() / len(self.offline_data[m])
                temp_train_loss /= len(train_idx_list)
                train_loss_list.append(temp_train_loss)
            train_loss_list = np.array(train_loss_list)
            train_loss = train_loss_list.mean()

            valid_loss_list = []
            for _ in range(ave_num):
                temp_valid_loss = 0
                for m in valid_idx_list:
                    temp_valid_loss += loss_fn(m).item() / len(self.offline_data[m])
                temp_valid_loss /= len(valid_idx_list)
                valid_loss_list.append(temp_valid_loss)
            valid_loss_list = np.array(valid_loss_list)
            valid_loss = valid_loss_list.mean()

        print("train_loss: ",train_loss)
        print("valid_loss: ",valid_loss)
        self.ell_tilde = (train_loss*len(train_idx_list) + valid_loss*len(valid_idx_list)) / (len(train_idx_list)+len(valid_idx_list))
        self.kappa_tilde = self.c_coeff*0.5*(1-self.gamma)/np.sqrt(self.ell_tilde-self.h_min_tilde)
        print("h_min_tilde", self.h_min_tilde, "ell_tilde", self.ell_tilde, "kappa_tilde", self.kappa_tilde)
        return train_loss, valid_loss
