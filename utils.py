import numpy as np
import math
import torch

log2pi = math.log(2*math.pi)


def torch_from_numpy(x):
    if type(x)==type(np.array([0])):
        return torch.from_numpy(x.astype(np.float32))
    return x


def log_gaussian(y, mu, logvar):
    # we assume variance matrix is diagonal.
    # log(det|Var|) = log(prod_i var_i) = sum_i log(var_i)
    y_ = y
    if y.flatten().shape[0] == mu.flatten().shape[0]:
        mu_ = mu
        logvar_ = logvar
    else:
        mu_ = torch.ones_like(y) * mu.reshape(1,-1)
        logvar_ = torch.ones_like(y) * logvar.reshape(1,-1)
    return - 0.5 * (((y_-mu_)**2) * torch.exp(-logvar_) + logvar_ + log2pi).sum(-1)


def kld(mu1, logvar1, mu2, logvar2):
    # we assume variance matrix is diagonal.
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # kld(p1|p2) = E_{z~p1}[ log p1(z) - log p2(z) ]
    mu1 = mu1.flatten()
    logvar1 = logvar1.flatten()
    mu2 = mu2.flatten()
    logvar2 = logvar2.flatten()
    tmp1 = 0.5 * (logvar2 - logvar1) # log (sigma2/sigma1)
    tmp2 = 0.5 * (torch.exp(logvar1)+(mu1-mu2)**2) / torch.exp(logvar2) # (sigma1^2+(mu1-mu2)^2)/(2*sigma2^2)
    return torch.sum(tmp1 + tmp2 - 0.5)


def kdl_var_approx(mulogvar_1, mulogvarlist_2):
    # Eq (20) in Approximating the Kullback Leibler Divergence Between Gaussian Mixture Models (2007)
    # https://www.researchgate.net/profile/John-Hershey-4/publication/4249249_Approximating_the_Kullback_Leibler_Divergence_Between_Gaussian_Mixture_Models/links/56b46b9d08ae61c480592ca9/Approximating-the-Kullback-Leibler-Divergence-Between-Gaussian-Mixture-Models.pdf
    # Assume f is single gaussian, and g is a mixture of gaussian with uniform weights, i.e.
    # f = N(mu1,logvar1), g = (1/M) * sum_m N(mu2_list[m], logvar2_list[m])
    # Then, fa = falpha, w_a_f = 1, and kld_fa_falph=1, resulting in numerator=1.
    # Eq (20) = - log( (1 / M) *  sum_m exp(-kld_fa_gb)) = log(M) - log(sum_m exp(-kld_fa_gb))
    z_dim = len(mulogvar_1)//2

    minus_kld_fa_gb_array = torch.stack([-kld(mulogvar_1[:z_dim], 
                                              mulogvar_1[z_dim:],
                                              mulogvarlist_2[m][:z_dim],
                                              mulogvarlist_2[m][z_dim:]) for m in range(len(mulogvarlist_2))])
    tmp_max = minus_kld_fa_gb_array.max()
    logsumexp_minus_kld_fa_gb = torch.log(torch.sum(torch.exp(minus_kld_fa_gb_array - tmp_max))) + tmp_max # log-sum-exp trick
    return np.log(len(mulogvarlist_2))-logsumexp_minus_kld_fa_gb


# def saz_array_to_sads_array(saz_array):
#     ds = saz_array[1:,:self.s_dim] - saz_array[:-1,:self.s_dim]
#     return np.hstack([saz_array[:-1,:self.s_dim+self.a_dim], ds])


# for sac
def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# for density ratio estimation
def logireg_loss(de_r, nu_r): # weight decay around 0.1
    # min [- (1/N_de)*sum_{j} log (1/(1+r(x_de^j)))   - (1/N_nu)*sum_{i} log (r(x_nu^i)/(1+r(x_nu^i))) ]
    return -torch.mean(-torch.log(1.+de_r)) -torch.mean(-torch.log(1.+nu_r) + torch.log(nu_r))
