import numpy as np
import torch
from torch import nn
import ipdb

from cobl_diffusion.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from cobl_diffusion.reward import single_cbf_reward_fn_pairwise, single_clf_reward_fn

SCALE_FACTOR = 10.0

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class WeightedLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss
    
class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)
        

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, n_timesteps=1000, dynamics='single', dt=0.1,
        loss_type='l1', clip_denoised=False, predict_epsilon=False,
        action_weight=1.0, loss_discount=1.0, loss_weights=None
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = 2
        self.model = model
        self.dynamics = dynamics
        self.dt = dt

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = nn.L1Loss()
        self.loss_pos = nn.L1Loss()
        self.loss_term = nn.L1Loss()


    def make_schedule(self, ddim_num_steps, ddim_eta=0):
        # ddim sampling parameters
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method='uniform', num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.n_timesteps, verbose=False)
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas.to(self.betas.device))
        self.register_buffer('ddim_alphas', ddim_alphas.to(self.betas.device))
        self.register_buffer('ddim_alphas_prev', torch.tensor(ddim_alphas_prev).to(self.betas.device))
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas).to(self.betas.device))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    #------------------------------------------ sampling ------------------------------------------#

    @torch.no_grad()
    def ddim_sample_loop(self, shape, init_state, goal, state, scale=0.1, temperature=0, 
                         u_hist=None, formula=None, approx_method="true", stl_temperature=1.0, 
                         obst=None, r=0.5, s_cbf=0.1, s_clf=0.1, **kwargs):
        # x: [bs, 4, time]
        # neigh: [bs, n_neigh, 2, time]

        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        subset_end = int(min(self.n_timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
        timesteps = self.ddim_timesteps[:subset_end]
        timesteps = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        x[:,:,:u_hist.shape[-1]] = u_hist.cuda()

        self.nom_state = state

        single_cbf_grad_fn = torch.func.grad(single_cbf_reward_fn_pairwise)
        single_clf_grad_fn = torch.func.grad(single_clf_reward_fn)
        batched_cbf_grad_fn = torch.vmap(single_cbf_grad_fn, in_dims=(0, None, None, None))
        batched_clf_grad_fn = torch.vmap(single_clf_grad_fn, in_dims=(0, None))

        for i, step in enumerate(timesteps):
            index = total_steps - i - 1
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            pred_x0 = self.model(x, pos=state[:,:2,:], time=t)

            a_t = torch.full((batch_size, 1, 1), self.ddim_alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1), self.ddim_alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1), self.ddim_sigmas[index], device=device)
            # direction pointing to x_t
            dir_xt = ((1. - a_prev) - (sigma_t**2)).sqrt() * (x-a_t.sqrt()*pred_x0)/((1. - a_t)).sqrt()
            noise = sigma_t * noise_like(x.shape, device, False) * temperature

            x = a_prev.sqrt() * pred_x0 + dir_xt + noise
            
            if i==0:
                continue

            for j in range(10):
                grad_cbf = batched_cbf_grad_fn(x, obst[0,:,:2,:], obst[0,:,2:,:], r)
                grad_clf = batched_clf_grad_fn(x, goal.squeeze(0))
                x_norm = torch.norm(x, dim=1, keepdim=True)
                grad_cbf_norm = torch.norm(grad_cbf, dim=1, keepdim=True)
                grad_clf_norm = torch.norm(grad_clf, dim=1, keepdim=True)
                normalized_grad_cbf = grad_cbf * x_norm / (grad_cbf_norm+1e-8)
                normalized_grad_clf = grad_clf * x_norm / (grad_clf_norm+1e-8)
                x = x + s_cbf * normalized_grad_cbf + s_clf * normalized_grad_clf

            state = self.get_state(x)
            state[:,:2,-1] = self.nom_state[:,:2,-1].clone()

        return x, state
        
    
    @torch.no_grad()
    def ddim_sample(self, init_state, goal, state, ddim_steps, scale=0.1,
                    temperature=0, u_hist=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = state.shape[2]

        self.make_schedule(ddim_steps)
        batch_size = init_state.shape[0]
        shape = (batch_size, self.transition_dim, horizon)

        control, state = self.ddim_sample_loop(shape=shape, init_state=init_state, goal=goal, state=state, scale=scale,
                                        temperature=temperature, u_hist=u_hist, *args, **kwargs)


        return control, state
    
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pos_noisy = self.q_sample(x_start=state[:,:2,:], t=t, noise=noise)

        pos_noisy[:,:,0] = state[:,:2,0]
        pos_noisy[:,:,-1] = state[:,:2,-1]

        x_recon = self.model(x_noisy, pos_noisy.detach(), t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)
            
        state_pred = self.get_state(x_recon)
        loss_pos = self.loss_pos(state_pred[:,:2,:], state[:,:2,:].detach())
        loss_term = self.loss_term(state_pred[:,:2,-1], state[:,:2,-1].detach())

        return loss, loss_pos, loss_term

    def loss(self, traj):
        batch_size = len(traj)
        x, state = self.extract_states(traj)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t)


    def extract_states(self, traj):
        if self.dynamics=='single':
            # traj: [x, y, xd, yd]
            traj = traj/SCALE_FACTOR
            x = traj[:,2:4,:]
            state = traj[:, :2, :]
            state = torch.cat([torch.zeros(state.shape[0], 2, 1).to(state.device), state], dim=2) # [bs, 2, time+1]
            state = state[:,:,:80]
        elif self.dynamics=='unicycle':
            # traj: [x, y, xd, yd, s, c, v, theta, omega]
            traj = traj/SCALE_FACTOR
            x = traj[:,(6,8),:]
            state = traj[:, (0,1,7), :]
        return x, state


    # scale factor is 10
    # computes state sequence given control sequence
    # assumes initial state is always zero.
    def get_state(self, control):
        if self.dynamics=='single':
            state = torch.cumsum(self.dt*control, dim=2)
            state = torch.cat([torch.zeros(control.shape[0], 2, 1).to(control.device), state], dim=2) # [bs, 2, time+1]
            state = state[:,:,:80]
        elif self.dynamics=='unicycle':
            theta = torch.cumsum(self.dt*control[:,1,:]*SCALE_FACTOR, dim=1)
            delta_x = control[:,0,:] * torch.cos(theta)
            delta_y = control[:,0,:] * torch.sin(theta)
            x = torch.cumsum(delta_x, dim=1)*self.dt
            y = torch.cumsum(delta_y, dim=1)*self.dt
            state = torch.cat([x.unsqueeze(1), y.unsqueeze(1), theta.unsqueeze(1)/SCALE_FACTOR], dim=1)
        return state