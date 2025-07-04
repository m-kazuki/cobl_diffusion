import os
import copy
import numpy as np
import torch
import einops
import pdb
import ipdb
import random

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        data_loader,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=64,
        train_lr=2e-5,
        gradient_accumulate_every=1,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=10000,
        label_freq=10000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader = data_loader
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, epoch):

        for traj in self.dataloader:
            loss, loss_pos, loss_term = self.model.loss(traj.cuda()) # loss.shape: scalar, infos[dict]: {a0_loss: 0.5789} <- action loss
            loss = loss + loss_pos + loss_term*10
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = (epoch//50) * 50
                self.save(label)

            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f}, {loss_pos:8.4f}')
                logpath = os.path.join(self.logdir, 'train.log')
                with open(logpath, 'a') as f:
                    log = f'(Epoch {epoch}) {self.step} steps: loss {loss:8.4f}, loss_pos {loss_pos:8.4f}, loss_term {loss_term:8.4f}\n'
                    f.write(log)
            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        savepath = os.path.join(self.logdir, 'latest.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')


    def load(self, epoch, loadpath=None):
        '''
            loads model and ema from disk
        '''
        if loadpath is None:
            loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        state = {k: v for k, v in data['model'].items()
                 if not 'loss' in k}
        dummy = self.model.state_dict()
        dummy.update(state)
        state = dummy
        # self.model.load_state_dict(data['model'], strict=False)
        self.model.load_state_dict(state, strict=False)

        state = {k: v for k, v in data['ema'].items()
                 if not 'loss' in k}
        dummy = self.model.state_dict()
        dummy.update(state)
        state = dummy
        # self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(state, strict=False)

        # self.ema_model.load_state_dict(data['ema'], strict=False)

