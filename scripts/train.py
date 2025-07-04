import ipdb
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '.'))
# from diffuser.datasets.dataset_trajdata import UnifiedDataset
# from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
from scripts.trainer import Trainer
# from cobl_diffusion.models.diffusion.ddpm import Diffusion
from cobl_diffusion.models.diffusion.diffusion_ddim_cobl  import GaussianDiffusion
from cobl_diffusion.modules.diffusionmodules.unet import UNetModel
import time
import torch


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#


class LightDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

        self.n_data = self.data.shape[0]
    
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = LightDataset("dataset/train80.pt")

dataloader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=1, # This can be set to 0 for single-threaded loading, if desired.
    pin_memory=True
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

horizon = 80

dynamics = 'single'

unet = UNetModel(
    horizon = horizon
).cuda()

diffusion = GaussianDiffusion(
    model = unet,
    horizon = horizon,
    dynamics = dynamics
).cuda()

timestr = time.strftime("%Y%m%d-%H%M")
output_path = "./logs/cobl_ddim/"+timestr
os.makedirs(output_path, exist_ok=True)

trainer = Trainer(
    diffusion_model=diffusion,
    data_loader=dataloader,
    results_folder=output_path
)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0])
# loss, _ = diffusion.loss(*batch)
# loss.backward()
# print('✓')

#-----------------------------------------------------------------------------#
#----------------------------- training settings -----------------------------#
#-----------------------------------------------------------------------------#

n_epochs = 201

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs}')
    trainer.train(i)

