# make sure you're logged in with \`huggingface-cli login\`
from diffusers import EnergyStableDiffusionPipeline, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
from diffusers.models.energy_unet_2d_condition import EnergyUNet2DConditionModel
from diffusers.utils.gamma_scheduler import get_gamma_scheduler
import torch
from coco_data_loader import text_image_pair
from PIL import Image
import os
import pandas as pd
import argparse
import torch.nn as nn
from torch_utils import distributed as dist
import numpy as np
import tqdm
import gc

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=30, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--s_noise', type=float, default=1.)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--second', action='store_true', default=False, help='second order ODE')
parser.add_argument('--sigma', action='store_true', default=False, help='use sigma')
parser.add_argument('--text_path', type=str, default='subset.csv')
parser.add_argument('--adj', type=int, default=0)
parser.add_argument('--gamma_attn', type=float, default=0.01, help="initial weight coefficient for attention term")
parser.add_argument('--gamma_norm', type=float, default=0.02, help="initial weight coefficient for normalization term")
parser.add_argument('--gamma_tau', type=float, default=1., help="Turn off gammas after some time. r == 1: gamma never turns off.")
args = parser.parse_args()


dist.init()

dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))
# define dataset / data_loader

#df = pd.read_csv('./coco/coco/subset.csv')
df = pd.read_csv(args.text_path)
all_text = list(df['caption'])
all_text = all_text[: args.max_cnt]

num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]


index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

model_id = "runwayml/stable-diffusion-v1-5"
unet = EnergyUNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    # torch_dtype=torch.float16,
    down_block_types=(
        "EnergyCrossAttnDownBlock2D", "EnergyCrossAttnDownBlock2D", "EnergyCrossAttnDownBlock2D", "DownBlock2D",
    ),
    mid_block_type="EnergyUNetMidBlock2DCrossAttn",
    up_block_types=(
        "UpBlock2D", "EnergyCrossAttnUpBlock2D", "EnergyCrossAttnUpBlock2D", "EnergyCrossAttnUpBlock2D"
    ),
)

gamma_attn = get_gamma_scheduler(name='reverse_step',
                                 gamma_tau=args.gamma_tau, gamma_src=args.gamma_attn)(num_time_steps=args.steps)
gamma_norm = get_gamma_scheduler(name='reverse_step',
                                 gamma_tau=args.gamma_tau, gamma_src=args.gamma_norm)(num_time_steps=args.steps)
print(gamma_attn)
print(gamma_norm)

##### load stable diffusion models #####
pipe = EnergyStableDiffusionPipeline.from_pretrained(model_id, unet=unet)#, torch_dtype=torch.float16)

dist.print0("default scheduler config:")
dist.print0(pipe.scheduler.config)
pipe = pipe.to("cuda")

if args.scheduler == 'DDPM':
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'DDIM':
    # recommend using DDIM with Restart
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_sigma = args.sigma
elif args.scheduler == 'SDE':
    pipe.scheduler = SDEScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'ODE':
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = False
elif args.scheduler == 'PNDM':
    pass
else:
    raise NotImplementedError

generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)

##### setup save configuration #######
if args.name is None:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}')
else:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}_name_{args.name}')

dist.print0("save images to {}".format(save_dir))

if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)

## generate images ## (YM_edited)
for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    text = list(mini_batch)
    # image = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, restart=args.restart, second_order=args.second, dist=dist, S_noise=args.s_noise).images
    SD_out = pipe(text, generator=generator, num_inference_steps=args.steps,
                         gamma_attn=gamma_attn, gamma_norm=gamma_norm,
                         guidance_scale=args.w, scheduler=args.scheduler)

    image = SD_out.images
    for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
        # image_torch = torch.from_numpy(image[text_idx])[0]
        # images_np = (image_torch * 255).clip(0, 255).to(torch.uint8).cpu().numpy()
        # print(images_np.shape)
        # print(images_np)
        # print(np.max(images_np))
        # print(np.min(images_np))
        # im = Image.fromarray(images_np)
        # im.save(os.path.join(save_dir, f'{global_idx}.png'))
        image[text_idx].save(os.path.join(save_dir, f'{global_idx+args.adj}.png'))
    gc.collect()
    torch.cuda.empty_cache()

# Done.
torch.distributed.barrier()
if dist.get_rank() == 0:
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(save_dir, 'subset.csv'))