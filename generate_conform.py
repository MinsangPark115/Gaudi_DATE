# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, ConformPipeline, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
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
import json
import random
import time

start_time = time.time()
print(start_time)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1024, 65535))
os.environ['RANK'] = '0'
os.environ['PT_HPU_LAZY_MODE'] = '0'
import habana_frameworks.torch.distributed.hccl
torch.distributed.init_process_group(backend='hccl')
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
torch.set_num_threads(4)
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
parser.add_argument('--text_path', type=str, default='subset_ae_repeat.csv')
parser.add_argument('--json_path', type=str, default='conform_prompt_token-groups.json')
parser.add_argument('--adj', type=int, default=0)
parser.add_argument('--max_iter_to_alter', type=int, default=25)
parser.add_argument('--fp16', action='store_true', default=False)
args = parser.parse_args()


# dist.init()

dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))
# define dataset / data_loader

#df = pd.read_csv('./coco/coco/subset.csv')
df = pd.read_csv(args.text_path)
all_text = list(df['caption'])
all_text = all_text[: args.max_cnt]

with open(args.json_path, "r") as f:
    json_token_groups = json.load(f)

num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]


index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]


##### load stable diffusion models #####
if args.fp16:
    sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
else:
    sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
dist.print0("default scheduler config:")
dist.print0(sd_pipeline.scheduler.config)
sd_pipeline = sd_pipeline.to("hpu") # cuda -> hpu

if args.scheduler == 'DDPM':
    sd_pipeline.scheduler = DDPMScheduler.from_config(sd_pipeline.scheduler.config)
elif args.scheduler == 'DDIM':
    # recommend using DDIM with Restart
    sd_pipeline.scheduler = DDIMScheduler.from_config(sd_pipeline.scheduler.config)
    sd_pipeline.scheduler.use_sigma = args.sigma
elif args.scheduler == 'SDE':
    sd_pipeline.scheduler = SDEScheduler.from_config(sd_pipeline.scheduler.config)
elif args.scheduler == 'ODE':
    sd_pipeline.scheduler = EulerDiscreteScheduler.from_config(sd_pipeline.scheduler.config)
    sd_pipeline.scheduler.use_karras_sigmas = False
else:
    raise NotImplementedError

pipe = ConformPipeline(
    vae=sd_pipeline.vae,
    text_encoder=sd_pipeline.text_encoder,
    tokenizer=sd_pipeline.tokenizer,
    unet=sd_pipeline.unet,
    scheduler=sd_pipeline.scheduler,
    safety_checker=sd_pipeline.safety_checker,
    feature_extractor=sd_pipeline.feature_extractor,
)

generator = torch.Generator(device="cpu").manual_seed(args.generate_seed) # cuda -> cpu

##### setup save configuration #######
if args.name is None:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}')
else:
    save_dir = os.path.join(args.save_path,
                            f'scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}_name_{args.name}')

dist.print0("save images to {}".format(save_dir))

if dist.get_rank() == 0 and not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)

## generate images ##
for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    text = list(mini_batch)
    token_groups = [json_token_groups[t] for t in text]

    if rank_batches_index[cnt][-1] < args.adj:
        skip_images = True
    else:
        skip_images = False

    image, _ = pipe(
        prompt=text,
        token_groups=token_groups,
        guidance_scale=args.w,
        generator=generator,
        num_inference_steps=args.steps,
        max_iter_to_alter=args.max_iter_to_alter,
        skip_images=skip_images,
    )
    if not skip_images:
        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            image[text_idx].save(os.path.join(save_dir, f'{global_idx}.png'))
    gc.collect()
    # torch.cuda.empty_cache()

# Done.
torch.distributed.barrier()
if dist.get_rank() == 0:
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(save_dir, 'subset.csv'))