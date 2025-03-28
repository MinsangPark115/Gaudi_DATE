import torch
from PIL import Image
import open_clip
from coco_data_loader import text_image_pair
import argparse
from tqdm import tqdm
import clip
import aesthetic_score
import numpy as np
import random
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1024, 65535))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['PT_HPU_LAZY_MODE'] = '0'

import habana_frameworks.torch.hpu as hpu

import habana_frameworks.torch.distributed.hccl
torch.distributed.init_process_group(backend='hccl')

import habana_frameworks.torch.core as htcore


parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=100, help='number of maximum geneated samples')
parser.add_argument('--csv_path', type=str, default='./generated_images/subset.csv')
parser.add_argument('--dir_path', type=str, default='./generated_images/subset')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--num_eval', type=int, default=5000)
parser.add_argument('--aes_path', type=str, default='./clip-refs/aesthetic-model.pth')
args = parser.parse_args()

# define dataset / data_loader
text2img_dataset = text_image_pair(dir_path=args.dir_path, csv_path=args.csv_path)
text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)

print("total length:", len(text2img_dataset))
model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model2, _ = clip.load("ViT-L/14", device='hpu')  #RN50x64 # cuda -> hpu
model = model.to("hpu").eval() # cuda -> hpu
model2 = model2.eval()
tokenizer = open_clip.get_tokenizer('ViT-g-14')


model_aes = aesthetic_score.MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
#s = torch.load("./clip-refs/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
s = torch.load(args.aes_path, map_location=torch.device("hpu"))

model_aes.load_state_dict(s)
model_aes.to("hpu") # cuda -> hpu
model_aes.eval()

# text = tokenizer(["a horse", "a dog", "a cat"])
cnt = 0.
total_clip_score = 0.
total_aesthetic_score = 0.

scores = np.zeros((args.num_eval,), dtype=np.float32)
with torch.no_grad(), torch.autocast(device_type="hpu"):
    for idx, (image, image2, text) in tqdm(enumerate(text2img_loader)):
        image = image.to("hpu").float()
        image2 = image2.to("hpu").float()
        text = list(text)
        text = tokenizer(text).to("hpu")
        # print('text:')
        # print(text.shape)
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text).float()
        # (bs, 1024)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # total_clip_score += (image_features * text_features).sum()
        # scores[args.bs*idx:args.bs*idx+len(image)] = (image_features * text_features).sum(dim=-1).cpu().numpy()

        # image_features = model2.encode_image(image)
        # im_emb_arr = aesthetic_score.normalized(image_features.cpu().detach().numpy())
        # total_aesthetic_score += model_aes(torch.from_numpy(im_emb_arr).to(image.device).type(torch.cuda.FloatTensor)).sum()

        total_clip_score += (image_features * text_features).sum().float()  # BFloat16 -> Float32로 변환
        scores[args.bs * idx : args.bs * idx + len(image)] = (image_features * text_features).sum(dim=-1).float().cpu().numpy()

        image_features = model2.encode_image(image)
        im_emb_arr = aesthetic_score.normalized(image_features.cpu().detach().float().numpy())  # Float32 변환
        total_aesthetic_score += model_aes(torch.from_numpy(im_emb_arr).to(image.device).float()).sum()


        cnt += len(image)

        if cnt >= args.num_eval:
            print(f'Evaluation complete! NUM EVAL: {cnt}')
            break

np.savez(f'{args.dir_path}/clip_scores.npz', scores=scores)

print("Average ClIP score :", total_clip_score.item() / cnt)
print("Average Aesthetic score :", total_aesthetic_score.item() / cnt)