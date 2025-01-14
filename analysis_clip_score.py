import torch
from PIL import Image
import open_clip
from coco_data_loader import text_image_pair, text_image_pair_pred
import argparse
from tqdm import tqdm
import clip
import aesthetic_score
import numpy as np

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--bs', type=int, default=10)
parser.add_argument('--max_cnt', type=int, default=100, help='number of maximum geneated samples')
parser.add_argument('--csv_path', type=str, default='./generated_images/subset.csv')
parser.add_argument('--dir_path', type=str, default='./generated_images/subset')
parser.add_argument('--scheduler', type=str, default='DDPM')
parser.add_argument('--num_eval', type=int, default=5000)
args = parser.parse_args()

# define dataset / data_loader
text2img_dataset = text_image_pair_pred(dir_path=args.dir_path, csv_path=args.csv_path)
text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)

print("total length:", len(text2img_dataset))
model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model = model.cuda().eval()
tokenizer = open_clip.get_tokenizer('ViT-g-14')

# text = tokenizer(["a horse", "a dog", "a cat"])
cnt = 0.
total_clip_score = 0.
total_aesthetic_score = 0.

lst_image = []
lst_convex = []
bool_convex = np.zeros(args.steps)

with torch.no_grad(), torch.cuda.amp.autocast():
    for idx, (image, lst_pred, text) in tqdm(enumerate(text2img_loader)):
        image = image.cuda().float()
        text = list(text)
        text = tokenizer(text).cuda()
        # print('text:')
        # print(text.shape)
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text).float()
        # (bs, 1024)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_clip_score = (image_features * text_features).sum(dim=-1)
        lst_image.append(image_clip_score)

        lst_c = []
        for k in range(len(lst_pred)):
            pred = lst_pred[k].cuda().float()
            pred_features = model.encode_image(pred).float()
            pred_features /= pred_features.norm(dim=-1, keepdim=True)
            pred_clip_score = (pred_features * text_features).sum(dim=-1)
            bool_convex[k] += (image_clip_score >= pred_clip_score).float().sum().item()
            lst_c.append(pred_clip_score)
        lst_convex.append(lst_c)
        cnt += len(image)
        if cnt >= args.num_eval:
            print(f'Evaluation complete! NUM EVAL: {cnt}')
            break

        print(idx, bool_convex)
import os
import pickle
image_path = os.path.join(f'lst_image.pickle')
with open(image_path, 'wb') as f:
    pickle.dump(lst_image, f)
pred_path = os.path.join(f'lst_convex.pickle')
with open(pred_path, 'wb') as f:
    pickle.dump(lst_convex, f)

print(bool_convex)

# print("Average ClIP score :", total_clip_score.item() / cnt)
# print("Average Aesthetic score :", total_aesthetic_score.item() / cnt)