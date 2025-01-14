import os
import torch
import argparse
import pandas as pd
import ImageReward as RM
import numpy as np
from tqdm import tqdm
import random
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(random.randint(1024, 65535))
os.environ['RANK'] = '0'
os.environ['PT_HPU_LAZY_MODE'] = '0'
import habana_frameworks.torch.distributed.hccl
torch.distributed.init_process_group(backend='hccl')

parser = argparse.ArgumentParser(description='ImageReward Evaluation')
parser.add_argument('--text_path', type=str, default='subset.csv')
parser.add_argument('--image_prefix', type=str, default='gen_images')
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum geneated samples')
parser.add_argument('--bs', type=int, default=4, help='batch size')
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.text_path)
    all_text = list(df['caption'])
    all_text = all_text[: args.max_cnt]

    img_prefix = args.image_prefix
    generations = [f"{pic_id}.png" for pic_id in range(0, args.max_cnt)]
    img_list = [os.path.join(img_prefix, img) for img in generations]

    model = RM.load("ImageReward-v1.0", device='hpu') # cuda->hpu
    avg_score = 0.

    scores = np.zeros((len(img_list),), dtype=np.float32)

    with torch.no_grad():
        for index in tqdm(range(0, len(img_list), args.bs)):
            score = model.score_batch(all_text[index:index+args.bs], img_list[index:index+args.bs])
            scores[index:index+args.bs] = score.cpu().numpy()
            avg_score += score.sum()
    np.savez(f'{img_prefix}/image_reward_scores.npz', scores=scores)
    avg_score /= len(img_list)
    print(avg_score)
