import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--path1', type=str, default='image_reward_scores.npz')
parser.add_argument('--path2', type=str, default='image_reward_scores.npz')
args = parser.parse_args()

if __name__ == "__main__":
    data1 = np.load(args.path1)
    data2 = np.load(args.path2)

    array1 = data1['scores']
    array2 = data2['scores']

    wins = np.sum(array2 > array1)

    win_ratio = wins / len(array1)

    print(f"# wins: {wins}")
    print(f"win percentage: {win_ratio * 100:.2f}%")
