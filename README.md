# DATE: Diffusion Adaptive Text Embedding

## Requirements

We utilized CUDA 11.4 on a single NVIDIA A100 GPU.

```
pip install -r requirements.txt
```


## Sampling with DATE

* Dynamic initialization of original text embedding
```.python
python generate_inter.py \
--steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=8 --skip_freq=10 \
--num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=TEST
```

* Fixed initialization of original text embedding
```.python
python generate_inter.py \
--steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=8 --skip_freq=10 \
--num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=0 --init_org --name=TEST
```

## Evaluation

* Zero-shot FID
  * Download `coco.npz` at [this link](https://www.dropbox.com/scl/fi/srrqj78mkm7idye9fpnaw/coco.npz?rlkey=o6exeybbe01a0rpe7sp21n8a0&dl=0).

```.python
python fid.py <directory_of_generated_images> <path_of_coco_npz_file>
```


* CLIP score (& Aesthetic score)
  * Download `aesthetic-model.pth` at [this link](https://www.dropbox.com/scl/fi/xiatomma3iw535t1ylozt/aesthetic-model.pth?rlkey=dd1uiq7559x3nxk17qsjgnork&dl=0).
```.python
python eval_clip_score.py --csv_path=subset.csv --dir_path=<directory_of_generated_images> --aes_path=<path_of_aesthetic-model_pth_file>
```

* ImageReward
```.python
python image_reward.py --image_prefix=<directory_of_generated_images>
```


## Acknowledgements

This work is heavily built upon the code from [Newbeeer/diffusion_restart_sampling](https://github.com/Newbeeer/diffusion_restart_sampling).
* Xu, Y., Deng, M., Cheng, X., Tian, Y., Liu, Z., & Jaakkola, T. (2023). Restart sampling for improving generative processes. Advances in Neural Information Processing Systems, 36, 76806-76838.