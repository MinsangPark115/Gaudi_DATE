def command_to_string(command):
    # example command : MASTER_PORT=8007 CUDA_VISIBLE_DEVICES=7 python3 generate_inter_v6.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=8 --skip_freq=10 --num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=inter_v6_skip10_num1_lr0.5_weight591_w8
    # output : scheduler_{args.scheduler}_steps_{args.steps}_restart_{args.restart}_w_{args.w}_second_{args.second}_seed_{args.generate_seed}_sigma_{args.sigma}_name_{args.name}
    com_list = command.split(" ")
    for i in range(len(com_list)):
        if com_list[i].startswith("--"):
            com_list[i] = com_list[i][2:]
        if "=" in com_list[i]:
            com_list[i] = com_list[i].split("=")
        else:
            com_list[i] = [com_list[i], None] 
    targ_list = ["scheduler", "steps", "restart", "w", "second", "seed", "sigma", "name"]
    output_str = ""
    for i in targ_list:
        if i =="seed":
            output_str += f"{i}_6_"
            continue
        targ = False
        for j in com_list:
            if j[0] == i:
                if i=="w":
                    output_str += f"{i}_{j[1]}.0_"
                    targ = True
                    break
                else:
                    output_str += f"{i}_{j[1]}_"
                    targ = True
                    break
        if targ == False:
            output_str += f"{i}_False_"
    output_str = output_str[:-1]

    # print(com_list)
    print(output_str)
targ_str ="MASTER_PORT=5000 CUDA_VISIBLE_DEVICES=0 python3 generate.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=8 --name=base_w8_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=5000 CUDA_VISIBLE_DEVICES=0 python3 generate.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=5 --name=base_w5_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=5000 CUDA_VISIBLE_DEVICES=0 python3 generate.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=3 --name=base_w3_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=5000 CUDA_VISIBLE_DEVICES=0 python3 generate.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=2 --name=base_w2_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=8001 CUDA_VISIBLE_DEVICES=1 python3 generate_inter.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=5 --skip_freq=10 --num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=inter_v3.2_skip10_num1_lr0.5_weight591_w5_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=8002 CUDA_VISIBLE_DEVICES=2 python3 generate_inter.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=3 --skip_freq=10 --num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=inter_v3.2_skip10_num1_lr0.5_weight591_w3_gaudi"
command_to_string(targ_str)
targ_str ="MASTER_PORT=8003 CUDA_VISIBLE_DEVICES=3 python3 generate_inter.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=2 --skip_freq=10 --num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=inter_v3.2_skip10_num1_lr0.5_weight591_w2_gaudi"
#  "MASTER_PORT=8007 CUDA_VISIBLE_DEVICES=7 python3 generate_inter_v6.py --steps=50 --scheduler=DDIM --save_path=gen_images --bs=4 --w=8 --skip_freq=1 --num_upt_prompt=1 --lr_upt_prompt=0.5 --weight_prior=591.36 --name=inter_v6_skip1_num1_lr0.5_weight591_w8"
command_to_string(targ_str)