wandb_usr=drumpt
dataset=modelnet40c_background_5
dataset_dir=../datasets/modelnet40_c/
classifier=dgcnn
t_max=0.8
t_min=0.02
steps=400
array=(0)
lam_h=10
lam_l=1
wd=0
lr=0.2
severity=5
optim_end_factor=0.05
mode=vis
weighted_reg=True
pow=1
optim=adamax
subsample=700
pre_trans=0
seed=2
denoising_thrs=100
batch_size=4

if [ "1" == "$pre_trans" ]; then
    exp_name=${classifier}_${corruption}_${severity}_lr${lr}_ef${optim_end_factor}_t_${t_max}_${t_min}_${steps}iters_betas_0.9_0.999_wd_0_pow${pow}weighted${weighted_reg}_lam_${lam_h}_${lam_l}_cosaneal_wd${wd}_${optim}_schedule_t_tlb_lr_linearLR_sub${subsample}_wRotation0.02_denoisingThrs${denoising_thrs}_trans${trans}_seed${seed}
    pre_trans=--pre_trans
else
    exp_name=${classifier}_${corruption}_${severity}
    pre_trans=""
fi

python3 adapt.py \
    --t_min ${t_min} \
    --t_max ${t_max} \
    --save_itmd 0 \
    --denoising_thrs ${denoising_thrs} \
    --random_seed ${seed} \
    ${pre_trans} \
    --pow ${pow} \
    --optim_end_factor ${optim_end_factor} \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --batch_size ${batch_size} \
    --scale_mode unit_std \
    --cls_scale_mode unit_norm \
    --dataset ${dataset} \
    --dataset_dir ${dataset_dir} \
    --exp_name ${exp_name} \
    --mode ${mode} \
    --model transformer \
    --resume \
    outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy \
    --lr ${lr} \
    --n_update ${steps} \
    --weight_decay ${wd} \
    --lam_l ${lam_l} \
    --lam_h ${lam_h} \
    --beta1 0.9 \
    --beta2 0.999 \
    --optim ${optim} \
    --subsample ${subsample} \
    --weighted_reg ${weighted_reg} \
    --wandb_usr ${wandb_usr}


# #for corruption in background cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling
# for corruption in background; do # cutout density density_inc distortion distortion_rbf distortion_rbf_inv gaussian impulse lidar occlusion rotation shear uniform upsampling
#     classifier=dgcnn
#     t_max=0.8
#     t_min=0.02
#     steps=400
#     array=(0)
#     lam_h=10
#     lam_l=1
#     wd=0
#     lr=0.2
#     severity=5
#     optim_end_factor=0.05
#     mode=eval
#     weighted_reg=True
#     pow=1
#     optim=adamax
#     subsample=700
#     pre_trans=0
#     seed=2
#     denoising_thrs=100
#     batch_size=32

#     if [ "1" == "$pre_trans" ]; then
#         exp_name=${classifier}_${corruption}_${severity}_lr${lr}_ef${optim_end_factor}_t_${t_max}_${t_min}_${steps}iters_betas_0.9_0.999_wd_0_pow${pow}weighted${weighted_reg}_lam_${lam_h}_${lam_l}_cosaneal_wd${wd}_${optim}_schedule_t_tlb_lr_linearLR_sub${subsample}_wRotation0.02_denoisingThrs${denoising_thrs}_trans${trans}_seed${seed}
#         pre_trans=--pre_trans
#     else
#         exp_name=${classifier}_${corruption}_${severity}
#         pre_trans=""
#     fi

#     CUDA_VISIBLE_DEVICES=2,3 python3 adapt.py \
#         --t_min ${t_min} \
#         --t_max ${t_max} \
#         --save_itmd 0 \
#         --denoising_thrs ${denoising_thrs} \
#         --random_seed ${seed} \
#         ${pre_trans} \
#         --pow ${pow} \
#         --optim_end_factor ${optim_end_factor} \
#         --diffusion_steps 500 \
#         --diffusion_noise_schedule polynomial_2 \
#         --batch_size ${batch_size} \
#         --scale_mode unit_std \
#         --cls_scale_mode unit_norm \
#         --dataset modelnet40c_${corruption}_${severity} \
#         --dataset_dir ../datasets/modelnet40_c/ \
#         --exp_name ${exp_name} \
#         --mode ${mode} \
#         --model transformer \
#         --resume \
#         outputs/diffusion_model_transformer_modelnet40/generative_model_ema_last.npy \
#         --lr ${lr} \
#         --n_update ${steps} \
#         --weight_decay ${wd} \
#         --lam_l ${lam_l} \
#         --lam_h ${lam_h} \
#         --beta1 0.9 \
#         --beta2 0.999 \
#         --optim ${optim} \
#         --subsample ${subsample} \
#         --weighted_reg ${weighted_reg} \
#         --wandb_usr ${wandb_usr}
# done