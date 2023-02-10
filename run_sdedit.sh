CUDA_VISIBLE_DEVICES=0 python3 main_sdedit.py \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --batch_size 8 \
    --classifier \
        '../GAST_ori/experiments/GAST_SPST_modelnet/model.ptdgcnn' \
    --scale 1 \
    --scale_mode unit_std \
    --cls_scale_mode unit_norm \
    --dataset modelnet \
    --self_ensemble \
    --K 3 \
    --t 0.2 \
    --random_seed 1 \
    --mode eval \
    --exp_name transformer_modelnet \
    --model transformer \
    --resume \
    outputs/unit_std_transformer_polynomial_2_500steps_resume/generative_model_ema_last.npy \
    --egsde \
    --lambda_s 0 \
    --lambda_i 1000 \
    --domain_cls \
    outputs/domain_classifier_DGCNN_shape_model_timecondGN_fullt.pt \
    --no_wandb \
    #--voxelization \
    #--voxel_resolution 5 \
    #--model pvd \
    #--resume \
    #outputs/unit_std_pvd_polynomial_2_500steps_resume_resume/generative_model_ema_last.npy \
    #--mode 'eval' \
    #--classifier \
    #    '../GAST_ori/experiments/GAST_scannet_nodup/model.ptdgcnn' \
    #--ddim \
    #--n_inversion_steps 50 \
    #--n_reverse_steps 50 \
    #--classifier \
    #    '../GAST_ori/experiments/GAST_SPST_scannet_lr1e-4_wd0_layernorm_bsz16/model.ptdgcnn' \
    #outputs/domain_classifier_DGCNN_shape_model_timecondGN.pt \
    #--n_subsample 128 \
    #--entropy_guided \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_fourth_polynomial_2_500steps/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_fourth_radius.json \
    #--no_wandb \
    #--model transformer \
    #--resume \
    #outputs/unit_std_transformer_polynomial_2_500steps_resume/generative_model_ema_last.npy \
    #    '../GAST_ori/experiments/GAST_SPST/model.ptdgcnn' \
    #--resume \
    #outputs/unit_val_shapenet_pointnet_scale1_polynomial_2_500steps_my_config/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_my.json \
    #--resume \
    #outputs/unit_val_shapenet_pointnet_scale1_polynomial_2_500steps_my_config/generative_model_ema_last.npy \
    #    '../DefRec_and_PCM/PointDA/experiments/defrec_pcm_shape2modelnet/model.ptdgcnn' \
    #     '../GAST/experiments/GAST_balanced_unitnorm_randomscale0.2/model.ptdgcnn' \
    #--keep_sub \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_add_feature_level_polynomial_2_500steps_resume/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_add_feature_level.json \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_use_radius_dec_polynomial_2_500steps/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_use_radius_dec.json \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_downsample_tcondpnet_polynomial_2_500steps_randomscale/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_downsample_tcondpnet.json \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radius_double_downsample_polynomial_2_500steps/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius_downsample.json \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radiusdouble_balanced_resume/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius.json \
    #--resume \
    #outputs/unit_std_shapenet_pointnet_wo_localcond_radiusdouble_balanced_linear1000/generative_model_ema_last.npy \
    #--dynamics_config \
    #pointnet2/exp_configs/mvp_configs/config_standard_attention_larger_radius.json \
    #    'best_model_shape_model.pt' \
    #    '../GAST/experiments/GAST_SPST/model.ptdgcnn' \
