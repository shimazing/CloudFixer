CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --model transformer \
    --exp_name \
    unit_std_transformer_polynomial_2_500steps_nozeromean_LRExponentialDecay0.9995_resume \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --batch_size 64 \
    --accum_grad 1 \
    --n_epochs 10000 \
    --scale 1 \
    --scale_mode unit_std \
    --jitter False \
    --lr 2e-4 \
    --lr_gamma 0.9995 \
    --no_zero_mean \
