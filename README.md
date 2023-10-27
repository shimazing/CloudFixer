# CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Domain Translation

## Environmental Setup
- Environment setting
```
conda create -n cloudfixer python=3.8.16
conda activate cloudfixer
bash set_env.sh
```
- Download and preprocess data by following https://github.com/zou-longkun/GAST in ./data/
- (Not necessary) Only for quantitative evaluation of DM Install PyTorchEMD by 
```
cd PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
``