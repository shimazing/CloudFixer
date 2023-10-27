# CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Domain Translation

### Environmental Setup
- Download  and preprocess data by following https://github.com/zou-longkun/GAST in ./data/
- Env setting
```
conda create -n [new_env_name] python=3.8
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install wandb h5py tqdm pandas scikit-learn
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'
```

- (Not necessary) Only for quantitative evaluation of DM) Install PyTorchEMD by 
```
cd PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```
  
### Trouble Shooting
- if 'RuntimeError: Ninja is required to load C++ extensions' occurs...
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
``` 
- ninja: build stopped: subcommand failed
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
```