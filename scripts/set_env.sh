pip install torch==2.0.0 torchvision==0.15.1 # please install an appropriate version for your environment
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # please refer to https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install -r requirements.txt

# for Point2Vec
pip install lightning lightning-bolts

# for PointMLP
git clone git@github.com:ma-xu/pointMLP-pytorch.git
cd pointMLP-pytorch
pip install pointnet2_ops_lib/.

# for PointNeXt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html # please change torch-2.0.1+cu117.html to an appropriate version for your environment
cd utils/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../chamfer_dist
python setup.py install --user

# for PointMAE
cd ./utils/chamfer_dist
python setup.py install --user
cd ../extensions/emd
python setup.py install --user
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl