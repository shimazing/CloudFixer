# pip install torch==2.0.0 torchvision==0.15.1 # Please install an appropriate version for your environment
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" #refer https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# pip install git+'https://github.com/otaheri/chamfer_distance'
# pip install -r requirements.txt

# point2vec
# python 3.10
pip install lightning lightning-bolts

# pointMLP
git clone git@github.com:ma-xu/pointMLP-pytorch.git
cd pointMLP-pytorch
pip install pointnet2_ops_lib/.

# PointNeXt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html #change torch-2.0.1+cu117.html according to the proper version

cd classifier/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

cd chamfer_dist/
python setup.py install --user

