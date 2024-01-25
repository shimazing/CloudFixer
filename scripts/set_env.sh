pip install torch==2.0.0 torchvision==0.15.1 # Please install an appropriate version for your environment
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install -r requirements.txt

# point2vec
# python 3.10
pip install lightning lightning-bolts

# PointNeXt
cd classifier/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

cd chamfer_dist/
python setup.py install --user
