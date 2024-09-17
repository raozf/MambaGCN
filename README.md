## Dataset

Download [ModelNet40](https://modelnet.cs.princeton.edu) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) [435M].

## Model Training

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name=dgcnn --model=dgcnn
