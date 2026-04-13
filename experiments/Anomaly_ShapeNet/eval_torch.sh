export PYTHONPATH=../../:$PYTHONPATH
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/zzh/MC3D-AD/open3d_deps_pack"
CUDA_VISIBLE_DEVICES=$2 python -m torch.distributed.launch --nproc_per_node=$1 --master_port 15002 --use_env ../../tools/train_val.py -e
