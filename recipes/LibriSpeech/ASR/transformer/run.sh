python -m torch.distributed.launch --nproc_per_node=4 train_1_distribute.py ./hparams/benchmark1/IntrlvFormer1_noseg.yaml --data_folder /home/ec2-user/experiment/LibriSpeech --auto_mix_prec --distributed_launch --distributed_backend='nccl'