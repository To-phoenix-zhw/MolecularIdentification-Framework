#! /bin/bash
python -u run.py --searchtimes 1 --mode case  --checkpoint_path checkpoints/almodel.pt  --pri true  --test_path ./data/dataset/testing-set  --num_iter 40
