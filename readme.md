# Learning Visual-Semantic Hierarchical Attribute Space for Interpretable Open-Set Recognition

The code repository for "Zhuo Xu and Xiang Xiang: Learning Visual-Semantic Hierarchical Attribute Space for Interpretable Open-set Recognition. In WACV, 2025" in PyTorch.

## Train
Please run the following commands:
```
python osr_hyre.py --att_choose_min=0.1 --att_choose_max=0.9 --batch_size=32 --image_size=448 --use_attribute=True --att_file="cub_100_200_select150att.pkl" --use_default_attribute=False --model='rs50' --dataset='cub'
```
## Evaluate
```
Please run the following commands for evaluation:
python test.py --dataset='cub'--att_choose_min=0.1 --att_choose_max=0.9 --image_size=448 --use_attribute=True --use_default_attribute=False --model='rs50' --att_file='cub_100_200_select150att.pkl'  --exp_id=cub_150att
```
This code is build upon [osr_closed_set_all_you_need](https://github.com/sgvaze/osr_closed_set_all_you_need), thanks for their wonderful work!
