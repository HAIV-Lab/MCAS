# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = '/new_data/xz/osr_closed_set_all_you_need-main/'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = '/new_data/xz/osr_closed_set_all_you_need-main/save'        # directory to store experiment output (checkpoints, logs, etc)
save_dir =  '/new_data/xz/osr_closed_set_all_you_need-main/save'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = '/new_data/xz/osr_closed_set_all_you_need-main/save/xz/osr_closed_set_all_you_need-main/log/{}/arpl_models/{}/checkpoints/model_best_Softmax.pth'
root_criterion_path = '/new_data/xz/osr_closed_set_all_you_need-main/save/xz/osr_closed_set_all_you_need-main/log/{}/arpl_models/{}/checkpoints/model_best_Softmax_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
                                
cub_root = "/new_data/xz/OSR/CUB_200_2011/"                                                  # CUB
aircraft_root = '/new_data/xz/OSR/aircraft/fgvc-aircraft-2013b'                      # FGVC-Aircraft                    
car_root = "/new_data/xz/OSR/scar/cars_{}/"        # Stanford Cars
meta_default_path = "/new_data/xz/OSR/scar/devkit/cars_{}.mat"  
awa_root = "/new_data/xz/OSR/AWA/"
lad_root = "/new_data/xz/OSR/LAD/"
                

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir ="/new_data/xz/osr_closed_set_all_you_need-main/data/open_set_splits/"

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = '/work/sagar/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar'
places_moco_path = '/work/sagar/pretrained_models/places/moco_v2_places.pth'
places_supervised_path = '/work/sagar/pretrained_models/places/supervised_places.pth'
imagenet_supervised_path = '/work/sagar/pretrained_models/imagenet/supervised_imagenet.pth'