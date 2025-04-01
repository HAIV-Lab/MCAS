import numpy as np
def get_super_class_targets(hyre_dict_name_index,thre):
    super_mask = {}
    for k,v in hyre_dict_name_index.items():
        if len(v) == 1:
            mask = []
        else:
            mask = []
            super_attribute=np.array(v)
            ave=np.mean(super_attribute,axis=0)
            for idx in range(len(ave)):
                if ave[idx]>thre:
                    mask.append(idx)
        super_mask[k] = mask
    return super_mask