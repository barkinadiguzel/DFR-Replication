import torch.nn as nn

from src.layers.resize_align import resize_align
from src.layers.aggregation import aggregate
from src.layers.concatenation import concatenate


class MultiScaleFeatureGenerator(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, features):
        aligned = []

        target_size = features[0].shape[-2:] 

        for f in features:
            f_resized = resize_align(f, target_size)    
            f_agg = aggregate(f_resized, self.stride)    
            aligned.append(f_agg)

        f_out = concatenate(aligned) 
        return f_out
