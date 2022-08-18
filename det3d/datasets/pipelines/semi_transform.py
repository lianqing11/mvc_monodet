from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import copy
@PIPELINES.register_module()
class SplitSourceTarget(object):
    def __init__(self, source_index=0, target_index=1):
        # if -1 random select from the image:
        self.source_index = source_index
        self.target_index = target_index
    
    def __call__(self, input_dict):
        target_img = copy.deepcopy(input_dict['img'])
        # target_img.data = target_img.data[:,self.target_index]
        # target_img_meta = input_dict['img']
        # img.data = img.data[:,self.source_index]
        # # source_img_meta = 
        
        # input_dict['target_img'] = target_img
        
        
        # import pdb; pdb.set_trace()