import numpy as np
import sys
import time
import subprocess
import os
import os.path as osp
from tools.manager import GPUManager
from tools.manager import repeat_mission, add_extra_arg, run_in_local, run_in_slurm
from tools.manager import check_available
import random
from optparse import OptionParser
from datetime import datetime
import shutil
import re
# from keras.layers import LSTM

if __name__ == "__main__":
    os.system("nvidia-smi")

    parser = OptionParser()
    parser.add_option('-i', dest='cmd_file', default=None)
    # parser.add_option("--dir", default="../semi_supervised")
    # parser.add_option('-t', dest='waiting_seconds', default=None)
    parser.add_option('-g', dest="available_gpu", default="")
    parser.add_option('-s', dest="save_dir", default="")
    parser.add_option('--ceph', action="store_true", default=False)
    parser.add_option('--tacc', action="store_true", default=False)
    parser.add_option('--select', default=None)
    parser.add_option('--repeat_time', default=1, type=int)
    parser.add_option('--extra_arg', default=None, type=str)
    parser.add_option('--slurm', default=None, type="str")
    parser.add_option('--slurm_partion', default="mm_det", type=str)

    (options, args) = parser.parse_args()
    # assume the sup dir is the running dir
    print("is tacc: ", options.tacc)

    sup_dir = os.path.dirname(options.cmd_file).replace("/slurm_env","").replace("/exps", "")
    sup_dir = sup_dir.replace("run_logs/", "")

    os.chdir(sup_dir)
    print(f"change dir: {sup_dir}")
    mission_queue = open(options.cmd_file).readlines()
    if options.select is not None:
        first, last = options.select.split(":")
        mission_queue = mission_queue[int(first): int(last)]

    if options.extra_arg is not None:
        mission_queue = add_extra_arg(mission_queue, options.extra_arg)
    if options.repeat_time > 1:
        mission_queue = repeat_mission(mission_queue, options.repeat_time)




    mission_queue = [i.replace("\n", "").strip() for i in mission_queue]
    mission_queue = [i for i in mission_queue if len(i) > 0]



    if options.slurm is None:
        run_in_local(mission_queue, sup_dir, options)
    else:
        run_in_slurm(mission_queue, sup_dir, options)

    # return None
