import argparse

def TrainOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    return parser

