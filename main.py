import argparse
import time

from train import *
from validate import * 
from ensemble import *


def parse_args():

  parser = argparse.ArgumentParser(description="AGIC-PART A", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--batchsize", default=8, type=int, dest="batchsize") 
  parser.add_argument("--epochs", default=1000, type=int, dest="epochs")
  parser.add_argument("--data_dir", default="./data/ROI/", type=str, dest="data_dir") 
  parser.add_argument("--trainmode", default='train', type=str, dest="trainmode")
  parser.add_argument("--fold_size", default=10, type=int, dest="fold_size")
  parser.add_argument("--prepare_data_dir", default='./data', type=str, dest="prepare_data_dir")
  parser.add_argument("--augmentation", default=3, type=int, dest="augmentation")
  parser.add_argument("--answer", default=1, type=int, dest="answer")
  
  return parser.parse_args()


def main():
  args = parse_args()
  # fold_val_set = train(args)
  # validate(fold_val_set)
  ensemble(args)
  

if __name__ == '__main__':
  since = time.time()
  main()
  elapsed_time = time.time() - since
  print("Elapsed time: {0:.0f}m {1:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
