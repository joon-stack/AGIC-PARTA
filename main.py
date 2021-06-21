import argparse
import time

from train import *

def parse_args():

  parser = argparse.ArgumentParser(description="AGIC-PART A", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--batchsize", default=25, type=int, dest="batchsize") 
  parser.add_argument("--epochs", default=50, type=int, dest="epochs")
  parser.add_argument("--train_dir", default="./Dataset-rps-bg/training/", type=str, dest="train_dir") 
  parser.add_argument("--val_dir", default="./Dataset-rps-bg/rps-test-set/", type=str, dest="val_dir")
  
  return parser.parse_args()


def main():
  args = parse_args()
  # pretrain()
  train(args)

if __name__ == '__main__':
  since = time.time()
  main()
  elapsed_time = time.time() - since
  print("Elapsed time: {0:.0f}m {1:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
