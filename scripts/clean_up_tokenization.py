import argparse
import torch
import os

def init_args():
  parser = argparse.ArgumentParser(description='Clean up the sequences')
  parser.add_argument('--seed', type=int, default=77777, metavar='S', help='random seed')
  parser.add_argument('--data_path', type=str, default="", help='data path')
  args = parser.parse_args()
  args.cuda = torch.cuda.is_available()
  return args


if __name__ == "__main__":
    args = init_args()
    print(args.data_path)
    with open(args.data_path, encoding="utf-8") as f:
        lines = [line.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
         for line in f.read().splitlines()]
    
    path, file = os.path.split(args.data_path)
    with open(path + "/train_cleaned.txt", "w") as f:
        for line in lines:
            f.write(line+"\n")
