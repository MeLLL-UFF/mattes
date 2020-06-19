import argparse
import torch
import os

def init_args():
  parser = argparse.ArgumentParser(description='Clean up the sequences')
  parser.add_argument('--seed', type=int, default=77777, metavar='S', help='random seed')
  parser.add_argument('--data_path', type=str, default="", help='data path')
  parser.add_argument("--train", action="store_true", 
    help="whether to save a train file")
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
            .replace(" :", ":")
            .replace(" ;", ";")
         for line in f.read().splitlines()]
    
    train =  args.train
    path, file = os.path.split(args.data_path)

    with open(path + "/cleaned_" + file, "w") as f:
        for line in lines:
            f.write(line+"\n")
