import argparse
import torch
import os
import numpy as np

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
    lines=[]
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.split(" ")[0]:
                lines.append(line.split(" ")[-11:])
                #print(len(line.split(" ")[-11:]))
                #print(line.split(" ")[-11:])

    np_line = np.array(lines)
    print(np_line.shape)
    #print(np_line.size)
    ##print(np_line[:,3])
    np_line = np.concatenate((np_line[:,0:1], np_line[:, 2:3], np_line[:, 4:5], np_line[:, 6:7]), axis = 1)
    np_float = np_line.astype(np.float)
    print(np_float)
    train =  args.train
    path, file = os.path.split(args.data_path)
    np.savetxt("Nov25030636" + "log.csv" ,np_float, delimiter = ",")

    
    

    '''
    with open(path + "/cleaned_" + file, "w") as f:
        for line in lines:
            f.write(line+"\n")
    '''