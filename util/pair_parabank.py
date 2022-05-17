import sys
from random import random
"""
Choose sentence-pairs based on the bidirectional model scores and sentence length.

Args:
    sys.argv[1]: input file of parabank2
    sys.argv[2]: output file for source sentences
    sys.argv[3]: output fiel for target sentences
"""

with open(sys.argv[1],'r') as f0, \
     open(sys.argv[2],'w') as f1, \
     open(sys.argv[3],'w') as f2, \
     open(sys.argv[4],'w') as f3, \
     open(sys.argv[5],'w') as f4:
    for line in f0.readlines():
        line = line.strip().split('\t')
        if len(line)<3:
            continue
        len0 = len(line[1].split())
        len1 = len(line[2].split())
        if 5<len0 and len0<50 and 5<len1 and len1<50:
            if (random()<=0.8):
                f1.write(line[1].strip()+'\n')
                f2.write(line[2].strip()+'\n')
            else:
                f3.write(line[1].strip()+'\n')
                f4.write(line[2].strip()+'\n')

