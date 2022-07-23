from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import pkg_resources
from transformers import AlbertTokenizer
import math
from cnn_classify import test, CNNClassify, BiLSTMClassify
from lm_lstm import LSTM_LM
import os
import torch
import numpy as np
import sys
sys.path.insert(2,'/home/scalercio/nlp/BARTScore')
from bart_score import BARTScorer

class Evaluator(object):

    def __init__(self):
        resource_package = __name__

        #yelp_acc_path = 'acc_yelp.bin'
        #yelp_ppl_path = 'ppl_yelp.binary'
        yelp_ref0_path = 'cleaned_test_1.txt'
        yelp_ref1_path = 'cleaned_test_0.txt'
        classifier_dir = "pretrained_classifer/shakespeare2"
        classifier_file_name = os.path.join(classifier_dir, "model.pt")
        print("Loading model from '{0}'".format(classifier_file_name))
        self.classifier = torch.load(classifier_file_name)

        lm0_dir = "pretrained_lm/shakespeare_style0"
        lm1_dir = "pretrained_lm/shakespeare_style1"
        lm0_file_name = os.path.join(lm0_dir, "model.pt")
        lm1_file_name = os.path.join(lm1_dir, "model.pt")
        print("Loading language models from '{0}' and {1}".format(lm0_file_name,lm1_file_name))
        self.lm0 = torch.load(lm0_file_name)
        self.lm1 = torch.load(lm1_file_name)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
        
        #yelp_acc_file = pkg_resources.resource_stream(resource_package, yelp_acc_path)
        #yelp_ppl_file = pkg_resources.resource_stream(resource_package, yelp_ppl_path)
        yelp_ref0_file = pkg_resources.resource_stream(resource_package, yelp_ref0_path)
        yelp_ref1_file = pkg_resources.resource_stream(resource_package, yelp_ref1_path)

        
        self.yelp_ref = []
        with open(yelp_ref0_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())
        with open(yelp_ref1_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())
        self.path_to_similarity_script = "/home/scalercio/nlp/style-transfer-paraphrase/style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py"
        self.bart_scorer = BARTScorer(device=torch.device('cuda' if True and torch.cuda.is_available() else 'cpu'), checkpoint='facebook/bart-large-cnn')
        self.bart_scorer.load(path='bart.pth')

    def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [self.tokenizer.tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = self.tokenizer.tokenize(text_transfered.lower().strip().replace("<unk>","_"))
        #print(texts_origin, text_transfered)
        return sentence_bleu(texts_origin, text_transfered) * 100

    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_0(self, texts_neg2pos):
        #assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        if texts_neg2pos[-1] == '':
            n = len(texts_neg2pos)-1
        else:
            n = len(texts_neg2pos)
        print(n)
        for x, y in zip(self.yelp_ref[0], texts_neg2pos):
            #with open('/home/ascalercio/nlp/language-transfer-style-portuguese/deep_yelp_bleu.txt', 'a+') as fl:
            #    print(('{:.4f}').format(self.nltk_bleu([x], y)
            #    ), file=fl)
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_1(self, texts_pos2neg):
        #assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        if texts_pos2neg[-1] == '':
            n = len(texts_pos2neg)-1
        else:
            n = len(texts_pos2neg)
        print(n)
        for x, y in zip(self.yelp_ref[1], texts_pos2neg):
            #print(x,y)
            #print(self.nltk_bleu([x], y))
            #with open('/home/ascalercio/nlp/language-transfer-style-portuguese/deep_yelp_bleu.txt', 'a+') as fl:
            #    print(('{:.4f}').format(self.nltk_bleu([x], y)
            #    ), file=fl)
            sum += self.nltk_bleu([x], y)
        return sum / n

    def ref_similarity_0(self, path_texts_neg2pos, model_name):
        #assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        command = "python " + self.path_to_similarity_script +\
                  " --generated_path " + path_texts_neg2pos   +\
                  " --reference_strs reference --reference_paths ~/nlp/mattes/evaluator/cleaned_test_1.txt " +\
                  "--output_path ~/nlp/mattes/save/" + model_name + "/0to1_sim.txt " +\
                  "--store_scores"
        print(command)
        os.system(command)
        sim_file = np.loadtxt(path_texts_neg2pos + ".pp_scores")
        return sim_file.mean()

    def ref_similarity_1(self, path_texts_pos2neg, model_name):
        command = "python " + self.path_to_similarity_script +\
                  " --generated_path " + path_texts_pos2neg  +\
                  " --reference_strs reference --reference_paths ~/nlp/mattes/evaluator/cleaned_test_0.txt " +\
                  "--output_path ~/nlp/mattes/save/" + model_name + "/1to0_sim.txt " +\
                  "--store_scores"
        print(command)
        os.system(command)
        sim_file = np.loadtxt(path_texts_pos2neg + ".pp_scores")
        return sim_file.mean()

    def ref_bartscore_0(self, texts_neg2pos):
        if texts_neg2pos[-1] == '':
            texts_neg2pos = texts_neg2pos[:-1]
        print(len(texts_neg2pos))
        assert len(texts_neg2pos) == (len(self.yelp_ref[0]))
        self.yelp_ref[0] = [text.lower().strip() for text in self.yelp_ref[0]]
        print(self.yelp_ref[0][-1:])
        print(texts_neg2pos[-1:])
        bart_score = self.bart_scorer.score(texts_neg2pos, self.yelp_ref[0])
        bart_score  = sum(bart_score) / len(bart_score)
        print(bart_score)

        return bart_score

    def ref_bartscore_1(self, texts_pos2neg):
        if texts_pos2neg[-1] == '':
            texts_pos2neg = texts_pos2neg[:-1]
        print(len(texts_pos2neg))
        assert len(texts_pos2neg) == (len(self.yelp_ref[1]))
        self.yelp_ref[1] = [text.lower().strip() for text in self.yelp_ref[1]]
        print(self.yelp_ref[1][-1:])
        print(texts_pos2neg[-1:])
        bart_score = self.bart_scorer.score(texts_pos2neg, self.yelp_ref[1])
        bart_score  = sum(bart_score) / len(bart_score)
        print(bart_score)

        return bart_score

    def yelp_ref_bleu(self, texts_neg2pos, texts_pos2neg):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 1000
        for x, y in zip(self.yelp_ref[0] + self.yelp_ref[1], texts_neg2pos + texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n