import torch
import time
from data_utils_yelp import DataUtil
from models import StyleTransformer, Discriminator
from train import train, auto_eval
from cnn_classify import test, CNNClassify, BiLSTMClassify
from lm_lstm import LSTM_LM


class Config():
    train_src_file0 = 'data/shakespeare/cleaned_train_0.txt'
    train_src_file1 = 'data/shakespeare/cleaned_train_1.txt'
    train_trg_file = 'data/shakespeare/train.attr'
    dev_src_file0 = 'data/shakespeare/cleaned_dev_0.txt'
    dev_src_file1 = 'data/shakespeare/cleaned_dev_1.txt'
    dev_trg_file = 'data/shakespeare/dev.attr'
    dev_trg_file0 = 'data/shakespeare/dev_0.attr'
    dev_trg_file1 = 'data/shakespeare/dev_1.attr'
    dev_trg_ref = 'data/shakespeare/cleaned_dev_ref.txt'
    trg_vocab  = 'data/shakespeare/attr.vocab'
    data_path = './data/shakespeare/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 64
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 32
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500#57500
    log_steps = 5
    eval_steps = 100
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(0.2, 0), (0.2, 115), (0.3, 5750)]
    temperature_config = [(1, 0), (1, 575), (0.8, 2300), (0.6, 5750)]

    slf_factor = 0.15
    cyc_factor = 0.3
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 1
    decode = False
    #max_len = 10000
    lambda_span = 10000
    word_mass = 0.5
    word_mask = 0.8
    word_keep = 0.1
    word_rand = 0.1
    albert_kd = True
    kd_alpha = 0.5
    kd_temperature = 10
    bert_dump0 = 'data/targets/teacher0'
    bert_dump1 = 'data/targets/teacher1'
    load_ckpt = False
    d_ckpt = 'save/Apr12011239/ckpts/1200_D.pth'
    f_ckpt = 'save/Apr12011239/ckpts/1200_F.pth'


def main():
    config = Config()
    data = DataUtil(config)
    print('Vocab size:', config.src_vocab_size)
    if config.load_ckpt:
        state_dict = torch.load(config.f_ckpt)
        model_F = StyleTransformer(config, data).to(config.device)
        model_F.load_state_dict(state_dict)
        state_dict = torch.load(config.d_ckpt)
        model_D = Discriminator(config, data).to(config.device)
        model_D.load_state_dict(state_dict)
    else:
        model_F = StyleTransformer(config, data).to(config.device)
        model_D = Discriminator(config, data).to(config.device)
    print(config.discriminator_method)
    
    train(config, data, model_F, model_D)
    

if __name__ == '__main__':
    main()
