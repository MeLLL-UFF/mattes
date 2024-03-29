import torch
import time
from data_utils_yelp import DataUtil
from models import StyleTransformer
from train import get_lengths
from cnn_classify import test, CNNClassify, BiLSTMClassify
from lm_lstm import LSTM_LM, lm_ppl
import os
from utils import tensor2text, list2text
from evaluator import EvaluatorGyafc
import numpy as np


class Config():
    train_src_file0 = 'data/gyafc/cleaned_train_0.txt'
    train_src_file1 = 'data/gyafc/cleaned_train_1.txt'
    train_trg_file = 'data/gyafc/train.attr'
    dev_src_file0 = 'data/gyafc/cleaned_test_0.txt'
    dev_src_file1 = 'data/gyafc/cleaned_test_1.txt'
    dev_trg_file = 'data/gyafc/dev.attr'
    dev_trg_file0 = 'data/gyafc/dev_0.attr'
    dev_trg_file1 = 'data/gyafc/dev_1.attr'
    dev_trg_ref = 'data/gyafc/cleaned_dev_ref.txt'
    trg_vocab  = 'data/gyafc/attr.vocab'
    data_path = './data/gyafc/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 32
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500#57500
    log_steps = 5
    eval_steps = 200
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(0.3, 0), (0.4, 230), (0.5, 11500)]
    temperature_config = [(1, 0), (1, 1150), (0.8, 4600), (0.6, 11500)]

    slf_factor = 0.1
    cyc_factor = 0.2
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
    albert_kd = False
    kd_alpha = 0.5
    kd_temperature = 5
    bert_dump0 = 'data/targets/teacher0'
    bert_dump1 = 'data/targets/teacher1'
    translate = True
    ckpt = 'save/Apr03055150/ckpts/1150_F.pth'
    model_name = 'best-no-para-and-kd-hm1-Apr03055150-1150testset-4ref'
    beam_size = 1
    valid_file_0 = False#'save/Mar31141244/ckpts/300_0'#'baseline_outputs/gyafc/styins/0to1'    
    valid_file_1 = False#'save/Mar31141244/ckpts/300_1'#'baseline_outputs/gyafc/styins/1to0'
    paraphrase = False
    direct_paraphrase = False

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

def auto_eval(config, data, model_F, model_name, temperature=1):
    if model_F:
        model_F.eval()
    vocab_size = len(data.tokenizer)
    eos_idx = config.eos_id
    config.save_folder = config.save_path + '/' + model_name
    os.makedirs(config.save_folder)
    print('Save Path:', config.save_folder)

    def inference(data, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        while True:
            if raw_style == 0:
                inp_tokens, _ , eop = data.next_dev0(dev_batch_size = 128, sort = False)
            else:
                inp_tokens, _ , eop = data.next_dev1(dev_batch_size = 128, sort = False)

            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(data, inp_tokens.cpu())
            raw_output += tensor2text(data, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(data, rev_log_probs.argmax(-1).cpu())
            if eop: break

        return gold_text, raw_output, rev_output

    def inference_direct_paraphrase(data, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        while True:
            if raw_style == 0:
                inp_tokens, _ , eop = data.next_dev0(dev_batch_size = 128, sort = False)
            else:
                inp_tokens, _ , eop = data.next_dev1(dev_batch_size = 128, sort = False)

            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            para_styles = torch.ones_like(raw_styles) * 2
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    para_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(data, inp_tokens.cpu())
            raw_output += tensor2text(data, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(data, rev_log_probs.argmax(-1).cpu())
            if eop: break

        return gold_text, raw_output, rev_output
    
    def inference_paraphrase(data, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        while True:
            if raw_style == 0:
                inp_tokens, _ , eop = data.next_dev0(dev_batch_size = 128, sort = False)
            else:
                inp_tokens, _ , eop = data.next_dev1(dev_batch_size = 128, sort = False)

            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
            para_styles = torch.ones_like(rev_styles) * 2
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                gen_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    para_styles,
                    generate=True,
                    differentiable_decode=True,
                    temperature=temperature,
                )

                gen_soft_tokens = gen_log_probs.exp()
                gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

                rev_log_probs = model_F(
                    gen_soft_tokens,
                    inp_tokens,
                    gen_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(data, inp_tokens.cpu())
            raw_output += tensor2text(data, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(data, rev_log_probs.argmax(-1).cpu())
            if eop: break

        return gold_text, raw_output, rev_output

    if config.translate == True:
        if config.direct_paraphrase:
            gold_text0, raw_output0, rev_output0 = inference_direct_paraphrase(data, 0)
            gold_text1, raw_output1, rev_output1 = inference_direct_paraphrase(data, 1)
        elif config.paraphrase == True:
            gold_text0, raw_output0, rev_output0 = inference_paraphrase(data, 0)
            gold_text1, raw_output1, rev_output1 = inference_paraphrase(data, 1)
        else:
            gold_text0, raw_output0, rev_output0 = inference(data, 0)
            gold_text1, raw_output1, rev_output1 = inference(data, 1)
        gold_text = (gold_text0, gold_text1)
        raw_output = (raw_output0,raw_output1)
        rev_output = (rev_output0, rev_output1)

        valid_file_0 = os.path.join( config.save_folder + '/', str(model_name) + '_0')
        valid_file_1 = os.path.join( config.save_folder + '/', str(model_name) + '_1')
        out_file_0 = open(valid_file_0, 'w', encoding='utf-8')
        out_file_1 = open(valid_file_1, 'w', encoding='utf-8')
        for i in range(len(rev_output[0])):
            line_0 = rev_output[0][i].strip()
            out_file_0.write(line_0 + '\n')
            out_file_0.flush()
        for i in range(len(rev_output[1])):
            line_1 = rev_output[1][i].strip()
            out_file_1.write(line_1 + '\n')
            out_file_1.flush()
        out_file_0.close()
        out_file_1.close()
    else:
        rev_output = []
        valid_file_0 = config.valid_file_0
        valid_file_1 = config.valid_file_1
        with open(valid_file_0, 'r', encoding='utf-8') as f:
            src_lines_0 = f.read().split('\n')
        rev_output.append(src_lines_0)

        with open(valid_file_1, 'r', encoding='utf-8') as f:
            src_lines_1 = f.read().split('\n')
        rev_output.append(src_lines_1)



    evaluator = EvaluatorGyafc()
    ref_text = evaluator.yelp_ref

    
    #acc_neg = evaluator.yelp_acc_0(rev_output[0])
    acc_mod, _ = test(evaluator.classifier, data, 128, valid_file_0, config.dev_trg_file0, negate = True)
    acc_cla, _ = test(evaluator.classifier, data, 128, valid_file_1, config.dev_trg_file1, negate = True)
    #acc_pos = evaluator.yelp_acc_1(rev_output[1])
    bleu_mod = evaluator.yelp_ref_bleu_0(rev_output[0])
    bleu_cla = evaluator.yelp_ref_bleu_1(rev_output[1])
    _ , ppl_mod = 0, 0#lm_ppl(evaluator.lm1, data, 128, valid_file_0, config.dev_trg_file0) #evaluator.yelp_ppl(rev_output[0])
    _ , ppl_cla = 0, 0#lm_ppl(evaluator.lm0, data, 128, valid_file_1, config.dev_trg_file1) #evaluator.yelp_ppl(rev_output[1])
    sim_mod = 0#evaluator.ref_similarity_0(valid_file_0, str(model_name))
    sim_cla = 0#evaluator.ref_similarity_1(valid_file_1, str(model_name))
    bartscore_mod = 0#evaluator.ref_bartscore_0(rev_output[0])
    bartscore_cla = 0#evaluator.ref_bartscore_1(rev_output[1])

    print(('[auto_eval] acc_cla: {:.4f} acc_mod: {:.4f} ' + \
          'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
          'sim_cla: {:.4f} sim_mod: {:.4f} ' + \
          'bartscore_cla: {:.4f} bartscore_mod: {:.4f} ' + \
          'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
              acc_cla, acc_mod, bleu_cla, bleu_mod, sim_cla, sim_mod, bartscore_cla, bartscore_mod, ppl_cla, ppl_mod
    ))

    # save output
    
    eval_log_file = config.save_folder + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        print(('{:18s}:  acc_cla: {:.4f} acc_mod: {:.4f} ' + \
               'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
               'sim_cla: {:.4f} sim_mod: {:.4f} ' + \
               'bartscore_cla: {:.4f} bartscore_mod: {:.4f} ' + \
               'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
            model_name, acc_cla, acc_mod, bleu_cla, bleu_mod, sim_cla, sim_mod, bartscore_cla, bartscore_mod, ppl_cla, ppl_mod
        ), file=fl)

    if config.translate == True:
        save_file = config.save_folder + '/' + str(model_name) + '.txt'
        with open(save_file, 'w') as fw:
            print(('[auto_eval] acc_cla: {:.4f} acc_mod: {:.4f} ' + \
                'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
                'sim_cla: {:.4f} sim_mod: {:.4f} ' + \
                'bartscore_cla: {:.4f} bartscore_mod: {:.4f} ' + \
                'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
                acc_cla, acc_mod, bleu_cla, bleu_mod, sim_cla, sim_mod, bartscore_cla, bartscore_mod, ppl_cla, ppl_mod
            ), file=fw)

            for idx in range(len(rev_output[0])):
                print('*' * 20, 'classic sample', '*' * 20, file=fw)
                print('[gold]', gold_text[0][idx], file=fw)
                print('[raw ]', raw_output[0][idx], file=fw)
                print('[rev ]', rev_output[0][idx], file=fw)
                print('[ref ]', ref_text[0][idx], file=fw)

            print('*' * 20, '********', '*' * 20, file=fw)

            for idx in range(len(rev_output[1])):
                print('*' * 20, 'modern sample', '*' * 20, file=fw)
                print('[gold]', gold_text[1][idx], file=fw)
                print('[raw ]', raw_output[1][idx], file=fw)
                print('[rev ]', rev_output[1][idx], file=fw)
                print('[ref ]', ref_text[1][idx], file=fw)

            print('*' * 20, '********', '*' * 20, file=fw)

def beam_eval(config, data, model_F, model_name, temperature=1):
    model_F.eval()
    vocab_size = len(data.tokenizer)
    eos_idx = config.eos_id
    config.save_folder = config.save_path + '/' + model_name
    os.makedirs(config.save_folder)
    print('Save Path:', config.save_folder)

    def beam_inference(data, raw_style):
        gold_text = []
        rev_output = []
        while True:
            if raw_style == 0:
                inp_tokens, _ , eop = data.next_dev0(dev_batch_size = 256, sort = False)
            else:
                inp_tokens, _ , eop = data.next_dev1(dev_batch_size = 256, sort = False)

            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles

            hyps = []
            batch_size = inp_tokens.size(0)

            for i in range(batch_size):
                x = inp_tokens[i,:].unsqueeze(0)
                inp_length = inp_lengths[i].unsqueeze(0)
                rev_style = rev_styles[i].unsqueeze(0)     
                if config.paraphrase:
                    para_style = torch.ones_like(rev_style)*2
                    gen_log_probs = model_F(
                        x,
                        None,
                        inp_length,
                        para_style,
                        generate=True,
                        differentiable_decode=True,
                        temperature=1,
                    )
                    gen_soft_tokens = gen_log_probs.exp()
                    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), config.eos_id)
                    hyp = model_F.translate_sent(gen_soft_tokens, gen_lengths, rev_style, temperature, max_len=config.max_length, beam_size=config.beam_size, poly_norm_m=1)[0]

                else:
                    hyp = model_F.translate_sent(x, inp_length, rev_style, temperature, max_len=config.max_length, beam_size=config.beam_size, poly_norm_m=1)[0]
                hyps.append(hyp.y)
                
            gold_text += tensor2text(data, inp_tokens.cpu())
            rev_output += list2text(data, hyps)
            if eop: break

        return gold_text, rev_output
    
    gold_text0, rev_output0 = beam_inference(data, 0)
    gold_text1, rev_output1 = beam_inference(data, 1)
    gold_text = (gold_text0, gold_text1)
    rev_output = (rev_output0, rev_output1)

    valid_file_0 = os.path.join( config.save_folder + '/' , str(model_name) + '_0')
    valid_file_1 = os.path.join( config.save_folder + '/' , str(model_name) + '_1')
    out_file_0 = open(valid_file_0, 'w', encoding='utf-8')
    out_file_1 = open(valid_file_1, 'w', encoding='utf-8')
    for i in range(len(rev_output[0])):
        line_0 = rev_output[0][i].strip()
        out_file_0.write(line_0 + '\n')
        out_file_0.flush()
    for i in range(len(rev_output[1])):
        line_1 = rev_output[1][i].strip()
        out_file_1.write(line_1 + '\n')
        out_file_1.flush()
    out_file_0.close()
    out_file_1.close()


    evaluator = EvaluatorGyafc()
    ref_text = evaluator.yelp_ref

    
    #acc_neg = evaluator.yelp_acc_0(rev_output[0])
    acc_mod, _ = test(evaluator.classifier, data, 128, valid_file_0, config.dev_trg_file0, negate = True)
    acc_cla, _ = test(evaluator.classifier, data, 128, valid_file_1, config.dev_trg_file1, negate = True)
    #acc_pos = evaluator.yelp_acc_1(rev_output[1])
    bleu_mod = evaluator.yelp_ref_bleu_0(rev_output[0])
    bleu_cla = evaluator.yelp_ref_bleu_1(rev_output[1])
    _ , ppl_mod = 0, 0#lm_ppl(evaluator.lm1, data, 128, valid_file_0, config.dev_trg_file0) #evaluator.yelp_ppl(rev_output[0])
    _ , ppl_cla = 0, 0#lm_ppl(evaluator.lm0, data, 128, valid_file_1, config.dev_trg_file1) #evaluator.yelp_ppl(rev_output[1])
    sim_mod = 0#evaluator.ref_similarity_0(valid_file_0, str(model_name))
    sim_cla = 0#evaluator.ref_similarity_1(valid_file_1, str(model_name))
    bartscore_mod = 0#evaluator.ref_bartscore_0(rev_output[0])
    bartscore_cla = 0#evaluator.ref_bartscore_1(rev_output[1])

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'classic sample', '*' * 20)
        print('[gold]', gold_text[0][idx])
        print('[rev ]', rev_output[0][idx])
        print('[ref ]', ref_text[0][idx])

    print('*' * 20, '********', '*' * 20)
    

    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'modern sample', '*' * 20)
        print('[gold]', gold_text[1][idx])
        print('[rev ]', rev_output[1][idx])
        print('[ref ]', ref_text[1][idx])

    print('*' * 20, '********', '*' * 20)

    print(('[auto_eval] acc_cla: {:.4f} acc_mod: {:.4f} ' + \
          'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
          'sim_cla: {:.4f} sim_mod: {:.4f} ' + \
          'bartscore_cla: {:.4f} bartscore_mod: {:.4f} ' + \
          'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
              acc_cla, acc_mod, bleu_cla, bleu_mod, sim_cla, sim_mod, bartscore_cla, bartscore_mod, ppl_cla, ppl_mod
    ))

    # save output
    eval_log_file = config.save_folder + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        print(('iter{:18s}:  acc_cla: {:.4f} acc_mod: {:.4f} ' + \
               'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
               'sim_cla: {:.4f} sim_mod: {:.4f} ' + \
               'bartscore_cla: {:.4f} bartscore_mod: {:.4f} ' + \
               'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
            model_name, acc_cla, acc_mod, bleu_cla, bleu_mod, sim_cla, sim_mod, bartscore_cla, bartscore_mod, ppl_cla, ppl_mod
        ), file=fl)

def main():
    config = Config()
    data = DataUtil(config)
    print('Vocab size:', config.src_vocab_size)
    if config.valid_file_0:
        assert config.translate == False
        assert config.beam_size == 1
        model_F = None
    else:        
        state_dict = torch.load(config.ckpt)
        model_F = StyleTransformer(config, data).to(config.device)
        model_F.load_state_dict(state_dict)
    
    if config.beam_size == 1:
        auto_eval(config, data, model_F, config.model_name)
    else:
        beam_eval(config, data, model_F, config.model_name)
    

if __name__ == '__main__':
    main()
