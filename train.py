import os
import time
import torch
import numpy as np
from torch import nn, optim
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from utils import tensor2text, calc_ppl, idx2onehot, add_noise, word_drop, kd_loss
from cnn_classify import test, CNNClassify, BiLSTMClassify
from lm_lstm import lm_ppl
import random

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

def batch_preprocess(batch, pad_idx, eos_idx, reverse=False):
    batch_0, batch_1, topk_logit0, topk_logit1, topk_index0, topk_index1 = batch
    diff = batch_0.size(1) - batch_1.size(1)
    if diff < 0:
        #pad = torch.full_like(batch_1[:, :-diff], pad_idx)
        pad = torch.full((batch_0.size(0), -diff), pad_idx, dtype = batch_1.dtype, layout=batch_1.layout, device=batch_1.device)
        batch_0 = torch.cat((batch_0, pad), 1)
        #pad_topk = torch.full_like(topk_logit1[:, :-diff], pad_idx)
        pad_topk = torch.full((topk_logit0.size(0), -diff, topk_logit0.size(2)), pad_idx, dtype = topk_logit1.dtype, layout=topk_logit1.layout, device=topk_logit1.device)
        topk_logit0 = torch.cat((topk_logit0, pad_topk), 1)
        topk_index0 = torch.cat((topk_index0, pad_topk.long()), 1)

    elif diff > 0:
        #pad = torch.full_like(batch_0[:, :diff], pad_idx)
        pad = torch.full((batch_1.size(0), diff), pad_idx, dtype = batch_0.dtype, layout=batch_0.layout, device=batch_0.device)
        batch_1 = torch.cat((batch_1, pad), 1)
        #pad_topk = torch.full_like(topk_logit0[:, :diff], pad_idx)
        pad_topk = torch.full((topk_logit1.size(0), diff, topk_logit1.size(2)), pad_idx, dtype = topk_logit0.dtype, layout=topk_logit0.layout, device=topk_logit0.device)
        topk_logit1 = torch.cat((topk_logit1, pad_topk), 1)
        topk_index1 = torch.cat((topk_index1, pad_topk.long()), 1)

    pos_styles = torch.ones_like(batch_0[:, 0])
    neg_styles = torch.zeros_like(batch_1[:, 0])

    if reverse:
        batch_0, batch_1 = batch_1, batch_0
        pos_styles, neg_styles = neg_styles, pos_styles
        
    tokens = torch.cat((batch_0, batch_1), 0)
    lengths = get_lengths(tokens, eos_idx)
    styles = torch.cat((neg_styles, pos_styles), 0)
    topk_logit = torch.cat((topk_logit0, topk_logit1), 0)
    topk_index = torch.cat((topk_index0, topk_index1), 0)


    return tokens, lengths, styles, topk_logit, topk_index

def batch_preprocess_eval(batch, pad_idx, eos_idx, reverse=False):
    batch_0, batch_1 = batch
    diff = batch_0.size(1) - batch_1.size(1)
    if diff < 0:
        pad = torch.full_like(batch_1[:, :-diff], pad_idx)
        batch_0 = torch.cat((batch_0, pad), 1)

    elif diff > 0:
        pad = torch.full_like(batch_0[:, :diff], pad_idx)
        batch_1 = torch.cat((batch_1, pad), 1)

    pos_styles = torch.ones_like(batch_0[:, 0])
    neg_styles = torch.zeros_like(batch_1[:, 0])

    if reverse:
        batch_0, batch_1 = batch_1, batch_0
        pos_styles, neg_styles = neg_styles, pos_styles
        
    tokens = torch.cat((batch_0, batch_1), 0)
    lengths = get_lengths(tokens, eos_idx)
    styles = torch.cat((neg_styles, pos_styles), 0)

    return tokens, lengths, styles     

def d_step(config, data, model_F, model_D, optimizer_D, batch, temperature):
    model_F.eval()
    pad_idx = config.pad_id
    eos_idx = config.eos_id
    vocab_size = len(data.tokenizer)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles, _ , _ = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    with torch.no_grad():
        raw_gen_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            raw_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
        rev_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    
    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    
    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

        

    if config.discriminator_method == 'Multi':
        gold_log_probs = model_D(inp_tokens, inp_lengths)
        gold_labels = raw_styles + 1

        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = raw_styles + 1
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)
    else:
        raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles)
        rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
        raw_gold_labels = torch.ones_like(raw_styles)
        rev_gold_labels = torch.zeros_like(rev_styles)
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

        
        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(raw_styles)
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    
    adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
    adv_labels = torch.cat((gold_labels, gen_labels), 0)
    adv_loss = loss_fn(adv_log_probs, adv_labels)
    assert len(adv_loss.size()) == 1
    adv_loss = adv_loss.sum() / batch_size
    loss = adv_loss
    
    optimizer_D.zero_grad()
    loss.backward()
    clip_grad_norm_(model_D.parameters(), 5)
    optimizer_D.step()

    model_F.train()

    return adv_loss.item()

def f_step(config, data, model_F, model_D, optimizer_F, batch, temperature, drop_decay,
           cyc_rec_enable=True):
    model_D.eval()
    
    pad_idx = config.pad_id
    eos_idx = config.eos_id
    unk_idx = config.unk_id
    vocab_size = len(data.tokenizer)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles, topk_logit, topk_index = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    token_mask = (inp_tokens != pad_idx).float()

    optimizer_F.zero_grad()

    # self reconstruction loss

    noise_inp_tokens = word_drop(
        inp_tokens,
        inp_lengths, 
        config.inp_drop_prob * drop_decay
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_F(
        noise_inp_tokens, 
        inp_tokens, 
        noise_inp_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor
    
    slf_rec_loss.backward()

    # cycle consistency loss

    if not cyc_rec_enable:
        optimizer_F.step()
        model_D.train()
        return slf_rec_loss.item(), 0, 0, inp_lengths.float().mean().item(), 0
    
    gen_log_probs = model_F(
        inp_tokens,
        None,
        inp_lengths,
        rev_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

    cyc_log_probs = model_F(
        gen_soft_tokens,
        inp_tokens,
        gen_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size
    token_mask = token_mask.view(-1)
    if config.albert_kd and config.kd_alpha > 0:
            cyc_log_probs = cyc_log_probs.view(-1,cyc_log_probs.size(2))
            loss_kd = kd_loss(cyc_log_probs, (topk_logit, topk_index),
                              config.kd_temperature, token_mask)
            loss_kd /= batch_size
            cyc_rec_loss = cyc_rec_loss * (1. - config.kd_alpha) + loss_kd * config.kd_alpha

    cyc_rec_loss *= config.cyc_factor

    # style consistency loss

    if config.discriminator_method == 'Multi':
        adv_labels = rev_styles + 1
        adv_log_porbs = model_D(gen_soft_tokens, gen_lengths)
    else:
        adv_labels = torch.ones_like(rev_styles)
        adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)
        
    adv_loss = loss_fn(adv_log_porbs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor
        
    (cyc_rec_loss + adv_loss).backward()
        
    # update parameters
    
    clip_grad_norm_(model_F.parameters(), 5)
    optimizer_F.step()

    model_D.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item(), inp_lengths.float().mean().item(), gen_lengths.float().mean().item()

def mask_word(w, config):
    _w_real = w
    _w_rand = np.random.randint(config.src_vocab_size, size=w.shape)
    _w_mask = np.full(w.shape, config.mask_id)
    probs = torch.multinomial(config.pred_probs, len(_w_real), replacement=True)
    _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
    return _w

def unfold_segments(segs):
    """Unfold the random mask segments, for example:
       The shuffle segment is [2, 0, 0, 2, 0], 
       so the masked segment is like:
       [1, 1, 0, 0, 1, 1, 0]
       [1, 2, 3, 4, 5, 6, 7] (positions)
       (1 means this token will be masked, otherwise not)
       We return the position of the masked tokens like:
       [1, 2, 5, 6]
    """
    pos = []
    curr = 1   # We do not mask the start token
    for l in segs:
        if l >= 1:
            pos.extend([curr + i for i in range(l)])
            curr += l
        else:
            curr += 1
    return np.array(pos)

def shuffle_segments(segs, unmasked_tokens):
    """
    We control 20% mask segment is at the start of sentences
               20% mask segment is at the end   of sentences
               60% mask segment is at random positions,
    """
    p = np.random.random()
    if p >= 0.8:
        shuf_segs = segs[1:] + unmasked_tokens
    elif p >= 0.6:
        shuf_segs = segs[:-1] + unmasked_tokens
    else:
        shuf_segs = segs + unmasked_tokens
    random.shuffle(shuf_segs)
    
    if p >= 0.8:
        shuf_segs = segs[0:1] + shuf_segs
    elif p >= 0.6:
        shuf_segs = shuf_segs + segs[-1:]
    return shuf_segs

def get_segments(mask_len, span_len):
    segs = []
    while mask_len >= span_len:
        segs.append(span_len)
        mask_len -= span_len
    if mask_len != 0:
        segs.append(mask_len)
    return segs

def restricted_mask_sent(x, l, config):
    """ Restricted mask sents
        if span_len is equal to 1, it can be viewed as
        discrete mask;
        if span_len -> inf, it can be viewed as 
        pure sentence mask
    """
    x = x.transpose(0, 1)
    span_len = config.lambda_span
    if span_len <= 0:
        span_len = 1
    max_len = 0
    positions, inputs, targets, outputs, = [], [], [], []
    mask_len = round(len(x[:, 0]) * config.word_mass)
    len2 = [mask_len for i in range(l.size(0))]
    
    unmasked_tokens = [0 for i in range(l[0] - mask_len - 1)]
    segs = get_segments(mask_len, span_len)
    
    for i in range(l.size(0)):
        words = np.array(x[:, i].tolist())
        shuf_segs = shuffle_segments(segs, unmasked_tokens)
        pos_i = unfold_segments(shuf_segs)
        output_i = words[pos_i].copy()
        target_i = words[pos_i - 1].copy()
        words[pos_i] = mask_word(words[pos_i], config)

        inputs.append(words)
        targets.append(target_i)
        outputs.append(output_i)
        positions.append(pos_i - 1)

    x1  = torch.LongTensor(max(l) , l.size(0)).fill_(config.pad_id)
    x2  = torch.LongTensor(mask_len, l.size(0)).fill_(config.pad_id)
    y   = torch.LongTensor(mask_len, l.size(0)).fill_(config.pad_id)
    pos = torch.LongTensor(mask_len, l.size(0)).fill_(config.pad_id)
    l1  = l.clone()
    l2  = torch.LongTensor(len2)
    for i in range(l.size(0)):
        x1[:, i].copy_(torch.LongTensor(inputs[i]))
        x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
        y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
        pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))
    y = torch.cat((x2[:1 , :].clone(), y), 0)
    pred_mask = y != config.pad_id
    #y = y.masked_select(pred_mask)
    return x1, l1, x2, l2, y, pred_mask, pos

def mass_step(config, data, model_F, model_D, optimizer_F, batch, temperature, drop_decay):
    '''assert lambda_coeff >= 0
    if lambda_coeff == 0:
        return
    params = self.params
    self.encoder.train()
    self.decoder.train()
    lang1_id = params.lang2id[lang]
    lang2_id = params.lang2id[lang]
    x_, len_ = self.get_batch('mass', lang)'''
    model_D.eval()
    
    pad_idx = config.pad_id
    eos_idx = config.eos_id
    unk_idx = config.unk_id
    vocab_size = len(data.tokenizer)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles, _ , _ = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    


    (x1, len1, x2, len2, y, pred_mask, positions) = restricted_mask_sent(inp_tokens, inp_lengths, config)
    x1 = x1.transpose(0, 1).to(inp_tokens.device)
    x2 = x2.transpose(0, 1).to(inp_tokens.device)
    y = y.transpose(0, 1).to(inp_tokens.device)
    positions = positions.transpose(0, 1).to(inp_tokens.device)
    len1 = len1.to(inp_tokens.device)
    len2 = len2.to(inp_tokens.device)
    enc_mask = (x1 == config.mask_id).to(inp_tokens.device)
    token_mask = (y != pad_idx).float()
    optimizer_F.zero_grad()

    slf_log_probs = model_F.mass(
        x1,
        len1,
        raw_styles,
        x2,
        len2,
        positions,
        enc_mask,
        generate=False,
        differentiable_decode=False,
        temperature=temperature        
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), y) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor
    
    slf_rec_loss.backward()

    optimizer_F.step()
    model_D.train()
    return slf_rec_loss.item(), 0, 0, token_mask.sum().item()/batch_size, 0
    '''
    langs1 = x1.clone().fill_(lang1_id)
    langs2 = x2.clone().fill_(lang2_id)
    
    x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)
    enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
    enc1 = enc1.transpose(0, 1)
        
    enc_mask = x1.ne(params.mask_index)
    enc_mask = enc_mask.transpose(0, 1)
    
    dec2 = self.decoder('fwd', 
                        x=x2, lengths=len2, langs=langs2, causal=True, 
                        src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)
    
    _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
    self.stats[('MA-%s' % lang)].append(loss.item())
    
    self.optimize(loss, ['encoder', 'decoder'])
    # number of processed sentences / words
    self.n_sentences += params.batch_size
    self.stats['processed_s'] += len2.size(0)
    self.stats['processed_w'] += (len2 - 1).sum().item()
    '''

def eval_step(config, data, model_F, model_D, batch, temperature):
    model_D.eval()
    model_F.eval()
    
    pad_idx = config.pad_id
    eos_idx = config.eos_id
    unk_idx = config.unk_id
    vocab_size = len(data.tokenizer)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess_eval(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    token_mask = (inp_tokens != pad_idx).float()

    # self reconstruction loss

    noise_inp_tokens = inp_tokens
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_F(
        noise_inp_tokens, 
        inp_tokens, 
        noise_inp_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor
    
    # cycle consistency loss
    
    gen_log_probs = model_F(
        inp_tokens,
        None,
        inp_lengths,
        rev_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

    cyc_log_probs = model_F(
        gen_soft_tokens,
        inp_tokens,
        gen_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size

    cyc_rec_loss *= config.cyc_factor

    # style consistency loss

    adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)
    if config.discriminator_method == 'Multi':
        adv_labels = rev_styles + 1
    else:
        adv_labels = torch.ones_like(rev_styles)
    adv_loss = loss_fn(adv_log_porbs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor
        
    model_D.train()
    model_F.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item(), inp_lengths.float().mean().item(), gen_lengths.float().mean().item()

def train(config, data, model_F, model_D):
    optimizer_F = optim.Adam(model_F.parameters(), lr=config.lr_F, weight_decay=config.L2)
    optimizer_D = optim.Adam(model_D.parameters(), lr=config.lr_D, weight_decay=config.L2)

    his_d_adv_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []
    batches_len = []
    batches_gen_len = []
    
    #writer = SummaryWriter(config.log_dir)
    if config.load_ckpt:
        global_step = int(config.d_ckpt.split("/")[-1].split("_")[0])
    else:
        global_step = 0
    model_F.train()
    model_D.train()

    config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
    os.makedirs(config.save_folder + '/ckpts')
    print('Save Path:', config.save_folder)

    if not config.load_ckpt:
        print('Model F pretraining......')
        #for i, batch in enumerate(train_iters):
        for i in range(config.F_pretrain_iter):
            batch, batch_size, eop = data.next_train()
            #print(batch[0], batch[1])
            slf_loss, cyc_loss, _ , batch_len, _ = f_step(config, data, model_F, model_D, optimizer_F, batch, 1.0, 1.0, False)
            his_f_slf_loss.append(slf_loss)
            his_f_cyc_loss.append(cyc_loss)
            batches_len.append(batch_len)

            if (i + 1) % 10 == 0:
                avrg_f_slf_loss = np.mean(his_f_slf_loss)
                avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
                avrg_batches_len = np.mean(batches_len)
                his_f_slf_loss = []
                his_f_cyc_loss = []
                batches_len = []
                print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}, len_batches:{:.2f}'.format(i + 1, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_batches_len))
            #if eop: break
    
    print('Training start......')

    def calc_temperature(temperature_config, step):
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature
    #batch_iters = iter(train_iters)
    while True:
        drop_decay = calc_temperature(config.drop_rate_config, global_step)
        temperature = calc_temperature(config.temperature_config, global_step)
        #batch = next(batch_iters)
        
        for _ in range(config.iter_D):
            batch, batch_size, eop = data.next_train()
            d_adv_loss = d_step(
                config, data, model_F, model_D, optimizer_D, batch, temperature
            )
            his_d_adv_loss.append(d_adv_loss)
            
        for _ in range(config.iter_F):
            batch, batch_size, eop = data.next_train()
            f_slf_loss, f_cyc_loss, f_adv_loss, batch_len , batch_gen_len = f_step(
                config, data, model_F, model_D, optimizer_F, batch, temperature, drop_decay
            )
            his_f_slf_loss.append(f_slf_loss)
            his_f_cyc_loss.append(f_cyc_loss)
            his_f_adv_loss.append(f_adv_loss)
            batches_len.append(batch_len)
            batches_gen_len.append(batch_gen_len)


        
        global_step += 1
        #writer.add_scalar('rec_loss', rec_loss.item(), global_step)
        #writer.add_scalar('loss', loss.item(), global_step)
            
            
        if global_step % config.log_steps == 0:
            avrg_d_adv_loss = np.mean(his_d_adv_loss)
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            avrg_f_adv_loss = np.mean(his_f_adv_loss)
            avrg_batches_len = np.mean(batches_len)
            avrg_batches_gen_len = np.mean(batches_gen_len)
            log_str = '[iter {}] d_adv_loss: {:.4f}  ' + \
                      'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                      'f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f}'
            print(log_str.format(
                global_step, avrg_d_adv_loss,
                avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,
                temperature, config.inp_drop_prob * drop_decay
            ))
            train_log_file = config.save_folder + '/train_log.txt'
            with open(train_log_file, 'a') as fl:
                print(('[iter {}] d_adv_loss: {:.4f}  ' + \
                       'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                       'f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f} ' + \
                       ' len_batches: {:.2f}  len_gen_batches: {:.2f}\n').format(
                    global_step, avrg_d_adv_loss, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss, temperature, config.inp_drop_prob * drop_decay, avrg_batches_len, avrg_batches_gen_len
                ), file=fl)
                
        if global_step % config.eval_steps == 0:
            his_d_adv_loss = []
            his_f_slf_loss = []
            his_f_cyc_loss = []
            his_f_adv_loss = []
            batches_len = []
            batches_gen_len = []
            
            #save model
            torch.save(model_F.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_F.pth')
            torch.save(model_D.state_dict(), config.save_folder + '/ckpts/' + str(global_step) + '_D.pth')
            auto_eval(config, data, model_F, model_D, global_step, temperature)
            #for path, sub_writer in writer.all_writers.items():
            #    sub_writer.flush()

def auto_eval(config, data, model_F, model_D, global_step, temperature):
    model_F.eval()
    vocab_size = len(data.tokenizer)
    eos_idx = config.eos_id

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
    
    gold_text0, raw_output0, rev_output0 = inference(data, 0)
    gold_text1, raw_output1, rev_output1 = inference(data, 1)

    valid_file_0 = os.path.join( config.save_folder + '/ckpts/' , str(global_step) + '_0')
    valid_file_1 = os.path.join( config.save_folder + '/ckpts/' , str(global_step) + '_1')
    out_file_0 = open(valid_file_0, 'w', encoding='utf-8')
    out_file_1 = open(valid_file_1, 'w', encoding='utf-8')
    for i in range(len(rev_output0)):
        line_0 = rev_output0[i].strip()
        out_file_0.write(line_0 + '\n')
        out_file_0.flush()

    for i in range(len(rev_output1)):
        line_1 = rev_output1[i].strip()
        out_file_1.write(line_1 + '\n')
        out_file_1.flush()

    out_file_0.close()
    out_file_1.close()


    evaluator = Evaluator()
    ref_text = evaluator.yelp_ref

    
    #acc_neg = evaluator.yelp_acc_0(rev_output[0])
    acc_mod, _ = test(evaluator.classifier, data, 128, valid_file_0, config.dev_trg_file0, negate = True)
    acc_cla, _ = test(evaluator.classifier, data, 128, valid_file_1, config.dev_trg_file1, negate = True)
    #acc_pos = evaluator.yelp_acc_1(rev_output[1])
    bleu_mod = evaluator.yelp_ref_bleu_0(rev_output0)
    bleu_cla = evaluator.yelp_ref_bleu_1(rev_output1)
    _ , ppl_mod = lm_ppl(evaluator.lm1, data, 128, valid_file_0, config.dev_trg_file0) #evaluator.yelp_ppl(rev_output[0])
    _ , ppl_cla = lm_ppl(evaluator.lm0, data, 128, valid_file_1, config.dev_trg_file1) #evaluator.yelp_ppl(rev_output[1])

    for k in range(5):
        idx = np.random.randint(len(rev_output0))
        print('*' * 20, 'classic sample', '*' * 20)
        print('[gold]', gold_text0[idx])
        print('[raw ]', raw_output0[idx])
        print('[rev ]', rev_output0[idx])
        print('[ref ]', ref_text[0][idx])

    print('*' * 20, '********', '*' * 20)
    

    for k in range(5):
        idx = np.random.randint(len(rev_output1))
        print('*' * 20, 'modern sample', '*' * 20)
        print('[gold]', gold_text1[idx])
        print('[raw ]', raw_output1[idx])
        print('[rev ]', rev_output1[idx])
        print('[ref ]', ref_text[1][idx])

    print('*' * 20, '********', '*' * 20)

    print(('[auto_eval] acc_cla: {:.4f} acc_mod: {:.4f} ' + \
          'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
          'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
              acc_cla, acc_mod, bleu_cla, bleu_mod, ppl_cla, ppl_mod,
    ))
    '''
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []
    batches_len = []
    batches_gen_len = []
    while True:
        batch, batch_size, eop = data.next_dev(dev_batch_size = 16)
        f_slf_loss, f_cyc_loss, f_adv_loss, batch_len , batch_gen_len = eval_step(
            config, data, model_F, model_D, batch, 1)
        his_f_slf_loss.append(f_slf_loss)
        his_f_cyc_loss.append(f_cyc_loss)
        his_f_adv_loss.append(f_adv_loss)
        batches_len.append(batch_len)
        batches_gen_len.append(batch_gen_len)


        if eop: break

    avrg_f_slf_loss = np.mean(his_f_slf_loss)
    avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
    avrg_f_adv_loss = np.mean(his_f_adv_loss)
    avrg_batches_len = np.mean(batches_len)
    avrg_batches_gen_len = np.mean(batches_gen_len)
    '''
    # save output
    save_file = config.save_folder + '/' + str(global_step) + '.txt'
    eval_log_file = config.save_folder + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        print(('iter{:5d}:  acc_cla: {:.4f} acc_mod: {:.4f} ' + \
               'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
               'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
            global_step, acc_cla, acc_mod, bleu_cla, bleu_mod, ppl_cla, ppl_mod
        ), file=fl)
    with open(save_file, 'w') as fw:
        print(('[auto_eval] acc_cla: {:.4f} acc_mod: {:.4f} ' + \
               'bleu_cla: {:.4f} bleu_mod: {:.4f} ' + \
               'ppl_cla: {:.4f} ppl_mod: {:.4f}\n').format(
            acc_cla, acc_mod, bleu_cla, bleu_mod, ppl_cla, ppl_mod,
        ), file=fw)

        for idx in range(len(rev_output0)):
            print('*' * 20, 'classic sample', '*' * 20, file=fw)
            print('[gold]', gold_text0[idx], file=fw)
            print('[raw ]', raw_output0[idx], file=fw)
            print('[rev ]', rev_output0[idx], file=fw)
            print('[ref ]', ref_text[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_output1)):
            print('*' * 20, 'modern sample', '*' * 20, file=fw)
            print('[gold]', gold_text1[idx], file=fw)
            print('[raw ]', raw_output1[idx], file=fw)
            print('[rev ]', rev_output1[idx], file=fw)
            print('[ref ]', ref_text[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

    model_F.train()