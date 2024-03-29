# Masked Transformer through Knowledge Distillation for Unsupervised Text Style Tranfer

This repository contains the code for the study `Masked Transformer through Knowledge Distillation for Unsupervised Text Style Tranfer`



## Requirements

* Python 3
* Pytorch == 1.7

A list of other required python packages is in the `requirements.txt` file.

You can install everything on your environment with `pip`:
```
pip3 install -r requirements.txt
```

## Setting Up
For training our model, it's necessary a pre-trained masked languagel. More specifically, we adopted [Albert](https://arxiv.org/abs/1909.11942) for all experiments.

1. Preprocessing

    Run the following command for preprocess our training data. You must specify dataset in the script file.
    ```
    bash scripts/prepare_deen.sh
    ```

2. Extract teacher soft label

    We first precompute hidden states (logits) of MLM teacher, for each domain, to speedup KD training ant then pre-compute the top-K logits to save memory

        ```
        # extract hidden states of teacher (domain 0)
        python dump_teacher_hiddens.py \
            --output ~/nlp/mattes/data/targets/teacher0 \
            --ckpt ~/nlp/mattes/masked_lm/lm0-large/pytorch_model.bin \
            --db ~/nlp/mattes/data/dump/NEGA.db

        # extract top-k logits (domain 0)
        python dump_teacher_topk.py --topk 64 --bert_hidden ~/nlp/mattes/data/targets/teacher0

        # extract hidden states of teacher (domain 1)
        python dump_teacher_hiddens.py \
            --output ~/nlp/mattes/data/targets/teacher1 \
            --ckpt ~/nlp/mattes/masked_lm/lm1-large/pytorch_model.bin \
            --db ~/nlp/mattes/data/dump/POSI.db

        # extract top-k logits (domain 1)
        python dump_teacher_topk.py --topk 64 --bert_hidden ~/nlp/mattes/data/targets/teacher1
        ```


## Usage

The hyperparameters for the Masked Transformer can be found in ''main.py''.

For our work, the most important are listed below:

```
    data_path : the path of the datasets
    save_path = where to save the checkpoing
    
    discriminator_method : the type of discriminator ('Multi' or 'Cond'). We used always 'Multi'
    max_length : the maximun sentence length 
    embed_size : the dimention of the token embedding
    d_model : the dimention of Transformer d_model parameter
    h : the number of Transformer attention head
    num_layers : the number of Transformer layer
    batch_size : the training batch size
    lr_F : the learning rate for the Style Transformer
    lr_D : the learning rate for the discriminator
    iter_D : the number of the discriminator update step pre training interation
    iter_F : the number of the Style Transformer update step pre training interation
    dropout : the dropout factor for the whole model

    log_steps : the number of steps to log model info
    eval_steps : the number of steps to evaluate model info

    slf_factor : the weight factor for the self reconstruction loss
    cyc_factor : the weight factor for the cycle reconstruction loss
    adv_factor : the weight factor for the style controlling loss
```

You can adjust them in the Config class from the ''main.py''.



To train the model, use the command:

```shell
python main.py
```

You can train from a checkpoit using `load_ckpt = True` and indicating the previously trained models in the `d_ckpt` and `f_ckpt` parameters.


To evaluate the model, we used a pre-trained convolutional classifier, the NLTK python package and recurrent neural language models to evaluate the style control, content preservation and fluency respectively. The evaluation related files for the Yelp and the Shakespeate datasets are placed in the ''evaluator'' folder.


## Outputs

Outputs generated by our best model, as well as the metrics achived by then on the test sets are located in the "save" folder.