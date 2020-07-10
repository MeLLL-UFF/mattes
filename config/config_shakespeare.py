

params0={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 500,
  "batch_size": 32,
  "dropout_in": 0.5,
  "dropout_out": 0.5,
  "train_src_file": "data/shakespeare/cleaned_train_0.txt",
  "train_trg_file": "data/shakespeare/train_0.attr",
  "dev_src_file": "data/shakespeare/cleaned_dev_0.txt",
  "dev_trg_file": "data/shakespeare/dev_0.attr",
  "trg_vocab": "data/shakespeare/attr.vocab"
}


params1={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 500,
  "batch_size": 32,
  "dropout_in": 0.5,
  "dropout_out": 0.5,
  "train_src_file": "data/shakespeare/cleaned_train_1.txt",
  "train_trg_file": "data/shakespeare/train_1.attr",
  "dev_src_file": "data/shakespeare/cleaned_dev_1.txt",
  "dev_trg_file": "data/shakespeare/dev_1.attr",
  "trg_vocab": "data/shakespeare/attr.vocab"
}


params_main={
  "lm_style0":"models/lm0-large/",
  "lm_style1":"models/lm1-large1/",
  "eval_cls": True
}
