

params0={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/yelp/cleaned_train_0.txt",
  "train_trg_file": "data/yelp/train_0.attr",
  "dev_src_file": "data/yelp/cleaned_dev_0.txt",
  "dev_trg_file": "data/yelp/dev_0.attr",
  "trg_vocab": "data/yelp/attr.vocab"
}


params1={
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/yelp/cleaned_train_1.txt",
  "train_trg_file": "data/yelp/train_1.attr",
  "dev_src_file": "data/yelp/cleaned_dev_1.txt",
  "dev_trg_file": "data/yelp/dev_1.attr",
  "trg_vocab": "data/yelp/attr.vocab"
}


params_main={
  "lm_style0":"models/lm0-large-yelp/",
  "lm_style1":"models/lm1-large-yelp/",
  "eval_cls": True
}
