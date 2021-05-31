# Reproducing MS MARCO TEK-Representations

Baseline is originally cloned from https://github.com/google-research/language/tree/master/language/tek_representations
and modified.

## Pretrained Models
Create the following directories and set up the variables
```
export model_dir=language/tek_representations/models
export base_dir=$model_dir/pretrained/base/roberta
export large_dir=$model_dir/pretrained/large/roberta
```
### Roberta Base
```
wget -O $base_dir/encoder.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json
wget -O $base_dir/vocab.bpe https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt 
git clone https://huggingface.co/roberta-base  $base_dir
```

### Roberta Large
```
wget -O $large_dir/encoder.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json 
wget -O $large_dir/vocab.bpe https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt 
git clone https://huggingface.co/roberta-large $large_dir
```

Clone BERT Repository in the project directory: 
https://github.com/google-research/bert.git

Get GPT-2 encoder from https://github.com/openai/gpt-2/blob/master/src/encoder.py

### Init Checkpoints

We are using the initial checkpoints for both Roberta base and large provided by Hugging Face.
In order to get tensorflow checkpoints, clone the following repository in the project directory:

`git clone https://github.com/vickyzayats/roberta_tf_ckpt.git`

```
for model in $base_dir,roberta-base $large_dir,roberta-large;do \
IFS=, read dir name <<< '$model'
export config=$dir/config.json 
export ckpt_out=$dir/ckpts
python -m language.tek_representations.roberta_tf_ckpt.convert_pytorch_checkpoint_to_tf --model_name=$name --config_file=$config --tf_cache_dir=$ckpt_out
done
```

Now that the models are all set. We can start the TEK-representations generation steps.
 
## Input Generation

Save the data directory into `data_dir`. Create a key-value store for the background corpus from the linked entities and reformat MS MARCO to
```
{ id: { 'text': text, 
        'wiki_links': [[wikidata_id, entity_start_pos, entity_end_pos], ...]
      },
  ...
}
```
This can be done through the following commands:
```
export input dir= $data_dir/msmarco_linked_entities
export out_dir = $data_dir/reformat_msmarco_linked_entities
python -m language.tek_representations.reformat_input --data_dir $input dir --out_dir $out_dir
```
## Pretraining
```
export pretraining_files=$out_dir/wiki.tfrecord
export background_corpus=$out_dir/background_corpus.json
export input=$(find $out_dir -type f -name '*linked_entities.json' | tr '\n' ',' | sed 's/,$/ /' | tr ' ' '\n')
```
Create pretraining data if key-value store is available
```
for dir in $base_dir $large_dir;do
python -m language.tek_representations.preprocess.create_pretraining_data \
--input_file=$input \
--output_file=$pretraining_files \
--background_corpus_file=$background_corpus \
--vocab_file=$dir \
--max_seq_length=512 \
--max_background_len=128
done
```
To run pretraining, enable `use_tpu` if tpu is available. If multiple gpus are used,
enable `use_gpu` and set `num_gpu_cores`.
```
for model in $base_dir,roberta_base.ckpt $large_dir,roberta_large.ckpt;do
IFS=, read dir ckpt <<< '$model'
python -m language.tek_representations.run_pretraining \
--do_train \
--do_eval \
--bert_config_file=$dir/config.json \
--init_checkpoint=$dir/ckpts/$ckpt \
--input_file=$pretraining_files \
--output_dir=$dir/msl512_mbg128 \
--train_batch_size=8 \
--eval_batch_size=8 \
--learning_rate=5e-05 \
--num_train_steps=200000 \
--save_checkpoints_steps=20000 \
done
```
**Note:** 

1. In the original settings suggested [here](https://arxiv.org/pdf/2004.12006.pdf), `train_batch_size` and `eval_batch_size` are both of 512 which causes OOM because of tensorflow inability to serve on multi-gpu models (**[Known Issue](https://github.com/tensorflow/serving/issues/311)** in tensorflow) . As a result, TPU is needed for larger batch sizes.
2. Make sure that `max_predictions_per_seq` and `max_seq_length` have the same values in `create_pretraining_data.py` and `run_pretraining.py` to avoid this [issue](https://github.com/google-research/bert/issues/75).

## Query-Passage Preprocessing
Convert MS MARCO Passage dataset to MRQA format
```
export mrqa_preprocessed=$data_dir/mrqa/msmarco_preprocessed

python -m language.tek_representations.preprocess.msmarco_to_mrqa \
--data_dir $data_dir/msmarco_linked_entities \
--qrels $data_dir/msmarco_passage \
--out_dir $mrqa_preprocessed
```
Preprocess data for both train and dev splits, and count the features to set the number of steps in the fine-tuning phase.
```
for model in $base_dir,base $large_dir,large;do \
IFS=, read dir name <<< "$model"

python -m language.tek_representations.preprocess.prepare_mrqa_data \
--input_data_dir=$mrqa_preprocessed \
--output_data_dir=$mrqa_preprocessed/$name/type.ngram-msl.512-mbg.128 \ 
--vocab_file=$dir \
--split=dev \
--background_type=ngram \
--corpus_file=$background_corpus \
--is_training=False

python -m language.tek_representations.preprocess.prepare_mrqa_data \
--input_data_dir=$mrqa_preprocessed \ 
--output_data_dir=$mrqa_preprocessed/$name/type.ngram-msl.512-mbg.128 \
--vocab_file=$dir \
--split=train \
--background_type=ngram  \
--corpus_file=$background_corpus \
--is_training=True 

python -m  language.tek_representations.preprocess.count_features \
--output_file=$mrqa_preprocessed/$name/counts.txt \
--preprocessed_dir=$mrqa_preprocessed/$name/type.ngram-msl.512-mbg.128 \
done
```
## Fine-tuning