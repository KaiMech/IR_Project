## Dense Passage Retrieval

### Generating Embeddings

The embedding generation script can be executed via the command line using a certain set of predefined arguments. These are:
- <b>--finetuned</b>: if this keyword is included, a finetuned model is used. If you want to use pretrained models, this arguments can be dropped.
- <b>--finetuned_model_path</b>: if --finetuned is used, a path to a finetuned model needs to be specified. In this project, the finetuned model paths are dpr_finetuned_models/{context, question}_encoder and dpr_finetuned_models_new_settings/{context, question}_encoder. These directories need to contain a config.json file and a model.safetensors file which fit the DPR models.
- <b>--use_query_encoder</b>: Include this argument if you want to use the question encoder model. If it is not included, the context encoder will be used. <b>Important note:</b> if you want to use a query encoder with finetuning but the finetuned model directory contains files for a different model, e.g., the context encoder, an error will occur!
- <b>--corpus_path</b>: path to the dataset you want to encode. The standard directory for this project ../data/tot25/subsets/*. If you want to use other files, you have to ensure that the provided file is a *.jsonl file.
- <b>--target_path</b>: path at which the final resulting embeddings will be stored. <b>Important note:</b> The embedding script generates multiple files, so it is best to provide an empty directory, so that you do not lose track of your files.

Here is an example invocation which uses a finetuned question encoder to generate embeddings for the dev1 queries of the eval20 subset:


```bash
python generate_embeddings.py --finetuned --finetuned_model_path dpr_finetuned_models/question_encoder --use_query_encoder --corpus_path ../data/tot25/subsets/eval20/eval20-queries-dev1.jsonl --target_path . --device cpu
```

Here is an example invocation which uses a pretrained context encoder to generate embeddings for the train80 corpus:
```bash
python generate_embeddings.py -corpus_path ../data/tot25/subsets/train80/train100k-corpus.jsonl --target_path data/train80/ --device cuda
```
All query embeddings are provided in the directory *DPR/data*. The corpus embeddings have to be regenerated.

### Evaluation
After embeddings are generated, they can be evaluated using the evaluation script. Although arbitrary embedding files might be used, you should pass embeddings for contexts and queries to get a meaningful result. It follows a description of parameters which need to be provided to run the script. The notes on directory structure will become much clearer when you actually try out the embedding generation procedure.
- <b>--ctx_embs_path</b>: path to the document embeddings. <b>Important note: context embeddings need to be stored in a directory called *shard01*/. If they are not, evaluation will not work. If you have many embedding files and you want to split them over multiple shards for memory efficiency, simply create multiple of such shard directories named *shard01*, *shard02* etc. in your context embedding directory</b>.
- <b>--query_embs_path</b>: path to the query embeddings. <b>Important note: this needs to be a path to a directory which directly contains the *.npy and *.txt files created during embedding generation, no shards!</b>
- <b>--topk</b>: indicates how many results will be retrieved for each query. Standard value for this task: $k=1000$
- <b>--results_path</b>: path at which the generated runfile will be stored. The suggested structure for runfile storage can be seen below.

Here, you can see some examples:
```bash
python evaluate.py --ctx_embs_path DPR/data/train80/embeddings/train80_embs_corpus/ --query_embs_path DPR/data/train80/embeddings/train80_embs_train_queries/ --topk 1000 --results_path DPR/runs/pretrained/train80/train.txt
```
```
python evaluate.py --ctx_embs_path DPR/data/train80/finetuned_new/train80_embs_corpus_ft/ --query_embs_path DPR/data/train80/finetuned_new/train80_embs_dev3_queries/ --topk 1000 --results_path DPR/runs/finetuned_new/train80/dev3.txt
```
```
python evaluate.py --ctx_embs_path DPR/data/eval20/embeddings_finetuned/eval20_embs_corpus_ft/ --query_embs_path DPR/data/eval20/embeddings_finetuned/eval20_embs_dev1_queries/ --topk 1000 --results_path DPR/runs/finetuned/eval20/dev1.txt
```

The following directory structure is suggested for storing runfiles, i.e., the parameter *--results_path* should be chosen accordingly.<br>
<b>If you want to use the automatic runfile evaluation script `evaluate_runfiles.sh` you have to follow this structure precisely!</b>
```
DPR/
├── runs/
    ├── finetuned/
            ├──train80/
            └──eval20/
    |── finetuned_new/
            ├──train80/
            └──eval20/
    └── pretrained/
            ├──train80/
            └──eval20/

```