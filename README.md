# NER-unlabeled-data-retrieval
[COLING 22] Domain-Specific NER via Retrieving Correlated Samples.

arxiv [http://arxiv.org/abs/2208.12995](http://arxiv.org/abs/2208.12995)

## Usage

This code depends on the [AllenNLP](https://github.com/allenai/allennlp) library, see `requirements.txt`.

To train a baseline NEZHA-BiLSTM-CRF model on the address dataset: 
`python main.py --config=add --cuda=0 --name=RUN_NAME` .

To train a baseline Cross-Encoder model on the address dataset: 
`python main.py --config=add-ret --cuda=0 --name=RUN_NAME` .

The above `add` and `add-ret` correspond to the filename in the `config/` dictionary. They could be replaced with `eco` and `eco-ret` to run the e-commerce dataset experiments.


Notes:
1. `plm_dir` and `plm_name` are used to set the path of pretrained models from huggingface or local filepath, they can be set to `plm_dir=""` and `plm_name="bert-base-chinese"` to load a Chinese BERT from huggingface.


## Data

The address domain dataset is comes from [CCKS2021中文地址要素解析数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=109339).

The e-commerce domain dataset is comes from [MultiDigraphNER](https://github.com/PhantomGrapes/MultiDigraphNER/tree/master/data/ecommerce).

For some reasons, we can not provide the testset of the address domain (not the `final_test.txt` in [CCKS2021中文地址要素解析数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=109339)), and all retrieved correlated texts. (I really hope I can...)

All experiments are conducted at Alibaba Damo Academy, the results in the paper are real.


## Cite

```
@inproceedings{zhang-etal-2022-domain,
title = "Domain-Specific NER via Retrieving Correlated Samples",
author = "Zhang, Xin  and
    Yong, Jiang  and
    Wang, Xiaobin  and
    Hu, Xuming  and
    Sun, Yueheng  and
    Xie, Pengjun  and
    Zhang, Meishan",
booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
month = oct,
year = "2022",
address = "Gyeongju, Republic of Korea",
publisher = "International Committee on Computational Linguistics"
}
```
