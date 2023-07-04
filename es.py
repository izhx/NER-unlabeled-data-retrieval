"""
code for elasticsearch
"""

from typing import Dict, Iterable, List
import os
import json
import time
import pickle
import requests


def printf(*args):
    print(time.asctime(), "-", os.getpid(), ":", *args)


def pickle_dump(obj, path):
    with open(path, mode="wb") as file:
        pickle.dump(obj, file)
        printf("dump", file.name)


def pickle_load(path):
    with open(path, mode="rb") as file:
        obj = pickle.load(file)
        printf("load", file.name)
    return obj


def jsonline_iter(path) -> Iterable[Dict]:
    with open(path) as file:
        for line in file:
            obj = json.loads(line)
            if obj:
                yield obj


def dump_jsonline(path: str, data: List):
    with open(path, mode='w') as file:
        for r in data:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")
    printf("dump", path)


def add_raw_copus(source="dedump"):
    """
    批量添加
    """
    config = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "standard_address": {
                        "tokenizer": "standard",
                        "filter": [ "lowercase", "keep_tokens" ]
                    }
                },
                "filter": {
                    "keep_tokens": {
                        "type": "keep_types",
                        "types": [ "<IDEOGRAPHIC>", "<ALPHANUM>" ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "standard_address"},
                "prov": {"type": "keyword"},
                "rid": {"type": "integer", "index": False},
            }
        }
    }
    # 创建索引
    res = requests.put(f"{ES}/{INDEX}", json=config)
    if res.status_code != 200:
        print(json.dumps(res.json(), indent=2))
        raise RuntimeError("failed to create index mapping!")

    # 自己预处理过的缓存文件
    import glob
    paths = glob.glob(f'files/raw/{source}/*.txt')

    def post_batch(batch: List):
        content = "\n".join(batch) + "\n"
        res = requests.post(url, data=content.encode("utf-8"), headers=HEADERS)
        if res.status_code != 200:
            print(res.json())
            failures.append((p, i, batch_size))
        batch.clear()

    url = f"{ES}/{INDEX}/_bulk"  # address 索引
    failures = list()
    batch_size = 10000
    for p in paths:
        printf(p)
        timer = time.time()
        with open(p) as file:  # 'files/raw/dedump/西藏.txt'
            batch = list()
            for i, line in enumerate(file):
                prov, city, dist, town, part, rid = line.strip().split(('\t'))
                text = ''.join(''.join((prov, city, dist, town, part)).split())
                # action
                batch.append('{"index":{}}')
                # 对应的 data
                batch.append(json.dumps(
                    {'prov': prov, 'text': text, 'rid': rid}, ensure_ascii=False
                ))
                if len(batch) >= batch_size:
                    post_batch(batch)
            else:
                if len(batch) > 0:
                    post_batch(batch)
            timer = time.time() - timer
            printf("time", timer, 'num', i + 1, 'from', p)
            timer = time.time()

    if len(failures) > 0:
        pickle_dump(failures, "files/fail.pkl")
    else:
        printf("success")


def query_body(text, prov=None, size=100):
    if prov is None:
        return {"size": size, "query": {"match": {"text": text}}}

    query = {
        "must": {"match": {"text": text}},
        "filter": {"term": {"prov": prov}}
    }
    return {"size": size, "query": {"bool": query}}


def search_one(text, prov=None, size=100):
    """ 默认 BM25 检索 """
    url = f"{ES}/{INDEX}/_search"  # 或者
    response = requests.get(url, json=query_body(text, prov, size))
    assert response.status_code == 200
    hits = response.json()['hits']['hits']
    array = [(h['_source']['text'], h['_score']) for h in hits]
    return array


def search_batch(tuples, size=100):
    """ 默认 BM25 检索 """
    url = f"{ES}/{INDEX}/_msearch"
    content = ""
    for text, prov in tuples:
        content += "{}\n"
        content += json.dumps(query_body(text, prov, size), ensure_ascii=False) + "\n"
    response = requests.get(url, data=content.encode("utf-8"), headers=HEADERS)
    assert response.status_code == 200
    results = list()
    for one in response.json()["responses"]:
        array = [(h['_source']['text'], h['_score']) for h in one['hits']['hits']]
        results.append(array)
    return results


def retrieval(path, batch_size=1000, to=""):
    from process_raw import PROVS
    # path = f"files/pesudo/{part}.json"
    instances = list(jsonline_iter(path))
    printf(path, len(instances))
    # data = list()

    def add_batch(batch: List, ids: List):
        timer = time.time()
        array = search_batch(batch, 101)
        for i, r in zip(ids, array):
            r = r[1:]
            # obj = dict(text=instances[i]['hits'][-1], hits=[i[0] for i in r], bm25=[i[1] for i in r])
            # data.append(obj)
            instances[i]['hits'] = [i[0] for i in r]
            instances[i]['bm25'] = [i[1] for i in r]
        timer = time.time() - timer
        printf(ids[0], '-', ids[-1], 'seconds:', timer)

    batch, ids = list(), list()
    for i, ins in enumerate(instances):
        text = ins['text']
        # text = ins['hits'][-1]
        if text[:2] in PROVS:
            prov = text[:2]
        elif text[:3] in PROVS:
            prov = text[:3]
        else:
            prov = None
        batch.append((text, prov))
        ids.append(i)
        if len(ids) >= batch_size:
            add_batch(batch, ids)
            batch, ids = list(), list()
    else:
        if len(batch) > 0:
            add_batch(batch, ids)

    dump_jsonline(f"{path}.retrieval-{to}.jsonl", instances)
    # dump_jsonline(path, instances)


def main(name='pesudo'):
    r = requests.get(ES)
    assert r.status_code == 200 and "tagline" in r.json()
    # add_raw_copus(path_to_raw_data)  # 批量添加地址
    # r1 = search_one("浙江省嘉兴市海宁市许村镇许巷王安桥")
    # print(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
    # for p in ('train', 'dev', 'test'):

    r = search_one("广东深圳市南山华侨城", None, 10)
    for t, s in r:
        print(t)
    # retrieval('path to train data', to=name)
    return


if __name__ == "__main__":
    ES = "http://localhost:9200"  # localhost:9200/<index>/[<id>]
    HEADERS = {"content-type": "application/json;charset=UTF-8"}
    INDEX = "address"
    main()

    # with open("files/retrieval-full/all.txt", mode='w') as file:
    #     for part in ('dev', 'test', 'train'):
    #         print(part)
    #         for ins in jsonline_iter(f"files/retrieval-full/{part}.json"):
    #             for text, _ in ins['matches']:
    #                 file.write(text + '\n')
    #     print(file.name)

"""
安装: https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.16.1-darwin-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.16.1-darwin-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.16.1-darwin-x86_64.tar.gz.sha512 
tar -xzf elasticsearch-7.16.1-darwin-x86_64.tar.gz
cd elasticsearch-7.16.1/
nohup ./bin/elasticsearch >> /dev/null 2>&1 &

"""
