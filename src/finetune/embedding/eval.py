import os
import time
import json
from tqdm import tqdm
from typing import Union, List, Tuple
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from FlagEmbedding import FlagModel


def main(
        bge: FlagModel, 
        test_file: Union[str, List[str]], 
        query_bs: int = 512, 
        query_max_len: int = 128, 
        candi_bs: int = 512, 
        candi_max_len: int = 128, 
        print_error: bool = False
        ) -> 'Tuple[float, float, float, int]':
    """
    Args:
        bge: FlagModel, 嵌入模型对象
        test_file: Union[str, list[str]], 测试文件路径，支持单个文件路径或文件路径列表
        query_bs: int, query 的批量大小，默认为512
        query_max_len: int, query 的最大长度，默认为128
        candi_bs: int, 候选语句的批量大小，默认为512
        candi_max_len: int, 候选语句的最大长度，默认为128
        print_error: bool, 是否打印错误样本，默认为False
    
    Returns:
        tuple[float, float, float, int]: 返回四个指标值，分别为： top 1 准确率，top 3 准确率，top 5 准确率，测试集大小
    
    """
    if isinstance(test_file, list):
        ds = []
        for file in test_file:
            with open(file, 'r', encoding='utf-8') as f:
                ds += [json.loads(line) for line in f]
    else:
        with open(test_file, 'r', encoding='utf-8') as f:
            ds = []
            for line in f:
                try:
                    ds.append(json.loads(line))
                except:
                    print(line)
                    break

    # 添加候选语句
    candidates = list(set([item["pos"][0] for item in ds]))   # 从正例
    for d in ds:   # 从负例
        if d["neg"] != []:
            candidates += d["neg"]
    candidates = list(set(candidates))
    print('The number of candidates: ', len(candidates))

    start = time.time()
    # 嵌入候选语句
    p_embeddings = bge.encode(candidates, candi_bs, candi_max_len)
    # 嵌入 query
    queries = [item['query'] for item in ds]
    q_embeddings = bge.encode_queries(queries, query_bs, query_max_len)
    end = time.time()
    print('Embedding time cost: ', end - start)

    # 推理评估
    cnt_top_1, cnt_top_3, cnt_top_5 = 0, 0, 0
    for i, item in tqdm(enumerate(ds)):
        q_embedding = q_embeddings[i]
        scores = q_embedding @ p_embeddings.T
        scores_with_id = sorted([(i, s) for i, s in enumerate(scores)], reverse=True, key=lambda x: x[1])
        
        # top 5
        if item['pos'][0] in [candidates[s[0]] for s in scores_with_id[: 5]]:
            cnt_top_5 += 1
        else:
            if print_error:
                print("top5 error: ", item, [candidates[s[0]] for s in scores_with_id[: 5]])
        # top 3
        if item['pos'][0] in [candidates[s[0]] for s in scores_with_id[: 3]]:
            cnt_top_3 += 1
        else:
            if print_error:
                print("top3 error: ", item, [candidates[s[0]] for s in scores_with_id[: 3]])
       # top 1
        if item['pos'][0] == candidates[scores_with_id[0][0]]:
            cnt_top_1 += 1
        else:
            if print_error:
                print("top1 error: ", item, candidates[scores_with_id[0][0]])

    print("top1 acc:", cnt_top_1 / len(ds), ", top3 acc: ", cnt_top_3 / len(ds), ", top5 acc: ", cnt_top_5 / len(ds))
    return cnt_top_1, cnt_top_3, cnt_top_5, len(ds)


if __name__ == '__main__':
    # 测试集文件路径
    test_file = './data/test/query_chunk.jsonl'
    # 加载模型
    bge = FlagModel('../models/checkpoint_2024_04_27_17_47_55',
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    
    main(bge=bge, test_file=test_file, print_error=False)
