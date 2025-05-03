#                  (HyDE)
#               _____↑↓______
#              |             |
#      query ->|  Retriever  |-> ···
#              |_____________|
#               ↑↓         ↑↓
#           embedding  VectorDatabase

import re
import requests
import pandas as pd
from tqdm import tqdm 
import concurrent.futures
from collections import defaultdict
from typing import List, Union, Dict

from logger import logger
from llm_chat_hf import LLM
from template import HYDE_TEMPLATE, API_LIST, GEN_QUERY, NL2PANDAS, IMAGE_GEN
from VectorDatabase import VectorDatabase
from embedding import Embedding, RerankEmbedding, BM25


class APIModel(LLM):
    api_list = API_LIST
    def __init__(self, template: str, api_name: str = None, model_path: str = None, agent_id: str = None):
        assert bool(api_name) ^ bool(model_path), "`api_name` or `model_path` must have one of them."
        super().__init__()
        self.template = template
        self.api_name = api_name
        self.model_path = model_path
        self.agent_id = agent_id
        self.api_config = self.api_list.get(api_name)
        self.model_gen = self._get_model_gen()
    
    def _get_model_gen(self) -> str:
        if self.api_config:
            model = lambda **kwargs: requests.post(
                self.api_config["url"], 
                headers=self.api_config["headers"], 
                json={"messages": [{"role": "user", "content": self.template.format(**kwargs)}], 
                      "model": self.api_name if self.agent_id is None else self.agent_id, "stream": False}
                ).json()["choices"][0]["message"]["content"]
            return model
        else:
            model = self.from_pretrained(self.model_path)
            return model.chat_completion
        
    def __call__(self, **kwargs) -> str:
        try:
            return self.model_gen(**kwargs)
        except Exception as e:
            logger.error(str(e))
            return None


class HyDE(APIModel):
    def __init__(self, api_name: str, **kwargs):
        super().__init__(HYDE_TEMPLATE, api_name, **kwargs)
    
    def __call__(self, query: str) -> str:
        try:
            return self.model_gen(query=query)
        except Exception as e:
            logger.error(str(e))
            return query


class Route:
    def __init__(self,
                 topk: int = 3,
                 embedding: Embedding = None, 
                 db: VectorDatabase = None, 
                 bm25: BM25 = None,
                 hyde: HyDE = None
                 ):
        self.topk = topk
        self.embedding = embedding
        self.bm25 = bm25
        self.db = db
        self.hyde = hyde
        assert bool(self.embedding) ^ bool(self.bm25), "You must select one of dense or bm25."
        assert (self.db and self.embedding) or self.embedding == None, "You must provide a database and a embedding."
    
    def __call__(self, query: str) -> List[int]:
        logger.info(f"Route running ... HyDE: {bool(self.hyde)}, Embedding: {'dense' if bool(self.embedding) else 'bm25'}, DB: {self.db.db_name if self.db else None}({self.db.amount if self.db else None})")
        # 1. HyDE or not
        if self.hyde:
            dummy_query = self.hyde(query).strip().split("搜索结果来自：")[0]
            logger.info(f"HyDE({self.hyde.api_name if self.hyde.api_name else self.hyde.model_path}): {query} --> {dummy_query}")
            query = dummy_query
        
        # 2. embedding or bm25
        if self.embedding:
            query_embedding = self.embedding(query, is_query=False if self.hyde else True)
            # 3. db: long or short
            return self.db(query_embedding, topk=self.topk)
        else:
            # 3. bm25: long or short
            return self.bm25([query], k=self.topk)



class Retriever:
    def __init__(self, 
                 route: Union[Route, List[Route]], 
                 reranker: RerankEmbedding = None, 
                 weights: List[float] = None):
        self.route = route if isinstance(route, list) else [route]
        self.n_routes = len(self.route)
        self.reranker = reranker
        self.weights = weights
        assert self.n_routes == 1 ^ bool(self.weights) or self.reranker, \
            "If routes > 1, you must provide weights. Or if routes = 1, you don't need to provide weights."
        assert bool(self.reranker) ^ bool(self.weights), "You must select one of reranker or weights."
        assert self.weights is None or (len(self.weights) == self.n_routes), "The length of weights must be equal to the number of routes."

    def __call__(self, 
                 query: str,
                 sent_list_list: List[List[str]],
                 topk: int = 3, 
                 lost_in_the_middle: bool = True,
                 return_lists: bool = False
                 ) -> Union[List[str], List[int]]:
        assert len(sent_list_list) == self.n_routes, "The length of `sent_list_list` must be equal to the number of routes."
        
        # 1. run multi-routes
        logger.info(f"Retriever is beginning, there are {self.n_routes} routes.")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交任务时记录索引
            indexed_results = {executor.submit(route, query): i for i, route in enumerate(self.route)}

            lists = [None] * len(self.route)
            for future in concurrent.futures.as_completed(indexed_results):
                index = indexed_results[future]
                result = future.result()
                lists[index] = result[0] if isinstance(result, list) else result.tolist()[0]

        if return_lists:
            return lists
        
        # 2. merge results
        logger.info(f"Retriever is merging results, merge mode: {'reranker' if self.reranker else 'weights'}.")
        if self.reranker:
            sentences = set()
            for i in range(self.n_routes):
                for idx in lists[i]:
                    sentences.add(sent_list_list[i][idx])
            sentences = list(sentences)
            segment_list = [sentences[idx] for idx in self.reranker(query, sentences, topk=topk)]
        else:
            sentence_score: Dict[str, float] = defaultdict(float)
            for i in range(self.n_routes):
                for idx in lists[i]:
                    # sent_score = route weight * (1 / rank)
                    sentence_score[sent_list_list[i][idx]] += self.weights[i] * 1 / (idx + 1)
            segment_list = [sent for sent, _ in sorted(sentence_score.items(), key=lambda x: x[1], reverse=True)][: topk] 
            
        # 3. Lost in the middle 
        if lost_in_the_middle:
            segment_list_order = [""] * len(segment_list)
            i, j = 0, 0
            while "" in segment_list_order:
                segment_list_order[i] = segment_list[j]
                if j+1 >= len(segment_list): break
                segment_list_order[-(i+1)] = segment_list[j+1]
                i += 1
                j += 2
            return segment_list_order
        else:
            return segment_list


class DummyQuery(APIModel):
    def __init__(self, sentences: List[str], api_name: str, embedding: Embedding, db_path: str = None, **kwargs) -> VectorDatabase:
        super().__init__(GEN_QUERY, api_name, **kwargs)
        self.sentences = sentences
        self.embedding = embedding
        
        if db_path:
            self.db = VectorDatabase.load(db_path)
            return self.db
        
        self.db = VectorDatabase(1024, db_name="dummy_query")
        self.sentence_queries: List[List[str]] = self._generate_queries()
        self.queries_embeddings = self.embedding.create_embedding(self.sentence_queries, max_len=128)
        self.db.create_index(self.queries_embeddings)
        self.db.save()
        return self.db
    
    def _generate_queries(self) -> List[List[str]]:
        responses = [self.__call__(context=sent) for sent in tqdm(self.sentences, desc="Generating queries")]
        return [line.split('. ', 1)[1] if '. ' in line else line for line in responses.splitlines()]


class SheetRetriever(APIModel):
    def __init__(self, api_name: str = "glm", agent_id: str = "6627bb82e931facbc8e093c9", **kwargs):
        super().__init__(NL2PANDAS, api_name,  agent_id=agent_id, **kwargs)
    
    def __call__(self, query: str, sheet: 'SheetDocument', n_lines: int = 15, return_code: bool = False) -> pd.DataFrame:
        md_code =  super().__call__(desc=sheet.desc, query=query)
        pd_code = self._markdown_to_plain(md_code)
        try:
            pd_result = eval(pd_code.replace("df", "sheet.df"))
        except:
            pd_result = "*语句执行错误，请重新提问生成！*"
        if type(pd_result) == pd.DataFrame:
            pd_str = pd_result.to_markdown(index=False)
            pd_str = f"根据您的提问生成了以下pandas语句：\n  ```python\n{pd_code}\n```\n  返回结果前{n_lines}行如下：\n  " + "\n".join(pd_str.split('\n')[: n_lines]) + f"  \n\n`共 {pd_result.shape[0]} 行 {pd_result.shape[1]} 列`"
        else:
            pd_str = f"根据您的提问生成了以下pandas语句：\n  ```python\n{pd_code}\n```\n  返回结果如下：  \n{pd_result}"
        if return_code:
            return pd_str, pd_code
        else:
            return pd_str

    
    def _markdown_to_plain(self, code_block) -> str:
        lines = code_block.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:-1]
        return '\n'.join(lines)
    
    
class ChartRetriever(APIModel):
    def __init__(self, api_name: str = "qwen1half-14b-chat", **kwargs):
        super().__init__(IMAGE_GEN, api_name, **kwargs)
    
    def __call__(self, query: str, desc: str) -> str:
        codes =  super().__call__(query=query, desc=desc)
        codes = re.findall(r'```.*?```', codes, re.DOTALL)
        codes = "" if codes == None else codes[0]
        codes = codes.replace("plt.show()", """plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.savefig('./tmp/figure.png')""").replace("```python\n", "").replace("\n```", "")
        logger.info(f"the chart code: {codes}")
        try:
            exec(codes)
            desc += "  \n![img](./tmp/figure.png)"
        except Exception as e:
            logger.error(f"生成图片失败！{str(e)}")
            try:
                codes =  super().__call__(query=query, desc=desc + f"请根据报错信息{str(e)}，修改以下代码：\n\n{codes}")
                codes = re.findall(r'```.*?```', codes, re.DOTALL)
                codes = "" if codes == None else codes[0]
                codes = codes.replace("plt.show()", """plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig('./tmp/figure.png')""").replace("```python\n", "").replace("\n```", "")
                logger.info(f"the retry chart code: {codes}")
                exec(codes)
                desc += "  \n![img](./tmp/figure.png)"
            except Exception as e:
                desc += f"  \n*已尝试两次生成图片均失败！{str(e)}*"
        return desc
        


if __name__ == "__main__":
    # hyde = HyDE(api_name="concise")
    # print(hyde("太阳从哪里升起？"))
    
    # sentences = []
    # from document import SheetDocument
    # sheet = SheetDocument.read_sheet("xxx.xlsx", 1)
    # sr = SheetRetriever()
    # query = "毕业于西北工业大学的男生有哪些"
    # sr(query, sheet)

    cr = ChartRetriever()
    print(cr("""请根据2023年的四个季度报告，分析招商瑞丰混合发起式在各个报告期末的境内股票投资中，对制造业的股票投资占基金资产净值的比例变化，并以季报时间为横轴、比例为纵轴绘制折线统计图。""", 
       """根据招商瑞丰混合发起式2023年的……"""))
    