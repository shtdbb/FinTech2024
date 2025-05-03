import numpy
import pickle
import jieba
import requests
from logger import logger
from template import API_LIST
from datetime import datetime
from rank_bm25 import BM25Okapi
from typing import List, Union
from FlagEmbedding import FlagModel, BGEM3FlagModel, FlagReranker, LayerWiseFlagLLMReranker, FlagLLMReranker


class Embedding:
    def __init__(self):
        self.sent_embeddings = None
    
    @classmethod
    def load_model(cls, model_path: str = 'models/bge-large-zh-v1.5') -> object:
        logger.info(f"Loading embedding model: {model_path} ...")
        cls.model = FlagModel(model_path, 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
        return cls()
    
    
    def create_embedding(self, sentences: List[str], batch_size: int = 512, max_len: int = 512) -> numpy.ndarray:
        """创建句子的 embedding

        Args:
            sentences (List[str]): 句子列表

        Returns:
            numpy.Ndarray: 句子的 embedding, (n_sentences, n_hiddens)
        """
        
        logger.info(f"Embedding documents ...")
        sentence_embeddings = self.model.encode(sentences, batch_size, max_len)
        
        self.sent_embeddings = sentence_embeddings
        logger.info(f"The shape of embeddings is {sentence_embeddings.shape}.")
        return sentence_embeddings
    
    
    def __call__(self, query: Union[str, List[str]], is_query: bool = True) -> numpy.ndarray:
        """embedding 查询

        Args:
            query (str/List[str]): 提问句子

        Returns:
            numpy.ndarray: 提问句子的 embedding, (1, n_hiddens) / (n, n_hiddens)
        """
        if is_query:
            return self.model.encode_queries(query, 8, 128)
        else:
            return self.model.encode(query, 8, 512)


    def save(self, path: str) -> None:
        """保存向量

        Args:
            path (str): 向量保存路径
        """
        
        pickle.dump(self.sent_embeddings, open(path, 'wb'))
        logger.info(f"[{str(datetime.now())[:19]}]\tSave embedding to: `{path}`.")

    def load(self, path: str) -> None:
        """加载向量

        Args:
            path (str): 向量保存路径
        """
        
        self.sent_embeddings = pickle.load(open(path, 'rb'))
        logger.info(f"[{str(datetime.now())[:19]}]\tLoad embedding from: `{path}`.")
        

class EmbeddingAPI:
    def __init__(self):
        self.api_config = API_LIST["embedding"]
        
    def __call__(self, query: Union[str, List[str]], is_query: bool = True) -> numpy.ndarray:
        response = requests.post(
                    self.api_config["url"], 
                    headers=self.api_config["headers"], 
                    json={"sentences": [query] if isinstance(query, str) else query, "is_query": is_query})
        try:
            embedding_list = response.json()["embeddings"]
        except:
            logger.error(f"Error: {response.text}")
            return None
        return numpy.array(embedding_list)


class RerankEmbedding:
    def __init__(self):
        self.type = None

    @classmethod
    def load_model(cls, model_path: str = 'models/bge-reranker-large') -> object:
        logger.info(f"Loading rerank model: {model_path} ...")
        if "layerwise" in model_path:
            cls.reranker = LayerWiseFlagLLMReranker(model_path, use_fp16=True)
            cls.type = "layerwise"
        elif "gemma" in model_path:
            cls.reranker = FlagLLMReranker(model_path, use_fp16=True)
            cls.type = "llm"
        else:
            cls.reranker = FlagReranker(model_path, use_fp16=True)
            cls.type = "normal"
        return cls()


    def __call__(self, query: str, sentences: List[str], topk: int = 3, normalize: bool = False, cutoff_layers: int = None, return_scores: bool = False) -> int:
        """获取最佳结果的上下文

        Args:
            query (str): 用户提问\n
            sentences (List[str]): 粗排上下文列表

        Returns:
            int: 最佳上下文索引
        """
        assert cutoff_layers == None or (cutoff_layers != None and self.type == "layerwise"), "cutoff_layers only works for layerwise reranker."
        pairs = [[query, s] for s in sentences]
        if self.type == "layerwise":
            scores = self.reranker.compute_score(pairs, cutoff_layers=cutoff_layers if not cutoff_layers else [28], normalize=normalize)
        else:
            scores = self.reranker.compute_score(pairs, normalize=normalize)
        sent_order = sorted([(i, s) for i, s in enumerate(scores)], key=lambda x: x[1], reverse=True)
        return [s[0] for s in sent_order[: topk]] if not return_scores else scores


class RerankEmbeddingAPI:
    def __init__(self):
        self.api_config = API_LIST["reranker"]
        
    def __call__(self, query: str, sentences: List[str], topk: int = 3, normalize: bool = False, cutoff_layers: int = None, return_scores: bool = False) -> List[List[int]]:
        response = requests.post(
                    self.api_config["url"], 
                    headers=self.api_config["headers"], 
                    json={"query": query, "sentences": sentences, "topk": topk, "normalize": normalize, "cutoff_layers": cutoff_layers, "return_scores": return_scores}).json()
        
        return response["indexes"] if not return_scores else response["scores"]

    
class BM25:
    def __init__(self, sentences: List[str], lang: str = 'zh'):
        self.sentences = sentences
        self.lang = lang
        logger.info(f"Building BM25 model ... sentences num: {len(sentences)}")
        self.sentences_splitted = self._split_words(self.sentences)
        self.bm25 = BM25Okapi(self.sentences_splitted)

    
    def _split_words(self, sentences: List[str]) -> List[List[str]]:
        if self.lang == 'zh':
            return [list(jieba.cut(s)) for s in sentences]
        else:
            return [s.split() for s in sentences]
    
    
    def __call__(self, quies: List[str], k: int = 3, return_scores: bool = False) -> List[List[int]]:
        assert isinstance(quies, list), "The input must be a list of queries."
        queries_splitted = self._split_words(quies)
        sents_scores = [self.bm25.get_scores(query_splitted).tolist()for query_splitted in queries_splitted]
        sents_order = [sorted([(i, s) for i, s in enumerate(sent_scores)], key=lambda x: x[1], reverse=True) for sent_scores in sents_scores]
        return [[s[0] for s in sent_order[: k]] for sent_order in sents_order] if not return_scores else sents_scores


if __name__ == '__main__':
    # from FlagEmbedding import FlagModel
    sentences_1 = "根据2023年中期报告，华泰柏瑞消费成长混合的应付赎回费是多少（保留2位有效数字，不加千位分隔符）"
    sentences_2 = ["东方新能源汽车主题混合型证券投资基金2023年中期报告"]
    # model = FlagModel('models/bge-large-zh', 
    #                 query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    #                 use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = EmbeddingAPI()
    # embeddings_1 = model(sentences_1)
    # embeddings_2 = model(sentences_2)
    # similarity = embeddings_1 @ embeddings_2.T
    # print(similarity)

    # # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
    # # corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
    # passages = ["东方新能源汽车主题混合型证券投资基金2023年中期报告"]
    # query = "根据2023年中期报告，华泰柏瑞消费成长混合的应付赎回费是多少（保留2位有效数字，不加千位分隔符）"
    # model = RerankEmbeddingAPI()
    # scores = model(query, passages)
    # print(scores)
    # print(passages[scores[0]])
    
    # bm25 = BM25(sentences_1 + sentences_2)
    # print(bm25(["样例文档-3"]))