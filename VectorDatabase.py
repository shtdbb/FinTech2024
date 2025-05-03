import re
import os
import json
import faiss
import pickle
import numpy as np
from logger import logger
from datetime import datetime
from sklearn.decomposition import PCA


class VectorDatabase(object):
    def __init__(self, 
                 n_hiddens: int = 768, 
                 measure = faiss.METRIC_INNER_PRODUCT, 
                 search_mode: str = 'HNSW64',
                 db_name: str = "default"
                 ) -> None:
        """初始化向量数据库

        Args:
            n_hiddens (int): 向量维数\n
            measure (Union[ faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2, faiss.METRIC_L1, faiss.METRIC_Linf, \n
                        faiss.METRIC_Linf, faiss.METRIC_Lp, faiss.METRIC_Canberra, faiss.METRIC_BrayCurtis, \n
                        faiss.METRIC_JensenShannon, faiss.METRIC_Jaccard ], optional): \n
                    度量指标, 默认为欧氏距离. Defaults to faiss.METRIC_L2. \n
            search_mode ([str: 'Flat', str: 'IVF100,Flat', str: 'PQ16', str:'IVF100,PQ16', \n
                            str: 'LSH', str: 'HNSW64'], optional): \n
                    检索算法. Defaults to 'IVF100,PQ16'.
        """
        
        self.n_hiddens = n_hiddens   # 向量维度
        self.measure = measure   # 度量方式
        self.search_mode = search_mode   # 检索算法
        self.db_name = db_name   # 数据库名称
        self.amount = -1   # 数据库数据量
        self.pca = None   # PCA 模型
        self.is_pca = False   # 数据是否降维
        self.pca_dim = None   # 降维后的维度
        
        
    def train_pca(self, db_data: np.ndarray, pca_dim) -> np.ndarray:
        """训练 PCA 模型

        Args:
            db_data (np.ndarray): 数据库张量, shape: (amount, n_hiddens)\n
            pca_dim (_type_): 降维的维数, [int, 'auto']

        Returns:
            np.ndarray: 降维后的数据库张量, shape: (amount, pca_dim)
        """
        
        assert pca_dim == 'auto' or pca_dim < self.n_hiddens, 'The PCA dimension must be less than n_hiddens.'
        self.is_pca = True

        logger.info(f"PCA training...")
        self.pca = PCA(n_components = ('mle' if pca_dim == 'auto' else pca_dim))
        self.pca.fit(db_data)
        

        self.pca_dim = self.pca.n_components_
            
        logger.info(f"Train finished, the dimension of PCA: {self.pca_dim}.")
        return self.pca.transform(db_data)
        
        
    def transfom_pca(self, query_data: np.ndarray) -> np.ndarray:
        """数据降维

        Args:
            query_data (np.ndarray): 查询张量, shape: (amount, n_hiddens) \n

        Returns:
            np.ndarray: 降维后数据
        """
        
        assert self.is_pca, 'PCA model is not trained.'
        logger.info(f"PCA transforming...")
        return self.pca.transform(query_data)
        
        
    def create_index(self, db_data: np.ndarray, pca_dim: int = None) -> None:
        """创建索引

        Args:
            db_data (np.ndarray): 待训练/添加张量 \n
            pca_dim (int, optional): (若需) 降维维度. Defaults to None.
        """
        
        assert len(db_data.shape) == 2, 'The dimension of database data must be `(amount, n_hiddens)`.'
        self.amount, self.n_hiddens = db_data.shape
        
        # 若需 PCA
        if pca_dim:
            db_data = self.train_pca(db_data, pca_dim)
            self.n_hiddens = self.pca_dim
        
        # 创建数据库索引
        self._index = faiss.index_factory(self.n_hiddens, self.search_mode, self.measure)
        db_data = db_data.astype(np.float32)
        logger.info(f"`index.is_trained: {self._index.is_trained}`")
        
        # 训练索引
        logger.info(f"Index building ...")
        self._index.train(db_data)
        self._index.add(db_data)
        
        logger.info(f"Index build finished, there are {self._index.ntotal} vectors with {self.n_hiddens} dim.")
        assert self.amount == self._index.ntotal, "The amount of database data is not equal to the amount of index."
    
    
    def __call__(self, query_data: np.ndarray, topk: int = 5, return_distances: bool = False) -> np.ndarray:
        """检索相似向量

        Args:
            query_data (np.ndarray): 待查询的张量, shape: (amount, n_hiddens) \n
            topk (int, optional): 返回最相似的 `topk` 个. Defaults to 5.\n
            return_distances (bool, optional): 是否返回距离列表. Defaults to False.

        Returns:
            np.ndarray:  \n
            - distance_list: (amount, topk) 距离度量大小 \n
            - index_list: (amount, topk) 相似向量的索引
        """
        if len(query_data.shape) == 1: query_data = query_data.reshape(1, -1)
        assert len(query_data.shape) == 2, 'The dimension of query data must be `(amount, n_hiddens)`.'
        
        if self.is_pca:
            query_data = self.transfom_pca(query_data)
            
        distance_list, index_list = self._index.search(query_data.astype(np.float32), topk)
        
        # distance_list(float): (amount, topk)
        # index_list(int): (amount, topk)
        if not return_distances:
            return index_list
        else:
            return distance_list, index_list
    
    
    def save(self, document_path: str = None) -> dict:
        """保存数据库

        Args:
            document_path (str, optional): 文件夹路径. Defaults to `vector_database_{DATETIME}`.
            
        Returns:
            str: 数据库状态 json 字符串
        """
        if document_path is None:
            document_path = f"vector_database_{self.db_name}_{re.sub(r'[-: ]', '_', str(datetime.now())[:19])}"
        
        if not os.path.exists(document_path):
            os.makedirs(document_path)
        
        if self.is_pca:
            pickle.dump(self.pca, open(f"{document_path}/pca_{self.pca_dim}.pkl", "wb"))
            
        pickle.dump(self._index, open(f"{document_path}/index_{self.n_hiddens}.pkl", "wb"))
        # faiss.write_index(self._index, f"{document_path}/index_{self.n_hiddens}.faissindex")
        
        state_dict = {
            "n_hiddens": self.n_hiddens,
            "measure": self.measure,
            "search_mode": self.search_mode,
            "amount": self.amount,
            "pca": f"{document_path}/pca_{self.pca_dim}.pkl" if self.is_pca else None,
            "is_pca": self.is_pca,
            "pca_dim": self.pca_dim,
            "_index": f"{document_path}/index_{self.n_hiddens}.pkl",
            # "_index": f"{document_path}/index_{self.n_hiddens}.faissindex",
        }
        
        json.dump(state_dict, open(f"{document_path}/state_dict.json", "w"))
        
        abstract_path = os.path.dirname(os.path.realpath(__file__))
        logger.info(f"Save vector database to: `{document_path if abstract_path in document_path else abstract_path + '/' + document_path}`.")
        return state_dict
        
            
            
    def load(self, document_path: str) -> object:
        """从文件夹加载数据库

        Args:
            file_name (str): 文件夹路径

        Returns:
            object: 数据库对象
        """
        
        abstract_path = os.path.dirname(os.path.realpath(__file__))
        logger.info(f"Load vector database from: `{document_path if abstract_path in document_path else abstract_path + '/' + document_path}`.")
        
        state_dict = json.load(open(f"{document_path}/state_dict.json", "r"))
        self.n_hiddens = state_dict["n_hiddens"]
        self.measure = state_dict["measure"]
        self.search_mode = state_dict["search_mode"]
        self.amount = state_dict["amount"]
        self.is_pca = state_dict["is_pca"]
        
        if self.is_pca:
            self.pca = pickle.load(open(state_dict["pca"], "rb"))
        else:
            self.pca = None
            
        self.pca_dim = state_dict["pca_dim"]
        self._index = pickle.load(open(state_dict["_index"], "rb"))
        # self._index = faiss.read_index(state_dict["_index"])
        # logger.info(f"Load vector database sucessfully.")
        
        return self
        
    
    
if __name__ == "__main__":
    db = VectorDatabase()
    # db_data = np.random.random((10000, 128))
    # db.create_index(db_data)
    query_data = np.random.random((2, 128))
    # print(db(query_data))
    # db.save()
    db.load('vector_database_2023_12_08_07_47_11')
    print(db(query_data))
