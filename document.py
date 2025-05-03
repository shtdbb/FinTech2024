import re
import os
import json
import fitz
import markdown
import pickle
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Union, List
from collections import defaultdict

from logger import logger
from Retriever import APIModel
from template import DOCUMENT_TEMPLATE, SHEET_DESC
from utils import get_random_id, sentence_split


class TextDocument:
    """文本类文档, 如: txt, log
    """
    def __init__(self):
        self.meta_data = defaultdict(list)   # 元数据
        """self.meta_data
        {
            '{ID}': 
                {
                    'type': 'TextDocument',
                    'file_name': 'xxx',
                    'file_path': 'xxx',
                    'sentences': (start_index, end_index),
                    'n_sentences': 0,
                },
        }
        """
        
        self.n_documents = 0   # 文档数量
        self.n_sentences = 0   # 句子数量
        self.para_split = '\n'   # 段落分割符
        self.last_index = -1   # 最后一个句子的索引
        self.sent_list = []   # 保存所有句子
        self.id_to_document = {}   # 句子索引到文档编号的映射: {0: '123456', 1: '123456', ...}
        
    
    def input_documents(self, documents: Union[str, list], max_length: int = 512, return_list: bool = True) -> Union[list, None]:
        """输入文档
        
        Args:
            documents (Union[str, list]): 单个文档路径或文档路径列表, 可以是字符串或字符串列表\n
            return_list (bool, optional): 是否返回句子列表. Defaults to True.
        
        Returns:
            list: 句子列表
        """
        
        assert isinstance(documents, list) or isinstance(documents, str), \
            f"`documents` must be a list or a string, your type of `document` is {type(documents)}."
        
        self.max_length = max_length   # 段落最大长度
        
        try:
            if isinstance(documents, list):
                self.n_documents += len(documents)
                
                for doc in documents:
                    logger.info(f"Process document `{doc}` ...")
                    
                    # 打开文件
                    with open(doc, 'r', encoding='utf-8') as f:
                        context = f.read()
                    
                    # 分割成段落
                    para_list = self.split_paragraphs(context)
                    
                    # 分组成句子
                    sent_list = self.group_sentences(para_list)
                    
                    # 打上格式
                    sent_list = [DOCUMENT_TEMPLATE.format(document_name=doc.split('/')[-1], 
                                                        contents_title=" ".join(self.extract_titles_from_markdown(sent)) 
                                                        if ".md"  == doc[-3: ] else "无",
                                                        context=sent) for sent in sent_list]
                    self.sent_list += sent_list
                    
                    # 更新元数据
                    doc_id = get_random_id()
                    self.meta_data[doc_id] = {
                        'type': 'TextDocument',
                        'file_name': doc.split('/')[-1],
                        'file_path': doc,
                        'sentences': (self.last_index + 1, self.last_index + len(sent_list)),
                        'n_sentences': len(sent_list),
                    }
                    
                    # 句子索引到文档编号的映射
                    for i in range(self.last_index + 1, self.last_index + len(sent_list) + 1):
                        self.id_to_document[i] = doc_id
                    
                    self.n_sentences += len(sent_list)
                    self.last_index = self.last_index + len(sent_list)
                    
            else:
                self.n_documents += 1
                logger.info(f"Process document `{documents}` ...")
                
                # 打开文件
                with open(documents, 'r', encoding='utf-8') as f:
                    context = f.read()
                
                # 分割成段落
                para_list = self.split_paragraphs(context)
                
                # 分组成句子
                sent_list = self.group_sentences(para_list)
                
                # 打上格式
                sent_list = [DOCUMENT_TEMPLATE.format(document_name=documents.split('/')[-1], 
                                                    contents_title=" ".join(self.extract_titles_from_markdown(sent)) 
                                                    if ".md"  == documents[-3: ] else "无",
                                                    context=sent) for sent in sent_list]
                
                self.sent_list += sent_list
                
                # 更新元数据
                doc_id = get_random_id()
                self.meta_data[doc_id] = {
                    'type': 'TextDocument',
                    'file_name': documents.split('/')[-1],
                    'file_path': documents,
                    'sentences': (self.last_index + 1, self.last_index + len(sent_list)),
                    'n_sentences': len(sent_list),
                }
                
                # 句子索引到文档编号的映射
                for i in range(self.last_index + 1, self.last_index + len(sent_list) + 1):
                    self.id_to_document[i] = doc_id
                
                self.n_sentences += len(sent_list)
                self.last_index = self.last_index + len(sent_list)
            
            if return_list:
                return sent_list
        except Exception as e:
            logger.error(f"{e}")


    def extract_titles_from_markdown(self, markdown_text) -> List[str]:
        # 将Markdown文本转换为HTML
        html = markdown.markdown(markdown_text)
        
        # 使用BeautifulSoup来解析HTML并提取标题
        soup = BeautifulSoup(html, 'html.parser')
        
        # 获取所有标题元素
        titles = [title.get_text() for title in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
        return titles if titles else ["无"]

        
    def save(self, save_path: str = f"documents_{re.sub(r'[-: ]', '_', str(datetime.now())[:19])}"):
        """保存文档解析结果

        Args:
            save_path (_type_, optional): 保存的文件夹路径. Defaults to f"documents_{DATETIME}".
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        pickle.dump(self.sent_list, open(f'{save_path}/sent_list.pkl', 'wb'))
        
        # 保存属性
        pickle.dump(self.id_to_document, open(f'{save_path}/id_to_document.pkl', 'wb'))
        json.dump({
            'meta_data': self.meta_data,
            'n_documents': self.n_documents,
            'n_sentences': self.n_sentences,
            'max_length': self.max_length,
            'para_split': self.para_split,
            'last_index': self.last_index,
            'sent_list': f'{save_path}/sent_list.pkl',
            'id_to_document': f'{save_path}/id_to_document.pkl',
        }, open(f'{save_path}/state_dict.json', 'w'))
        
        logger.info(f"Save document to: `{save_path}`")
        
        
    def load(self, load_path: str):
        """加载文档解析结果
        
        Args:
            load_path (str): 加载的文件夹路径
        """
        assert os.path.exists(load_path), f"`load_path` must be a existing path, your `load_path` is {load_path}."
        
        # 加载属性
        state_dict = json.load(open(f'{load_path}/state_dict.json', 'r'))
        self.meta_data = state_dict['meta_data']
        self.n_documents = state_dict['n_documents']
        self.n_sentences = state_dict['n_sentences']
        self.max_length = state_dict['max_length']
        self.para_split = state_dict['para_split']
        self.last_index = state_dict['last_index']
        self.sent_list = pickle.load(open(state_dict['sent_list'], 'rb'))
        self.id_to_document = pickle.load(open(state_dict['id_to_document'], 'rb'))
        
        logger.info(f"Load document from: `{load_path}`.")
    
    
    def split_paragraphs(self, context: str) -> List[List[str,]]:
        """全文分割成段落
        
        Args:
            context (str): 全文
        
        Returns:
            list: 每个元素为段落列表， 每个段落列表为 n 个句子
        """
        
        assert isinstance(context, str), f"`context` must be a string, your type of `context` is {type(context)}."
        
        paragraph_list = []
        for para in context.split(self.para_split):
            if para.strip() != '':
                paragraph_list.append(["\n" + sent + "\n" if sent[:2] == "##" else sent for sent in sentence_split(para.strip())])
        return paragraph_list
    
    
    def group_sentences(self, paragraph_list: List[List[str,]], n_overlap: int = 1) -> List[str,]:
        """将段落列表分组成齐长的句子列表
        
        Args:
            paragraph_list (List[List[str,]]): 段落列表, [['xx', 'x'], ['xxx'], ...]
            n_overlap (int, optional): 句子重叠数, 将前后文的最近 n 个句子加入, 新增 2*n 个句子. Defaults to 1.
        
        Returns:
            List[str,]: 句子列表, ['xxxx', 'xxxx', ...]
        """
        
        assert n_overlap >= 0, f"`n_overlap` must not be negative, your `n_overlap` is {n_overlap}."
        
        """
        # 根据长度分组
        sentence_list = []
        for i in range(len(paragraph_list)):   
            sentences, group = [], []   # [['xxx', ], ], ['xx', ]
            length = 0
            for j in range(len(paragraph_list[i])):
                if len(paragraph_list[i][j]) > (self.max_length - n_overlap * 50 * 2):   # 若单个句子超长
                    paragraph_list[i][j] = paragraph_list[i][j][: len(paragraph_list[i][j]) // 2]
                    paragraph_list[i].insert(j + 1, paragraph_list[i][j][len(paragraph_list[i][j]) // 2: ])
                    
                length += len(paragraph_list[i][j])
                
                if j == len(paragraph_list[i]) - 1:   # 最后一个句子
                    if length > (self.max_length - n_overlap * 50 * 2):
                        sentences.append(group)
                        sentences.append([paragraph_list[i][j]])
                    else:
                        group.append(paragraph_list[i][j])
                        sentences.append(group)
                    break
                
                if length > (self.max_length - n_overlap * 50 * 2):   # 超长
                    sentences.append(group)
                    group = []
                    length = 0
                else:   # 未超长
                    group.append(paragraph_list[i][j])
                    
            sentence_list.append(sentences)
        
        # 句子重叠
        sentences = []
        new = copy.deepcopy(sentence_list)
        for p in range(len(sentence_list)):   # 每个段落
            for g in range(len(sentence_list[p])):   # 每个组
                if len(sentence_list[p]) == 1:   # 段落就一个组
                    # TODO: 未考虑句子重叠大于 1 的细节
                    new[p][g] = ([sentence_list[p-1][-1][-1]] if p > 0 else []) + \
                                                sentence_list[p][g] + \
                                                    ([sentence_list[p+1][0][0]] if p < (len(sentence_list) - 1) else [])
                
                else:
                    new[p][g] = [sentence_list[p][g-1][-1]] if g>0 else ([sentence_list[p-1][-1][-1]] if p>0 else []) + \
                                sentence_list[p][g] + \
            [sentence_list[p][g+1][0]] if g<(len(sentence_list[p])-1) else ([sentence_list[p+1][0][0]] if p<len(sentence_list)-1 else [])
                    
                sentences.append(" ".join(new[p][g]))
        
        # 短句合并
        i = 0
        while i < len(sentences):
            if len(sentences[i]) < self.max_length // 2:
                if 0 < i < len(sentences) - 1:
                    if len(sentences[i-1]) <= len(sentences[i+1]):
                        sentences[i-1] += " " + sentences[i]
                        sentences.pop(i)
                    else:
                        sentences[i+1] = sentences[i] + " " + sentences[i+1]
                        sentences.pop(i)
                elif i == 0:
                    sentences[i+1] = sentences[i] + " " + sentences[i+1]
                    sentences.pop(i)
                else:
                    sentences[i-1] += " " + sentences[i]
                    sentences.pop()
            else:
                i += 1
        """        
        paragraph_list = [[" ".join(p)] if "|" == p[0][0] else p for p in paragraph_list]
        sent_list = sum(paragraph_list, [])
        
        length, left, right = 0, 0, 0
        sentences = []
        while right < len(sent_list):
            if sent_list[right][0] == "|" and sent_list[right][-1] == "|":   # 有表格
                left = right
                for i in range(left + 1, len(sent_list) - 1):
                    if "|" not in sent_list[i+1]:
                        right = i
                        break
                sentences.append("\n".join(sent_list[left - 2 if left > 1 else 0: right + 3]))
                left = right + 1 - n_overlap
            else:
                length += len(sent_list[right])
                
                if length > self.max_length:
                    sentences.append(" ".join(sent_list[left: right + 1]))
                    length = 0
                    left = right + 1 - n_overlap
                    
                if right == len(sent_list) - 1 and length <= self.max_length:
                    sentences.append(" ".join(sent_list[left: right + 1]))
                    
            right += 1

        return sentences


class PDFDocument(TextDocument):
    def __init__(self):
        super().__init__()
        
    def input_documents(self, pdf_path: Union[str, list] , **kwargs) -> List[str]:
        if isinstance(pdf_path, str):
            logger.info(f"Transfer PDF to txt: `{pdf_path}`.")
            try:
                with fitz.open(pdf_path) as pdf:
                    text = ""
                    # 遍历PDF中的每一页
                    for page in pdf:
                        # 提取当前页的文本
                        text += page.get_text()
            except:
                pdf_path_new = ".".join(pdf_path.split(".")[: -1]) + "." + pdf_path.split(".")[-1].lower()
                os.rename(pdf_path, pdf_path_new)
                pdf_path = pdf_path_new
                with fitz.open(pdf_path) as pdf:
                    text = ""
                    # 遍历PDF中的每一页
                    for page in pdf:
                        # 提取当前页的文本
                        text += page.get_text()
            text = re.sub(r'([^\s,]*),([^\s,]*)', r'\1\2', text)
            txt_path = pdf_path.replace(".pdf", ".txt")
            
            logger.info(f"Extract PDF to txt file: `{txt_path}`.")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            return super().input_documents(txt_path, **kwargs)
        else:
            txts_path = []
            for path in pdf_path:
                with fitz.open(path) as pdf:
                    text = ""
                    # 遍历PDF中的每一页
                    for page in pdf:
                        # 提取当前页的文本
                        text += page.get_text()
                text = re.sub(r'([^\s,]*),([^\s,]*)', r'\1\2', text)
                txt_path = path.replace(".pdf", ".txt")
                with open(txt_path, "w") as f:
                    f.write(text)
                txts_path.append(txt_path)
            return super().input_documents(txts_path, **kwargs)


class SheetDocument(TextDocument):
    def __init__(self):
        super().__init__()

    @classmethod
    def read_sheet(cls, excel_path: Union[str, list], header: int = None) -> 'SheetDocument':
        file_type = "csv" if excel_path.split('.')[-1] == "csv" else "xlsx"
        cls.df = pd.read_excel(excel_path, header=header) if file_type == "xlsx" else pd.read_csv(excel_path, sep=",", header=header)
        cls.markdown = cls.df.to_markdown(index=False).split('\n')[: 10]
        cls.markdown = "\\n".join(cls.markdown)
        cls.desc = cls.get_desc(cls)
        return cls()
    
    def get_desc(self, api_name: str = "glm") -> str:
        assert self.markdown is not None, "Please read the sheet first."
        model = APIModel(SHEET_DESC, api_name)
        self.desc = model(markdown=self.markdown)
        return self.desc

    
if __name__ == '__main__':
    txt = TextDocument()
    txt.input_documents("data/md/财通裕惠63个月定期开放债券型证券投资基金2023年第2季度报告.md")
    # pdf = PDFDocument()
    # sent_list = pdf.input_documents("./tmp/LLM_Survey_Chinese.pdf")
#     # txt.save()
#     txt.load("documents_2023_12_08_03_56_52")
#     pass
    # sheet = SheetDocument.read_sheet("2024年中国农业发展银行软件开发中心校园招聘拟招录人员名单（第一批）.xlsx")
    # print(sheet.desc)
    