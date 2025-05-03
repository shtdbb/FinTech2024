
import random
from ltp import StnSplit
from typing import Union, List


def sentence_split(paragraph: Union[str, List[str,]]) -> list:
    """段落分割成句子

    Args:
        paragraph (Union[str, List[str,]]): 段落, 可以是字符串或字符串列表

    Returns:
        Union[str, List[str,]]: 分割后的句子列表
    """
    assert isinstance(paragraph, list) or isinstance(paragraph, str), "`paragraph` must be a list or a string."
    if isinstance(paragraph, list):
        sentence_list = StnSplit().batch_split(paragraph)   # ['xxx', 'xxxxx'] -> ['xx', 'x', 'xxx', 'xx']

    else:
        sentence_list = StnSplit().split(paragraph)   # 'xxx' -> ['xx', 'x']
    return sentence_list


def get_random_id(mode = 'int', length = 8) -> str:
    """生成随机 id

    Args:
        mode (str, optional): id 模式, ['int', 'a+0', 'A+0', 'strict']. Defaults to 'int'.
        length (int, optional): id 长度. Defaults to 6.
    """
    
    assert mode in ['int', 'a+0', 'A+0', 'strict'] and length > 0, \
        f"The length of id must be greater than 0, your `length` is {length} and `mode` is {mode}."
    
    char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    if mode == 'int':
        return "".join(random.choices(char_list[: 10], k=length))
    elif mode == 'a+0':
        return "".join(random.choices(char_list[: 10] + char_list[-26: ], k=length))
    elif mode == 'A+0':
        return "".join(random.choices(char_list[: 36], k=length))
    else:
        return "".join(random.choices(char_list, k=length))


if __name__ == '__main__':
    print(get_random_id())