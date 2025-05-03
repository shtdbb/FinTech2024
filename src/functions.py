import re
import json
import pytz
import math
import numpy
import urllib
import requests
from datetime import datetime
from bs4 import BeautifulSoup

week = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '日'}

tools = [
    {"type": "function",
        "function": {
        "name": "run_python",
        "description": "用于执行满足用户意图的Python代码功能，使用exec函数实现，返回执行结果变量`result`。",
        "parameters": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "string",
                    "description": "Python代码，请务必在代码中设置存储运行结果的变量`result`，如'''import random\nresult = random.randint(0, 100)'''。"
                    },
                },
            "required": ["codes"]
            }
        }
    }
]

def calculate(what: str):
    try:
        return {'result': eval(what)}
    except:
        return {'error': '表达式不符合Python规范，请修改后重新调用。'}


def run_python(codes: str):
    try:
        exec(codes)
        try:
            return {'result': result}
        except:
            return {'error': '代码缺少返回值变量`result`，请添加后重新调用。'}
    except:
        return {'error': '代码不符合Python规范，请修改后重新调用。'}


def get_datetime(**kwargs) -> dict:
    """获取当前日期和时间

    Returns:
        dict: {'date': '...', 'time': '...', 'weekday': '...'}
    """
    beijing_tz = pytz.timezone('Asia/Shanghai')
    time = datetime.now().astimezone(beijing_tz)
    return {'date': time.strftime("今天是%Y年%m月%d日"), 'time': time.strftime("当前为%H时%M分%S秒"), 'weekday': f"星期{week[str(time.weekday() + 1)]}"}


if __name__ == '__main__':
    from pprint import pprint
    print(wikipedia("Three Body", 1, True))