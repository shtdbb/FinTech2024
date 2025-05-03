import re
import glob
from tqdm import tqdm

def drop_noise(lines: list[dict]):
    i = 0
    while i < len(lines):
        if lines[i]["inside"] == "" and lines[i]["type"] == "text":
            lines.pop(i)
        elif lines[i]["type"] == "页眉" or lines[i]["type"] == "页脚":
            lines.pop(i)
        elif type(lines[i]["inside"]) == str and re.match(r'.*\.{2,}\d{1,2}$', lines[i]["inside"]):
            lines.pop(i)
        else:
            i += 1
    return lines


def extract_title(lines: list[dict]):
    except_level = [1, 1, 1, 1]
    for i in range(len(lines)):
        if lines[i]["type"] != "text": continue
        text = lines[i]["inside"]
        level_1_number = re.match(r'§\d{1,2}', text)   # 二级标题
        level_2_number = re.match(r'^\d{1,2}\.\d{1,2}', text)   # 三级标题
        level_3_number = re.match(r'^\d{1,2}\.\d{1,2}\.\d{1,2}', text)   # 四级标题
        level_4_number = re.match(r'^\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}', text)   # 五级标题
        if level_1_number and int(level_1_number.group()[1: ]) == except_level[0]:
            lines[i]["inside"] = "\n## " + text.replace(level_1_number.group(), level_1_number.group() + " ")
            except_level[0] += 1
            except_level[1] = 1
            except_level[2] = 1
            except_level[3] = 1
        elif level_2_number and int(level_2_number.group().split('.')[0]) == except_level[0] - 1 \
                    and int(level_2_number.group().split('.')[1]) == except_level[1]:
            lines[i]["inside"] = "\n### " + text[: len(level_2_number.group())] + " " + text[len(level_2_number.group()): ]
            except_level[1] += 1
            except_level[2] = 1
            except_level[3] = 1
        elif level_3_number and int(level_3_number.group().split('.')[0]) == except_level[0] - 1 \
                    and int(level_3_number.group().split('.')[1]) == except_level[1] - 1 \
                    and int(level_3_number.group().split('.')[2]) == except_level[2]:
            lines[i]["inside"] = "\n#### " + text[: len(level_3_number.group())] + " " + text[len(level_3_number.group()): ]
            except_level[2] += 1
            except_level[3] = 1
        elif level_4_number and int(level_4_number.group().split('.')[0]) == except_level[0] - 1 \
                    and int(level_4_number.group().split('.')[1]) == except_level[1] - 1 \
                    and int(level_4_number.group().split('.')[2]) == except_level[2] - 1 \
                    and int(level_4_number.group().split('.')[3]) == except_level[3]:
            lines[i]["inside"] = "\n##### " + text[: len(level_4_number.group())] + " " + text[len(level_4_number.group()): ]
            except_level[3] += 1
    return lines


def extract_sheet(lines: list[dict]):
    head, tail = 0, 0
    while head < len(lines):
        if lines[head]["type"] == "text": 
            head += 1
        else:   # 发现表格
            for i in range(head + 1, len(lines) - 1):
                if lines[i + 1]["type"] == "text":
                    tail = i
                    break
                elif i == len(lines) - 2:
                    tail = len(lines) - 1
            n_feature = len(lines[head]["inside"])
            header = f"""| {" | ".join([" " for _ in range(n_feature)])} |  
| -{"- | -".join(["-" for _ in range(n_feature)])}- |  
| {" | ".join(lines[head]["inside"])} |"""
            lines[head]["inside"] = header
            for i in range(head + 1, tail + 1):
                lines[i]["inside"] = "| " + " | ".join(lines[i]["inside"]) + " |"
            
            head = tail + 1
    return lines


def process(lines: list[dict]) -> str:
    lines = drop_noise(lines)
    lines = extract_title(lines)
    lines = extract_sheet(lines)
    
    content = ""
    for line in lines:
        content += line["inside"] + "  \n"
    content = re.sub(r'([^\s,]*),([^\s,]*)', r'\1\2', content)
    return content


if __name__ == "__main__":
    folder_path = 'D:\data\desktop\招商训练营\\announcements'
    file_paths = glob.glob(f'{folder_path}/*.txt')
    for file in tqdm(file_paths):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [eval(line.strip()) for line in lines]
        for i in range(len(lines)):
            try:
                lines[i]["inside"] = eval(lines[i]["inside"])
            except:
                continue

        content = process(lines)
        with open(file.replace(".txt", ".md"), "w", encoding="utf-8") as f:
            f.write(content)
