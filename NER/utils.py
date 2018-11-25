# -*- coding: utf-8 -*-
"""
@version: 
@time: 2018/11/24
@software: PyCharm
@file: Corpus
"""
import re

# 定义测试集section名的映射
document = "文书"

_section_map = (
    ("head", "首部"),
    ("party_info", "当事人信息"),
    ("case_info", "案件基本情况"),
    ("judge_principle", "裁判原则"),
    ("judgment", "判决结果"),
    ("ending", "尾部"),
    ("apply_record", "上诉记录")
)

section_map = dict(_section_map)

# 定义训练集目标实体名映射

entity = "实体"

_entity_map = (
    ("BIR", "出生信息"),
    ("NAT", "名族"),
    ("LOC", "居住地"),
    ("SEX", "性别"),
    ("HJ", "户籍"),
    ("EDU", "文化背景"),
    ("JOB", "职务"),
    ("ORG", "单位"),
    ("POL", "政治面貌"),
    ("PER", "被告人姓名"),
    ("T", "犯罪时间"),
    ("MON", "涉案金额")
)

entity_map = dict(_entity_map)

_tag_mean_map = (
    ("nr", "PER"),
    ("ns", "LOC"),
    ("nt", "ORG"),
    ("t", "T")
)

tag_mean_map = dict(_tag_mean_map)

_model_config = (
    ("algorithm", "lbfgs"),
    ("c1", "0.1"),
    ("c2", "0.1"),
    ("max_iterations", 100),
    ("model_path", "{}.pkl")
)

model_config = dict(_model_config)

_test_config = (
    ("test_path", "./data/{}.json"),
    ("output_path", "{}.json")
)

test_config = dict(_test_config)

_regex_map = (
    ("edu", "文化"),
    ("pol", "中共党员"),
    ("nat", "族"),
    ("curator", "检察院"),
    ("money", "元")
)

regex_map = dict(_regex_map)

regex_pattern = "[0-9]+(.[0-9]+)?([百千万]*)(余?)元"


def expand_list(nested_list):
    """
        将高维list转换为一维list
    """
    for item in nested_list:
        if isinstance(item, list):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item


def b_2_q(b_str):
    """
        半角转全角
    """
    q_str = ""
    for uchar in b_str:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif 126 >= inside_code >= 32:
            inside_code += 65248
        q_str += chr(inside_code)
    return q_str


def q_2_b(q_str):
    """
        全角转半角
    """
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格
            inside_code = 32
        elif 65374 >= inside_code >= 65281:
            inside_code -= 65248
        b_str += chr(inside_code)
    return b_str


def deal_with_entity(entity_list, sentence, flag):
    obj = dict()
    if flag == 0:
        for index in range(len(entity_list)):
            if entity_list[index][1] == "PER":
                obj[entity_map["PER"]] = entity_list[index][0]
            if entity_list[index][1] == "LOC":
                if obj.get(entity_map["LOC"], None) is None:
                    obj[entity_map["LOC"]] = entity_list[index][0]
                else:
                    obj[entity_map["HJ"]] = entity_list[index][0]
            if entity_list[index][1] == 'T':
                obj[entity_map['BIR']] = entity_list[index][0]
            if entity_list[index][1] == 'ORG' and regex_map["curator"] not in entity_list[index][0]:
                obj[entity_map['ORG']] = entity_list[index][0]
        sentence = sentence.replace("。", ",")
        sentence_list = sentence.split(",")
        for index in range(len(sentence_list)):
            if regex_map["edu"] in sentence_list[index]:
                if obj.get(entity_map["EDU"], None) is None:
                    obj[entity_map["EDU"]] = sentence_list[index]
            if regex_map["pol"] in sentence_list[index]:
                if obj.get(entity_map["POL"], None) is None:
                    obj[entity_map["POL"]] = sentence_list[index]
            if regex_map["nat"] in sentence_list[index]:
                if obj.get(entity_map["NAT"], None) is None:
                    obj[entity_map["NAT"]] = sentence_list[index]
    if flag == 1:
        for index in range(len(entity_list)):
            if entity_list[index][1] == 'T':
                obj[entity_map['T']] = entity_list[index][0]
        sentence = sentence.replace("。", ",")
        sentence_list = sentence.split(",")
        for index in range(len(sentence_list)):
            if re.search(regex_pattern, sentence_list[index]) is not None:
                obj[entity_map['MON']] = re.search(regex_pattern, sentence_list[index]).group()
    return obj
