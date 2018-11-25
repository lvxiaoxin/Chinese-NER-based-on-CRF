# -*- coding: utf-8 -*-
"""
@version: 
@time: 2018/11/24
@software: PyCharm
@file: singleton
"""
from NER.model import NerModel

__model = None


def get_model():
    global __model
    if not __model:
        __model = NerModel()
    return __model
