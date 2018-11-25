# -*- coding: utf-8 -*-
"""
@version: 
@time: 2018/11/24
@software: PyCharm
@file: main
"""
import json

from NER.singleton import get_model
from NER.utils import test_config, document, entity, section_map
import codecs

if __name__ == '__main__':
    model = get_model()
    model.train()
    with open(test_config.get("test_path").format("testset"), encoding='utf-8') as f:
        content = f.read()
    testset = json.loads(content)
    entity_list = []
    for obj in testset:
        sentence_0 = obj.get(document, {}).get(section_map["party_info"])
        if len(list(sentence_0.keys())) == 0:
            sentence_0 = ""
        elif len(list(sentence_0.keys())) == 1:
            sentence_0 = obj.get(document, {}).get(section_map["party_info"])["0"]
        else:
            sentence_0 = obj.get(document, {}).get(section_map["party_info"])["1"]
        sentence_1 = obj.get(document, {}).get(section_map["case_info"])
        if len(list(sentence_1.keys())) == 0:
            sentence_1 = ""
        else:
            sentence_1 = obj.get(document, {}).get(section_map["case_info"])["0"]
        entity_0 = model.predict(sentence_0, 0)
        entity_1 = model.predict(sentence_1, 1)
        entity_obj = entity_0.copy()
        entity_obj.update(entity_1)
        print("--> ", entity_obj)
        entity_list.append({entity: entity_obj})
    with codecs.open(test_config.get("output_path").format("output"), 'w', encoding='utf-8') as f:
        content = json.dumps(entity_list, indent=4, ensure_ascii=False)
        f.write(content)

