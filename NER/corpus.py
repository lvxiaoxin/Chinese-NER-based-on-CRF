# -*- coding: utf-8 -*-
"""
@version: 
@time: 2018/11/24
@software: PyCharm
@file: Corpus
"""
import codecs
import re

from NER.utils import q_2_b, tag_mean_map

__corpus = None


class Corpus(object):

    def __init__(self):
        self.origin_corpus = self.read_corpus("./data/corpus.txt")
        self.pro_corpus = self.pre_process(self.origin_corpus)
        self.save_pro_corpus(self.pro_corpus)
        self.word_seq = []
        self.pos_seq = []
        self.tag_seq = []

    def read_corpus(self, path):
        with open(path, encoding='utf-8') as f:
            corpus = f.readlines()
        print("-> 完成训练集{0}的读入".format(path))
        return corpus

    def save_pro_corpus(self, pro_corpus):
        with codecs.open("./data/pro_corpus.txt", 'w', encoding='utf-8') as f:
            for line in pro_corpus:
                f.write(line)
                f.write("\n")
        print("-> 保存预处理数据")

    def pre_process(self, origin_corpus):
        pro_corpus = []
        for line in origin_corpus:
            words = q_2_b(line.strip("")).split('  ')
            pro_words = self.process_big_seq(words)
            pro_words = self.process_nr(pro_words)
            pro_words = self.process_t(pro_words)
            pro_corpus.append('  '.join(pro_words[1:]))
        print("-> 完成数据预处理")
        return pro_corpus

    def process_nr(self, words):
        pro_words = []
        index = 0
        while True:
            word = words[index] if index < len(words) else ''
            if '/nr' in word:
                next_index = index + 1
                if next_index < len(words) and '/nr' in words[next_index]:
                    pro_words.append(word.replace('/nr', '') + words[next_index])
                    index = next_index
                else:
                    pro_words.append(word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def process_t(self, words):
        pro_words = []
        index = 0
        temp = ''
        while True:
            word = words[index] if index < len(words) else ''
            if '/t' in word:
                temp = temp.replace('/t', '') + word
            elif temp:
                pro_words.append(temp)
                pro_words.append(word)
                temp = ''
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def process_big_seq(self, words):
        pro_words = []
        index = 0
        temp = ''
        while True:
            word = words[index] if index < len(words) else ''
            if '[' in word:
                temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=word.replace('[', ''))
            elif ']' in word:
                w = word.split(']')
                temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=w[0])
                pro_words.append(temp + '/' + w[1])
                temp = ''
            elif temp:
                temp += re.sub(pattern='/[a-zA-Z]*', repl='', string=word)
            elif word:
                pro_words.append(word)
            else:
                break
            index += 1
        return pro_words

    def initialize(self):
        pro_corpus = self.read_corpus("./data/pro_corpus.txt")
        corpus_list = [line.strip().split('  ') for line in pro_corpus if line.strip()]
        del pro_corpus
        self.init_sequence(corpus_list)

    def init_sequence(self, corpus_list):
        words_seq = [[word.split('/')[0] for word in words] for words in corpus_list]
        pos_seq = [[word.split('/')[1] for word in words] for words in corpus_list]
        tag_seq = [[self.pos_2_tag(p) for p in pos] for pos in pos_seq]
        self.pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                         for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
        self.tag_seq = [[[self.perform_tag(tag_seq[index][i], w) for w in range(len(words_seq[index][i]))]
                         for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
        self.pos_seq = [['un'] + [self.perform_pos(p) for pos in pos_seq for p in pos] + ['un'] for pos_seq in
                        self.pos_seq]
        self.tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in self.tag_seq]
        self.word_seq = [['<BOS>'] + [w for word in word_seq for w in word] + ['<EOS>'] for word_seq in words_seq]
        print("-> 完成字序列、词性序列、标记序列的初始化")

    def pos_2_tag(self, pos):
        return tag_mean_map[pos] if pos in tag_mean_map else '0'

    def perform_tag(self, tag, index):
        if index == 0 and tag != '0':
            return 'B_{}'.format(tag)
        elif tag != '0':
            return 'I_{}'.format(tag)
        else:
            return tag

    def perform_pos(self, pos):
        if pos in tag_mean_map.keys() and pos != 't':
            return 'n'
        else:
            return pos

    def generator(self):
        print("-> 以 {0} 的窗口大小，分割字序列".format(3))
        word_grams = [self.segment_by_window(word_list) for word_list in self.word_seq]
        print("-> 根据特征模板，提取特征")
        features = self.feature_extractor(word_grams)

        return features, self.tag_seq

    def segment_by_window(self, word_list=None, window_size=3):
        all_posible_words = []
        begin, end = 0, window_size
        for _ in range(1, len(word_list)):
            if end > len(word_list):
                break
            all_posible_words.append(word_list[begin:end])
            begin += 1
            end += 1
        return all_posible_words

    def feature_extractor(self, word_grams):
        features, features_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {
                    "w-1": word_gram[0],
                    "w": word_gram[1],
                    "w+1": word_gram[2],

                    "w-1:w": word_gram[0] + word_gram[1],
                    "w:w+1": word_gram[1] + word_gram[2],

                    "bias": 1.0
                }

                features.append(feature)
            features_list.append(features)
            features = []
        return features_list


def get_corpus():
    global __corpus
    if not __corpus:
        __corpus = Corpus()
    return __corpus


if __name__ == '__main__':
    c = Corpus()
    c.initialize()
