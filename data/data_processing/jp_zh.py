# 从数据库得到数据，生成的数据存入data/jp_zh_data/

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import codecs
import collections
from operator import itemgetter
import pymysql
import MeCab

# 生成训练集train，验证集dev，测试集test
def getData(path, lang):
    # 请输入自己的数据库
    connect = pymysql.Connect(
        host="",
        port=3306,
        user="",
        passwd="",
        db="jp",
        charset='utf8'
    )
    print("写入中，请等待……")
    cursor = connect.cursor()

    if lang == "zh":
        sql = "select cn from jp_cn"
    elif lang == "jp":
        sql = "select jp from jp_cn"
    else:
        print("Something wrong with lang for sql")

    cursor.execute(sql)
    number = cursor.fetchall()
    loan_count = 0

    if lang == "zh":
        # 英文单词或数字按照独立单词处理，其余中文以汉字形式分割
        eng_and_num = 'abcdefghijklmnopqrstuvwxyz0123456789'

        file1 = path + "train.zh"
        file2 = path + "dev.zh"
        file3 = path + "test.zh"
        fp1 = open(file1, "w")
        fp2 = open(file2, "w")
        fp3 = open(file3, "w")

        for loanNumber in number:
            output = ''
            buffer = ''
            for s in loanNumber[0]:
                if s in eng_and_num or s in eng_and_num.upper():
                    buffer += s
                else:
                    if buffer:
                        output += buffer
                        output += " "
                    buffer = ''
                    if s == '\n':
                        output += s
                    else:
                        output += s
                        output += " "

            if buffer:
                output += buffer

            newline = output

            loan_count += 1
            if (loan_count < 200000):
                fp1.write(newline)
            elif (loan_count < 220000):
                fp2.write(newline)
            else:
                fp3.write(newline)

    elif lang == "jp":
        # 用mecab对处理日语
        mecab = MeCab.Tagger("-Owakati")

        file1 = path + "train.jp"
        file2 = path + "dev.jp"
        file3 = path + "test.jp"
        fp1 = open(file1, "w")
        fp2 = open(file2, "w")
        fp3 = open(file3, "w")

        for loanNumber in number:
            newline = mecab.parse(str(loanNumber[0]) + "\n")
            loan_count += 1
            if (loan_count < 200000):
                fp1.write(newline)
            elif (loan_count < 220000):
                fp2.write(newline)
            else:
                fp3.write(newline)
    else:
        print("Cannot take sentence of lang")

    fp1.close()
    fp2.close()
    fp3.close()
    cursor.close()
    connect.close()
    print("写入完成,共写入%d条数据……" % loan_count)


def deal(lang):
    # 可能需要新建这个文件夹，不然会报错
    ROOT_PATH = "../jp_zh_data/"
    getData(ROOT_PATH, lang)

    # 生成词汇表
    # 训练集数据文件
    if lang == "zh":
        RAW_DATA = ROOT_PATH + "train.zh"
        # 输出的词汇表文件
        VOCAB_OUTPUT = ROOT_PATH + "vocab.zh"
        # 中文词汇表单词个数
        VOCAB_SIZE = 6000
    elif lang == "jp":
        RAW_DATA = ROOT_PATH + "train.jp"
        VOCAB_OUTPUT = ROOT_PATH + "vocab.jp"
        VOCAB_SIZE = 10000
    else:
        print("Cannot find lang for vocab")

    # 统计单词出现的频率
    counter = collections.Counter()
    with codecs.open(RAW_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # 按照词频顺序对单词进行排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 在后面处理机器翻译数据时，需要将"<unk>"和句子起始符"<sos>"，句子终止符"<eos>"加入词汇表，并从词汇表中删除低频词汇。
    sorted_words = ["<unk>", "<s>", "</s>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]

    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + "\n")


if __name__ == "__main__":
    lang = ["zh", "jp"]
    for i in lang:
        deal(i)
