# data/data_processing/app/jp_input.py #
# 输入app的句子为完整的句子，我们需要对它进行分词，我们使用的工具是mecab
# 假设输入的句子在model/jp2zh_model/input.jp，处理好的句子存入model/jp2zh_model/my_infer_file.jp，然后输入到模型中进行翻译
# 需要安装MeCab

import MeCab
# coding = utf-8

def HandleWithMecab():
    file1 = open('../../../model/jp2zh_model/input.jp', 'r', encoding='utf-8') # 要分词的文件
    file2 = open('../../../model/jp2zh_model/my_infer_file.jp', 'w', encoding='utf-8') # 生成有空格的文件
    mecab = MeCab.Tagger("-Owakati")
    try:
        for line in file1.readlines():
            newline = mecab.parse(line)
            file2.write(newline)
    finally:
        file1.close()
        file2.close()

if __name__ == '__main__':
    HandleWithMecab()

