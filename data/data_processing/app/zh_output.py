# data/data_processing/app/zh_output.py #
# 模型输出的句子包含空格，我们把它去掉，再输出给用户
# 假设要处理的句子在model/jp2zh_model/output_infer.zh，最终输出的句子存入model/jp2zh_model/output.zh，然后返回给用户

# coding = utf-8

def clearBlankSpace():
    file1 = open('../../../model/jp2zh_model/output_infer.zh', 'r', encoding='utf-8') # 要去掉空格的文件
    file2 = open('../../../model/jp2zh_model/output.zh', 'w', encoding='utf-8') # 生成没有空格的文件
    eng_and_num = 'abcdefghijklmnopqrstuvwxyz0123456789'
    try:
        for line in file1.readlines():
            newline = line.replace(" ", "")
            file2.write(newline)
    finally:
        file1.close()
        file2.close()

if __name__ == '__main__':
    clearBlankSpace()

