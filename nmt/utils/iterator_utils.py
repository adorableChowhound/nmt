# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""For loading data into NMT models."""
# 数据预处理过程

from __future__ import print_function

import collections

import tensorflow as tf

from ..utils import vocab_utils


__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None,
                       use_char_encode=False):
  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])

  if use_char_encode:
    # Convert the word strings to character ids
    src_dataset = src_dataset.map(
        lambda src: tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]))
  else:
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

  # Add in the word counts.
  if use_char_encode:
    src_dataset = src_dataset.map(
        lambda src: (src,
                     tf.to_int32(
                         tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN)))
  else:
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)

# 训练数据的处理代码
# 超参数详解(https://github.com/luozhouyang/csdn-blogs/blob/master/tensorflow_nmt/tensorflow_nmt_hparams.md)
def get_iterator(src_dataset,                       # 源数据集
                 tgt_dataset,                       # 目标数据集
                 src_vocab_table,                   # 源数据单词查找表，单词和int类型数据的对应表
                 tgt_vocab_table,                   # 目标数据单词查找表，单词和int类型数据的对应表
                 batch_size,                        # 批大小
                 sos,                               # 句子开始标记
                 eos,                               # 句子结尾标记
                 random_seed,                       # 随机种子，用来打乱数据集
                 num_buckets,                       # 桶数量
                 src_max_len=None,                  # 源数据最大长度
                 tgt_max_len=None,                  # 目标数据最大长度
                 num_parallel_calls=4,              # 并发处理数据的并发数
                 output_buffer_size=None,           # 输出缓冲区大小
                 skip_count=None,                   # 跳过数据行数
                 num_shards=1,                      # 将数据集分片的数量，分布式训练中有用
                 shard_index=0,                     # 数据集分片后的id
                 reshuffle_each_iteration=True,     # 是否每次迭代都重新打乱顺序
                 use_char_encode=False):
# 1. 数据集的处理过程
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  # 将sos和eos标记表示成一个整数
  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  # 将sos和eos标记表示成一个整数
  # 用改整数来表示这两个标记，并且将这两个整数转型为int32类型
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  # 通过zip操作将源数据集和目标数据集合并在一起
  # 张量变化举例： [src_dataset] + [tgt_dataset] ---> [src_dataset, tgt_dataset]
  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  # 数据集分片，分布式训练的时候可以分片来提高训练速度
  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  # 跳过数据，比如一些文件的头尾信息行
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  # 随机打乱数据，切断相邻数据之间的联系
  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  # 将每一行数据，根据“空格”切分开来
  # 这个步骤可以并发处理，用num_parallel_calls指定并发量
  # 通过prefetch来预获取一定数据到缓冲区，提升数据吞吐能力
  # 张量变化举例： ['上海　浦东', '上海　浦东'] ---> [['上海', '浦东'], ['上海', '浦东']]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  # 过滤掉长度为0的数据
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # 限制源数据最大长度
  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # 限制目标数据的最大长度
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  # 通过map操作将字符串转换为数字
  # 张量变化举例： [['上海', '浦东'], ['上海', '浦东']] ---> [[1, 2], [1, 2]]
  if use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.reshape(vocab_utils.tokens_to_bytes(src), [-1]),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  # 给目标数据加上sos, eos标记
  # 张量变化举例： [[1, 2], [1, 2]] ---> [[1, 2], [sos_id, 1, 2], [1, 2, eos_id]]
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  # 增加长度信息
  if use_char_encode:
    # 张量变化举例 [[1, 2], [sos_id, 1, 2], [1, 2, eos_id]] ---> [[1, 2], [sos_id, 1, 2], [1, 2, eos_id], [src_size], [tgt_size]]
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out,
            tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
            tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

# 2. 数据对齐处理
  # 数据的对齐, 并且将数据集按照batch_size完成分批
  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  # x: dataset 对象
  def batching_func(x):
    return x.padded_batch(                              # 对齐数据的同时，也将数据集按照batch_size进行分批
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        # 对齐数据的形状
        padded_shapes=(
            tf.TensorShape([None]),  # src              # 因为数据长度不定，因此设置None
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len              # 数据长度张量，实际上不需要对齐
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        # 对齐数据的值
        padding_values=(
            src_eos_id,  # src                          # 用src_eos_id填充到 src 的末尾
            tgt_eos_id,  # tgt_input                    # 用tgt_eos_id填充到 tgt_input 的末尾
            tgt_eos_id,  # tgt_output                   # 用tgt_eos_id填充到 tgt_output 的末尾
            0,  # src_len -- unused
            0))  # tgt_len -- unused

# 3. num_buckets分桶的作用

  # 判断我们指定的参数num_buckets是否大于１
        # 如果是那么就进入分桶的过程
        # 如果不满足条件就直接做对齐操作
  if num_buckets > 1:

    # key_func: 将我们的数据集(由源数据和目标数据成对组成)按照一定的方式进行分类
    # 根据我们数据集每一行的数据长度，将它放到合适的桶里面去，然后返回该数据所在桶的索引
    # 目的：相似长度的数据放在一起，能够提升计算效率
    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      # 根据src_max_len来计算bucket_width
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    # reduce_func：把刚刚分桶好的数据，做一个对齐
    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    # 此时数据集就已经成为了一个对齐（也就是说有固定长度）的数据集了

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  # 使用迭代器，从处理好的数据集获取一批一批的数据来训练
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())             # get_next(): 获取之前我们处理好的批量数据
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
