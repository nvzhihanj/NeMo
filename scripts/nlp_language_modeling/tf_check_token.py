import os
import sys
import tensorflow as tf

saved_unprocessed_dataset_filename = "gs://cnn_dailymail_public/mlperf/cnn_dailymail/preprocess/cnn_dailymail-test.tfrecord-00000-of-00001"
saved_processed_dataset_filename = "gs://cnn_dailymail_public/mlperf/cnn_dailymail/seqio_cache_tasks/cnn_dailymail_3.0.0/cnn_dailymail-test.tfrecord-00000-of-00001"
saved_processed_dataset_filename = "gs://cnn_dailymail_public/mlperf/cnn_dailymail/seqio_cache_tasks/cnn_dailymail_3.0.0/cnn_dailymail-validation.tfrecord-00000-of-00001"
saved_processed_dataset_filenames = [saved_processed_dataset_filename]
saved_processed_dataset = tf.data.TFRecordDataset(saved_processed_dataset_filenames)

for saved_processed_record in saved_processed_dataset.take(1):
  saved_processed_example = tf.train.Example()
  saved_processed_example.ParseFromString(saved_processed_record.numpy())
  targets_tokenized = saved_processed_example.features.feature['targets_pretokenized'].bytes_list.value
  targets_tokens = saved_processed_example.features.feature['targets'].int64_list.value
  inputs_pretokenized = saved_processed_example.features.feature['inputs_pretokenized'].bytes_list.value
  inputs_tokens = saved_processed_example.features.feature['inputs'].int64_list.value

  print(inputs_tokens)
  print(inputs_pretokenized)


