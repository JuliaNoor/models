import csv
import os
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

import numpy as np
from scipy.io import wavfile
import six

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from subprocess import call

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string('input_video_label', '/home/shakil/PSVA/data/output/incident/video_path_label_incident.txt',
                    'TSV file with lines "<video_file_path>\t<start_time>\t<end_time>\t<class_label>" where '
                    ' and <labels> '
	            'must be an integer list joined with semi-colon ";"')
  flags.DEFINE_string( 'tfrecord_file', '/home/shakil/PSVA/data/output/incident/feature_label.tfrecord',
		    'Path to a TFRecord file where embeddings will be written.')

  flags.DEFINE_string(
                     'pca_params', 'vggish_pca_params.npz',
    		     'Path to the VGGish PCA parameters file.')

  flags.DEFINE_string(
 			'checkpoint', 'vggish_model.ckpt',
			'Path to the VGGish checkpoint file.')

def _int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)

def main(unused_argv):
  print("Input file: " +FLAGS.input_video_label)
  print("Output tfrecord file: "  + FLAGS.tfrecord_file)

  writer = tf.python_io.TFRecordWriter(
  FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

  for wav_file,st_time, end_time, label in csv.reader(open(FLAGS.input_video_label),delimiter='\t'):
    print( wav_file,st_time, end_time, label)
    if (os.path.isfile(wav_file)): 
      examples_batch = vggish_input.wavfile_to_examples(wav_file)
      #print(examples_batch)
      pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

      with tf.Graph().as_default(), tf.Session() as sess:
       # Define the model in inference mode, load the checkpoint, and
       # locate input and output tensors.
       vggish_slim.define_vggish_slim(training=False)
       vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
       features_tensor = sess.graph.get_tensor_by_name(
          vggish_params.INPUT_TENSOR_NAME)
       embedding_tensor = sess.graph.get_tensor_by_name(
          vggish_params.OUTPUT_TENSOR_NAME)

       # Run inference and postprocessing.
       [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
       #print(embedding_batch)
       postprocessed_batch = pproc.postprocess(embedding_batch)
       #print(postprocessed_batch)

       # Write the postprocessed embeddings as a SequenceExample, in a similar
       # format as the features released in AudioSet. Each row of the batch of
       # embeddings corresponds to roughly a second of audio (96 10ms frames), and
       # the rows are written as a sequence of bytes-valued features, where each
       # feature value contains the 128 bytes of the whitened quantized embedding.
       seq_example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            vggish_params.LABELS_FEATURE_KEY:
                _int64_list_feature(sorted(map(int, label))),
            vggish_params.VIDEO_FILE_KEY_FEATURE_KEY:
                _bytes_feature(_make_bytes(map(ord, wav_file))),
        }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[embedding.tobytes()]))
                            for embedding in postprocessed_batch
                        ]
                    )
            }
        )
       )

       #print(seq_example)
       if writer:
         writer.write(seq_example.SerializeToString())
       
      tf.reset_default_graph()

  if writer:
    writer.close()


if __name__ == '__main__':
  app.run(main)
