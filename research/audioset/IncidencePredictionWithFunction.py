import csv
import os
import time

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import numpy as np
from scipy.io import wavfile
import six

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from subprocess import call

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string('input_wav_file', '',
	            'input audio file in wav format')
#  flags.DEFINE_string( 'tfrecord_file', '/mnt/disks/disk-1/data/youtube_video/incident/feature_label_eval.tfrecord',
#		    'Path to a TFRecord file where embeddings will be written.')

  flags.DEFINE_string(
                     'pca_params', 'vggish_pca_params.npz',
    		     'Path to the VGGish PCA parameters file.')

  flags.DEFINE_string(
 			'checkpoint', 'vggish_model.ckpt',
			'Path to the VGGish checkpoint file.')

if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "/Users/atislam/data/train_dir/temp_model/",
                      "The directory to load the model files from.")
  flags.DEFINE_string("checkpoint_file", "",
                      "If provided, this specific checkpoint file will be "
                      "used for inference. Otherwise, the latest checkpoint "
                      "from the train_dir' argument will be used instead.")
  flags.DEFINE_string("output_file", "/Users/atislam/data/test_dir/temp_model/predictions.csv",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", True,
      "If set, then --input_data_pattern must be frame-level features. "
      "Otherwise, --input_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  
  flags.DEFINE_integer(
      "batch_size", 1,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "audio_embedding", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "128", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 2,
                       "How many predictions to output per video.")
  flags.DEFINE_string("model", "FrameLevelLogisticModel",
                       "What type of model")
    


def _int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)

 #return pointer to array of prediction
def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield line[0][0]
    
    

def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch

def inference(video_batch_val,num_frames_batch_val, checkpoint_file, train_dir,out_file_location, batch_size=1, top_k=2):
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, gfile.Open(out_file_location, "w+") as out_file:
    
    if checkpoint_file:
      if not gfile.Exists(checkpoint_file + ".meta"):
        logging.fatal("Unable to find checkpoint file at provided location '%s'" % checkpoint_file)
      latest_checkpoint = checkpoint_file
    else:
      latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("Number of threads -----------------------------------"+str(len(threads))+"------------------")
    num_examples_processed = 0
    start_time = time.time()
    #out_file.write("VideoId,LabelConfidencePairs\n")

    try:      
      #video_id_batch_val, video_batch_val,num_frames_batch_val = sess.run([video_id_batch, video_batch, num_frames_batch])
      predictions_val, = sess.run([predictions_tensor], feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
      now = time.time()
      num_examples_processed += len(video_batch_val)
      num_classes = predictions_val.shape[1]
      logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
      print("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
      video_id_batch_val = np.array(['1'], dtype = bytes)
      ite = format_lines(video_id_batch_val, predictions_val, top_k) #return pointer to array of predicted classes
      
      classes = [line for line in ite]
      return(classes[0]) #returning the prediction of the first sample; ignoring the others assuming there are none
 


    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def extract_n_predict(input_wav_file, pca_params, checkpoint, checkpoint_file, train_dir, output_file):
    print("Input file: " +input_wav_file)

    
    if (os.path.isfile(input_wav_file)):
      examples_batch = vggish_input.wavfile_to_examples(input_wav_file)
      #print(examples_batch)
      pproc = vggish_postprocess.Postprocessor(pca_params)

      with tf.Graph().as_default(), tf.Session() as sess:
       # Define the model in inference mode, load the checkpoint, and
       # locate input and output tensors.
       vggish_slim.define_vggish_slim(training=False)
       vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
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
       num_frames_batch_val = np.array([postprocessed_batch.shape[0]],dtype=np.int32)
    
       video_batch_val = np.zeros((1, 300, 128), dtype=np.float32)
       video_batch_val[0,0:postprocessed_batch.shape[0],:] = utils.Dequantize(postprocessed_batch.astype(float),2,-2)
    

 #  extract_n_predict()
       predicted_class = inference(video_batch_val ,num_frames_batch_val, checkpoint_file, train_dir, output_file)
       return(predicted_class)
      tf.reset_default_graph()

      
def main(unused_argv):
    predicted_class = extract_n_predict(FLAGS.input_wav_file, FLAGS.pca_params, FLAGS.checkpoint, FLAGS.checkpoint_file, FLAGS.train_dir, FLAGS.output_file)
    print(predicted_class)

  


if __name__ == '__main__':
  app.run(main)
