import csv
import os
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

import numpy as np
from scipy.io import wavfile
import six

from pydub import AudioSegment

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from subprocess import call

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string('input_youtube_id_tsv', '/home/shakil/PSVA/data/output/eval/youtube_eval.txt',
                    'TSV file with lines "<id>\t<start_time>\t<end_time>\t<label>" where '
                    ' and <labels> '
	            'must be an integer list joined with semi-colon ";"')

  flags.DEFINE_string('output_dir','/home/shakil/PSVA/data/output/eval', 'where to save the wav file')

def main(unused_argv):
  print(FLAGS.input_youtube_id_tsv)


  f= open(FLAGS.output_dir+"/video_path_label_eval.txt","w+")
  i=0

  for youtube_id,st_time, end_time, label in csv.reader(open(FLAGS.input_youtube_id_tsv),delimiter='\t'):
    try:    
      i = i+1
      print( youtube_id,st_time, end_time, label)
      #wav_file = FLAGS.output_dir+'/'+str(i)+'.wav'
      wav_file = FLAGS.output_dir+'/'+str(i)
      sh_cmd = 'youtube-dl -v -o \''+wav_file+'.%(ext)s\'' + ' -x --audio-format wav \'https://www.youtube.com/watch?v=' + youtube_id+'\''
      print(sh_cmd)
      #call(sh_cmd);
      os.system(sh_cmd)
      print('--0')
      wav_file = FLAGS.output_dir+'/'+str(i)+'.wav'
      if (os.path.isfile(wav_file)): 
        t1 = float(st_time) * 1000
        t2 = float(end_time) * 1000
        newAudio = AudioSegment.from_wav(wav_file)
        newAudio = newAudio[t1:t2]
        print('--1')
        new_wav_file = FLAGS.output_dir+'/'+str(i)+'_cut_eval.wav'
        newAudio.export(new_wav_file, format="wav") 
        print('--2')
        f.write(new_wav_file+"\t"+st_time+"\t"+end_time+"\t"+label+"\r\n")    
      #examples_batch = vggish_input.wavfile_to_examples(wav_file)
      #print(examples_batch)
    except:
      print('An error occurred.') 



  f.close()

if __name__ == '__main__':
  app.run(main)
