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
  flags.DEFINE_string('input_youtube_id','vgv7iQ47z0o',
                    ' ')

  flags.DEFINE_string('output_dir','/Users/julia/PSVA/data/output/game', 'where to save the wav file')

def main(unused_argv):


  f= open(FLAGS.output_dir+"/game_video_path.txt","w+")
  
  i=0
  youtube_id = "vgv7iQ47z0o"
  wav_file = FLAGS.output_dir+'/'+str(i)
  sh_cmd = 'youtube-dl --no-check-certificate -v -o \''+wav_file+'.%(ext)s\'' + ' -x --audio-format wav \'https://www.youtube.com/watch?v=' + youtube_id+'\''
  print(sh_cmd)

  os.system(sh_cmd)
  print('--0')
  for j in range(1,3600,10) :
    try:    
      
      st_time = j
      end_time = j+10
      label = 1
      print( youtube_id,st_time, end_time, label)
      #wav_file = FLAGS.output_dir+'/'+str(i)+'.wav'

      wav_file = FLAGS.output_dir+'/'+str(i)+'.wav'
      print(wav_file)
      if (os.path.isfile(wav_file)): 
        t1 = float(st_time) * 1000
        t2 = float(end_time) * 1000
        newAudio = AudioSegment.from_wav(wav_file)
        newAudio = newAudio[t1:t2]
        print('--1')
        new_wav_file = FLAGS.output_dir+'/'+str(j)+'_cut.wav'
        newAudio.export(new_wav_file, format="wav") 
        print('--2')
        f.write(new_wav_file+"\t"+str(st_time)+"\t"+str(end_time)+"\t"+str(label)+"\r\n")
        print('--3')

        #rm_cmd = 'rm -rf '+wav_file
        #os.system(rm_cmd)
      #examples_batch = vggish_input.wavfile_to_examples(wav_file)
      #print(examples_batch)
    except:
      print('An error occurred.') 



  f.close()

if __name__ == '__main__':
  app.run(main)
