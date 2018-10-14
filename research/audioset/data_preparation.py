import csv
import os
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

import numpy as np
from scipy.io import wavfile
import six

from pydub import AudioSegment


import re
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from subprocess import call

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string('input', '/Users/julia/PSVA/data/Google_audioset/balanced_train_segments.csv',
                    'CSV file with lines <id>,<start_time>,<end_time>,"<label>" where '
                    ' and <labels> ')
	            

  flags.DEFINE_string('output_dir','/Users/julia/PSVA/data/output/balanced/', 'where to save the tsv file')

def main(unused_argv):
  print(FLAGS.input)


  f= open(FLAGS.output_dir+"/youtube_balanced_train.txt","w+")

#  for youtube_id,st_time, end_time, label in csv.reader(open(FLAGS.input),delimiter=','):
  count = 0
  for x in open(FLAGS.input):
    
     # print(x) 
      if x.startswith('#'):
        continue
    #(youtube_id,st_time, end_time, label) = csv.reader(x,delimiter=',', quotechar='"',quoting=csv.QUOTE_ALL, skipinitialspace=True)
      ll = re.split(', ',x)
      youtube_id = ll[0]
      st_time = ll[1]
      end_time = ll[2]
      label = ll[3]
      #print(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label)     
      target = "/m/03qc9zr"
      if label.find(target)== -1:
        label ="0"
        count = count + 1
        if count <=1000:
          f.write(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label+ '\n')
      else:
        label = "1"
        f.write(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label+ '\n')

        
  f.close()

if __name__ == '__main__':
  app.run(main)
