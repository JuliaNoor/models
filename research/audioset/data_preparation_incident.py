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
  flags.DEFINE_string('input', '/home/shakil/PSVA/data/Google_audioset/unbalanced_train_segments.csv',
                    'CSV file with lines <id>,<start_time>,<end_time>,"<label>" where '
                    ' and <labels> ')
	            

  flags.DEFINE_string('output_dir', '/home/shakil/PSVA/data/output/incident', 'where to save the tsv file')

def main(unused_argv):
  print(FLAGS.input)


  f= open(FLAGS.output_dir+"/youtube_incident_train.txt","w+")
  print(FLAGS.output_dir+"/youtube_incident_train.txt")

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
      label = ll[3].strip().replace("\"","")
      #print(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label)     
      label_list = label.split(',')
      label_set = set(label_list)
      #print(len(label_set))

      positive_target = {'/m/03qc9zr', '/m/07p6fty', '/t/dd00135', '/m/03qc9zr', '/m/03j1ly', '/m/04qvtq', '/m/012n7d', '/m/012ndj', '/m/01y3hg', '/m/0c3f7m', '/m/014zdl', '/m/032s66', '/m/04zjc'}
      
      negative_target = {'/m/09x0r', '/t/dd00003', '/t/dd00004', '/t/dd00005', '/t/dd00031', '/m/07r660_', '/m/0342h', '/m/02sgy', '/m/07pkxdp', '/m/05kq4', '/m/07qjznl', '/m/081rb'}

      if len(negative_target.intersection(label_set)) > 0:
        label ="0"
        count = count + 1
        if count <=15000:
          f.write(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label+ '\n')
      elif len(positive_target.intersection(label_set))> 0:
        label = "1"
        f.write(youtube_id + '\t' + st_time + '\t' + end_time + '\t' + label+ '\n')
      
        
  f.close()

if __name__ == '__main__':
  app.run(main)
