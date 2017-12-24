from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import gmtime, strftime

import tensorflow as tf


tf.app.flags.DEFINE_integer("task_index", None, "Task index, should be >= 0.")
tf.app.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.app.flags.DEFINE_string("ps_hosts", None, "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", None, "Comma-separated list of hostname:port pairs")


def main():
  s = tf.constant("Hello, Tensorflow!")
  print(tf.Session().run(s))
  import os
  print('git repo', os.listdir('/opt/tf-models'))
  print('module dir', os.path.dirname(os.path.realpath(__file__)))

  while True:
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

if __name__ == '__main__':
  main()