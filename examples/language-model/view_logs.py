"""
Copyright 2016 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kur.loggers import BinaryLogger
import sys


log_dir_name = sys.argv[1]

training_loss = BinaryLogger.load_column(log_dir_name, 'training_loss_total')
validation_loss = BinaryLogger.load_column(log_dir_name, 'validation_loss_total')

plt.xlabel('Epoch')
plt.ylabel('Loss')
epoch = list(range(1, 1+len(training_loss)))
t_line, = plt.plot(epoch, training_loss, 'co-', label='Training Loss')
v_line, = plt.plot(epoch, validation_loss, 'mo-', label='Validation Loss')
plt.legend(handles=[t_line, v_line])
plt.savefig('loss.pdf')
