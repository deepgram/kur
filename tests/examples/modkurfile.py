"""
Copyright 2017 Deepgram

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

def modify_kurfile(data):

	for k in ('train', 'validate', 'test', 'evaluate'):
		if k not in data:
			continue

		if 'weights' in data[k]:
			del data[k]['weights']

		if 'provider' not in data[k]:
			data[k]['provider'] = {}
		data[k]['provider']['num_batches'] = 1
		data[k]['provider']['batch_size'] = 2

	if 'train' in data:
		if 'checkpoint' in data['train']:
			del data['train']['checkpoint']
		data['train']['epochs'] = 2
		if 'log' in data['train']:
			del data['train']['log']

	if 'evaluate' in data:
		if 'destination' in data['evaluate']:
			del data['evaluate']['destination']

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
