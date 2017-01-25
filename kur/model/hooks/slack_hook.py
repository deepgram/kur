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

import logging
import json

import urllib.request

from . import TrainingHook

logger = logging.getLogger(__name__)

###############################################################################
class SlackHook(TrainingHook):
	""" Training hook for posting to Slack.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the training hook.
		"""
		return 'slack'

	###########################################################################
	def __init__(self, channel, url, user=None, icon=None, title=None,
		*args, **kwargs):
		""" Creates a new Slack training hook.
		"""
		super().__init__(*args, **kwargs)

		self.username = user or 'kur'
		self.icon = icon or 'dragon'
		self.url = url
		self.channel = channel
		self.title = title

	###########################################################################
	def notify(self, status, info=None):
		""" Sends the Slack message.
		"""

		logger.debug('Slack hook received message.')

		info = info or {}

		if status is TrainingHook.EPOCH_END:
			epoch = info.pop('epoch', None)
			total_epochs = info.pop('total_epochs', None)
			text = 'Finished epoch {} of {}.'.format(epoch+1, total_epochs)
		elif status is TrainingHook.TRAINING_END:
			text = 'Training has ended.'
		elif status is TrainingHook.TRAINING_START:
			text = 'Started training.'
		else:
			return

		if self.title is not None:
			text = '{}: {}'.format(self.title, text)

		if info:
			text = '{} More information: {}'.format(text,
				', '.join('{} = {}'.format(k, v) for k, v in info.items())
			)

		data = {
			'channel' : self.channel,
			'username' : self.username,
			'icon_emoji' : ':{}:'.format(self.icon),
			'text' : text
		}
		data = json.dumps(data)
		data = data.encode('utf-8')

		request = urllib.request.Request(
			self.url,
			data=data,
			headers={
				'Content-type' : 'application/json'
			},
			method='POST'
		)
		director = urllib.request.build_opener()

		try:
			response = director.open(request)
		except:									# pylint: disable=bare-except
			logger.exception('Failed to connect to Slack. Make sure the URL '
				'and channel are correct. If the channel was newly created, '
				'it might take a little time for Slack to catch up.')
		else:
			if response.code != 200:
				logger.error('Failed to post Slack notification. Make sure '
				'the URL and channel are correct. If the channel was newly '
				'created, it might take a little time for Slack to catch up.')

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
