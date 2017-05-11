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

import os
import logging

import urllib.request

from . import TrainingHook, EvaluationHook
from .transcript import Transcript
from ...utils import prepare_multipart, prepare_json, UploadFile
from ...supplier import SpeechRecognitionSupplier

logger = logging.getLogger(__name__)

###############################################################################
class SlackHook(TrainingHook, EvaluationHook):
	""" Hook for posting to Slack.
	"""

	FILES_UPLOAD = 'https://slack.com/api/files.upload'

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'slack'

	###########################################################################
	def __init__(self, channel, url=None, user=None, icon=None, title=None,
		token=None, extra_files=None, *args, **kwargs):
		""" Creates a new Slack hook.
		"""
		super().__init__(*args, **kwargs)

		if extra_files and isinstance(extra_files, str):
			extra_files = [extra_files]
		elif extra_files is None:
			extra_files = []

		self.username = user or 'kur'
		self.icon = icon or 'dragon'
		self.url = url
		self.channel = channel
		self.title = title
		self.token = token
		self.extra_files = extra_files

		if url is None and token is None:
			raise ValueError('Slack hook requires at least one of "url" or '
				'"token" is defined.')

		if extra_files and token is None:
			raise ValueError('A Slack "token" is required to upload files.')

	###########################################################################
	def upload_message(self, filename, text=None):
		""" Sends a message to Slack and uploads a file.
		"""
		logger.debug('Upload file: %s', filename)

		if not self.token:
			logger.warning('Skipping this upload -- "token" was not defined.')
			return

		if not os.path.isfile(filename):
			logger.warning('Skipping this upload -- file does not exist: %s',
				filename)
			return

		data = {
			'token' : self.token,
			'file' : UploadFile(filename),
			'filename' : filename,
			'channels' : self.channel
		}

		if text:
			data['initial_comment'] = text

		if self.title is not None:
			data['title'] = self.title

		data, header = prepare_multipart(data)
		self._submit(SlackHook.FILES_UPLOAD, data, header)

	###########################################################################
	def send_message(self, text, info=None):
		""" Sends a message to Slack.
		"""
		logger.debug('Sending Slack message.')

		if not self.url:
			logger.warning('Skipping this message -- "url" was not defined.')
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

		data, header = prepare_json(data)
		self._submit(self.url, data, header)

	###########################################################################
	def _submit(self, url, data, header):
		""" Submits a POST request to Slack.
		"""
		request = urllib.request.Request(
			url,
			data=data,
			headers=header,
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

	###########################################################################
	def notify(self, status, log=None, info=None):
		""" Sends the Slack message.
		"""

		logger.debug('Slack hook received training message.')

		info = info or {}

		if status is TrainingHook.EPOCH_END:
			epoch = info.pop('epoch', None)
			total_epochs = info.pop('total_epochs', None)
			text = 'Finished epoch {} of {}.'.format(epoch, total_epochs)
		elif status is TrainingHook.TRAINING_END:
			text = 'Training has ended.'
		elif status is TrainingHook.TRAINING_START:
			text = 'Started training.'
		else:
			text = None

		if text:
			self.send_message(text, info)

		if status in (
			TrainingHook.EPOCH_END,
			TrainingHook.VALIDATION_END
		):
			self.upload_extra_files()

	###########################################################################
	def upload_extra_files(self):
		""" Uploads `extra_files` to Slack.
		"""
		if self.token:
			for filename in self.extra_files:
				if os.path.isfile(filename):
					self.upload_message(filename)
				else:
					logger.debug('Skipping file upload -- file does not '
						'exist: %s', filename)
		else:
			logger.debug('"token" is not defined. Skipping uploads.')

	###########################################################################
	def apply(self, current, original, model=None):
		""" Sends a Slack message in response to an evaluation hook.
		"""

		logger.debug('Slack hook received non-training message.')

		data, truth = current

		upload = False
		if self.token is not None:
			if isinstance(data, Transcript) \
					and original[1] is not None \
					and 'audio_source' in original[1]:
				path = SpeechRecognitionSupplier.find_audio_path(
					original[1]['audio_source'][0]
				)
				if path is not None:
					upload = True

		text = 'Truth = "{}", Prediction = "{}"'.format(truth, data),
		if upload:
			self.upload_message(path, text)
		elif self.url is not None:
			self.send_message(text)
		else:
			logger.warning('Failed to post to Slack. The "slack" hook was '
				'given enough information for uploading only, and not enough '
				'for message posting. However, the data you are working with '
				'does not provide enough information for file uploading.')

		self.upload_extra_files()

		return current

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
