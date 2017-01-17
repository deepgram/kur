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

import os
import hashlib
import tempfile
import logging
import urllib.request

import tqdm

logger = logging.getLogger(__name__)

###############################################################################
def get_hash(path):
	""" Returns the SHA256 hash of a file.

		# Arguments

		path: str. Path to the file to produce a hash for.

		# Return value

		64-character lowercase string of hexadecimal digits representing the
		32-byte SHA256 hash of the file content.
	"""
	with open(path, 'rb') as fh:
		data = fh.read()
	sha = hashlib.sha256()
	sha.update(data)
	return sha.hexdigest()

###############################################################################
def download(url, target):
	""" Downloads a URL.
	"""
	response = urllib.request.urlopen(url)
	with open(target, 'wb') as fh:
		with tqdm.tqdm(
			total=int(response.info().get('Content-Length')),
			unit='bytes',
			unit_scale=True,
			desc='Downloading'
		) as pbar:
			while True:
				chunk = response.read(8192)
				if not chunk:
					break
				fh.write(chunk)
				pbar.update(len(chunk))

	logger.info('File downloaded: %s', target)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
