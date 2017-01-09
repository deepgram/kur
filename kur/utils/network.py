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

import tqdm
import urllib.request

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
def do_download(url, target):
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

###############################################################################
def download_file(url, sha256=None, target_dir=None):
	""" Downloads a URL to the system temporary directory.

		# Arguments

		url: str. The URL of the resource to download.
		sha256: str or None (default: None). The SHA256 hash of the resource
			content, or None to skip verification.
		target_dir: str or None (default: None). The target directory to store
			the file in. If it is None, it defaults to the system temp
			directory.

		# Return value

		The path of the file on the local system.

		# Notes

		This will only download the file if it doesn't already exist or if its
		checksum fails. If the checksum fails after a download, an exception is
		raised.
	"""

	if target_dir is None:
		target_dir = tempfile.gettempdir()

	if sha256 is not None:
		sha256 = sha256.lower()

	_, filename = os.path.split(url)
	target = os.path.join(target_dir, filename)

	# If the file already exists locally, verify its contents.
	if os.path.isfile(target):
		if sha256 is not None:
			if get_hash(target) == sha256:
				logger.info('File already present (checksum passed): %s',
					target)
				return target

			logger.info('Corrupt file present. Re-downloading: %s', url)
		else:
			logger.info('File already present (skipping checksum): %s', target)
			return target
	else:
		logger.info('File does not exist. Downloading: %s', url)

	do_download(url, target)

	if sha256 is not None:
		if get_hash(target) != sha256:
			raise ValueError('Failed integrity check.')

	return target

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
