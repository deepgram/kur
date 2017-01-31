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
import logging
import mimetypes
import urllib.request
import urllib.parse
import uuid
import json
import io
from collections import namedtuple

import tqdm

logger = logging.getLogger(__name__)

###############################################################################
UploadFile = namedtuple('UploadFile', 'filename')
UploadFileData = namedtuple('UploadFileData', ('filename', 'data'))

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

###############################################################################
def prepare_json(data):
	""" Prepares a JSON-encoded HTTP message.
	"""
	data = json.dumps(data)
	data = data.encode('utf-8')
	header = {
		'Content-type' : 'application/json'
	}
	return (data, header)

###############################################################################
def prepare_multipart(data):
	""" Prepares an HTTP multipart message.
	"""
	# This is where we'll store our message as we build it.
	buffer = io.BytesIO()

	# Create our boundary ID.
	boundary = uuid.uuid4().hex

	# Loop over all entries in data.
	for k, v in data.items():

		# Do we need to upload a file?
		if isinstance(v, (UploadFile, UploadFileData)):
			buffer.write(
				'--{}\r\n'
				'Content-Disposition: form-data; name="{}"; filename="{}"\r\n'
				'Content-Type: {}\r\n'
				'\r\n'
					.format(
						boundary,
						urllib.parse.quote(k),
						urllib.parse.quote(
							os.path.basename(v.filename)
						),
						mimetypes.guess_type(v.filename)[0] or
							'application/octet-stream'
					).encode('utf-8')
			)
			if isinstance(v, UploadFile):
				with open(v.filename, 'rb') as fh:
					buffer.write(fh.read())
			else:
				if not isinstance(v.data, bytes):
					buffer.write(v.data.encode('utf-8'))
				else:
					buffer.write(v.data)
			buffer.write(b'\r\n')

		# Or just upload a simple value?
		else:
			buffer.write(
				'--{}\r\n'
				'Content-Disposition: form-data; name="{}"\r\n'
				'\r\n'
				'{}\r\n'
					.format(boundary, urllib.parse.quote(k), v).encode('utf-8')
			)

	# Write out the footer.
	buffer.write('--{}--\r\n'.format(boundary).encode('utf-8'))

	# Return everything.
	data = buffer.getvalue()
	header = {
		'Content-type' : 'multipart/form-data; boundary={}'.format(boundary)
	}

	return (data, header)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
