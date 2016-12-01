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

import gzip
import os
import struct
import warnings
import numpy

###############################################################################
def _read(fh, num_bytes):
	""" Reads a number of bytes, throwing an exception if that many bytes isn't
		available.
	"""
	result = fh.read(num_bytes)
	if len(result) != num_bytes:
		raise IOError('Unexpected end-of-file.')
	return result

###############################################################################
def save(filename, data, append=False):
	""" Saves data to an IDX file.
	"""
	if not isinstance(data, numpy.ndarray):
		data = numpy.array(data)

	data_info = {
		'f' : (0x0D, '>f4'),
		'u' : (0x0C, '>i4'),
		'i' : (0x0C, '>i4')
	}
	data_type = data_info.get(data.dtype.kind, None)
	if not data_type:
		raise ValueError('Data type is not understood: {}'.format(data.dtype))

	needs_copy = any((
		data.dtype.itemsize != int(data_type[1][-1]),
		data.dtype.byteorder != data_type[1][0],
		data.dtype.kind != data_type[1][-2]
	))
	if needs_copy:
		data = data.astype(data_type[1])

	if os.path.isfile(filename) and append:

		# Verify the header and update it.
		with open(filename, 'r+b') as fh:
			magic = _read(fh, 4)
			if magic[0] != 0 or magic[1] != 0:
				raise IOError('Bad header in IDX file.')
			if magic[2] != data_type[0]:
				raise IOError('Data type mis-match.')
			if magic[3] != data.ndim:
				raise IOError('Data dimensionality mis-match.')

			mark = fh.tell()
			shape = tuple(
				struct.unpack('>I', _read(fh, 4))[0]
				for _ in range(data.ndim)
			)

			if shape[1:] != data.shape[1:]:
				raise IOError('Data shape mis-match.')

			fh.seek(mark)
			fh.write(struct.pack('>I', shape[0] + data.shape[0]))

	else:

		# Write the new header.
		with open(filename, 'wb') as fh:
			fh.write(bytes([0, 0, data_type[0], data.ndim]))
			for shape in data.shape:
				fh.write(struct.pack('>I', shape))

	# Write the data.
	with open(filename, 'ab') as fh:
		fh.write(data.tostring())

###############################################################################
def load(filename):
	""" Loads an IDX file.

		Reference for file format: http://yann.lecun.com/exdb/mnist/
	"""

	def opener(filename):
		""" Returns a file handle to a binary file, optionally gzipped.
		"""
		with open(filename, 'rb') as fh:
			magic = fh.read(2)

		# Test if the GZIP magic number is present.
		# Reference: http://www.gzip.org/zlib/rfc-gzip.html#file-format
		if magic == b'\x1f\x8b':
			return gzip.open
		return open

	with opener(filename)(filename, 'rb') as fh:
		magic = _read(fh, 4)
		if magic[0] != 0 or magic[1] != 0:
			raise IOError('Bad header in IDX file.')

		data_info = {
			0x08 : ('B', 1),
			0x09 : ('b', 1),
			0x0B : ('h', 2),
			0x0C : ('i', 4),
			0x0D : ('f', 4),
			0x0E : ('d', 8)
		}

		data_type = data_info.get(magic[2])
		if data_type is None:
			raise IOError(
				'Bad data type in IDX header: 0x{:02X}'.format(magic[2]))

		ndim = magic[3]
		shape = tuple(
			struct.unpack('>I', _read(fh, 4))[0] for _ in range(ndim)
		)

		num_entries = numpy.product(shape)

		entry_type, entry_size = data_type
		total_size = entry_size * num_entries

		data = _read(fh, total_size)

		try:
			_read(fh, 1)
		except IOError:
			pass
		else:
			warnings.warn('IDX file has additional data at the end of the '
				'file. Ignoring this data...')

	result = numpy.fromstring(data, dtype='>{}'.format(entry_type))
	result = result.reshape(shape)
	return result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
