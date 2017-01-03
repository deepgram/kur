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

import functools
import logging
import os

import numpy

# NOTE: These are special requirements needed for processing audio data.
# pylint: disable=import-error
import magic						# python-magic
import scipy.io.wavfile as wav		# scipy
from pydub import AudioSegment		# pydub
import python_speech_features		# python_speech_features
import stft							# stft
# pylint: enable=import-error

from .. import __homepage__

logger = logging.getLogger(__name__)

###############################################################################
def load_wav(filename):
	""" Loads a WAV file.
	"""
	rate, sig = wav.read(filename)

	# Flatten stereo to mono
	if len(sig.shape) > 1:
		sig = numpy.array(sig.mean(axis=1), dtype=sig.dtype)

	return {
		'signal' : sig,
		'sample_rate' : rate,
		'sample_width' : sig.dtype.itemsize * 8,
		'channels' : 1
	}

###############################################################################
def load_pydub(filename):
	""" Loads an MP3 or FLAC file.
	"""
	data = AudioSegment.from_file(filename)

	if data.channels > 1:
		data = functools.reduce(
			lambda x, y: x.overlay(y),
			data.split_to_mono()
		)

	raw = data.get_array_of_samples()
	raw = numpy.frombuffer(raw, dtype=raw.typecode)

	return {
		'signal' : raw,
		'sample_rate' : data.frame_rate,
		'sample_width' : data.sample_width * 8,
		'channels' : data.channels
	}

###############################################################################
def load_audio(filename):
	""" Loads an audio file.

		# Arguments

		filename: str. The file to load.

		# Return value

		A dictionary with the following keys/values:
			- signal: A numpy array of raw audio data
			- sample_rate: The sample rate, in Hz
			- sample_width: The sample width, in bits
			- channels: The number of channels (always 1--see below)

		# Notes

		The returned audio data is mono. If the source file was stereo, it is
		downmixed to mono before being returned.
	"""
	# Read off magic numbers and return MIME types
	mime_magic = magic.Magic(mime=True)
	ftype = mime_magic.from_file(filename)
	if isinstance(ftype, bytes):
		ftype = ftype.decode('utf-8')

	# Define the loaders for each supported file type.
	loaders = {
		'audio/x-wav' : load_wav,
		'audio/mpeg' : load_pydub,
		'audio/x-flac' : load_pydub
	}

	if ftype not in loaders:
		raise IOError('No loader available for filetype ({}) for file: {}'
			.format(ftype, filename))

	try:
		return loaders[ftype](filename)
	except:
		logger.exception('Failed to load audio file: %s. Its MIME type is: '
			'%s. The most likely cause is that you do not have FFMPEG '
			'installed. Check out our troubleshooting guide for more '
			'information: %s', filename, ftype,
			os.path.join(
				__homepage__,
				'troubleshooting.html#couldn-t-find-ffmpeg-or-avconv'
			)
		)
		raise

###############################################################################
def get_audio_features(audio, feature_type, **kwargs):
	""" Returns audio features.

		# Arguments

		audio: dict or str. If dict, it should have keys, values as returned by
			`load_audio()`. If str, it should be a file that will be passed to
			`load_audio()`.
		feature_type. str. One of:
			- raw: Returns raw audio data (1-dimensional)
			- mfcc: Returns MFCC features
			- spec: Returns a spectrogram
		kwargs. Additional arguments that depend on `feature_type`:
			- For 'raw': no additional parameters
			- For 'mfcc':
				- features: int (default: 13). The number of MFCC features to
				  keep.
				- low_freq: int (default: None). The low-frequency cutoff.
				- high_freq: int (default: None). The high-frequency cutoff.
			- For 'spec':
				- low_freq: int (default: None). The low-frequency cutoff.
				- high_freq: int (default: None). The high-frequency cutoff.
	"""
	if isinstance(audio, str):
		audio = load_audio(audio)

	if feature_type == 'raw':
		return audio['signal']

	elif feature_type == 'mfcc':
		num_features = kwargs.get('features') or 13
		return python_speech_features.mfcc(
			audio['signal'],
			audio['sample_rate'],
			numcep=num_features,
			nfilt=num_features*2,
			lowfreq=kwargs.get('low_freq') or 0,
			highfreq=kwargs.get('high_freq') or None
		)

	elif feature_type == 'spec':
		# Window size, in seconds
		window_size = 0.020
		# Step size, in seconds
		step_size = 0.010

		# Calculate the spectrogram
		spec = stft.spectrogram(
			audio['signal'],
			framelength=int(window_size * audio['sample_rate']),
			hopsize=int(step_size * audio['sample_rate'])
		)
		# At this point, `spec` is shape (frequency, time).

		# Apply frequency cutoffs, if necessary
		low_freq = kwargs.get('low_freq')
		high_freq = kwargs.get('high_freq')
		if low_freq or high_freq:
			# Number of frequency bins
			num_bins = spec.shape[0]
			# Width of each frequency bin.
			delta_freq = 1 / window_size

			# Calculate the bin that a frequency would fall into.
			get_bin = lambda f, alt: \
				(
					min(
						max(int(f / delta_freq + 0.5), 0), num_bins
					)
					if f else alt
				)
			spec = spec[get_bin(low_freq, 0):get_bin(high_freq, num_bins)]

		# Format `spec` as (time, frequency)
		spec = spec.T
		return spec

	else:
		raise ValueError('Unsupported feature type: {}'.format(feature_type))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
