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

from .. import __homepage__

logger = logging.getLogger(__name__)

###############################################################################
class DependencyError(Exception):
	""" Exception class for missing dependencies.
	"""
	pass

###############################################################################
def load_wav(filename):
	""" Loads a WAV file.
	"""
	import scipy.io.wavfile as wav				# pylint: disable=import-error
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
	try:
		from pydub import AudioSegment
		data = AudioSegment.from_file(filename)
	except ImportError:
		logger.exception('"pydub" is a required Python dependency for '
			'handling this audio file: %s.', filename)
		raise
	except FileNotFoundError:
		if os.path.isfile(filename):
			raise DependencyError()
		else:
			raise

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
def scale_signal(audio):
	""" Returns an audio signal scaled so that the max/min values are +1/-1.
	"""
	return audio['signal'] / 2**(audio['sample_width'] - 1)

###############################################################################
def get_mime_type(filename):
	""" Returns the MIME type associated with a particular audio file.
	"""
	try:
		import magic
	except ImportError:
		if get_mime_type.warn:
			logger.warning('Python package "magic" could not be loaded, '
				'possibly because system library "libmagic" could not be '
				'found. We are falling back on our own heuristics.')
			get_mime_type.warn = False

		ext = os.path.splitext(filename)[1].lower()
		return {
			'.wav' : 'audio/x-wav',
			'.mp3' : 'audio/mpeg',
			'.flac' : 'audio/x-flac'
		}.get(ext, 'unknown')
	else:
		# Read off magic numbers and return MIME types
		mime_magic = magic.Magic(mime=True)
		ftype = mime_magic.from_file(filename)
		if isinstance(ftype, bytes):
			ftype = ftype.decode('utf-8')
		return ftype

get_mime_type.warn = True

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
	ftype = get_mime_type(filename)

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
	except DependencyError:
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
	except KeyboardInterrupt:
		raise
	except:
		logger.exception('Failed to load audio file: %s. Its MIME type is: '
			'%s. Check out our troubleshooting guide for more information: %s',
			filename, ftype, os.path.join(__homepage__, 'troubleshooting.html')
		)
		raise

###############################################################################
def get_audio_features(audio, feature_type, on_error=None, **kwargs):
	""" Returns audio features.

		# Arguments

		audio: dict or str. If dict, it should have keys, values as returned by
			`load_audio()`. If str, it should be a file that will be passed to
			`load_audio()`.
		feature_type. str. One of:
			- raw: Returns raw audio data (1-dimensional)
			- mfcc: Returns MFCC features
			- spec: Returns a spectrogram
		on_error: str or None (default: None). One of:
			- 'raise' or None: let the error propagate (no special catching)
			- 'suppress': catch the error and return None
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
	assert on_error in (None, 'suppress', 'raise')

	if isinstance(audio, str):
		original_path = audio
		audio = load_audio(audio)
	else:
		original_path = None

	if len(audio['signal']) < 1:
		logger.error('Failed to produce audio features while processing file '
			'%s. Length: %d. Sample rate: %d.', original_path,
			len(audio['signal']), audio['sample_rate'])
		if on_error == 'suppress':
			return None
		else:
			raise ValueError('Audio data is too short.')

	if feature_type == 'raw':
		return audio['signal']

	elif feature_type == 'mfcc':
		try:
			import python_speech_features
		except ImportError:
			logger.exception('"python_speech_features" is a required Python '
				'dependency for calculating MFCC features.')
			raise

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

		signal = scale_signal(audio)

		hop_size = int(step_size * audio['sample_rate'])
		frame_size = int(window_size * audio['sample_rate'])

		if len(signal) < frame_size:
			logger.error('Failed to produce FFT while processing file '
				'%s. Original length: %d. Hop size: %d. Frame size: %d. '
				'Sample rate: %d.', original_path, len(signal), hop_size,
				frame_size, audio['sample_rate'])
			if on_error == 'suppress':
				return None
			else:
				raise ValueError('Audio data is too short.')

		# Cleave off any samples that do not cleanly fit into our step size.
		remove = (len(signal) - frame_size) % hop_size
		if remove:
			clean = signal[:-remove]
		else:
			clean = signal

		# Optimization: instead of doing a for loop or list comprehension to
		# apply the window to the signal, we can just create a new view into
		# the data with each window.
		num_frames = (len(clean) - frame_size) // hop_size + 1
		frames = numpy.lib.stride_tricks.as_strided(
			clean,
			shape=(frame_size, num_frames),
			strides=(clean.strides[0], clean.strides[0] * hop_size)
		)

		filter_window = numpy.hanning(frame_size)
		fft = numpy.fft.rfft(
			frames * numpy.expand_dims(filter_window, -1),
			axis=0
		)
		norm = numpy.absolute(fft)**2

		scale = numpy.sum(filter_window**2) * audio['sample_rate']
		scaled = norm
		scaled[1:-1] /= scale/2
		scaled[[0, -1]] /= scale

		spec = scaled
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
						max(int(f / delta_freq + 0.5), 0) + 1, num_bins
					)
					if f else alt
				)
			spec = spec[get_bin(low_freq, 0):get_bin(high_freq, num_bins)]

		spec = numpy.log(spec + 1e-14)
		# Format `spec` as (time, frequency)
		spec = spec.T
		return spec

	else:
		raise ValueError('Unsupported feature type: {}'.format(feature_type))

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
