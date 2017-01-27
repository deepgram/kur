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
import gzip
import tarfile
import logging
import tempfile

from .network import get_hash, download

logger = logging.getLogger(__name__)

###############################################################################
def canonicalize(path):
	""" Returns an absolute, canonical path.
	"""
	return os.path.realpath(os.path.abspath(path))

###############################################################################
def sandbox_extract(path, dest, allow_outside_links=True,
	allow_absolute=False):
	""" Safely extract a tar archive, avoiding paths with try to "escape" from
		the destination path (the "sandbox"), since tar archives can accept
		files with absolute paths (e.g, "/") or relative paths ("../..", "..").

		# Arguments

		path: str. The path to the tarball to extract.
		dest: str. The destination folder to extract into.
		allow_outside_links: bool (default: True). If True, links inside the
			archive are allowed to point to files outside of the archive. If
			False, all links must resolve to files/directories in the archive.
		allow_absolute: bool (default: False). If True, files/links are allowed
			to have absolute paths; if False, all paths in the archive must be
			relative. In either case, any absolute paths must ultimately "end
			up" back inside the sandbox.

		# Return value

		A list of extracted filenames.
	"""

	def path_is_safe(path):
		""" We will paths that leave the current directory tree.
		"""
		if allow_absolute:
			# We allow absolute filenames, as long as they land inside our
			# destination directory.
			path = canonicalize(path)
			normalized_dest = canonicalize(dest)
			if not path.startswith(normalized_dest):
				return False
			# We could still have this problem:
			#   path = '/asd/fg__ha_ha_i_escaped/try_and_stop_me'
			#   normalized_dest = '/asd/fg'
			# So we need to demand that this falls on a directory boundary.
			suffix = path[len(normalized_dest):]
			return suffix.startswith('/')
		else:
			# We will simply forbid absolute paths altogether, as well as
			# relative paths that escape from our destination directory.
			path = os.path.normpath(path)
			if path.startswith('/') or path == '..' or path.startswith('../'):
				return False
			return True

	def link_is_safe(member):
		""" For links, they must resolve inside the current directory tree.
			Note: you still need to check that `path_is_safe(member.name)`.
		"""
		if not member.issym() and not member.islnk():
			return True
		return path_is_safe(
			os.path.join(os.path.dirname(member.name), member.linkname)
		)

	with tarfile.open(path) as tar:
		# Get a list of members that can be safely extracted.
		members = [
			member for member in tar.getmembers()
			if path_is_safe(member.name) and
				(allow_outside_links or link_is_safe(member))
		]
		# Extract the safe members
		tar.extractall(path=dest, members=members)
		# Return the list of file paths.
		return [canonicalize(os.path.join(dest, member.name))
			for member in members]

###############################################################################
def is_gzipped(filename):
	""" Returns True if the target filename looks like a GZIP'd file.
	"""
	with open(filename, 'rb') as fh:
		return fh.read(2) == b'\x1f\x8b'

###############################################################################
def gzip_extract(path, dest):
	""" Extracts a GZIP'd file into a target directory.
	"""
	with gzip.open(path, 'rb') as fh:
		content = fh.read()

	target = os.path.join(dest, os.path.basename(path))
	if path.endswith('.gz'):
		target = target[:-3]
	else:
		parts = os.path.splitext(target)
		if parts[1]:
			target = parts[0]
		else:
			target = target + '.content'

	with open(target, 'wb') as fh:
		fh.write(content)

	return canonicalize(target)

###############################################################################
def unpack(path, dest=None, recursive=False, ignore_error=False):
	""" Extracts an archive.

		# Arguments

		path: str. Filename to archive to extract.
		recursive: bool (default: False). If False, only the file at `path` is
			extracted. If True, then if any extracted files are themselves
			compressed, they will be recursively extracted as well.
		ignore_error: bool (default: False). If True, errors that result from
			trying to extract uncompressed files will be silently ignored.
		dest: str or None (default: None). Directory to extract to (will be
			created if it doesn't exist). If None, defaults to the same
			directory as `path`.
	"""
	if dest is None:
		dest = os.path.dirname(path)
	if not os.path.isdir(dest):
		if os.path.exists(dest):
			raise ValueError('Cannot extract to target location: {}'.format(
				dest))
		os.makedirs(dest, exist_ok=True)

	if os.path.isfile(path) and tarfile.is_tarfile(path):
		extracted = sandbox_extract(path, dest)
	elif os.path.isfile(path) and is_gzipped(path):
		extracted = gzip_extract(path, dest)
	else:
		if ignore_error:
			extracted = []
		else:
			raise ValueError('Unknown or unsupported file type: {}'
				.format(path))

	if recursive:
		for filename in list(extracted):
			extracted.extend(unpack(
				filename,
				os.path.dirname(filename),
				recursive=True,
				ignore_error=True
			))

	return extracted

###############################################################################
def install(url=None, path=None, checksum=None):
	""" Ensure that the data source exists locally.

		# Return value

		A tuple `(path, is_packed)`, where `path` is the path to the target on
		the local machine, and `is_packed` is a boolean which indicates
		whether or not `path` appears to be a single file (packed) or a
		directory (unpacked, as if it were a tar archive that was already
		extracted).
	"""
	if url is None:
		# Expect a path to an existing source
		if path is None:
			raise ValueError('Either "url" or "path" needs to be '
				'specified in the data supplier.')
		path = os.path.expanduser(os.path.expandvars(path))
		if os.path.isfile(path):
			# Perfect. Checksum it.
			if checksum is not None:
				actual = get_hash(path)
				if actual.lower() != checksum.lower():
					raise ValueError('Input file "{}" failed its '
						'checksum.'.format(path))
			return path, True
		elif os.path.isdir(path):
			return path, False
		else:
			raise ValueError('"path" was specified in a data supplier, but '
				'the path does not exist. Check that the path is correct, or '
				'specify a URL to download data.')
	else:
		if path is None:
			# URL, but no path: use temporary directory as path.
			path = tempfile.gettempdir()
		else:
			path = os.path.expanduser(os.path.expandvars(path))

		if not os.path.exists(path):
			# Create the necessary directories and download the file.
			os.makedirs(path, exist_ok=True)

		if os.path.isdir(path):
			# It's a directory that exists. Let's look for the would-be
			# downloaded file.
			_, filename = os.path.split(url)
			path = os.path.join(path, filename)

		if os.path.isfile(path):
			# File already exists. Checksum it.
			if checksum is not None:
				if get_hash(path).lower() == checksum.lower():
					logger.debug('File exists and passed checksum: %s',
						path)
					return path, True
				else:
					# Checksum fails -> redownload
					logger.warning('Input file "%s" failed its checksum. '
						'Redownloading...', path)
			else:
				logger.debug('File exists, but there is no checksum: %s',
					path)
				return path, True

		# Need to download the file.
		download(url, path)
		if checksum is not None:
			if get_hash(path).lower() != checksum.lower():
				raise ValueError('Failed to download URL: {}. The '
					'integrity check failed.'.format(url))
			else:
				logger.debug('Downloaded file passed checksum: %s', path)
		else:
			logger.debug('Downloaded file, but there is not checksum: %s',
				path)
		return path, True

	raise ValueError('Unhandled download path. This is a bug.')

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
