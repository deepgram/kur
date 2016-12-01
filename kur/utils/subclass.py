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

###############################################################################
def get_subclasses(cls, recursive=True):
	""" Enumerates all subclasses of a given class.

		# Arguments

		cls: class. The class to enumerate subclasses for.
		recursive: bool (default: True). If True, recursively finds all
			sub-classes.

		# Return value

		A list of subclasses of `cls`.
	"""
	sub = cls.__subclasses__()
	if recursive:
		for cls in sub:
			sub.extend(get_subclasses(cls, recursive=True))
	return sub

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
