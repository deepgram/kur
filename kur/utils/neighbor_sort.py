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

import math
import bisect

import numpy

###############################################################################
def argsort(data, batch_size, neighborhood=None, growth_factor=None,
	uniform=False):
	""" Returns indices which sort the dataset into batches of neighboring
		entries.

		# Arguments

		data: list or numpy array. The data to use in producing the sort
			indices.
		batch_size: int. The size of each batch.
		neighborhood: int or float (default: None). The target size to draw
			neighboring samples from. If None, it defaults to 10% of the range
			of `data`.
		growth_factor: float (default: None). If there are not enough data
			points in the neighborhood to construct an entire batch, then the
			neighborhood must expand to include other, more distant points.
			This parameter controls how much the current neighborhood should
			exand when necessary. It is a fraction of the current neighborhood
			size. If this is None, it defaults to 0.4.
		uniform: bool (default: False). If True, candidate neighborhoods are
			selected uniformly at random from all data points not previously
			associated with a neighborhood. If False, neighborhoods are
			systematically "grown" from one end of the data towards the other.

		# Return value

		A numpy array of int32's that contains the indices that would sort the
		array into neighborhoods.
	"""

	num_entries = len(data)
	num_batches = math.ceil(num_entries / batch_size)
	num_perfect_batches = int(num_entries / batch_size)

	# Sort the data.
	sort_permutation = numpy.argsort(data)
	data = data[sort_permutation]

	# Calculate the neighborhood size if it wasn't given to us.
	if not neighborhood:
		data_range = data[-1] - data[0]
		neighborhood = data_range / 10

	if not uniform:
		neighborhood *= 2

	if not growth_factor:
		growth_factor = 0.4

	# This mask tracks which entries we've already assigned to neighborhoods.
	mask = numpy.ones(num_entries, dtype=bool)

	# We will put our results here.
	result = numpy.empty(num_entries, dtype=numpy.int32)

	if not uniform:
		batch_ordering = numpy.random.permutation(num_perfect_batches)

	# Create each batch
	for num_batch in range(num_batches):

		offset = batch_size * num_batch
		needed = min(num_entries - offset, batch_size)

		# Choose our target "value" for the neighborhood.
		if uniform:
			seed_index = numpy.random.choice(num_entries - offset)
		else:
			if numpy.random.random_sample() >= 0.5:
				seed_index = num_entries - offset - 1
			else:
				seed_index = 0

		seed_value = data[mask][seed_index]

		# Find a neighborhood that has enough data.
		current_neighborhood = neighborhood / 2
		while True:
			lower_bound = bisect.bisect_left(
				data[mask],
				seed_value - current_neighborhood
			)
			upper_bound = bisect.bisect_right(
				data[mask],
				seed_value + current_neighborhood
			)

			candidates = upper_bound - lower_bound
			if candidates < needed:
				current_neighborhood *= (1 + growth_factor)
			else:
				break

		# Select which points to assign to this neighborhood.
		selected = numpy.random.choice(
			candidates, size=needed, replace=False
		) + lower_bound

		# Keep the seed.
		if not uniform:
			if seed_index not in selected:
				selected[0] = seed_index

		# Update the results.
		if not uniform and num_batch < num_perfect_batches:
			# For non-uniform sampling, we still want the end results in a
			# pseudo-random order.
			offset = batch_ordering[num_batch]*batch_size

		result[offset:offset+needed] = sort_permutation[mask][selected]

		# Update the mask.
		mask[numpy.nonzero(mask)[0][selected]] = False

	return result

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
