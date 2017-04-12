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

import threading
import logging
import ctypes
from functools import wraps, partial

logger = logging.getLogger(__name__)

###########################################################################
def locked(lock):
	""" Wrap function in a thread-safe lock. This is similar to Java's
		`synchronized(this)`.
	"""

	###########################################################################
	def wrapper(func):
		""" Closure for the actual decorator.
		"""

		#######################################################################
		@wraps(func)
		def wrapped(*args, **kwargs):
			""" The wrapped (locked) function call.
			"""
			with lock:
				func(*args, **kwargs)
		return wrapped

	return wrapper

###############################################################################
class CudaError(Exception):
	""" Exception class for CUDA errors.
	"""
	pass

###############################################################################
# The lock
_LOCK = threading.RLock()

###############################################################################
def ready(func):
	""" Decorator for a class instance method which requires that the class's
		"_ready" property be set.
	"""

	###########################################################################
	@wraps(func)
	def wrapper(self, *args, **kwargs):
		""" The wrapped function.
		"""
		if not self._ready:
			raise ValueError('CUDA has not been initialized for this object. '
				'Be sure to only use this object only within Python context '
				'management.')
		return func(self, *args, **kwargs)

	return wrapper

###############################################################################
class DeviceHandleRaw(ctypes.Structure):
	pass
DeviceHandle = ctypes.POINTER(DeviceHandleRaw)

###############################################################################
class DeviceMemory(ctypes.Structure):
	_fields_ = [
		('total', ctypes.c_ulonglong),
		('free', ctypes.c_ulonglong),
		('used', ctypes.c_ulonglong)
	]

###############################################################################
class DeviceUtilization(ctypes.Structure):
	_fields_ = [
		('gpu', ctypes.c_uint),
		('memory', ctypes.c_uint),
	]

################################################################################
class Device:
	""" Manages individual GPU devices.

		# References

		- http://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
	"""

	###########################################################################
	def __init__(self, handle, context):
		self.handle = handle
		self.context = context

	###########################################################################
	def __str__(self):
		return 'Device(index="{}", uuid="{}")' \
			.format(self.index(), self.uuid())

	###########################################################################
	def __repr__(self):
		return str(self)

	###########################################################################
	def __getattr__(self, name):
		if hasattr(self.context, name):
			attr = getattr(self.context, name)
			if callable(attr) and hasattr(attr, '_device') and attr._device:
				return partial(attr, self)
		raise AttributeError('No such attribute for a device: {}'.format(name))

###############################################################################
def mark_device(func):
	func._device = True
	return func

###############################################################################
class CudaContext:

	NVIDIA_ML = 'libnvidia-ml.so.1'
	_dll = None
	_ref = 0
	_ptr = {}

	###########################################################################
	def __init__(self):
		self._ready = False
		self._devices = None

	###########################################################################
	def __enter__(self):
		self._init()
		self._devices = {}
		self._ready = True
		return self

	###########################################################################
	def __exit__(self, exc_type, exc_val, exc_tb):
		self._ready = False
		self._del()

	###########################################################################
	@classmethod
	@locked(_LOCK)
	def _init(cls):
		if cls._dll is None:
			logger.trace('Loading NVIDIA ML library.')
			try:
				cls._dll = ctypes.CDLL(cls.NVIDIA_ML)
			except OSError:
				raise CudaError('Failed to find NVIDIA ML library.')

		if not cls._ref:
			logger.trace('Initializing NVIDIA ML.')
			init = cls._get_ptr('nvmlInit_v2')
			if init():
				raise CudaError('Failed to initialize NVIDIA ML.')

		cls._ref += 1

	###########################################################################
	@classmethod
	def _get_ptr(cls, function_name):
		result = cls._ptr.get(function_name)
		if result is None:
			try:
				result = getattr(cls._dll, function_name)
			except AttributeError:
				logger.exception('No such function found in the NVIDIA ML '
					'library: %s')
				raise
			cls._ptr[function_name] = result
		return result

	###########################################################################
	@classmethod
	@locked(_LOCK)
	def _del(cls):
		cls._ref -= 1
		if not cls._ref:
			logger.trace('Shutting down NVIDIA ML.')
			shutdown = cls._get_ptr('nvmlShutdown')
			if shutdown():
				raise CudaError('Failed to shutdown NVIDIA ML.')

	###########################################################################
	def __iter__(self):
		""" Iterates over devices.
		"""
		return (self.device(i) for i in range(len(self)))

	###########################################################################
	@ready
	def __len__(self):
		""" Retrieves the number of GPU devices.
		"""
		result = ctypes.c_uint()
		func = self._get_ptr('nvmlDeviceGetCount_v2')
		if func(ctypes.byref(result)):
			raise CudaError('Failed to shutdown NVIDIA ML.')
		return result.value

	###########################################################################
	@ready
	def _get_handle_by_index(self, index):
		c_index = ctypes.c_uint(index)
		handle = DeviceHandle()
		func = self._get_ptr('nvmlDeviceGetHandleByIndex_v2')
		if func(c_index, ctypes.byref(handle)):
			raise CudaError('Failed to get device by index: {}'.format(index))
		return handle

	###########################################################################
	@ready
	def _get_handle_by_uuid(self, uuid):
		c_uuid = ctypes.c_char_p(uuid)
		handle = DeviceHandle()
		func = self._get_ptr('nvmlDeviceGetHandleByUUID')
		if func(c_uuid, ctypes.byref(handle)):
			raise CudaError('Failed to get device by UUID: {}'.format(uuid))
		return handle

	###########################################################################
	def __getitem__(self, device):
		""" A convenience function for retrieving devices by index.
		"""
		return self.device(device)

	###########################################################################
	def device(self, device):
		""" Retrieves a Device handler.

			# Arguments

			device: int or str. If an integer, it is the zero-based index of
				the device to retrieve. If a string, it is the UUID of the
				device to retrieve.

			# Return value

			A Device instance. If no such device can be found, an exception is
			raised.
		"""
		if isinstance(device, int):
			if device < 0 or device >= len(self):
				raise ValueError('Out-of-range device index: {}'
					.format(device))
			if device not in self._devices:
				handle = Device(self._get_handle_by_index(device), self)
				self._devices[device] = handle
			else:
				handle = self._devices[device]
		elif isinstance(device, (str, bytes)):
			if isinstance(device, str):
				device = device.encode('utf-8')
			handle = Device(self._get_handle_by_uuid(device), self)
			index = handle.index()
			if index not in self._devices:
				self._devices[index] = handle
			else:
				# Don't accidentally create additional handles that you give to
				# the user.
				handle = self._devices[index]
		else:
			raise TypeError(
				'Index must be a integer index, or a str/bytes UUID.')

		return handle

	###########################################################################
	@mark_device
	@ready
	def name(self, device):
		""" Returns the name of the device as a string.
		"""
		logger.trace('Calculating device name.')
		buffer_size = 64
		name = ctypes.create_string_buffer(buffer_size)
		func = self._get_ptr('nvmlDeviceGetName')
		if func(device.handle, name, ctypes.c_uint(buffer_size)):
			raise CudaError('Failed to get device name.')
		return name.value.decode('utf-8')

	###########################################################################
	@mark_device
	@ready
	def uuid(self, device):
		""" Returns the device UUID as a string.
		"""
		logger.trace('Calculating device UUID.')
		buffer_size = 80
		uuid = ctypes.create_string_buffer(buffer_size)
		func = self._get_ptr('nvmlDeviceGetUUID')
		if func(device.handle, uuid, ctypes.c_uint(buffer_size)):
			raise CudaError('Failed to get UUID for device.')
		return uuid.value.decode('utf-8')

	###########################################################################
	@mark_device
	@ready
	def index(self, device):
		""" Returns the zero-based index of the device.
		"""
		logger.trace('Calculating device index.')
		index = ctypes.c_uint()
		func = self._get_ptr('nvmlDeviceGetIndex')
		if func(device.handle, ctypes.byref(index)):
			raise CudaError('Failed to get index for device.')
		return index.value

	###########################################################################
	@mark_device
	def kernel_utilization(self, device):
		""" Returns the kernel utilization of the device, as an integer between
			0 (not utilized) and 100 (fully utilized).
		"""
		return self._utilization(device).gpu

	###########################################################################
	@mark_device
	def memory_utilization(self, device):
		""" Returns the memory utilization of the device, as an integer between
			0 (not utilized) and 100 (fully utilized).
		"""
		return self._utilization(device).memory

	###########################################################################
	@ready
	def _utilization(self, device):
		logger.trace('Calculating device utilization.')
		result = DeviceUtilization()
		func = self._get_ptr('nvmlDeviceGetUtilizationRates')
		if func(device.handle, ctypes.byref(result)):
			raise CudaError('Failed to get device utilization.')
		return result

	###########################################################################
	@mark_device
	def used_memory(self, device):
		""" Returns the used memory on the device, in MiB.
		"""
		return self._memory(device).used / 2**20

	###########################################################################
	@mark_device
	def free_memory(self, device):
		""" Returns the free memory on the device, in MiB.
		"""
		return self._memory(device).free / 2**20

	###########################################################################
	@mark_device
	def total_memory(self, device):
		""" Returns the total memory on the device, in MiB.
		"""
		return self._memory(device).total / 2**20

	###########################################################################
	@ready
	def _memory(self, device):
		logger.trace('Calculating device memory footprint.')
		result = DeviceMemory()
		func = self._get_ptr('nvmlDeviceGetMemoryInfo')
		if func(device.handle, ctypes.byref(result)):
			raise CudaError('Failed to get memory utilization.')
		return result

	###########################################################################
	@mark_device
	def business(self, device):
		""" Calculates how busy a device is ("busy-ness").

			# Return value

			An integer between 0 (idle) and 100 (fully loaded) indicating how
			busy the device is.
		"""
		logger.trace('Calculating device business...')
		memory = self._memory(device)
		utilization = self._utilization(device)
		x = (
			100 * memory.used / memory.total,
			utilization.gpu,
			utilization.memory
		)
		return int(sum(x) / len(x))

	###########################################################################
	@mark_device
	def is_busy(self, device):
		""" Convenience function for checking if a device is in use.
		"""
		return self.business(device) >= 5

	###########################################################################
	def rank_available(self, num=None):
		""" Returns devices in the order of increasing usage.

			# Arguments

			num: int or None. If None, returns all devices. If an integer, then
				returns up to `num` devices.d$x t

			# Return value

			A list of devices, ordered by increasing usage (so that the idling
			devices would come first). If two devices tie for usage, then the
			device with the lower index is ordered first.
		"""
		logger.trace('Calculating device availability...')
		if num is not None and (num <= 0 or num > len(self)):
			raise IndexError('Invalid number of instances to return. Use None '
				'to return all available.')

		devices = sorted(
			self,
			key=lambda device: (device.business(), device.index())
		)
		if not devices:
			raise ValueError('No GPUs found.')

		if num is None:
			return devices
		else:
			return devices[:num]

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
