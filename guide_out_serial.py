#!/usr/bin/env python

# Copyright (C) 2015 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import fcntl
import struct
import sys
import termios
import time
import threading
import numpy as np
from guide_out import GuideOutBase

class GuideOut(GuideOutBase):

	def __init__(self):
		GuideOutBase.__init__(self)
		device = '/dev/ttyUSB0'
		self.fileobj = open(device, 'rw')

	def set_pin(self, d):
		GuideOutBase.set_pin(self, d)
		# Get current flags
		p = struct.pack('I', 0)
		flags = fcntl.ioctl(self.fileobj.fileno(), termios.TIOCMGET, p)

		# Convert four byte string to integer
		flags = struct.unpack('I', flags)[0]

		if d > 0:
			flags |= termios.TIOCM_RTS
		else:
			flags &= ~termios.TIOCM_RTS

		if d < 0:
			flags |= termios.TIOCM_DTR
		else:
			flags &= ~termios.TIOCM_DTR

		# Set new flags
		p = struct.pack('I', flags)
		fcntl.ioctl(self.fileobj.fileno(), termios.TIOCMSET, p)


