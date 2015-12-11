#!/usr/bin/env python

# Copyright (C) 2015 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import time
import threading
import numpy as np
from guide_out import GuideOutBase

from pyA20.gpio import gpio
from pyA20.gpio import port
from pyA20.gpio import connector


class GuideOut(GuideOutBase):

	def __init__(self):
		GuideOutBase.__init__(self)
		self.st4pins = [ port.PD14, port.PC4, port.PC7, port.PA7 ]

		for p in self.st4pins:
			gpio.setcfg(p, gpio.OUTPUT)  #Configure LED1 as output

		

	def set_pin(self, d):
		GuideOutBase.set_pin(self, d)

		if d > 0:
			gpio.output(self.st4pins[3], 1)
		else:
			gpio.output(self.st4pins[3], 0)

		if d < 0:
			gpio.output(self.st4pins[0], 1)
		else:
			gpio.output(self.st4pins[0], 0)



