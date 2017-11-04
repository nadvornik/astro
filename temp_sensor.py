#!/usr/bin/env python

# Copyright (C) 2015 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import numpy as np
import threading
import subprocess
import atexit
import time
import sys
import logging

log = logging.getLogger()

class TempSensor(threading.Thread):
	def __init__(self, status):
		threading.Thread.__init__(self)
		self.daemon = True
		self.history = []
		self.terminating = False
		self.bin_name = "./temp_sensor"
		self.cmd = None
		self.status = status
		self.status['temp'] = 0.0
		self.status['rhum'] = 0.0
		
		self.start()
	
	def run(self):
		while True:
			if self.cmd is None or self.cmd.poll() is not None:
				restart = self.cmd is not None
				if restart:
					log.error("temp_sensor exited with %d\n" % (self.cmd.poll()))
				if self.terminating:
					return
				self.cmd = subprocess.Popen([self.bin_name], close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 )
				if not restart:
					atexit.register(self.terminate)
			try:
				line = self.cmd.stdout.readline()
				(temp, hum) = [float(s) for s in line.split()]
				self.history.append((time.time(), temp, hum))
				(self.status['temp'], self.status['rhum']) = (temp, hum)
				log.info("temp %f, hum %f" % (temp, hum))
			except:
				pass
	
	def terminate(self):
		self.terminating = True
		if self.cmd is not None and self.cmd.poll() is None:
			self.cmd.terminate()
		self.join()


	def get(self):
		return (self.status['temp'], self.status['rhum'])

if __name__ == "__main__":
	t = TempSensor({})
	while True:
		print(t.get())
		time.sleep(10)
		
#usb.close()
