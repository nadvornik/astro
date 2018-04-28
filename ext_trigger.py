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
from cmd import cmdQueue

log = logging.getLogger()

class ExtTrigger(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.daemon = True
		self.terminating = False
		self.bin_name = "./ext_trigger"
		self.cmd = None
		
		self.start()
	
	def run(self):
		while True:
			if self.cmd is None or self.cmd.poll() is not None:
				restart = self.cmd is not None
				if restart:
					log.error("ext_trigger exited with %d\n" % (self.cmd.poll()))
				if self.terminating:
					return
				self.cmd = subprocess.Popen([self.bin_name], close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 )
				if not restart:
					atexit.register(self.terminate)
			try:
				line = self.cmd.stdout.readline()
				cmdQueue.put(line.strip())
				log.info("ext trigger %s" % (line))
			except:
				pass
	
	def terminate(self):
		self.terminating = True
		if self.cmd is not None and self.cmd.poll() is None:
			self.cmd.terminate()
		self.join()



if __name__ == "__main__":
	t = ExtTrigger()
	while True:
		time.sleep(10)
		
#usb.close()
