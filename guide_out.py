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

class GuideOut(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.daemon = True
		self.history = []
		self.cmd = subprocess.Popen(['./guide_out_rt'], close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 )
		atexit.register(self.cmd.terminate)
		self.start()
	
	def run(self):
		while self.cmd.poll() is None:
			try:
				line = self.cmd.stdout.readline()
				(d, tsec, tusec) = [int(s) for s in line.split()]
				self.history.append((d, tsec + tusec / 1000000.0))
			except:
				pass
		print "guide_out exited"
	
	def out(self, d, t = 0):
		self.cmd.stdin.write("%d %d\n" % (d, int(t * 1000000)))
	
	def recent_avg(self, t = None):
		t1 = time.time()
		if (t == None):
			t0 = 0
		else:
			t0 = t1 - t
		avg = 0
		for (d, ti) in reversed(self.history):
			if (ti > t0):
				avg = avg + d * (t1 - ti)
				t1 = ti
			else:
				avg = avg + d * (t1 - t0)
				break
		return avg
	
	def save(self, fn):
		np.save(fn, np.array(self.history))

if __name__ == "__main__":
	g = GuideOut()
	while True:
		g.out(0, 0)
		time.sleep(5)
		print "0x5s", g.recent_avg(5)
		g.out(1, 0)
		print "0x5s1x0", g.recent_avg(5)
		time.sleep(5)
		print "1x5s", g.recent_avg(5)
		g.out(-1, 1)
		time.sleep(5)
		print "-1x1s,0x4s", g.recent_avg(5)
		
#usb.close()
