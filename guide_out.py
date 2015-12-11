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

class TimerThread(threading.Thread):
	def __init__(self, t, func, args=()):
		threading.Thread.__init__(self)
		self.t = t
		self.func = func
		self.args = args
		self.disabled = threading.Event()

	def stop(self):
		self.disabled.set()

	def run(self):
		if not self.disabled.wait(self.t):
			self.func(*self.args)


class GuideOutBase:
	def __init__(self):
		self.tt = None
		self.history = []
	
	def set_pin(self, d):
		print "set pin ", d
		self.history.insert(0, (d, time.time()))
	
	def out(self, d, t = 0):
		if self.tt:
			self.tt.stop()
			self.tt =None
		self.set_pin(d)
		if t > 0.0:
			self.tt = TimerThread(t, self.set_pin, (0,))
			self.tt.start()
	
	def recent_avg(self, t = None):
		t1 = time.time()
		if (t == None):
			t0 = 0
		else:
			t0 = t1 - t
		avg = 0
		for (d, ti) in self.history:
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
	g = GuideOutBase()
	while True:
		g.out(0, 0)
		time.sleep(5)
		print g.recent_avg(5)
		g.out(1, 0)
		print g.recent_avg(5)
		time.sleep(5)
		print g.recent_avg(5)
		g.out(-1, 1)
		time.sleep(5)
		print g.recent_avg(5)

#usb.close()
