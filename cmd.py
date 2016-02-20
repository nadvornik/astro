#!/usr/bin/python

import Queue

class CmdQueue:
	def __init__(self):
		self.dict = {}
	
	def register(self, tid):
		self.dict[tid] = Queue.Queue()
		print "register", tid
	
	def get(self, tid, timeout):
		q = self.dict[tid]
		try:
			cmd = q.get(block=True, timeout= timeout/1000.0)
			print tid, ':', cmd
			return cmd
		except Queue.Empty:
			return None
			
	def put(self, entry):
		for q in self.dict.values():
			q.put(entry)
			
cmdQueue = CmdQueue()
	