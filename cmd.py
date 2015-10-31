#!/usr/bin/python

import Queue

class CmdQueue:
	def __init__(self):
		self.dict = {}
	
	def register(self, id):
		self.dict[id] = Queue.Queue()
		print "register", id
	
	def get(self, id, timeout):
		q = self.dict[id]
		try:
			cmd = q.get(block=True, timeout= timeout/1000.0)
			print id, ':', cmd
			return cmd
		except Queue.Empty:
			return None
			
	def put(self, entry):
		for q in self.dict.values():
			q.put(entry)
			
cmdQueue = CmdQueue()
	