#!/usr/bin/python

import Queue
import logging

log = logging.getLogger()


class CmdQueue:
	def __init__(self):
		self.dict = {}
	
	def register(self, tid):
		self.dict[tid] = Queue.Queue()
		log.info("register %s", tid)
	
	def get(self, tid, timeout = None):
		q = self.dict[tid]
		try:
			cmd = q.get(block=True, timeout= timeout)
			log.info("%s:%s", tid, cmd)
			return cmd
		except Queue.Empty:
			return None
			
	def put(self, entry, target = None):
		if target is None or target == '':
			for q in self.dict.values():
				q.put(entry)
		else:
			if target in self.dict:
				self.dict[target].put(entry)
			
cmdQueue = CmdQueue()

