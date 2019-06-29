#!/usr/bin/python

import queue
import logging

log = logging.getLogger()


class CmdQueue:
	def __init__(self):
		self.dict = {}
	
	def register(self, tid):
		self.dict[tid] = queue.Queue()
		log.info("register %s", tid)
	
	def get(self, tid, timeout = None):
		q = self.dict[tid]
		try:
			cmd = q.get(block=True, timeout= timeout)
			log.info("%s:%s", tid, cmd)
			return cmd
		except queue.Empty:
			return None
			
	def put(self, entry, target = None):
		if target is None or target == '':
			for q in self.dict.values():
				q.put(entry)
		else:
			if target in self.dict:
				self.dict[target].put(entry)

	def send_exit(self, signum, frame):
		log.info("signal %s", signum)
		self.put('exit')
cmdQueue = CmdQueue()

