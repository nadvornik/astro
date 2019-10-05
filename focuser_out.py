import time
import sys
import subprocess
import atexit
import logging
import threading

log = logging.getLogger()


class FocuserOut:
	def __init__(self):
		self.pos = 0
		self.rt_cmd = None
		self.testmode = True
		self.lock = threading.Lock()
		try:
			self.rt_cmd = subprocess.Popen("./focuser_out", close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 )
			atexit.register(self.rt_cmd.terminate)
			self.testmode = False
		except:
			log.exception('Cant start external focuser')
			log.info("Focuser test mode")

	
	def move(self, m):
		with self.lock:
			self.pos += m
			if self.rt_cmd is not None:
				try:
					self.rt_cmd.stdin.write(("%d\n" % (m)).encode())
					self.rt_cmd.stdin.flush()
					line = self.rt_cmd.stdout.readline().decode()
					self.pos = int(line)
				except:
					log.exception('Unexpected error')			
			log.info("focuser %d", self.pos)

			
			
	
	def cmd(self, cmd):
		if cmd == "f-3":
			self.move(-384)
		if cmd == "f-2":
			self.move(-128)
		if cmd == "f-1":
			self.move(-16)
		if cmd == "f+3":
			self.move(384)
		if cmd == "f+2":
			self.move(128)
		if cmd == "f+1":
			self.move(16)

	def get_pos(self):
		return self.pos

if __name__ == "__main__":
	focuser = FocuserOut()
	
	while True:
		focuser.move(3000)
		focuser.move(-3000)
		