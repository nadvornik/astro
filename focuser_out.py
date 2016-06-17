class FocuserOut:
	def __init__(self):
		self.pos = 0
	
	def move(self, m):
		step = 1
		if m < 0:
			step = -1

		for i in range(0, m, step):
			self.pos += step
			print self.pos
	
	def cmd(self, cmd):
		if cmd == "f-3":
			self.move(-50)
		if cmd == "f-2":
			self.move(-10)
		if cmd == "f-1":
			self.move(-1)
		if cmd == "f+3":
			self.move(50)
		if cmd == "f+2":
			self.move(10)
		if cmd == "f+1":
			self.move(1)
