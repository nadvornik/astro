import time


class FocuserOut:
	def __init__(self):
		self.pos = 0
		self.testmode = False
		try:
			from pyA20.gpio import gpio
			from pyA20.gpio import port
			from pyA20.gpio import connector
			global pins

			gpio.init() #Initialize module. Always called first
			pins = [ port.PA8, port.PA9, port.PA10, port.PA20 ]

		
			for p in pins:
				gpio.setcfg(p, gpio.OUTPUT)  #Configure LED1 as output
				gpio.output(p, 1)
				time.sleep(0.01)
				gpio.output(p, 0)
		
			gpio.output(pins[0], 1)
		except:
			print "Focuser test mode"
			self.testmode = True


	
	def move(self, m):
		step = 1
		if m < 0:
			step = -1

		for i in range(0, m, step):
			self.pos += step
			
			if not self.testmode:
				for p in pins:
					gpio.output(p, 0)
				gpio.output(pins[ self.pos % 4 ], 1)
				time.sleep(0.002)
		print "focuser %d" % self.pos

			
			
	
	def cmd(self, cmd):
		if cmd == "f-3":
			self.move(-3000)
		if cmd == "f-2":
			self.move(-600)
		if cmd == "f-1":
			self.move(-50)
		if cmd == "f+3":
			self.move(3000)
		if cmd == "f+2":
			self.move(600)
		if cmd == "f+1":
			self.move(50)


if __name__ == "__main__":
	focuser = FocuserOut()
	
	while True:
		focuser.move(5000)
		focuser.move(-5000)
		