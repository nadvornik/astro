import time
import sys
import threading
import indi_python.indi_base as indi

import logging

log = logging.getLogger()

class FocuserIndi:
	def __init__(self, driver, device):
		self.driver = driver
		self.device = device
		self.pos0 = None
		self.lock = threading.Lock()
	
	
	def cmd(self, cmd):
		with self.lock:
			try:
				if self.driver.checkValue(self.device, "CONNECTION", "CONNECT") == 'Off':
					self.driver.sendClientMessageWait(self.device, "CONNECTION", {"CONNECT": "On"})
					time.sleep(1);
				
				while self.driver[self.device]["ABS_FOCUS_POSITION"].getAttr('state') == 'Busy':
					time.sleep(0.2)

				pos = int(self.driver[self.device]["ABS_FOCUS_POSITION"]["FOCUS_ABSOLUTE_POSITION"].getValue())
				
				if cmd == "f-3":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos - 384})
				if cmd == "f-2":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos - 128})
				if cmd == "f-1":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos - 16})
				if cmd == "f+3":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos + 384})
				if cmd == "f+2":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos + 128})
				if cmd == "f+1":
					self.driver.sendClientMessage(self.device, "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": pos + 16})
			except:
				log.exception("focuser move")

	def get_pos(self):
		with self.lock:
			try:
				if self.driver.checkValue(self.device, "CONNECTION", "CONNECT") == 'Off':
					self.driver.sendClientMessageWait(self.device, "CONNECTION", {"CONNECT": "On"})
					time.sleep(1);

				while self.driver[self.device]["ABS_FOCUS_POSITION"].getAttr('state') == 'Busy':
					time.sleep(0.2)

				pos = int(self.driver[self.device]["ABS_FOCUS_POSITION"]["FOCUS_ABSOLUTE_POSITION"].getValue())
				if self.pos0 is None:
					self.pos0 = pos
				return pos - self.pos0
			except:
				log.exception("focuser pos")
				return 0
