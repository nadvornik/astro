#!/usr/bin/python

import webserver
import cv2
import numpy as np
import io
from PIL import Image
from cmd import cmdQueue
import Queue
import threading

class MyGUI_CV2(threading.Thread):
	cmds = {
		ord('q') : 'exit',
		27 : 'exit',
		ord(' ') : 'solver-reset',
		ord('d') : 'dark',
		ord('g') : 'guider',
		ord('f') : 'navigator',
		
		ord('t') : 'test-capture',
		ord('y') : 'capture',
		
		ord('1') : 'disp-normal',
		ord('2') : 'disp-zoom-2',
		ord('3') : 'disp-zoom-3',
		ord('4') : 'disp-zoom-4',
		ord('5') : 'disp-zoom-8',
		ord('6') : 'disp-zoom-16',
		ord('7') : 'disp-zoom-deg50',
		ord('8') : 'disp-zoom-deg100',
		ord('9') : 'disp-zoom-deg180',
		ord('0') : 'disp-match',
		ord('-') : 'disp-orig',
		ord('=') : 'disp-df-cor',
		ord(' ') : ' ',
		ord('a') : 'z0',
		ord('z') : 'z1',
		ord('x') : 'f-3',
		ord('c') : 'f-2',
		ord('v') : 'f-1',
		ord('b') : 'f+1',
		ord('n') : 'f+2',
		ord('m') : 'f+3',

		ord('j') : 'left',
		ord('l') : 'right',
		ord('i') : 'up',
		ord('k') : 'down',
		
		ord('p') : 'polar-reset',
		ord('o') : 'polar-align',
		
		ord('s') : 'save',
		}

	def __init__(self):
		threading.Thread.__init__(self)
		self.queue = Queue.Queue()
		self.stop = False
		self.start()

	def namedWindow(self, name):
		self.queue.put((name, None))
	
	
	def imshow(self, name, img):
		self.queue.put((name, img))

	def imshow_jpg(self, name, jpg):
		pil_image = Image.open(io.BytesIO(jpg))
		img = np.array(pil_image)
		self.queue.put((name, img))
	
	
	def run(self):
		while not self.stop:
			key = cv2.waitKey(10)
			if key in MyGUI_CV2.cmds:
				cmdQueue.put(MyGUI_CV2.cmds[key])

			try:
				(name, img) = self.queue.get(block=False)
			except Queue.Empty:
				continue
			if img is None:
				cv2.namedWindow(name, cv2.WINDOW_NORMAL)
			else:
				cv2.imshow(name, img)
		
	def __enter__(self):
		pass

	def __exit__(self, type, value, traceback):
		self.stop = True
		pass
		

class MyGUI_Web:
	def __init__(self):
		self.server = webserver.ServerThread()
		self.server.start()
	
	def namedWindow(self, name):
		webserver.mjpeglist.add(name)
	
	def imshow(self, name, img):
		pil_image = Image.fromarray(img)
		webserver.mjpeglist.update(name, pil_image)

	def imshow_jpg(self, name, jpg):
		webserver.mjpeglist.update(name,jpg)
	
	def __enter__(self):
		pass

	def __exit__(self, type, value, traceback):
		self.server.shutdown()

#ui = MyGUI_CV2()
ui = MyGUI_Web()
