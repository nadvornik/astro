#!/usr/bin/python3

import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cgi
import threading
import io
import time
import os
import sys
from cmd import cmdQueue
import subprocess
from stacktraces import stacktraces
import socket
#import exceptions
import logging

from websocket import HTTPWebSocketsMixIn

log = logging.getLogger()

class MjpegBuf:
	def __init__(self):
		self.condition = threading.Condition()
		self.buf = None
		self.encoded = True
		self.seq = -1

	def update(self, image):
		with self.condition:
			self.buf = image
			self.encoded = False
			self.condition.notify_all()
			self.seq += 1

	def update_jpg(self, jpg):
		with self.condition:
			self.buf = jpg
			self.encoded = True
			self.condition.notify_all()
			self.seq += 1

	def serve(self, handler, seq):
		t0 = time.time()
		with self.condition:
			while self.buf is None or seq == self.seq + 1:
				if self.buf is None or time.time() > t0 + 25:
					handler.send_response(503)
					handler.send_header('Connection', 'close')
					handler.end_headers()
					log.info("req timeout")
					return
#					raise EOFError()
				self.condition.wait(10)
			if not self.encoded:
				encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
				ret, file_data = cv2.imencode('.jpg', self.buf, encode_param)
				self.buf = file_data.tobytes()
				del file_data
				self.encoded = True
				
			buf = self.buf
			seq = self.seq
				
		l = len(buf)
		handler.send_response(200)
		handler.send_header('Content-type','image/jpeg')
		handler.send_header('Content-length',str(l))
		handler.send_header('X-seq',str(seq))
		handler.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
		handler.send_header('Pragma', 'no-cache')
		handler.send_header('Expires', '0')
		handler.end_headers()
		handler.wfile.write(buf)
		handler.wfile.flush()


class MjpegList:
	def __init__(self):
		self.dict = {}
	
	def add(self, name):
		self.dict[name] = MjpegBuf()
	
	def get(self, name):
		if name in self.dict:
			return self.dict[name]
		else:
			return None
	
	def update(self, name, image):
		self.dict[name].update(image)

	def update_jpg(self, name, jpg):
		self.dict[name].update_jpg(jpg)



class Handler(BaseHTTPRequestHandler, HTTPWebSocketsMixIn):

	wbufsize = 1024 * 1024
	timeout = 120
	protocol_version = 'HTTP/1.1'

	def do_GET(self):
		s_path = self.path.split('?')
		path = self.path
		args = ''
		if len(s_path) == 2:
			path, args = s_path
		
		base = os.path.basename(path)
		name, ext = os.path.splitext(base)
		log.error(path)
		if self.path == '/':
			self.send_response(301)
			self.send_header('Location', 'index.html')
			self.send_header('Connection', 'close')
			self.end_headers()
			return
		elif ext == '.jpg':
			s_args = args.split('=')
			if len(s_args) == 2 and s_args[0] == 'seq':
				seq = int(s_args[1])
			else:
				seq = 0
			
			mjpeg = mjpeglist.get(name)
			if mjpeg is None:
				self.send_response(404)
				self.send_header('Connection', 'close')
				self.end_headers()
				return
			
			mjpeg.serve(self, seq)
			return
		elif base == 'log.html':
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.send_header('Connection', 'close')
			self.end_headers()
			self.wfile.write("<html><head><title>Log</title></head>")
			self.wfile.write("<body><pre>")
			self.wfile.write(subprocess.check_output(['journalctl', '-u', 'navigate.service', '-n', '300']))
			self.wfile.write("</bre></body></html>")
			return
		elif base == 'status.json':
			global status
			if status is not None:
				self.send_response(200)
				self.send_header('Content-type','application/json')
				c = status.to_json().encode()
				self.send_header('Content-length', str(len(c)))
				self.end_headers()
				self.wfile.write(c)
			else:
				self.send_response(404)
				self.send_header('Connection', 'close')
				self.end_headers()
				return
			return
		elif ext == '.html' or ext == '.js' or  ext == '.css' or ext == '.map':
			try:
				log.error("ui/build/" + path)
				f = open("ui/build/" + path, "rb")
			except:
				self.send_response(404)
				self.send_header('Connection', 'close')
				self.end_headers()
				return
			c = f.read()
			f.close()
			self.send_response(200)
			self.send_header('Content-length', str(len(c)))
			if ext == '.html':
				self.send_header('Content-type','text/html; charset=utf-8')
			elif ext == '.js':
				self.send_header('Content-type','application/javascript')
			elif ext == '.css':
				self.send_header('Content-type','text/css')
			elif ext == '.map':
				self.send_header('Content-type','application/json')
			self.end_headers()
			self.wfile.write(c)
			return
		elif path == '/websocket':
			if self.headers.get("Upgrade", None) == "websocket":
				try:
					self.indi_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					self.indi_socket.connect(("localhost", 7624))
				except:
					log.exception("Websocket INDI connect failed")
					self.send_response(500)
					self.end_headers()
					return

				self.log_message("handshake")
				self.handshake()
				self.log_message("websocket started1")
				threading.Thread(target=self.websocket_writer).start()
				self.log_message("websocket started2")
				self.read_messages()
				return

		self.send_response(404)
		self.send_header('Connection', 'close')
		self.end_headers()

	def websocket_writer(self):
		try:
			while True:
				msg = self.indi_socket.recv(1000000)
				if len(msg) == 0:
					log.error("ws indiserver received 0")
					break
				self.send_message(msg)
		except:
			log.exception("websocket_writer closed")

	def on_ws_message(self, message):
		self.indi_socket.send(message)

	def do_POST(self):
		ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
		if ctype == 'multipart/form-data':
			postvars = cgi.parse_multipart(self.rfile, pdict)
		elif ctype == 'application/x-www-form-urlencoded':
			length = int(self.headers.get('content-length'))
			postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
		else:
			postvars = {}
	
		base = os.path.basename(self.path)
		if base == 'button':
			self.send_response(200)
			self.send_header('Content-type','text/text')
			self.send_header('Connection', 'close')
			self.end_headers()
			self.wfile.write(b"ok")
			
			cmd = postvars[b'cmd'][0].decode()
			try:
				tgt = postvars[b'tgt'][0].decode()
			except:
				tgt = None
			if cmd == 'exit' or cmd == 'shutdown' or cmd == 'stacktrace':
				stacktraces()
			cmdQueue.put(cmd, target = tgt)
			return
		self.send_response(404)
		self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

	def handle_error(self, request, client_address):
		ex, val = sys.exc_info()[:2]
		#if 'Broken pipe' not in val:
		log.exception('Unexpected error')
		#pass

class ServerThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.server = ThreadedHTTPServer(('', 8080), Handler)
		self.server.daemon_threads = True
		self.daemon = True

	def run(self):
		self.server.serve_forever()

	def shutdown(self):
		time.sleep(3)
		self.server.shutdown()
		
mjpeglist = MjpegList()
status = None




if __name__ == '__main__':
	server = ServerThread()
	server.start()
	
	mjpeglist.add('capture')

	i = 0
	while True:
#		pil_image = Image.open("testimg2_" + str(i % 5 * 4 + 3) + ".tif")
#		mjpeglist.update('capture', pil_image)
		time.sleep(10)

		i += 1
