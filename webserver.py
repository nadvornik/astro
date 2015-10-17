#!/usr/bin/python

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import cgi
import threading
from PIL import Image
import StringIO
import time
import os
import Queue

class MjpegBuf:
	def __init__(self):
		self.condition = threading.Condition()
		self.buf = ''

	def update(self, pil_image):
		tmpFile = StringIO.StringIO()
		pil_image.save(tmpFile,'JPEG')
		with self.condition:
			self.buf = tmpFile.getvalue()
			#lock.release()
			self.condition.notify_all()

	def serve(self, handler):
		with self.condition:
			while True:
				handler.wfile.write("--jpegBoundary\r\n")
				self.condition.wait()
				l = len(self.buf)
				handler.send_header('Content-type','image/jpeg')
				handler.send_header('Content-length',str(l))
				handler.end_headers()
				handler.wfile.write(self.buf)

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
	
	def update(self, name, pil_image):
		self.dict[name].update(pil_image)



class Handler(BaseHTTPRequestHandler):
    
	def do_GET(self):
		base = os.path.basename(self.path)
		name, ext = os.path.splitext(base)
		print name, ext
		if ext == '.mjpg':
			print name
			mjpeg = mjpeglist.get(name)
			if mjpeg is None:
				self.send_response(404)
				self.end_headers()
				return
			
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpegBoundary')
			self.end_headers()
			mjpeg.serve(self)
			return
		elif base == 'index.html':
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			f = open(base)
			self.wfile.write(f.read())
			f.close()
			return
		elif base == 'jquery.min.js':
			self.send_response(200)
			self.send_header('Content-type','application/javascript')
			self.end_headers()
			f = open(base)
			self.wfile.write(f.read())
			f.close()
			return
		self.send_response(404)
		self.end_headers()

	def do_POST(self):
		ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
		if ctype == 'multipart/form-data':
			postvars = cgi.parse_multipart(self.rfile, pdict)
		elif ctype == 'application/x-www-form-urlencoded':
			length = int(self.headers.getheader('content-length'))
			postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
		else:
			postvars = {}
	
		base = os.path.basename(self.path)
		if base == 'button':
			self.send_response(200)
			self.send_header('Content-type','text/text')
			self.end_headers()
			self.wfile.write("ok")
			cmdqueue.put(postvars['key'][0])
			return
		self.send_response(404)
		self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class ServerThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.server = ThreadedHTTPServer(('', 8080), Handler)
		self.server.daemon_threads = True

	def run(self):
		print 'Starting server, use <Ctrl-C> to stop'
		self.server.serve_forever()

	def shutdown(self):
		self.server.shutdown()
		
mjpeglist = MjpegList()
cmdqueue = Queue.Queue()

if __name__ == '__main__':
	server = ServerThread()
	server.start()
	
	mjpeglist.add('cam')

	i = 0
	while True:
		pil_image = Image.open("testimg12_" + str(i % 5 * 4 + 3) + ".tif")
		mjpeglist.update('cam', pil_image)
		time.sleep(1)

		i += 1
