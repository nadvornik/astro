import cv2
import numpy as np

import os
import sys
import subprocess
import signal

import pyfits
from astrometry.util.util import Tan, anwcs_new_tan
from astrometry.blind.plotstuff import *
import threading
import math
import tempfile
import shutil
import time

class Solver(threading.Thread):
	def __init__(self, sources_img = None, sources_list = None, field_w = None, field_h = None, ra = None, dec = None, field_deg = None, radius = None):
		threading.Thread.__init__(self)
		self.sources_img = sources_img
		self.sources_list = sources_list
		self.field_w = field_w
		self.field_h = field_h
		if self.sources_img is not None:
			(self.field_h, self.field_w) = sources_img.shape
		self.ra = ra
		self.dec = dec
		self.field_deg = field_deg
		self.radius = radius
		if self.ra is not None and field_deg is not None and radius is None:
			self.radius = field_deg * 2.0
		self.solved = False
		self.cmd = []
		
	
	def run(self):
		tmp_dir = tempfile.mkdtemp()
		if (self.sources_img is None):
			tbhdu = pyfits.BinTableHDU.from_columns([
				pyfits.Column(name='X', format='E', array=self.sources_list[:, 1]),
				pyfits.Column(name='Y', format='E', array=self.sources_list[:, 0]),
				pyfits.Column(name='FLUX', format='E', array=self.sources_list[:, 2])
				])
			prihdr = pyfits.Header()
			prihdr['IMAGEW'] = self.field_w
			prihdr['IMAGEH'] = self.field_h
			prihdu = pyfits.PrimaryHDU(header=prihdr)
			thdulist = pyfits.HDUList([prihdu, tbhdu])
			thdulist.writeto(tmp_dir + '/field.fits', clobber=True)
		else:
			cv2.imwrite(tmp_dir + "/field.tif", self.sources_img)
		
		cmd_s = ['solve-field', '-O',  '--objs', '20', '--depth', '20', '-E', '2', '--no-plots', '--no-remove-lines', '--no-fits2fits', '--crpix-center', '--tweak-order', '1' ] #, '--no-tweak'] #, '-z', '2']
		
		if self.ra is not None:
			cmd_s = cmd_s + ['--ra',  str(self.ra)]
			if self.radius is not None and self.radius > 0:
				cmd_s = cmd_s + ['--radius', str(self.radius)]
		if self.dec is not None:
			cmd_s = cmd_s + ['--dec', str(self.dec)]
		
		if self.field_deg is not None:
			cmd_s = cmd_s + ['--scale-low', str(self.field_deg * 0.95), '--scale-high', str(self.field_deg * 1.05), '--odds-to-solve', '1e6']
		else:
			cmd_s = cmd_s + ['--odds-to-solve', '1e8']
		
		if (self.sources_img is None):
			cmd_s = cmd_s + ['--sort-column', 'FLUX', tmp_dir + "/field.fits"]
		else:
			cmd_s = cmd_s + [tmp_dir + "/field.tif", '-z', '2']

		conf_list = ['conf-41', 'conf-42-1', 'conf-42-2' ]
		for conf in conf_list:
			cmd_s1 = cmd_s + [ '--config', conf + '.cfg', '--out', conf ]
			print cmd_s1
			cmd = subprocess.Popen(cmd_s1, preexec_fn=os.setpgrp)
			self.cmd.append(cmd)
		
		solved = None
		while solved is None:
			running = False
			for cmd, conf in zip(self.cmd, conf_list):
				if cmd.poll() is not None:
					if os.path.exists(tmp_dir + '/' + conf + '.solved'):
						solved = tmp_dir + '/' + conf
						break
				else:
					running = True
			if not running:
				break
			if solved is not None:
				break
			time.sleep(0.1)
		
		self.terminate(wait = False)

		if solved is None or not os.path.exists(solved + ".solved"):
			self.ra = None
			self.dec = None
			self.field_deg = None
			shutil.rmtree(tmp_dir)
			return
	
		self.wcs = Tan(solved + ".wcs", 0)
		self.ra, self.dec = self.wcs.radec_center()
		self.field_deg = self.field_w * self.wcs.pixel_scale() / 3600
		
		ind = pyfits.open(solved + '-indx.xyls')
		tbdata = ind[1].data
		self.ind_sources = []
		self.ind_radec = []
		for l in tbdata:
			x = np.clip(int(l['X']), 0, self.field_w - 1)
			y = np.clip(int(l['Y']), 0, self.field_h - 1)
			self.ind_sources.append((x,y))
			self.ind_radec.append(self.wcs.pixelxy2radec(l['X'], l['Y']))
		shutil.rmtree(tmp_dir)
		self.solved = True


	def terminate(self, wait = True):
		for cmd in self.cmd:
			if cmd.poll() is not None:
				continue
		
			try:
				pgid = os.getpgid(cmd.pid)
				os.killpg(pgid, signal.SIGTERM)
			except:
				print "Unexpected error:", sys.exc_info()
			try:
				cmd.terminate()
			except:
				print "Unexpected error:", sys.exc_info()
				
		if wait:
			self.join()
				

	

class Plotter:
	def __init__(self, wcs):
		self.wcs = wcs

	def plot(self, img = None, off = [0., 0.], extra = [], extra_lines = [], grid = True, scale = 1):
		field_w = self.wcs.get_width()
		field_h = self.wcs.get_height()

		try:
			scale = float(scale)
		except ValueError:
			if scale.startswith('deg'):
				scale = float(scale[3:]) / (self.wcs.pixel_scale() * field_w / 3600.0)
			else:
				raise

		# the field moved by given offset pixels from the position in self.wcs
		(crpix1, crpix2) = self.wcs.crpix
		ra2, dec2 = self.wcs.pixelxy2radec(crpix1 - off[1], crpix2 - off[0])
		

		# plot extra area with the original image in the center
		xborder = field_w * (scale - 1) / 2.0
		yborder = field_h * (scale - 1) / 2.0
		nw = field_w * scale
		nh = field_h * scale
	
		new_wcs = Tan(self.wcs)
		new_wcs.set_crval(ra2, dec2)

		new_wcs.set_width(nw)
		new_wcs.set_height(nh)
		
		new_wcs.set_crpix(crpix1 + xborder, crpix2 + yborder)

		#thumbnail size and pos
		tw = int(field_w / scale)
		th = int(field_h / scale)
		tx = int(xborder / scale)
		ty = int(yborder / scale)


		
		plot = Plotstuff(outformat=PLOTSTUFF_FORMAT_PPM)
		plot.wcs = anwcs_new_tan(new_wcs)
		plot.scale_wcs(1.0 / scale)

		plot.set_size_from_wcs()

		plot.color = 'black'
		plot.alpha = 1.0
		plot.plot('fill')

		plot.color = 'gray'
		
		plot_area_deg = new_wcs.pixel_scale() * scale * field_w / 3600.0
		print "plot area", plot_area_deg
		
		
		if grid:
			log_step = [1, 2, 5];
			log_ra = int(np.floor(np.log10(plot_area_deg / (0.01 + math.cos(math.radians(dec2)))) * 3)) - 1
			log_dec = int(np.floor(np.log10(plot_area_deg) * 3)) - 1
			grid_step_ra = 10 ** int(log_ra / 3) * log_step[log_ra % 3]
			grid_step_dec = 10 ** int(log_dec / 3) * log_step[log_dec % 3]
			grid_step_ra = min(10, grid_step_ra)
			grid_step_dec = min(10, grid_step_dec)
			print "grid", plot_area_deg, log_ra, log_dec, grid_step_ra, grid_step_dec
			plot.plot_grid(grid_step_ra, grid_step_dec, grid_step_ra, grid_step_dec)

		ann = plot.annotations
		ann.NGC = (plot_area_deg < 60)
		ann.constellations = (plot_area_deg > 10)
		ann.constellation_labels = (plot_area_deg > 10)
		ann.constellation_labels_long = False
		ann.bright = (plot_area_deg < 60)
		ann.ngc_fraction = 0.05 / scale

		ann.HD = (plot_area_deg < 9)
		ann.HD_labels = (plot_area_deg < 3)
		ann.hd_catalog = "hd.fits"

		for (r, d, name) in extra:
			ann.add_target(r, d, name)

		for (r1, d1, r2, d2) in extra_lines:
			plot.move_to_radec(r1, d1)
			plot.line_to_radec(r2, d2)
			plot.stroke()

		plot.color = 'green'
		plot.lw = 2
		plot.valign = 'B'
		plot.halign = 'C'
		plot.label_offset_x = 0;
		plot.label_offset_y = -20;

		plot.plot('annotations')
		
		# frame around the image
		if scale > 1:
			plot.color = 'blue'
			plot.polygon([(tx, ty), (tx + tw, ty), (tx + tw, ty + th), (tx, ty + th)])
            		plot.close_path()
            		plot.stroke()

		plot_image = plot.get_image_as_numpy()

		bg = np.zeros_like(plot_image)
		if img is not None:
			if scale == 1:
				thumb = img
			else:
				thumb = cv2.resize(img, (tw, th))
			bg[ty:ty+th, tx:tx+tw, 0] = bg[ty:ty+th, tx:tx+tw, 1] = bg[ty:ty+th, tx:tx+tw, 2] = thumb
		
		res=cv2.add(plot_image, bg)

		return res



if __name__ == "__main__":
	img = cv2.imread("field.tif")
	img = np.amin(img, axis = 2)

	solver = Solver(sources_img = img)
	solver.start()
	solver.join()
	
	plotter=Plotter(solver.wcs)
	plot = plotter.plot(img, scale = 5)
	cv2.imshow("plot", plot)
	cv2.waitKey(0)
