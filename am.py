import cv2
import numpy as np

import os
import subprocess

import pyfits
from astrometry.util.util import anwcs
from astrometry.blind.plotstuff import *
import threading


class Solver(threading.Thread):
	def __init__(self, sources_img = None, sources_list = None, field_w = None, field_h = None, ra = None, dec = None, field_deg = None):
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
		self.solved = False
		self.cmd = None
		
	
	def run(self):
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
			thdulist.writeto('field.fits', clobber=True)
		else:
			cv2.imwrite("field.tif", self.sources_img)
		
		if os.path.exists("field.solved"):
			os.remove("field.solved")
		
		cmd_s = ['solve-field', '-O',  '--objs', '20', '--depth', '20', '-E', '2', '--no-plots', '--no-remove-lines', '--no-fits2fits', '--crpix-center', '--no-tweak']
		
		if self.ra is not None:
			cmd_s = cmd_s + ['--ra',  str(self.ra)]
		if self.dec is not None:
			cmd_s = cmd_s + ['--dec', str(self.dec)]
		
		if self.field_deg is not None:
			cmd_s = cmd_s + ['--radius', str(self.field_deg * 2), '--scale-low', str(self.field_deg * 0.95), '--scale-high', str(self.field_deg * 1.05), '--odds-to-solve', '1e6']
		else:
			cmd_s = cmd_s + ['--odds-to-solve', '1e8']
		
		if (self.sources_img is None):
			cmd_s = cmd_s + ['--sort-column', 'FLUX', 'field.fits']
		else:
			cmd_s = cmd_s + ['field.tif']

		self.cmd = subprocess.Popen(cmd_s)
		self.cmd.wait()

		if not os.path.exists("field.solved"):
			self.ra = None
			self.dec = None
			self.field_deg = None
			return
	
		self.solved = True
		self.wcs = anwcs('field.wcs',0)
		self.ra, self.dec = self.wcs.get_center()
		self.field_deg = self.field_w * self.wcs.pixel_scale() / 3600
		
		ind = pyfits.open('field-indx.xyls')
		tbdata = ind[1].data
		self.ind_sources = []
		for l in tbdata:
			x = np.clip(int(l['X']), 0, self.field_w - 1)
			y = np.clip(int(l['Y']), 0, self.field_h - 1)
			self.ind_sources.append((x,y))


	def terminate(self, wait = True):
		if self.cmd is not None:
			self.cmd.terminate()
		if (wait):
			self.join()
	

class Plotter:
	def __init__(self, wcs):
		self.wcs = wcs

	def plot(self, img, off):
		self.wcs.writeto('off1_field.wcs')
		
		ra, dec = self.wcs.get_center()
		ok, x0, y0 = self.wcs.radec2pixelxy(ra, dec)
		ok, ra2, dec2 = self.wcs.pixelxy2radec(x0 - off[1], y0 - off[0])
		print ra,dec, ra2, dec2
		
		wcs = pyfits.open('off1_field.wcs')
		wcs_h = wcs[0].header
		wcs_h['CRVAL1'] = ra2
		wcs_h['CRVAL2'] = dec2
		wcs.writeto('off2_field.wcs', clobber=True)

		
		plot = Plotstuff(outformat=PLOTSTUFF_FORMAT_PPM, wcsfn='off2_field.wcs')

		plot.set_size_from_wcs()

		plot.color = 'black'
		plot.alpha = 1.0
		plot.plot('fill')

		plot.color = 'gray'
		grid_step = 10 ** round (np.log10(self.wcs.pixel_scale() /3600) + 2)
		plot.plot_grid(grid_step, grid_step, grid_step * 2, grid_step * 2)

		ann = plot.annotations
		ann.NGC = True
		ann.constellations = False
		ann.constellation_labels = False
		ann.constellation_labels_long = False
		ann.bright = True
		ann.ngc_fraction = 0.05

		ann.HD = True
		ann.HD_labels = True
		ann.hd_catalog = "hd.fits"

		plot.color = 'green'
		plot.lw = 2
		plot.valign = 'B'
		plot.halign = 'C'
		plot.label_offset_x = 0;
		plot.label_offset_y = -20;

		plot.plot('annotations')


		plot_image = plot.get_image_as_numpy()
	
		bg = np.zeros_like(plot_image)
		bg[:, :, 0] = bg[:, :, 1] = bg[:, :, 2] = img
	
		res=cv2.add(plot_image, bg)
		return res

		
	
	
	def plot_viewfinder(self, img, scale, res_w = 1200):
		field_w = self.wcs.get_width()
		field_h = self.wcs.get_height()
		downscale = res_w / (field_w * scale)
	
		xoff = field_w * (scale - 1) / 2.0
		yoff = field_h * (scale - 1) / 2.0
		nw = field_w * scale
		nh = field_h * scale
	
		wcs = pyfits.open('field.wcs')
		wcs_h = wcs[0].header
		wcs_h['IMAGEW'] = nw
		wcs_h['IMAGEH'] = nh
		wcs_h['CRPIX1'] = wcs_h['CRPIX1'] + xoff
		wcs_h['CRPIX2'] = wcs_h['CRPIX2'] + yoff
		wcs.writeto('nfield.wcs', clobber=True)

		#thumbnail size and pos
		tw = int(field_w * downscale)
		th = int(field_h * downscale)
		tx = int(xoff * downscale)
		ty = int(yoff * downscale)

		plot = Plotstuff(outformat=PLOTSTUFF_FORMAT_PPM, wcsfn='nfield.wcs')

		plot.scale_wcs(downscale)
		plot.set_size_from_wcs()

		plot.color = 'black'
		plot.alpha = 1.0
		plot.plot('fill')

		plot.color = 'gray'
		grid_step = 10 ** round (np.log10(self.wcs.pixel_scale() /3600) + 3)
		plot.plot_grid(grid_step, grid_step, grid_step * 2, grid_step * 2)
		
		ann = plot.annotations
		ann.NGC = True
		ann.constellations = True
		ann.constellation_labels = True
		ann.constellation_labels_long = False
		ann.bright = True
		ann.ngc_fraction = 0.001

		ann.HD = True
		ann.HD_labels = True
		ann.hd_catalog = "hd.fits"

		plot.color = 'green'
		plot.lw = 2
		plot.valign = 'B'
		plot.halign = 'C'
		plot.label_offset_x = 0;
		plot.label_offset_y = -20;

		plot.plot('annotations')

		plot.color = 'blue'
		plot.polygon([(tx, ty), (tx + tw, ty), (tx + tw, ty + th), (tx, ty + th)])
            	plot.close_path()
            	plot.stroke()
            	
		plot_image = plot.get_image_as_numpy()

		thumb = cv2.resize(img, (tw, th))
		bg = np.zeros_like(plot_image)
		bg[ty:ty+th, tx:tx+tw, 0] = bg[ty:ty+th, tx:tx+tw, 1] = bg[ty:ty+th, tx:tx+tw, 2] = thumb
		
		res=cv2.add(plot_image, bg)
		return res


if __name__ == "__main__":
	img = cv2.imread("field.tif")
	img = np.amin(img, axis = 2)

	solver = Solver(sources_img = img, field_deg=0.57)
	solver.start()
	solver.join()
	
	plotter=Plotter(solver.wcs)
	plot = plotter.plot_viewfinder(img, 25)
	cv2.imshow("plot", plot)
	cv2.waitKey(0)
	