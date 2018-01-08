import os
import cv2
import numpy as np
import torch
from screeninfo import get_monitors
import tkinter as tk
from torch.autograd import Variable

from net import Net
from option import Options
import utils
from utils import StyleLoader

def run_demo(args, mirror=False):
	# VARS
	# scaling is to resize the images (cam and art)
	scaling=1.0
	# source_scaling is to resize the used painting 
	source_scaling=1.0
	# x is along the horizontal, y is along the vertical
	# (0,0) is top left
	# art position in fractions of the screen
	art_x = 0.5
	art_y = 0.4
	# source painting position in fractions of the screen
	source_x = 0.4
	source_y = 0.1
	# camera position in fractions of the scteen
	cam_x = 0.05
	cam_y = 0.4

	root = tk.Tk()
	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()
	# fix to resolution of external screen - then it pops to that screen on my linux laptop (hack)
	screen_width=1920
	screen_height=1080
	# calculate positions of images in screen
	cposx=int(cam_x*screen_width)
	sposx=int(source_x*screen_width)
	aposx=int(art_x*screen_width)
	cposy=int(cam_y*screen_height)
	sposy=int(source_y*screen_height)
	aposy=int(art_y*screen_height)
	# read background image
	bimg=cv2.imread('bg.png')
	# resize background to full screen
	bimg = cv2.resize(bimg,(screen_width, screen_height), interpolation = cv2.INTER_CUBIC)
	#code to get focus in window
	cv2.namedWindow("GetFocus", cv2.WINDOW_NORMAL)
	img = np.zeros((100, 100, 1), dtype = "uint8")
	cv2.imshow("GetFocus", img);
	cv2.setWindowProperty("GetFocus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.setWindowProperty("GetFocus", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
	cv2.destroyWindow("GetFocus")
	#load style model
	style_model = Net(ngf=args.ngf)
	style_model.load_state_dict(torch.load(args.model))
	style_model.eval()
	if args.cuda:
		style_loader = StyleLoader(args.style_folder, args.style_size)
		style_model.cuda()
	else:
		style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
	height =  args.demo_size
	width = int(4.0/3*args.demo_size)
	swidth = int(source_scaling*width/4)
	sheight = int(source_scaling*height/4)
	if args.record:
		fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
	cam = cv2.VideoCapture(1)
	cam.set(3, width)
	cam.set(4, height)
	key = 0
	idx = 0
	#start unfrozen
	freeze2art = False 
	#load styles on first use only - speeds up
	style_loaded =False
	#stop when q pressed
	stopped = False
	dimg = bimg.copy()
	while not stopped:
		# read frame
		if not freeze2art:
			ret_val, img = cam.read()
			if not ret_val:
				continue
			if mirror: 
				img = cv2.flip(img, 1)
		cimg = img.copy()
		# if the f has been pressed
		if freeze2art:
			img = np.array(img).transpose(2, 0, 1)
			# changing style (n or p has been pressed or first run)
			if not style_loaded:
				style_v = style_loader.get(idx)
				style_v = Variable(style_v.data, volatile=True)
				style_model.setTarget(style_v)
				style_loaded = True

			img=torch.from_numpy(img).unsqueeze(0).float()
			if args.cuda:
				img=img.cuda()

			img = Variable(img, volatile=True)
			img = style_model(img)

			if args.cuda:
				simg = style_v.cpu().data[0].numpy()
				img = img.cpu().clamp(0, 255).data[0].numpy()
			else:
				simg = style_v.data[0].numpy()
				img = img.clamp(0, 255).data[0].numpy()
			img = img.transpose(1, 2, 0).astype('uint8')
			simg = simg.transpose(1, 2, 0).astype('uint8')

			# display
			#   resize the used painting
			simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
			#   include in the left image
			#   resize the cam image
			cih,ciw = cimg.shape[:2]
			ncih = int(scaling*cih)
			nciw = int(scaling*ciw)
			ccimg = cv2.resize(cimg,(nciw, ncih), interpolation = cv2.INTER_CUBIC)
			cih,ciw = ccimg.shape[:2]
			# position of the cam image to copy into
			a=cposy
			b=cposy+cih
			c=cposx
			d=cposx+ciw
			dimg[a:b,c:d,:]=ccimg
			# position of the source painting image to copy into
			a=sposy
			b=sposy+sheight
			c=sposx
			d=sposx+swidth
			dimg[a:b,c:d,:]=simg
			#   resize the art image
			aih,aiw = img.shape[:2]
			naih = int(scaling*aih)
			naiw = int(scaling*aiw)
			aaimg = cv2.resize(img,(naiw, naih), interpolation = cv2.INTER_CUBIC)
			aih,aiw = aaimg.shape[:2]
			# position of the art image to copy into
			a=aposy
			b=aposy+aih
			c=aposx
			d=aposx+aiw
			dimg[a:b,c:d,:]=aaimg
		else:
			# load on style change or first use
			if not style_loaded:
				style_v = style_loader.get(idx)
				style_v = Variable(style_v.data, volatile=True)
				style_model.setTarget(style_v)
				style_loaded = True
			if args.cuda:
				simg = style_v.cpu().data[0].numpy()
			else:
				simg = style_v.data[0].numpy()
			simg = simg.transpose(1, 2, 0).astype('uint8')

			# display
			#   resize the used painting
			simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
			dimg = bimg.copy()
			#   include in the image
			# position of the source painting image to copy into
			a=sposy
			b=sposy+sheight
			c=sposx
			d=sposx+swidth
			dimg[a:b,c:d,:]=simg
			#   resize the cam image
			cih,ciw = img.shape[:2]
			ncih = int(scaling*cih)
			nciw = int(scaling*ciw)
			ccimg = cv2.resize(img,(nciw, ncih), interpolation = cv2.INTER_CUBIC)
			cih,ciw = ccimg.shape[:2]
			# position of the cam image to copy into
			a=cposy
			b=cposy+cih
			c=cposx
			d=cposx+ciw
			dimg[a:b,c:d,:]=ccimg
		# put the image in the window
		cv2.imshow('Selfie 2 Art', dimg)
		if freeze2art:
			resumed = False
			#wait for r (resume) or q (quit)
			while not resumed:
				key = cv2.waitKey(1)
				if key == ord('r'):
					freeze2art = False
					resumed = True
				if key == ord('q'):
					stopped = True
					resumed = True
		# wait for keys
		key = cv2.waitKey(1)
		if key == 27: 
			stopped = True
		if key == ord('q'):
			stopped = True
		if key == ord('f'):
			freeze2art = True
		if key == ord('n'):
			idx+=1
			if idx>8:
				idx=0
			style_loaded = False
		if key == ord('p'):
			idx-=1
			if idx<0:
				idx=8
			style_loaded = False
	cam.release()
	if args.record:
		out.release()
	cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)

if __name__ == '__main__':
	main()
