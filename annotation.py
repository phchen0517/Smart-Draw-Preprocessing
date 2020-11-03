# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import os, sys, zipfile
import shutil
import numpy as np
import skimage.io as io
import pylab
import json
from pycocotools.coco import COCO
import csv
import numpy
import math


def inbox( x1, y1, w1, h1, x2, y2, w2, h2 ) :
	if x1 > x2+w2 :
		return 0
	if y1 > y2+h2 :
		return 0
	if x1 + w1 < x2 :
		return 0
	if y1+h1<y2 :
		return 0
    
	colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
	rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
	
	overlap_area = colInt * rowInt
	area1 = w1 * h1
	area2 = w2 * h2

	if area1 + area2 - overlap_area == 0 :
		return 1
	else :
		return overlap_area / (area1 + area2 - overlap_area)
		

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

json_file = 'instances_train2017.json' # # Object Instance 型別的標註

data = json.load( open( json_file, 'r' ) )

outputFile = open( 'output.csv', 'w' )
writer = csv.writer( outputFile )

writer.writerow('')

for pic in range ( len( data['images'] ) ) :
	annotation = []
	imgID = data['images'][pic]['id']
	picHeight = data['images'][pic]['height']
	picWidth = data['images'][pic]['width']
	#coco = COCO( json_file )
	#img = coco.loadImgs( [imgID] )
	#I = io.imread( 'val2017/%s' % img[0]['file_name'] )


	for ann in data['annotations'] :
		if ann['image_id'] == imgID :
			temp = []
			
			x1 = ann['bbox'][0]
			x2 = ann['bbox'][0] + ann['bbox'][2]
			y1 = ann['bbox'][1]
			y2 = ann['bbox'][1] + ann['bbox'][3]
			area = ann['bbox'][2] * ann['bbox'][3]
			center_x = x1 + ann['bbox'][2] / 2
			center_y = y1 + ann['bbox'][3] / 2
			diagonal = round( numpy.sqrt( numpy.square( ann['bbox'][2] ) + numpy.square( ann['bbox'][3] ) ) , 3 )


			if ann['bbox'][2] > ( picWidth / 50 ) and ann['bbox'][3] > ( picHeight / 50 ) :

				# 從80類裡面找到他的種類
				for cate in data['categories'] :
					if cate['id'] == ann['category_id'] :
						temp.append( cate['name'] ) # temp[0]


				temp.append( x1 ) # temp[1]
				temp.append( x2 ) # temp[2]
				temp.append( y1 ) # temp[3]
				temp.append( y2 ) # temp[4]
				temp.append( area ) # temp[5]
				temp.append( center_x ) # temp[6]
				temp.append( center_y ) # temp[7]
				temp.append( ann['bbox'][2] ) # temp[8] width
				temp.append( ann['bbox'][3] ) # temp[9] height
				temp.append( diagonal ) # temp[10]

				annotation.append( temp )


	# 做組合比較
	compare = []
	for i in range ( len( annotation ) ) :
		for j in range( i+1, len( annotation ) ) :
			temp = []

			temp.append( imgID )
			
			if annotation[i][5] < annotation[j][5] : # 用i當基準(小)
				temp.append( annotation[i][0] )
				temp.append( annotation[i][8] )
				temp.append( annotation[i][9] )
				temp.append( annotation[j][0] )
				temp.append( annotation[j][8] )
				temp.append( annotation[j][9] )

				w_mul = annotation[j][8] / annotation[i][8]
				h_mul = annotation[j][9] / annotation[i][9]
				temp.append( w_mul )
				temp.append( h_mul )

				width = annotation[i][6] - annotation[j][6]
				height = annotation[i][7] - annotation[j][7]

				# box
				boxi = ( annotation[i][1], annotation[i][2], annotation[i][3], annotation[i][4] ) #(x1, x2, y1, y2)
				boxj = ( annotation[i][1], annotation[j][2], annotation[j][3], annotation[j][4] )
				result = inbox( annotation[i][1], annotation[i][2], annotation[i][3], annotation[i][4], annotation[i][1], annotation[j][2], annotation[j][3], annotation[j][4] )

				# 判斷左右or上下/inoutbox
				if annotation[i][1] > annotation[j][1] and annotation[i][2] < annotation[j][2] :
					if annotation[i][7] > annotation[j][7] :
						temp.append( 'down' )
						temp.append( 'up' )
					else :
						temp.append( 'up' )
						temp.append( 'down' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				elif annotation[i][3] > annotation[j][3] and annotation[i][4] < annotation[j][4] :
					if annotation[i][6] > annotation[j][6] :
						temp.append( 'right' )
						temp.append( 'left' )
					else :
						temp.append( 'left' )
						temp.append( 'right' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				elif height > width :
					if annotation[i][7] > annotation[j][7] :
						temp.append( 'down' )
						temp.append( 'up' )
					else :
						temp.append( 'up' )
						temp.append( 'down' )

					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				else :
					if annotation[i][6] > annotation[j][6] :
						temp.append( 'right' )
						temp.append( 'left' )
					else :
						temp.append( 'left' )
						temp.append( 'right' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )

				# distance & angle
				x_distance = annotation[j][6] - annotation[i][6]
				y_distance = annotation[j][7] - annotation[i][7]


				distance = round( numpy.sqrt( numpy.square( x_distance ) + numpy.square( y_distance ) ) / annotation[i][10] , 3 )
				temp.append( distance )

				angle = round( math.atan2( height, width ) / math.pi * 180 )

				if angle < 0 :
					angle = angle + 360

				temp.append( angle )

				
			
			else : # 用j當基準(小)
				temp.append( annotation[j][0] )
				temp.append( annotation[j][8] )
				temp.append( annotation[j][9] )
				temp.append( annotation[i][0] )
				temp.append( annotation[i][8] )
				temp.append( annotation[i][9] )

				w_mul = annotation[i][8] / annotation[j][8]
				h_mul = annotation[i][9] / annotation[j][9]
				temp.append( w_mul )
				temp.append( h_mul )

				width = annotation[j][6] - annotation[i][6]
				height = annotation[j][7] - annotation[i][7]


				# box
				boxi = ( annotation[i][1], annotation[i][2], annotation[i][3], annotation[i][4] ) #(x1, x2, y1, y2)
				boxj = ( annotation[i][1], annotation[j][2], annotation[j][3], annotation[j][4] )
				result = inbox( annotation[i][1], annotation[i][2], annotation[i][3], annotation[i][4], annotation[i][1], annotation[j][2], annotation[j][3], annotation[j][4] )

				# 判斷左右or上下/inoutbox
				if annotation[j][1] > annotation[i][1] and annotation[j][2] < annotation[i][2] :
					if annotation[j][7] > annotation[i][7] :
						temp.append( 'down' )
						temp.append( 'up' )
					else :
						temp.append( 'up' )
						temp.append( 'down' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				elif annotation[j][3] > annotation[i][3] and annotation[j][4] < annotation[i][4] :
					if annotation[j][6] > annotation[i][6] :
						temp.append( 'right' )
						temp.append( 'left' )
					else :
						temp.append( 'left' )
						temp.append( 'right' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				elif height > width :
					if annotation[j][7] > annotation[i][7] :
						temp.append( 'down' )
						temp.append( 'up' )
					else :
						temp.append( 'up' )
						temp.append( 'down' )

					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )
				else :
					if annotation[j][6] > annotation[i][6] :
						temp.append( 'right' )
						temp.append( 'left' )
					else :
						temp.append( 'left' )
						temp.append( 'right' )
					
					temp.append( result )
					if result >= 0.5 :
						temp.append( 'inbox' )
					else :
						temp.append( 'outbox' )

				# distance & angle
				x_distance = annotation[i][6] - annotation[j][6]
				y_distance = annotation[i][7] - annotation[j][7]


				distance = round( numpy.sqrt( numpy.square( x_distance ) + numpy.square( y_distance ) ) / annotation[j][10] , 3 )
				temp.append( distance )

				angle = round( math.atan2( height, width ) / math.pi * 180 )

				if angle < 0 :
					angle = angle + 360

				temp.append( angle )


			
			compare.append( temp )
		
	writer.writerows( compare )

outputFile.close()