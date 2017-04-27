import os
import cv2
import numpy as np

import video
from common import anorm2, draw_str
import re
import pdb
import h5py
os.chdir('/scratch/ys1297/LSTM_tracking/resource/vot2016/')

#temp = os.listdir('./')
#for item in temp:
#    if os.path.isdir(item):
#        # cd to subfolder
#        folder_name = './'+item
#        os.chdir(folder_name)
#        
#        with open('groundtruth.txt','r') as txt_file:
#            line = True
#            count = 0
#            total_num = len(txt_file.readlines())
#            h5file_name = '../'+ item+'.h5'
#            file = h5py.File(h5file_name,'w')
#            file.create_dataset('image',(total_num,224,224,3))
#            file.create_dataset('label',(total_num,8))
#        with open('groundtruth.txt','r') as txt_file:
#            
#            while True:
#                line = txt_file.readline()
#                if count == total_num:
#                    break
#                line = re.sub('\n','',line)
#                #pdb.set_trace()
#                nums = [(float(num)) for num in re.split(',',line)]
#                
#                count += 1
#                file_name = str(count).zfill(8)+'.jpg'
#                img = cv2.imread(file_name)
#                height,width = img.shape[0],img.shape[1]
#
#
#                img = cv2.resize(img,(224,224))
#                
#                
#                nums[0:8:2] = [int(item/width*224.0) for item in nums[0:8:2]]
#                nums[1:8:2] = [int(item/height*224.0) for item in nums[1:8:2]]
#                file['image'][count-1] = img
#                file['label'][count-1] = np.array(nums).squeeze()
#                print count
#            file.close()
#
#        os.chdir('../')

temp = os.listdir('./')
for item in temp:
    if os.path.isdir(item):
        # cd to subfolder
        folder_name = './'+item
        os.chdir(folder_name)

        with open('groundtruth.txt','r') as txt_file:
            line = True
            count = 0
            total_num = len(txt_file.readlines())
            h5file_name = '../'+ item+'origin.h5'
            file = h5py.File(h5file_name,'w')
	    img = cv2.imread("00000001.jpg")
	    height,width = img.shape[0],img.shape[1]
	    file.create_dataset('image',(total_num,height,width,3))
	    file.create_dataset('label',(total_num,8))

	with open('groundtruth.txt','r') as txt_file:

            while True:
                line = txt_file.readline()
                if count == total_num:
                    break
                line = re.sub('\n','',line)
                #pdb.set_trace()
                nums = [int(float(num)) for num in re.split(',',line)]

                count += 1
                file_name = str(count).zfill(8)+'.jpg'
                img = cv2.imread(file_name)
		file['image'][count-1] = img
                file['label'][count-1] = np.array(nums).squeeze()
		print count
            file.close()

        os.chdir('../')
		




#        cv2.line(img,(nums[0],nums[1]),(nums[2],nums[3]),(255,0,0),2)
#        cv2.line(img,(nums[2],nums[3]),(nums[4],nums[5]),(255,0,0),2)
#        cv2.line(img,(nums[4],nums[5]),(nums[6],nums[7]),(255,0,0),2)
#        cv2.line(img,(nums[6],nums[7]),(nums[0],nums[1]),(255,0,0),2)
#        
#        cv2.imshow('fret',img)
#        pdb.set_trace()
#        cv2.destroyWindow('fret')
