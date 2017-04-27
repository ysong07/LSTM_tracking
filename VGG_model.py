import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
from matplotlib.path import Path
import os
import cv2 
data_dict =np.load('../model/params.npy').item()
def feature_extract(batch_frames,mask_frames):
    def avg_pool( bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
            
    def max_pool( bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(bottom, name):
        with tf.variable_scope(name):
            filt = get_conv_filter(name)
        
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)
        
        relu = tf.nn.relu(bias)
        return relu
    def get_conv_filter(name):
        return tf.constant(data_dict[name][0], name="filter")
    
    def get_bias(name):
        return tf.constant(data_dict[name][1], name="biases")
    
    def get_fc_weight( name):
        return tf.constant(data_dict[name][0], name="weights")
    
    
    VGG_MEAN = [103.939, 116.779, 123.68]
    
    img_shape = [1,224,224]
    rgb_scaled = batch_frames
    red = rgb_scaled[:,:,:,0]
    green = rgb_scaled[:,:,:,1]
    blue = rgb_scaled[:,:,:,2]
    bgr= tf.stack([blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2]])
    bgr= tf.transpose(bgr, [1,2,3,0])
    #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
   
    conv1_1 = conv_layer(bgr, "conv1_1")
    conv1_2 = conv_layer(conv1_1, "conv1_2")
    conv1_2 = tf.multiply(conv1_2,mask_frames)

    pool1 = max_pool(conv1_2, 'pool1')
    
    conv2_1 = conv_layer(pool1, "conv2_1")
    conv2_2 = conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')
    
    conv3_1 = conv_layer(pool2, "conv3_1")
    conv3_2 = conv_layer(conv3_1, "conv3_2")
    conv3_3 = conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')
    
    conv4_1 = conv_layer(pool3, "conv4_1")
    conv4_2 = conv_layer(conv4_1, "conv4_2")
    conv4_3 = conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')
    
    #conv5_1 = conv_layer(pool4, "conv5_1")
    #conv5_2 = conv_layer(conv5_1, "conv5_2")
    #conv5_3 = conv_layer(conv5_2, "conv5_3")
    #pool5 = max_pool(conv5_3, 'pool5')
    
    
    return pool4

def construct_mask(poly_verts,nx=224,ny=224,nchan=64):
    #nx, ny = 10, 10
    #poly_verts = [(1,1), (5,1), (5,9),(3,2),(1,1)]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx,1))

    grid = np.repeat(grid,nchan,axis=2)
    return grid
def crop_image_include_coundary(output,input,center):
  height,width = output.shape[0],output.shape[1]
  min_x = max(0,int(center[0]-height/2))
  min_out_x = max(0,int(height/2-center[0]))

  min_y = max(0,int(center[1]-width/2))
  min_out_y = max(0,int(width/2-center[1]))

  max_x = min(input.shape[0],int(center[0]+height/2))
  max_out_x = min(height, height+ input.shape[0]- int(center[0]+height/2))
  max_y =  min(input.shape[1],int(center[1]+width/2))
  max_out_y = min(width, width+ input.shape[1]- int(center[1]+width/2))

  try:
    output[min_out_x:max_out_x,min_out_y:max_out_y,:] = input[min_x:max_x,min_y:max_y,:]
  except:
    pdb.set_trace()

  return output


if __name__ == '__main__':


    #""" pre-store features """
    #os.chdir('../data')
    #for item in os.listdir('./'):
    #  file_name = item
    #  print file_name
    #  file = h5py.File(file_name,'a')
    #  img = file['image']
    #  label = file['label']

    #  try:   
    #    file.create_dataset('mask',(img.shape[0],224,224,1))
    #    file.create_dataset('feature',(img.shape[0],14,14,512)) 
    #    feature = []
    #    sess = tf.Session()
    #    input_img = tf.placeholder(tf.float32, shape=[1,224,224,3])
    #    mask =  tf.placeholder(tf.float32, shape=[1,224,224,64]) # 64: number of channel of conv1

    #    output_feature = feature_extract(input_img,mask)
    #    for i in range(img.shape[0]):
    #      input_ = img[i,:,:,:].reshape(1,224,224,3)
    #      list_pos = label[i]
    #      poly_verts = [(list_pos[0],list_pos[1]),(list_pos[2],list_pos[3]),(list_pos[4],list_pos[5]),(list_pos[6],list_pos[7]),(list_pos[0],list_pos[1])]
    #      save_mask = construct_mask(poly_verts,nchan=1) # construct spacial mask for saving
    #      file['mask'][i] = save_mask
    #      
    #      #mask_ = construct_mask(poly_verts)
    #      #mask_ = np.reshape(mask_,(1,224,224,64))
    #      mask_ = np.ones((1,224,224,64))
    #      temp = sess.run(output_feature,feed_dict = {input_img:input_,mask:mask_})	
    #      file['feature'][i] = temp
    #      #feature.append(temp)
    #  except:
    #    pass
    #  file.close()

    sess = tf.Session()	
    input_img = tf.placeholder(tf.float32, shape=[1,224,224,3])
    mask =  tf.placeholder(tf.float32, shape=[1,224,224,64]) # 64: number of channel of conv1
    output_feature = feature_extract(input_img,mask)
    os.chdir('../data_origin')
    for item in os.listdir('./'):
      file_name = item
      print file_name
      file = h5py.File(file_name,'a')
      img = file['image']
      label = file['label']
   
      try:
        file.create_dataset('mask',(img.shape[0],224,224,1))
        file.create_dataset('feature',(img.shape[0],14,14,512))
        file.create_dataset('first_frame_feature',(img.shape[0],14,14,512)) 
      except:
	pass
      try:
        file.create_dataset('crop_image',(img.shape[0],224,224,3))
      except:
	pass
      feature = []
      for i in range(img.shape[0]):
          pos = label[i]
	  center = [int((pos[1]+pos[3]+pos[5]+pos[7])/4.0),int((pos[0]+pos[2]+pos[4]+pos[6])/4.0)]
	  frame = img[i]
	  list_pos = pos
	  poly_verts = [(list_pos[0],list_pos[1]),(list_pos[2],list_pos[3]),(list_pos[4],list_pos[5]),(list_pos[6],list_pos[7]),(list_pos[0],list_pos[1])]
	  mask_img = construct_mask(poly_verts,nx = frame.shape[1],ny= frame.shape[0],nchan=1)

	  crop_frame = crop_image_include_coundary(np.zeros((224*2,224*2,3)),frame,center)
	  frame = cv2.resize(crop_frame,(224,224))
	  file['crop_image'][i] = frame
	  #crop_mask = crop_image_include_coundary(np.zeros((224*2,224*2,1)),mask_img,center)
	  #frame = cv2.resize(crop_frame,(224,224))
          #mask_img = cv2.resize(crop_mask,(224,224))
	  #mask_input=  np.repeat(mask_img.reshape(1,224,224,1),64,axis=3)
	  #file['mask'][i] =  mask_img.reshape(224,224,1)
	  #file['first_frame_feature'][i] = sess.run(output_feature,feed_dict = {input_img:frame.reshape(1,224,224,3),mask:mask_input})  	    
          #file['feature'][i] = sess.run(output_feature,feed_dict = {input_img:frame.reshape(1,224,224,3),mask:np.ones((1,224,224,64))}) 
	  print i
      file.close()

	    
	
	    
	


 
      
	
