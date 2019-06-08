import numpy as np
import tarfile
import matplotlib
import time
import uuid
import zipfile
import os
import six.moves.urllib as urllib
import sys

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# This shorthand makes one module visible to the other
import sys
sys.path.append('../') 

from utils import ops as utils_ops
from matplotlib import pylab
from pylab import *

# import easygui
# image_input = easygui.fileopenbox()


# This shorthand makes one module visible to the other
import sys
sys.path.append('../') 

from utils import ops as utils_ops

# To provide the label to the i/p images
from utils import label_map_util  

from utils import visualization_utils as vis_util 

use_model = 'ssd_mobilenet_v1_coco_2018_01_28'
file_type = use_model + ".tar.gz"

# Tensorflow works on frozen graph methodology,
path_to_checkpoint = use_model + '/frozen_inference_graph.pb'

# name of the objects that can be used to add correct label for each box
path_for_labels = os.path.join('data', 'mscoco_label_map.pbtxt')

num_of_classes = 90

#Frozen Tensorflow model in memory
#loading the frozen inference graph in the memory
#definition of the graph that is to be used

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')

#load the labels and the category, also the category index from the dataset
#label maps map indices to category names
category_index = label_map_util.create_category_index_from_labelmap(path_for_labels, use_display_name=True)

#conversion of images to numpy array
def make_image_to_numpy_array(image):
    (image_width, image_height) = image.size
    return np.array(image.getdata()).reshape(
    (image_height, image_width, 3)).astype(np.uint8)
    
#path to the test images

# path_for_test_image_dir = "test_images"
# test_image_path = [ os.path.join(path_for_test_image_dir, 'image{}.jpg'.format(i)) for i in range(1, 3)]
# test_image_path = os.path.join(image_input)

#size of the output image
image_size = (12, 8)

#deduction of the image
def inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
        #Handling the input and output tensors
            ops = tf.get_default_graph().get_operations()
            #giving names to tensor
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            #for a single image, detecting the boxes, scores, classes etc
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                #this is done for the single image.
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                #here reframing is done for to translate the mask from box coordinates to image coordinates
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0,0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0,0,0], [real_num_detection, -1, -1])

                #reframing of detection mask

                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, 
                                                                                      detection_boxes,
                                                                                     image.shape[1],
                                                                                     image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

            #the output generated is in the form of float32,datatype.Conversion is needed

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

#This happens to be the final for loop, where image is taken from the path, and inference is done one by one

path = 'images'

n = 1

# for image_path in os.listdir(path):

def final(image_path):

    input_path = os.path.join(path, image_path)
    image = Image.open(input_path)

    #loading the image into the numpy array


    image_to_np = make_image_to_numpy_array(image)
    #expansion of the image is done, as it is based on the category
    #axis 0 here allows here means expansion column wise
    image_np_expanded = np.expand_dims(image_to_np, axis=0)

    #detection of the image

    output_dict = inference_for_single_image(image_np_expanded, detection_graph)

    #visualization
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_to_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks = output_dict.get('detection_masks'),
        use_normalized_coordinates = True,
        line_thickness = 8)

    plt.figure(figsize = image_size)
    plt.imshow(image_to_np)
    plt.savefig('out_here/' + image_path)

    return 'The file has successfully processed'
    

    

    # print("Image %d is stored in out_here folder" %(n))

    # n = n+1
