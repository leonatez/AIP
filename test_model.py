# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

import imageio
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys
from datetime import datetime

def enhance_img(image_upload):
    
    tf.compat.v1.disable_v2_behavior()

    # settings:
    phone = "iphone_orig"
    dped_dir = 'img/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"
    resize_level = 1

    # get all available image resolutions
    res_sizes = utils.get_resolutions()

    # get the specified image resolution
    # IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)
    get_size = imageio.imread("img/" + image_upload)
    IMAGE_HEIGHT = int(get_size.shape[0]/resize_level)
    IMAGE_WIDTH = int(get_size.shape[1]/resize_level)
    IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3

    # disable gpu if specified
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

    # create placeholders for input images
    x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # generate enhanced image
    enhanced = resnet(x_image)

    with tf.compat.v1.Session(config=config) as sess:

        # load pre-trained model
        now1 = datetime.now()
        # tf.compat.v1.reset_default_graph()
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models_orig/" + phone)
        complete_loading_time = datetime.now() - now1

        # load training image and crop it if necessary
        now = datetime.now()
        print("Preprocessing picture....")
        image = np.float16(np.array(Image.fromarray(imageio.imread("img/" + image_upload)).resize([IMAGE_WIDTH, IMAGE_HEIGHT]))) / 255
        image = image[:,:,:3]
        image_crop_2d = np.reshape(image, [1, IMAGE_SIZE])

        complete_loading_image_time = datetime.now() - now
                
            
        # get enhanced image
        now = datetime.now()
        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
        sess.close()
        print(sess)
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        complete_enhancement_time = datetime.now() - now
        photo_name = image_upload.rsplit(".", 1)[0]

        # save the results as .png images
        now = datetime.now()
        print("Saving picture....")
        enhanced_image = (enhanced_image*250).astype(np.uint8)
        imageio.imwrite("visual_results/" + photo_name + ".png", enhanced_image)
        pics_name = "visual_results/" + photo_name + ".png"
        complete_writing_time = datetime.now() - now
        total_time = datetime.now()-now1

        #PERFORMANCE REPORT
        print('='*50)
        print('PERFORMANCE REPORT FOR ', pics_name)
        print('='*50)
        print('Operation is successful after: ', total_time)
        print('-'*50)
        print("Loading model completed after ",complete_loading_time)
        print("Loading image completed after ",complete_loading_image_time)
        print("Enhancement is successul at", complete_enhancement_time)
        print("Saving image is successful after: ", complete_writing_time)
        print('='*50)

    tf.compat.v1.reset_default_graph()

    return pics_name
    
    