import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras
import tensorflow as tf
import numpy as np

def visualisation_GradCam(mdl,last_conv_layer_name = 'conv5_block3_out',pred_index=None):
    '''
    Deep Visualisation with Grad-CAM method and Guided Grad-CAM

    Parameter: 
    mdl - model to be visualized
    last_conv_layer_name - the name of last convoluntional layer, here conv5_block3_out, to get the values for `last_conv_layer_name` use `model.summary()`
    pred_index - prediction index

    Reference:
    1. https://keras.io/examples/vision/grad_cam/
    2. https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
    '''

    #img_path = input('Enter the path for the test image for GradCam: ') # enter the path for visualisation
    img_path = 'diabetic_retinopathy/dataset/1. Original Images/a. Training Set/IDRiD_001.jpg' # here we visualize with a specific image
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img,tf.float32)
    img = tf.image.resize_with_pad(img, 256, 256, antialias=True)
    img_showed = tf.cast(img,tf.float32)/255.0
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model([mdl.input], [mdl.get_layer(last_conv_layer_name).output, mdl.output]) #A mapping that outputs target convolution and output

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = np.argmax(preds[0])
            loss = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    output = conv_outputs[0]

    # Apply guided backpropagation
    grads = tape.gradient(loss,conv_outputs)[0]
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads 

    #The average of the gradients are taken here
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    weights_gradcam = tf.reduce_mean(grads, axis=(0, 1))

    #Based on the prominence of the gradients, a heatmap is generated
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    cam_gradcam = np.ones(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:,:,index]

    for index, w in enumerate(weights_gradcam):
        cam_gradcam += w * output[:,:,index]

    cam = cv2.resize(cam.numpy(), (256, 256))
    cam_gradcam = cv2.resize(cam_gradcam.numpy(), (256, 256))

    cam = np.maximum(cam, 0)
    cam_gradcam = np.maximum(cam_gradcam, 0)

    # Heatmap Visualization
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap_gradcam = (cam_gradcam - cam_gradcam.min()) / (cam_gradcam.max() - cam_gradcam.min())

    img = np.asarray(img).astype('float32')

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    
    cam_gradcam = cv2.applyColorMap(np.uint8(255*heatmap_gradcam), cv2.COLORMAP_JET)
    output_image_gradcam = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam_gradcam, 1.0, 0)

    b, g, r = cv2.split(output_image)
    output_image = cv2.merge([r, g, b])

    b, g, r = cv2.split(output_image_gradcam)
    output_image_gradcam = cv2.merge([r, g, b])

    #Plot the images
    plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    plt.title('Test Image with Diabetic Retinopathy')
    plt.imshow(img_showed)

    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM ')
    plt.imshow(output_image)

    plt.subplot(1, 3, 3)
    plt.title('Grad-CAM + Guided Backpropagation')
    plt.imshow(output_image_gradcam)

    plt.savefig('diabetic_retinopathy/images/deep_visualisation.jpg')


