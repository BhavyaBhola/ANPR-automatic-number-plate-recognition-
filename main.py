import tensorflow as tf
import numpy as np
from six import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


def load_images_into_numpyarray(path):
    img_data = tf.io.gfile.GFile(path,'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width , im_height) = image.size

    return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)


def detect(interpreter,input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #preprocessed_image,shapes = detection_model.preprocess()
    interpreter.set_tensor(input_details[0]['index'],input_tensor.numpy())
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[2]['index'])

    return scores , boxes , classes



interpreter = tf.lite.Interpreter(model_path=r"C:\Users\91976\Desktop\programming\AI and Ml\projects\ANPR(automatic name plate recognition)\model.tflite")
interpreter.allocate_tensors()

label_id_offset = 1
test_images_np = []

image_path = r"C:\Users\91976\Desktop\programming\AI and Ml\projects\ANPR(automatic name plate recognition)\images\Greennumberplate-640x640(2)(1).jpg"
test_images_np.append(np.expand_dims(load_images_into_numpyarray(image_path),axis=0))

for i in range(len(test_images_np)):
    input_tensor = tf.convert_to_tensor(test_images_np[i],dtype=tf.float32)
    scores , boxes , classes = detect(interpreter,input_tensor)
    bbox = boxes[0][0]*640
    
    im = tf.image.crop_to_bounding_box(test_images_np[i] , int(bbox[0]) , int(bbox[1]) , int(bbox[2]-bbox[0]) , int(bbox[3] - bbox[1]))
    plt.axis('off')
    plt.imshow(np.squeeze(im)) 
    plt.savefig('output\output.png',dpi=300)   

