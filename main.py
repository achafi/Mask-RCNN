
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
%matplotlib inline
import matplotlib.pyplot as plt
from keras.preprocessing import image
from matplotlib.patches import Rectangle
from mrcnn.visualize import display_instances 

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80


rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)


# load photograph
input_image = "...."
img = image.load_img(input_image)
plt.imshow(img)


# # make prediction
img = image.img_to_array(img)
results = rcnn.detect([img], verbose=0)


# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = plt.imread(filename)
     # plot the image
     plt.imshow(data)
     # get the context for drawing boxes
     ax = plt.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     plt.show()
     

draw_image_with_boxes(input_image, results[0]['rois'])


# 
# define classes that the coco model knowns about
class_names = ['bicycle', 'car', 'motorcycle', 'airplane','bus', 'train', 'truck', 'boat']

# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])