import numpy as np
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
#from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model

# Load Keras model
model = load_model('full_CNN_model.h5')

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def increase_value(img, value=30,sval=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(hsv)
    
    print("H__________________________________________")
    print(h)
    print("S__________________________________________")
    print(s)
    print("V__________________________________________")
    print(v)
    print(type(v))


    lim = 255 - value
    v[v > lim] = 255 # for values > lim set to 255 for overflow 
    v[v <= lim] += value
    
    slim=255-sval
    s[s > slim] = 255 # for values > lim set to 255 for overflow 
    s[s <= slim] += sval

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    image=image[500:1050,600:1800]
    image = increase_value(image, value=70)
    #print(image.shape)
    

    # Get image ready for feeding into model
    small_img = resize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = resize(lane_drawn, (image.shape[0], image.shape[1], 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

lanes = Lanes()

# Where to save the output video
vid_output = 'proj_reg_vid.mp4'

# Location of the input video
clip1 = VideoFileClip("project_new.mp4")

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)
