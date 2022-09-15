import numpy as np
import pandas as pd
from sklearn import preprocessing
from PIL import Image
import shutil, os

def rletobbox(rle, shape, image_filename, counter):
    '''
    rle: run length encoded image mask as string
    shape: (height, width) of an image in which RLE was produced
    image_filename: name of image whose bounding boxes we are going to extract
    
    Returns:
        (x0, y0, x1, y1) tuple describing the bounding box of RLE mask
    '''
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2)) # an array of start, length pairs
    a[:, 0] -= 1  # 'start' is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    
    if np.any(y1 > shape[0]):
        # got 'y' overrun, meaning that there are pixels in a mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
        
    x0 = a[:, 0] // shape[0]
    x1 = (a[:, 0] + a[:, 1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)

    bounding_boxes = np.array([x0, y0, x1, y1])
    normalized_array = preprocessing.normalize([bounding_boxes])
    
    x0 = normalized_array[0][0]
    y0 = normalized_array[0][1]
    x1 = normalized_array[0][2]
    y1 = normalized_array[0][3]
    
    if x1 > shape[1]:
        # just went out of image dimension
        raise ValueError("Invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (x1, shape[1]))
    
    if counter <= 8000:
        with open('./val_labels/'+image_filename+'.txt', 'w') as f:
            f.write(f'0 {x0} {y0} {x1} {y1}')
        #print("file created sucessfully")
    elif(counter > 8000 and counter <= 16000):
        with open('./test_labels/'+image_filename+'.txt', 'w') as f:
            f.write(f'0 {x0} {y0} {x1} {y1}') 
    elif(counter > 16000):
        with open('./train_labels/'+image_filename+'.txt', 'w') as f:
            f.write(f'0 {x0} {y0} {x1} {y1}')
        #print("file created sucessfully")



counter = 1

df = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
for _, image_filename, encoded_pixel in df[~df.EncodedPixels.isnull()].itertuples():
    image = Image.open('../input/airbus-ship-detection/train_v2/'+image_filename)
    image_bbox_filename = os.path.splitext(image_filename)[0]
    rletobbox(encoded_pixel, (image.height, image.width), image_bbox_filename, counter)
    
    if (counter <= 8000):
        shutil.copy('../input/airbus-ship-detection/train_v2/'+image_filename, './val_images')
        print("file copied scuesfully to val", counter)
        
    elif(counter > 8000 and counter <= 16000):
        shutil.copy('../input/airbus-ship-detection/train_v2/'+image_filename, './test_images')
        print("file copied scuesfully to test", counter)
    elif(counter > 16000):
        shutil.copy('../input/airbus-ship-detection/train_v2/'+image_filename, './train_images')
        print("file copied scuesfully to train", counter)
        
    counter += 1