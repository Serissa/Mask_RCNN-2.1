"""
Mask R-CNN
Train on the toy pig dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 pig_train.py train --dataset=/path/to/pig/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 pig_train.py train --dataset=/path/to/pig/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 pig_train.py train --dataset=/path/to/pig/dataset --weights=imagenet

    # Apply color splash to an image
    python3 pig_train.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 pig_train.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from config import Config
import utils
import model as modellib
import cv2 as cv
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

############################################################
#  Configurations
############################################################


class PigConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pig"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PigDataset(utils.Dataset):

    def load_pig(self, dataset_dir, subset):
        """Load a subset of the pig dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("pig", 1, "pig")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "pig",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a pig dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pig":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pig":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def bounding_box(image, rois):
    for bbox in rois:
        cv.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 3, 4, 0)  # 用rectangle对图像进行画框
    return image


def center_point_splash(image, mask, output_path=None):
    draw = ImageDraw.Draw(image)
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        font = ImageFont.truetype('simsun.ttc', 40)
        shape = mask.shape
        dim = shape[2]
        count = 1
        for i in range(dim):
            mask1 = mask[:, :, i]
            mask1 = mask1 + 0
            gray = np.array(mask1, dtype='uint8')
            kernel = np.ones((20, 20), np.uint8)
            erosion = cv.erode(gray, kernel)  # 腐蚀

            im, contours, hierarchy = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            if len(contours):
                cnt = contours[0]
                M = cv.moments(cnt)
                # print(M)
                cx = int(M['m10']/(M['m00']+1))
                cy = int(M['m01']/(M['m00']+1))
                """
                if mask1[cy, cx] == 0:
                    num = 1
                    avg = 0
                    for j in range(shape[0]):
                        if mask1[cy, j] == 1:
                            avg += j
                            num += 1
                    cx = round(avg/num)
                """
                draw.text((cx, cy), str(count), fill=(255, 0, 0), font=font)
                count += 1
    image.save(output_path, 'jpeg')



def detect_and_color_splash(model, image_path=None, video_path=None, output_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        (filepath, tempfilename) = os.path.split(image_path)
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # bounding box
        splash = bounding_box(splash, r['rois'])
        # moments 质心
        img = Image.open(image_path)
        center_point_splash(img, r['masks'], output_path + 'result' + tempfilename)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Testing
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect pigs.')
    parser.add_argument('--weights',
                        default='D:/tensorflow/Mask_RCNN-2.1/samples/pig/models/pig20181022T1553/mask_rcnn_pig_0040.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs',
                        default='D:/tensorflow/Mask_RCNN-2.1/samples/pig/logs/',
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image',
                        default='D:/tensorflow/Mask_RCNN-2.1/samples/pig/logs/',
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video',required=False,
                        default='D:/tensorflow/Mask_RCNN-2.1/samples/pig/models/',
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)

    # Configurations
    class InferenceConfig(PigConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # weights file to load
    weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # evaluate
    for i in os.listdir(args.image):
        if i.endswith('.jpg'):
            image_path = os.path.join(args.image, i)
            detect_and_color_splash(model, image_path, video_path=args.video, output_path=args.logs)


