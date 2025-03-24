'''
**YOLOv8: Object Detection and Segmentation**

**1 Introduction to YOLOv8**
YOLOv8 (You Only Look Once, version 8) is an advanced real-time object detection and segmentation model. It is designed for high-speed and high-accuracy image analysis using deep learning techniques.

**Key Features:**
- **Real-time Object Detection**: Processes images/videos in a single pass.
- **Instance Segmentation**: Generates pixel-perfect masks around detected objects.
- **Bounding Box Prediction**: Provides coordinates for detected objects.
- **High Efficiency**: Optimized for both CPU and GPU inference.

**2 How YOLOv8 Works**

**1. Feature Extraction with CNNs**
YOLOv8 employs a **Convolutional Neural Network (CNN)** to extract features from images:
- **Edge Detection**: Identifies object boundaries.
- **Texture Recognition**: Distinguishes objects based on surface patterns.
- **Shape Detection**: Recognizes object contours and structure.

**2. Grid-based Object Detection**
- The input image is divided into a **grid of cells**.
- Each cell predicts **bounding boxes, class scores, and segmentation masks**.
- A non-max suppression (NMS) algorithm removes redundant detections.

**3. Instance Segmentation (YOLOv8-Seg)**
- In addition to bounding boxes, YOLOv8-Seg generates **segmentation masks**.
- The **binary mask** classifies each pixel as either part of an object (interested) (**1 - white**) or background (not interested) (**0 - black**).
- Techniques like **contour detection, edge refinement, and up-sampling** improve mask quality.

**3 Input and Output Formats**

**1. Input Format**
YOLOv8 processes image frames in:
- **NumPy Arrays** (`H x W x C` for RGB images)
- **Torch Tensors** (`B x C x H x W` in PyTorch format) (batch size, channels, height, width)
- **Common image formats**: JPG, PNG, MP4 (for video processing)
and the color format is RGB (for frames), only the masks are (binary 0 or 1 which is converted to grayscale (0 or 255))

**2. Output Format**
| Output | Format | Description |
|--------|--------|-------------|
| **Bounding Box** | `[x, y, w, h]` | Object's location |
| **Class Label** | `int` | Predicted class index |
| **Confidence Score** | `float` | Detection probability (0-1) |
| **Segmentation Mask** | `NumPy/Tensor` | Binary mask of detected object |

Example Output for YOLOv8-Seg:
[
    {
        "bbox": [120, 50, 300, 400],  # x, y, width, height
        "class": "cat",
        "confidence": 0.95,
        "mask": [Segmentation mask (1 = object, 0 = background)
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
    }
]

**4 Mask Generation Techniques**
YOLOv8-Seg refines segmentation masks using:
- **Binary Masking**: Classifies each pixel as object (`1`) or background (`0`).
- **Edge Detection**: Finds boundaries using Sobel or Canny filters.
- **Contour Detection**: Identifies object shapes.
- **Up-sampling (Deconvolution)**: Improves mask resolution.
'''

import cv2
from ultralytics import YOLO
import numpy as np
import torch

class CustomSegmentationWithYolo():
    def __init__(self, erode_size=5, erode_intensity=2):
        '''
        yolov8 -> model
        m -> Medium size variant
        seg -> segmentation variant (not just object detection)
        '''
        self.model = YOLO('yolov8m-seg.pt') # Initializing yolov8 model for segmentation (.pt file automatically loads both the model defination and learned weights when passed to class YOLO())
        self.erode_size = erode_size
        self.erode_intensity = erode_intensity
        self.background_image = cv2.imread("./static/default-office-animated.png") #numpy format
         
    def generate_mask_from_result(self,results):
        # results refers to the predictions made by the yolo model for each frame. Results contains bounding boxes (x,y,w,h), class label, confidence scores and segmentation masks (pixel perfect)
        for result in results:
            # mask and boxes are both in numpy format (can also be in torch tensor format)
            if result.masks:
                # get array results
                masks = result.masks.data #contains binary segmentation mask
                boxes = result.boxes.data #contains bounding box coordinates (x,y,w,h), confidence score, and class indices
                
                # extract classes
                clss = boxes[:, 5] #since boxes contains [x,y,w,h,confidence,class], to get the class we use boxes[:,5] (all rows and the 5th column)

                '''
                YOLO is trained on COCO dataset in which people have a class label of 0.
                hence when yolo processes our frames the location of results where the class label is 0 is selected .
                Since yolo has already processed our frames it also has the masks.
                We get the mask where people exist by giving the selected locations (where classes==0(people's indices)) to the masks[]
                '''
                
                # get indices of results where class is 0 (people in COCO)
                people_indices = torch.where(clss == 0) # we torch.where() as classes is a pytorch tensor (hence we dont use np) and torch.where() returns indices (np.where returns values) 
                
                # use these indices to extract the relevant masks
                people_masks = masks[people_indices]

                if len(people_masks) == 0: # if there are no valid mask for people in that frame we return none (people may exist but mask is not valid (sometimes detections may fail))
                        return None
                
                '''
                we combine all the people_mask into one single mask dim=0 ==> merge using logical OR (+) . Eg:
                Person 1:      Person 2:      Person 3:      Merged Mask:   
                000011000      000000011      000110000      000011011
                000111100  +   000001111   +  000111000  =   000111111
                001111110      000011111      001111000      001111111

                uint8 (unsigned 8 bit int) ==> standard format for opencv and numpy.
                YOLO masks are binary (0 or 1).Images use grayscale values from 0 to 255. Multiplying by 255 converts the mask to black(background) & white(people) for visualization
                '''
                
                # scale for visualizing results
                people_mask = torch.any(people_masks, dim=0).to(torch.uint8) * 255

                # Erode the mask to bring boundaries inward
                '''
                Erosion scans the image with a small kernel (a tiny matrix) and replaces the pixel value with 
                the minimum value under the kernel. So if even one pixel under the kernel is black, the pixel 
                being processed will turn black. This has the effect of eating away the white areas, shrinking the mask.
                '''
                # iterations - number of times to apply the erosion
                # kernel - This is a small matrix (typically a square) that slides over the image.
                # Create a structuring element (kernel) for erosion  
                # A square matrix of size (erode_size x erode_size) filled with ones
                kernel = np.ones((self.erode_size, self.erode_size), np.uint8)
                
                '''
                Convert the mask from a Torch tensor to a NumPy array (since OpenCV expects NumPy arrays)
                Perform erosion using the kernel, with a specified number of iterations
                Each erosion iteration removes boundary pixels of objects in the mask 
                iterations=self.erode_intensity determines how many times erosion is applied
                purpose - Reduces Noise, Refines Segmentation (Removes rough edges), Separates Connected Objects (Helps break apart overlapping objects)
                '''
                
                eroded_mask = cv2.erode(people_mask.cpu().numpy(), kernel, iterations=self.erode_intensity)

                # save to file
                return eroded_mask
            else:
                 return None # return none if no people exist in that frame


    def apply_blur_with_mask(self,frame, mask, blur_strength=21):
        # blur_str is the kernel size for the blur operation
        blur_strength=(blur_strength, blur_strength)
        blurred_frame = cv2.GaussianBlur(frame, blur_strength, 0) #blurring using gaussian blur, 0 ==> standard deviation (auto-calculated)
        # note: in blurred frame the entire frame including the object and background are blurred.
        # while return the frame we send the object back from the original frame and the background from the blurred frame

        '''
        yolos mask output is not always exactly 1.  eg: 0.9999
        OpenCV expects binary masks to be integer-based (np.uint8), not floating-point.
        The operation (mask > 0).astype(np.uint8) ensures:
        mask > 0 converts values to True (1) or False (0)(but this is in boolean format).
        .astype(np.uint8) converts True → 1 and False → 0 in unsigned 8 bit integer format required by opencv
        '''
        mask = (mask > 0).astype(np.uint8)
        
        #The segmentation mask is originally single-channel (grayscale).
        #To apply it to an RGB image, we expand it into 3 channels (H x W x 3) each pixel is 0 ==> [0,0,0] or 1 ==> [1,1,1].
        mask_3d = cv2.merge([mask, mask, mask])
        
        # check individually mask_3d == 1 ie from [1,1,1] == 1 we get [true, true, true]
        # if equal then return the "frame" (original frame - person (interested area))
        # if not equal ie, mask_3d == 0, then return the "blurred_frame" (blurred frame - background (not interested area))
        result_frame = np.where(mask_3d == 1, frame, blurred_frame) #we use np.where() for element wise selection between 2 arrays (frame and blurred frame) and also opencv loads images as np arrays (mask_3d,frame,blurred_frame)
        
        '''
        we can use np.where(mask[:,:,newaxis] == 255, frame, blurred_frame) instead of: 
        mask = (mask > 0).astype(np.uint8),
        mask_3d = cv2.merge([mask, mask, mask]),
        result_frame = np.where(mask_3d == 1, frame, blurred_frame) 
        np.where(mask[:,:,newaxis] == 255, frame, blurred_frame) is more memory efficient as it does not duplicate the mask whereas merge() duplicates the mask.
        we are using merge() method to try diferent ways
        '''

        return result_frame
    
    def apply_black_background(self, frame, mask):
        # Create a black background
        black_background = np.zeros_like(frame) #creates a black image (all 0) of the same shape as that of the frame

        #mask[:,:,newaxis] ==> all rows, all cols, and a newdim ==> (h,w,1), np.where scaled it to (h,w,3) for rgb format automatically.
        #since mask is 0 or 255(white => object), if it is object we send the original frame else we send the black bg 
        result_frame = np.where(mask[:, :, np.newaxis] == 255, frame, black_background)
        return result_frame

    def apply_custom_background(self, frame, mask):
        # Load the background image
        background_image = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0])) # resizes the bg image to frame.shape[1] ==> width of frame (no of cols) and frame.shape[0] ==> height of frame (no of rows)
        # Apply the mask
        result_frame = np.where(mask[:, :, np.newaxis] == 255, frame, background_image)
        return result_frame