# All the code related to streaming like opening camera, closing camera, streaming, listing webcams etc are written here for modularity
import cv2
import pyvirtualcam
import numpy as np
import cv2
import torch
import pyvirtualcam
from engine import CustomSegmentationWithYolo


class Streaming(CustomSegmentationWithYolo): # streaming class inherits customersegmentationwithyolo class
    def __init__(self, in_source=None, out_source=None, fps=None, blur_strength=None, cam_fps=15, background="none"):
        super().__init__(erode_size=5, erode_intensity=2) #initialize the default constructor of super class (here customersegmentationwithyolo)
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.running = False
        self.original_fps = cam_fps
        self.background = background
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Device selected/found for inference : {self.device}")

    def update_streaming_config(self, in_source=None, out_source=None, fps=None, blur_strength=None, background="none"):
        # Updates the configuration settings for video streaming.
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background

    def update_cam_fps(self, fps):
        self.original_fps = fps

    def update_running_status(self, running_status=False):
        self.running = running_status
    
    def stream_video(self):
        # Starts video streaming using OpenCV and sends the processed frames to a virtual camera.
        self.running = True # Flag to control the streaming loop
        print(f"Retreiving feed from source({self.input_source}), FPS : {self.fps}, Blur Strength : {self.blur_strength}")
        cap = cv2.VideoCapture(int(self.input_source)) # Open the webcam using the class VideoCapture
        frame_idx = 0 # Keeps track of the total number of frames processed

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try : 
            # Attempt to get the original (default) FPS of the webcam
            self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        except Exception as e: 
            # If FPS is unavailable, print a warning message
            print(f"Webcam({self.input_source}), live fps not available. Setting fps to {self.original_fps}. Exception info : {e}")

        # Calculate frame interval for FPS adjustment
        # Frame interval determines how often we process and send a frame
        if self.fps:
            if self.fps>self.original_fps:
                self.fps = self.original_fps # Limit FPS to original value if requested FPS is too high (device wont be able to support it)
                frame_interval = int(self.original_fps / self.fps) # Calculate interval based on FPS (helps us decide how many frames to skip to match required fps)
            else:
                frame_interval = int(self.original_fps / self.fps) # Normal FPS adjustment
        else:
            frame_interval=1 # Default interval (every frame is processed)

        # Create a virtual camera with the specified width, height, and FPS using the Camera class
        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
            print(f"Virtual camera running at {width}x{height} {self.fps}fps")

            while self.running and cap.isOpened(): # Loop while the camera is open and streaming is active
                ret, frame = cap.read() # ret - bool value indicating whether frame was captured or not, frame - read a frame from the input source.
                # Note - frame is a numpy array of bgr values in Opencv

                if not ret:
                    break # Exit the loop if no valid frame is received

                # Process frame only at the calculated interval (to match the user required fps eg original fps - 30, user given fps - 15 , frame_interval = 30/15=2, process every 2nd frame)
                if frame_idx % frame_interval == 0:
                    '''
                    predict() method of yolo takes:
                    source - The video frame to be processed.
                    save(bool) - If True, saves the output image/video with detected objects.
                    save_txt(bool) - If True, saves the detected objects' details (class, bounding box, confidence) as a .txt file.
                    stream(bool) - f True, returns results as a generator (streamed), allowing real-time processing.
                    retina_masks(bool) - By default, YOLOv8 uses low-resolution masks to save computation time.When retina_masks=True, it increases the mask resolution relative to the input image size (typically 4x finer resolution).
                        This results in smoother edges and better-defined object boundaries.
                    verbose(bool) - If False, suppresses extra logging messages.
                    device - Specifies whether to run inference on CPU or GPU ('cuda') or MAC ('mps')
                    
                    the output stored in results is:
                    result.boxes: bounding boxes, class labels, confidence score
                    result.masks: segmentation masks
                    result.prob: class probabilities (used for classification)
                    result.org_img: original input image/frame
                    result.names: dictionary containing class index to name mapping from COCO dataset
                    '''
                    results = self.model.predict(source=frame, save=False, save_txt=False, stream=True, retina_masks=True, verbose=False, device=self.device)
                    # used for segmentation (like a person in a webcam feed) or applying effects (like background blur or replacement).
                    mask = self.generate_mask_from_result(results)
                    

                    if mask is not None:
                        if self.background == "blur":
                            result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
                        elif self.background == "none":
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            result_frame = self.apply_custom_background(frame, mask)
                            

                # Send the processed frame to the virtual camera
                # OpenCV uses BGR color format, while virtual cameras use RGB, so conversion is required
                cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                # Ensures that the next frame is processed at the correct time according to FPS
                cam.sleep_until_next_frame()

        cap.release() # Release the camera resource when streaming stops

    def list_available_devices(self):
        devices = []
        for i in range(5): #Most devices have 5 or less than 5 webcams hence range is 5
            cap = cv2.VideoCapture(i) # VideoCapture is a class in cv2 that takes the index(no of webcams in a device). Note: it can also take file with extension .mp4 etc.
            if cap.isOpened(): # isOpened function of the VideoCapture class checks if the webcam at that index is working or not. If it exists and is working it returns True and if there is no webcam at that index then it returns false.
                # From this we can know the actual number of webcams present on the system
                devices.append({"id": i, "name": f"Camera {i}"}) #creates a list of dictionaries and is sent to the frontend
                cap.release()
        return devices


    if __name__ == "__main__":
        print(list_available_devices())
