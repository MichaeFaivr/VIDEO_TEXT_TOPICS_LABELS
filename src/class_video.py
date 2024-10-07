"""
Class for video attributes, methods for object detection from a predefined list of object types
voir si possible de faire de l'héritage de classe
"""

import numpy as np
import random

"""
purpose: embark all features and methods for video object detection

** Attributes: **
- frame_size is a tuple
- duration will be trimmed at 10 mins to avoid too heavy files
- type of video

** Methods: **
- landmarking
- start_obj_detection
- end_obj_detection
- nb of objects of this type  in the sequence
- main color of the object if only one of this type in the sequence

** Input: **
- a video with objects to be detected

** Output: **
- a Json file with the start/end sequence
- list of object types and list of boxes coordinates

** Example of sequence with multiple object types

"""

# voir s'il faut faire une classe mère et de l'héritage

## Random Images from Video -> Extraction/Labelling of predefined features: 
## Random time
class VideoClass():
    def __init__(self, frame_size, duration, video_type):
        self.frame_size = frame_size
        self.duration   = duration
        self.video_type = video_type

    def landmarking(self):
        pass

    def object_labelling(self):
        pass

    def random_frame(self):
        self.random_time = random.randrange(start=0, stop=self.duration)
        # comment extraire l'image associée à ce time dans la vidéo?
