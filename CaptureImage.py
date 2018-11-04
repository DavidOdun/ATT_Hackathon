#!/usr/bin/env python3
###############################################################################
#
#  Project:  MLontheEdgeCodeStory
#  File:     Edge.py
#  Authors:  David (Seun) Odun-Ayo
#  Emails:   dodunayo@nd.edu | dave_seun@yahoo.com
#  Description: This is the main script in the Code Story Project. This script
#  sets up a connection to Azure IoTHub, Twilio, and Azure Blob Storage. As well,
#  the script is contanstly taking pictures of the world around and making ML model
#  predictions on the images taken. Images are saved on Azure Blob Storage and 
#  there is a constant check for new updates the ML model.
#  Requires: Python 3.5.3
#
###############################################################################
import cv2
import ellmanager as emanager
import io
import json
import logging
import model
import numpy as numpy
import os
import picamera
import random
import shutil
import subprocess
import sys
import termios
import time
import tty
import zipfile
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.split(os.path.realpath(__file__))[0]

class PiImageDetection():
    
    def __init__(self):
        # Intialize Azure Blob Container Properties
        self.picture_container_name = 'edgeimages'
        self.video_container_name = 'edgevideos'
        self.model_container_name = 'edgemodels'
        self.json_container_name = 'edgejson'

        # Intialize Azure IoTHub Config Properties
        self.capture_rate = 30.0
        self.prediction_threshold = 0.2
        self.camera_res_len = 256
        self.camera_res_wid = 256
        self.video_capture_length = 30
        self.video_preroll = 5
        self.capture_video = False
        self.send_twilio_sms = True
      
    def run_shell(self, cmd):
        """
        Used for running shell commands
        """
        output = subprocess.check_output(cmd.split(' '))
        logging.debug('Running shell command')
        if output is None:
            logging.debug('Error Running Shell Command. Exiting...')
            sys.exit(1)
        return str(output.rstrip().decode())

    def save_video(self, input_path, output_path, rename_path):
        # Convert each indivudul .h264 to mp4 
        mp4_box = "MP4Box -fps {0} -quiet -add {1} {2}".format(self.capture_rate, input_path, output_path)
        # Call the OS to perform the compressing
        self.run_shell(mp4_box)
        # Remove the .h264 file to save space on the RPI
        os.remove(input_path)
        # Rename for Better Convention Understanding
        os.rename(output_path, rename_path)
        logging.debug('Video Saved')

    def model_predict(self, image):
        # Open the required categories.txt file used for identify the labels for recognized images
        with open("categories.txt", "r") as cat_file:
            categories = cat_file.read().splitlines()

        # Determine the right size and shape that the model wants
        input_shape = model.get_default_input_shape()
        # Get the given image ready for use with the model
        input_data = emanager.prepare_image_for_model(image, input_shape.columns, input_shape.rows)
        # Make the Model Prediction
        prediction = model.predict(input_data)
        # Return the max top 2 predictions if they exits
        top_2 = emanager.get_top_n(prediction, 2)
        # Make a decision on what to do based on the return prediction values
        if (len(top_2) < 1):
            # If nothing, return nothing
            return None, None
        else:
            # Something was recongized, give the name based on the categories file and give the value
            word = categories[top_2[0][0]]
            predict_value = top_2[0][1]
            return word, predict_value

    def write_json_to_file(self, video_time, word_prediction, predicition_value, video_name, json_path):
        # Template for description of the image and video taken
        json_message = {
            'Description': {
                'sysTime':               str(datetime.now().isoformat()) + 'Z',
                'videoStartTime':        str(video_time.isoformat()) + 'Z',
                'prediction(s)':         word_prediction,
                'predictionConfidence':  str(predicition_value),
                'videoName':             video_name
            }
        }
        logging.debug("Rewriting Json to File")
        # Write Json Message to file
        with open(json_path, 'w') as json_file:
            json.dump(json_message, json_file)


    def get_video(self):
        # Define Variables
        capture_time = self.video_capture_length
        preroll = self.video_preroll
        capture_video = self.capture_video
        camera_res = (self.camera_res_len, self.camera_res_wid)
        image = numpy.empty((camera_res[1], camera_res[0],3), dtype=numpy.uint8)
        capture_counter = 0

        # Set up Circular Buffer Settings
        video_stream = picamera.PiCameraCircularIO(camera_device, seconds=capture_time)
        camera_device.start_preview()
        camera_device.start_recording(video_stream, format='h264')
        my_now = datetime.now()

        while True:
            if capture_counter < 8:
                # Set up a waiting time difference
                my_later = datetime.now()
                difference = my_later-my_now
                seconds_past = difference.seconds
                camera_device.wait_recording(1)

                logging.debug('Analyzing Surroundings')
                if seconds_past > preroll+1:
                    # Take Picture for the Model
                    camera_device.capture(image,'bgr', resize=camera_res, use_video_port=True)
                    camera_device.wait_recording(1)
                    
                    # Take Picture for Azure
                    image_name = "image-{0}.jpg".format(my_later.strftime("%Y%m%d%H%M%S"))
                    image_path = "{0}/{1}".format(SCRIPT_DIR, image_name)
                    camera_device.capture(image_path)
                    camera_device.wait_recording(1)

                    #print("Prediction Threshold: {}".format(self.prediction_threshold))
                    # Make Prediction with the first picture
                    logging.debug('Prediction Captured')
                    word, predict_value = self.model_predict(image)
                    
                    # Give time here for model predictions
                    camera_device.wait_recording(3)
                    logging.debug('Prediction Returned')
                    my_now = datetime.now()
                    
                    if word is None:
                        logging.debug('No Event Registered')
                        capture_video = False
                        # Format specifically for the Good Folder
                        bad_image_folder = "{0}/badimages".format(self.picture_container_name)
                        # Send Picture to the Bad Images Folder on Azure that can be used to retrain
                    elif word is not None and predict_value < self.prediction_threshold:
                        logging.debug('Prediction Value Too Low')
                        capture_video = False
                        # Format Specifically for the Good FOlder
                        bad_image_folder = "{0}/badimages".format(self.picture_container_name)
                        # Send Picture to the Bad Images Folder on Azure that can be used to retrain
                        camera_device.wait_recording(2)
                    else:
                        # See what we got back from the model
                        logging.debug('Event Registered')
                        capture_video=True
                        print('Prediction(s): {}'.format(word))
                        # Format specifically for the Good Folder
                        good_image_folder = "{0}/goodimages".format(self.picture_container_name)
                        # Send the Picture to the Good Images Folder on Azure
                        camera_device.wait_recording(2)
                        # Once it is uploaded, delete the image
                        os.remove(image_path)
                        break
                    # If we don;t break by finidng the right predicition stay in the loop
                    seconds_past = 0
                    capture_counter = capture_counter + 1
                    # Delete the image from the OS folder to save space
                    os.remove(image_path)
            else:
                camera_device.stop_recording()
                return

        ## Create diretory to save the video that we get if we are told to capture video
        start_time = my_later
        base_dir = SCRIPT_DIR
        video_dir = "myvideos"
        video_dir_path ="{0}/{1}".format(base_dir, video_dir)

        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)

        video_start_time = start_time - timedelta(seconds=preroll)

        ## We will have two seperate files, one for before and after the event had been triggered
        #Before:
        before_event =         "video-{0}-{1}.h264".format("before", video_start_time.strftime("%Y%m%d%H%M%S"))
        before_event_path =    "{0}/{1}/{2}".format(base_dir, video_dir, before_event)
        before_mp4 =           before_event.replace('.h264', '.mp4')
        before_mp4_path =      "{0}/{1}/{2}".format(base_dir, video_dir, before_mp4)
        before_path_temp =      "{0}.tmp".format(before_mp4_path)

        # After:
        after_event =         "video-{0}-{1}.h264".format("after", video_start_time.strftime("%Y%m%d%H%M%S"))
        after_event_path =    "{0}/{1}/{2}".format(base_dir, video_dir, after_event)
        after_mp4 =           after_event.replace('.h264', '.mp4')
        after_mp4_path =      "{0}/{1}/{2}".format(base_dir, video_dir, after_mp4)
        after_path_temp =     "{0}.tmp".format(after_mp4_path)

        # Full combined video path
        full_path =           "video-{0}-{1}.mp4".format("full", video_start_time.strftime("%Y%m%d%H%M%S"))
        full_video_path =     "{0}/{1}/{2}".format(base_dir, video_dir, full_path)

        # Create a json file to a reference the given event
        json_file_name = "video-description-{0}.json".format(video_start_time.strftime("%Y%m%d%H%M%S"))
        json_file_path = "{0}/{1}/{2}".format(base_dir,video_dir, json_file_name)

        if capture_video == True:
            # Save the video to a file path specified
            camera_device.split_recording(after_event_path)
            video_stream.copy_to(before_event_path, seconds=preroll)
            camera_device.wait_recording(preroll+5)
                    
            # Convert to MP4 format for viewing
            self.save_video(before_event_path, before_path_temp, before_mp4_path)
            self.save_video(after_event_path, after_path_temp, after_mp4_path)

            # Upload Before Videos to Azure Blob Storage
            before_video_folder = "{0}/{1}".format(self.video_container_name, 'beforevideo')

            # Upload After Videos to Azure Blob Storage
            after_video_folder = "{0}/{1}".format(self.video_container_name, 'aftervideo')

            # Combine the two mp4 videos into one and save it
            full_video = "MP4Box -cat {0} -cat {1} -new {2}".format(before_mp4_path, after_mp4_path, full_video_path)
            self.run_shell(full_video)
            logging.debug('Combining Full Video')
            
            # Upload Video to Azure Blob Storage
            full_video_folder = "{0}/{1}".format(self.video_container_name, 'fullvideo')

            # Create json and fill it with information
            self.write_json_to_file(video_start_time, word, predict_value, full_path, json_file_path)
        
            # End Things
            shutil.rmtree(video_dir_path)
            camera_device.stop_recording()

    def main(self):
        # Define Globals
        global camera_device

        # Intialize Log Properties
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') 

        # Intilize Camera properties 
        camera_device = picamera.PiCamera()
        camera_device.resolution = (1280, 720)
        camera_device.framerate = self.capture_rate
        
        if camera_device is None:
            logging.debug("No Camera Device Found.")
            sys.exit(1)
                
        # Intialize the updates Json File
        update_json_path = "{0}/{1}.json".format(SCRIPT_DIR, 'updatehistory')
        
        while True:
            logging.debug('Starting Edge.py')
            # Began running and stay running the entire project.
            self.get_video()

    

if __name__ == '__main__':
    mydetector = PiImageDetection()
    mydetector.main()
