"""
This is a class that is able to detect blinks using a webcam feed.
Requires:
dlib
opencv
numpy
"""
import dlib
import cv2
import numpy as np
import time
import threading
import queue
import logging
import serial

# TODO logging

class BlinkDetector(threading.Thread):

    def __init__(self, camfeed=0, serial_port='/dev/ttyUSB0', blink_threshold=0.75, live_display=True, 
        shape_detector_path='classifiers/shape_predictor_68_face_landmarks.dat',
        detector_upscaling=0, save_frames=False):
        """ camfeed [int] default=0 used to select VideoCapture source for cv2.VideoCapture(camfeed)
            blink_threshold [float] default=0.75 the threshold value for detecting closed eyes
            live_display [bool] default=True If True camera feed and blink counter will be displayed for every frame
            shape_detector_path for dlib.shape_predictor()
            detector_upscaling [int] default=0 upscaling factor for dlib face detector
            save_frames [bool] default=False Extreme debug mode, every frame is saved """
        super().__init__()
        # mandatory properties
        self.stop_req = False
        self.cam = cv2.VideoCapture(camfeed)
        self.blink_threshold = blink_threshold
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_detector_path)
        self.detector_upscaling = detector_upscaling
            # history is containing the eye status values (open/closed)
        self.history = queue.deque(maxlen=3) 
            # every detected blink is incrementing this
        self.blink_counter = 0

        # properties for calculating blinks per minute ratio
        self.face_present = threading.Event()
        self.blinks_per_10s = 0 # optimal value is around 3
        self.blink_timestamps = [] # stores all the timestamps when a blink was detected in the last 10s
        self.total_time_elapsed = 0 # time elapsen while face was present, used for long term analysis
        self.serial = serial.Serial(serial_port, baudrate=115200)

        # live display properties
        self.live_display = live_display
        self.q_live_display = queue.deque()
        
        # frame saver properties
        self.save_frames = save_frames
        if save_frames:
            self.save_stop_flag = False
            self.q_in = queue.deque()
            self.q_results = queue.deque()
            
    def stop_det(self):
        self.stop_req = True

    def save_images(self, q_in, dir_prefix='image_results/'):
        """ q_in is a queue with tuples like (image, eye_openness_ratio) """
        print('Image saver thread started!!!')
        index = 1
        while not self.stop_req:
            try:
                img, ratio, left, right = q_in.pop()
                img = cv2.putText(img, str(ratio), (0,355), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0)) 
                img = cv2.polylines(img, [np.array(left, np.int32)], True, (0,255,0),2)
                img = cv2.polylines(img, [np.array(right, np.int32)], True, (0,255,0),2)

                cv2.imwrite(dir_prefix+f'{index}.png', img)
                index += 1
            except IndexError:
                #empty queue
                time.sleep(2)
            if index % 200 == 0:
                print("-- 200 IMAGES SAVED! --")
 
    def display(self, q_in):
        """ Expects the image, the eyes as dlib points the eye closeness ratio
            and the blink counter and displays it in a dlib.image_window """ 
        win = dlib.image_window()
        while not self.stop_req:
            try:
                live_in = q_in.pop()
                win.clear_overlay()
                img = live_in['frame']                
                blink_counter = live_in['counter']
                blink_rate = live_in['blink_rate']
                img = cv2.putText(img, str(blink_counter), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                img = cv2.putText(img, str(self.blinks_per_10s), (60, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                win.set_image(img)
                ratio = live_in['ratio'] # not used for now
                try:
                    left_eye = live_in['left']
                    right_eye = live_in['right']
                except KeyError:
                    continue # no eyes to draw
                for p in left_eye:
                    win.add_overlay_circle(p,2)
                for p in right_eye:
                    win.add_overlay_circle(p,2)

            except IndexError:
                # empty queue nothing to display
                time.sleep(0.05)
        print("Display done")

    def alert(self, msg):
        self.serial.write(msg)


    def calculate_blink_rate(self):
        prev_blink_cnt = 0
        alert_status = 'on'
        while not self.stop_req:
            self.blinks_per_10s = len(self.blink_timestamps)

            if self.blinks_per_10s > 3 and alert_status == 'on':
                print('Alert ON')
                self.alert(b'off')
                alert_status = 'off'
            elif 3 >= self.blinks_per_10s and alert_status == 'off':
                print('Alert OFF')
                self.alert(b'on')
                alert_status = 'on'

            if self.blink_counter > prev_blink_cnt:
                print("timestamp added")
                self.blink_timestamps.append(time.time())
                prev_blink_cnt = self.blink_counter
            try:
                for t_blink in self.blink_timestamps:
                    if time.time() > t_blink + 10.0:
                        self.blink_timestamps.remove(t_blink)
                    else:
                        break
            except IndexError:
                time.sleep(0.1)




    @staticmethod
    def dlibpt_to_narray(pts):
        """ Converts a dlib.Point or dlib.Points objects and converts them to np.array(s) """
        ret = []
        for pt in list(pts):
            ret.append(np.array([pt.x, pt.y]))

        ret = ret if len(ret) > 1 else ret[0]
        return ret

    @staticmethod
    def eye_open_ratio(eye_dlib):
        """ Expects the 6 dlib.Point from the shape predictor defining one eye. 
        Uses these landmarks to calculate a value that is indicating eye openness:
            1   2  
        0           3   RATIO = [distance(1,5)+distance(2,4)] / distance(0,3)
            5   4
          """
        eye = BlinkDetector.dlibpt_to_narray(eye_dlib)
        wi = np.linalg.norm(eye[0]-eye[3])
        d1 = np.linalg.norm(eye[1]-eye[5])
        d2 = np.linalg.norm(eye[2]-eye[4])
        return (d1+d2)/wi

    


    def detect_blink(self,result):
        """  Stores the last 3 results each a boolean value describing whether the eyes were open or not
        Only returns true if the following pattern is recognized: False, False, True """
        self.history.append(result)
        try:
            if self.history[0] == False and self.history[1] == False and self.history[2] == True:
                return True
        except IndexError:
            # not enough items in history yet
            pass

        return False


    def run(self):
        """ Main loop here """
        self.display_thread = threading.Thread(target=self.display, args=[self.q_live_display]) if self.live_display else None
        if self.display_thread:
            self.display_thread.start()
        self.save_thread = threading.Thread(target=self.save_images, args=[self.q_in]) if self.save_frames else None
        if self.save_thread:
            self.save_thread.start()
        self.calc_rate_thread = threading.Thread(target=self.calculate_blink_rate, args=[])
        self.calc_rate_thread.start()

        while not self.stop_req:
            live_frame = {}  #dict {'frame': np.array, 'left'&'right':dlib.points, 'ratio':float, 'counter':int} 
            
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detects = self.face_detector(gray, self.detector_upscaling) # no upscaling
            if 1 > len(detects):
                self.face_present.clear() # signal that no face is detected, blink rate should not be counted
            else:
                self.face_present.set()
            if len(detects) > 1:
                print("Warning: more than one face detected!")

            # TODO: Check for multiple faces detected
            # TODO: Check for 0 faces detected - also exclude from statistics
            left_eye = None
            right_eye = None
            for k, d in enumerate(detects):
                shape = self.shape_predictor(gray, d)
                left_eye = shape.parts()[42:48]
                right_eye = shape.parts()[36:42]    
                
                
            if len(detects) == 1:
                # Calc eye openness ratio and check for a blink
                ratio = self.eye_open_ratio(right_eye) + self.eye_open_ratio(left_eye)
                eye_closed = False if ratio > self.blink_threshold else True
                if self.detect_blink(eye_closed):
                    self.blink_counter += 1
            else:
                ratio = -1.0

            # send data for the live display
            live_frame = {'frame':img, 'counter':self.blink_counter, 'ratio':ratio}
            if left_eye and right_eye:
                live_frame['left'] = left_eye
                live_frame['right'] = right_eye
            live_frame['blink_rate'] = len(self.blink_timestamps)
            self.q_live_display.append(live_frame)
        print("MAIN LOOP STOPPED")

        if self.save_thread:
            self.save_thread.join()
            print("SAVE THREAD STOPPED")
        if self.display_thread:
            self.display_thread.join()
            print("DISPLAY THREAD STOPPED")
        self.calc_rate_thread.join()
        # free resources


if __name__ == '__main__':
    det = BlinkDetector()
    det.start()
    print("Blink detection started...")
    while True:
        try:
            print(f"{det.blink_counter} blinks detected.", end='\r')
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

    print('Stopping...')
    det.stop_det()
    det.join()    

