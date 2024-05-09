import time

import cv2
import pandas as pd
from sequence_utils import VOTSequence
from particle_filter import ParticleTracker
import glob
import numpy as np
# import utils

# set the path to directory where you have the sequences
dataset_path = '/home/aljaz/FAKS/nmrv/vot14' 
sequence = 'bolt'  # choose the sequence you want to test

win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 1
font = cv2.FONT_HERSHEY_PLAIN

# visualization and setup parameters

def run_tracking(name):


    sequence = VOTSequence(dataset_path, name)
    init_frame = 0
    n_failures = 0

    # create parameters and tracker objects
    # parameters = CorTracker(*params)
    tracker = ParticleTracker(n_particles=10, alpha = 0.05)

    time_all = 0

    # initialize visualization window
    # sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    overlaps = 0
    nr_frames = sequence.length()

    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        # if show_gt:
        #     sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        # sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        # sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        # sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        # sequence.show_image(img, video_delay, frame_idx)
        # time.sleep(0.1)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
            overlaps += o
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            # print('Tracking failure at frame %d' % frame_idx)
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1
        
   
        # if frame_idx >= 20:
        #     break

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    cv2.destroyAllWindows()

    return n_failures, (sequence.length() / time_all), overlaps / nr_frames


n_failures, fps, overlap = run_tracking(sequence)
