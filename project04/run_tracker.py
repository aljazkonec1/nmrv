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
sequence = 'skating'  # choose the sequence you want to test

win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 1
font = cv2.FONT_HERSHEY_PLAIN

# visualization and setup parameters

def run_tracking(name, comb = (0.1, 16, 50, 2)):

    alpha, n_bins, sigma, q = comb

    sequence = VOTSequence(dataset_path, name)
    init_frame = 0
    n_failures = 0

    # create parameters and tracker objects
    # parameters = CorTracker(*params)
    tracker = ParticleTracker(n_particles=100, alpha=alpha, n_bins=n_bins, sigma=sigma, q_qucient=q)

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
        particles = np.array([[0,0]])
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
            
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox, particles = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        #scatterplot particles onto image
        for x, y in zip(particles[:, 0], particles[:, 1]):
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        sequence.show_image(img, video_delay, frame_idx)

        time.sleep(0.1)

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


# n_failures, fps, overlap = run_tracking(sequence)


videos = glob.glob(dataset_path + '/*')

videos.remove(dataset_path + '/list.txt')
# videos = ["skating", "tunnel", "woman", "motocross", "hand2", "fish1", "fish2", "david", "diving"]
all_failures = {}
fps = {}

failures = 0
# alphas = [ 0.05, 0.075, 0.1, 0.125]
# n_bins = [16]
# sigmas = [25, 50, 100]
# q_qucients = [1, 2, 3, 4, 5]

alphas = [0.05]
n_bins = [16]
sigmas = [25]
q_qucients = [8]

combinations = [[a, b, c, d] for a in alphas for b in n_bins for c in sigmas for d in q_qucients]
print(combinations)
results = {}

for comb in combinations:
    comb_str = str(comb[0]) + '-' + str(comb[1])+'-' + str(comb[2]) + '-' + str(comb[3])
    print(comb_str)
    all_failures = []
    fps = []
    os = []
    for vid in videos:
        sequence = vid.split('/')[-1]
        print(sequence)
        n_failures, f, o = run_tracking(sequence, comb)
        all_failures.append(n_failures)
        fps.append(f)
        os.append(o)

    results[str(comb)] = (comb[0], comb[1], comb[2], comb[3], sum(all_failures), int(sum(fps) / len(fps)), np.round(sum(os) / len(os), 3))
    print('Total failures: %d' % sum(all_failures))
    print('Average FPS: %.1f' % (sum(fps) / len(fps)))

df = pd.DataFrame(results)
df = df.transpose()
df = df.rename(columns={0: 'Alpha', 1: 'Bins', 2: 'sigma', 3: 'Quocient', 4: 'Failures', 5: 'FPS', 6: 'Overlap'})
df = df.rename_axis('Combination')
print(df)
# df.to_csv('combinations.csv')
