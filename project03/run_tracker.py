import time

import cv2
import pandas as pd
from sequence_utils import VOTSequence
from tracker import CorTracker, CorrelationTracker
import glob
import numpy as np
import utils

# set the path to directory where you have the sequences
dataset_path = '/home/aljaz/FAKS/nmrv/vot14' 
sequence = 'bolt'  # choose the sequence you want to test

win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 1
font = cv2.FONT_HERSHEY_PLAIN

# visualization and setup parameters

def run_tracking(name, params = (2,0.01,0.7,1)):

    # create sequence object
    if params == None: 
        params = (2,0.01,0.7,1)

    sequence = VOTSequence(dataset_path, name)
    init_frame = 0
    n_failures = 0

    # create parameters and tracker objects
    # parameters = CorTracker(*params)
    tracker = CorrelationTracker(*params)

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
        # time.sleep(0.04)

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


videos = glob.glob(dataset_path + '/*')

videos.remove(dataset_path + '/list.txt')

all_failures = {}
fps = {}

failures = 0
kernel_sizes = [ 2, 10]
alphas = [ 0.1, 0.125, 0.2]
lambdas = [0.6, 0.7]
search_windows = [1, 1.1]

combinations = [(k, a, l, w) for k in kernel_sizes for a in alphas for l in lambdas for w in search_windows]

# results = {}

# for comb in combinations:
#     # comb_str = str(comb[0]) + '-' + str(comb[2])
#     # print(comb_str)
#     all_failures = []
#     fps = []
#     os = []
#     for vid in videos:
#         sequence = vid.split('/')[-1]
#         # print(sequence)
#         n_failures, f, o = run_tracking(sequence, comb)
#         all_failures.append(n_failures)
#         fps.append(f)
#         os.append(o)

#     results[str(comb)] = (comb[0], comb[1], comb[2], comb[3], sum(all_failures), int(sum(fps) / len(fps)), np.round(sum(os) / len(os), 3))
#     print('Total failures: %d' % sum(all_failures))
#     print('Average FPS: %.1f' % (sum(fps) / len(fps)))

# print(results)

# df = pd.DataFrame(results)
# df = df.transpose()
# df = df.rename(columns={0: 'Sigma', 1: 'Alpha', 2: 'Lambda', 3: 'Window', 4: 'Failures', 5: 'FPS', 6: 'Overlap'})
# df = df.rename_axis('Combination')
# df.to_csv('project03/combinations.csv')

kernel_sizes = [ 1, 2, 3, 4, 5, 10]
alphas = [ 0.01, 0.02, 0.05, 0.1, 0.125, 0.2, 0.3]
lambdas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
search_windows = [1, 1.1, 1.2, 1.5, 2]

#changing sigmas

results = {}

for sig in kernel_sizes:
    print(sig)
    all_failures = []
    fps = []
    os = []
    for vid in videos:
        sequence = vid.split('/')[-1]
        # print(sequence)
        n_failures, f, o = run_tracking(sequence, (sig, 0.1, 0.7, 1))
        all_failures.append(n_failures)
        fps.append(f)
        os.append(o)
    
    results[sig] = (sum(all_failures), int(sum(fps) / len(fps)), np.round(sum(os) / len(os), 3))
    print('Total failures: %d' % sum(all_failures))
    print('Average FPS: %.1f' % (sum(fps) / len(fps)))

print(results)
df = pd.DataFrame(results)
df = df.transpose()
df = df.rename(columns={0: 'Failures', 1: 'FPS', 2: 'Overlap'})
df = df.rename_axis('Sigma')
df.to_csv('project03/sigma_results.csv')

# # #changing alphas
results = {}

for alpha in alphas:
    print(alpha)
    all_failures = []
    fps = []
    os = []
    for vid in videos:
        sequence = vid.split('/')[-1]
        # print(sequence)
        n_failures, f, o = run_tracking(sequence, (2, alpha, 0.7, 1))
        all_failures.append(n_failures)
        fps.append(f)
        os.append(o)
    
    results[alpha] = (sum(all_failures), int((sum(fps) / len(fps))), np.round(sum(os) / len(os), 3))
    print('Total failures: %d' % sum(all_failures))
    print('Average FPS: %.1f' % int((sum(fps) / len(fps))))

print(results)
df = pd.DataFrame(results)
df = df.transpose()
df = df.rename(columns={0: 'Failures', 1: 'FPS', 2: 'Overlap'})
df = df.rename_axis('Alpha')
df.to_csv('project03/alpha_results.csv')

# chanigng window sizes

results = {}

for window in search_windows:
    print(window)
    all_failures = []
    fps = []
    os = []
    for vid in videos:
        sequence = vid.split('/')[-1]
        # print(sequence)
        n_failures, f, o = run_tracking(sequence, (2, 0.1, 0.7, window))
        all_failures.append(n_failures)
        fps.append(f)
        os.append(o)
    
    results[window] = (sum(all_failures), int((sum(fps) / len(fps))), np.round(sum(os) / len(os), 3))
    print('Total failures: %d' % sum(all_failures))
    print('Average FPS: %.1f' % int((sum(fps) / len(fps))))

print(results)
df = pd.DataFrame(results)
df = df.transpose()
df = df.rename(columns={0: 'Failures', 1: 'FPS', 2: 'Overlap'})
df = df.rename_axis('Window size multiplier')
df.to_csv('project03/window_results.csv')

# chanigng lambda

results = {}

for l in lambdas:
    print(l)
    all_failures = []
    fps = []
    os = []
    for vid in videos:
        sequence = vid.split('/')[-1]
        # print(sequence)
        n_failures, f, o = run_tracking(sequence, (2, 0.1, l, 1))
        all_failures.append(n_failures)
        fps.append(f)
        os.append(o)
    
    results[l] = (sum(all_failures), int((sum(fps) / len(fps))), np.round(sum(os) / len(os), 3))
    print('Total failures: %d' % sum(all_failures))
    print('Average FPS: %.1f' % int((sum(fps) / len(fps))))

print(results)
df = pd.DataFrame(results)
df = df.transpose()
df = df.rename(columns={0: 'Failures', 1: 'FPS', 2: 'Overlap'})
df = df.rename_axis('Lambda')
df.to_csv('project03/lambda_results.csv')

