import time

import cv2

from sequence_utils import VOTSequence
from ncc_tracker_example import NCCTracker, NCCParams
from ms_tracker import MeanShiftTracker, MSParams
import glob

# set the path to directory where you have the sequences
dataset_path = '/home/aljaz/FAKS/nmrv/project02/vot14' 
sequence = 'bolt'  # choose the sequence you want to test

win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN

# visualization and setup parameters

def run_tracking(name, params = (15, 16, 0.01)):

    # create sequence object
    if params == None: 
        params = (15, 16, 0.01)

    sequence = VOTSequence(dataset_path, name)
    init_frame = 0
    n_failures = 0

    # create parameters and tracker objects
    parameters = MSParams(*params)
    tracker = MeanShiftTracker(parameters)

    time_all = 0

    # initialize visualization window
    # sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
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
        # sequence.show_image(img, video_delay)
        # time.sleep(0.05)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    cv2.destroyAllWindows()

    return n_failures, (sequence.length() / time_all)


videos = glob.glob(dataset_path + '/*')

videos.remove(dataset_path + '/list.txt')

all_failures = {}
fps = {}
kernel_sizes = [11,15, 21, 31, 51]
alphas = [0.01, 0.02, 0.05, 0.1]

for vid in videos:
    sequence = vid.split('/')[-1]
    # print(sequence)
    n_failures, f = run_tracking(sequence, (15, 16, 0.01))
    all_failures[sequence] = n_failures
    fps[sequence] = f

print(all_failures)
print(fps)



# combinations = [(k, 16, a) for k in kernel_sizes for a in alphas]

# results = {}

# for comb in combinations:
#     comb_str = str(comb[0]) + '-' + str(comb[2])
#     print(comb_str)
#     all_failures = []
#     fps = []
#     for vid in videos:
#         sequence = vid.split('/')[-1]
#         # print(sequence)
#         n_failures, f = run_tracking(sequence, comb)
#         all_failures.append(n_failures)
#         fps.append(f)
#     results[comb_str] = (sum(all_failures), sum(fps) / len(fps))
#     print('Total failures: %d' % sum(all_failures))
#     print('Average FPS: %.1f' % (sum(fps) / len(fps)))

# print(results)