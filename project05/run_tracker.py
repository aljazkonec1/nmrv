import argparse
import os
import cv2
import time
from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from matplotlib import pyplot as plt
from siamfc import TrackerSiamFC

def evaluate_tracker(dataset_path, network_path, results_dir, visualize=False):
    time_all = 0
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    tracker = TrackerSiamFC(net_path=network_path, longterm=True, n_samples=30)
    nr_frames = 0
    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)
        nr_frames = sequence.length()

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        t_ = time.time()
        tracker.init(img, gt_rect)
        time_all += time.time() - t_
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):

            img = cv2.imread(sequence.frame(i))
            t_ = time.time()
            prediction, score = tracker.update(img)
            time_all += time.time() - t_

            results.append(prediction)
            scores.append([score])

            if visualize:
                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)
                if i >= 800:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plt.show()
                cv2.imshow('win', img)
                # time.sleep(0.05)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)
        print("Average FPS: %.1f" % (nr_frames / time_all))
        save_results(results, bboxes_path)
        save_results(scores, scores_path)


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')

args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, visualize=False)
