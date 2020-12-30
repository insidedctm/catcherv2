import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from DlFallClassifier import DlFallClassifier
import argparse
from evaluate_common import evaluate_classifier
import sys

def parse_args():
    DESCRIPTION='''
    Evaluate Catcher classifiers. Runs the selected classifier on the test set and produces a plot of the confusion matrix
    '''
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('root_dir', default='.', help='Folder containing the nofalls/falls folders')
    args = parser.parse_args()
    return args

args = parse_args()  

with tf.Session() as sess:
    classifier = DlFallClassifier(sess, "checkpoint/NaiveTransferLearner_9.ckpt")
    evaluate_classifier(classifier, args.root_dir)
