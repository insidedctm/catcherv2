import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from DlFallClassifier import DlFallClassifier
from Conv3DFallClassifier import Conv3DFallClassifier
import argparse
from evaluate_common import evaluate_classifier
import sys

models = {
    'NaiveTransferLearner': DlFallClassifier,
    'Conv3D'              : Conv3DFallClassifier
}

def parse_args():
    DESCRIPTION='''
    Evaluate Catcher classifiers. Runs the selected classifier on the test set and produces a plot of the confusion matrix
    '''
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('root_dir', default='.', help='Folder containing the nofalls/falls folders')
    parser.add_argument('classifier', default='NaiveTransferLearner', help='NaiveTransferLearner|Conv3D')
    args = parser.parse_args()
    return args

args = parse_args()  

with tf.Session() as sess:
    if args.classifier == 'NaiveTransferLearner':
        classifier = DlFallClassifier(sess, "checkpoint/NaiveTransferLearner_9.ckpt")
    elif args.classifier == 'Conv3D':
        classifier = Conv3DFallClassifier(sess, "checkpoint/CatcherConv3D_Test3_3.ckpt")
    else:
        print(f"Unknown classifier, must be one of {models.keys()}")

    evaluate_classifier(classifier, args.root_dir)
