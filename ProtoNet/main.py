import sys
sys.path.append('../src')

import os
from model import ProtoNet

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Few-shot Learning by Prototypical Network"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=20000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of batch')
    parser.set_defaults(test=False)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory name to save models')
    parser.add_argument('--predict', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load', type=int, default=0)


    parser.add_argument('--way', type=int, default=10, metavar='WAY',
                        help='The number of ways in training')
    parser.add_argument('--query', type=int, default=8, metavar='QUERY',
                        help='The number of query in training')
    parser.add_argument('--shot', type=int, default=32, metavar='SHOT',
                        help='The number of examples per class used in training')
    parser.add_argument('--test_way', type=int, default=10, metavar='TEST_WAY', 
                        help='The number of ways in testing')
    parser.add_argument('--test_query', type=int, default=8, metavar='TEST_QUERY',
                        help='The number of query in testing')
    parser.add_argument('--test_shot', type=int, default=32, metavar='TEST_SHOT',
                        help='The number of examples per class used in testing')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    return args

def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for transfer model

        print args.predict
        model = ProtoNet(sess,
                learning_rate=args.lr,
                epoch=args.epoch,
                checkpoint_dir=args.checkpoint_dir,
                log_dir=args.log_dir,                
                way=args.way,
                query=args.query,
                shot=args.shot,
                test_way=args.test_way,
                test_query=args.test_query,
                test_shot=args.test_shot,
                continue_learn=(args.load==1))

        # build graph
        model.build_model()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        if (args.predict == 1):
            model.pred()
            print(" [*] Prediction finished!")
        else:
            model.train()
            print(" [*] Training finished!")

if __name__ == '__main__':
    main()
