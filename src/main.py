import os
from model import transfer_model

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='BLSD', choices=['BLSD', 'kaggle'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=800, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.set_defaults(test=False)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--model_name', type=str, default='default',
                        help='Current model name')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory name to save models')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
#    check_folder(args.checkpoint_dir)

    # --result_dir
 #   check_folder(args.result_dir)

    # --result_dir
  #  check_folder(args.log_dir)

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

        model = transfer_model(sess,
                epoch=args.epoch,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                log_dir=args.log_dir)

        # build graph
        model.build_model()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        if (args.predict):
            model.pred()
            print(" [*] Prediction finished!")
        else:
            model.train()
            print(" [*] Training finished!")

if __name__ == '__main__':
    main()
