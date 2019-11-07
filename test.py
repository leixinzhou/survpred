import argparse
from argparse import Namespace


from models import LSAIOWA96

from result_helpers import OneClassResultHelper
from result_helpers import VideoAnomalyDetectionResultHelper
from utils import set_random_seed


def test_iowa96():
    # type: () -> None
    """
    Performs One-class classification tests on iowa96
    """

    # Build dataset and model
    dataset = IOWA96(path='data/MNIST')
    model = LSAIOWA96(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()

    # Set up result helper and perform test
    helper = OneClassResultHelper(dataset, model, checkpoints_dir='checkpoints/mnist/', output_file='IOWA96.txt')
    helper.test_one_class_classification()





def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                             'Choose among `mnist`, `cifar10`, `ucsd-ped2`, `shanghaitech`', metavar='')

    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Lock seeds
    set_random_seed(30101990)

    # Run test
    if args.dataset == 'mnist':
        test_mnist()
    elif args.dataset == 'cifar10':
        test_cifar()
    elif args.dataset == 'ucsd-ped2':
        test_ucsdped2()
    elif args.dataset == 'shanghaitech':
        test_shanghaitech()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')


# Entry point
if __name__ == '__main__':
    main()
