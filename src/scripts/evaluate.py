import argparse
from evaluation.eval_model import evaluate
from utils import add_universal_args

def add_args(parser: argparse.ArgumentParser) -> None:
    add_universal_args(parser=parser)
    parser.add_argument()

if __name__ == '__main__':
    # make an argument parser object
    parser = argparse.ArgumentParser()

    # add arguments
    add_args(parser)

    # parse the arguments, updating the config path and checkpoint path if applicable


    # run eval with updated paths. Inside this method is where the rest of the configs will be adjusted
    evaluate(config_path=config_path, checkpoint_path=checkpoint_path)

    print()




