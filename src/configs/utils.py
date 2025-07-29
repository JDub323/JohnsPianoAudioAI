import yaml
import argparse

# the main helper function to be used by scripts in the scripts file. 
# the get_args_func should be a function with no params which returns an 
# argparse.ArgumentParser object
# TODO: put variable types on this func
def get_configs(get_args_func):
    # collect all command line arguments from the function passed. This allows for custom command line
    # arguments which are only exposed for evaluate, predict, train, or build corpus individually
    args = get_args_func()

    # update the config path if that is what the user specified, and load the configs (it is a namespace)
    if args.config_path != None:
        configs = load_configs(path=args.config_path)
    else:
        configs = load_configs(path='../configs/default.yaml')

    # change the configs based on the command line args
    apply_args_to_configs(configs=configs, args=args)

    # return the configs to be used by the rest of the program
    return configs

# changes the configs based on the command line args
def apply_args_to_configs(configs, args: argparse.Namespace) -> None:
    for k, v in vars(args).items():
        if v is None:
            continue
        section, key = k.split('.')
        configs[section][key] = v

# returns the configs after loading them
def load_configs(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# adds the arguments to the argparser which may be used by predict.py, train.py, or evaluate.py 
# this is to avoid repetition
# as a rule, I will make the args parsed without stated defaults. There can be limited choices. 
# this is because I use "None" as a default for an argument that was not stated. This makes it 
# impossible to pass "None" as an argument, which I hope will not bite me in the future
def add_universal_args(parser: argparse.ArgumentParser) -> None:
    # Config Path
    parser.add_argument('--config_path', type=str)

    # Dataset
    parser.add_argument('--dataset.name', type=str)
    parser.add_argument('--dataset.data_root', type=str)
    parser.add_argument('--dataset.sample_rate', type=int)
    parser.add_argument('--dataset.segment_length_seconds', type=int)
    parser.add_argument('--dataset.input_representation', type=str, choices=["cqt", "mel", "spectrogram"])
    parser.add_argument('--dataset.n_bins', type=int)
    parser.add_argument('--dataset.hop_length', type=int)
    parser.add_argument('--dataset.normalize_audio', type=bool)
    parser.add_argument('--dataset.add_noise', type=bool)

    # Model
    parser.add_argument('--model.type', type=str)
    parser.add_argument('--model.input_channels', type=int)
    parser.add_argument('--model.output_channels', type=int)
    parser.add_argument('--model.hidden_size', type=int)
    parser.add_argument('--model.depth', type=int)
    parser.add_argument('--model.dropout', type=float)
    parser.add_argument('--model.use_batch_norm', type=bool)

    # Training
    parser.add_argument('--training.batch_size', type=int)
    parser.add_argument('--training.epochs', type=int)
    parser.add_argument('--training.learning_rate', type=float)
    parser.add_argument('--training.optimizer', type=str)
    parser.add_argument('--training.scheduler', type=str)
    parser.add_argument('--training.weight_decay', type=float)
    parser.add_argument('--training.grad_clip', type=float)
    parser.add_argument('--training.loss_function', type=str)

    # Evaluation
    parser.add_argument('--evaluation.metrics', nargs='+')
    parser.add_argument('--evaluation.threshold', type=float)
    parser.add_argument('--evaluation.use_mir_eval', type=bool)

    # Logging
    parser.add_argument('--logging.log_dir', type=str)
    parser.add_argument('--logging.checkpoint_dir', type=str)
    parser.add_argument('--logging.save_best_only', type=bool)
    parser.add_argument('--logging.save_every_n_epochs', type=int)

    # Experiment
    parser.add_argument('--experiment.name', type=str)
    parser.add_argument('--experiment.seed', type=int)
    parser.add_argument('--experiment.device', type=str)
    parser.add_argument('--experiment.num_workers', type=int)
    parser.add_argument('--experiment.debug', type=bool)
    parser.add_argument('--experiment.use_wandb', type=bool)

    # Inference
    parser.add_argument('--inference.threshold', type=float)
    parser.add_argument('--inference.apply_smoothing', type=bool)
    parser.add_argument('--inference.smoothing_window_size', type=int)
    parser.add_argument('--inference.output_format', type=str)


