import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Name of the new/trained model to train/sample from",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--loss", action="store_true", help="Whether to evaluate the model test loss")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--count", type=int, default=1, help="Structures to sample")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sampling_order_path",
        type=str,
        help="Sampling order file name",
    )
    parser.add_argument(
        "--noise_original",
        type=bool,
        default=False,
        help="noise original structure instead of completely random init"
    )

    args = parser.parse_args()
    args.log_path = os.path.join("exp", "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if not args.sample and not args.loss:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        logger = logging.getLogger()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

        
    elif args.sample:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)
        
        if not hasattr(args, 'sampling_order_path'):
            raise RuntimeError("No sampling order path is given")
        with open(os.path.join("sampling_orders", args.sampling_order_path), "r") as f:
            sampling_order = yaml.safe_load(f)

        new_config.sampling_order = []
        os.makedirs(os.path.join("exp", "cif_samples"), exist_ok=True)
        os.makedirs(os.path.join("exp", "cif_samples", args.doc), exist_ok=True)
        os.makedirs(os.path.join("exp", "cif_samples", args.doc, sampling_order['name']), exist_ok=True)
        overwrite = args.ni
        for order in sampling_order['orders']:
            if type(order['space_group']) is int:
                order['space_group'] = [order['space_group']]
            for space_group in order['space_group']:
                new_config.sampling_order.append(argparse.Namespace())
                composition_name = ["".join((element[0], str(element[1]))) for element in zip(order['composition'].keys(), order['composition'].values())]
                composition_name = "".join(composition_name)
                image_folder = composition_name + '_' + str(space_group)
                image_folder = os.path.join(
                    "exp", "cif_samples", args.doc, sampling_order['name'], image_folder
                )
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                else:
                    if not overwrite:
                        response = input(
                            f"Image folder already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True
                    if overwrite:
                        shutil.rmtree(image_folder)
                        os.makedirs(image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)
                if not order['only_final']:
                    for i in range(1, order['count']+1):
                        os.makedirs(os.path.join(image_folder, str(i)), exist_ok=True)
                    finals_dir = os.path.join(image_folder, "finals")
                else:
                    finals_dir = image_folder
                new_config.sampling_order[-1].image_folder = image_folder
                new_config.sampling_order[-1].composition = order['composition']
                new_config.sampling_order[-1].space_group = space_group
                new_config.sampling_order[-1].random_lattice = order['random_lattice']
                new_config.sampling_order[-1].random_positions = order['random_positions']
                new_config.sampling_order[-1].count = order['count']
                new_config.sampling_order[-1].T = order['T']
                new_config.sampling_order[-1].only_final = order['only_final']
                if 'template' in order:
                    new_config.sampling_order[-1].template = order['template']
                else:
                    if not order['random_lattice'] or not order['random_positions']:
                        raise RuntimeError("Invalid order given, not template structure for fixed lattice/positions")
                os.makedirs(finals_dir, exist_ok=True)
                new_config.sampling_order[-1].finals_dir = finals_dir

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) and not '_dict' in key:
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Exp instance id = {}".format(os.getpid()))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.loss:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
