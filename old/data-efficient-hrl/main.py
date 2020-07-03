import numpy as np
from hiro.train_hiro import run_hiro
from hiro.eval_hiro import eval_hiro
from parser import build_parser, get_parser_args

parser = build_parser()
args = get_parser_args(parser)

# Seed AFTER the hyperparameters are set
np.random.seed(args.seed)
if args.eval_only:
    eval_hiro(args)
elif args.hiro_only:
    run_hiro(args)
else:
    args.man_noise_sigma *= 5
    # run_nopt(args)
