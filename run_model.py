import argparse

from lib.pipeline import run_model
from lib.utils import general_arguments, str2bool, str2float
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def add_other_args(parser):
    for arg in general_arguments:
        if general_arguments[arg] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=None)
        elif general_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=None)
        elif general_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='HiGRU', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='OI', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=False,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--exp_id', type=str,
                        default='66311', help='id of experiment')
    parser.add_argument('--input_window', type=int, default=12,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--output_window', type=int, default=12,
                        help='whether re-train model if the model is \
                             trained before')
    # 增加其他可选的参数yi
    add_other_args(parser)
    # 解析参数
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)


