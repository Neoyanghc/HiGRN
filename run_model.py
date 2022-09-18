"""
训练并评估单一模型的脚本
"""

import argparse

from lib.pipeline import run_model
from lib.utils import general_arguments, str2bool, str2float
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"


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
                        default='state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='HiGRN', help='the name of model')
    parser.add_argument('--dataset', type=str,
                        default='OI', help='the name of dataset')
    # parser.add_argument('--model', type=str,
    #                     default='LSTM', help='the name of model')
    # parser.add_argument('--model', type=str,
    #                     default='HiGRU', help='the name of model')
    # parser.add_argument('--dataset', type=str,
    #                     default='Ocean_sensor_2022_2', help='the name of dataset')
    # parser.add_argument('--model', type=str,
    #                     default='GRU', help='the name of model')
    # parser.add_argument('--dataset', type=str,
    #                     default='METR_LA', help='the name of dataset')
    parser.add_argument('--config_file', type=str,
                        default=None, help='the file name of config file')
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--exp_id', type=str,
                        default=None, help='id of experiment')
    parser.add_argument('--input_window', type=int, default=12,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument('--output_window', type=int, default=12,
                        help='whether re-train model if the model is \
                             trained before')
    # parser.add_argument('--max_epoch', type=int, default=100,
    #                     help='max_epoch')
    parser.add_argument('--len_trend', type=int, default=1,
                        help='max_epoch')
    parser.add_argument('--len_closeness', type=int, default=3,
                        help='max_epoch')
    parser.add_argument('--len_period', type=int, default=1,
                        help='max_epoch')
    parser.add_argument('--add_day_in_week', type=str2bool,
                        default=True, help='id of experiment')
    parser.add_argument('--add_time_in_day', type=str2bool,
                        default=False, help='id of experiment')
    # parser.add_argument('--load_external', type=str2bool,
    #                     default=True, help='id of experiment')
    # parser.add_argument('--train_loss', type=str,
    #                     default='masked_mse', help='loss function')
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


