import _init_path
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_epoch_interval', type=int, default=10, help='evaluate checkpoints every N epochs')
    parser.add_argument('--topk_best_ckpt', type=int, default=3, help='keep top-k checkpoints by eval metric')
    parser.add_argument('--best_metric', type=str, default='NDS', help='metric key used to rank checkpoints')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        epoch_int = int(float(epoch_id))
        if epoch_int < args.start_epoch:
            continue
        if args.eval_epoch_interval > 1 and epoch_int % args.eval_epoch_interval != 0:
            continue
        if float(epoch_id) not in evaluated_ckpt_list:
            return epoch_id, cur_ckpt
    return -1, None


def _best_record_path(eval_output_dir):
    return eval_output_dir / 'best_ckpt_record.json'


def _load_best_records(record_path):
    if not record_path.exists():
        return []
    try:
        with open(record_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save_best_records(record_path, records):
    with open(record_path, 'w') as f:
        json.dump(records, f, indent=2)


def _extract_metric_value(tb_dict, metric_name):
    if metric_name in tb_dict:
        return float(tb_dict[metric_name])

    # Fallback for case-insensitive key matches.
    lower_to_key = {str(k).lower(): k for k in tb_dict.keys()}
    metric_lower = metric_name.lower()
    if metric_lower in lower_to_key:
        return float(tb_dict[lower_to_key[metric_lower]])
    return None


def _safe_metric_token(metric_name):
    return re.sub(r'[^0-9a-zA-Z_.-]+', '_', str(metric_name))


def _update_topk_best_ckpts(eval_output_dir, ckpt_dir, cur_ckpt, cur_epoch_id,
                            metric_name, metric_value, topk, eval_epoch_interval, logger):
    topk = max(int(topk), 0)
    if topk == 0:
        return

    record_path = _best_record_path(eval_output_dir)
    records = _load_best_records(record_path)

    # Remove stale files from historical records.
    valid_records = []
    for item in records:
        ckpt_path = item.get('ckpt_path', '')
        if ckpt_path and os.path.exists(ckpt_path):
            valid_records.append(item)
    records = valid_records

    cur_item = {
        'epoch': int(float(cur_epoch_id)),
        'metric_name': metric_name,
        'metric_value': float(metric_value),
        'ckpt_path': str(cur_ckpt),
    }

    replaced = False
    for idx, item in enumerate(records):
        if item.get('ckpt_path', '') == str(cur_ckpt):
            records[idx] = cur_item
            replaced = True
            break
    if not replaced:
        records.append(cur_item)

    records.sort(key=lambda x: float(x.get('metric_value', float('-inf'))), reverse=True)
    keep_records = records[:topk]
    keep_paths = {item['ckpt_path'] for item in keep_records}

    # Keep only top-k candidate checkpoints in ckpt_dir.
    ckpt_pattern = os.path.join(str(ckpt_dir), 'checkpoint_epoch_*.pth')
    current_epoch = int(float(cur_epoch_id))
    for ckpt_path in glob.glob(ckpt_pattern):
        matched = re.findall(r'checkpoint_epoch_(.*)\.pth', os.path.basename(ckpt_path))
        if len(matched) == 0:
            continue
        try:
            epoch_int = int(float(matched[-1]))
        except ValueError:
            continue

        # Never remove future checkpoints that are not evaluated yet.
        if epoch_int > current_epoch:
            continue
        if eval_epoch_interval > 1 and epoch_int % eval_epoch_interval != 0:
            continue

        if ckpt_path not in keep_paths:
            try:
                os.remove(ckpt_path)
            except OSError as err:
                logger.warning('Failed to remove ckpt %s: %s', ckpt_path, err)

    # Mirror top-k files into a dedicated folder for quick access.
    best_dir = Path(ckpt_dir) / 'best_ckpt'
    best_dir.mkdir(parents=True, exist_ok=True)
    for file_path in best_dir.glob('*.pth'):
        try:
            file_path.unlink()
        except OSError:
            pass

    safe_metric_name = _safe_metric_token(metric_name)
    for rank, item in enumerate(keep_records, start=1):
        src = Path(item['ckpt_path'])
        if not src.exists():
            continue
        dst = best_dir / f'top{rank}_epoch_{item["epoch"]}_{safe_metric_name}_{item["metric_value"]:.4f}.pth'
        shutil.copy2(str(src), str(dst))

    _save_best_records(record_path, keep_records)
    logger.info('Top-%d best checkpoints by %s: %s', topk, metric_name,
                ', '.join([f"epoch {x['epoch']} ({x['metric_value']:.4f})" for x in keep_records]))


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            metric_value = _extract_metric_value(tb_dict, args.best_metric)
            if metric_value is None:
                logger.warning('Metric "%s" not found in eval result keys: %s. Skip top-k checkpoint update.',
                               args.best_metric, sorted(tb_dict.keys()))
            else:
                _update_topk_best_ckpts(
                    eval_output_dir=eval_output_dir,
                    ckpt_dir=ckpt_dir,
                    cur_ckpt=cur_ckpt,
                    cur_epoch_id=cur_epoch_id,
                    metric_name=args.best_metric,
                    metric_value=metric_value,
                    topk=args.topk_best_ckpt,
                    eval_epoch_interval=max(1, args.eval_epoch_interval),
                    logger=logger
                )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
