import torch
import tqdm
import os
import sys
import time
import glob
import utils
import numpy as np
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset

from model_search import Network
from architect import Architect

parser = argparse.ArgumentParser("AutoDSN.")
parser.add_argument('--dataset_name', default='criteo')
parser.add_argument(
    '--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
parser.add_argument('--model_name', default='nasfm')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--epochs', type=int, default=15,
                    help='num of training epochs')
parser.add_argument('--weight_decay', type=float,
                    default=1e-6, help='weight decay')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--save_dir', type=str, default='chkpt',
                    help='path to save the model')
parser.add_argument('--embed_dim', type=int, default=16,
                    help='dimension of embedding')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--genotype_path', type=str,
                    default='./', help='path to save genotype')
args = parser.parse_args()

args.save = 'search-{}-{}-{}'.format(args.save,
                                     time.strftime("%Y%m%d-%H%M%S"), args.dataset_name)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s", args)
    device = torch.device(args.device)
    dataset = get_dataset(args.dataset_name, args.dataset_path)
    field_dims = dataset.field_dims
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_queue = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8)
    valid_queue = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=8)
    test_queue = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=8)

    model = Network(field_dims, args.embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    criterion = criterion.cuda()

    arch_params = list(map(id, model.arch_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_params,
                           model.parameters())

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(
        params=weight_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    architect = Architect(model, criterion, args)

    for epoch in range(args.epochs):
        lr = args.learning_rate
        logging.info('epoch %d lr %e', epoch, lr)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype[:, 0])

        # training
        train(train_queue, valid_queue, model, architect,
              criterion, optimizer, lr, device)

        # validation
        auc = test(valid_queue, model, criterion, device)
        print('epoch:', epoch, 'validation: auc:', auc)
        logging.info('valid_auc %f', auc)
        utils.save(model, os.path.join(args.save, 'weights.pt'))

    np.save(os.path.join(args.genotype_path, 'genotype.npy'),
            genotype)  # save genotype


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device):
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(train_queue, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)

        # get a random minibatch from the search queue with replacement
        fields_search, target_search = next(iter(valid_queue))
        fields_search = fields_search.to(device)
        target_search = target_search.to(device)

        architect.step(fields, target.float(), fields_search,
                       target_search.float(), lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(fields)
        loss = criterion(logits, target.float())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % args.report_freq == 0:
            print('    - loss:', total_loss / args.report_freq)
            total_loss = 0


def test(valid_queue, model, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(valid_queue, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


if __name__ == '__main__':
    main()
