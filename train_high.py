import torch
import tqdm
import os
import setproctitle
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
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset #, MovieLens100kDataset

from select_path import select_genotype
# from model import NetworkCTR as Network
from model import NetworkCTR_Sparse as Network

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
parser.add_argument('--genotype_path', type=str,
                    default='./', help='path to genotype')
parser.add_argument('--topk', type=int, default=8,
                    help='select topk feature interactions')
parser.add_argument('--topks', default='',
                    help='select topk feature interactions')
parser.add_argument('--high_order', action='store_true',
                    default=False, help='use high order')
args = parser.parse_args()
if args.high_order:
    interaction_type = 'high_order'
else:
    interaction_type = '2nd_order'

args.save = 'eval/eval-{}-{}-{}'.format(interaction_type,
                                   time.strftime("%Y%m%d-%H%M%S-%f"), args.dataset_name)
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
    if name == 'movielens100k':
        return MovieLens100kDataset(path)
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
    if args.high_order:
        genotype = select_genotype(os.path.join(
            args.genotype_path, 'genotype.npy'), high_order=True, topk=args.topk, topks=args.topks)  # load genotype saved
        logging.info('genotype_2nd = %s', genotype['2nd'][:, 0])
        logging.info('genotype_3rd = %s', genotype['3rd'][:, 0])
    else:
        genotype = select_genotype(os.path.join(
            args.genotype_path, 'genotype.npy'), high_order=False, topk=args.topk, topks=args.topks) # load genotype saved
        logging.info('genotype_ = %s', genotype[:, 0])
    args.num_fields = len(field_dims)
    args.genotype_3rd = genotype['3rd']
    args.genotype_2nd = genotype['2nd']
    if args.high_order:
        count3 = 0
        field31, field32, field33 = list(), list(), list()
        for i in range(args.num_fields - 2):
            for j in range(i + 1, args.num_fields-1):
                for k in range(j + 1, args.num_fields):
                    if args.genotype_3rd[count3] != 0:
                        field31.append(i), field32.append(j), field33.append(k)
                    count3 += 1
        fields_3order = [field31, field32, field33]
        count2 = 0
        field21, field22 = list(), list()
        for i in range(args.num_fields - 1):
            for j in range(i + 1, args.num_fields):
                if args.genotype_2nd[count2] != 0:
                    field21.append(i), field22.append(j)
                count2 += 1
        fields_2order = [field21, field22]
        fields_list = [fields_2order, fields_3order]

    model = Network(field_dims, args.embed_dim,
                    genotype, args.high_order, fields_list).to(device)
    criterion = torch.nn.BCELoss()
    criterion = criterion.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_epoch = 0
    best_auc = 0.0
    start_time = time.time()
    test_auc = test(test_queue, model, criterion, device)
    print('init', test_auc, time.time()-start_time)
    time.sleep(10)
    for epoch in range(args.epochs):
        lr = args.learning_rate
        logging.info('epoch %d lr %e', epoch, lr)

        # training from scratch
        train(train_queue, valid_queue, model,
              criterion, optimizer, lr, device)

        # validation
        auc = test(valid_queue, model, criterion, device)
        print('epoch:', epoch, 'validation: auc:', auc)
        logging.info('valid_auc %f', auc)

        # test
        test_auc = test(test_queue, model, criterion, device)
        if test_auc > best_auc:
            best_epoch = epoch
            best_auc = test_auc
            utils.save(model, os.path.join(args.save, 'weights.pt'))
        logging.info('epoch %d test auc %e', epoch, test_auc)
        print('best_epoch:{} best_test_auc:{}'.format(best_epoch, best_auc))


def train(train_queue, valid_queue, model, criterion, optimizer, lr, device):
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(train_queue, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
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
