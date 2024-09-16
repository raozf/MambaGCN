
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from MambaGCN.data import ModelNet40
from MambaGCN.model import  DGCNN
import numpy as np
from torch.utils.data import DataLoader
from MambaGCN.util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
import MinkowskiEngine as ME
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True,  drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False,  drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    model = DGCNN(args).to(device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*10, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                                                     T_0=250,
                                                                     T_mult=2,
                                                                     eta_min=1e-5,
                                                                     last_epoch=-1)
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):                         #data (B,N,3)
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            logits, logits2, logits3, logits4, logits5 = model(data)
            loss1 = criterion(logits, label)  
            loss2 = criterion(logits2, label)
            loss3 = criterion(logits3, label)
            loss4 = criterion(logits4, label)
            loss5 = criterion(logits5, label)
            loss = loss1 + 0.5*loss2 + 0.25*loss3 + 0.25*loss4 + 0.5*loss5
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        scheduler.step()
        if epoch%10==0:
            io.cprint(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_pred2 = []
        test_pred3 = []
        test_true = []
        test_pred4 = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits, logits2, logits3, logits4, logits5 = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            preds2 = logits3.max(dim=1)[1]
            preds3 = logits4.max(dim=1)[1]
            preds4 = logits5.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            test_pred2.append(preds2.detach().cpu().numpy())
            test_pred3.append(preds3.detach().cpu().numpy())
            test_pred4.append(preds4.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_pred2 = np.concatenate(test_pred2)
        test_pred3 = np.concatenate(test_pred3)
        test_pred4 = np.concatenate(test_pred4)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_acc2 = metrics.accuracy_score(test_true, test_pred2)
        avg_per_class_acc2 = metrics.balanced_accuracy_score(test_true, test_pred2)
        test_acc3 = metrics.accuracy_score(test_true, test_pred3)
        avg_per_class_acc3 = metrics.balanced_accuracy_score(test_true, test_pred3)
        test_acc4 = metrics.accuracy_score(test_true, test_pred4)
        avg_per_class_acc4 = metrics.balanced_accuracy_score(test_true, test_pred4)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f,test acc2: %.6f, test avg acc2: %.6f,test acc3: %.6f, test avg acc3: %.6f,test acc4: %.6f, test avg acc4: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              test_acc2,
                                                                              avg_per_class_acc2,
                                                                              test_acc3,
                                                                              avg_per_class_acc3,
                                                                              test_acc4,
                                                                              avg_per_class_acc4)
                                                                              
        io.cprint(outstr)
        if test_acc4 >= best_test_acc:
            best_test_acc = test_acc4
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    test_pred2 = []
    test_pred3 = []
    test_true = []
    test_pred4 = []
    for data, label in test_loader:   
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits, logits2, logits3, logits4 = model(data)
        preds = logits.max(dim=1)[1]
        preds2 = logits2.max(dim=1)[1]
        preds3 = logits3.max(dim=1)[1]
        preds4 = logits4.max(dim=1)[1]
        count += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        test_pred2.append(preds2.detach().cpu().numpy())
        test_pred3.append(preds3.detach().cpu().numpy())
        test_pred4.append(preds4.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_pred2 = np.concatenate(test_pred2)
    test_pred3 = np.concatenate(test_pred3)
    test_pred4 = np.concatenate(test_pred4)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    test_acc2 = metrics.accuracy_score(test_true, test_pred2)
    avg_per_class_acc2 = metrics.balanced_accuracy_score(test_true, test_pred2)
    test_acc3 = metrics.accuracy_score(test_true, test_pred3)
    avg_per_class_acc3 = metrics.balanced_accuracy_score(test_true, test_pred3)
    test_acc4 = metrics.accuracy_score(test_true, test_pred4)
    avg_per_class_acc4 = metrics.balanced_accuracy_score(test_true, test_pred4)
    outstr = 'test acc: %.6f, test avg acc: %.6f,test acc2: %.6f, test avg acc2: %.6f,test acc3: %.6f, test avg acc3: %.6f,test acc4: %.6f, test avg acc4: %.6f' % (
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              test_acc2,
                                                                              avg_per_class_acc2,
                                                                              test_acc3,
                                                                              avg_per_class_acc3,
                                                                              test_acc4,
                                                                              avg_per_class_acc4)
                                                                              
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=700, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
