import argparse
from Models.SSL import *
# from Models.test import *
from datetime import timedelta
import time
import copy
import sys

desc = 'new'
print(100*'*')
print(desc)
print(100*'*')

modelName = 'resnet50' 
dataset = 'ISIC2018' #'ISIC2019' 

GPU_ids = 0
pretrain = True
type = 'SSL'  #'FS' #

w_class = 1
w_proto = 0
w_pl = 0
w_consist = 0

use_cw = True
projectionDim = 512 
thr = 0.9
ema_pl = 0.8
ema_fea = 0.8
ema_model = 0.995
torch.cuda.set_device(GPU_ids)


bs_l, bs_u = 16, 48
lr = 1e-4 

parser = argparse.ArgumentParser()
parser.add_argument("--bs_l", type=int, default=bs_l)
parser.add_argument("--bs_u", type=int, default=bs_u)
parser.add_argument("--pretrain", type=int, default=pretrain)
parser.add_argument("--optimizer", default='ADAM') #'LARS''SGD'
parser.add_argument("--addValDataWithTrain", default=False)
parser.add_argument("--modelName", default=modelName)
parser.add_argument("--dataset", default=dataset)
parser.add_argument("--w_pl", default=w_pl)
parser.add_argument("--w_proto", default=w_proto)
parser.add_argument("--w_class", default=w_class)
parser.add_argument("--w_consist", default=w_consist)
parser.add_argument("--projectionDim", default=projectionDim)
parser.add_argument("--useEMA", default=True)
parser.add_argument("--gpuid", type=int, default=GPU_ids)
parser.add_argument("--lr", default=lr) #1e-3 for SGD, 1e-4 for ADAM
parser.add_argument("--type", default=type)
parser.add_argument("--temperature", default=1)
parser.add_argument("--thr", default=thr)
parser.add_argument("--use_cw", default=use_cw)
parser.add_argument("--ema_pl", default=ema_pl)
parser.add_argument("--ema_fea", default=ema_fea)
parser.add_argument("--ema_model", default=ema_model)
opt = parser.parse_args()

opt.output_dir='Results'
os.makedirs('args.output_dir', exist_ok=True)
dirname = os.path.join('/storage/scratch1/phd23-pg-skin-classification/multi_level_stablization/',opt.output_dir)
if not os.path.exists(dirname):
    os.makedirs(dirname)


fn = os.path.join(dirname, '_bs' + str(bs_l) + '_ds' + str(dataset)  + 'Results_01.txt')
print(fn)
sys.stdout = open(fn, 'w')
print(fn)


def printVals(desc, paraArr, mvArr, stdvArr):
    paraArr = np.stack(paraArr, axis=0)
    mvArr = np.stack(mvArr, axis=0)
    stdvArr = np.stack(stdvArr, axis=0)
    print(150*'-')
    print(desc)
    for r in range(len(paraArr)):
        para = paraArr[r]
        for e in para:
            print(e, ':', end='')
        print(end='|\t')
        mv = mvArr[r]
        stdv = stdvArr[r]
        for i in range(len(mv)):
            if i % len(desc) == 0:
                print(' |\t', end='')
            print(mv[i], '+', stdv[i], end = ' :')
        print()
    print(150 * '-')


def testOne(opt):
    print(10 * '-')
    re_all = []
    for i in range(3):
        st = time.time()
        print(30*'-')
        print(i)
        print(30 * '-')
        opt.itr = i
        opt.seed = 500 * (i + 1)
        print(opt)
        fm = SSL(opt)
        re, desc = fm.iterate()
        re_all.append(re)
        print(opt)
        elapsed_time = (time.time() - st)
        print("execution time: " + str(timedelta(seconds=elapsed_time)))
    re_all = np.stack(re_all, axis=0)
    re_all_mean = np.round(np.mean(re_all, axis=0), 2)
    re_all_std = np.round(np.std(re_all, axis=0), 2)
    return re_all_mean, re_all_std, desc


para = []
re_mvArr = []
re_stdArr = []
detail = ''

def execute(opt):
    print(desc)
    print([type, p, opt.thr, opt.w_class, opt.w_pl, opt.ema_pl, opt.w_consist, opt.ema_fea, opt.temperature])
    para.append([type, p, opt.thr, opt.w_class, opt.w_pl, opt.ema_pl, opt.w_consist, opt.ema_fea, opt.temperature])
    re_m, re_std, desc_re = testOne(opt)
    re_mvArr.append(re_m)
    re_stdArr.append(re_std)
    printVals(desc_re, para, re_mvArr, re_stdArr)

############################################################################################################   
# # for p in [0.1, 0.05, 0.02, 0.2]:
for p in [0.05]:
    opt.pL = p
    for w_pl in [0,1]:
        opt.w_pl = w_pl
        if opt.w_pl  > 0 :
            for empl in [0,0.8,0.9]:
                opt.ema_pl = empl
                if(opt.ema_pl > 0):
                    for w_consist in [0,100]:
                        opt.w_consist = w_consist
                        if opt.w_consist  > 0 :
                            for ema_fea in [0,0.8]:
                                    opt.ema_fea = ema_fea
                                    execute(opt)

                        else:
                            opt.ema_fea = 0
                            execute(opt)
                else:
                    opt.ema_fea = 0
                    opt.w_consist = 0
                    opt.ema_pl= 0
                    execute(opt)

        else:
            opt.ema_pl= 0
            for w_consist in [0,100]:
                opt.w_consist = w_consist
                if opt.w_consist  > 0 :
                    for ema_fea in [0,0.8]:
                            opt.ema_fea = ema_fea
                            # print(opt.w_consist)
                            # print(opt.ema_fea)
                            execute(opt)
                else:
                    opt.ema_fea = 0
                    execute(opt)

############################################################################################################   
# For different ema_pl 
# for p in [0.1, 0.05, 0.02, 0.2]:
# for p in [0.02]:
#     opt.pL = p
#     for w_pl in [1]:
#         opt.w_pl = w_pl
#         if opt.w_pl  > 0 :
#             # for empl in [0,0.5,0.6,0.7,0.8,0.9,0.95]:
#             for empl in [0.9]:
#                 opt.ema_pl = empl
#                 if(opt.ema_pl > 0):
#                     # opt.ema_pl = 0.8
#                     for w_consist in [0,10,100,1000]:
#                         opt.w_consist = w_consist
#                         if opt.w_consist  > 0 :
#                             for ema_fea in [0.8]:
#                                     opt.ema_fea = ema_fea
#                                     execute(opt)

#                         else:
#                             opt.ema_fea = 0
#                             # print(opt.w_consist)
#                             # print(opt.ema_fea)
#                             execute(opt)
#                 else:
#                     opt.ema_fea = 0
#                     opt.w_consist = 0
#                     opt.ema_pl= 0
#                     execute(opt)

#         else:
#             opt.ema_pl= 0
#             for w_consist in [0,100]:
#                 opt.w_consist = w_consist
#                 if opt.w_consist  > 0 :
#                     for ema_fea in [0,0.8]:
#                             opt.ema_fea = ema_fea
#                             # print(opt.w_consist)
#                             # print(opt.ema_fea)
#                             execute(opt)
#                 else:
#                     opt.ema_fea = 0
#                     # print(opt.w_consist)
#                     # print(opt.ema_fea)
#                     execute(opt)

############################################################################################################
# #For different thresholds
# # for p in [0.1, 0.05, 0.02, 0.2]:
# for p in [0.02]:
#     opt.pL = p
#     for w_pl in [1]:
#         opt.w_pl = w_pl
#         if opt.w_pl  > 0 :
#             for empl in [0.8,0.9]:
#                 opt.ema_pl = empl
#                 if(opt.ema_pl > 0):
#                     # opt.ema_pl = 0.8
#                     for th in [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]:
#                         opt.thr = th
#                         opt.w_consist = 0
#                         opt.ema_fea = 0
#                         execute(opt)
#                 else:
#                     opt.ema_fea = 0
#                     opt.w_consist = 0
#                     opt.ema_pl= 0
#                     execute(opt)

#         else:
#             opt.ema_pl= 0
#             for w_consist in [0,100]:
#                 opt.w_consist = w_consist
#                 if opt.w_consist  > 0 :
#                     for ema_fea in [0,0.8]:
#                             opt.ema_fea = ema_fea
#                             # print(opt.w_consist)
#                             # print(opt.ema_fea)
#                             execute(opt)
#                 else:
#                     opt.ema_fea = 0
#                     # print(opt.w_consist)
#                     # print(opt.ema_fea)
#                     execute(opt)

############################################################################################################