#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2021

@author: HW
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())

with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for i, data in enumerate(reader):
        # if i >= 10000:
        #     break
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
            time_l = int(curdate[20:-1])
            time_f = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))*1000+time_l
            item = data['item_id'], int(time_f)
        elif opt.dataset == 'diginetica':
            item = data['itemId'], int(data['timeframe'])
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    l = list(sess_clicks)
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [[c[0] for c in sorted_clicks], [c[1] for c in sorted_clicks]]  #
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s][0]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s][0]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s][0]
    timeseq = sess_clicks[s][1]
    tmp_curseq, tmp_timeseq = [], []
    for i, item in enumerate(curseq):
        if iid_counts[item] >= 5:
            tmp_curseq.append(item)
            tmp_timeseq.append(timeseq[i])

    if len(tmp_curseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = tmp_curseq, tmp_timeseq


# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324

print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        train_time = sess_clicks[s][1]

        seq = sess_clicks[s][0]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs.append([outseq, train_time])
    # print(item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []

    for s, date in tes_sess:
        test_time = sess_clicks[s][1]
        seq = sess_clicks[s][0]
        get_time = []
        outseq = []
        for i, se in enumerate(seq):
            if se in item_dict:
                outseq += [item_dict[se]]
                get_time += [test_time[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        if len(outseq) == len(get_time):
            test_seqs.append([outseq, get_time])
        else:
            print('------------------------------error------------------------------')
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):

    out_seqs = []
    out_times = []
    out_dates = []
    labs = []
    ids = []

    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        iseqs_item = seq[0]
        iseqs_time = seq[1]
        iseqs_time = [iseqs_time[-1] -t for t in iseqs_time]
        for i in range(1, len(iseqs_item)):
            tar = iseqs_item[-i]
            labs += [tar]
            out_seqs += [iseqs_item[:-i]]
            out_times += [iseqs_time[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_times, out_dates, labs, ids
tr_seqs, tr_times, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_times, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
# for i in range(len(te_seqs)):
#     if len(te_seqs[i]) == len(te_times[i]):
#         print('te_seqs', te_seqs[i])
#         print('te_times', te_times[i])
#         continue
#     else:
#         print('-------------------------error-------------------')
def process_times(p_seqs, p_times):
    deal_times = []
    deal_items = []
    for i in range(len(p_seqs)):
        # print(p_seqs[i], p_times[i])
        if len(p_seqs[i]) == len(p_times[i]):
            tmp_list = []
            tmp_dict = dict()
            tis = p_times[i]
            if len(p_seqs[i]) == 1:
                deal_times.append(p_times[i])
                deal_items.append(p_seqs[i])

            else:
                for j, it in enumerate(p_seqs[i]):
                    if it not in tmp_list:
                        tmp_list.append(it)

                    if it in tmp_dict:
                        tmp_dict[it] += [tis[j]]
                    else:
                        tmp_dict[it] = [tis[j]]
                # print(tmp_dict)
                one_time = []

                for tp in tmp_list:
                    if len(tmp_dict[tp]) == 1:
                        one_time += tmp_dict[tp]
                    else:
                        mean_time = [int(np.mean(np.array(tmp_dict[tp])))]
                        tmp_dict[tp] = mean_time
                        one_time += mean_time
                # if len(one_time) == len(tmp_list):
                deal_times.append(one_time)
                deal_items.append(tmp_list)

        else:
            print('________________________________', 'error')
            print(p_seqs[i])
            print(p_times[i])
    for i in range(len(deal_items)):
        # print('p_seqs', p_seqs[i])
        # print('deal_items',deal_items[i])
        # print('deal_times',deal_times[i])
        t_input = list(np.unique(np.array(p_seqs[i])))
        if len(t_input) == len(deal_items[i]) and len(t_input) == len(deal_times[i]):
            continue
        else:
            print('出错')
            print(p_seqs[i])
            print(deal_items[i])
            print(deal_times[i])
    return deal_items, deal_times

tr_deal_items, tr_deal_times = process_times(tr_seqs, tr_times)
print('process_times train finish')
te_deal_items, te_deal_times = process_times(te_seqs, te_times)
print('process_times test finish')

for i in range(len(tr_seqs)):
    print(tr_seqs[i], tr_labs[i], tr_deal_times[i], tr_deal_items[i])
tra = (tr_seqs, tr_labs, tr_deal_times, tr_deal_items)
tes = (te_seqs, te_labs, te_deal_times, te_deal_items)


all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('./test_data/diginetica'):
        os.makedirs('./test_data/diginetica')
    pickle.dump(tra, open('./test_data/diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('./test_data/diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('./test_data/diginetica/all_train_seq.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('./test_data/yoochoose1_4'):
        os.makedirs('./test_data/yoochoose1_4')
    if not os.path.exists('./test_data/yoochoose1_64'):
        os.makedirs('./test_data/yoochoose1_64')
    pickle.dump(tes, open('./test_data/yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('./test_data/yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))
    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:], tr_deal_times[-split4:], tr_deal_items[-split4:]), \
                  (tr_seqs[-split64:], tr_labs[-split64:], tr_deal_times[-split64:], tr_deal_items[-split64:])

    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('./test_data/yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('./test_data/yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('./test_data/yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('./test_data/yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('./test_data/sample/train.txt', 'wb'))
    pickle.dump(tes, open('./test_data/sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('./test_data/sample/all_train_seq.txt', 'wb'))

print('Done.')
