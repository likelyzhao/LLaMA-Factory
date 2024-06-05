import pandas as pd

## load gt
gt = pd.read_csv('evaluation/ceval3/val/electricv3_val.csv')
gtans = gt['answer']

#load dt 
import json 
with open('saveeval/qwen7B-1.5b-chat-v2-shengnengdataset/results.json') as f:
    dt = json.load(f)

print(dt)

neglist = []
poslist = []

for  id, item in enumerate(gtans):
    #print(id, item)
    #print(dt['electricv2'][str(id)])
    if item != dt['electricv3'][str(id)]:
        neglist.append(id)

        print(gt['question'][id])
        print('right ans=' , item, 'ans=', gt[dt['electricv3'][str(id)]][id])
        #print('wrong')
    else:
        poslist.append(id)
        pass
        #print(gt['question'][id], "ans=", gt[item][id], "right")

negpd = gt.iloc[neglist]
pospd = gt.iloc[poslist]

total = 500 
posnum = total = len(neglist)
import random 
random.shuffle(poslist)
poslist = poslist[:posnum]

totallist = neglist + poslist
random.shuffle(totallist)
totaldf = gt.iloc[totallist]

totaldf.to_csv('electricv4_val_shuffle.csv')

print('neglist', neglist)