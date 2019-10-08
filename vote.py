import glob
import pandas as pd 
import sys
import numpy as np


filenames = sorted(glob.glob(sys.argv[1] +'*.csv'))

def most_freq(lst):
    return max(set(lst),key=lst.count)




def main(filenames):

    csv_list=[]

    for filename in filenames:
        csv_list.append(pd.read_csv(filename,index_col=0))
    res=pd.concat(csv_list,axis=1,ignore_index=True)
    has_nan=res.isnull().any().any()
    print(has_nan)
    if has_nan:
        print('nan exist in your csv file ! check it')
        return 0
    labelx=[]
    for ind,row in res.iterrows():
        row=row.tolist()
        labelx.append(most_freq(row))

    file = open("vote.csv","w")
    file.write("image_name,label\n")
    for i,(filename,data) in enumerate(zip(res.index.tolist(),labelx)):
        file.write("%s,%d\n"%(filename,int(data)))

    file.close()

if __name__ == '__main__':
    main(filenames)
