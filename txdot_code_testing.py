# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:14:24 2019

@author: daizh
"""
import pandas as pd
import sys
import numpy
from collections import OrderedDict
import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
from datetime import datetime, timedelta
#test how to find a small list from a bigger list
#list1=[[1,2,3,4],[5,6,7,8]]
#list2=[1,2]
#for i in list1:
#    print i
#    if list2[0] in i and list2[1] in i:
#        print 'yes'
#    else:
#        print 'n/a'
#+++++++++++++++++++========================        

#++++++++++++++++++++++++++++==================
##test how to use group by to group data points with same identity columns together
##df = pd.read_csv("C:\\Users\\daizh\\Desktop\Ren\\txdot\\uh_dcs_2018.csv")
#d = [{'city':'houston','year':2010,'pop':800},{'city':'dallas','year':2001,'pop':500,
#     'area':500},{'city':'austin','pop':100},{'city':'houston','year':2011,'pop':900},{'city':'dallas','year':2002,'pop':550}]
#df = pd.DataFrame(d)
#print df
#
##create lagged variable for population
#df['pop_lag'] = df.groupby(['city'])['pop'].shift(1)
#print df
#==================================++++++++++++++++++++++

#+++++++++++++++++++++++===========================
#using for loop to find information of selected road section
#df = pd.read_csv("C:\\Users\\daizh\\Desktop\Ren\\txdot\\uh_dcs_2018.csv")
#the first 50000 row of the original file was selected to test coding

#convert each row of the file to list

#
#list_of_all_rows=df.values.tolist()
#
##convert identity columns to list
#section_of_interest=['FM0256k', '0740', 1.0]
#for d in list_of_all_rows:
#    if section_of_interest[0] == d[4] and section_of_interest[1] == d[5] and section_of_interest[2] == d[6]:
#        print d
#        print (d[1],d[15])
#        print '+++++++++++++++++++++++++++++++++++++====================='

#++++++++++++++++++++++++++++++++=============================
#write a function that search through the entire dataset to extract information
# of certain road section   
#def search_road(x,y,z):
#    section_of_interest=[x,y,z]
#    result=[]
#    for d in list_of_all_rows:
#        if section_of_interest[0] == d[4] and section_of_interest[1] == d[5] and section_of_interest[2] == d[6]:
#            result.append(d)
#    return result
#    return result[1]
#    return result[15]
#x=search_road('BF1187CK', '0564', 0.5)
#    
#print x

#++++++++++++++++++++++++++++=========================
#delete data points that has a condition score which is better than the previous
# year
#
#df=df[(df['TX_CONDITION_SCORE']<df['condition_lag'])]
#print df
#sys.exit()
#
#df = df.sort_values(['ROUTE_NAME','TX_MRKR_ID_FROM','TX_MRKR_OFFSET_FROM',
#               'TX_MRKR_ID_TO','TX_MRKR_OFFSET_TO','EFF_YEAR'])
#
#print df
#
##
#for i in df['TX_CONDITION_SCORE']:
#    print i 

#++++++++++++++++===============================
df = pd.read_csv("C:\\Users\\daizh\\Desktop\Wilson\\txdot\\uh_wh.csv")
df = df.sort_values(['ROUTE_NAME','TX_MRKR_ID_FROM','TX_MRKR_OFFSET_FROM',
               'TX_MRKR_ID_TO','TX_MRKR_OFFSET_TO','EFF_YEAR'])
df = df.dropna(subset=['TX_FINAL_WORK_CODE'])
df = df.dropna(subset=['COMPLETION_DATE'])

df['date'] = pd.to_datetime(df['COMPLETION_DATE'])
df['year']= df['date'].dt.year
count_row = df['year'].shape[0]
for i in range(0,count_row ):
        print (i);
        if df['year'].iloc[i]>2019:
                df['year'].iloc[i] -= 100;
    
        
df.to_csv("C:\\Users\\daizh\\Desktop\Wilson\\txdot\\uh_wh_date.csv")







