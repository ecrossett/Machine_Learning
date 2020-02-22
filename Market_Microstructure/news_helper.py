import pandas as pd
import numpy as np
import datetime as dt
import dask.dataframe as dsk

filename = 'newsfeed_0.csv'
starttime = '10:30:00'
endtime = '15:00:00'
pivotname = 'newsTickerCounts'

def filterNews(filename,starttime,endtime,pivotname):
    '''Takes a dataframe, converts timezone, filters by time range, creates
    a pivot table sorting tickers by frequency and saves to .csv file.'''
    df = pd.read_csv(filename)
    df.TimeOfArrival = pd.to_datetime(df.TimeOfArrival)
    df.TimeOfArrival = df.TimeOfArrival.dt.tz_convert('US/Eastern')
    df.TimeOfArrival = df.TimeOfArrival.dt.localize(None)
    df = df.set_index(df.TimeOfArrival)
    df = df[((df.index.strftime('%H:%M:%S') >= starttime) & (df.index.strftime('%H:%M:%S') <= endtime))]
    df = df[(df.index.weekday < 5)]
    
    startdate = str(df.index.date[0])
    enddate = str(df.index.date[-1])
    
    dfpivot = df.pivot_table(index=df.Ticker,aggfunc={'Relevance':np.mean,'SUID':'count'})
    dfpivot = dfpivot.sort_values('SUID', ascending=False)
    
    dfpivot.to_csv(pivotname + startdate + 'TO' + enddate + '.csv')
    
    return dfpivot
    

def dictFilter(df):
	'''Takes a dataframe, converts timezone, filters by time range.'''
    starttime = '10:30:00'
    endtime = '15:00:00'
    
    df.TimeOfArrival = pd.to_datetime(df.TimeOfArrival)
    try:
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_convert('US/Eastern')
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_localize(None)
        df = df.set_index(df.TimeOfArrival)
        df = df[((df.index.strftime('%H:%M:%S') >= starttime) & (df.index.strftime('%H:%M:%S') <= endtime))]
        df = df[(df.index.weekday < 5)]
        df = df.rename(columns={'SUID':'Frequency'})
        df = df.rename(columns={'Relevance':'Avg Relevance'})
    

    except:
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_localize(None)
        df = df.set_index(df.TimeOfArrival)
        df = df[((df.index.strftime('%H:%M:%S') >= starttime) & (df.index.strftime('%H:%M:%S') <= endtime))]
        df = df[(df.index.weekday < 5)]
        df = df.rename(columns={'SUID':'Frequency'})
        df = df.rename(columns={'Relevance':'Avg Relevance'})
    

    return df

def dictFilterPivot(df):
	'''Takes a dataframe, converts timezone, filters by time range, creates
    a pivot table sorting tickers by frequency.'''
    starttime = '10:30:00'
    endtime = '15:00:00'
    
    df.TimeOfArrival = pd.to_datetime(df.TimeOfArrival)
    try:
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_convert('US/Eastern')
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_localize(None)
        df = df.set_index(df.TimeOfArrival)
        df = df[((df.index.strftime('%H:%M:%S') >= starttime) & (df.index.strftime('%H:%M:%S') <= endtime))]
        df = df[(df.index.weekday < 5)]
        df = df.rename(columns={'SUID':'Frequency'})
        df = df.rename(columns={'Relevance':'Avg Relevance'})
    
        dfpivot = df.pivot_table(index=df.Ticker,aggfunc={'Avg Relevance':np.mean,'Frequency':'count'})
        dfpivot = dfpivot.sort_values('Frequency', ascending=False)
    
    except:
        df.TimeOfArrival = df.TimeOfArrival.dt.tz_localize(None)
        df = df.set_index(df.TimeOfArrival)
        df = df[((df.index.strftime('%H:%M:%S') >= starttime) & (df.index.strftime('%H:%M:%S') <= endtime))]
        df = df[(df.index.weekday < 5)]
        df = df.rename(columns={'SUID':'Frequency'})
        df = df.rename(columns={'Relevance':'Avg Relevance'})
    
        dfpivot = df.pivot_table(index=df.Ticker,aggfunc={'Avg Relevance':np.mean,'Frequency':'count'})
        dfpivot = dfpivot.sort_values('Frequency', ascending=False)
    
    
    
    return dfpivot
    
filename = 'News_Analytics_20130713_20170101.csv'

def splitLargeFileinChuncks(filename):
    '''Takes filename as input, returns large file in chunks as a 
    dictionary list of dataframes.'''

    dfList = {}
    df_iter = pd.read_csv(filename,chunksize = 1000000,iterator=True)
    for iter_num, chunk in enumerate(df_iter,1):
        dfList[iter_num] = chunk
    return dfList

def getPivotList(dfList):
	'''Takes a dictionary list of dataframes as input, returns a 
	dictionary list of pivot tables for each key.'''
    pivotList = {}

    for key, value in dfList.items():
        pivotList[key] = dictFilterPivot(value)
    return pivotList

def saveDataFrames(dfList):
	'''Takes a dictionary list of dataframes as input, saves each
	dataframe as .csv file'''
    for key, value in dfList.items():
        df = dictFilter(value)
        df.to_csv(pivotname + '_' + str(df.index.date[0]) + 'TO' + str(df.index.date[-1]) + '.csv')
        
        
def savePivots(pivotList):
	'''Saves each dataframe in a dictionary list to .csv file.'''
    df = pd.DataFrame
    for key,value in pivotList.items():
        value.to_csv(pivotname + '_' + str(key) + '.csv') 

def concatPivots(pivotList):
	'''Concatenate dictionary list of pivot tables, compute avg, sum pivots
	and save each to .csv'''
    #df = pd.DataFrame
    #for key,value in pivotList.items():
    dfnew = pd.concat(pivotList)
    pivotAvg = dfnew.mean(level=1).sort_values('Frequency',ascending=False)
    pivotSum = dfnew.sum(level=1).sort_values('Frequency',ascending=False)
    dfnew.to_csv('TickerCountByBatch_20130713_20170101.csv')
    pivotAvg.to_csv('TickerCountAvg_20130713_20170101.csv')
    pivotSum.to_csv('TickerCountSum_20130713_20170101.csv')
    return dfnew

