import pandas as pd
import numpy as np
import pickle
import sklearn.preprocessing as preprocessing
from sklearn_pandas import DataFrameMapper

# Read the bids, replace NaN values with a dash because NaN are string categories we don't know
bids = pd.read_csv('bids.csv', header=0)
bids.fillna('-', inplace=True)

# Load train and test bidders lists (nothing really interesting in that list)
train = pd.read_csv('train.csv', header=0, index_col=0)
test = pd.read_csv('test.csv', header=0, index_col=0)

# Join those 2 datasets together (-1 outcome meaning unknown i.e. test)
test['outcome'] = -1.0
bidders = pd.concat((train, test))

def computeStatsCat(series, normalizeCount = 1.0):

    n = float(series.shape[0])
    counts = series.value_counts()

    nbUnique = counts.count()
    hiFreq = counts[0] / n
    loFreq = counts[-1] / n
    argmax = counts.index[0]
    stdFreq = np.std(counts / n)

    return (nbUnique, loFreq, hiFreq, stdFreq, argmax)

# Compute stats of numerical series without caring about interval between values
def computeStatsNumNoIntervals(series):

    min = series.min()
    max = series.max()
    mean = np.mean(series)
    std = np.std(series)
    perc20 = np.percentile(series, 20)
    perc50 = np.percentile(series, 50)
    perc80 = np.percentile(series, 80)

    return (min, max, mean, std, perc20, perc50, perc80)

# Compute stats of a numerical series, taking intervals between values into account
def computeStatsNum(series, copy = True):

    if copy:
        series = series.copy()

    series.sort()
    intervals = series[1:].as_matrix() - series[:-1].as_matrix()
    if len(intervals) < 1:
        intervals = np.array([0])

    nb = series.shape[0]
    min = series.min()
    max = series.max()
    range = max - min
    intervalsMin = np.min(intervals)
    intervalsMax = np.max(intervals)
    intervalsMean = np.mean(intervals)
    intervalsStd = np.std(intervals)
    intervals25 = np.percentile(intervals, 25)
    intervals50 = np.percentile(intervals, 50)
    intervals75 = np.percentile(intervals, 75)

    return (nb, min, max, range,
            intervalsMin, intervalsMax, intervalsMean, intervalsStd,
            intervals25, intervals50, intervals75)

# Compute stats about a numerical column of table, with stats on sub-groups of this column (auctions in our case).
def computeStatsNumWithGroupBy(table, column, groupby):

    # get series and groups
    series = table[column]
    groups = table.groupby(groupby)

    # global stats
    (nb, min, max, range,
    intervalsMin, intervalsMax, intervalsMean, intervalsStd,
    intervals25, intervals50, intervals75) = computeStatsNum(series)

    # stats by group
    #group = per bidder_id per auction
    X = []
    for _, group in groups:
        (grpNb, _, _, grpRange, grpIntervalsMin, grpIntervalsMax, grpIntervalsMean, grpIntervalsStd, _, _, _) = computeStatsNum(group[column])
        X.append([grpNb, grpRange, grpIntervalsMin, grpIntervalsMax, grpIntervalsMean, grpIntervalsStd])
    X = np.array(X)

    grpNbMean = np.mean(X[:,0]) #
    number mean from X[:,0] #which is grpNb from return statement which is grpNb
    grpNbStd = np.std(X[:,0])
    grpRangeMean = np.mean(X[:,1])
    grpRangeStd = np.std(X[:,1])
    grpIntervalsMinMin = np.min(X[:,2])
    grpIntervalsMinMean = np.mean(X[:,2])
    grpIntervalsMaxMax = np.max(X[:,3])
    grpIntervalsMaxMean = np.mean(X[:,3])
    grpIntervalsMean = np.mean(X[:,4])
    grpIntervalsMeanStd = np.std(X[:,4])
    grpIntervalsStd = np.mean(X[:,5])


    return (nb, min, max, range,
            intervalsMin, intervalsMax, intervalsMean, intervalsStd,
            intervals25, intervals50, intervals75,
            grpNbMean, grpNbStd, grpRangeMean, grpRangeStd,
            grpIntervalsMinMin, grpIntervalsMinMean, grpIntervalsMaxMax, grpIntervalsMaxMean,
            grpIntervalsMean, grpIntervalsMeanStd, grpIntervalsStd)

# Init vars
Xids = []
X = []

# Old init for stats about merchadises
# merchandises = bids.merchandise.value_counts()

# For each bidder
for bidder, group in bids.groupby('bidder_id'):

    # Compute the stats
    (nbUniqueIP, loFreqIP, hiFreqIP, stdFreqIP, IP) = computeStatsCat(group.ip)
    (nbUniqueDevice, loFreqDevice, hiFreqDevice, stdFreqDevice, device) = computeStatsCat(group.device)
    (nbUniqueMerch, loFreqMerch, hiFreqMerch, stdFreqMerch, merch) = computeStatsCat(group.merchandise)
    (nbUniqueCountry, loFreqCountry, hiFreqCountry, stdFreqCountry, country) = computeStatsCat(group.country)
    (nbUniqueUrl, loFreqUrl, hiFreqUrl, stdFreqUrl, url) = computeStatsCat(group.url)
    (nbUniqueAuction, loFreqAuction, hiFreqAuction, stdFreqAuction, auction) = computeStatsCat(group.auction)
    (auctionNb, auctionMin, auctionMax, auctionRange,
    auctionIntervalsMin, auctionIntervalsMax, auctionIntervalsMean, auctionIntervalsStd,
    auctionIntervals25, auctionIntervals50, auctionIntervals75,
    auctionGrpNbMean, auctionGrpNbStd, auctionGrpRangeMean, auctionGrpRangeStd,
    auctionGrpIntervalsMinMin, auctionGrpIntervalsMinMean, auctionGrpIntervalsMaxMax, auctionGrpIntervalsMaxMean,
    auctionGrpIntervalsMean, auctionGrpIntervalsMeanStd, auctionGrpIntervalsStd) = computeStatsNumWithGroupBy(group, 'time', 'auction')
#compute stats in group table on column 'time' by 'auction'
    x = [nbUniqueIP, loFreqIP, hiFreqIP, stdFreqIP, #IP,
          nbUniqueDevice, loFreqDevice, hiFreqDevice, stdFreqDevice, #device,
          nbUniqueMerch, loFreqMerch, hiFreqMerch, stdFreqMerch, merch,
          nbUniqueCountry, loFreqCountry, hiFreqCountry, stdFreqCountry, country,
          nbUniqueUrl, loFreqUrl, hiFreqUrl, stdFreqUrl, #url,
          nbUniqueAuction, loFreqAuction, hiFreqAuction, stdFreqAuction, #auction
          auctionNb, auctionMin, auctionMax, auctionRange,
          auctionIntervalsMin, auctionIntervalsMax, auctionIntervalsMean, auctionIntervalsStd,
          auctionIntervals25, auctionIntervals50, auctionIntervals75,
          auctionGrpNbMean, auctionGrpNbStd, auctionGrpRangeMean, auctionGrpRangeStd,
          auctionGrpIntervalsMinMin, auctionGrpIntervalsMinMean, auctionGrpIntervalsMaxMax, auctionGrpIntervalsMaxMean,
          auctionGrpIntervalsMean, auctionGrpIntervalsMeanStd, auctionGrpIntervalsStd]

    Xids.append(bidder)
    X.append(x)

# Features labels
Xcols = ['nbUniqueIP', 'loFreqIP', 'hiFreqIP', 'stdFreqIP', #'IP',
              'nbUniqueDevice', 'loFreqDevice', 'hiFreqDevice', 'stdFreqDevice', #'device',
              'nbUniqueMerch', 'loFreqMerch', 'hiFreqMerch', 'stdFreqMerch', 'merch',
              'nbUniqueCountry', 'loFreqCountry', 'hiFreqCountry', 'stdFreqCountry', 'country',
              'nbUniqueUrl', 'loFreqUrl', 'hiFreqUrl', 'stdFreqUrl', #'url',
              'nbUniqueAuction', 'loFreqAuction', 'hiFreqAuction', 'stdFreqAuction','auctionNb', 'auctionMin', 'auctionMax', 'auctionRange',
              'auctionIntervalsMin', 'auctionIntervalsMax', 'auctionIntervalsMean', 'auctionIntervalsStd',
              'auctionIntervals25', 'auctionIntervals50', 'auctionIntervals75',
              'auctionGrpNbMean', 'auctionGrpNbStd', 'auctionGrpRangeMean', 'auctionGrpRangeStd',
              'auctionGrpIntervalsMinMin', 'auctionGrpIntervalsMinMean', 'auctionGrpIntervalsMaxMax', 'auctionGrpIntervalsMaxMean',
              'auctionGrpIntervalsMean', 'auctionGrpIntervalsMeanStd', 'auctionGrpIntervalsStd']

# Create a pandas dataset, remove NaN from dataset and show it
dataset = pd.DataFrame(X,index=Xids, columns=Xcols)
dataset.fillna(0.0, inplace=True)
dataset

# First lets join the dataset with outcome because it might be a useful info in the future ;)
datasetFull = dataset.join(bidders[['outcome']])
types = datasetFull.dtypes

# Create a mapper that "standard scale" numbers and binarize categories
mapperArg = []
for col, colType in types.iteritems():
    if col == 'outcome':
        continue
    if colType.name == 'float64' or colType.name =='int64':
        mapperArg.append((col, sklearn.preprocessing.StandardScaler()))
    else:
        mapperArg.append((col, sklearn.preprocessing.LabelBinarizer()))
mapper = DataFrameMapper(mapperArg)

# Apply the mapper to create the cdataset
Xids = datasetFull.index.tolist()
X = mapper.fit_transform(datasetFull)
y = datasetFull[['outcome']].as_matrix()

# Last check!
print bidders['outcome'].value_counts()

# Save in pickle file
pickle.dump([Xids, X, y], open('Xy.pkl', 'wb'))
