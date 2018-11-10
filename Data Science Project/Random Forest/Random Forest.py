import numpy as np
import pandas as pd 
import cv2
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import time
from datetime import timedelta
#%matplotlib inline

data=pd.read_csv('train.csv')
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#Split arrays or matrices into random train and test subsets
#Insted of all rows this function will give random rows using shuffel

start_time = time.time()

rf=RandomForestClassifier(n_estimators=10)
rf.fit(df_x,df_y)    #Build a forest of trees from the training set (X, y).


pred=rf.predict(x_test)    #Predict regression target for X.

s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
end_time = time.time()
time_dif = end_time - start_time
# Print the time-usage.
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

#Method-1 : Input image file 
test_img = cv2.imread('test.png')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(test_img, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
mask_inv = mask_inv.reshape(28,28).astype('uint8')
height, width = test_img.shape
x = []
for i in range(height):
    for j in range(width):
        #print(mask_inv[i,j])
        x.append(mask_inv[i,j])
z = np.array(x)
z=z.reshape(1,-1).astype('uint8')
pred=rf.predict(z)
print(pred)       
plt.imshow(mask_inv)

#Method-2 : Input file from .CSV format 
test_a=data.iloc[11,1:].values
test_a=test_a.reshape(1,-1).astype('uint8')
plt.imshow(test_a)
pred=rf.predict(test_a)
print(pred)


# Cross-Validation
def train_test_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', 'default')
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    if test_size == 'default':
        test_size = None
        if train_size is not None:
            warnings.warn("From version 0.21, test_size will always "
                          "complement train_size unless both "
                          "are specified.",
                          FutureWarning)

    if test_size is None and train_size is None:
        test_size = 0.25

    arrays = indexable(*arrays)

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        n_samples = _num_samples(arrays[0])
        n_train, n_test = _validate_shuffle_split(n_samples, test_size,
                                                  train_size)

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=test_size,
                     train_size=train_size,
                     random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))
