from sklearn.externals import joblib
import h5py
import numpy as np
import csv
t2c=joblib.load('token_to_cui')
t2c[1]='NOCUI'
t2c[2]='NOCUI'


def cui2vec(inputy):
    with open('/home/soumyar/cui2vec_pretrained.csv', 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row in csvreader:
            if row[0]==inputy:
                return np.array(list(map(float, row[1:])))


embedding_matrix = np.zeros((1148 + 1, 500))
for i in range(1,1148):
    if t2c[i] !='NOCUI':
        embedding_matrix[i]= cui2vec(t2c[i])

        

h5f = h5py.File('embedding.h5', 'w')
h5f.create_dataset('embedding_dataset', data=embedding_matrix)
h5f.close()
print('DONE!')
