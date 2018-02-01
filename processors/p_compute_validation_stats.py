import numpy as np
import json

import sys,os
parent_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from mlpy import readmda

processor_name='spikeforest.compute_validation_stats'
processor_version='0.11'
def compute_validation_stats(*,confusion_matrix,output,output_format='json'):
    """
    Compute validation stats from a confusion matrix (see ms3.confusion_matrix). The first dimension (rows) of the confusion matrix should correspond to ground truth.

    Parameters
    ----------
    confusion_matrix : INPUT
        The path of the confusion matrix in .mda format. The first dimension (rows) should correspond to ground truth. The final row and final column correspond to unclassified events.
    output : OUTPUT
        The output file
    
    output_format : string
        For now this should always be 'json'
    """
    print(type(confusion_matrix))
    if type(confusion_matrix)==str:
        CM=readmda(confusion_matrix)
    else:
        CM=confusion_matrix
    K1=CM.shape[0]-1
    K2=CM.shape[1]-1
    if (K1<0) or (K2<0):
        print ('Error: not enough rows or columns in confusion matrix')
        return False
    row_sums=np.sum(CM,axis=1)
    row_sums=np.maximum(1,row_sums) # do not permit zeros in denominator
    col_sums=np.sum(CM,axis=0)
    col_sums=np.maximum(1,col_sums) # do not permit zeros in denominator
    accuracies=np.zeros(K1)
    num_correct=np.zeros(K1).astype(int)
    num_false_positives=np.zeros(K1).astype(int)
    num_false_negatives=np.zeros(K1).astype(int)
    for k1 in range(1,K1+1):
        row=CM[k1-1,:]
        tmp=row/(col_sums+row_sums[k1-1]-row)
        maxind=np.argmax(tmp[0:K2])
        accuracies[k1-1]=tmp[maxind]
        num_correct[k1-1]=row[maxind]
        num_false_positives[k1-1]=col_sums[maxind]-row[maxind]
        num_false_negatives[k1-1]=row_sums[k1-1]-row[maxind]
    
    obj={'accuracies':accuracies.tolist(),'num_correct':num_correct.tolist(),'num_false_positives':num_false_positives.tolist(),'num_false_negatives':num_false_negatives.tolist()};
    with open(output, 'w') as outfile:
        json.dump(obj, outfile, indent=4, sort_keys=True)    
    return True

def test_compute_validation_stats():
    CM=np.floor(np.random.uniform(0,100,(5,7)))
    if not compute_validation_stats(confusion_matrix=CM,output='tmp.json'):
        print ('compute_validation_stats returned with error')
        return False
    obj = json.load(open('tmp.json'))
    print(json.dumps(obj,indent=4,sort_keys=True))

compute_validation_stats.test=test_compute_validation_stats
compute_validation_stats.name = processor_name
compute_validation_stats.version = processor_version

if __name__ == '__main__':
    print ('Running test')
    test_compute_validation_stats()
