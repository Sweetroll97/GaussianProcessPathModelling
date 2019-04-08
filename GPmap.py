import numpy as np;
import statistics as ss;
import pandas as ps;
import csv;

class trajectory:
    xs = np.array([], (float));
    ys = np.array([], (float));
    timestamp = np.array([], (float));
    
    def __init__(self, id):
        self.id = id;
        
    def add_point(self,x,y, time):
        self.xs.append(x);
        self.ys.append(y);
        self.timestamp.append(time);
        
    def get_trajectory():     
        print("not implemented");
        
def readcsvfile():
    with open('testfile.csv') as data:
        data = csv.reader(data, delimiter=',')
        linenr = 0;
        for row in data:
            if(linenr == 0):
                print(f'column names are {", ".join(row)}');
                linenr = linenr + 1;
            else:
                print(f'\t{row[0]} works in the {row[1]}');
                linenr = linenr + 1;
                
readcsvfile();