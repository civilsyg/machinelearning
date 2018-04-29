import pandas as pd
import numpy as np
clsGMM = pd.read_csv('clsGMM.csv', sep=',')
clsHie = pd.read_csv('clsHie.csv',sep = ',' )

clsGMM = clsGMM['0'].values
clsHie = clsHie['0'].values-1

evalue = sum(clsGMM==clsHie) / len(clsGMM)
