#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:32:42 2018

@author: ibenfjordkjaersgaard
"""
import sys
sys.path.append('/Users/ibenfjordkjaersgaard/Library/Mobile Documents/com~apple~CloudDocs/Documents/Uni/Semester 4/Machine learning og data mining/02450Toolbox_Python/Tools')

from similarity import binarize2 

from projekt3 import pimaData


pimaData = pimaData

binarize2(pimaData,['a','b','c','d'])