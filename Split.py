# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:14:26 2016

@author: it-lab412
"""

def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq