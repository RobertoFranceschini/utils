#!/usr/bin/env python3
# coding: utf-8 
# # Import

# In[1]:


from __future__ import print_function
import math
import re
import sys
import os
import json
import subprocess
from subprocess import check_call
from difflib import SequenceMatcher
import scipy.sparse
import pandas as pd
from pandas.io.json import json_normalize
import pypdt
import numpy as np


# # Fits

# In[ ]:


def fit_histogram_data(_func,bins,n,p0=None,bounds=(-np.inf,np.inf)):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # A 1-d sigma should contain values of standard deviations of errors in ydata. In this case, the optimized function is chisq = sum((r / sigma) ** 2).    
    sigma=np.sqrt(n)
    popt, pcov = curve_fit(_func, midpoints(bins), n,sigma=sigma,p0=p0)
    return popt, pcov


# In[ ]:


def fit_pandas_data(_func,_data,p0=None,bounds=(-np.inf,np.inf)):
    _x=_data['x']
    _y=_data['y']
    _weights=_data['weights']

    popt, pcov = curve_fit(_func, _x, _y,sigma=_weights,p0=p0,bounds=bounds)
   
    #_const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #_linear=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #print('{:+.1uS}'.format(_const) , '( {:+.1uS}'.format(60*_linear)," )"+"* t/h" )
    #
    #lin=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #
    return popt, pcov


# # Numpy Manipulations

# In[ ]:


def np_sort_by_function_of_row(_mat,f,row='row'): # _mat needs to be np.transpose(cuts_matrix)
    _final_touch=np.transpose
    if row=='column':
        _mat=np.transpose(_mat)
        
    _t=np.array([ np.append(cut_results, f(cut_results)) for cut_results in _mat  ])
    res=np.sort(np.transpose(_t))
    if row=='column':
        return res
    return _final_touch(res)


# In[ ]:


def snake_flatten_matrix(m):
    return np.array([m[i][::-1] if i%2!=0 else m[i] for i in range(len(m))]).flatten()


# In[ ]:


def midpoints(bins):
    return (bins[:-1]+bins[1:])/2


# # Logical

# In[ ]:


def boolstring2binary(c):
     return int(''.join([ str(int(gt(r,0))) for r in c ]) , 2) 


# In[17]:


def test(d):  
    return d['relation'](d['values'],d['threshold'])


# In[18]:


def lt(a,b):
    return a<b
def le(a,b):
    return a<=b
def gt(a,b):
    return a>b
def ge(a,b):
    return a>=b
def eq(a,b):
    return a==b
def bt(a,b):
    return b[0] < a < b[1]


# # File I/O

# In[2]:


def read_file_to_lines(file_name):
    _xml_groups=[]
    file = open(file_name,"r")
    for line in file:
        _xml_groups.append(line)
    return _xml_groups


# In[3]:


def write_lines_to_file(mylines,filename,mode='a',final_line=False):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("%s" % item)
    if final_line:
        thefile.write("\n")      


# In[4]:


def write_lines_to_file_newline(mylines,filename,mode='a'):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("\n%s" % item)


# In[5]:


def filejson2dictionary(fn):
    with open(fn) as json_data:
        d = json.load(json_data)
    return d


# In[6]:


def change_tag_in_file(filename=None,tag=None,text=None):
     # Read in the file
    filedata = None
    if filename != None and tag != None and text != None :
        filedata = read_file_to_lines(filename)
        write_lines_to_file(filedata,filename+'.bak',mode='w',final_line=False)

        newfiledata = []
        for line in filedata:
            newline = line.replace(tag, text)
            newfiledata.append(newline)
            if line != newline:
                print(line, '---> ',  newline )


        # Write the file out again
        write_lines_to_file(newfiledata,filename,mode='w',final_line=False)


# # Strings analysis

# In[7]:


def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print( query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(xrange(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l / step]
        bmv_r = match_values[p_l / step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print( "\n" + str(f))
                print( "ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print( "lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print( "rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print( "rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print( "Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value


# # Lists

# In[8]:


def sort_by_ith(data,i):
    return sorted(data, key=lambda tup: tup[i])


# In[9]:


def flattenOnce(tags_times):
    return [y for x in tags_times for y in x]


# In[10]:


def arange(a,b,s):
    return np.arange(a,b+s,s)


# # Strings

# In[11]:


def remove_multiple_spaces(string):
    return re.sub(' +',' ',string)


# In[12]:


def ToString(x):
    return str(x)


# In[13]:


def dashed_to_year(stri):
    #print stri
    if not stri==None:
        fields=stri.split('-')
        fields1=fields[0].split(',')
        fields2=fields1[0].split('/')
        fields3=fields2[0].split('.')
        res=[]
        if len(fields3) == 2:
            #print "got a dot"
            res=[fields3[1]]
        else:
            res=fields3
        #       fields5=fields4[0].split()
        #       if len(fields5) == 2:
        #           fields6=reversed(fields5)
        #       else:
        #           fields6=fields5

        #print fields[0]
        #print int(fields[0])
        try:
            #print stri
            #print fields, fields1, fields2, fields3, res
            return int(res[0])
        except ValueError:
            #date_object = datetime.strptime(fields, '%Y')
            print(stri)
            return 0
    else:
        return -1


# # Dictionaries

# In[14]:


def dict2string(dictio):
    res=[]
    for key,value in dictio.items():
        res.extend([str(value)])
    return "_".join(res)


# # Number manipulations

# In[15]:


def num(s):
    try:
        return float(s)
    except ValueError:
        return s
def chop(n,eps):
	if abs(n)<eps:
		return 0
	else:
		return n


# In[16]:


def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

