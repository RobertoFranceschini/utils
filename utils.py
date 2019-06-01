#!/usr/bin/env python3
# coding: utf-8 
# # Import

# In[112]:


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
from scipy.optimize import curve_fit
import uncertainties
import NumpyClasses as npc
import sympy
from sympy.utilities.lambdify import lambdify
import scipy.interpolate
import matplotlib.pyplot as plt
import unicodedata
import matplotlib


# In[ ]:


def unload_symbols():
    for letter in dir(sympy.abc):
        if not letter.startswith('_'):
            del globals()[letter]


# # MatplotLib

# In[134]:


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# In[ ]:


def hide_xlabels_every_step(step):
    ax = plt.gca()
    for label in ax.get_xaxis().get_ticklabels()[::step]:
        label.set_visible(False)


# In[135]:


def Plot(f,var,minvar,maxvar,PlotPoints=30,PlotLabel=[None,None],PlotTitle=None,xbins=20,ybins=20,x_tick_rotation=0,y_tick_rotation=0,lower_xtick_density=None,xscilimits=None,yscilimits=None,fmt='-',**args):
        """
        xscilimits=(0,0) to apply scientific notation to all numbers
        https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ticklabel_format
        (m, n), pair of integers; if style is ‘sci’, scientific notation will be used for numbers outside the range 10`m`:sup: to 10`n`:sup:. Use (0,0) to include all numbers.
        """
        _x=np.arange(minvar,maxvar,(maxvar-minvar)/PlotPoints)
        if type(f) is scipy.interpolate.interpolate.interp1d:
            _y=f(_x)
        elif isinstance(var, tuple(sympy.core.all_classes)):
            print('is sympy')
            _y=np.array([ f.subs(var,__x) for __x in _x ])
        elif type(f) is np.float64:
            _y=np.full( (1, len(_x) ) , f )[0]
        else:
            _y=f(_x)
        fig = plt.plot(_x,_y,fmt,**args)
        if PlotLabel[0] is not None:
            plt.xlabel(PlotLabel[0])
        if PlotLabel[1] is not None:
            plt.ylabel(PlotLabel[1])
        if PlotTitle is not None:
            plt.title(PlotTitle)
            
        # Adjust plot    
        plt.xticks(rotation=x_tick_rotation)
        plt.yticks(rotation=y_tick_rotation)
        if xscilimits is not None:
            plt.ticklabel_format(style='sci', axis='x', scilimits=xscilimits)
        if yscilimits is not None:
            plt.ticklabel_format(style='sci', axis='y', scilimits=yscilimits)
        
        plt.locator_params(axis='y', nbins=ybins)
        plt.locator_params(axis='x', nbins=xbins)
        if lower_xtick_density is not None:
            hide_xlabels_every_step(lower_xtick_density)
        return fig 


# In[ ]:



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, range_display=[18,30], **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            #print(data[i,j])
            if range_display[0] <= data[i, j] <= range_display[1]:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
            #else:
            #    texts.append('')

    return texts


# In[130]:


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={},grid_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
        
        
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    
    gridkw = {'draw_grid':False}
    gridkw.update(grid_kw)
    
    if gridkw['draw_grid']:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=True)

    return im, cbar


# In[131]:


harvest=np.array([[4,1],[2,3]])
vegetables=['Carrots','Potatoes']
farmers=['Joe','Sam']


# In[132]:


im, cbar = heatmap(harvest, vegetables, farmers, ax=None, cmap="YlGn", cbarlabel='$\chi$-Label',grid_kw={'draw_grid':False})
texts = annotate_heatmap(im, valfmt="{x:.2f}",range_display=[2,3],verticalalignment='baseline')


# # Pandas

# In[ ]:


def columns2strings(raw_data):
    stri=''
    for col in raw_data.columns:
        stri+='\''+col+'\''+','
    return stri
        #np.std(raw_data['Gruppi'])
    #return stri


# # Fits

# In[ ]:


def fit_histogram_data(_func,bins,n,p0=None,bounds=(-np.inf,np.inf),**kw):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # A 1-d sigma should contain values of standard deviations of errors in ydata. In this case, the optimized function is chisq = sum((r / sigma) ** 2).    
    sigma=np.sqrt(n)
    popt, pcov = curve_fit(_func, midpoints(bins), n,sigma=sigma,p0=p0,absolute_sigma=True, **kw)
    return popt, pcov


# In[173]:


def fit_pandas_data(_func,_data,x=0,y=1,weights=2,error_format='latex',significant_digits=1,**kw):
    debug=False
    if type(x)==int and type(y)==int and type(weights)==int:
        _x=_data.iloc[:, x ]
        _y=_data.iloc[:, y ]
        _weights=_data.iloc[:, weights ]
    else:
        _x=_data[x]
        _y=_data[y]
        try:
            _weights=_data[weights]
        except KeyError:
            print('Weights for this fit were not found.')
            _weights=None
    if debug:
        print(_data)
        print(_x)
        print(_y)

        
    popt, pcov = curve_fit(_func, _x, _y, sigma=_weights, absolute_sigma=True, **kw)
    if debug:
        print(popt)
    #_const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #_linear=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #print('{:+.1uS}'.format(_const) , '( {:+.1uS}'.format(60*_linear)," )"+"* t/h" )
    #
    #lin=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #
    
    c95 = sympy.symbols('c95',real=True,positive=False)
    best_func_sympy=_func( *([c95]+list(popt)) )
    best_func_numpy=lambdify([c95],best_func_sympy,"numpy")
    
    raw_pars=[     uncertainties.ufloat( popt[p], np.sqrt(pcov[p,p] ) )  for p in range(len(popt) ) ]
    if debug:
        print(raw_pars)
    #https://pythonhosted.org/uncertainties/user_guide.html#global-formatting
    if error_format=='scientific':
        err_format=':.'+str(significant_digits)+'uS' #scientific shorthand notation   
    if error_format=='latex':
        err_format=':.'+str(significant_digits)+'uSL'
    if error_format=='plusminus':
        err_format=':.'+str(significant_digits)+'uP'
    
    refined_pars=[ p.format(err_format) for p in raw_pars ]
    if debug:
        print(refined_pars)
    result=npc.generic()
    result.parameters = popt
    result.covariance = pcov
    result.raw_parameters = raw_pars
    result.refined_parameters = refined_pars
    result.bestfit=best_func_numpy
    result.bestfit_sympy=best_func_sympy
    
    return result #popt, pcov


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


# In[ ]:


def sumQuadrature(list):
    '''
    list should be formatted as the output of the n output of numpy histogram, that is a list of arrays
    
    [array([18325., 10511.,  7899.,  6668.,  5479.,  4875.,  4175.,  3849.,
         3406.,  3051.,  2858.,  2518.,  2455.,  2244.,  2102.,  1889.,
         1736.,  1606.,  1555.,  1512.,  1435.,  1342.,  1260.,  1174.,
         1143.,  1091.,  1037.,   969.,   944.,   892.,     0.]),
     array([18341., 10623.,  7947.,  6509.,  5553.,  4800.,  4133.,  3727.,
         3504.,  3040.,  2809.,  2576.,  2456.,  2199.,  2078.,  1929.,
         1787.,  1652.,  1586.,  1441.,  1386.,  1265.,  1268.,  1254.,
         1114.,  1078.,  1095.,  1020.,   917.,   913.,     0.])]
    
    '''
    return  np.sqrt( np.sum( np.array([ np.power(el,2) for el in list ]) , axis=0 ) ) 


# In[ ]:


def versor(_beta):
    return _beta/np.sqrt(np.dot(_beta,_beta))


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

# In[113]:


def read_file_to_lines(file_name):
    _xml_groups=[]
    file = open(file_name,"r")
    for line in file:
        _xml_groups.append(line)
    return _xml_groups


# In[114]:


def write_lines_to_file(mylines,filename,mode='a',final_line=False):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("%s" % item)
    if final_line:
        thefile.write("\n")      


# In[115]:


def write_lines_to_file_newline(mylines,filename,mode='a'):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("\n%s" % item)


# In[116]:


def filejson2dictionary(fn):
    with open(fn) as json_data:
        d = json.load(json_data)
    return d


# In[117]:


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


# In[19]:


def MGPandasFromHTML(url):
    MGpage=pd.read_html(url,header=0)[-1]
    MGpage['CleanBanner']=MGpage['Banner'].apply(isstring)
    MGPage=pd.DataFrame((MGpage[MGpage['Banner'].str.contains('run')]).query('CleanBanner==True'))
    return MGPage


# # Strings analysis

# In[ ]:


def isGreek(token):
    """
    https://docs.sympy.org/latest/_modules/sympy/parsing/sympy_parser.html#split_symbols_custom
    from _token_splittable(token):
    """
    if '_' in token:
        return False
    else:
        try:
            return not not unicodedata.lookup('GREEK SMALL LETTER ' + token)
        except KeyError:
            return False


# In[ ]:


def isstring(s):
    return isinstance(s, str)

def nextto(s,string='ebeam1'):
    try:
        return s.split(string)[1].split('_')[1]
    except IndexError:
        return s
    
def measurementFromString(s,err='±'):
    return list(map(lambda x: float(x), s.split(err) ) )


# In[118]:


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

# In[119]:


def sort_by_ith(data,i):
    return sorted(data, key=lambda tup: tup[i])


# In[120]:


def flattenOnce(tags_times):
    return [y for x in tags_times for y in x]


# In[133]:


def arange(a,b,s):
    return np.arange(a,b+s,s)
def linspace(start,stop,step):
    return np.linspace(start, stop, num=(stop-start)/step, endpoint=True)


# In[134]:


linspace(0,2,0.2)


# # Strings

# In[122]:


def remove_multiple_spaces(string):
    return re.sub(' +',' ',string)


# In[123]:


def ToString(x):
    return str(x)


# In[124]:


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

# In[125]:


def dict2string(dictio):
    res=[]
    for key,value in dictio.items():
        res.extend([str(value)])
    return "_".join(res)


# # Number manipulations

# In[ ]:


def logticks(basis=[1,2,5],orders=[-1.,-2.,-3.,-4.]):
    return np.array(list(map(lambda x: np.array(basis)*np.power(10,x),np.array(orders) ))).flatten()


# In[126]:


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


# In[ ]:


# Define function for string formatting of scientific notation
# https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if not exponent:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


# In[127]:


def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """
    def _to_precision(x,p):
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
    
    _res = [_to_precision(_x,p) for _x in x ]
    
    if len(_res)==1:
        _res=_res[0]
   
    return _res

