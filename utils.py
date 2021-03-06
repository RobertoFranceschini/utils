#!/usr/bin/env python3
# coding: utf-8 
# # Import

# In[922]:


#from __future__ import print_function
import math
import re
import sys
import os
import json
import subprocess
import inspect
from termcolor import colored
from inspect import signature
from subprocess import check_call
from difflib import SequenceMatcher
import scipy.sparse
import pandas as pd
from pandas.io.json import json_normalize
import pypdt
import numpy as np
from scipy.optimize import curve_fit, minimize
import uncertainties
import matplotlib.pyplot as plt
import NumpyClasses as npc
#
import sympy
import sympy as sp
from sympy.utilities.lambdify import lambdify
#
import scipy.interpolate
import unicodedata
import matplotlib
import ray, time
#
from bokeh.io import show, output_file, save, export_png, export_svgs
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure


# In[ ]:


def unload_symbols():
    for letter in dir(sympy.abc):
        if not letter.startswith('_'):
            del globals()[letter]


# In[ ]:


from IPython.display import display as _display
from IPython.display import Math as _Math
from IPython.display import Latex as _Latex
def typeset(string):
    _display(_Math(string))


# # Ray

# In[ ]:


def check_RayObjectsIdDict(analysisResult, DEBUG=False):
    remaining_ids=0
    for k in analysisResult:
        if type(analysisResult[k]) is ray._raylet.ObjectID: # the item in the list is not yet ready or not yet retrived
            if DEBUG: print(k,' is still ObjectId',analysisResult[k])
            # check if this ID is ready
            ready_ids, _remaining_ids = ray.wait([analysisResult[k]],timeout=0)
            remaining_ids+=len(_remaining_ids)
            if len(ready_ids)>0:
                if DEBUG: print(k ,' is ready at ID', ready_ids)
                analysisResult[k] = ray.get(analysisResult[k])
                
    if DEBUG: print(remaining_ids, ' ID still remaining')
    return remaining_ids


# In[ ]:


def check_RayObjectsIdList(LHEsubevents, DEBUG=False):
    remaining_ids=0
    for th,k in enumerate(LHEsubevents):
        if type(k) is ray._raylet.ObjectID: # the item in the list is not yet ready or not yet retrived
            if DEBUG: print(k,' is still ObjectId')
            # check if this ID is ready
            ready_ids, _remaining_ids = ray.wait([k],timeout=0)
            remaining_ids+=len(_remaining_ids)
            if len(ready_ids)>0:
                if DEBUG: print(k ,' is ready')
                LHEsubevents[th] = ray.get(k)
                
    if DEBUG: print(remaining_ids, ' ID still remaining')
    return remaining_ids


# In[ ]:


def wait_for_ray_iterable(analysisResult,method=check_RayObjectsIdDict,DEBUG=False):
    start = time.time()
    while method(analysisResult, DEBUG=False) > 0:
        method(analysisResult, DEBUG=DEBUG)
        time.sleep(5)
    end = time.time()
    print(end - start, 'seconds')


# # MatplotLib

# http://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle/

# ```python
# fig_handle = plt.figure()
# plt.plot([1,2,3],[1,2,3])
# with open('123.pickle', 'wb') as f: 
#     pickle.dump(fig_handle, f) 
# ```

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
                     "{x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
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


# In[189]:


def histo2D(data, row_labels=None, col_labels=None, ax=None, labels_format=['{:.2g}','{:.2g}'] ,cbarlabel="",offsetx=0.5,offsety=-0.5, reverse=True,transpose=True,minor=False, **kwargs):
    '''
    `data` is a NPCHistogram with `bins` and `counts` data member. 
    '''
    
    if type(labels_format) is str:
        labels_format=[labels_format,labels_format]

    if row_labels is None:
        row_labels=[labels_format[0].format(r) for r in data.bins[0]]
    if col_labels is None:
        col_labels=[labels_format[1].format(c) for c in data.bins[1]]
    
    return heatmap(data.counts, row_labels , col_labels , ax=ax, cbarlabel=cbarlabel,offsetx=offsetx,offsety=offsety, reverse=reverse,transpose=transpose, minor=minor, **kwargs) 


# In[ ]:


def PandasXYZ2Dhisto(dataframe=None,var1=None,var2=None,f=None,interpolation='quadric'):
    '''
    unlike histo2D this is takes a DataFrame where the `x` `y` `z=f(x,y)`  are stored 
    this requires a full grid, otherwise the only possible plot is 
    `plt.tripcolor(dataframe['var1'],dataframe['var2'],dataframe['sensitvity'])`
    '''
    if (None not in [var1,var2,f]) and (type(dataframe) is  pd.DataFrame):

        _x=np.unique(np.array(dataframe[var1]))
        _y=np.unique(np.array(dataframe[var2]))

        _z = np.array( [  [ np.array(dataframe.query(var1+'=='+str(f1)+' & '+var2+'=='+str(f2))[f])[0]  for f2 in _y  ] for f1 in _x])

        _dummy=npc.NumpyHistogramData( counts=_z, bins=(_x,_y)   ) 

        methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                   'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                   'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


        fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(18, 12),
                                subplot_kw={'xticks': [], 'yticks': []})

        for ax, interp_method in zip(axs.flat, methods):
            ax.imshow(_z, interpolation=interp_method, cmap='viridis')
            ax.set_title(str(interp_method))

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.imshow(_dummy.counts, interpolation=interpolation)


# In[268]:


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={},grid_kw={}, cbarlabel="",offsetx=0,offsety=0, reverse=False,transpose=False,minor=True, **kwargs):
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
    
 
    harvest=np.array([[4,1],[2,3]])
    vegetables=['Carrots','Potatoes']
    farmers=['Joe','Sam']
    im, cbar = heatmap(harvest, vegetables, farmers, ax=None, cmap="YlGn", cbarlabel='$\chi$-Label',grid_kw={'draw_grid':False})
    texts = annotate_heatmap(im, valfmt="{x:.2f}",range_display=[2,3],verticalalignment='baseline')
    
    
    
    """
    
   
    rev=2*int(False)-1 # 1 if True -1 if False
    if transpose:
        data=np.transpose(data)
    if not ax:
        ax = plt.gca()

       
    # Plot the heatmap
    im = ax.imshow(data[::rev], **kwargs)
    

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1])-offsetx)
    ax.set_yticks(np.arange(data.shape[0])-offsety)
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels[::rev])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
        
        
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=minor)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=minor)
    
    gridkw = {'draw_grid':False}
    gridkw.update(grid_kw)
    
    if gridkw['draw_grid']:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=True)

    return im, cbar


# In[ ]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def spiderPlot(properties,df):
    ''' 
    Plots a polygon plot (radar-plot, spider-plot) of the values in the columns `properties` of the dataframe `df`
    `df` must contain only one row at this point 
    '''
    N = len(properties)

    theta = radar_factory(N, frame='polygon')
    fig, axes = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    axes.set_rgrids([1,3, 9,27])
    color='g'
    d = df.to_numpy()[0]

    axes.set_varlabels(properties)
    axes.plot(theta, d, color=color)
    axes.fill(theta, d, facecolor=color, alpha=0.25)

    plt.show()
    return df


# In[ ]:


def find_range(chiS,a_scale=(-2e3/5,2e3/5,50)):
    _mg=np.meshgrid(np.linspace(*a_scale),np.linspace(*a_scale))
    return np.min(sp.lambdify(chiS.free_symbols,chiS)(*_mg)), np.max(sp.lambdify(chiS.free_symbols,chiS)(*_mg))


# In[239]:


def contour_plot(Z,x_seq=(-2e-3,2e-3,50),y_seq=(-2e-3,2e-3,50),label_format='%1.0f',levels=np.arange(0, 100, 10),cmap="YlGn",shades_alpha=1.0,contour_color='k',contour_alpha=1.0,labels=True,label_color='k',label_alpha=1.0,xlabel='$x$ label',ylabel='ylabel',title='title',fontsize=14,fig=None,ax=None,return_fig=False,colorbar=True):
    x, y = np.meshgrid(np.linspace(*x_seq), np.linspace(*y_seq))
    z=Z(x,y)
    #
    if type(levels)==int:
        levels=np.linspace(np.min(z),np.max(z),levels)
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots()
    im=ax.contourf(x, y, z,levels,cmap=cmap,alpha=shades_alpha)
    CS=ax.contour(x, y, z, levels, colors=contour_color,alpha=contour_alpha)
    if labels:
        ax.clabel(CS, levels, inline=1, fmt=label_format,colors=label_color, fontsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim
    if colorbar:
        fig.colorbar(im, orientation='vertical', shrink=0.99)
    if return_fig:
        return fig, ax


# In[ ]:


def cross(x,y,style='rx',**kwargs):
    '''
    kwargs e.g. markersize=7,marker='x',mfc="C1", mec="C2"
    '''
    _x, _y = np.meshgrid(np.array([x]), np.array(y))
    plt.plot(_x,_y,style,**kwargs)


# #  Bokeh

# In[ ]:


def makeDateSeriesPlot(data,file_name_root='plot',output_format='png',ylabel='Temperature [C]',xdata='Date',ydata='Temperature'):
    """
    A function that makes a plot in a file `file_name_root` with extension `output`  from a sample of data `data`.
    Data is a Pandas Dataframe and plots `ydata` against a datetime `xdata`. 
    """
    p = figure(plot_height=300, plot_width=800, tools="xpan,reset,save,hover", toolbar_location='left',
               x_axis_type="datetime", x_axis_location="above",
               background_fill_color="#efefef", x_range=(data[xdata].min(), data[xdata].max()))

    p.line(xdata, ydata, source=data)
    p.yaxis.axis_label = ylabel

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=130, plot_width=800, y_range=p.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line(xdata, ydata, source=data)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool


    if output_format=='web':
        show(column(p, select))
    if output_format=='html':
        output_file(file_name_root+".html")
        save(column(p, select))
    if output_format=='svg':
        p.output_backend = "svg"
        export_svgs(column(p, select), filename=file_name_root+".svg")
    if output_format=='png':
        export_png(column(p, select), filename=file_name_root+".png")


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


# In[169]:


def fit_pandas_data(_func,_data,x=0,y=1,weights=2,error_format='latex',significant_digits=1,debug=True,**kw):
    '''
    This functions fits data in `R^n` with a real function `_func` with target space `R`
    
    _func may be a vectorized function, which takes a matrix MxN of inputs and treat each of the M row as apoint in R^N
    
    The output is given a generic python object `result` which, by `duck typing`,  has properties
    
    `result.x` The fitted x data in R^n
    `result.y` The value of the data in R
    `result.parameters` The best fit parameters
    `result.covariance` The covariance matrix
    `result.raw_parameters` The parameters with errors as `ufloat`
    `result.refined_parameters` The paramters formatted within significant digits
    ## sympy outputs
    `result.bestfit_sympy` The best fit function as sympy *expression*
    `result.bestfit_sympy_sympy` The fitted functional form as sympy *expression*
    ## nympy outputs
    `result.bestfit`    The best fit function as numpy *function* (obtained via lambdify).
                        This function takes N positional arguments (printed if `debug` is asked).
                        These arguments are taken from an np.array, thus are *guranteed* to be *ordered* _0 _1 ...
    `result.bestfit_Rn` The best fit function as numpy *function* (obtained straigh from _func).
                        This function takes 1  argument that is a vector in R^N (printed if `debug` is asked)
    `result.residuals` The *absolute* residuals between the fitted function and the data `y`
    `result.relative_residuals` The *relative* residuals between the fitted function and the data `y`
    
    '''
    #debug=False
    if type(x)==int and type(y)==int and type(weights)==int:
        _x=_data.iloc[:, x ]
        _y=_data.iloc[:, y ]
        _weights=_data.iloc[:, weights ]
    else:
        if debug:
            print('domain is ', x)
        # _x is the variable that I will put in the curve_fit
        _x=np.array(_data[x]) # if x is a list this will be a matrix
        if debug:
            print(_x)
        _y=_data[y]
        try:
            _weights=_data[weights]
        except KeyError:
            print('Weights for this fit were not found.')
            _weights=None
    if debug:
        print('_data:',_data)
        print('_x:',_x)
        print('_y:',_y)

        
    popt, pcov = curve_fit(_func, _x, _y, sigma=_weights, absolute_sigma=True, **kw)
    if debug:
        print('popt=',popt)
    #_const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #_linear=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #print('{:+.1uS}'.format(_const) , '( {:+.1uS}'.format(60*_linear)," )"+"* t/h" )
    #
    #lin=ufloat(popt[1],np.sqrt(pcov[1,1]))
    #const=ufloat(popt[0],np.sqrt(pcov[0,0]))
    #
    # MAKING SYMPY OUTPUTS
    c95s = sp.symarray('c_95',  _x.shape[-1] ) #these are guranteed to be ordered _0 _1 ...
    # if debug:
    print('c_95s', colored(c95s,'blue') , ' type ', type(c95s))
    print('list(c_95s)', colored(list(c95s),'blue') , ' type ', type(c95s))
    print('np.array([c_95s])', colored( np.array([c95s]), 'blue'), 'for  best_func_numpy=lambdify(c95s,best_func_sympy,"numpy") ' ) 
    _p = sp.symarray('p',  popt.shape[-1] )
    #c95 = sympy.symbols('c95',real=True)
    best_func_sympy_sympy=_func( np.array([c95s]),*_p )   # needs to feed an array because _func is vectorized
    best_func_sympy=_func( np.array([c95s]),*list(popt) ) # needs to feed an array because _func is vectorized
    

    
    # MAKING NUMPY OUTPUTS
    def best_func_numpy_from_func(x_in_Rn): # in this way it is vectorized
        return _func( x_in_Rn  ,*list(popt) )
    
    ## ****** NUMPY VIA LAMBDIFY ***** 
    best_func_numpy=lambdify(c95s,best_func_sympy,"numpy") # c95s are a np.array thus guranteed to be *ordered* _0 _1 ...
    sig = signature( best_func_numpy )
    print( 'signature of best_func_numpy ', colored(str(sig),'yellow') )
    
    #if debug:
    # **** TEST **** THE TWO FUNCTIONS ARGUMENTS ARE ORDERED IN THE SAME WAY
    x_test = np.random.rand(_x.shape[-1])
    
    best_func_numpy_value = best_func_numpy(*x_test) 
    if debug: print('best_func_numpy' ,best_func_numpy_value  ) # the lambdified function take a tuple of positional arguments, cannot feed the vector itself
    best_func_sympy_value = best_func_sympy[0].subs({'c_95_'+str(_i):_x_test_i  for _i,_x_test_i in enumerate(x_test)   })
    if debug:  print('best_func_sympy' , best_func_sympy_value )  
    best_func_numpyRn_value = best_func_numpy_from_func(np.array([x_test]))
    if debug: print('best_func_numpyRn_value' , best_func_numpyRn_value )
    
    if (np.abs(best_func_numpy_value[0]/best_func_sympy_value -1) < 1e-5) and (np.abs(best_func_numpy_value[0]/best_func_numpyRn_value[0] -1) < 1e-5):
        if debug: print(colored('SYMPY AND NUMPY VALUES AGREE at 1e-5','green'))
    else:
        print(best_func_numpy_value[0],'!=',best_func_sympy_value)
        print(best_func_numpy_value[0],'!=',best_func_numpyRn_value[0])
    
    
    
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
        print("result.refined_parameters=",refined_pars)
    if debug:
        print("result.bestfit_sympy=",best_func_sympy)
    result=npc.generic()
    result.x=_x
    result.y=_y
    result.parameters = popt
    result.covariance = pcov
    result.raw_parameters = raw_pars
    result.refined_parameters = refined_pars
    ## numpy outputs
    result.bestfit=best_func_numpy
    result.bestfit_Rn=best_func_numpy_from_func
    ## sympy outputs
    result.bestfit_sympy=best_func_sympy
    result.bestfit_sympy_sympy=best_func_sympy_sympy
    #
    result.residuals=np.array(list(map(lambda x: result.bestfit(*x), result.x )))-np.array([result.y]).T
    result.relative_residuals=(np.array(list(map(lambda x: result.bestfit(*x), result.x )))/np.array([result.y]).T)-1
    return result #popt, pcov


# In[ ]:


def test_fit_function_arguments_are_consistent(fit_res,x_test=None,debug=False):
    if x_test is None:
        from inspect import signature
        sig = signature( fit_res.bestfit )
        #print()
        x_test = np.random.rand(len(sig.parameters))



    if debug: print(x_test)
    best_func_numpy_value = fit_res.bestfit(*x_test) 
    if debug: print('best_func_numpy' ,best_func_numpy_value  ) # the lambdified function take a tuple of positional arguments, cannot feed the vector itself
    best_func_sympy_value = fit_res.bestfit_sympy[0].subs({'c_95_'+str(_i):_x_test_i  for _i,_x_test_i in enumerate(x_test)   })
    if debug: print('best_func_sympy' , best_func_sympy_value )  
    best_func_numpyRn_value = fit_res.bestfit_Rn(np.array([x_test]))
    if debug: print('best_func_numpyRn_value' , best_func_numpyRn_value )


    if (np.abs(best_func_numpy_value[0]/best_func_sympy_value -1) < 1e-5) and (np.abs(best_func_numpy_value[0]/best_func_numpyRn_value[0] -1) < 1e-5):
        if debug: print(colored('SYMPY AND NUMPY VALUES AGREE at 1e-5','green'))
        return True
    else:
        print(best_func_numpy_value[0],'!=',best_func_sympy_value)
        print(best_func_numpy_value[0],'!=',best_func_numpyRn_value[0])
        return False


# In[26]:


def profiled_limits(observable,nCouplings=None,_lumi=0,_nsigma=2,events_numbers_format="{:6.1f}",ratios_number_format='{:.4g}', results_number_format='{:.4g}', tol=1e-7, maxiter=10000,DEBUG=False,marginalized=False,make_bound=True,fluctuation=None):
    '''
    - `observable` is callable, e.g. a numpy function. It must take N positional arguments in the ordering of `utils.sort_symbols`. \
       - It is `not a sympy` object. If a sympy expression is turned into numpy by lambdidy u.sort_symbols must be used to feed arguments to lambdidy.
    
    - `fluctuation` if it is left `None` is computed from _nsigma and the value at zero, otherwise
        I assume the fluctuation is a fixed number (float) for which to solve `deviation=fluctuation`
    
    Usage:
    for a vector-able `observable` functions, that is a functions that always returns a list, even if 1D
    e.g.  f(x)->[Sin(x)]
    pointers must be used throughout
    `profiled_limits(lambda *x: fit_res.bestfit(*x)[0] , _lumi=luminosity(eb),_nsigma=2)`
     
    '''
    _result=[]
    def inverse_coupling(x):
        return sp.sign(x)/sp.sqrt(sp.Abs(x))

    if nCouplings is None:
        # figure out the numner of couplings
        from inspect import signature
        sig = signature(observable)
        nCouplings=len(sig.parameters)
    if DEBUG: print(nCouplings,' couplings')
    # Number of events
    # SMcrosssection=np.array(fit_res.bestfit(*np.zeros(nCouplings)))[0]
    SMcrosssection=observable(*np.zeros(nCouplings) ) #all variavbles are zero, the ordering of the positional paramters is irrelevant
    SMevents=_lumi*SMcrosssection
    if DEBUG: print('Expected events in the SM', events_numbers_format.format(SMevents))
    if fluctuation is None:
        fluctuation=_nsigma*np.sqrt( _lumi*SMcrosssection)
    # else I assume the fluctuation is a fixed number (float) for which to solve `deviation=fluctuation`    
    if DEBUG: print(str(_nsigma)+'sigma fluctuation in the SM', events_numbers_format.format(fluctuation))

    # Treating the predictions
    c95i=sp.symarray('c_95',nCouplings) # these are an np.array, thus gurantedd to be ordered _0 _1 ... 

    
    ### THE ORDER OF THE OBSERVABLE arguments list MUST BE THE SAME AS FOR THE SYMBOLS free-symbols set
    
    print_signature(observable)
    deviation = _lumi* np.abs( observable(*c95i) - SMcrosssection )  
    # this assumes that the observable takes N positional arguments ordered
    # the sympy expression `deviation` might have free_symbols ordered in a different way.
    print('deviation.free_symbols',colored(deviation.free_symbols,'magenta'),colored('it may differ from the rest of lists that are signatures','magenta'))
    print('sort_symbols(deviation)',colored( sort_symbols(deviation),'yellow'))
    if not make_bound:
        return deviation, fluctuation
    else:
        for _i,_c95 in enumerate(c95i): 
            print('***************'+str(_i)+'*********************')
            # for each couplings I can do the profiled bound
            c95 = sp.symbols('c95',real=True)
            if DEBUG: print(c95i,_c95)
            point=list(np.zeros(nCouplings))
            point[_i]=c95
            if DEBUG: print(point)
            #
            deviation_this_direction = sp_replace(deviation,point,debug=DEBUG) # sp_replace uses sort_symbols, this it is guaranteed to call 0 the direction of c_95_0 and so on.
            if DEBUG: print('deviation',deviation)
            if DEBUG: print('deviation_this_direction',deviation_this_direction)
            sols = sp.solveset( sp.Eq(deviation_this_direction,fluctuation) ,c95,domain=sp.S.Reals)
            if DEBUG: print(sols)
            phys_sols=sorted(sols, key=lambda x: np.abs(x) )[0:2]
            print( 'couplings allowed range', colored(  [ float(results_number_format.format(_n)) for _n in sorted(phys_sols) ]  ,'cyan') )
            deviation.collect(c95)
            g95inverseroot=list(map(lambda x: sp.sign(x)/sp.sqrt(sp.Abs(x)), phys_sols ) )
            _result+=[g95inverseroot]
            if DEBUG: print(_c95,'$CW^{-1/2}$'+'@'+str(_nsigma)+r'$\sigma$'+' CL=', colored( [ results_number_format.format(_n) for _n in g95inverseroot ], 'blue') )
            print(_c95,'$CW^{-1/2}$'+'@'+str(_nsigma)+r'$\sigma$'+' CL=', colored( [ float(results_number_format.format(_n)) for _n in sorted(g95inverseroot) ], 'blue') )
            if DEBUG: print('equivalent $\hat{S}=$',np.mean( list(map(lambda x: 0.08**2*sp.Abs(x) , phys_sols ))) )
            BSMcontributions=np.array([ _lumi*( observable(*point).subs(c95,cc) - SMcrosssection ) for cc in phys_sols ])
            if DEBUG: print('Expected events  BSM - SM',  [ events_numbers_format.format(_n) for _n in BSMcontributions]   )
            if DEBUG: 
                if SMevents>0:
                    print('BSM/SM', [ ratios_number_format.format(_n) for _n in BSMcontributions/SMevents]  )
            if marginalized:
                _dev=sp.lambdify( sort_symbols(deviation) ,deviation)
                def con(x,func,value):
                        '''
                        constraint x**2+y**2=1 (a unit circle) can be passed as a list of arguments 'args':(lambda x,y: x**2+y**2,1,) 
                        if the function of the constraints is a vectorized function (i.e. always returns a list even if 1D)
                        `'args':( lambda *x: fit_res.bestfit(*x)[0],18,)` must be passed
                        '''
                        return func(*x) - value
                cons = {'type':'eq', 'fun': con, 'args':( _dev,fluctuation,)}
                minimization=minimize(lambda x: x[_i], np.zeros(nCouplings), constraints=cons , tol=tol,options={'maxiter':maxiter})
                cons = {'type':'eq', 'fun': con, 'args':( _dev,fluctuation,)}
                maximization=minimize(lambda x: -x[_i], np.zeros(nCouplings), constraints=cons , tol=tol,options={'maxiter':maxiter})
                if (maximization['status'] ==0) and (minimization['status'] ==0):
                    print('marginalized: [', results_number_format.format( inverse_coupling(minimization['fun'])  ),' ', results_number_format.format(-inverse_coupling(maximization['fun'])), ']')
                    print('marginalized couplings: [', results_number_format.format( (minimization['fun'])  ),' ', results_number_format.format( -(maximization['fun'])), ']')
                else:
                    print( colored('extremization error','red') )
                    print( colored(minimization['status'],'red') )
                    print( colored(maximization['status'],'red') )
            print('************************************')
    return _result


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


def snake_bins(hist2d,op=lambda x,y: '('+'{:.2g}'.format(x)+','+"{:.2g}".format(y)+')'):  
    A=hist2d.bins[0][:-1]
    B=hist2d.bins[1][:-1]
    r = np.empty((len(A),len(B)),dtype=object)
    counter = 0
    bins=[]
    labels=[]
    for i,a in enumerate(A):
        for j,b in enumerate(B):
            label=op(a,b)
            r[i,j] = np.array([counter, op(a,b) ],dtype=object) # op = ufunc in question
            bins=bins+[counter]
            labels=labels+[label]
            counter+=1
    return np.array(bins),labels#r, bins, labels


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


# # Sympy

# In[80]:


def sort_symbols(_chiSquare):
    return sorted(list( _chiSquare.free_symbols ),key=lambda s: s.name)

def ordered_point2sp_sub(mypoint,expression,debug=False):
    sorted_symbols = sort_symbols( expression )
    if debug:
        print('sorted_symbols ', colored(sorted_symbols, 'yellow'))
    
    myPoint={ symbol:mypoint[i] for i,symbol in enumerate( sorted_symbols )}
    return myPoint

def sp_replace(expression, point,debug=False):
    '''
    This functions facilitates the usage of a symbolic functions in a
    quicker format, closer to `f([0,1,2,3])`
    If one wants to use the names of the variables instead, it is always 
    possible to use the default sympy 
    `expression.subs({p0:1,p1:2,p3:3,p4:4})`
    '''
    return expression.subs( ordered_point2sp_sub(point,expression,debug=debug) )


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

# In[923]:


def read_file_to_lines(file_name):
    _xml_groups=[]
    file = open(file_name,"r")
    for line in file:
        _xml_groups.append(line)
    return _xml_groups


# In[924]:


def write_lines_to_file(mylines,filename,mode='a',final_line=False):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("%s" % item)
    if final_line:
        thefile.write("\n")      


# In[925]:


def write_lines_to_file_newline(mylines,filename,mode='a'):
    thefile = open(filename, mode=mode)
    for item in mylines:
          thefile.write("\n%s" % item)


# In[926]:


def filejson2dictionary(fn):
    with open(fn) as json_data:
        d = json.load(json_data)
    return d


# In[927]:


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

# def float_next_to(x,k):
#     try:
#         res=float(u.nextto(x,string=k))
#     except ValueError as e:
#         print('working on',k,'-->',e)
#         res=missing
#     return res 
    

def float_next_to(x,k,permissive=False,permissive_faulty_value=np.nan):
    try:
        res=float(nextto(x,string=k))
    except ValueError as e:
        print('working on',k,'-->',e)
        if not permissive:
            res=missing
        else:
            try:
                res=missing
            except NameError:
                res=permissive_faulty_value
    return res 

def filename_to_parameters(parameters,df,col):                      
    for k in parameters:
        df[str(k)]=df[col].apply(lambda x:  float_next_to(x,k)  ) 



def measurementFromString(s,err='±'):
    return list(map(lambda x: float(x), s.split(err) ) )


# In[928]:


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

# In[929]:


def sort_by_ith(data,i):
    return sorted(data, key=lambda tup: tup[i])


# In[67]:


def generate_patterns(pattern='squares',zero=0,small=1e-12,mid=1e-11,large =1e-10,names=['GeR','G3L'],make_plot=False,fine=False):
    
    def minus(x):
        if type(x)==str:
            return '-'+x
        else:
            return -x
        
    #diagonals 9 total
    diagonalPlus=[[minus(large),minus(large)],[minus(small),minus(small)],[zero,zero],[small,small],[large,large]]
    smallDiagonalMinus=[[minus(small),small],[small,minus(small)]]
    BigHorizontal=[[zero,large],[zero,minus(large)]]
    # squares 9 or 13 total
    SmallSquare=[[small,small],[minus(small),minus(small)],[small,minus(small)],[minus(small),small]]
    MidSquare=[[mid,mid],[minus(mid),minus(mid)],[mid,minus(mid)],[minus(mid),mid]]
    BigSquare=[[large,large],[minus(large),minus(large)],[large,minus(large)],[minus(large),large]]
    ###
    if pattern=='diagonals':
        res=diagonalPlus+smallDiagonalMinus+BigHorizontal            
    if pattern=='squares':
        res=[[zero,zero]]+SmallSquare+BigSquare
        if fine:
            res+=MidSquare
            
    ###
    _L = [ [{names[0]:a},{names[1]:b}] for a,b in res ] 
    #_L = utils.generate_patterns(zero=0,small=1e-12,mid=1e-11,large=1e-10,names=['x','y'])
    if make_plot:
        _d=pd.DataFrame([{  k:v    for _r in r  for k,v in _r.items()  } for r in _L ])
        plt.plot(names[0],names[1],'.',data=_d)
    return _L


# In[69]:


generate_patterns(zero=0,small=1e-11,large='1e-9',pattern='squares')


# In[930]:


def flattenOnce(tags_times):
    return [y for x in tags_times for y in x]


# In[931]:


def arange(a,b,s):
    return np.arange(a,b+s,s)
def linspace(start,stop,step):
    return np.linspace(start, stop, num=int((stop-start)/step)+1, endpoint=True)


# In[44]:


linspace(0,2,0.2)


# # Dictionaries

# In[932]:


def dict2string(dictio):
    res=[]
    for key,value in dictio.items():
        res.extend([str(value)])
    return "_".join(res)


# In[3]:


def retain_options_of_function(myfunct,optArgs,bonus=None):
    
    acceptable=[p.name for p in inspect.signature(myfunct).parameters.values()]
    if bonus is not None:
        acceptable=acceptable+bonus
    filtered_mydict = {k: v for k, v in optArgs.items() if k in acceptable}
    return filtered_mydict


# In[5]:


def multiply_dictionaries(params_list_of_dicts,DEBUG=False):
    first_result=[]
    for k,v in params_list_of_dicts[0].items():
            for _v in v:
                if DEBUG: print('adding ',k,':',_v)
                res={}
                res[k]=[_v]
                first_result.append(res)
    if DEBUG: print(first_result)
    last_result=first_result
    for r in range(len(params_list_of_dicts)-1): #each of the new dictionaries in the list
        # extend current with r-th component
        current_plus_rth_parameter=[]
        for k,v in params_list_of_dicts[1+r].items():
            for _v in v:
                for res_ith in range(len(last_result)):
                    current_plus_rth_parameter_vth_value=last_result[res_ith].copy()
                    if DEBUG: print('adding ',k,':',_v)
                    if DEBUG: print('adding to ',current_plus_rth_parameter_vth_value)
                    current_plus_rth_parameter_vth_value[k]=[_v]
                    if DEBUG: print('result ',current_plus_rth_parameter_vth_value) # print extended
                    current_plus_rth_parameter.append(current_plus_rth_parameter_vth_value)
                    if DEBUG: print(current_plus_rth_parameter)
        if DEBUG:  print('appending to last result')
        last_result=current_plus_rth_parameter
        if DEBUG:  print(last_result)

    return last_result


# In[6]:


def dictionary_outer(d1,d2):
    return np.array([ [ [ [ {k1:[v1[i1]], k2:[v2[i2]]} for i2 in range(len(v2)) ] for i1 in range(len(v1)) ]                       for k1,v1 in d1.items() ] for k2,v2 in d2.items() ]).flatten()


# In[ ]:


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# # Strings

# In[ ]:


def unprotect_string(s):
    return s.replace("\'",'')


# In[933]:


def remove_multiple_spaces(string):
    return re.sub(' +',' ',string)


# In[934]:


def ToString(x):
    return str(x)


# In[935]:


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


# #  Functions

# In[ ]:


def print_signature(chiSquare_lambdify):
    sig = signature( chiSquare_lambdify )
    print('signature ', colored(str(sig),'yellow'))
    return sig


# # Number manipulations

# In[ ]:


def logticks(basis=[1,2,5],orders=[-1.,-2.,-3.,-4.]):
    return np.array(list(map(lambda x: np.array(basis)*np.power(10,x),np.array(orders) ))).flatten()


# In[936]:


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


# In[937]:


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


# # LHE

# ```python
# import lhef, math
# LHEfile=lhef.readLHE("unweighted_events.lhe")
# nprinted=0
# debug=False
# nPrint=11
# costheta_values=[] # create a vector where to store the computed values of costheta
# abs_costheta_values=[] # and one for the abs
# theta_values=[] # and one for the abs
# 
# for e in LHEfile: # loop on the events
#     for p in e.particles: # loop on the particles of each event
#         if p.status == 1 and p.id == 24: # check it is a final state and is a Chi+
#             lv=p.fourvector() # make four vector
#             obs=lv.cosTheta() # obtain the cosTheta
#             costheta_values.append(obs) # append it to the vector of results
#             abs_costheta_values.append(math.fabs(obs)) # and the abs(cosTheta)
#             theta=lv.theta() # obtain the theta angle
#             theta_values.append(theta)
#             if nprinted <nPrint: 
#                 if debug: print(p.px, p.py, p.pz, p.e)
#                 if debug: print(obs)
#                 nprinted+=1
# 
# plt.hist(costheta_values,bins=np.linspace(-1,1,num=50),density=True)
# plt.title(r"$W^+$ $\cos\theta$ Distribution")
# plt.xlabel(r"$\cos\theta^\star$")
# plt.ylabel(r"$d\sigma/d\cos\theta$")
# plt.show()
# ```

# In[ ]:




