import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
import glob
import os
from typing import Tuple
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import pathlib
import json
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from scipy.spatial import ConvexHull
import pandas as pd 


def getFileList(path:str, extension:str) -> list:
    """Returns the files in a given path 
    with a given extension. 

    Args:
        path (str): path string
        extension (str): the extension of files

    Returns:
        list: list of file paths (string values)
    """
    fileList = []
    for fn in os.listdir(path):
        if extension in fn:
            fileList.append('{0}/{1}'.format(path, fn))
    fileList.sort()
    return fileList


def gammaAdjust(im:np.ndarray, gamma:float) -> np.ndarray:
    """Adjust the gamma of an image

    Args:
        im (numpy.ndarray): opnecv image
        gamma (float): gamma value

    Returns:
        numpy.ndarray: opencv imge
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(im, table)


def rgb2hsv0(img:np.ndarray) -> np.ndarray:
    """Fast rgb to hsv conversion using numpy arrays

    Args:
        img (numpy.ndarray): numpy array representation of cv2 RGB image

    Returns:
        numpy.ndarray: HSV image packed in 3D numpy array
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0] * 2
    s = hsv[:,:,1] / 2.55
    v = hsv[:,:,2] / 2.55
    return np.array([h, s, v]).transpose(1, -1, 0)


def hsvFilter(img:np.ndarray, hSlice=(0,360), sSlice=(0,100), vSlice=(0,100)) -> np.ndarray:
    imm  = img.copy()
    hsv = rgb2hsv0(img)
    if hSlice[0] != 0 or hSlice[1] != 360:
        mask = np.logical_and((hsv[...,0] >= hSlice[0]), (hsv[...,0] < hSlice[1]))
        imm[np.logical_not(mask)] = 0
    if sSlice[0] != 0 or sSlice[1] != 100:
        mask = np.logical_and((hsv[...,1] >= sSlice[0]), (hsv[...,1] < sSlice[1]))
        imm[np.logical_not(mask)] = 0
    if vSlice[0] != 0 or vSlice[1] != 100:
        mask = np.logical_and((hsv[...,2] >= vSlice[0]), (hsv[...,2] < vSlice[1]))
        imm[np.logical_not(mask)] = 0
    return imm
    


def hsvHistogramScatter(img, bins:list) -> dict:
    """Gets RGB image, converts it to HSV, bins coordinates
    and returns this binned coordinate list in a dict.
    This function generates histograms for fast plotting.

    Args:
        img (numpy.ndarray): cv2 image
        bins (list): bin size list in the form of [h, s, v]

    Returns:
        dict: keys->3D coordinates as tuples, values->counts
    """
    hsv = rgb2hsv0(img)
    hsvb = np.floor(hsv / np.array(bins)) * np.array(bins)
    hsvScatter = {}
    for i in range(0, hsvb.shape[0]):
        for j in range(0, hsvb.shape[1]):
            hsvc = (hsvb[i,j,0], hsvb[i,j,1], hsvb[i,j,2])
            if hsvc not in hsvScatter:
                hsvScatter[hsvc] = 1
            else:
                hsvScatter[hsvc] = hsvScatter[hsvc] +1
    return hsvScatter


def hsvHistogramCubic(img, bins:list) -> np.ndarray:
    """Generates 3D HSV histogram of an image. Given HSV ranges are binned and 
    the histogram is returned as a matrix in the form of 3D numpy array.
    This form of histogram is suitable for distance matrix calculations by using EMD

    Args:
        img (numpy.ndarray): cv2 image
        bins (list): bin size list for h, s and v components
        
    Returns:
        np.ndarray: 3D histogram matrix
    """
    hRange = (0,360)
    sRange = (0,100)
    vRange = (0, 20)
    hsv = rgb2hsv0(img)
    hbins = int((hRange[1] - hRange[0]) / bins[0])
    sbins = int((sRange[1] - sRange[0]) / bins[1])
    vbins = int((vRange[1] - vRange[0]) / bins[2])
    histogram = [[[0 for v in range(0, vbins)] for s in range(0, sbins)] for h in range(0, hbins)]
    hsvb = np.floor(hsv / np.array(bins))
    for i in range(0, hsvb.shape[0]):
        for j in range(0, hsvb.shape[1]):
            h = int(hsvb[i,j,0])
            if h >= hbins:
                h -= 1
            s = int(hsvb[i,j,1])
            if s >= sbins:
                s -= 1
            v = int(hsvb[i,j,1])
            if v >= vbins:
                v -= 1
            histogram[h][s][v] += 1
    hr = [h for h in range(hRange[0], hRange[1], bins[0])]
    sr = [s for s in range(sRange[0], sRange[1], bins[1])] 
    vr = [v for v in range(vRange[0], vRange[1], bins[2])]
    return (np.array(histogram), hr, sr, vr)


def plotHistogram3D(hist:dict, axs:mpl.axes, rot=(10,0)) -> (list, list, list):
    hc = []
    sc = []
    vc = []
    cl = []
    cn = []
    for k,v in hist.items():
        hc.append(k[0])
        sc.append(k[1])
        vc.append(k[2])
        cl.append(k[0]*100/360)
        cn.append(math.log(v)*32)
    axs.scatter(hc, sc, vc, s=cn, c=cl, cmap='hsv')
    axs.set_xlabel('hue', fontsize=20, rotation=0)
    axs.set_ylabel('saturation', fontsize=20, rotation=0)
    axs.set_zlabel('value', fontsize=20, rotation=60)
    axs.view_init(*rot)
    return (hc, sc, vc)


def filterScatterHist(scatterHist:dict, hSlice=(0,360), sSlice=(0,255), vSlice=(0,255), cSlice=(0,1000000000)) -> dict:
    outHist = {}
    for k,v in scatterHist.items():
        if k[0] >= hSlice[0] and k[0] < hSlice[1] and k[1] >= sSlice[0] and k[1] < sSlice[1] and k[2] >= vSlice[0] and k[2] < vSlice[1]:
            if v >= cSlice[0] and v < cSlice[1]:
                outHist[k] = v
    return outHist


def hist3Dpanel(img:np.ndarray, fig:Figure, bins=[18,5,1]) -> dict:
    axs = fig.add_subplot(221)
    axs.imshow(gammaAdjust(img, 2.5))
    hsv = hsvHistogramScatter(img, bins)
    axs = fig.add_subplot(224, projection='3d')
    plotHistogram3D(hsv, axs, rot=(5, 45+270))
    axs = fig.add_subplot(222, projection='3d')
    plotHistogram3D(hsv, axs, rot=(5, 270))
    axs = fig.add_subplot(223, projection='3d')
    plotHistogram3D(hsv, axs, rot=(5, 0))
    plt.show()
    return hsv


def convexHull(scatterHist:dict, fig:Figure, pointSize=(18,5,1), hSlice=(0,360), sSlice=(0,255), vSlice=(0,255), cmin=0) -> (float, float):
    h = filterScatterHist(scatterHist, hSlice, sSlice, vSlice, cSlice=(cmin, 1E12))
    coords = []
    pixels = 0
    for k,v in h.items():
        coords.append(list(k))
        k2 = []
        for i, j in zip(k, pointSize):
            k2.append(i+j)
        coords.append(k2)
        pixels += v
    ch = ConvexHull(np.array(coords), qhull_options='QJ')
    axs = fig.add_subplot(221, projection='3d')
    hc, sc, vc = plotHistogram3D(h, axs, rot=(0,270))
    axs.plot(hc, sc, vc)
    axs = fig.add_subplot(222, projection='3d')
    plotHistogram3D(h, axs, rot=(0,0))
    axs = fig.add_subplot(223, projection='3d')
    plotHistogram3D(h, axs, rot=(90,270))
    axs = fig.add_subplot(224, projection='3d')
    plotHistogram3D(h, axs, rot=(10,300))
    return (ch.volume, pixels)


def batchConvexHull(scatterHistSet:dict, outdir:str, pointSize=(18,5,1), hSlice=(0,360), sSlice=(0,255), vSlice=(0,255), cmin=0) -> pd.DataFrame:
    volNdist = []
    for k,v in scatterHistSet.items():
        fig = plt.figure()
        fig.set_size_inches(w=24, h=24)
        ch, p = convexHull(v, fig, pointSize, hSlice, sSlice, vSlice, cmin=10)
        fig.suptitle('{0}   Vch:{1}   pix:{2}'.format(k, ch, p), fontsize=32)
        fig.savefig('{0}/{1}_convexHull.png'.format(outdir, k))
        plt.show()
        plt.close(fig)
        volNdist.append({'name': k, 'V_ch': ch, 'pixels': p})
    return pd.DataFrame(volNdist)


def processImage(fname:str, outpath:str, bins=[18,5,1]) -> (dict, np.ndarray):
    im = cv2.imread(fname)
    imgtitle = '{0}'.format(pathlib.Path(fname).stem)
    print('processing: {0}'.format(imgtitle))
    fig:Figure = plt.figure()
    fig.set_size_inches(w=24, h=24)
    fig.suptitle(imgtitle, fontsize=32)
    scatterHist = hist3Dpanel(im, fig, bins)
    fig.savefig('{0}/{1}_histograms.png'.format(outpath, imgtitle))
    plt.show()
    plt.close(fig)
    cubicHist, hc, sc, vc = hsvHistogramCubic(im, bins)
    return cubicHist, scatterHist


def batchProcessImages(sourcepath:str, ext:str, outpath:str) -> dict:
    hhCubic = {}
    hhScatter = {}
    for f in getFileList(sourcepath, ext):
        print(f)
        cubicHist, scatterHist = processImage(f, outpath)
        hhCubic[pathlib.Path(f).stem] = cubicHist
        hhScatter[pathlib.Path(f).stem] = scatterHist
    return hhCubic, hhScatter


def emd(hist1:np.ndarray, hist2:np.ndarray) -> float:
    """Calculates earth movers distance between two histograms
    (either 1D or 2D). If the histograms are in 2D, they are flattened.
    Wasserstein distance between the histograms is returned. 

    Args:
        hist1 (numpy.ndarray): Histogram1 (1D/2D)
        hist2 (numpy.ndarray): Histogram2 (1D/2D)

    Returns:
        float: Wasserstein distance
    """
    if len(hist1.shape) > 1:
        hist1 = hist1.flatten()
    if len(hist2.shape) > 1:
        hist2 = hist2.flatten()
    return wasserstein_distance(hist1, hist2)


def pairwiseEmd(histograms:list) -> list:
    matrix = []
    for i in range(len(histograms)):
        for j in range(i+1, len(histograms)):
            wd = emd(histograms[i], histograms[j])
            matrix.append(wd)
    return matrix


def heatmap(histograms:dict, outpath:str):
    names = []
    hists = []
    for k,v in histograms.items():
        names.append(k)
        hists.append(v)
    e = pairwiseEmd(hists)
    Y = linkage(e, method='single', metric='braycurtis')
    sqdist = squareform(e)
    # heatmap
    fig = plt.figure()
    hmsize = len(names)
    fig.set_size_inches(w=hmsize, h=hmsize)
    leftDendAxs = fig.add_axes([0, 0, 0.15, 0.9]) # [left, bottom, width, height]
    Z = dendrogram(Y, labels=names, orientation='left')
    ssqdist = sqdist[:,Z['leaves']]
    ssqdist = ssqdist[Z['leaves'],:]
    rightHeatmapAxs = fig.add_axes([0.15, 0, 0.9, 0.9])
    rightHeatmapAxs.matshow(ssqdist, origin='bottom', cmap='afmhot')
    rightHeatmapAxs.set_xticklabels(['']+Z['ivl'], rotation=90, size='30')
    rightHeatmapAxs.yaxis.set_ticks_position('right')
    rightHeatmapAxs.set_yticklabels(['']+Z['ivl'], size='30')
    rightHeatmapAxs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    rightHeatmapAxs.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.savefig('{0}/heatmap.png'.format(outpath))
    plt.show()
    plt.close(fig)
    # dendrogram
    fig = plt.figure()
    fig.set_size_inches(w=hmsize, h=hmsize/2)
    Z = dendrogram(Y, labels=names, leaf_font_size=30, leaf_rotation=90)
    fig.savefig('{0}/dendrogram.png'.format(outpath))
    plt.show()
    plt.close(fig)
    