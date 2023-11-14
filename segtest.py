import gc
import os
import time

from skimage.morphology import local_maxima, remove_small_objects, remove_small_holes
import glob

from numpy import zeros, dot, ix_, array
from pyfem.util.shapeFunctions import getElemShapeData
import scipy.linalg
import scipy
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import measure, filters
from PointCloudExtract import PointCloud
from PIL import Image
from itertools import product
import objgraph
from memory_profiler import profile



# ------------

def getDofs(nodes):
    n = 3 * len(nodes)

    dofs = zeros(n, dtype=int)

    dofs[0:n:3] = 3 * nodes
    dofs[1:n:3] = 3 * nodes + 1
    dofs[2:n:3] = 3 * nodes + 2

    return dofs


# --------------------

def getBmatrix(dhdx):
    n = 3 * len(dhdx)

    B = zeros(shape=(6, n))

    for i, dp in enumerate(dhdx):
        B[0, i * 3] = dp[0]
        B[1, i * 3 + 1] = dp[1]
        B[2, i * 3 + 2] = dp[2]
        B[3, [i * 3, 3 * i + 1]] = [dp[1], dp[0]]
        B[4, [i * 3 + 1, 3 * i + 2]] = [dp[2], dp[1]]
        B[5, [i * 3, 3 * i + 2]] = [dp[2], dp[0]]

    return B

# @profile()
def solveconstrainEqModify(preIndex, K, Fext, Fin):
    v_size = int(K.shape[0] / 3)

    currv = np.zeros((v_size,))
    currv[preIndex] = 1
    Index_s = []
    for ii in range(v_size):
        if currv[ii] == 0:
            Index_s.append([ii])
    print(1)
    Indexs0 = (np.array(Index_s)).reshape(len(Index_s), )
    Indexs1 = np.zeros((3 * len(Indexs0),), dtype=int)
    Indexs1[0:3 * len(Indexs0):3] = 3 * Indexs0
    Indexs1[1:3 * len(Indexs0):3] = 3 * Indexs0 + 1
    Indexs1[2:3 * len(Indexs0):3] = 3 * Indexs0 + 2
    Indexs1 = Indexs1.tolist()
    b = Fext[Indexs1] - Fin[Indexs1]
    K = scipy.sparse.csr_matrix(K).copy()
    KK = K[Indexs1, :][:, Indexs1]
    KK = scipy.sparse.csr_matrix(KK)
    # aa = scipy.linalg.solve(np.dot(KK.T, KK), np.dot(KK.T, b.T))
    aa = scipy.sparse.linalg.spsolve(KK, b)
    sol = np.zeros((3 * v_size,))
    sol[Indexs1] = aa
    del K, KK
    gc.collect()
    return sol, aa


def calculateK(Nodeset, Elements, D):
    totDof = 3 * len(Nodeset)
    try:
        K = np.zeros((totDof, totDof), dtype=np.float64)
    except:
        K = scipy.sparse.lil_matrix((totDof, totDof), dtype=np.float64)

    for elemNodes in Elements:
        elemDofs = getDofs(elemNodes)
        sData = getElemShapeData(Nodeset[elemNodes, :])
        for iData in sData:
            b = getBmatrix(iData.dhdx)
            Kint = dot(b.transpose(), dot(D, b)) * iData.weight
            K[ix_(elemDofs, elemDofs)] += Kint

        # del sData
        # gc.collect()

    return K


def calculatefint(K, Nodeset, Elements, D, sol):
    totDof = K.shape[0]
    fint = zeros(totDof)
    nodalStress = zeros(shape=(len(Nodeset), 6))
    nodalCount = zeros(len(Nodeset))
    Nodeset1 = np.zeros((Nodeset.shape), dtype=float)
    for i in range(Nodeset.shape[0]):
        Nodeset1[i, 0] = float(Nodeset[i, 0]) + sol[3 * i]
        Nodeset1[i, 1] = float(Nodeset[i, 1]) + sol[3 * i + 1]
        Nodeset1[i, 2] = float(Nodeset[i, 2]) + sol[3 * i + 2]
    for elem in Elements:
        elemDofs = getDofs(elem)
        sData = getElemShapeData(Nodeset[elem, :])
        for iData in sData:
            b = getBmatrix(iData.dhdx)
            strain = dot(b, sol[elemDofs])
            stress = dot(D, strain)
            fint[elemDofs] += dot(b.transpose(), stress) * iData.weight
            nodalStress[elem, :] += stress
            nodalCount[elem] += 1
    return fint, Nodeset1



# @profile()
def do_loop(BinaryImg, dispmap):
    PointCloud1 = PointCloud(BinaryImg)
    PointCloud1.setNodeset()
    PointCloud1.setBoudnum()
    PointCloud1.setElements()
    PointCloud1.setUppernum()
    Nodeset = PointCloud1.Nodeset
    Nodeset = np.hstack((np.arange(0, Nodeset.shape[0], 1, dtype=int).reshape(-1, 1), Nodeset))
    Elements = PointCloud1.Elements
    preIndex = Nodeset[PointCloud1.Boudnum][:, 0].T
    foreIndex = Nodeset[::ModelHeight, 0]
    # Nodeset, Elements, Boudnode, ImaL, Imab = Grid.traceRegionBoundaryM(BinaryImg)
    Nodeset = Nodeset[:, [1, 2, 3]]

    D = np.zeros((6, 6))
    nu = 0.05
    E = 2.e4

    D[0, :] = [1 - nu, nu, nu, 0, 0, 0]
    D[1, :] = [nu, 1 - nu, nu, 0, 0, 0]
    D[2, :] = [nu, nu, 1 - nu, 0, 0, 0]
    D[3, 3] = (1 - 2 * nu) / 2
    D[4, 4] = (1 - 2 * nu) / 2
    D[5, 5] = (1 - 2 * nu) / 2
    D = (E / ((1 + nu) * (1 - 2 * nu))) * D

    # Calculate K
    print(1)
    time1 = time.time()
    print(Nodeset.shape[0], Nodeset.shape[1])
    if Nodeset.shape[0] > 480000:
        print('too large area input at {}'.format(Nodeset.shape[0]))
        return
    K = calculateK(Nodeset, Elements, D)
    # fixnode = Boudnode.disp
    # fixforce = Boudnode.force
    # preIndex = (fixnode[:, 0]).T
    # foreIndex = (fixforce[:, 0]).T
    Fext0 = np.zeros((3, int(K.shape[0] / 3)))

    Fext0[2, foreIndex] = 80
    Fin = np.zeros((3, int(K.shape[0] / 3)))

    # Fext = Fext.reshape(K.shape[0],1,order='F')
    Fext = np.zeros((K.shape[0]))
    for i in range(int(K.shape[0] / 3)):
        Fext[3 * i] = Fext0[0, i]
        Fext[3 * i + 1] = Fext0[1, i]
        Fext[3 * i + 2] = Fext0[2, i]

    Fin = Fin.reshape(K.shape[0], )
    print(2)
    time2 = time.time()
    sol, aa = solveconstrainEqModify(preIndex, K, Fext, Fin)
    print(3)
    time3 = time.time()
    fint, Nodeset1 = calculatefint(K, Nodeset, Elements, D, sol)
    dispmap[np.int16(np.round(Nodeset[::3, 0])), np.int16(np.round(Nodeset[::3, 1]))] = Nodeset1[::3, 2]
                                                                                        #* 255 // np.max(Nodeset1[::3, 2])
    print(ii)
    del K
    gc.collect()
    time4 = time.time()
    gc.collect()
    print(
        'num={}, \t set={}, \t modeling time={}, cal K time = {}, solving time = {}, final time = {}, tot time = {}'.format(
            num, Labeledarea.shape[0], time1 - time0, time2 - time1, time3 - time2, time4 - time3, time4 - time1))


def get_img_info(PicList, Filemask, pic_num):
    if Filemask == '/*.npy':
        temp_img = np.load(PicList[pic_num])
    else:
        temp_img = cv2.imread(PicList[pic_num], cv2.IMREAD_UNCHANGED)
    return temp_img.shape[0], temp_img.shape[1]

def pre_process(I, area_threshold=60): #default param for cellular
    if I.ndim == 3:
        I = I[:, :, 0]

    I = np.float64(I)
    I[I >= 0.5] = 1
    I[I < 0.5] = 0
    I[:, 0] = 0
    I[:, -1] = 0
    I[0, :] = 0
    I[-1, :] = 0
    I[I > 1] = 1
    kernal = np.ones((3, 3), dtype=np.uint16)
    kernal[1, 1] = 6
    I = cv2.filter2D(I, -1, kernal)
    I = (I >= 9).astype(np.uint8)
    I = cv2.filter2D(I, 0, kernal)
    I = (I >= 9).astype(np.uint8)
    kernal = np.array([[0, 1, 0], [1, 6, 1], [0, 1, 0]])
    I = cv2.filter2D(I, 0, kernal)
    I = (I >= 8).astype(np.uint8)
    # I = np.int32(remove_small_holes(I, area_threshold=area_threshold))
    return I

if __name__ == '__main__':
    for dir_num in range(1):
        FilePath = r'/home/fty/picture/plantcell/result'
        try:
            os.makedirs(FilePath + '/result/seg3')
            os.makedirs(FilePath + '/result/vis3')
        except FileExistsError:
            print('Path exits')

        FileMask = r'/*.png'
        PicList = sorted(glob.glob(FilePath+FileMask))
        if not os.getenv('LD_LIBRARY_PATH'):
            os.environ['LD_LIBRARY_PATH'] = '/usr/local/MATLAB/MATLAB_Runtime/R2023a/runtime/glnxa64'

        import marker_watershed
        mark_watsh = marker_watershed.initialize()

        MapPath = FilePath + '/prob_map'
        MapMask = r'/*.png'
        MapList = sorted(glob.glob(MapPath+MapMask))

        m, n = get_img_info(PicList, FileMask, pic_num=0)
        # m = n = 500
        ModelHeight = 3
        BinaryImg = np.zeros((m, n, ModelHeight+1))
        dispmap = np.zeros((m, n))
        Labeledimg = np.zeros((m, n), dtype=np.int32)

        ii = 0
        pic_num = 0
        # for pic_num, ii in product(range(2), range(10)):
        while True:
            print(ii)
            num = ii + 1
            ii += 1
            if pic_num == len(PicList):
                break

            if num == 1:
                # ##############for CPM only
                # forePath = '/home/fty/picture/CPM_semantic/cpm17/test/Images'
                # foreMask = '/*.png'
                # foreList = sorted(glob.glob(forePath+foreMask))
                # fore = cv2.imread(foreList[pic_num], cv2.IMREAD_UNCHANGED)
                # if fore.shape[0] != 600:
                #     ii = 0
                #     pic_num += 1
                #     continue
                # ##############

                if FileMask == '/*.npy':
                    I = np.load(PicList[pic_num])
                else:
                    I = cv2.imread(PicList[pic_num], cv2.IMREAD_UNCHANGED)
                # if pic_num == 28:
                #     I = cv2.imread('/home/fty/picture/EM_ISBI_2012/3d_map/seg/prob_map/output28.png')
                #     I[I < 130] = 0
                #     I[I > 0] = 255
                # I = Image.fromarray(I)
                # I = I.resize((500, 500), resample=Image.ANTIALIAS)
                # I = np.array(I, dtype=np.uint8)
                # I[I > 0] = 1

                # assert I.shape[0] == m and I.shape[1] == n
                dispmap = np.zeros((I.shape[0], I.shape[1]))
                I = pre_process(I, area_threshold=300)
                dispmap[:] = np.zeros((I.shape[0], I.shape[1]))
                Labeledimg[:] = remove_small_holes(I, 50)
                Labeledimg[:] = measure.label(I, 0, connectivity=1)
                # Labeledimg[:] = remove_small_objects(Labeledimg, min_size=20)

            if num > np.max(Labeledimg):
                ii = 0
                Filename = PicList[pic_num].split('/')[-1][:-4]
                cv2.imwrite(FilePath+'/result/seg3/'+Filename+'origin_dispmap.png', dispmap)
                np.save(FilePath+'/result/seg3/'+Filename+'origin_dispmap.npy', dispmap)
                Peaks = np.array(np.where(local_maxima(dispmap, connectivity=2) == True)).T
                np.save(FilePath + '/result/seg3/' + Filename + '_peaks.npy', np.int16(Peaks))
                if pic_num == 162:
                    Peaks = np.vstack((Peaks, np.array([240, 52])))
                pic_wh = np.min((I.shape[0], I.shape[1]))
                # I = np.ascontiguousarray(I[:pic_wh, :pic_wh])
                ##################EM only
                # prob_map = cv2.imread(MapList[pic_num])
                # if prob_map.ndim == 3:
                #     prob_map = prob_map[:pic_wh, :pic_wh, 1]
                # I = I * prob_map.copy()
                #################


                seg = mark_watsh.marker_watershed(I, Peaks)


                seg = np.array(seg, dtype=np.int16)
                seg = remove_small_objects(seg, min_size=40, connectivity=1)
                cv2.imwrite(FilePath+'/result/seg3/'+Filename+'.png', np.int16(seg))
                np.save(FilePath+'/result/seg3/'+Filename+'_seg.npy', np.int16(seg))
                plt.figure()
                plt.imshow(seg)
                plt.plot(Peaks[:, 1], Peaks[:, 0], '.')
                # plt.show()
                plt.savefig(FilePath+'/result/vis3/'+Filename+'.png')
                plt.close('all')
                print('next slide')
                pic_num += 1


            Labeledarea = np.array(np.where(Labeledimg == num)).T
            if Labeledarea.shape[0] == 0:
                continue

            BinaryImg[:] = np.zeros((m, n, ModelHeight+1))
            BinaryImg[Labeledarea[:, 0], Labeledarea[:, 1], :ModelHeight] = 1
            time0 = time.time()
            do_loop(BinaryImg, dispmap)

                # objgraph.show_growth()
                # objgraph.show_most_common_types(limit=50)

            # if num == np.max(Labeledimg):
            #     ii = 0
            #     Peaks = np.array(np.where(local_maxima(dispmap, connectivity=2) == True)).T
            #     Peaks = Peaks[:, [-1, 0]]
            #     pic_wh = np.min((I.shape[0], I.shape[1]))
            #     I = np.ascontiguousarray(I[:pic_wh, :pic_wh])
            #     seg = mark_watsh.marker_watershed(I, Peaks)
            #     seg = np.array(seg, dtype=np.int16)
            #     seg = remove_small_objects(seg, min_size=8, connectivity=1)
            #     Filename = PicList[pic_num].split('/')[-1][:-4]
            #     cv2.imwrite(FilePath+'/result/seg1/'+Filename+'.png', np.int16(seg))
            #     plt.figure()
            #     plt.imshow(seg)
            #     plt.plot(Peaks[:, 1], Peaks[:, 0], '.')
            #     # plt.show()
            #     plt.savefig(FilePath+'/result/vis1/'+Filename+'.png')
            #     plt.close('all')
            #     print('next slide')
            #     pic_num += 1
        mark_watsh.terminate()
        # cv2.imwrite('/home/fty/picture/plot/0807/dispmap.png', dispmap)






