import cv2
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool

def create_spatialglimpse_function(img_h=480, img_w=640, fovH=64, fovW=64):
    fovHalfH = fovH / 2
    fovHalfW = fovW / 2
    glimpseInpImg = T.dtensor3('glimpseInpImg')
    glimpseInpLoc_y, glimpseInpLoc_x = T.dscalars('gilY', 'gilX') # each lies between -1 and 1
    glimpseLocOnImg_y = T.cast(((glimpseInpLoc_y + 1) / 2.0) * img_h, 'int32')
    glimpseLocOnImg_x = T.cast(((glimpseInpLoc_x + 1) / 2.0) * img_w, 'int32')

    y1 = T.max((glimpseLocOnImg_y - fovHalfH, 0))
    y2 = T.min((glimpseLocOnImg_y + fovHalfH, img_h))
    x1 = T.max((glimpseLocOnImg_x - fovHalfW, 0))
    x2 = T.min((glimpseLocOnImg_x + fovHalfW, img_w))

    y3 = T.max((glimpseLocOnImg_y - fovH, 0))
    y4 = T.min((glimpseLocOnImg_y + fovH, img_h))
    x3 = T.max((glimpseLocOnImg_x - fovW, 0))
    x4 = T.min((glimpseLocOnImg_x + fovW, img_w))

    y5 = T.max((glimpseLocOnImg_y - 2*fovH, 0))
    y6 = T.min((glimpseLocOnImg_y + 2*fovH, img_h))
    x5 = T.max((glimpseLocOnImg_x - 2*fovW, 0))
    x6 = T.min((glimpseLocOnImg_x + 2*fovW, img_w))

    glimpse1= glimpseInpImg[:, y1:y2, x1:x2]
    if T.lt(glimpse1.shape[1], fovH):
        pad = T.zeros((glimpse1.shape[0], fovH - glimpse1.shape[1], glimpse1.shape[2]))
        if T.eq(y1, 0):
            glimpse1 = T.concatenate((pad, glimpse1), 1)
        else:
            glimpse1 = T.concatenate((glimpse1, pad), 1)
    if T.lt(glimpse1.shape[2], fovW):
        pad = T.zeros((glimpse1.shape[0], glimpse1.shape[1], fovW - glimpse1.shape[2]))
        if T.eq(x1, 0):
            glimpse1 = T.concatenate((pad, glimpse1), 2)
        else:
            glimpse1 = T.concatenate((glimpse1, pad), 2)

    glimpse2 = glimpseInpImg[:, y3:y4, x3:x4]
    if T.lt(glimpse2.shape[1], 2*fovH):
        pad = T.zeros((glimpse2.shape[0], 2*fovH - glimpse2.shape[1], glimpse2.shape[2]))
        if T.eq(y3, 0):
            glimpse2 = T.concatenate((pad, glimpse2), 1)
        else:
            glimpse2 = T.concatenate((glimpse2, pad), 1)
    if T.lt(glimpse2.shape[2], 2*fovW):
        pad = T.zeros((glimpse2.shape[0], glimpse2.shape[1], 2*fovW - glimpse2.shape[2]))
        if T.eq(x3, 0):
            glimpse2 = T.concatenate((pad, glimpse2), 2)
        else:
            glimpse2 = T.concatenate((glimpse2, pad), 2)

    glimpse2 = T.signal.pool.pool_2d(glimpse2, (2, 2), ignore_border=True, mode='average_exc_pad')

    glimpse3 = glimpseInpImg[:, y5:y6, x5:x6]
    if T.lt(glimpse3.shape[1], 4*fovH):
        pad = T.zeros((glimpse3.shape[0], 4*fovH - glimpse3.shape[1], glimpse3.shape[2]))
        if T.eq(y5, 0):
            glimpse3 = T.concatenate((pad, glimpse3), 1)
        else:
            glimpse3 = T.concatenate((glimpse3, pad), 1)
    if T.lt(glimpse3.shape[2], 4*fovW):
        pad = T.zeros((glimpse3.shape[0], glimpse3.shape[1], 4*fovW - glimpse3.shape[2]))
        if T.eq(x5, 0):
            glimpse3 = T.concatenate((pad, glimpse3), 2)
        else:
            glimpse3 = T.concatenate((glimpse3, pad), 2)
    glimpse3 = pool.pool_2d(glimpse3, (4, 4), ignore_border=True, mode='average_exc_pad')

    glimpse1 = T.cast(glimpse1, 'uint8')
    glimpse2 = T.cast(glimpse2, 'uint8')
    glimpse3 = T.cast(glimpse3, 'uint8')

    fun = theano.function([glimpseInpImg, glimpseInpLoc_y, glimpseInpLoc_x], [glimpse1, glimpse2, glimpse3])
    return fun


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('1.jpg'), cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    func = create_spatialglimpse_function()
    l1, l2 = (np.random.rand(2) - 0.5) * 2
    g1, g2, g3 = func(img, l1, l2)

    p = g1.transpose(1, 2, 0)
    q = g2.transpose(1, 2, 0)
    r = g3.transpose(1, 2, 0)

    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(p)
    plt.subplot(1, 3, 2)
    plt.imshow(q)
    plt.subplot(1, 3, 3)
    plt.imshow(r)
    plt.show()
