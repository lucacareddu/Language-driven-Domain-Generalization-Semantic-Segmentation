import matplotlib.pyplot as plt
import numpy as np

from utils.colors import CITY_PALETTE


def visualize(img, lbl, pred=None, fullscreen=True):
    img = img.transpose(2,1,0) * 255
    lbl = lbl.transpose()
    color_lbl = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)

    if pred is not None:
        pred = pred.transpose()
        color_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(CITY_PALETTE):
        color_lbl[lbl == label, :] = color
        if pred is not None:
            color_pred[pred == label, :] = color

    fig, axs = plt.subplots(2,2) if pred is not None else plt.subplots(1,2)

    if fullscreen:
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    if pred is not None:
        axs[0,0].imshow(np.uint8(img))
        axs[0,1].imshow(color_lbl)
        axs[1,0].imshow(color_pred)
        
        res = np.array(img) * 0.5 + color_pred * 0.5
        axs[1,1].imshow(np.uint8(res))
    else:
        axs[0].imshow(np.uint8(img))
        axs[1].imshow(color_lbl)

    plt.show()
