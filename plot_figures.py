#############################
#    Author: Xuejing Lei    #
# Modified by: Zohreh Azizi #
#############################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_images(gen_img_all,h,w,label="",figsize=(10,10),show=True,save=False,fdir='./figures/',fn_pre="img"):

    if not os.path.exists(fdir):
        os.makedirs(fdir)
    N = len(gen_img_all)
    tt_num = int(N/w/h)
    if N % (w*h) != 0:
        tt_num += 1
    for tt in range(tt_num):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(label)
        fig.tight_layout()
        if tt<int(N/w/h):
            for idx, i in enumerate(range(w*h)):
                fig.add_subplot(h, w, idx+1)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
                plt.margins(0,0)
                if gen_img_all.shape[-1]==1 or len(gen_img_all.shape)==3:
                    plt.imshow(gen_img_all[tt*w*h+i].squeeze(), cmap='gray')# , cmap='gray')
                else:
                    plt.imshow(gen_img_all[tt*w*h+i])
        elif not tt*w*h == N:
            for idx, i in enumerate(range(N-tt*w*h)):
                fig.add_subplot(h, w, idx+1)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
                plt.margins(0,0)
                if gen_img_all.shape[-1]==1 or len(gen_img_all.shape)==3:
                    plt.imshow(gen_img_all[tt*w*h+i].squeeze(), cmap='gray')# , cmap='gray')
                else:
                    plt.imshow(gen_img_all[tt*w*h+i])
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        if show:
            plt.show()
        if save:
            fig.savefig(os.path.join(fdir,"{}_{}.jpg".format(fn_pre,tt)))
        plt.close()
