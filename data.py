#############################
#    Author: Xuejing Lei    #
# Modified by: Zohreh Azizi #
#############################

import h5py
import pickle
import numpy as np
import os
import cv2
import platform
import matplotlib.pyplot as plt
import matplotlib
import sys

def read_data(path, fn):
    filename = os.path.join(path,fn)
    with open(filename, 'rb') as f:
        prefix = f.read(2)
        d_type = f.read(1).decode('utf8')
        dim = int.from_bytes(f.read(1),byteorder='big')
        dim_list = tuple(int.from_bytes(f.read(4),byteorder='big') for d in range(dim))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(dim_list)

def import_data2(dsdir, name, patch_size=None, patch_stride=None, gray=False, shuffle_index = None, num_total_train=None, use_num_images=None):
    if name=="mnist":
        train_images = read_data(dsdir+name+'/','train-images.idx3-ubyte')
        train_labels = read_data(dsdir+name+'/','train-labels.idx1-ubyte')
        test_images = read_data(dsdir+name+'/','t10k-images.idx3-ubyte')
        test_labels = read_data(dsdir+name+'/','t10k-labels.idx1-ubyte')

        train_images=train_images/255.
        test_images=test_images/255.

        val_images=None

    elif name=='fashion-mnist':
        train_images = read_data(dsdir+name+'/','train-images-idx3-ubyte')
        train_labels = read_data(dsdir+name+'/','train-labels-idx1-ubyte')
        test_images = read_data(dsdir+name+'/','t10k-images-idx3-ubyte')
        test_labels = read_data(dsdir+name+'/','t10k-labels-idx1-ubyte')

        train_images=train_images/255.
        test_images=test_images/255.

        val_images=None

    elif "celeba" in name:
        if "celebahq" in name:
            tsize = ''.join(c for c in name if c.isdigit())
            if tsize is not '':
                tsize = int(tsize)
            else:
                print('Please specify image size')
                quit()
            dsname = ''.join(c for c in name if not c.isdigit())
            print(tsize, dsname)

            if os.path.exists(os.path.join(dsdir, dsname, '{}.h5'.format(name))):
                f = h5py.File(os.path.join(dsdir, dsname, '{}.h5'.format(name)), 'r')
                train_images = f['train'][:]
                val_images = []
                train_labels = np.zeros((len(train_images),))
                test_images = 0
                test_labels = 0
            else:
                imgdir = os.path.join(dsdir, dsname, 'CelebA-HQ-img')
                train_images = []
                test_images = 0
                for idx, filename in enumerate(os.listdir(imgdir)):

                    fn = os.path.join(imgdir, filename)
                    img = matplotlib.image.imread(fn)

                    img = cv2.resize(img,(tsize, tsize),cv2.INTER_LINEAR)

                    img = img / 255.

                    train_images.append(img)
                    if (idx+1) % 10000 == 0:
                        print(len(train_images))
                train_images = np.array(train_images)

                print("Saved: train {}".format(train_images.shape))

                with h5py.File(os.path.join(dsdir, dsname, '{}.h5'.format(name)), "w") as f:
                    dset = f.create_dataset("train", data=train_images) #chunks=(20000)

                train_labels = np.zeros((len(train_images),))
                test_labels = 0

        else:
            tsize = ''.join(c for c in name if c.isdigit())
            if tsize is not '':
                tsize = int(tsize)
            else:
                print('Please specify image size: celeba32 or celeba64')
                quit()
            dsname = ''.join(c for c in name if not c.isdigit())
            print(tsize, dsname)

            if os.path.exists(os.path.join(dsdir, dsname, '{}.h5'.format(name))):
                f = h5py.File(os.path.join(dsdir, dsname, '{}.h5'.format(name)), 'r')
                train_images = f['train'][:]
                val_images = f['validation'][:]
                train_labels = f['train_labels'][:]
                test_images = f['test'][:]
                test_labels = f['test_labels'][:]
                val_labels = f['validation_labels'][:]
            else:
                imgdir = os.path.join(dsdir, dsname, 'img_align_celeba')
                with open(os.path.join(dsdir, dsname,'list_eval_partition.txt'), 'r') as f:
                    lines = f.readlines()
                with open(os.path.join(dsdir, dsname,'list_attr_celeba.txt'), 'r') as f:
                    attr_lines = f.readlines()
                train_images = []
                val_images = []
                test_images = []
                train_labels = []
                val_labels = []
                test_labels = []
                train_images_np = None
                val_images_np = None
                test_images_np = None

                print('num_total_train: ', num_total_train)
                print('use_num_images: ', use_num_images)

                if shuffle_index is not None:
                    for i in range(use_num_images):
                        linelist = lines[shuffle_index[i]].split()
                        attr_lineslist = attr_lines[shuffle_index[i]+2].split()
                        assert linelist[0]==attr_lineslist[0]
                        labels = [int(j) for j in attr_lineslist[1:]]
                        labels = np.asarray(labels)
                        fn = os.path.join(imgdir, linelist[0])
                        img = matplotlib.image.imread(fn)
                        osh = (img.shape[0] - 160) // 2
                        osw = (img.shape[1] - 160) // 2
                        img = img[osh:img.shape[0]-osh, osw:img.shape[1]-osw]
                        img = cv2.resize(img,(tsize, tsize),cv2.INTER_LINEAR)
                        img = img / 255.
                        if int(linelist[1]) == 0:
                            train_images.append(img)
                            train_labels.append(labels)
                        elif int(linelist[1]) == 1:
                            val_images.append(img)
                            val_labels.append(labels)
                        elif int(linelist[1]) == 2:
                            test_images.append(img)
                            test_labels.append(labels)
                        if (i+1) % 10000 == 0:
                            print(len(train_images),len(val_images), len(test_images))

                    print('****', len(train_images),len(val_images), len(test_images))

                else:
                    for idx, line in enumerate(lines):
                        linelist = line.split()
                        attr_lineslist = attr_lines[idx+2].split()
                        assert linelist[0]==attr_lineslist[0]
                        labels = [int(i) for i in attr_lineslist[1:]]
                        labels = np.asarray(labels)
                        fn = os.path.join(imgdir, linelist[0])
                        img = matplotlib.image.imread(fn)

                        osh = (img.shape[0] - 160) // 2
                        osw = (img.shape[1] - 160) // 2
                        img = img[osh:img.shape[0]-osh, osw:img.shape[1]-osw]

                        img = cv2.resize(img,(tsize, tsize),cv2.INTER_LINEAR)

                        img = img / 255.

                        if int(linelist[1]) == 0:

                            train_images.append(img)
                            train_labels.append(labels)
                        elif int(linelist[1]) == 1:
                            val_images.append(img)
                            val_labels.append(labels)
                        elif int(linelist[1]) == 2:
                            test_images.append(img)
                            test_labels.append(labels)
                        if (idx+1) % 10000 == 0:
                            print(len(train_images),len(val_images), len(test_images))

                train_images = np.array(train_images)
                print('train_images shape: ', train_images.shape)
                val_images = np.array(val_images)
                print('val_images shape: ', val_images.shape)
                test_images = np.array(test_images)
                print('test_images shape: ', test_images.shape)

                train_labels = np.array(train_labels)
                print('train_labels shape: ', train_labels.shape)
                val_labels = np.array(val_labels)
                print('val_labels shape: ', val_labels.shape)
                test_labels = np.array(test_labels)
                print('test_labels shape: ', test_labels.shape)

                print("Saved: train {}, validation {}, test {}.".format(train_images.shape, val_images.shape, test_images.shape))

                if shuffle_index is None:
                    with h5py.File(os.path.join(dsdir, dsname, '{}.h5'.format(name)), "w") as f:
                        dset = f.create_dataset("train", data=train_images) #chunks=(20000)
                        dset = f.create_dataset("validation", data=val_images)
                        dset = f.create_dataset("test", data=test_images)
                        dset = f.create_dataset("train_labels", data=train_labels)
                        dset = f.create_dataset("validation_labels", data=val_labels)
                        dset = f.create_dataset("test_labels", data=test_labels)

    if len(train_images.shape) == 3:
        train_images = train_images.reshape((train_images.shape[0],train_images.shape[1],train_images.shape[2],1))
        test_images = test_images.reshape((test_images.shape[0],test_images.shape[1],test_images.shape[2],1))

    if name=='lsun-bedroom' or name=='texture' or 'celeba' in name:
        class_list=[0]
    else:
        class_list=[0,1,2,3,4,5,6,7,8,9]
    return train_images, train_labels, test_images, test_labels, class_list, val_images

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def Normalize(data, a=0, b=1, ABS=False):
    # Normalized to [a, b]
    data = data.astype(float)
    # print(type(data))
    if ABS:
        data_new = np.abs(data)
    else:
        data_new = data
    k = float(b-a) / (np.max(data_new) - np.min(data_new))
    # data_new = data_new - data_new.min()
    # print data_new.min(), data_new.max()

    return (data_new - data_new.min())*k + a

if __name__ == '__main__':

    import_data2('./datasets/', 'celeba32')


    