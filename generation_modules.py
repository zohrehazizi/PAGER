########################
# Author: Zohreh Azizi #
########################

import numpy as np
import os 
import pickle
import cv2 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plot_figures as pf
from sklearn.mixture import GaussianMixture
import copy
from torch.nn import Fold
from torch.nn import Unfold
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import solve
import faiss
from torch_ssl import torchSaab, sslModel
from torch_regression import image_regressor
from time import time
from torch_configs import device, convert_to_torch, convert_to_numpy

report_timing = True

class timerClass():
    def __init__(self):
        self.events = ['start']
        self.times = [time()]
    def register(self, eventname):
        self.events.append(eventname)
        self.times.append(time())
    def print(self):
        if report_timing:
            totalTime = self.times[-1]-self.times[0]
            for i in range(1, len(self.events)):
                event = self.events[i]
                time = self.times[i]
                time_prev = self.times[i-1]
                print(f"{event}: {time-time_prev}, percentage: {(time-time_prev)/(totalTime)*100}")    

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, n_threads=10, init=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = (device == 'cuda') #torch.cuda.device_count()>0     
        faiss.omp_set_num_threads(n_threads)       
        self.__version__ = faiss.__version__
        self.init = init

    def load(self, cluster_centers_, labels_):
        if self.gpu != False:
            self.kmeans = faiss.Kmeans(d=cluster_centers_.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    gpu=self.gpu,
                                    )
        else:
            self.kmeans = faiss.Kmeans(d=cluster_centers_.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,)
        print('cluster_centers_.shape', cluster_centers_.shape)
        self.kmeans.train(cluster_centers_)
        self.kmeans.cluster_centers_ = cluster_centers_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = labels_

    def fit(self, X):
        if self.gpu != False:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    gpu=self.gpu,
                                    )
        else:
            self.kmeans = faiss.Kmeans(d=X.shape[1],
                                    k=self.n_clusters,
                                    niter=self.max_iter,
                                    nredo=self.n_init,
                                    )
        dummyKmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1, init=self.init).fit(X)
        init_centroids = dummyKmeans.cluster_centers_.astype('float32')
        X = np.ascontiguousarray(X.astype('float32'))
        self.kmeans.train(X, init_centroids=init_centroids)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]
        self.labels_ = self.kmeans.index.search(X.astype(np.float32), 1)[1].ravel()
        return self
        
    def predict(self, X):
        X = np.ascontiguousarray(X.astype('float32'))
        return self.kmeans.index.search(X.astype(np.float32), 1)[1].ravel()
    
    def inverse_predict(self, label):
        return self.cluster_centers[label]

class ac_leveled_gen():
    def __init__(self, N, n_clusters, pca_components, params_path, apply_regression=False, sharpen_approx_AC=False, stopRecurseN=1, regressorArgs={'type':'linear'}):
        self.N = N #N is the log of image_size, e.g., if image size is 32 then N=5 as 2**5=32
        self.stopRecurseN = stopRecurseN
        self.children = [[None,None],[None,None]]
        self.DC_saab_model = None
        self.AC_saab_model = None
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.params_path = params_path
        self.regressorArgs = copy.deepcopy(regressorArgs)
        self.apply_regression = apply_regression
        self.sharpen_approx_AC = sharpen_approx_AC
        self.kernel= np.array([[-2, -2, -2],
                               [-2, 32,-2],
                               [-2, -2, -2]])
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        
    def downSample(self, image):
        h,w = image.shape[0:2]
        dim = (int(h/2),int(w/2))
        ds = cv2.resize(image,dim,interpolation = cv2.INTER_LANCZOS4)
        if len(ds.shape)==2:
            ds = np.expand_dims(ds, axis=-1)
        return ds
    
    def upSample(self, image):
        h,w = image.shape[0:2]
        dim = (h*2,w*2)
        us = cv2.resize(image,dim,interpolation = cv2.INTER_LANCZOS4)
        if len(us.shape)==2:
            us = np.expand_dims(us, axis=-1)
        return us
    
    def run_inverse(self, samples, model):
        with torch.no_grad():
            if samples.ndim==2:
                samples = np.expand_dims(samples, axis=(2,3))
            model.set_inverse()
            reversed_samples = model(samples).transpose(0,2,3,1)
            model.set_forward()
            return reversed_samples

    def run_forward(self, images, model):
        with torch.no_grad():
            #images: <n_images,n,n,3>
            images = images.transpose(0,3,1,2)
            model.set_forward()
            features = model(images)
            return features.squeeze(-1).squeeze(-1)

    def get_DC_and_AC(self,images):
        print('--- in ac_leveled_gen get_DC_and_AC ----')
        DC = np.asarray([self.downSample(img) for img in images])
        DC_upsampled = np.asarray([self.upSample(img) for img in DC])
        AC = np.asarray([img - imgDC for img,imgDC in zip(images,DC_upsampled)])

        return DC, AC

    def load(self):
        # print("fitting model with DC shape {} and AC shape {}".format(DC.shape,AC.shape))
        n_clusters = self.n_clusters
        pca_components = self.pca_components
               
        path_to_dc = os.path.join(self.params_path, "DC_model")
        path_to_ac = os.path.join(self.params_path, "AC_model")
        self.DC_saab_model = torch.load(path_to_dc, map_location=device)
        self.AC_saab_model = torch.load(path_to_ac, map_location=device)

        path_to_kmeans = os.path.join(self.params_path, 'dc_kmeans_'+str(self.n_clusters))       
        foo = open(path_to_kmeans, 'rb')
        DC_kmeans = pickle.load(foo)
        foo.close()
        self.DC_kmeans = DC_kmeans
        
        path_to_ac_codebooks = os.path.join(self.params_path,'ac_codebook_'+str(self.n_clusters))
        foo = open(path_to_ac_codebooks, 'rb')
        self.AC_codebook, self.AC_codebook_rgb = pickle.load(foo)
        foo.close()

        if self.apply_regression:
            path_to_regressor = os.path.join(self.params_path, 'regressor')
            foo = open(path_to_regressor, 'rb')
            self.regressor = pickle.load(foo)
            foo.close()

        if self.N>self.stopRecurseN:
            #break the images into 4 windows and redo all the stuff
            for i in range(2):
                for j in range(2):
                    path_to_child = os.path.join(self.params_path, 'child-{}-{}'.format(i,j))
                    self.children[i][j] = ac_leveled_gen(N=self.N-1, 
                        n_clusters=self.n_clusters, 
                        pca_components=self.pca_components, 
                        params_path=path_to_child,
                        apply_regression=self.apply_regression,
                        sharpen_approx_AC=self.sharpen_approx_AC,
                        stopRecurseN=self.stopRecurseN,
                        regressorArgs=self.regressorArgs)
                    self.children[i][j].load()

    def gen_ac_dc_saab(self, DC, AC):
        timer = timerClass()
        path_to_dc = os.path.join(self.params_path, "DC_model")
        path_to_ac = os.path.join(self.params_path, "AC_model")
        if os.path.exists(path_to_dc):
            self.DC_saab_model = torch.load(path_to_dc, map_location=device)
            timer.register("gen_ac_dc_saab---->load DC_saab_model")
            DC_features = self.run_forward(DC, self.DC_saab_model)
            timer.register("gen_ac_dc_saab---->forward DC_saab_model")

        else:
            self.DC_saab_model, DC_features = self.learn_saab(DC)
            timer.register("gen_ac_dc_saab---->fit_predict DC_saab_model")
            torch.save(self.DC_saab_model, path_to_dc)
            timer.register("gen_ac_dc_saab---->dump DC_saab_model")
        if os.path.exists(path_to_ac):
            self.AC_saab_model = torch.load(path_to_ac, map_location=device)
            timer.register("gen_ac_dc_saab---->load AC_saab_model")
            AC_features = self.run_forward(AC, self.AC_saab_model)
            timer.register("gen_ac_dc_saab---->forrward AC_saab_model")
        else:
            self.AC_saab_model, AC_features = self.learn_saab(AC)
            timer.register("gen_ac_dc_saab---->fit_predict AC_saab_model")
            torch.save(self.AC_saab_model, path_to_ac)
            timer.register("gen_ac_dc_saab---->dump AC_saab_model")
        DC_features = np.squeeze(DC_features)
        AC_features = np.squeeze(AC_features)
        timer.register("gen_ac_dc_saab---->finish")
        timer.print()
        print("###### finished gen_ac_dc_saab #######")
        return DC_features, AC_features
    
    def apply_pca(self, features, pca_components):
        pca = PCA(n_components = pca_components, svd_solver = 'full')
        features = pca.fit_transform(features)
        return features, pca

    def learn_ac_rep(self, DC_features, AC_features):
        path_to_kmeans = os.path.join(self.params_path, 'dc_kmeans_'+str(self.n_clusters))
        if os.path.exists(path_to_kmeans):
            foo = open(path_to_kmeans, 'rb')
            DC_kmeans = pickle.load(foo)
            foo.close()
        else:
            DC_kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++').fit(DC_features)
            foo = open(path_to_kmeans, 'wb')
            pickle.dump(DC_kmeans,foo)
            foo.close()
        self.DC_kmeans = DC_kmeans
        labels = DC_kmeans.labels_

        path_to_ac_codebooks = os.path.join(self.params_path,'ac_codebook_'+str(self.n_clusters))
        if os.path.exists(path_to_ac_codebooks):
            foo = open(path_to_ac_codebooks, 'rb')
            self.AC_codebook, self.AC_codebook_rgb = pickle.load(foo)
            foo.close()
        else:
            AC_codebook = [np.mean(AC_features[labels==i],axis=0) for i in range(n_clusters)]
            self.AC_codebook = np.expand_dims(np.asarray(AC_codebook),axis=[2,3])
            self.AC_codebook_rgb = self.run_inverse(self.AC_codebook, self.AC_saab_model)
            foo = open(path_to_ac_codebooks, 'wb')
            pickle.dump([self.AC_codebook, self.AC_codebook_rgb], foo)
            foo.close()

        approximated_AC = np.asarray([self.AC_codebook_rgb[i] for i in labels])
        
        return approximated_AC
    
    def fit_predict_regressor(self, X, Y):
        path_to_regressor = os.path.join(self.params_path, 'regressor')
        with torch.no_grad():
            if os.path.exists(path_to_regressor):
                self.regressor = torch.load(path_to_regressor, map_location=device)
            else:
                self.regressor = image_regressor(regressorArgs=copy.deepcopy(self.regressorArgs))
                self.regressor.fit(X,Y)
                torch.save(self.regressor,path_to_regressor)
            return self.regressor(X)

    def fit(self, DC, AC):
        # print("fitting model with DC shape {} and AC shape {}".format(DC.shape,AC.shape))
        
        
        DC_features, AC_features = self.gen_ac_dc_saab(DC, AC)  
        
        # print("DC features shape: {}, AC features shape: {}".format(DC_features.shape,AC_features.shape))
        if self.pca_components is not None:
            DC_features, pca_dc = self.apply_pca(DC_features, self.pca_components)
        else:
            pca_dc = None 
        
        approximated_AC = self.learn_ac_rep(DC_features, AC_features)

        if self.sharpen_approx_AC:
            approximated_AC = [cv2.filter2D(src=ac, ddepth=-1, kernel=self.kernel) for ac in approximated_AC]
            approximated_AC = np.asarray(approximated_AC)
           
        if DC.shape[-2]==approximated_AC.shape[-2]:
            next_DC = DC+approximated_AC
            orig_images = DC+AC
        else:
            DC_upsampled = np.asarray([self.upSample(img) for img in DC])
            next_DC = DC_upsampled+approximated_AC
            orig_images = DC_upsampled+AC
        
        if self.apply_regression:
            next_DC = self.fit_predict_regressor(next_DC,orig_images)

        next_AC = orig_images-next_DC

        w, h = next_AC.shape[1:3]
        if self.N>self.stopRecurseN:
            #break the images into 4 windows and redo all the stuff
            for i in range(2):
                for j in range(2):
                    path_to_child = os.path.join(self.params_path, 'child-{}-{}'.format(i,j))
                    self.children[i][j] = ac_leveled_gen(N=self.N-1, 
                        n_clusters=self.n_clusters, 
                        pca_components=self.pca_components, 
                        params_path=path_to_child,
                        apply_regression=self.apply_regression,
                        sharpen_approx_AC=self.sharpen_approx_AC,
                        stopRecurseN=self.stopRecurseN,
                        regressorArgs=self.regressorArgs)

                    cropped_DC = self.crop(next_DC, i,j)#[:,int(i*h/2):int((i+1)*h/2),int(j*w/2):int((j+1)*w/2),:]
                    cropped_AC = self.crop(next_AC, i,j)#[:,int(i*h/2):int((i+1)*h/2),int(j*w/2):int((j+1)*w/2),:]
                    self.children[i][j].fit(cropped_DC, cropped_AC)

        ####### test #######
        """
        test_dc = DC_upsampled[0:25]
        test_ac = np.asarray([self.AC_codebook_rgb[i] for i in labels[0:25]])
        test_im = test_dc + test_ac
        fig_path = 'figures'
        pf.plot_images(test_im,5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC+PredictedAC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(DC_upsampled[0:25]+AC[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC+AC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(DC_upsampled[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(test_ac[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='AC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        """
    
    def plot_figs(self, DC, enhanced, AC=None):
        if AC is None:
            mode = '_test'
        else:
            mode = '_train'
        pf.plot_images(DC[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=self.params_path,fn_pre='DC'+mode)
        pf.plot_images(enhanced[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=self.params_path,fn_pre='enhanced'+mode)
        if AC is not None:
            pf.plot_images(AC[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=self.params_path,fn_pre='AC'+mode)
            pf.plot_images(enhanced[0:25]+AC[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=self.params_path,fn_pre='original'+mode)
        
    def transform(self, DC):
        print("transforming", DC.shape)
        images = np.moveaxis(DC, 3, 1)
        h, w = images.shape[2:]
        assert h==w 
        nHops = int(np.log2(h)+0.001)

        features_DC = self.run_forward(images, self.DC_saab_model)
        labels = self.DC_kmeans.predict(features_DC)
        approximated_AC = np.asarray([self.AC_codebook_rgb[i] for i in labels])
        if self.sharpen_approx_AC:
            approximated_AC = [cv2.filter2D(src=ac, ddepth=-1, kernel=self.kernel) for ac in approximated_AC]
            approximated_AC = np.asarray(approximated_AC)
        if DC.shape[-2]==approximated_AC.shape[-2]:
            next_DC = DC+approximated_AC
        else:
            DC_upsampled = np.asarray([self.upSample(img) for img in DC])
            next_DC = DC_upsampled+approximated_AC
        if self.apply_regression:
            next_DC = self.regressor(next_DC)
        w, h = approximated_AC.shape[1:3]
        if self.N>self.stopRecurseN:
            enhanced_cropped = [[None,None],[None,None]]
            
            #break the images into 4 windows and use the children to enhance each window
            for i in range(2):
                for j in range(2):
                    cropped_DC = self.crop(next_DC, i,j)#[:,int(i*h/2):int((i+1)*h/2),int(j*w/2):int((j+1)*w/2),:]
                    enhanced_cropped[i][j] = self.children[i][j].transform(cropped_DC)
            output = self.stich(enhanced_cropped)
        else:
            output = next_DC
        return output

    def learn_saab(self, images):
        # images, mean, std = normalize_images(images)
        images = images.transpose(0,3,1,2)
        h, w = images.shape[2:]
        assert h==w 
        nHops = int(np.log2(h)+0.001)
        channelwise = []
        kernel_size = [(2,2) for _ in range(nHops)]
        stride_size = [(2,2) for _ in range(nHops)]
        layers = [torchSaab(kernel_size=(2,2), stride=(2,2), channelwise=(l!=0)) for l in range(nHops)] 

        model = sslModel(layers).eval()
        with torch.no_grad():
            features = model.fit(images)
        return model, features

    def crop(self, images, i , j):
        h = images.shape[1]
        w = images.shape[2]
        cropped = images[:,int(i*h/2):int((i+1)*h/2),int(j*w/2):int((j+1)*w/2),:]
        return cropped
                    
    def stich(self, images):
        dim = images[0][0].shape[2]
        n = images[0][0].shape[0]
        stiched = np.zeros((n, dim*2,dim*2, images[0][0].shape[3]))
        for i in range(2):
            for j in range(2):
                stiched[:,i*dim:(i+1)*dim,j*dim:(j+1)*dim,:] = images[i][j]
        return stiched

class classwise_ac_leveled_gmm(ac_leveled_gen):
    def __init__(self, N, n_clusters, ac_clusters, pca_components, params_path, reg_covar=1e-6, apply_regression=False, sharpen_approx_AC=False, stopRecurseN=1, regressorArgs={'type':'linear'}, pca_list = None, n_clusters_sampling=None):
        super().__init__(N, n_clusters=n_clusters,
                         pca_components=pca_components, 
                         params_path=params_path, 
                         apply_regression=apply_regression, 
                         sharpen_approx_AC=sharpen_approx_AC, 
                         stopRecurseN=stopRecurseN, 
                         regressorArgs=regressorArgs)
        
        self.ac_clusters = ac_clusters
        self.n_clusters_sampling = n_clusters_sampling
        self.reg_covar = reg_covar
    
    def load(self):
        # print("fitting model with DC shape {} and AC shape {}".format(DC.shape,AC.shape))
        n_clusters = self.n_clusters
        pca_components = self.pca_components
               
        path_to_dc = os.path.join(self.params_path, "DC_model")
        path_to_ac = os.path.join(self.params_path, "AC_model")
        self.DC_saab_model = torch.load(path_to_dc, map_location=device)
        self.AC_saab_model = torch.load(path_to_ac, map_location=device)
        
        path_to_gmm = os.path.join(self.params_path, 'dc_gmm_'+str(self.n_clusters))       
        foo = open(path_to_gmm, 'rb')
        self.dc_gmm = pickle.load(foo)
        foo.close()
        # dc_labels = self.dc_gmm.predict(DC_features)

        path_to_ac_gmms = os.path.join(self.params_path,f'ac_gmm_{self.n_clusters}_{self.ac_clusters}')
        foo = open(path_to_ac_gmms, 'rb')
        self.ac_gmms = pickle.load(foo)
        foo.close()

        if self.pca_components is not None:
            assert self.n_clusters_sampling is not None, "n_clusters_sampling should not be None"
            self.pca_dc_gmms = {}
            self.pca_dc = {}
            for n_components in pca_components:
                path_to_pca = os.path.join(self.params_path, f"pca_{n_components}")
                path_to_pca_gmm = os.path.join(self.params_path, f"pca_dc_{n_components}_gmm_{self.n_clusters_sampling}")
                foo = open(path_to_pca, 'rb')
                pca_dc = pickle.load(foo)
                foo.close()
                self.pca_dc[n_components] = pca_dc
                foo = open(path_to_pca_gmm, 'rb')
                pca_dc_gmm = pickle.load(foo)
                foo.close()
                self.pca_dc_gmms[n_components] = pca_dc_gmm


        if self.apply_regression and self.N<7:
            path_to_regressor = os.path.join(self.params_path, 'regressor')
            foo = open(path_to_regressor, 'rb')
            self.regressor = pickle.load(foo)
            foo.close()

        if self.N>self.stopRecurseN:
            #break the images into 4 windows and redo all the stuff
            for i in range(2):
                for j in range(2):
                    path_to_child = os.path.join(self.params_path, 'child-{}-{}'.format(i,j))
                    self.children[i][j] = ac_leveled_gmm(N=self.N-1, 
                            n_clusters=self.n_clusters, 
                            ac_clusters=self.ac_clusters,
                            pca_components=None, 
                            params_path=path_to_child,
                            reg_covar = self.reg_covar,
                            apply_regression=self.apply_regression,
                            sharpen_approx_AC=self.sharpen_approx_AC,
                            stopRecurseN=self.stopRecurseN,
                            regressorArgs=self.regressorArgs)
                    self.children[i][j].load()

    def fit(self, DC, AC, classes):
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        #print('class_ids, num_classes', class_ids, num_classes)
        timer = timerClass()
        DC_features, AC_features = self.gen_ac_dc_saab(DC, AC)
        timer.register('learned saab')
        
        if self.pca_components is not None:
            assert self.n_clusters_sampling is not None, "n_clusters_sampling should not be None"
            self.pca_dc_gmms = {}
            self.pca_dc = {}
            for n_components in self.pca_components:
                path_to_pca = os.path.join(self.params_path, f"pca_{n_components}")
                path_to_pca_gmm = os.path.join(self.params_path, f"pca_dc_{n_components}_gmm_{self.n_clusters_sampling}")
                if (not os.path.exists(path_to_pca)) or (not os.path.exists(path_to_pca_gmm)):
                    pca_dc = []
                    pca_dc_gmm = []
                    for c in range(num_classes):
                        f = DC_features[classes==c]
                        pca_features, p_dc = self.apply_pca(f, n_components)
                        pca_dc.append(p_dc)
                        g = GaussianMixture(n_components=self.n_clusters_sampling, covariance_type='diag', reg_covar = self.reg_covar).fit(pca_features)
                        pca_dc_gmm.append(g)
                    timer.register('pca-GMM computated')
                    foo = open(path_to_pca, 'wb')
                    pickle.dump(pca_dc, foo)
                    foo.close()
                    foo = open(path_to_pca_gmm, 'wb')
                    pickle.dump(pca_dc_gmm,foo)
                    foo.close()
                    timer.register('pca-GMM dumped')
                else:
                    foo = open(path_to_pca, 'rb')
                    pca_dc = pickle.load(foo)
                    foo.close()
                    foo = open(path_to_pca_gmm, 'rb')
                    pca_dc_gmm = pickle.load(foo)
                    foo.close()
                    timer.register('pca-GMM loaded')
                self.pca_dc_gmms[n_components] = pca_dc_gmm
                self.pca_dc[n_components] = pca_dc
        
        if self.N>self.stopRecurseN or (self.apply_regression and self.N<7): 
            approximated_AC = self.learn_ac_rep(DC_features, AC_features, classes=classes, return_AC=True)
            timer.register('learned ac rep')
            if self.sharpen_approx_AC:
                approximated_AC = [cv2.filter2D(src=ac, ddepth=-1, kernel=self.kernel) for ac in approximated_AC]
                approximated_AC = np.asarray(approximated_AC)
                timer.register('resize approximated AC')
            if DC.shape[-2]==approximated_AC.shape[-2]:
                next_DC = DC+approximated_AC
                orig_images = DC+AC
            else:
                DC_upsampled = np.asarray([self.upSample(img) for img in DC])
                next_DC = DC_upsampled+approximated_AC
                orig_images = DC_upsampled+AC            
            if self.apply_regression and self.N<7:               
                path_to_regressor = os.path.join(self.params_path, 'regressor')
                
                if os.path.exists(path_to_regressor):
                    self.regressor = torch.load(path_to_regressor, map_location=device)
                    timer.register('regressor loaded')
                else:
                    self.regressor = [image_regressor(regressorArgs=copy.deepcopy(self.regressorArgs)) for c in range(num_classes)]
                    for c in range(num_classes):
                        self.regressor[c].fit(next_DC[classes==c],orig_images[classes==c])
                    timer.register('trained regressors')
                    torch.save(self.regressor,path_to_regressor)
                    timer.register('saved regressors')
                for c in range(num_classes):
                    next_DC[classes==c] = self.regressor[c](next_DC[classes==c])
                timer.register('computed regressed images')
            next_AC = orig_images-next_DC
            w, h = next_AC.shape[1:3]
            timer.register("finished")
            timer.print()
            if self.N>self.stopRecurseN:
                for i in range(2):
                    for j in range(2):
                        path_to_child = os.path.join(self.params_path, 'child-{}-{}'.format(i,j))
                        self.children[i][j] = classwise_ac_leveled_gmm(N=self.N-1, 
                                                            n_clusters=self.n_clusters, 
                                                            ac_clusters=self.ac_clusters,
                                                            pca_components=None, 
                                                            params_path=path_to_child,
                                                            reg_covar=self.reg_covar,
                                                            apply_regression=self.apply_regression,
                                                            sharpen_approx_AC=self.sharpen_approx_AC,
                                                            stopRecurseN=self.stopRecurseN,
                                                            regressorArgs=self.regressorArgs)

                        cropped_DC = self.crop(next_DC, i,j)
                        cropped_AC = self.crop(next_AC, i,j)
                        self.children[i][j].fit(cropped_DC, cropped_AC, classes)


        else:
            self.learn_ac_rep(DC_features, AC_features, classes=classes, return_AC=False)
            timer.register('learned ac rep')
            timer.register("finished")
            timer.print()
        ####### test #######
        """
        test_dc = DC_upsampled[0:25]
        test_ac = np.asarray([self.AC_codebook_rgb[i] for i in labels[0:25]])
        test_im = test_dc + test_ac
        fig_path = 'figures'
        pf.plot_images(test_im,5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC+PredictedAC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(DC_upsampled[0:25]+AC[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC+AC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(DC_upsampled[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='DC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        pf.plot_images(test_ac[0:25],5,5,figsize=(20,20),show=False,save=True,fdir=fig_path,fn_pre='AC_N='+str(self.N)+"_Clusters="+str(n_clusters))
        """

    def sample_ac(self, dc_labels, classes, sample_mean=False):
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        ac_dim = self.ac_gmms[0][0].means_.shape[1]
        ac_features = np.zeros((len(dc_labels), ac_dim))
        for c in class_ids:
            for dc_cluster_id in range(self.n_clusters):
                condition = np.logical_and(dc_labels==dc_cluster_id, classes==c)
                num_samples = np.sum(condition)
                if num_samples>0:
                    if sample_mean:
                        if num_samples>1:
                            _, lvals = self.ac_gmms[c][dc_cluster_id].sample(num_samples)
                        else:
                            lvals = [0]
                        ac_feature = np.asarray([np.asarray(self.ac_gmms[c][dc_cluster_id].means_)[label] for label in lvals])
                    else:
                        ac_feature, _ = self.ac_gmms[c][dc_cluster_id].sample(num_samples)
                    ac_features[condition] = ac_feature
        return ac_features

    def sample_ac_labels(self, dc_labels, classes):
        # ac_dim = self.ac_gmms[0].means_.shape[1]
        # ac_features = np.zeros((len(dc_labels), ac_dim))
        ac_labels = np.ones(len(dc_labels), dtype='int32')*100000
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        for c in class_ids:
            for dc_cluster_id in range(self.n_clusters):
                condition = np.logical_and(dc_labels==dc_cluster_id, classes==c)
                num_samples = np.sum(condition)
                if num_samples>0:
                    _, lvals = self.ac_gmms[c][dc_cluster_id].sample(num_samples)
                    ac_labels[condition] = lvals
        return ac_labels

    def learn_ac_rep(self, DC_features, AC_features, classes, return_AC=False):
        timer = timerClass()
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        path_to_gmm = os.path.join(self.params_path, 'dc_gmm_'+str(self.n_clusters))
        if os.path.exists(path_to_gmm):
            foo = open(path_to_gmm, 'rb')
            dc_gmm = pickle.load(foo)
            foo.close()
        else:
            dc_gmm = [GaussianMixture(n_components=self.n_clusters, covariance_type='diag', reg_covar = self.reg_covar).fit(DC_features[classes==c]) for c in class_ids] 
            timer.register('learn_ac_rep---->learned dc gmm')
            foo = open(path_to_gmm, 'wb')
            pickle.dump(dc_gmm,foo)
            foo.close()
            timer.register('learn_ac_rep---->dumped dc gmm')
        self.dc_gmm = dc_gmm
        dc_labels = np.ones(len(DC_features),dtype=np.int32)*100000
        for c in class_ids:
            dc_labels[classes==c] = self.dc_gmm[c].predict(DC_features[classes==c])
        timer.register('learn_ac_rep---->predict dc gmm labels')

        path_to_ac_gmms = os.path.join(self.params_path,f'ac_gmm_{self.n_clusters}_{self.ac_clusters}')
        if os.path.exists(path_to_ac_gmms):
            foo = open(path_to_ac_gmms, 'rb')
            self.ac_gmms = pickle.load(foo)
            foo.close()
            timer.register('learn_ac_rep---->loaded ac_gmms')
        else:
            ac_gmms = []
            for c in class_ids:
                class_ac_gmms = []
                for i in range(self.n_clusters):
                    condition = np.logical_and(classes==c, dc_labels==i)
                    features_this_cluster = AC_features[condition]
                    if len(features_this_cluster)>1:
                        n_components = min(len(features_this_cluster), self.ac_clusters)
                        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar = self.reg_covar).fit(features_this_cluster)
                        if n_components==1:
                            gmm.weights_[0] = 1.0
                    else:
                        print("ac features length less than 1!")
                        dim = features_this_cluster.shape[1]
                        random_data = np.random.rand(10,dim)
                        gmm = GaussianMixture(n_components=1, covariance_type='diag', reg_covar = self.reg_covar).fit(random_data)
                        gmm.means_ = features_this_cluster #np.asarray(features_this_cluster)
                        #print('ac gmm mean shape: ',gmm.means_.shape)
                        gmm.covariances_ = np.zeros([1,dim])
                        gmm.weights_[0] = 1.0
                    class_ac_gmms.append(gmm)
                ac_gmms.append(class_ac_gmms)
            timer.register('learn_ac_rep---->learned ac gmms')
            foo = open(path_to_ac_gmms, 'wb')
            pickle.dump(ac_gmms, foo)
            foo.close()
            timer.register('learn_ac_rep---->dump ac gmms')
            self.ac_gmms = ac_gmms
        

        if return_AC:
            approximated_AC_features = self.sample_ac(dc_labels, classes, sample_mean=True)
            timer.register('learn_ac_rep---->sample ac')
            approximated_AC = self.run_inverse(approximated_AC_features, self.AC_saab_model)
            timer.register('learn_ac_rep---->run inverse')
            timer.register('learn_ac_rep---->finished')
            timer.print()
            print("######## finished learn_ac_rep ########")
            return approximated_AC
        else:
            timer.register('learn_ac_rep---->finished')
            timer.print()
            print("######## finished learn_ac_rep ########")
            return None

    def transform(self, DC, classes, maxN, sample_mean=False, stopRecurseN=1, stopSampleN=1, apply_canny_mask=False, canny_mask=None):
        class_ids = np.unique(classes)
        num_classes = len(class_ids)

        if self.N>stopSampleN:

            print('DC shape: ', DC.shape)
            if DC.shape[2] == 2**(maxN):
                DC_downsampled = np.asarray([self.downSample(img) for img in DC])
                features_DC = self.run_forward(DC_downsampled, self.DC_saab_model)
            else:
                features_DC = self.run_forward(DC, self.DC_saab_model)
            print('DC_features shape: ', features_DC.shape)
            dc_labels = np.ones(len(features_DC),dtype=np.int32)*100000


            for c in class_ids:
                dc_labels[classes==c] = self.dc_gmm[c].predict(features_DC[classes==c])

            ac_features = self.sample_ac(dc_labels, classes, sample_mean=sample_mean)
            approximated_AC = self.run_inverse(ac_features, self.AC_saab_model)
            
            if self.sharpen_approx_AC:
                approximated_AC = [cv2.filter2D(src=ac, ddepth=-1, kernel=self.kernel) for ac in approximated_AC]
                approximated_AC = np.asarray(approximated_AC)

            if apply_canny_mask:
                try:
                    approximated_AC = ((approximated_AC*canny_mask)+approximated_AC)/2
                    print("canny mask applied on approximated AC!")
                except:
                    approximated_AC = approximated_AC
                    print("canny mask was not given!")

            if DC.shape[-2]==approximated_AC.shape[-2]:
                next_DC = DC+approximated_AC
            else:
                DC_upsampled = np.asarray([self.upSample(img) for img in DC])
                next_DC = DC_upsampled+approximated_AC
                
        else:
            next_DC=DC

        

        if self.apply_regression and self.N<7:
            for c in class_ids:
                next_DC[classes==c] = self.regressor[c](next_DC[classes==c])
            #print("coeffs at transform", self.regressor.regressor.coef_)
        # w, h = approximated_AC.shape[1:3] 
        if apply_canny_mask:
            next_canny_mask = self.find_canny_mask(next_DC)

        if self.N>stopRecurseN:
            enhanced_cropped = [[None,None],[None,None]]
            #break the images into 4 windows and use the children to enhance each window
            for i in range(2):
                for j in range(2):
                    cropped_DC = self.crop(next_DC, i,j)#[:,int(i*h/2):int((i+1)*h/2),int(j*w/2):int((j+1)*w/2),:]
                    if apply_canny_mask:
                        cropped_canny = self.crop(next_canny_mask, i,j)
                    else:
                        cropped_canny = None
                    enhanced_cropped[i][j] = self.children[i][j].transform(DC=cropped_DC, classes=classes, maxN=maxN, sample_mean=sample_mean, stopRecurseN=stopRecurseN, stopSampleN=stopSampleN, apply_canny_mask=apply_canny_mask, canny_mask=cropped_canny)
            output = self.stich(enhanced_cropped)
        else:
            output = next_DC
        return output

    def find_canny_mask(self, image):
        image[image<0]=0
        image[image>1]=1
        image = (image*255).astype('uint8')
        return np.expand_dims((np.asarray([cv2.Canny(image=im, threshold1=50, threshold2=50) for im in image]).astype('float64')/255), axis=3) 

    def sample_dc(self, classes, cov_multiplier=None, pca_components=None):
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        size = len(classes)

        if pca_components is None:
            if cov_multiplier is not None:
                original_covariance = copy.deepcopy([g.covariances_ for g in self.dc_gmm])
                for g in self.dc_gmm:
                    g.covariances_ = g.covariances_*cov_multiplier

            dim = self.dc_gmm[0].means_.shape[1]
            samples = np.zeros((size,dim))
            labels = np.zeros(size)
            for c in class_ids:
                size_c = int(np.sum(classes==c))
                s, l = self.dc_gmm[c].sample(size_c)
                samples[classes==c]=s 
                labels[classes==c]=l
            if cov_multiplier is not None:
                for g, cov in zip(self.dc_gmm,original_covariance):
                    g.covariances_ = cov

        else:
            pca_gmm = self.pca_dc_gmms[pca_components]
            pca = self.pca_dc[pca_components]

            if cov_multiplier is not None:
                original_covariance = copy.deepcopy([g.covariances_ for g in pca_gmm])
                for g in pca_gmm:
                    g.covariances_ = g.covariances_*cov_multiplier


            dim = self.dc_gmm[0].means_.shape[1]
            samples = np.zeros((size,dim))
            labels = np.zeros(size)
            for c in class_ids:
                size_c = int(np.sum(classes==c))
                s, l = pca_gmm[c].sample(size_c)
                samples[classes==c] = pca[c].inverse_transform(s)
                labels[classes==c]=l 
            if cov_multiplier is not None:
                for g, cov in zip(pca_gmm,original_covariance):
                    g.covariances_ = cov

        return self.run_inverse(samples, self.DC_saab_model)

    def adjust_covariances(self, cov_multiplier):
        for class_gmm in self.ac_gmms:
            for gmm in class_gmm:
                gmm.covariances_ = gmm.covariances_*cov_multiplier
        for i in range(2):
            for j in range(2):
                if self.children[i][j] is not None:
                    self.children[i][j].adjust_covariances(cov_multiplier)

    def get_regressor_size(self):
        this_level = np.sum([r.get_size() for r in self.regressor])
        if self.children[0][0] is not None:
            lower_level = np.sum([self.children[i][j].get_regressor_size() for i in range(2) for j in range(2)])
        else:
            lower_level = 0 
        return this_level+lower_level
    
    def get_dc_gmm_size(self):
        this_level = 0
        for i in range(len(self.dc_gmm)):
            this_level += (self.dc_gmm[i].means_.size+self.dc_gmm[i].covariances_.size+self.dc_gmm[i].weights_.size)
        if self.children[0][0] is not None:
            lower_level = np.sum([self.children[i][j].get_dc_gmm_size() for i in range(2) for j in range(2)])
        else:
            lower_level = 0 
        return this_level+lower_level

    def get_ac_gmm_size(self):
        this_level = 0
        for i in range(len(self.ac_gmms)):
            for j in range(len(self.ac_gmms[i])):
                this_level += (self.ac_gmms[i][j].means_.size+self.ac_gmms[i][j].covariances_.size+self.ac_gmms[i][j].weights_.size)
        if self.children[0][0] is not None:
            lower_level = np.sum([self.children[i][j].get_ac_gmm_size() for i in range(2) for j in range(2)])
        else:
            lower_level = 0 
        return this_level+lower_level
    
    def get_sampling_params_size(self):
        size_pca = {}
        for key, class_pcas in self.pca_dc.items():
            size_pca[key] = 0
            for pca in class_pcas:
                size_pca[key] += (pca.components_.size+pca.explained_variance_.size+pca.mean_.size)
        size_gmm = {}
        for key, class_gmms in self.pca_dc_gmms.items():
            size_gmm[key] = 0
            for gmm in class_gmms:
                size_gmm[key] += (gmm.means_.size+gmm.covariances_.size+gmm.weights_.size)
        params_size = {key: size_pca[key]+size_gmm[key] for key in size_gmm.keys()}
        return np.max([v for k,v in params_size.items()])

    def get_dc_model_size(self):
        this_level = self.DC_saab_model.get_size()
        if self.children[0][0] is not None:
            lower_level = np.sum([self.children[i][j].get_dc_model_size() for i in range(2) for j in range(2)])
        else:
            lower_level = 0 
        return this_level+lower_level

    def get_ac_model_size(self):
        this_level = self.AC_saab_model.get_size()
        if self.children[0][0] is not None:
            lower_level = np.sum([self.children[i][j].get_ac_model_size() for i in range(2) for j in range(2)])
        else:
            lower_level = 0 
        return this_level+lower_level

    def get_size(self):
        return {'regressor': self.get_regressor_size(),
        'dc_gmm': self.get_dc_gmm_size(),
        'ac_gmm': self.get_ac_gmm_size(),
        'sampling_params': self.get_sampling_params_size(),
        'DC_saab': self.get_dc_model_size(),
        'AC_saab': self.get_ac_model_size()
        }

class resolution_enhancer():
    def __init__(self, minN, maxN, n_clusters, ac_clusters, pca_components, params_path, 
                        apply_regression=False, stopRecurseN=1, 
                        regressorArgs={'type':'ridge', 'alpha':0.1}, pca_list = None, n_clusters_sampling=None, 
                        booster_params_all=None, min_pixels=None, max_pixels=None):
        print('---- in resolution_enhancer ----')
        self.minN = minN
        self.maxN = maxN
        self.n_clusters = n_clusters
        self.ac_clusters = ac_clusters
        self.pca_components = pca_components
        self.params_path = params_path
        self.apply_regression = apply_regression
        self.stopRecurseN = stopRecurseN
        self.regressorArgs = apply_regression
        self.pca_list = pca_list
        self.n_clusters_sampling = n_clusters_sampling
        self.boost_quality = booster_params_all is not None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        
        #self.apply_regression_outside = apply_regression_outside
        
        
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        self.children = []
        for N in range(self.minN, self.maxN+1):
            path = os.path.join(self.params_path, 'width-'+str(2**N))
            child = ac_leveled_gmm(N=N,n_clusters=n_clusters,ac_clusters=ac_clusters, pca_components=pca_components, params_path=path, 
                apply_regression=apply_regression, stopRecurseN=stopRecurseN, regressorArgs={'type':'ridge', 'alpha':0.1}, 
                pca_list = pca_list, n_clusters_sampling=n_clusters_sampling)
            self.children.append(child)
        if self.boost_quality:
            self.quality_boosters=[]
            for i,N in enumerate(range(self.minN, self.maxN+1)):
                booster_params = copy.deepcopy(booster_params_all[i])
                booster_params["N"] = N 
                booster_params["params_path"] = self.children[i].params_path
                self.quality_booste
    def fit(self, images):
        print('---- in resolution_enhancer fit ----')
        h,w = images.shape[1:3]
        print('images shape: ', images.shape)
        assert np.log2(h)==self.maxN
        assert h==w
        for i in range(len(self.children)):
            child = self.children[i]
            width = 2**child.N
            try:
                child.load()
            except:
                print('---- in resolution_enhancer fit except ----')
                if images.shape[2]!=width:
                    low_res_images = np.asarray([cv2.resize(im,(width,width),interpolation=cv2.INTER_LANCZOS4) for im in images])
                else:
                    low_res_images = images
                print('---- in resolution_enhancer fit except after resizing ----')
                DC_train, AC_train = child.get_DC_and_AC(low_res_images)
                child.fit(DC_train, AC_train)
            if self.boost_quality:
                try:
                    self.quality_boosters[i].load()
                except: 
                    low_res_images = np.asarray([cv2.resize(im,(width,width),interpolation=cv2.INTER_LANCZOS4) for im in images])
                    DC_train, AC_train = child.get_DC_and_AC(low_res_images)  
                    Y = child.transform(DC_train, maxN=child.N, stopSampleN=child.N-1)
                    residue = low_res_images-Y
                    self.quality_boosters[i].fit(Y, residue)

    def load(self):
        print('---- in resolution_enhancer load ----')
        for i in range(len(self.children)):
            child = self.children[i]
            child.load()
            if self.boost_quality:
                self.quality_boosters[i].load()
    
    def fit_pqr(self, images):
        print('---- in resolution_enhancer fit pqr ----')
        h,w = images.shape[1:3]
        assert np.log2(h)==self.maxN
        assert h==w
        self.rgb_pqr_objs = []
        for i in range(len(self.children)):
            child = self.children[i]
            width = 2**child.N
            if width!=w:
                images_child = np.asarray([cv2.resize(im,(width,width),interpolation=cv2.INTER_LANCZOS4) for im in images])
            else:
                images_child = images
            rgb_pqr_obj = rgb_pqr()
            rgb_pqr_obj.fit(images_child)
            self.rgb_pqr_objs.append(rgb_pqr_obj)

    def transform(self, images, sample_mean=False, stopRecurseN=1, stopSampleN=1, do_sharpen=False, do_histogram_eq=False, do_boost=False, apply_canny_mask=False, canny_mask=None):
        print('---- in resolution_enhancer transfrom ----')
        allResImages = []
        h,w = images.shape[1:3]
        assert np.log2(h)==(self.minN-1)
        assert h==w
        print("sample mean", sample_mean)
        for i,child in enumerate(self.children):
            # images = child.transform(images, sample_mean, stopRecurseN, stopSampleN=max(child.N-1,1))     # without adding canny mask for AC sampling
            
            images = child.transform(DC=images, maxN=child.N, sample_mean=sample_mean, stopRecurseN=stopRecurseN, stopSampleN=stopSampleN, apply_canny_mask=apply_canny_mask, canny_mask=canny_mask)
            
            if do_sharpen:
                assert hasattr(self, 'rgb_pqr_objs'), "you have to call enhancer.fit_pqr first"
                images =self.sharpen_images(images, self.rgb_pqr_objs[i])
            if do_histogram_eq:
                assert hasattr(self, 'rgb_pqr_objs'), "you have to call enhancer.fit_pqr first"
                images =self.histogram_equalization(images, self.rgb_pqr_objs[i])
            if do_boost:
                images = self.quality_boosters[i].predict(images)
            if self.min_pixels is not None:
                for channel in range(images.shape[3]):
                    images[:,:,:,channel] = np.clip(images[:,:,:, channel], self.min_pixels[channel], self.max_pixels[channel])
            allResImages.append(images)
        return allResImages
    
    def adjust_covariances(self, cov_multiplier):
        print('---- in resolution_enhancer adjust_covariances ----')
        for child in self.children:
            child.adjust_covariances(cov_multiplier)
    
    def histogram_equalization(self, images, rgb_pqr_obj):
        print('---- in resolution_enhancer histogram_equalization ----')
        images_pqr = rgb_pqr_obj.transform_rgb2pqr(images)
        p = images_pqr[:,:,:,0]
        clahe = cv2.createCLAHE(clipLimit=1.0 ,tileGridSize=(2,2))
        minP = np.min(p)
        maxP = np.max(p)
        p = p-minP
        p = p/(maxP-minP)
        p = p*255
        p = p.astype('uint8')
        #sharp_p = [cv2.equalizeHist(im) for im in p]
        sharp_p = [clahe.apply(im) for im in p]
        sharp_p = np.asarray(sharp_p, dtype=np.float32)
        sharp_p/=255
        sharp_p*=(maxP-minP)
        sharp_p+=minP
        images_pqr[:,:,:,0] = sharp_p
        return rgb_pqr_obj.transform_pqr2rgb(images_pqr)
    
    def sharpen_images(self, images, rgb_pqr_obj, kernel_multiplier=0.3):
        print('---- in resolution_enhancer sharpen_images ----')
        images_pqr = rgb_pqr_obj.transform_rgb2pqr(images)
        p = images_pqr[:,:,:,0]
        sharpening_kernel = kernel_multiplier*np.array([[0,-1,0],
                                      [-1,5,-1],
                                      [0,-1,0]])

        sharp_p = cv2.filter2D(src=p, ddepth=-1, kernel=sharpening_kernel) 
        images_pqr[:,:,:,0] = sharp_p
        return rgb_pqr_obj.transform_pqr2rgb(images_pqr)
    
    def generate(self, size, level=0, cov_multiplier=None, pca_components=None):
        print('---- in resolution_enhancer generate ----')
        child = self.children[level]
        samples = child.sample_dc(size=size, cov_multiplier=cov_multiplier, pca_components=pca_components)
        return samples

class quality_booster():
    def __init__(self, N, params_path, n_neighbors, patch_size=(4,4), stride=4, padding = 0, dilation=1, n_clusters=100, sample_numbers=None, per_class_samples=None, throw_threshold=1e-4):
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_clusters = n_clusters
        self.N = N
        self.params_path = params_path
        self.n_neighbors = n_neighbors
        self.sample_numbers = sample_numbers
        self.per_class_samples = per_class_samples
        self.patch_extractor = Unfold(kernel_size=patch_size, dilation=dilation, padding=padding, stride=stride)
        self.throw_threshold = throw_threshold

        if not os.path.exists(params_path):
            os.makedirs(params_path)
    
    def load(self):
        timer = timerClass()
        path_to_params = os.path.join(self.params_path, f'booster_params_nclustres_{self.n_clusters}_patch_{self.patch_size}_stride_{self.stride}_samples_{self.sample_numbers}')
        with open(path_to_params, 'rb') as foo:
            centers, labels , self.Y_patches, self.res_patches, self.knns = pickle.load(foo)
            # centers, labels , self.Y_patches, self.res_patches = pickle.load(foo)
            # self.knns = [FaissNearestNeighbors(self.n_neighbors).fit(Y_patches) for Y_patches in self.Y_patches]
        
        # all_vals = [int(knn.n_samples_fit_) for knn in self.knns]
        # print(f" knn-min: {min(all_vals)}, max: {max(all_vals)}, average: {np.mean(all_vals)}, std: {np.std(all_vals)}")

        timer.register("quality_booster----> load ----> read from file")
        self.Y_kmeans = FaissKMeans(n_clusters=self.n_clusters, init='k-means++')
        self.Y_kmeans.load(centers, labels)
        timer.register("quality_booster----> load ----> load Faiss object")
        self.set_nneighbors(self.n_neighbors)
        timer.register("quality_booster----> load ----> finished")
        timer.print()
        print("################# loaded quality booster #################")
        
    def set_nneighbors(self, n_neighbors):
        for knn in self.knns:
            knn.n_neighbors = n_neighbors

    def fit(self, Y, residue):
        try:
            self.load()
        except:
            timer = timerClass()
            assert len(Y.shape)==4
            Y_torch = Y.transpose(0,3,1,2)
            Y_torch = torch.tensor(Y_torch)
            Y_patches = self.patch_extractor(Y_torch).numpy() #<n,p*p*3, patches_per_image>
            Y_patches = Y_patches.transpose(0,2,1) #<n, patches_per_image, p*p*3>
            Y_patches = Y_patches.reshape(Y_patches.shape[0]*Y_patches.shape[1], -1) #<nxpatches_per_image, p*p*3>

            assert len(residue.shape)==4
            res_torch = residue.transpose(0,3,1,2)
            res_torch = torch.tensor(res_torch)
            res_patches = self.patch_extractor(res_torch).numpy() #<n,p*p*3, patches_per_image>
            res_patches = res_patches.transpose(0,2,1) #<n, patches_per_image, p*p*3>
            res_patches = res_patches.reshape(res_patches.shape[0]*res_patches.shape[1], -1) #<nxpatches_per_image, p*p*3>

            sum_per_patch = np.sum(np.abs(Y_patches), axis=1)
            Y_patches = Y_patches[sum_per_patch>self.throw_threshold]
            res_patches = res_patches[sum_per_patch>self.throw_threshold]
            Y_patches = np.concatenate((Y_patches,np.zeros((1,Y_patches.shape[-1]))), axis=0)
            res_patches = np.concatenate((res_patches,np.zeros((1,res_patches.shape[-1]))), axis=0)

            timer.register('quality_booster----> fit ----> patches extracted')

            
            if self.sample_numbers is not None:
                indices = np.random.permutation(Y_patches.shape[0])[0:self.sample_numbers]
                Y_patches = np.take(Y_patches, indices, axis=0)
                res_patches = np.take(res_patches, indices, axis=0)
                timer.register('quality_booster----> fit ----> patches sampled')

            path_to_params = os.path.join(self.params_path, f'booster_params_nclustres_{self.n_clusters}_patch_{self.patch_size}_stride_{self.stride}_samples_{self.sample_numbers}')
 
            self.Y_kmeans = FaissKMeans(n_clusters=self.n_clusters, init='k-means++').fit(Y_patches)
            labels = np.asarray(self.Y_kmeans.labels_)
            timer.register('quality_booster----> fit ----> kmeans learned, labels extracted')

            self.Y_patches = []
            self.res_patches = []
            self.knns = []
            for i in range(self.n_clusters):
                per_class_Y_patches = Y_patches[labels==i]
                per_class_res_patches = res_patches[labels==i]
                if self.per_class_samples is not None:
                    if per_class_Y_patches.shape[0]>self.per_class_samples:
                        indices = np.random.permutation(per_class_Y_patches.shape[0])[0:self.per_class_samples]
                        per_class_Y_patches = np.take(per_class_Y_patches, indices, axis=0)
                        per_class_res_patches = np.take(per_class_res_patches, indices, axis=0)
                self.Y_patches.append(per_class_Y_patches)
                self.res_patches.append(per_class_res_patches)
                #self.knns.append(NearestNeighbors(n_neighbors=self.n_neighbors).fit(Y_patches[labels==i]))
                self.knns.append(FaissNearestNeighbors(n_neighbors=self.n_neighbors).fit(per_class_Y_patches))
            
            timer.register('quality_booster----> fit ----> organized data based on labels')
            foo = open(path_to_params, 'wb')
            pickle.dump([self.Y_kmeans.cluster_centers_, labels, self.Y_patches, self.res_patches, self.knns],foo)
            #pickle.dump([self.Y_kmeans.cluster_centers_, labels, self.Y_patches, self.res_patches],foo)
            foo.close()
            timer.register('quality_booster----> fit ----> dumped data')
            timer.print()
            print("################# fitted quality booster #################")


            
            #centroids = []
            #for i in range(self.n_clusters):
            #    this_cluster_residues = self.res_patches[self.Y_labels==i]
            #    centroid = np.mean(this_cluster_residues, axis=0)
            #    centroids.append(centroid)
            #centroids = np.asarray(centroids)
            #self.centroids = centroids
            
    def predict(self, X, replace=False, canny_mask=None, save_output=False):
        timer = timerClass()
        assert len(X.shape)==4
        X_torch = X.transpose(0,3,1,2)
        n,c,h,w = X_torch.shape
        X_torch = torch.tensor(X_torch)
        X_patches = self.patch_extractor(X_torch).numpy() #<n,p*p*3, patches_per_image>
        X_patches = X_patches.transpose(0,2,1) #<n, patches_per_image, p*p*3>
        n, n_patches, dim = X_patches.shape
        X_patches = X_patches.reshape(X_patches.shape[0]*X_patches.shape[1], -1) #<nxpatches_per_image, p*p*3>
        labels = self.Y_kmeans.predict(X_patches)
        residues_to_add = np.zeros_like(X_patches)
        divisor = np.ones((c,h,w))
        divisor = np.expand_dims(divisor, axis=0)
        divisor_torch = torch.tensor(divisor)
        divisor_patches = self.patch_extractor(divisor_torch) #<n,p*p*3, patches_per_image>
        timer.register('quality_booster ----> predict ----> patches extracted')

        def parallel_LLE(G, R, neighbors_res):
            G = convert_to_torch(G)
            R = convert_to_torch(R)
            neighbors_res = convert_to_torch(neighbors_res)
            eye = torch.cat([torch.eye(G.size(1)).unsqueeze(0) for _ in range(R.size(0))]).to(device)
            R = R.unsqueeze(-1).unsqueeze(-1) * eye
            G = G+R
            V = torch.ones((1,G.size(2),1),dtype=G.dtype).to(device)
            WW = torch.linalg.solve(G,V)
            WW = WW/torch.sum(WW, dim=(1,2), keepdims=True)
            return convert_to_numpy(torch.matmul(WW.moveaxis(1,2),neighbors_res).squeeze(1))
        reg = 1e-3
        class_ids = np.unique(labels)
        
        A_batches = []
        neighbor_batches = []
        index_mapping = {}
        start = 0

        for i in class_ids:
            this_cluster_knn = self.knns[i]
            data = X_patches[labels==i]
            if len(data)!=0:
                dists, indices = this_cluster_knn.kneighbors(data)
                neigh = np.take(self.Y_patches[i], indices, axis=0) #<nsamples, n_neighbors, dim>
                assert neigh.shape[1]==self.n_neighbors
                A_batch = neigh - np.expand_dims(data,axis=1) #<nsamples, n_neighbors, dim>
                neigh_batch = np.take(self.res_patches[i], indices, axis=0) #<nsamples, n_neighbors, dim>
                A_batches.append(A_batch)
                neighbor_batches.append(neigh_batch)
                index_mapping[i] = (start, start+len(A_batch))
                start+=len(A_batch)
        timer.register('quality_booster ----> predict ----> knns extracted')
        A = np.concatenate(A_batches, axis=0)
        neighbors_res = np.concatenate(neighbor_batches,axis=0)
        V = np.ones((A.shape[1],1), dtype=data.dtype)
        G = np.matmul(A, A.transpose(0,2,1)) #<nsamples, n_neighbors, n_neighbors>
        traces = np.trace(G, axis1=1, axis2=2)
        R = np.zeros(G.shape[0])
        R[traces<0]=reg
        R[traces>=0]=reg*traces[traces>=0]
        computed_res = parallel_LLE(G, R, neighbors_res)
        for i in class_ids:
            if i in index_mapping.keys():
                start, end = index_mapping[i]  
                residues_to_add[labels==i] = computed_res[start:end]


        timer.register('quality_booster ----> predict ----> LLE computed')
        res_patches = residues_to_add.reshape(n,n_patches,dim)
        res_patches = res_patches.transpose(0,2,1)
        res_patches = torch.tensor(res_patches)
        res_reconst = Fold(output_size=(h,w), kernel_size=self.patch_size, dilation=self.dilation, padding=self.padding, stride=self.stride)(res_patches)
        divisor_reconst = Fold(output_size=(h,w), kernel_size=self.patch_size, dilation=self.dilation, padding=self.padding, stride=self.stride)(divisor_patches)
        res_reconst = res_reconst.numpy().transpose(0,2,3,1)
        divisor_reconst = divisor_reconst.numpy().transpose(0,2,3,1)
        res_reconst = res_reconst/divisor_reconst
        if replace:
            result = res_reconst
        else:
            if canny_mask is not None:
                res_reconst_masked = res_reconst*canny_mask
                res_reconst = (res_reconst_masked+res_reconst)/2
            result = X+res_reconst
        timer.register('quality_booster ----> predict ----> patches reshaped to image')
        timer.print()
        print("################# predicted via quality booster #################")
        if save_output:
            self.output = result
        return result

    def get_size(self):
        size_1 = np.sum([np.size(y) for y in self.Y_patches])
        size_2 = np.sum([np.size(y) for y in self.res_patches])
        size_3 = np.size(self.Y_kmeans.cluster_centers_)
        return size_1+size_2+size_3

class classwise_quality_booster():
    def __init__(self, num_classes, params_path, **kwargs):
        
        self.num_classes = num_classes
        #print('self.num_classes', self.num_classes)
        self.params_path = params_path
        #print("self.params_path: ", self.params_path)
        self.boosters = [quality_booster(params_path=self.params_path+f"/boosters/booster_{i}", **kwargs) for i in range(self.num_classes)]

    def load(self):
        for booster in self.boosters:
            booster.load()
    
    def set_nneighbors(self, n_neighbors):
        for booster in self.boosters:
            booster.set_nneighbors(n_neighbors) 

    def fit(self, Y, residue, classes):
        try:
            self.load()
        except:
            class_ids = np.unique(classes)
            num_classes = len(class_ids)
            for c, booster in enumerate(self.boosters):
                booster.fit(Y[classes==c], residue[classes==c])
                
    def predict(self, X, classes, replace, canny_mask=None):
        class_ids = np.unique(classes)
        num_classes = len(class_ids)
        output = np.zeros_like(X)
        
        for c in class_ids:
            output[classes==c] = self.boosters[c].predict(X[classes==c], replace, canny_mask)
        

        return output
    def get_size(self):
        return np.sum([c.get_size() for c in self.boosters])

class FaissNearestNeighbors():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X.astype(np.float32)))
        return self
    def kneighbors(self, X):
        distances, indices = self.index.search(np.ascontiguousarray(X.astype(np.float32)), k=self.n_neighbors)
        return distances, indices
