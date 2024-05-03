########################
# Author: Zohreh Azizi #
########################

import numpy as np
import os 
import pickle
import cv2 
import plot_figures as pf
import data
from eval_metrics import calculate_FID
import copy
from generation_modules import classwise_ac_leveled_gmm, classwise_quality_booster



if __name__ == '__main__':
    datasets_dir = './datasets/'
    ds_name = 'celeba32'
    #ds_name = 'celeba64'
    #ds_name = 'celeba128'
    n_clusters=100
    n_clusters_booster=100
    n_clusters_sampling=500
    ac_clusters=3
    per_class_samples_booster = None#500
    pca_components = None#0.85 #pca for generation core
    gen_cov_multiplier = 0.5 #covariance multiplier for generation core
    enh_cov_multiplier = 0#0.05 #covariance multiplier for resolution enhancer 
    n_neighbors = 2 #neighbors for booster
    apply_canny_booster = True 
    apply_canny_enhancer = True
    apply_regression_enhancer = True
    minN = 3
    maxN = 5
    apply_resize = False
    apply_pad = False
    replace = False
    reg_covar = 1e-6
    selected_attribute = 'Male'

    images_train, labels_train, images_test, labels_test, _, _ = data.import_data2(datasets_dir, ds_name)

    dsname = ''.join(c for c in ds_name if not c.isdigit())
    with open(os.path.join(datasets_dir, dsname,'list_attr_celeba.txt'), 'r') as f:
        attributes = f.readlines()[1].split()
    attr_index = attributes.index(selected_attribute)

    labels_train = labels_train[:,attr_index]
    labels_test = labels_test[:,attr_index]

    labels_train = ((labels_train+1)/2).astype('int32')
    labels_test = ((labels_test+1)/2).astype('int32')

    labels_train = labels_train*0
    labels_test = labels_test*0

    training_size = 100000
    images_train = images_train[0:training_size]
    labels_train = labels_train[0:training_size]

    test_size = 10000#images_test.shape[0]
    images_test = images_test[0:test_size]
    labels_test = labels_test[0:test_size]

    print('labels_train', labels_train)
    print('labels_test', labels_test)

    params_path = f'./params_{training_size}_{ds_name}'

    #gen_classes = np.random.randint(low=0, high=2, size=test_size)
    gen_classes = np.random.randint(low=0, high=1, size=test_size)
    num_classes=1
    #gen_classes = labels_test
    print('gen_classes',gen_classes)


    if apply_resize:
        images_test = np.asarray([cv2.resize(im,(2**(minN-1),2**(minN-1)),interpolation=cv2.INTER_LANCZOS4) for im in images_test])
    elif apply_pad:
        images_test = np.asarray([cv2.copyMakeBorder(im,2,2,2,2,cv2.BORDER_CONSTANT,(0,0,0)) for im in images_test])
        images_test = np.asarray([cv2.resize(im,(2**(minN-1),2**(minN-1)),interpolation=cv2.INTER_LANCZOS4) for im in images_test])
    if len(images_test.shape)<4:
        images_test = np.expand_dims(images_test,axis=-1)
    
    for n in range(minN, maxN+1):
        child_param_dir = os.path.join(params_path,f'width-{2**n}')
        if not os.path.exists(child_param_dir):
            os.makedirs(child_param_dir)
        child_fig_dir = os.path.join(child_param_dir,f'figures')

        
        if 2**(n)!=images_train.shape[1]:
            if apply_resize:
                images_train_GT = np.asarray([cv2.resize(im,(2**n,2**n),interpolation=cv2.INTER_LANCZOS4) for im in images_train])
            elif apply_pad:
                images_train_GT = np.asarray([cv2.copyMakeBorder(im,2,2,2,2,cv2.BORDER_CONSTANT,(0,0,0)) for im in images_train])
            else:
                images_train_GT = np.asarray([cv2.resize(im,(2**n,2**n),interpolation=cv2.INTER_LANCZOS4) for im in images_train])        
        else:
            images_train_GT = images_train

        if len(images_train_GT.shape)<4:
            images_train_GT = np.expand_dims(images_train_GT,axis=-1)
            

        if 'mnist' in ds_name:
            stopRecurseN = n-1
        else:
            stopRecurseN = 1

        if 'mnist' in ds_name:
            stopSampleN = n-1
        else:
            stopSampleN = 1
        
        child = classwise_ac_leveled_gmm(N=n, 
                                        n_clusters=n_clusters, 
                                        ac_clusters=ac_clusters, 
                                        pca_components=None,#[pca_components], 
                                        params_path=child_param_dir, 
                                        apply_regression=apply_regression_enhancer, 
                                        reg_covar = reg_covar,
                                        sharpen_approx_AC=False,
                                        stopRecurseN=stopRecurseN, 
                                        regressorArgs={'type':'ridge', 'alpha':0.1}, 
                                        pca_list = None, 
                                        n_clusters_sampling=n_clusters_sampling)
        DC, AC = child.get_DC_and_AC(images_train_GT)
        child.fit(DC, AC, labels_train)


        train_residue_fn = os.path.join(child_param_dir, f'residue_train')
        train_enhanced_fn = os.path.join(child_param_dir, f'enhanced_train')

       
        
        if os.path.exists(train_residue_fn) and os.path.exists(train_enhanced_fn):
            foo = open(train_residue_fn, 'rb')
            residue = pickle.load(foo)
            foo.close()
            foo = open(train_enhanced_fn, 'rb')
            enhanced = pickle.load(foo)
            foo.close()
        else:
            images_enhanced = child.transform(DC, labels_train, maxN=n, sample_mean=True, stopRecurseN=stopRecurseN, stopSampleN=stopSampleN, apply_canny_mask=False, canny_mask=None)
            residue = images_train_GT-images_enhanced
            foo = open(train_residue_fn, 'wb')
            pickle.dump(residue, foo)
            foo.close()
            foo = open(train_enhanced_fn, 'wb')
            pickle.dump(images_enhanced, foo)
            foo.close()
            enhanced = images_enhanced

        patch_size_1 = 2**3
        stride_1 = 2**3
        QualityBooster_1 = classwise_quality_booster(num_classes=num_classes, params_path = child_param_dir, N=n, n_neighbors = n_neighbors, patch_size=(patch_size_1,patch_size_1), stride=stride_1, padding=0, dilation=1, n_clusters=n_clusters_booster, per_class_samples=per_class_samples_booster)
        
        try:
            QualityBooster_1.load()
            print("quality_booster loaded!!")
        except: 
            if replace:
                QualityBooster_1.fit(enhanced, residue+enhanced, labels_train)
            else:
                QualityBooster_1.fit(enhanced, residue, labels_train)
            print("quality_booster fitted!!")


        if n==minN:
            #images_test = images_test
            #new_images_test = [cv2.resize(im,(2**(n-1),2**(n-1)),interpolation=cv2.INTER_LANCZOS4) for im in images_test]
            #images_test = np.asarray(new_images_test)
            #gen_classes = labels_test
            images_test =child.sample_dc(gen_classes, cov_multiplier=gen_cov_multiplier, pca_components=pca_components)
            #boosted_fn = os.path.join(params_path, f'width-64/boosted_test_32_64')
            #foo = open(boosted_fn, 'rb')
            #images_test = pickle.load(foo)
            #foo.close()
        else:
            images_test = boosted

        fid_score_low_res = None#calculate_FID(images_test, ds_name)
        
        
        pf.plot_images(images_test[0:100],10,10,figsize=(20,20),show=False,save=True,fdir=child_fig_dir,fn_pre=f'low_res_fid_{fid_score_low_res}')
        
        #exit()
        if enh_cov_multiplier==0:
            images_enhanced = child.transform(images_test,gen_classes,maxN=n, sample_mean=True, stopRecurseN=stopRecurseN, stopSampleN=n-1, apply_canny_mask=apply_canny_enhancer, canny_mask=None)
        else:
            child.adjust_covariances(enh_cov_multiplier)
            images_enhanced = child.transform(images_test,gen_classes,maxN=n, sample_mean=False, stopRecurseN=stopRecurseN, stopSampleN=n-1, apply_canny_mask=apply_canny_enhancer, canny_mask=None)
            child.adjust_covariances(1/enh_cov_multiplier)

        fid_score_enhanced = None#calculate_FID(images_enhanced, ds_name)
        
        if apply_canny_enhancer:
            fn_enhanced = "enhanced_canny_"
        else:
            fn_enhanced = "enhanced_"

        pf.plot_images(images_enhanced[0:100],10,10,figsize=(20,20),show=False,save=True,fdir=child_fig_dir,fn_pre=fn_enhanced+f'fid_{fid_score_enhanced}')
        #boosted = images_enhanced
        enhanced_fn = os.path.join(child_param_dir, f'enhanced_test')
        foo = open(enhanced_fn, 'wb')
        pickle.dump(images_enhanced, foo)
        foo.close()
        
        # Canny mask
        images_enhanced_cp = copy.deepcopy(images_enhanced)
        images_enhanced_cp[images_enhanced_cp<0]=0
        images_enhanced_cp[images_enhanced_cp>1]=1
        images_enhanced_cp = images_enhanced_cp*255
        images_enhanced_cp = images_enhanced_cp.astype('uint8')
        enhanced_edge_canny = [cv2.Canny(image=im, threshold1=50, threshold2=50) for im in images_enhanced_cp]
        enhanced_edge_canny = np.asarray(enhanced_edge_canny)
        enhanced_edge_canny = enhanced_edge_canny.astype('float64')/255
        enhanced_edge_canny = np.expand_dims(enhanced_edge_canny, axis=3)
        boosted = QualityBooster_1.predict(images_enhanced, gen_classes, replace, apply_canny_booster)
        print('boosteed shape', boosted.shape)

        if apply_resize:
            boosted_resized = [cv2.resize(im,(28,28),interpolation=cv2.INTER_LANCZOS4) for im in boosted]
            boosted_resized = np.asarray(boosted_resized)
            boosted_resized = np.expand_dims(boosted_resized, axis=3)
            print('boosted_resized shape', boosted_resized.shape)
            print('calculating FID in 28 x 28...')
        elif apply_pad:
            boosted_resized = boosted[:,2:-2,2:-2,:]    
            print('boosted_resized shape', boosted_resized.shape)
            print('calculating FID in 28 x 28...') 
        else:
            boosted_resized = boosted
        if n==5:
            fid_score_boost = calculate_FID(boosted_resized, ds_name)
        else:
            fid_score_boost = None
        


        if apply_canny_booster:
            fn_boosted = fn_enhanced+"boosted_canny_"
        else:
            fn_boosted = fn_enhanced+"boosted_"

        pf.plot_images(boosted_resized[0:100],10,10,figsize=(20,20),show=False,save=True,fdir=child_fig_dir,fn_pre=fn_boosted+f'fid_{fid_score_boost}')
        
        boosted_fn = os.path.join(child_param_dir, f'boosted_test')
        foo = open(boosted_fn, 'wb')
        pickle.dump(boosted, foo)
        foo.close()

        boosted_classes_fn = os.path.join(child_param_dir, f'boosted_classes')
        foo = open(boosted_classes_fn, 'wb')
        pickle.dump(gen_classes, foo)
        foo.close()

        print('----FID score:', fid_score_boost)
        