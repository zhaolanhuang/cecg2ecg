from joblib import Parallel, delayed
import numpy as np
import wfdb
import wfdb.processing
import scipy.io
import logging as log


def pad_to_next_pow_2(sig):
    n = sig.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2,y+1))
    sig_padded = np.pad(sig, ((0, nextpow2-n), (0,0)), mode='constant')
    return sig_padded

def resample_measurement(meas_idx, cECG1, cECG2, cECG3, refECG, ann_manual1, ann_manual2, ann_manual3, ann_osea, fs, storage_path):
    if (fs != 250):
        log.debug('Meas {}: build annotations'.format(meas_idx))
        symbol = list()
        for c in range(0, ann_manual1.shape[0]):
         symbol.append('N')
        ann_manual1 = ann_manual1[:, 0]
        wfdb.wrann('ann1_test', 'ann', ann_manual1, symbol)

        symbol = list()
        for c in range(0, ann_manual2.shape[0]):
         symbol.append('N')
        ann_manual2 = ann_manual2[:, 0]
        wfdb.wrann('ann2_test', 'ann', ann_manual2, symbol)

        ''' channel 3 is not annotated!
        symbol = list()
        for c in range(0, ann_manual3.shape[0]):
         symbol.append('N')
        ann_manual3 = ann_manual3[:, 0]
        wfdb.wrann('ann3_test', 'ann', ann_manual3, symbol)
        '''

        symbol = list()
        for c in range(0, ann_osea.shape[0]):
         symbol.append('N')
        ann_osea = ann_osea[:, 0]
        wfdb.wrann('ann4_test', 'ann', ann_osea, symbol)

        ann_manual1 = wfdb.rdann('ann1_test', 'ann')
        ann_manual2 = wfdb.rdann('ann2_test', 'ann')
        #ann_manual3 = wfdb.rdann('ann3_test', 'ann')
        ann_osea = wfdb.rdann('ann4_test', 'ann')

        log.debug('Meas {}: resampling {}'.format(meas_idx, 'cECG1'))
        n_resampled = int(np.floor(cECG1.shape[0] * 250/fs))
        sig_padded = pad_to_next_pow_2(cECG1)
        temp_sig0, annotation_man1 = wfdb.processing.resample_singlechan(sig_padded[:,0], ann_manual1, fs, 250)
        temp_sig0 = temp_sig0[0:n_resampled]

        log.debug('Meas {}: resampling {}'.format(meas_idx, 'cECG2'))
        n_resampled = int(np.floor(cECG2.shape[0] * 250/fs))
        sig_padded = pad_to_next_pow_2(cECG2)
        temp_sig1, annotation_man2 = wfdb.processing.resample_singlechan(sig_padded[:,0], ann_manual2, fs, 250)
        temp_sig1 = temp_sig1[0:n_resampled]

        log.debug('Meas {}: resampling {}'.format(meas_idx, 'cECG3'))
        n_resampled = int(np.floor(cECG3.shape[0] * 250/fs))
        sig_padded = pad_to_next_pow_2(cECG3)
        # using ann_osea instead of ann_manual3 since channel 3 is not annotated!
        temp_sig2, annotation_man3 =  wfdb.processing.resample_singlechan(sig_padded[:,0], ann_osea, fs, 250)
        temp_sig2 = temp_sig2[0:n_resampled]
        
        log.debug('Meas {}: resampling {}'.format(meas_idx, 'refECG'))
        n_resampled = int(np.floor(refECG.shape[0] * 250/fs))
        sig_padded = pad_to_next_pow_2(refECG)
        temp_sig_ref, annotation_osea =  wfdb.processing.resample_singlechan(sig_padded[:,0], ann_osea, fs, 250)
        temp_sig_ref = temp_sig_ref[0:n_resampled]

        sig = np.zeros((int(temp_sig0.shape[0]), 3))
        sig[:, 0] = temp_sig0
        sig[:, 1] = temp_sig1
        sig[:, 2] = temp_sig2
    else:
        log.debug('Meas {}: sampling rate already at 250 Hz')
        sig = np.zeros((int(cECG1.shape[0]), 3))
        sig[:, 0] = cECG1[:,0]
        sig[:, 1] = cECG2[:,0]
        sig[:, 2] = cECG3[:,0]
        annotation_man1 = ann_manual1
        annotation_man2 = ann_manual2
        annotation_osea = ann_osea

    log.info('Meas {}: writing results to {}'.format(meas_idx, storage_path))
    np.save(storage_path + 'cECG1' + 'UnoViS_auto2012_' + str(meas_idx+1) + '.hea.npy', temp_sig0)
    np.save(storage_path + 'cECG2' + 'UnoViS_auto2012_' + str(meas_idx+1) + '.hea.npy', temp_sig1)
    np.save(storage_path + 'cECG3' + 'UnoViS_auto2012_' + str(meas_idx+1) + '.hea.npy', temp_sig2)
    np.save(storage_path + 'annotations_man_cECG1_' + str(meas_idx+1), annotation_man1)
    np.save(storage_path + 'annotations_man_cECG2_' + str(meas_idx+1), annotation_man2)
    #np.save(storage_path + 'annotations_man_cECG3_' + str(meas_idx+1), annotation_man2)
    np.save(storage_path + 'annotations_UnoViS_auto2012_' + str(meas_idx+1) + '.hea.npy', annotation_osea.sample)

def resample_unovis_auto2012(unovis_auto2012_mat_path, resampled_storage_path):
    mat = scipy.io.loadmat(unovis_auto2012_mat_path)

    meas_idx = list(range(0,31))
    cECG1_ls = list();
    cECG2_ls = list();
    cECG3_ls = list();
    refECG_ls = list();

    ann_manual1_ls = list();
    ann_manual2_ls = list();
    ann_manual3_ls = list();
    ann_osea_ls = list();
    fs_ls = list();

    for i in meas_idx:
        cECG1_ls.append(mat['unovis'][0, i]['channels'][0, 0]['data'])
        cECG2_ls.append(mat['unovis'][0, i]['channels'][0, 1]['data'])
        cECG3_ls.append(mat['unovis'][0, i]['channels'][0, 2]['data'])
        refECG_ls.append(mat['unovis'][0, i]['channels'][0, 3]['data'])

        ann_manual1_ls.append(mat['unovis'][0, i]['channels'][0, 0]['ann'][0, 1]['loc'])
        ann_manual2_ls.append(mat['unovis'][0, i]['channels'][0, 1]['ann'][0, 1]['loc'])
        ann_manual3_ls.append(mat['unovis'][0, i]['channels'][0, 2]['ann'][0, 1]['loc'])
        ann_osea_ls.append(mat['unovis'][0, i]['channels'][0, 3]['ann'][0, 0]['loc'])
        fs_ls.append(mat['unovis'][0, i]['channels'][0, 0]['fs'])


    results = Parallel(n_jobs=8)(delayed(resample_measurement)(i, cECG1, cECG2, cECG3, refECG, ann_manual1, ann_manual2, ann_manual3, ann_osea, fs, resampled_storage_path) for i, cECG1, cECG2, cECG3, refECG, ann_manual1, ann_manual2, ann_manual3, ann_osea, fs in zip(meas_idx, cECG1_ls, cECG2_ls, cECG3_ls, refECG_ls, ann_manual1_ls, ann_manual2_ls, ann_manual3_ls, ann_osea_ls, fs_ls))