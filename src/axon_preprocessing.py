import datetime as dt
import json
import os
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy.signal import peak_prominences
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import butter, filtfilt
import scipy
from sklearn.decomposition import PCA
from scipy.io import loadmat
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
from scipy.signal import savgol_filter
from ruptures.base import BaseCost


# global variables, double check these match your data
from scipy.sparse import csr_matrix

# noinspection PyProtectedMember
from scipy.sparse.csgraph._traversal import connected_components

from sklearn.cluster import KMeans
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import ruptures as rpt
import re
import numpy.ma as ma

fig_path = 'D:\\Axon\\Methods paper'
data_path = 'D:\\Axon\\Analysis'


def load_axon(animal, day):

    # load imaging
    save_path = os.path.join(data_path, animal, day)
    multi_plane = os.path.join(save_path, 'suite2p', 'combined')
    # raw_red = None
    if os.path.exists(multi_plane):
        raw = np.load(os.path.join(multi_plane, 'F.npy'))
    else:
        raw = np.load(os.path.join(save_path, 'suite2p', 'plane0', 'F.npy'))
        # red_path = os.path.join(save_path, 'suite2p', 'plane0', 'F_chan2.npy')
        # if os.path.exists(red_path):
        #     raw_red = np.load(red_path)

    return raw


def save_axon(grouped_mat, group_map, save_path):

    print(f'saving to {save_path}')

    df = pd.DataFrame(grouped_mat.transpose(), columns=[f'{n}' for n in range(grouped_mat.shape[0])])
    df.to_csv(os.path.join(save_path, 'grouped_axons.csv'))

    with open(os.path.join(save_path, 'group_map.pickle'), "wb") as output_file:
        pickle.dump(group_map, output_file)

    print('saved!')


def smoothing(raw, raw_red=None, window_size = 10):
    v = np.log(raw)
    s = savgol_filter(v, window_length=window_size, polyorder=1)

    if raw_red is not None:
        v_red = np.log(raw_red)
        s_red = savgol_filter(v_red, window_length=window_size, polyorder=1)
        covs = np.array([np.cov(green, red) for green, red in zip(s, s_red)])
        s = s - s_red * (covs[:, 0, 1, None] / covs[:, 1, 1, None])

    s_reconstructed = np.exp(s)

    return s_reconstructed


class SelectROI:
    def __init__(self, transients_low = 0.03, transients_high = 0.13, drifts_low = 0.0001, drifts_high=0.01, fs=15.49):
        self.transients_low = transients_low
        self.transients_high = transients_high
        self.drifts_low = drifts_low
        self.drifts_high = drifts_high
        self.fs = fs
        self.nyquist = self.fs/2

    def select_transient_bands(self, data, power_thresh=0.3):

        # select potential rois using normalized power bands
        n_neuron, n_frames = np.shape(data)
        lowf_power = np.zeros(n_neuron)
        power_neuron = np.zeros((n_neuron, int(np.floor(n_frames / 2) + 1)))
        for neuron in range(n_neuron):  # loop over neurons
            f, Pxx = periodogram(data[neuron, :], fs=self.fs)
            power_neuron[neuron, :] = Pxx
            ind_min = np.argmax(f > self.transients_low) - 1
            ind_max = np.argmax(f > self.transients_high) - 1
            ind_drift_min = np.argmax(f > self.drifts_low)
            lowf_power[neuron] = np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max]) / np.trapz(Pxx[ind_drift_min::],
                                                                                                 f[ind_drift_min::])

        rois = np.where(lowf_power > power_thresh)[0]
        print(f'{len(rois)} rois selected using power > {power_thresh}')

        plt.hist(lowf_power)
        plt.xlabel(f'low f ({self.transients_low} to {self.transients_high}) power distribution')
        plt.ylabel('roi count')
        plt.plot([power_thresh, power_thresh], [0, len(rois)*2], 'r--')
        #plt.savefig(os.path.join(fig_path, 'roifft.svg'), format='svg')
        plt.show()

        total_power = np.sum(power_neuron, axis=1)
        norm_power = power_neuron / total_power[:, np.newaxis]
        mean_norm_power = np.mean(norm_power, axis=0)
        plt.semilogy(f, mean_norm_power, color='black', label='mean normalized power')
        mean_roi_power = np.mean(norm_power[rois, :], axis=0)
        plt.semilogy(f, mean_roi_power, color='green', label='selected rois normalized power')
        plt.legend()
        plt.xlim([self.drifts_low, self.transients_high+self.transients_low])
        plt.title('normalized power')
        plt.show()

        # mean_roi_power = np.exp(np.mean(np.log(power_neuron)[rois, :], axis=0))
        # mean_power = np.exp(np.mean(np.log(power_neuron), axis=0))
        # # std_roi_power = np.exp(np.std(np.log(power_neuron)[rois, :], axis=0))
        # plt.semilogy(f, mean_roi_power, color='green', label='selected rois')
        # plt.semilogy(f, mean_power, color='black', label='all rois')
        # plt.xlim([self.drifts_low, self.drifts_high+self.drifts_low])
        # plt.title('absolute power')
        # plt.show()

        f_range = np.where((f > self.drifts_low) & (f < self.drifts_high))[0]
        drift_power = np.sum(np.log(power_neuron[:, f_range]), axis=1)
        rois = rois[np.argsort(drift_power[rois])]

        return rois, power_neuron, f


def convert_df_f(data, quantile=0.08, scale_window=500):

    baseline = (
        pd.DataFrame(data)
        .T.rolling(scale_window, center=True, min_periods=0)
        .quantile(quantile)
        .values.T
    )

    df_f = data / baseline -1
    #df_f[df_f<0] = 0

    return df_f

def f_to_fc3(arr, axis=1, highbound=2, lowbound=0.5):
    # lowbound original setting is 0.5, highbound original is 2
    if axis == 0:
        arr = np.asarray(arr) - np.median(arr, axis=axis)
    else:
        arr = np.asarray(arr) - np.median(arr, axis=axis)[:, None]
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, axis=0)

    if axis == 0:
        arr = arr.T

    std = np.std(arr, axis=1)
    is_spike = arr > highbound * std[:, None]

    a, b = arr.shape
    for i in range(a):
        for j in range(b - 1):
            if not is_spike[i, j]:
                continue
            if is_spike[i, j + 1]:
                continue

            if arr[i, j + 1] > lowbound * std[i]:
                is_spike[i, j + 1] = True

    arr[~is_spike] = 0
    if axis == 0:
        arr = arr.T
    return arr


# def find_short_intervals(change_points, threshold):
#     change_points = np.insert(np.array(change_points), 0, 0)
#     differences = np.diff(change_points)
#     short_interval_indices = np.where(differences < threshold)[0]
#     if len(short_interval_indices)>0:
#         short_intervals = [np.arange(change_points[i], change_points[i + 1]) for i in short_interval_indices]
#         return np.hstack(short_intervals)
#     else:
#         return []


def find_short_intervals(change_points, thresh):
    # Calculate the differences between consecutive elements
    np_array = np.insert(np.array(change_points), 0, 0)
    differences = np.diff(np_array)

    # Find the indices where the difference is smaller than the threshold
    indices = np.where(differences < thresh)[0]

    # Extract the elements that have differences smaller than the threshold
    # Include both elements in each pair (np_array[i] and np_array[i + 1])
    pairs = [(np_array[i], np_array[i + 1]) for i in indices]

    return pairs


def find_transients(fc3, rois, prominence_thresh=0.1, amplitude=0.12, interval=10, width_min=5, width_max=100):

    transients_mat = np.zeros_like(fc3)

    for n in range(fc3.shape[0]):
        peaks, properties = find_peaks(fc3[n, :], prominence=prominence_thresh, height=amplitude, distance=interval,
                                       width=[width_min, width_max])
        for left_idx, right_idx in zip(properties['left_bases'], properties['right_bases']):
            transients_mat[n, left_idx:right_idx+1] = fc3[n, left_idx:right_idx+1]

    final_rois = np.where(np.sum(transients_mat, axis=1)>0)[0]

    return transients_mat[final_rois, :], rois[final_rois]


def group_columns(data, n_comp, group_ids, rois):

    grouped_components = []
    group_map = {}
    pca = PCA(n_components=1)

    for gid in range(n_comp):
        idx = np.where(group_ids == gid)[0]
        grouping_ind = list(idx)
        group_map[gid] = rois[grouping_ind]
        roi_group = data[idx].T
        if roi_group.shape[-1] == 1:
            result = roi_group
        else:
            result = pca.fit_transform(roi_group)
            result /= pca.components_.sum(1)
        result = result - np.quantile(result, 0.05)
        grouped_components.append(np.squeeze(result))

    return np.vstack(grouped_components), group_map

class AxonImaging:

    def __init__(self, animal, day, env):

        self.animal = animal
        self.day = day
        self.path = os.path.join(data_path, animal, day)
        self.env = env
        self._beh_params = None

    @property
    def beh_params(self):
        if self._beh_params is None:
            self._load_beh_params()
        return self._beh_params

    def _load_beh_params(self):
        pattern = re.compile(r'.*-all-cond(\d?)\.mat$')
        onlyfiles = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        beh_file_name = [f for f in onlyfiles if pattern.match(f)][0]
        print(f'found beh file: {beh_file_name} '
              f'\namong files: {onlyfiles} ')

        axon_beh = scipy.io.loadmat(os.path.join(self.path, beh_file_name))
        self._beh_params = {
            'switch_frame': axon_beh['end_frame'][:, 0],
            'start_frame': axon_beh['start_frame'][:, 0],
            'ybinned': axon_beh['behavior']['ybinned'][0, :][0][0, 1:],
            'lap': axon_beh['E'][0, 1:]
        }

        assert len(self.env) == len(self._beh_params['switch_frame']), "# of env does not match with behavioral file"

    def _save_dict(self, data_dict: {}):

        filename = os.path.join(self.path, f'{self.animal}_{self.day}_roi_power')
        with open(filename, 'wb') as file:
            pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'saved to {filename}')

    def _load_dict(self):
        filename = os.path.join(self.path, f'{self.animal}_{self.day}_roi_power')
        if not os.path.exists(filename):
            print(f'{filename} does not exist. '
                  f'\nrun self.rank_rois() first.')
            return None

        with open(filename, 'rb') as file:
            return pickle.load(file)

    def rank_rois(self, select_roi: SelectROI, smooth_window = 10):

        raw = load_axon(self.animal, self.day)
        self._load_beh_params()
        raw_dict = dict(zip(self.env, np.split(raw, self._beh_params['switch_frame'], axis=1)[:-1]))
        proc_dict = {}

        for env, data in raw_dict.items():
            s = smoothing(data, window_size=smooth_window)
            rois, power_roi, f = select_roi.select_transient_bands(s, power_thresh=0.3)
            proc_dict[env] = dict(zip(['data', 'roi', 'power_roi', 'freq'], [s, rois, power_roi, f]))

        self._save_dict(proc_dict)

        return proc_dict

    def remove_common_drift(self, proc_dict=None, cost_fun = None, plot_n = 5, max_drift_frames = 500):

        pca = PCA(n_components=1)

        if cost_fun is None:
            model = rpt.BottomUp(model='l2')
        else:
            model = rpt.BottomUp(custom_cost=cost_fun)

        if proc_dict is None:
            proc_dict = self._load_dict()

        for env in proc_dict:
            data = proc_dict[env]['data']
            rois = proc_dict[env]['roi']

            z_data = scipy.stats.zscore(data[rois, :], axis=1)
            z1 = pca.fit_transform(z_data.transpose())

            plt.plot(z1)
            plt.title(f'{self.animal} {self.day} in {env} PCA 1st component')
            plt.show()

            n_breakpoints = int(input(f"{self.animal} {self.day} in {env} # breakpoints to find"))

            if n_breakpoints>0:

                algo_binseg = model.fit(z1)
                optimal_change_points = algo_binseg.predict(n_bkps=n_breakpoints)
                print(optimal_change_points)
                outlier_indices = find_short_intervals(optimal_change_points, thresh=max_drift_frames)
                rpt.display(z1, optimal_change_points)
                #plt.plot(outlier_indices, [0]*len(outlier_indices), color='black', linewidth=3)
                plt.title(f'{self.animal} {self.day} in {env} {n_breakpoints} breakpoints')
                plt.savefig(os.path.join(self.path, f'{self.animal}_{self.day}_{env}_ROI_PCA.svg'))
                plt.show()
                print(outlier_indices)

                for n in range(plot_n):
                    rpt.display(z_data[n, :], optimal_change_points)
                    plt.title(f'cell {rois[n]}: {n_breakpoints} breakpoints')
                    plt.savefig(os.path.join(self.path, f'{self.animal}_{self.day}_{env}_roi{rois[n]}.svg'))
                    plt.show()

                verified = int(input(f"are breakpoints reasonable in {self.animal} {self.day} in {env}? 0: no, 1: yes"))

                if verified == 1:
                    proc_dict[env]['del frames'] = outlier_indices
                    self._save_dict(proc_dict)

                else:
                    print(f'user interrupt: {self.animal} {self.day} in {env}')

        return proc_dict

    def plot_choose_rois(self, proc_dict=None, worst_n=10):

        if proc_dict is None:
            proc_dict = self._load_dict()

        for env in proc_dict:
            selected_rois = worst_n
            plotted_rois = 1

            while selected_rois >= plotted_rois:
                worst_rois = proc_dict[env]['roi'][-plotted_rois: -selected_rois: -1]

                for n in range(len(worst_rois)):
                    plt.plot(proc_dict[env]['data'][worst_rois[n], :])
                    plt.title(f'{self.animal} {self.day} in {env} worst {n + plotted_rois} roi: {worst_rois[n]}')
                    plt.savefig(os.path.join(self.path, f'{self.animal}_{self.day}_{env}_roi{worst_rois[n]}'))
                    plt.show()

                plotted_rois = plotted_rois + n
                selected_rois = int(input(f"{self.animal} {self.day} in {env} worst # rois to delete"))

            proc_dict[env]['del roi'] = selected_rois

        self._save_dict(proc_dict)

        return proc_dict

    def combine_rois(self, proc_dict=None, corr_thresh=0.7):

        if proc_dict is None:
            proc_dict = self._load_dict()

        final_data_list = []
        deleted_frames = []
        total_frames = 0

        for env, proc_data in proc_dict.items():
            rois = proc_data['roi']

            if 'del roi' in proc_data:
                print(f'found rois to delete in {self.animal} {self.day} {self.env}')
                del_rois = proc_data['del roi']
                rois = rois[:-del_rois]

            data = proc_data['data'][rois, :]

            if 'del frames' in proc_data:
                print(f'found frames to delete in {self.animal} {self.day} {self.env}')
                #del_frames = proc_data['del frames']
                del_frames = np.hstack([np.arange(n[0], n[1]) for n in proc_data['del frames']])
                data = np.delete(data, np.unique(del_frames), axis=1)
                deleted_frames.append(del_frames+total_frames)

            data = convert_df_f(data)
            data = f_to_fc3(data)
            data, selected_rois = find_transients(data, rois)
            final_mat = np.zeros((proc_data['data'].shape[0], data.shape[1]))
            final_mat[selected_rois, :] = data
            final_data_list.append(final_mat)
            total_frames = total_frames+proc_data['data'].shape[1]

        final_mat = np.hstack(final_data_list)

        if len(self.env) == 1:
            combined_mat, group_map = self._group_by_cluster(final_mat, corr_thresh=corr_thresh)
        elif len(self.env) > 1:
            combined_mat, group_map = self._group_by_corr(final_mat, corr_thresh=corr_thresh)
        else:
            print(f'error {self.animal} {self.day}')


        if len(deleted_frames)>0:
            deleted_frames = np.hstack(deleted_frames)
            combined_mat = np.insert(combined_mat, deleted_frames-np.arange(len(deleted_frames)), np.nan, axis=1)

        save_axon(combined_mat, group_map, self.path)

        df = self._add_behavior(combined_mat)

        return combined_mat, group_map, df

    def _group_by_cluster(self, final_mat, corr_thresh=0.7, stopping_thresh_perc=10, min_best_score=0.4):

        active_rois = np.where(np.sum(final_mat, axis=1) > 0)[0]
        cov_nov = np.corrcoef(final_mat[active_rois, :])
        np.fill_diagonal(cov_nov, np.nan)
        max_corr = np.nanmax(cov_nov, axis=0)

        grouping_rois = np.where(max_corr > corr_thresh)[0]
        singleton_rois = np.where((max_corr>0) & (max_corr<corr_thresh))[0]

        overall_corr = cov_nov[grouping_rois, :]
        overall_corr = overall_corr[:, grouping_rois]
        np.fill_diagonal(overall_corr, 1)
        n_max_cluster = len(grouping_rois)
        print(f'{self.animal} {self.day} # potential non-singleton ROIs: {n_max_cluster}, '
              f'# singleton ROIs: {len(singleton_rois)}')
        plt.hist(max_corr)
        plt.title('max corr per ROI')
        plt.show()

        cluster_number_vector = np.arange(np.max((int(n_max_cluster / 100), 2)), n_max_cluster)
        silhouette_all = np.zeros((len(cluster_number_vector)))
        clusters_all = np.zeros((len(cluster_number_vector), n_max_cluster))
        stopping_thresh = int(np.ceil(n_max_cluster * stopping_thresh_perc/100))

        # Scale the flattened covariance matrices
        scaler = StandardScaler()
        flattened_cov_matrices_scaled = scaler.fit_transform(overall_corr)

        best_k = -1
        best_score = -1
        dropping_count = 0

        for i_cluster, n_cluster in enumerate(cluster_number_vector):
            h_clustering = AgglomerativeClustering(n_clusters=n_cluster)
            # overall_corr_selected = np.squeeze(overall_corr_corrected[:,:,ind])[~np.isnan(np.squeeze(overall_corr_corrected[:,:,ind]))]
            # n_comp, group_ids = connected_components(csr_matrix(overall_corr_corrected[:,:,ind]), directed=False, return_labels=True
            #    )
            # print(n_comp)

            clusters = h_clustering.fit_predict(flattened_cov_matrices_scaled)
            clusters_all[i_cluster, :] = clusters

            # Calculate silhouette score
            silhouette_avg = silhouette_score(flattened_cov_matrices_scaled, clusters)
            silhouette_all[i_cluster] = silhouette_avg

            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = i_cluster
                dropping_count = 0
                print(f"cluster {n_cluster} Silhouette Score: {silhouette_avg}, new highest")
            else:
                dropping_count = dropping_count + 1
                print(f"cluster {n_cluster} Silhouette Score: {silhouette_avg}, counting {dropping_count}")

            if dropping_count >= stopping_thresh:
                break

        plt.plot(silhouette_all[:i_cluster], label='Silhouette', color='tab:cyan')
        plt.scatter(best_k, best_score, color='red')
        plt.title(f'Hierarchical Clustering in {self.animal} {self.day}')
        plt.xlabel('Num of Clusters')
        plt.ylabel('Performance Metrics')
        plt.xlim([len(overall_corr) / 100, i_cluster])
        plt.legend()
        plt.savefig(os.path.join(self.path, f'{self.animal}_{self.day}_cluster_silhouette.png'))
        plt.show()

        if best_score < min_best_score:
            print(f'{self.animal} {self.day} no significant cluster groups found, start grouping by correlation')
            n_comp, group_ids = AxonImaging._group_by_corr(final_mat, corr_thresh)
            return n_comp, group_ids

        print(f'{self.animal} {self.day} best K: {best_k}, best Silhouette: {best_score}')

        grouped_mat, group_map = group_columns(final_mat[active_rois[grouping_rois], :], best_k, clusters_all[best_k],
                                               active_rois[grouping_rois])
        combined_mat = np.vstack([grouped_mat, final_mat[active_rois[singleton_rois], :]])
        group_map.update(dict(zip(np.arange(best_k, best_k+len(singleton_rois)), active_rois[singleton_rois])))

        return combined_mat, group_map  # , grouping_df.sort_values(by='raw roi').sort_values(by='group id')

    @staticmethod
    def _group_by_corr(final_mat, corr_thresh):

        active_rois = np.where(np.sum(final_mat, axis=1) > 0)[0]
        cov_nov = np.corrcoef(final_mat[active_rois, :])
        n_comp, group_ids = connected_components(csr_matrix(cov_nov > corr_thresh), directed=False,
                                                 return_labels=True)
        combined_mat, group_map = group_columns(final_mat[active_rois, :], n_comp, group_ids, active_rois)

        return combined_mat, group_map

    def _add_behavior(self, imaging_mat):
        # Create DataFrame from imaging_mat
        df = pd.DataFrame(imaging_mat.transpose(), columns=[str(n) for n in range(imaging_mat.shape[0])])

        # Set 'env' column based on behavioral parameters
        for start_frame, switch_frame, env_value in zip(self.beh_params['start_frame'],
                                                        self.beh_params['switch_frame'],
                                                        self.env):
            df.loc[start_frame:switch_frame, 'env'] = env_value

        # Drop rows with NaN in 'env' column
        df.dropna(subset=['env'], inplace=True)

        # Add additional columns
        df['lap'] = self.beh_params['lap']
        df['ybinned'] = self.beh_params['ybinned']
        df['mouse'] = self.animal
        df['day'] = self.day

        # Clean up lap values and save to Parquet file
        df['lap'] = df['lap'].sub(1).where(df['lap'].ne(0))  # Subtract 1 from 'lap' and replace 0 with NaN
        df = df.dropna(subset=['lap'])  # Drop rows with NaN in 'lap' column
        df = df.loc[:, (df != 0).any(axis=0)]   # drop columns with only zeros
        df.to_parquet(os.path.join(self.path, f'{self.animal}_{self.day}_axon.parquet'), compression='gzip')

        print(f'Saved to parquet in folder {self.path}')
        return df
