import os
import numpy as np
import pickle
import argparse
import sys
#import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
#import visualization
import utils

def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(BASELINE_DIR, os.pardir))

# Exection:
# python evaluate.py --normal_results_path '../../data/results/k128_s007_nostd_sumd_pt32_pl32_num/'

parser = argparse.ArgumentParser()
parser.add_argument('--normal_results_path', default='../../data/results/ss_0.05_normal/', help='Log dir [default: log]')
parser.add_argument('--data_path', type=str, default='/data/pclouds/', help='Relative path to data directory')
# parser.add_argument('--normal_results_path', default='./data/results/k256_s007_nostd_sumd_pt32_pl32_dist_c/', help='Log dir [default: log]')
# parser.add_argument('--data_path', type=str, default='/data/nyuv2_surfacenormal_metadata/pcloud_test_5000/', help='Relative path to data directory')
parser.add_argument('--sparse_patches', type=int, default=False, help='True for pclouds, False for nyuv2. Evaluate on a sparse subset or on the entire point cloud')
parser.add_argument('--dataset_list', type=str, default=['testset_vardensity_striped'], nargs='+',
                    help='list of .txt files containing sets of point cloud names for evaluation')
FLAGS = parser.parse_args()
EXPORT = False  # export some visualizations
PC_PATH =  FLAGS.data_path
normal_results_path =  FLAGS.normal_results_path
results_path = os.path.abspath(os.path.join(normal_results_path, os.pardir))
sparse_patches = FLAGS.sparse_patches

if not os.path.exists(normal_results_path):
    ValueError('Incorrect normal results path...')

#dataset_list = FLAGS.dataset_list
dataset_list = ['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise',
                'testset_vardensity_gradient','testset_vardensity_striped']  # list of files with evaluation lists, 'testset_vardensity_striped'
#dataset_list = ['testset_all'] # for NYUv2 data

RMS_not_oriented = []
for dataset in dataset_list:

    normal_gt_filenames = PC_PATH + dataset + '.txt'

    normal_gt_path = PC_PATH

    # get all shape names in the dataset
    shape_names = []
    with open(normal_gt_filenames) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    outdir = os.path.join(normal_results_path, 'summary/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    LOG_FOUT = open(os.path.join(outdir, dataset + '_evaluation_results.txt'), 'w')
    if EXPORT:
        file_path = os.path.join(normal_results_path, 'images')
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)

    experts_exist = False
    rms = []
    rms_o = []
    all_ang = []
    pgp10 = []
    pgp5 = []
    for i, shape in enumerate(shape_names):
        print('Processing ' + shape + '...')

        if EXPORT:
            # Organize the output folders
            idx_1 = shape.find('_noise_white_')
            idx_2 = shape.find('_ddist_')
            if idx_1 == -1 and idx_2 == -1:
                base_shape_name = shape
            elif idx_1 == -1:
                base_shape_name = shape[:idx_2]
            else:
                base_shape_name = shape[:idx_1]

            vis_output_path = os.path.join(file_path, base_shape_name)
            if not os.path.exists(vis_output_path):
                os.makedirs(vis_output_path)
            gt_normals_vis_output_path = os.path.join(vis_output_path, 'normal_gt')
            if not os.path.exists(gt_normals_vis_output_path):
                os.makedirs(gt_normals_vis_output_path)
            pred_normals_vis_output_path = os.path.join(vis_output_path, 'normal_pred')
            if not os.path.exists(pred_normals_vis_output_path):
                os.makedirs(pred_normals_vis_output_path)
            phi_teta_vis_output_path = os.path.join(vis_output_path, 'phi_teta_domain')
            if not os.path.exists(phi_teta_vis_output_path):
                os.makedirs(phi_teta_vis_output_path)

        # load the data
        points = np.loadtxt(os.path.join(normal_gt_path, shape + '.xyz')).astype('float32')
        normals_gt = np.loadtxt(os.path.join(normal_gt_path, shape + '.normals')).astype('float32')
        normals_results = np.loadtxt(os.path.join(normal_results_path, shape + '.normals')).astype('float32')
        if sparse_patches:
            points_idx = np.loadtxt(os.path.join(normal_gt_path, shape + '.pidx')).astype('int')
        else:
            points_idx = list(range(np.shape(points)[0]))

        if os.path.exists(os.path.join(normal_results_path, shape + '.experts')):
            experts_exist = True
            experts = np.loadtxt(os.path.join(normal_results_path, shape + '.experts'))
            params = pickle.load(open(os.path.join(results_path, 'parameters.p'), "rb"))
            n_experts = params.n_experts
            if EXPORT:
                experts_vis_output_path = os.path.join(vis_output_path, 'experts_labels')
                if not os.path.exists(experts_vis_output_path):
                    os.makedirs(experts_vis_output_path)
        n_points = points.shape[0]
        n_normals = normals_results.shape[0]
        if n_points != n_normals:
            sparse_normals = True
        else:
            sparse_normals = False
        # points_idx = np.random.choice(range(0, 100000), 5000)

        points = points[points_idx, :]
        normals_gt = normals_gt[points_idx, :]
        if sparse_patches and not sparse_normals:
            normals_results = normals_results[points_idx, :]
        else:
            normals_results = normals_results[:, :]

        normal_gt_norm = l2_norm(normals_gt)
        normal_results_norm = l2_norm(normals_results)
        normals_results = np.divide(normals_results, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normals_gt = np.divide(normals_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        # Not oriented rms
        nn = np.sum(np.multiply(normals_gt, normals_results), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))  #  unoriented

        # error metrics
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))  # portion of good points
        pgp5_shape = sum([j < 5.0 for j in ang]) / float(len(ang))  # portion of good points
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        # Oriented rms
        rms_o.append(np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn))))))

        diff = np.arccos(nn)
        diff_inv = np.arccos(-nn)
        unoriented_normals = normals_results
        unoriented_normals[diff_inv < diff, :] = -normals_results[diff_inv < diff, :]

        # Visualization
        if EXPORT:
            #  For additional visualization options see MATALAB visualization code
            phi_gt, teta_gt = utils.euclidean_to_spherical(normals_gt)
            phi_pred_unoirented, teta_pred_unoirented = utils.euclidean_to_spherical(unoriented_normals)

            filename_pc_gt = os.path.join(gt_normals_vis_output_path, shape + '_pc_normals_gt')
            filename_pc_pred = os.path.join(pred_normals_vis_output_path, shape + '_pc_normals_pred')
            filename_phi_teta = os.path.join(phi_teta_vis_output_path, shape + '_phi_theta_domain')
            footnote = 'RMS unoriented= ' + str(rms[i]) + ', PGP5= ' + str(pgp5[i]) + ', PGP10= ' + str(pgp10[i])
            points = points - np.tile(np.mean(points, axis=0), [points.shape[0], 1])

            ax = visualization.draw_phi_teta_domain(phi_gt, teta_gt, color='k', display=False, export=True, format='png',
                                                    filename='phi_teta_gt', title=r'$\theta(\phi)$'+' '+shape)
            visualization.draw_line_segments(phi_gt, teta_gt, phi_pred_unoirented, teta_pred_unoirented, ax=ax, display=False,
                                             export=True, filename=filename_phi_teta, format='png', footnote=footnote)
            if experts_exist:
                cmap = visualization.discrete_cmap(n_experts, 'nipy_spectral')
                visualization.draw_phi_teta_domain(phi_pred_unoirented, teta_pred_unoirented, color=experts,
                                                   display=False, export=True, format='png',
                                                   filename=filename_phi_teta, ax=ax,
                                                   title=None, cmap=cmap, n_labels=n_experts)
            else:
                visualization.draw_phi_teta_domain(phi_pred_unoirented, teta_pred_unoirented, color='r', display=False,
                                                   export=True, format='png', filename='phi_teta_pred', ax=ax, title=None)

    avg_rms = np.mean(rms)
    RMS_not_oriented.append(avg_rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5 = np.mean(pgp5)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))

    LOG_FOUT.close()


LOG_FOUT = open(os.path.join(outdir, 'summary.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string('RMS not oriented:'+ str(RMS_not_oriented))
sum = 0
for i in range(len(RMS_not_oriented)):
    sum += RMS_not_oriented[i]
log_string('avg RMS not oriented:' + str(sum/len(RMS_not_oriented)))

