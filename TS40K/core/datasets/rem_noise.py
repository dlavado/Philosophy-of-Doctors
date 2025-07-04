import torch
import numpy as np

import sys

from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.pointcloud_processing as eda

def noise_removing_score(original_class_freqs, dbscan_class_freqs, class_weights=None):
    """
    Compute the scoring of a noise removal algorithm based on the class frequencies before and after the noise removal;

    Parameters
    ----------
    `original_class_freqs` - torch.Tensor:
        tensor with the class frequencies of the original sample

    `dbscan_class_freqs` - torch.Tensor:
        tensor with the class frequencies

    `class_weights` - list:
        list with the weights of the classes, the first element should be the weight of the noise class

    Returns
    -------
    `score` - float:
        score of the noise removal algorithm, ranges from -1 to 1, with 1 being the best score
    """
    noise_removed = original_class_freqs[0] - dbscan_class_freqs[0] # Noise points removed
    non_noise_removed = torch.sum(original_class_freqs[1:]) - torch.sum(dbscan_class_freqs[1:]) # Non-noise points removed

    # Normalize by the total number of points
    noise_rem_ratio = noise_removed / original_class_freqs[0] # Fraction of noise removed
    non_noise_rem_ratio = non_noise_removed / torch.sum(original_class_freqs[1:]) # Fraction of non-noise points removed

    # Calculate a weighted score
    score = noise_rem_ratio - non_noise_rem_ratio  # Maximize noise removal, minimize non-noise removal
    if class_weights is not None:
        return score * class_weights[0] + (1 - class_weights[0]) * (1 - non_noise_rem_ratio)
    return score       


def greedy_search_rem_noise(dataset, num_samples=100):
    """
    Perform a greedy search to find the best parameters for DBSCAN noise removal algorithm

    Parameters
    ----------

    `dataset` - torch.utils.data.Dataset:
        dataset with the pointcloud samples

    `num_samples` - int:
        number of samples to use for the greedy search

    Returns
    -------

    """
    # `num_samples` - int: number of samples to use for the greedy search

    num_samples = min(num_samples, len(dataset))

    # do a greedy exploration of DBSCAN parameters that leads to the best removal of noise while keeping most of the other points

    # these values should group the points from intended classes and leave the noise points out
    eps_values = torch.linspace(0.1, 0.35, 10)
    min_point_values = torch.linspace(50, 500, 10)
    
    noise_removal_perf = {}
    scores_by_params = {}
    
    for i in tqdm(range(0, num_samples), desc="Computing Noise Removal..."):
        x, y  = dataset[i]
        y = y.squeeze().long()
        gt_class_freqs = torch.bincount(y, minlength=6)

        noise_removal_perf[i] = {}

        np_sample = np.concatenate((x.numpy(), y.numpy()[:, None]), axis=1)
        for eps in eps_values:
            for min_points in min_point_values:
 
                clean_sample = eda.remove_noise(np_sample, eps=eps, min_points=min_points)
                noise_in_og_sample = np.sum(np_sample[:, -1] == 0)
                noise_in_clean_sample = np.sum(clean_sample[:, -1] == 0)
                points_removed = np_sample.shape[0] - clean_sample.shape[0]
                noise_points_removed = noise_in_og_sample - noise_in_clean_sample
                print(f"Original sample: {np_sample.shape[0]}, Clean sample: {clean_sample.shape[0]};")
                print(f"Noise Points in Original Sample: {noise_in_og_sample}, Noise Points in Clean Sample: {noise_in_clean_sample}")
                print(f"Num Points removed: {points_removed};")
    
                print(f"Num Noise Points removed: {noise_points_removed};")
                
                print(f"Density of Noise removed from the OG sample = {1 - noise_in_clean_sample / noise_in_og_sample}")
                print(f"Density of Non-Noise removed from the OG sample = {(points_removed - noise_points_removed) / (np_sample.shape[0] - noise_in_og_sample)}")
                print(f"Density of Noise in Clean Sample = {noise_in_clean_sample / clean_sample.shape[0]}")
                print(f"Non-Noise Points Removed: {points_removed - (noise_in_og_sample - noise_in_clean_sample)}")

                clean_y = torch.from_numpy(clean_sample[:, -1])
                clean_y = clean_y.squeeze().long()
                dbscan_class_freqs = torch.bincount(clean_y, minlength=6)

                # compute the score
                score_value = noise_removing_score(gt_class_freqs, dbscan_class_freqs)
                noise_rem_density = 1 - noise_in_clean_sample / noise_in_og_sample
                
                noise_removal_perf[i][(eps, min_points)] = [score_value, noise_rem_density]

                if (eps, min_points) not in scores_by_params:
                    scores_by_params[(eps, min_points)] = ([], [])
                
                scores_by_params[(eps, min_points)][0].append(score_value)
                scores_by_params[(eps, min_points)][1].append(noise_rem_density)

                print(f"Sample {i} with eps: {eps}, min_points: {min_points} has score: {score_value:.5f} and noise_rem_density of {noise_rem_density:.5f} \n\n")

                # if score_value > 0.5:
                #     eda.plot_pointcloud(clean_sample[:, :-1], clean_sample[:, -1], use_preset_colors=True)
        
        # best score for the sample
        # best_params = max(noise_removal_perf[i], key=noise_removal_perf[i].get)
        # print(f"Best parameters for sample {i}: {best_params} with score: {noise_removal_perf[i][best_params]}")
        best_params = max(scores_by_params, key=lambda x: np.mean(scores_by_params[x][0]))
        print(f"Best parameters for sample {i}: {best_params} with score: {np.mean(scores_by_params[best_params][0])}")
        best_params = max(scores_by_params, key=lambda x: np.mean(scores_by_params[x][1]))
        print(f"Best parameters for sample {i}: {best_params} with noise removal density: {np.mean(scores_by_params[best_params][1])}")
        print("\n\n\n")

    for key in scores_by_params:
        print(f"Parameters: {key} with mean score: {np.mean(scores_by_params[key][0])}")
        print(f"Parameters: {key} with mean noise removal density: {np.mean(scores_by_params[key][1])}")

    # Calculate the mean score for each parameter combination
    mean_scores_by_params = {params: np.mean(tup[0]) for params, tup in scores_by_params.items()}
    mean_remdens_by_params = {params: np.mean(tup[1]) for params, tup in scores_by_params.items()}

    # Get the best parameters based on mean score
    best_params = max(mean_scores_by_params, key=mean_scores_by_params.get)
    best_mean_score = mean_scores_by_params[best_params]
    print(f"Best parameters across samples: {best_params} with mean score: {best_mean_score:.5f}")
    print(f"Best parameters across samples: {best_params} with mean noise removal density: {mean_remdens_by_params[best_params]:.5f}")

    return noise_removal_perf


def remove_noise_binary_search(dataset):
    # Define the initial search ranges for eps and min_points
    eps_range = [0.1, 0.5]
    min_points_range = [120, 300]
    
    # Set a max iteration limit for binary search
    max_iter = 20

    # Binary search function for eps and min_points
    def binary_search_dbscan(x, y, max_iter, eps_range, min_points_range):
        eps_low, eps_high = eps_range
        min_pts_low, min_pts_high = min_points_range
        
        best_eps = None
        best_min_pts = None
        best_score = -float('inf')
        
        for _ in range(max_iter):
            eps_mid = (eps_low + eps_high) / 2
            min_pts_mid = (min_pts_low + min_pts_high) // 2
            
            # Apply DBSCAN with mid values
            np_sample = np.concatenate((x.numpy(), y.numpy()[:, None]), axis=1)
            clean_sample = eda.remove_noise(np_sample, eps=eps_mid, min_points=min_pts_mid)
            
            clean_y = torch.from_numpy(clean_sample[:, -1])
            clean_y = clean_y.squeeze().long()
            dbscan_class_freqs = torch.bincount(clean_y, minlength=6)

            # add me some print statements to see the density of noise removed and the density of non-noise removed
            noise_in_og_sample = np.sum(np_sample[:, -1] == 0)
            noise_in_clean_sample = np.sum(clean_sample[:, -1] == 0)
            points_removed = np_sample.shape[0] - clean_sample.shape[0]
            noise_points_removed = noise_in_og_sample - noise_in_clean_sample
            print(f"#OG noise: {noise_in_og_sample}, #Clean noise: {noise_in_clean_sample}")
            if points_removed == 0:
                print("No points removed")
            else:
                print(f"Removed {points_removed} (noise: {noise_points_removed / points_removed});")
            print(f"Rem Density ---> Noise: {1 - noise_in_clean_sample / noise_in_og_sample:.5f}; Non-Noise: {(points_removed - noise_points_removed) / (np_sample.shape[0] - noise_in_og_sample):.5f}")

            # Compute ground truth frequencies
            gt_class_freqs = torch.bincount(y, minlength=6)

            # Compute score
            score_value = noise_removing_score(gt_class_freqs, dbscan_class_freqs)
            print(f"Score for eps={eps_mid:.3f}, min_points={min_pts_mid}: {score_value:.5f}")
            
            # Update the best parameters based on score
            if score_value > best_score:
                best_score = score_value
                best_eps, best_min_pts = eps_mid, min_pts_mid
            
            # Narrow the search space based on the score
            if score_value > 0:  # Adjust condition based on trend
                eps_low, min_pts_low = eps_mid, min_pts_mid
            else:
                eps_high, min_pts_high = eps_mid, min_pts_mid
        
        return best_eps, best_min_pts, best_score
    
  
 
    noise_removal_perf = {}
    
    for i in tqdm(range(0, len(dataset)), desc="Computing Noise Removal..."):
        x, y = dataset[i]
        y = y.squeeze().long()
        
        # Perform binary search for the best eps and min_points
        print(f"{'='*6} Sample {i} {'='*6}")
        best_eps, best_min_pts, best_score = binary_search_dbscan(x, y, max_iter, eps_range, min_points_range)
        
        noise_removal_perf[i] = {
            'best_eps': best_eps,
            'best_min_pts': best_min_pts,
            'best_score': best_score
        }
        
        print(f"Sample {i} best eps: {best_eps}, best min_points: {best_min_pts}, score: {best_score:.5f}")
        print("\n\n")
    
    # Return the best parameters for each sample
    return noise_removal_perf


def get_best_params_across_samples(noise_removal_perf):
    scores_by_params = {}
    
    # Aggregate scores across samples for each parameter pair
    for sample_id in noise_removal_perf:
        for params, result in noise_removal_perf[sample_id].items():
            if params not in scores_by_params:
                scores_by_params[params] = []
            scores_by_params[params].append(result['best_score'])
    
    # Calculate the mean score for each parameter combination
    mean_scores_by_params = {params: np.mean(scores) for params, scores in scores_by_params.items()}
    
    # Get the best parameters based on mean score
    best_params = max(mean_scores_by_params, key=mean_scores_by_params.get)
    best_mean_score = mean_scores_by_params[best_params]
    
    return best_params, best_mean_score



if __name__ == "__main__":
    from core.datasets.TS40K import TS40K_FULL, TS40K_FULL_Preprocessed
    from utils import constants

    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH,
        split='fit',
        sample_types='all',
    )
    

    # ts40k = TS40K_FULL(constants.TS40K_FULL_PATH,
    #                     split='fit',
    #                     sample_types='all',
    #                     task='sem_seg',
    #                     transform=None,
    #                     load_into_memory=False
    #                 )

    greedy_search_rem_noise(ts40k)

    # noise_removal_perf = remove_noise_binary_search(ts40k)
    # best_params, best_mean_score = get_best_params_across_samples(noise_removal_perf)

    # print(f"Best parameters across samples: {best_params} with mean score: {best_mean_score:.5f}")


