if True:
    import torch
    import numpy as np
    from matplotlib import pyplot as plt
    dset = 'c4'
    test_a = torch.load(f'quant_headacts_{dset}_V_16_opt-1.3b_.pt')
    for k_ in test_a[0].keys():
        if isinstance(k_, int):
            headacts_ = [tz[k_] for tz in test_a.values()]
            data = np.asarray(headacts_).squeeze()
            normalized_data = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))
            plt.imsave(f'headacts_{dset}_{k_}.png', normalized_data, cmap='hot')
    ldic = [x for x in test_a[0].keys() if isinstance(x, int)]
    for layer in ldic:
        layer_activations = [test_a[dp][layer] for dp in test_a if isinstance(test_a[dp], dict) and layer in test_a[dp]]
        data_matrix = np.stack(layer_activations).squeeze()
        cross_corr_matrix = np.corrcoef(data_matrix.T) 
        normalized_cross_corr = (cross_corr_matrix - np.min(cross_corr_matrix)) / (np.max(cross_corr_matrix) - np.min(cross_corr_matrix))
        plt.figure()
        img = plt.imshow(normalized_cross_corr, cmap='hot', aspect='auto')
        plt.colorbar(img)
        plt.savefig(f'dcrosscorr_{dset}_{layer}.png')
        plt.close()
