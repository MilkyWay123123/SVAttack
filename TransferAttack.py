'''
Sample transfer test
'''
from utils.loadModel import *
from feeder.feeder import *
import torch
import argparse
from omegaconf import OmegaConf

def transfer_rate(model_name, save_path, device='cuda:0'):
    target_classifier = getModel(model_name)

    samples_x_list = np.load(f'{save_path}/samples_x_list.npy')
    attck_samples_x_list = np.load(f'{save_path}/attck_samples_x_list.npy', allow_pickle=False)
    attck_samples_y_list = np.load(f'{save_path}/attck_samples_y_list.npy')

    error_num = 0
    sample_num = len(attck_samples_x_list)

    for i in range(sample_num):
        tx = samples_x_list[i]
        adData = attck_samples_x_list[i]

        tx = tx.reshape(1, tx.shape[0], tx.shape[1], tx.shape[2], tx.shape[3])
        adData = adData.reshape(1, adData.shape[0], adData.shape[1], adData.shape[2], adData.shape[3])

        tx = torch.tensor(tx).float().cuda(device)
        adData = torch.tensor(adData).float().cuda(device)

        truth_label = torch.argmax(target_classifier(tx), axis=1)
        pred = torch.argmax(target_classifier(adData), axis=1)

        # Filtering abnormal data
        if truth_label != attck_samples_y_list[i]:
            sample_num -= 1
        else:
            if pred != attck_samples_y_list[i]:
                error_num += 1
    print(f'迁移率 {error_num / sample_num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mn", type=str, default="agcn")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = 'cuda:0'
    transfer_rate(args.mn, config.save_path, device)
