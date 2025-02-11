'''
阻尼器
'''
import torch

hookratio = 0.1


# 注册钩子
def register_hooks(classifier, ratio, model_name):
    hookratio = ratio
    if model_name == 'stgcn' or model_name == 'stgcn120':
        print('注册stgcn阻尼器')
        for name, layer in classifier.named_modules():  # stgcn
            if ('module.st_gcn_networks.4.gcn' in name or 'module.st_gcn_networks.5.gcn' in name
                    or 'module.st_gcn_networks.6.gcn' in name or 'module.st_gcn_networks.7.gcn' in name
                    or 'module.st_gcn_networks.8.gcn' in name or 'module.st_gcn_networks.9.gcn' in name):
                print(f'stgcn={name}')
                hook = layer.register_full_backward_hook(adjust_array)
    elif model_name == 'agcn' or model_name == 'agcn120':
        print('注册agcn阻尼器')
        for name, layer in classifier.named_modules():  # agcn
            if ('l4.gcn1' in name or 'l5.gcn1' in name or 'l6.gcn1' in name or 'l7.gcn1' == name or 'l8.gcn1' in name
                    or 'l9.gcn1' in name or 'l10.gcn1' in name):
                print(f'agcn={name}')
                hook = layer.register_full_backward_hook(adjust_array)
    elif model_name == 'ctrgcn' or model_name == 'ctrgcn120':
        print('注册ctrgcn阻尼器')
        for name, layer in classifier.named_modules():  # ctrgcn
            if (
                    'l4.gcn' in name or 'l5.gcn' in name or 'l6.gcn' in name or 'l7.gcn' in name or 'l8.gcn' in name or 'l9.gcn' in name or 'l10.gcn' in name):
                print(f'ctrgcn={name}')
                hook = layer.register_full_backward_hook(adjust_array)
    else:
        print('-----------------------------------------')
        print('未注册阻尼器')
        print('-----------------------------------------')


def adjust_array(module, grad_input, grad_output):
    result_tensor = []
    for i in range(len(grad_input)):
        data = grad_input[i]
        data_flat = data.reshape(-1)
        size = len(data_flat)
        topk_values, topk_indices = torch.topk(data_flat, int(size * 0.9))
        data_flat[topk_indices] *= hookratio
        result_tensor.append(torch.reshape(data_flat, data.shape))
    return result_tensor
