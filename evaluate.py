import argparse

import toml

from dataset.dfcla_2 import LAVDFDataModule
from metrics import AP, AR
from model import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch.multiprocessing
import torch


parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--model", type=str, default=None)

def class_wise_accuracy(y_true, y_prob, threshold):
    """
    计算二分类中每个类别的准确率
    
    参数:
    y_true: 真实标签，形状为 (n_samples,)
    y_prob: 预测概率，形状为 (n_samples,)
    threshold: 分类阈值，默认0.5
    
    返回:
    dict: 包含每个类别的准确率
    """
    # 将概率转换为预测类别
    acc_0 = []
    acc_1 = []
    for th in threshold:
        y_pred = (y_prob >= th).astype(int)
        
        # 获取每个类别的索引
        class_0_indices = np.where(y_true == 0)[0]
        class_1_indices = np.where(y_true == 1)[0]
        
        # 计算每个类别的准确率
        acc_class_0 = accuracy_score(y_true[class_0_indices], y_pred[class_0_indices])
        acc_class_1 = accuracy_score(y_true[class_1_indices], y_pred[class_1_indices])
        
        acc_0.append(acc_class_0)
        acc_1.append(acc_class_1)        

    return {
        'class_0_accuracy': acc_0,
        'class_1_accuracy': acc_1,
    }


def calculate_accuracy(labels, scores, threshold):
    # 将预测分数转换为预测标签
    accs = []
    for th in threshold:
        predictions = (scores >= th).astype(int)
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        accs.append(accuracy)        

    return accs

def evaluate(model, data_loader):

    output_dict = forward(
        model=model, 
        data_loader=data_loader) 
    #print(output_dict.shape)

    statistics = {}

    # Clipwise statistics
    statistics['AUC'] = roc_auc_score(
        output_dict['target'], output_dict['clipwise_output'][:, 1])

    
    #statistics['AUC'] = roc_auc_score(
    #    output_dict['target'], output_dict['clipwise_output'])

    statistics['ACC'] = calculate_accuracy(
        output_dict['target'], output_dict['clipwise_output'][:, 1], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    statistics['F1'] = f1_score(
        output_dict['target'], [1 if p >= 0.5 else 0 for p in output_dict['clipwise_output'][:, 1]])

    statistics['ACC_all'] = class_wise_accuracy(
        output_dict['target'], output_dict['clipwise_output'][:, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    return statistics, output_dict

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        # raise Exception("Error!")
        return x

    return x.to(device)

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def forward(model, data_loader):

    device = next(model.parameters()).device #see model device
    output_dict = {}
    
    # Evaluate on mini-batch
    for n, wav_dic in enumerate(data_loader):
        print('batch_number', n)
        video, audio, n_frames, target = wav_dic

        #print("label : " , target)
        video = move_data_to_device(video, device)
        audio = move_data_to_device(audio, device)
        n_frames = move_data_to_device(n_frames, device)
        
        #print(video.shape)
        #print(audio.shape)
        video = rearrange(video, "b t c h w -> b c t h w")
        audio = rearrange(audio, "b t c -> b c t")
        
        with torch.no_grad():
            model.eval()
            output = model(video, audio, n_frames)
            if isinstance(output, tuple) :
                batch_output = output[0]
            else:
                batch_output = output
            #batch_output = batch_output.squeeze(-1).mean(1)
            #batch_output = batch_output.squeeze(-1)
            batch_output = torch.nn.functional.softmax(batch_output.squeeze(-1))
            print(batch_output, target)
            #print(batch_output.shape)
        #batch_output = torch.clamp(batch_output, 1, 5)
        append_to_dict(output_dict, 'clipwise_output', 
            batch_output.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target)
        
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    config = toml.load(args.config)
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    print(args.checkpoint)
    print(args.model)
    model = eval(args.model)

    # prepare dataset
    dm = LAVDFDataModule(root=args.data_root,
        frame_padding=config["num_frames"],
        max_duration=config["max_duration"],
        batch_size=8, num_workers=4,
        get_meta_attr=model.get_meta_attr)
    dm.setup()


    # prepare model

    #test_dataset = dm.test_dataset
    test_loader = dm.test_dataloader

    model = model.load_from_checkpoint(args.checkpoint)

    if args.gpus >= 1:
        model = model.cuda()

    (statistics, output_dict) = evaluate(model, test_loader())
    print(statistics['AUC'])
    print(statistics['ACC'])
    print(statistics['F1'])
    print(statistics['ACC_all'])

