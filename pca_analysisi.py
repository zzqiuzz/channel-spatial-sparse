import torch
import torch.nn as nn
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import torch
import os
parser = argparse.ArgumentParser(description='PCA analysis')
parser.add_argument('-b','--batchsize',default=256,type=int, metavar='batchsize',
                    help='batchsize for pca.')
parser.add_argument('-o','--output',default='./pca_ans',help='path to save pca analysis.')
parser.add_argument('-f','--feature',type=str,help='path to features..')
args = parser.parse_args()
def select(data):
    if len(data.shape) == 3: # batchsize == 1
        
    elif len(data.shape) == 4: #batchsize >1

    else:
        print("Not Implemented Yet.")
def pca(newdata):
    newdata = np.array(newdata)
    newdata = newdata.reshape(newdata.shape[0],-1)
    print("shape after reshape is :{}".format(newdata.shape))
    newdata = np.mat(newdata)
    newdata_mean = newdata - newdata.mean(axis=1)
    newdata_cov = np.cov(newdata_mean)
    eigenValue,eigenVector = np.linalg.eig(newdata_cov)
    print("shape:{},eigenValue is: {}".format(eigenValue.shape, eigenValue))
    print("shape:{},eigenVectors is :{}".format(eigenVector.shape, eigenVector))
    x_axis = np.arange(eigenValue.size)
    save_path = str(args.output) + '/batch' + str(args.batchsize) + '_' + str(layer_name) + '_pca.jpg' 
    plt.xlabel('Index')
    plt.ylabel('eigenValue')
    title = str(layer_name) + '_batch_' + str(args.batchsize)
    plt.title(title)
    plt.plot(x_axis,eigenValue)
    plt.savefig(save_path)
    plt.show()

def main():
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    filename = str(args.feature)
    npz = np.load(filename)
    print(npz.files)
    layer_name = npz[npz.files[0]] 
    layer_in_data = npz[npz.files[1]]
    layer_in_data = np.squeeze(layer_in_data)
    feature = npz[npz.files[2]]
    feature = np.squeeze(feature)
    in_shape = npz[npz.files[3]].reshape(-1)
    feature_shape = npz[npz.files[4]].reshape(-1)
    weight = npz[npz.files[5]]
    print("weight shape is: {}".format(weight.shape))

    ########
    #bn = np.load("layer1_1_bn2.npz")
    #print(bn.files)
    #feature_bn = npz[npz.files[2]]
    #feature_bn = np.squeeze(feature_bn)
    m = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
    m.weight = nn.Parameter(torch.Tensor(weight))
    output = m(torch.Tensor(layer_in_data))
    _feature = torch.Tensor(feature)
    ans = _feature.sub(output)

    #########feature pca starts
    print("feature shape is:{}".format(feature_shape))
    print("in_data_shape is:{}".format(in_shape))
    N = feature_shape[0]
    C = feature_shape[1]
    H = feature_shape[2]
    W = feature_shape[3]
    feature = feature.reshape(N,C,H,W)
    if args.batchsize == 1:
        newdata = feature[0,:,:,:] 
    else:
        newdata = []
        for i in range(0,C):
            newdata.append(feature[:,i,:,:])
    print("Shape of data for PCA:{} ".format(newdata.shape))
    pca(newdata)
    print("pca finished.")
    #select some important filters for pca again
    select(newdata)
if __name__ == '__main__':
    main()
