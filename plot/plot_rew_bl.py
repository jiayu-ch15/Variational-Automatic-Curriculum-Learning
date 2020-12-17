import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv

def main():
    scenario = 'box_locking'
    save_dir = './' + scenario + '/'
    save_name = 'main_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')

    # reverse
    exp_name = 'boxlocking_reverse'
    # data_dir =  './' + exp_name + '.csv'
    x_step1 = []
    y_seed1 = []
    x_step2 = []
    y_seed2 = []
    x_step3 = []
    y_seed3 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '_seed1' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step1.append(row[0])
            y_seed1.append(row[1:][0])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:][0])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:][0])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
    for j in range(len(y_seed1)):
        y_seed1[j] = np.float(y_seed1[j])
    for i in range(length):
        x_step2[i] = np.float(x_step2[i])
    for j in range(length):
        y_seed2[j] = np.float(y_seed2[j])
    
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for j in range(len(y_seed3)):
    #     y_seed3[j] = np.float(y_seed3[j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length]),axis=1)
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    plt.plot(timesteps,mean_seed,label='reverse goal generation')
    plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # mix
    exp_name = 'boxlocking'
    # data_dir =  './' + exp_name + '.csv'
    x_step1 = []
    y_seed1 = []
    x_step2 = []
    y_seed2 = []
    x_step3 = []
    y_seed3 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '_seed1' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step1.append(row[0])
            y_seed1.append(row[1:][0])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:][0])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:][0])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
    for j in range(len(y_seed1)):
        y_seed1[j] = np.float(y_seed1[j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
    for j in range(len(y_seed2)):
        y_seed2[j] = np.float(y_seed2[j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
    for j in range(len(y_seed3)):
        y_seed3[j] = np.float(y_seed3[j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[:-50]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)[:-50]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    plt.plot(timesteps,mean_seed,label='ours')
    plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    plt.title('main results')


    plt.xlabel('timesteps')
    plt.ylabel('success rate')
    plt.legend()
    plt.savefig(save_dir + save_name + '.jpg')

if __name__ == "__main__":
    main()