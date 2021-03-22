import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv
from scipy.interpolate import make_interp_spline

def main():
    scenario = 'box_locking'
    save_dir = './' + scenario + '/'
    save_name = 'floor6-lock'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')
    plt.figure(figsize=(8,6))

    # # agent_first
    # exp_name = 'agent_first'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step1.append(row[0])
    #         y_seed1.append(row[1:][0])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:][0])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:][0])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for j in range(len(y_seed1)):
    #     y_seed1[j] = np.float(y_seed1[j])
    # for i in range(length):
    #     x_step2[i] = np.float(x_step2[i])
    # for j in range(length):
    #     y_seed2[j] = np.float(y_seed2[j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for j in range(len(y_seed3)):
    #     y_seed3[j] = np.float(y_seed3[j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='brown')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),50)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # # box_first
    # exp_name = 'box_first'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step1.append(row[0])
    #         y_seed1.append(row[1:][0])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:][0])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:][0])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for j in range(len(y_seed1)):
    #     y_seed1[j] = np.float(y_seed1[j])
    # for i in range(length):
    #     x_step2[i] = np.float(x_step2[i])
    # for j in range(length):
    #     y_seed2[j] = np.float(y_seed2[j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for j in range(len(y_seed3)):
    #     y_seed3[j] = np.float(y_seed3[j])
    # # x_step = np.stack((x_step1[0:length],x_step2[0:length]),axis=1)
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='mediumpurple')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),50)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # floor6-lock
    exp_name = 'ours6-lock'
    # data_dir =  './' + exp_name + '.csv'
    x_step = []
    y_seed1 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step.append(row[0])
            y_seed1.append(row[1])
    x_step = x_step[1:610]
    y_seed1 = y_seed1[1:610]
    length = len(x_step)
    for i in range(len(x_step)):
        x_step[i] = np.float(x_step[i])
        y_seed1[i] = np.float(y_seed1[i])
    timesteps = np.array(x_step)
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),100)
    mean_smooth1 = make_interp_spline(timesteps,y_seed1)(xnew)
    plt.plot(xnew,mean_smooth1,color='mediumpurple')
    # end region

    exp_name = 'rcg6-lock'
    # data_dir =  './' + exp_name + '.csv'
    x_step = []
    y_seed1 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step.append(row[0])
            y_seed1.append(row[1])
    x_step = x_step[1:]
    y_seed1 = y_seed1[1:]
    length = len(x_step)
    for i in range(len(x_step)):
        x_step[i] = np.float(x_step[i])
        y_seed1[i] = np.float(y_seed1[i])
    timesteps = np.array(x_step)
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),100)
    mean_smooth1 = make_interp_spline(timesteps,y_seed1)(xnew)
    plt.plot(xnew,mean_smooth1,color='brown')
    # end region

    # # floor6-return
    # exp_name = 'ours6-return'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed1 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed1.append(row[1])
    # x_step = x_step[1:610]
    # y_seed1 = y_seed1[1:610]
    # length = len(x_step)
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     y_seed1[i] = np.float(y_seed1[i])
    # timesteps = np.array(x_step)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),100)
    # mean_smooth1 = make_interp_spline(timesteps,y_seed1)(xnew)
    # plt.plot(xnew,mean_smooth1,color='mediumpurple')
    # # end region

    # exp_name = 'rcg6-return'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed1 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed1.append(row[1])
    # x_step = x_step[1:]
    # y_seed1 = y_seed1[1:]
    # length = len(x_step)
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     y_seed1[i] = np.float(y_seed1[i])
    # timesteps = np.array(x_step)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),100)
    # mean_smooth1 = make_interp_spline(timesteps,y_seed1)(xnew)
    # plt.plot(xnew,mean_smooth1,color='brown')
    # # end region
 

    font = {
            'weight': 'normal',
            'size': 24,
            }
    plt.tick_params(labelsize=24)
    plt.xlabel('timesteps'r'$(\times 10^{7})$',font)
    plt.ylabel('lock rate',font)
    plt.legend()
    plt.savefig(save_dir + save_name + '.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()