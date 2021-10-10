import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv
from scipy.interpolate import make_interp_spline


def main():
    scenario = 'speaker_listener'
    save_dir = './' + scenario + '/'
    save_name = 'sl_2agents'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')
    plt.figure(figsize=(8,6))

    # region main results 4agents
    # ours
    exp_name = 'sl_VACL'
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
            y_seed1.append(row[1:])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    # length = min((len(x_step1),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1680]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1680]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='RCG')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='brown')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='mediumpurple')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # end region

    # reverse 4agents
    exp_name = 'sl_reverse'
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
            y_seed1.append(row[1:])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    # length = min((len(x_step1),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:5000]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:5000]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='RCG')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='brown')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='brown')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # end region

    # gan 4agents
    exp_name = 'sl_gan'
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
            y_seed1.append(row[1:])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    # length = min((len(x_step1),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:2500]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:2500]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='AGG')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='steelblue')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # end region

    # unif
    exp_name = 'sl_unif'
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
            y_seed1.append(row[1:])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    # length = min((len(x_step1),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:5000]
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:5000]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='MAPPO')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='coral')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='coral')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='coral')
    # end region

    exp_name = 'sl_amigo'
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
            y_seed1.append(row[1:])
    data_dir =  './' + exp_name + '_seed2' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step2.append(row[0])
            y_seed2.append(row[1:])
    data_dir =  './' + exp_name + '_seed3' + '.csv'
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step3.append(row[0])
            y_seed3.append(row[1:])
    x_step1 = x_step1[1:]
    y_seed1 = y_seed1[1:]
    x_step2 = x_step2[1:]
    y_seed2 = y_seed2[1:]
    x_step3 = x_step3[1:]
    y_seed3 = y_seed3[1:]
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i])
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i])
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i])
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:5000]
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:5000]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,color='steelblue')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='green')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='green')
    # end region

    font = {
            'weight': 'normal',
            'size': 24,
            }
    plt.tick_params(labelsize=24)
    plt.xlabel('timesteps' + r'$(\times 10^{7})$',font)
    plt.ylabel('coverage rate',font)
    # plt.ylabel('average reward',font)
    # plt.legend()
    plt.savefig(save_dir + save_name + '.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()