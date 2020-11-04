import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv

def main():
    scenario = 'simple_spread'
    save_dir = './' + scenario + '/'
    save_name = 'mix&phase'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # mix
    exp_name = 'mix4to8_sp'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    plt.plot(timesteps,mean_seed,label=exp_name)
    plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # warmup6iter
    exp_name = 'warmup6iter_sp'
    data_dir =  './' + exp_name + '.csv'
    x_step = []
    y_seed = []
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step.append(row[0])
            y_seed.append(row[1:])
    x_step = x_step[1:]
    y_seed = y_seed[1:]
    # 剔除坏点
    x_step = x_step[0:371]
    y_seed = y_seed[0:371]
    for i in range(len(x_step)):
        x_step[i] = np.float(x_step[i])
        for j in range(len(y_seed[i])):
            y_seed[i][j] = np.float(y_seed[i][j])
    x_step = np.array(x_step)
    # np.array(y_seed)[:,2]
    y_seed = np.stack((np.array(y_seed)[:,0],np.array(y_seed)[:,2]),axis=1)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    plt.plot(x_step,mean_seed,label=exp_name)
    plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # withoutwarmup
    exp_name = 'withoutwarmup_sp'
    data_dir =  './' + exp_name + '.csv'
    x_step = []
    y_seed = []
    with open(data_dir,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x_step.append(row[0])
            y_seed.append(row[1:])
    x_step = x_step[1:]
    y_seed = y_seed[1:]
    for i in range(len(x_step)):
        x_step[i] = np.float(x_step[i])
        for j in range(len(y_seed[i])):
            y_seed[i][j] = np.float(y_seed[i][j])
    x_step = np.array(x_step)
    y_seed = np.stack((np.array(y_seed)[:,0],np.array(y_seed)[:,2]),axis=1)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    plt.plot(x_step,mean_seed,label=exp_name)
    plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    plt.xlabel('timesteps')
    plt.ylabel('coverage rate')
    plt.legend()
    plt.savefig(save_dir + save_name + '.jpg')

    # scenario = 'simple_spread'
    # save_dir = './' + scenario + '/'
    # save_name = 'sp_tricks'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # solved_sp
    # exp_name = 'solved_sp'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_sp
    # exp_name = 'diversified_sp'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_novelty_sp
    # exp_name = 'diversified_novelty_sp'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_novelty_parentsampling_sp
    # exp_name = 'diversified_novelty_parentsampling_sp'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # plt.xlabel('timesteps')
    # plt.ylabel('coverage rate')
    # plt.legend()
    # plt.savefig(save_dir + save_name + '.jpg')


    # scenario = 'push_ball'
    # save_dir = './' + scenario + '/'
    # save_name = 'pb_tricks'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # solved_sp
    # exp_name = 'solved_pb'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_sp
    # exp_name = 'diversified_pb'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_novelty_sp
    # exp_name = 'diversified_novelty_pb'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # # diversified_novelty_parentsampling_sp
    # exp_name = 'diversified_novelty_parentsampling_pb'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:]
    # y_seed = y_seed[1:]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # plt.plot(x_step,mean_seed,label=exp_name)
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # plt.xlabel('timesteps')
    # plt.ylabel('coverage rate')
    # plt.legend()
    # plt.savefig(save_dir + save_name + '.jpg')

if __name__ == "__main__":
    main()