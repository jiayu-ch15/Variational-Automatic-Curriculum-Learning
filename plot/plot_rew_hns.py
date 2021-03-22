import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv

def main():
    scenario = 'hide_and_seek'
    save_dir = './' + scenario + '/'
    save_name = 'main_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')

    # ours
    exp_name = 'hns_phase'
    # data_dir =  './' + exp_name + '.csv'
    x_step1 = []
    y_seed1 = []
    x_step2 = []
    y_seed2 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '1' + '.json'
    with open(data_dir,'r') as csvfile:
        plots = json.load(csvfile)
        for env in plots:
            x_step1.append(env[1])
            y_seed1.append(env[2])
    data_dir =  './' + exp_name + '2' + '.json'
    with open(data_dir,'r') as csvfile:
        plots = json.load(csvfile)
        for env in plots:
            x_step2.append(env[1]+x_step1[-1])
            y_seed2.append(env[2])
    x_step = x_step1 + x_step2
    y_seed = y_seed1 + y_seed2
    plt.plot(x_step,y_seed,label='ours')

    # reverse
    exp_name = 'hns_reverse'
    # data_dir =  './' + exp_name + '.csv'
    x_step1 = []
    y_seed1 = []
    x_step2 = []
    y_seed2 = []
    # load data ranking by seed
    data_dir =  './' + exp_name + '.json'
    with open(data_dir,'r') as csvfile:
        plots = json.load(csvfile)
        for env in plots:
            x_step1.append(env[1])
            y_seed1.append(env[2])
    plt.plot(x_step1,y_seed1,label='reverse goal generation')

    # # add_easy
    # exp_name = 'hns_add_easy'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots:
    #         x_step1.append(env[1])
    #         y_seed1.append(env[2])
    # plt.plot(x_step1,y_seed1,label='uniform + easy case')
    plt.title('main_results')

    font = {
            'weight': 'normal',
            'size': 15,
            }
    plt.tick_params(labelsize=12)
    plt.xlabel('timesteps',font)
    plt.ylabel('success rate',font)
    plt.legend()
    plt.savefig(save_dir + save_name + '.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()