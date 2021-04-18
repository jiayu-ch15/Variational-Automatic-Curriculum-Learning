import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
import csv
from scipy.interpolate import make_interp_spline


def main():
    scenario = 'simple_spread'
    save_dir = './' + scenario + '/'
    save_name = 'transfer_mix_decay'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')
    plt.figure(figsize=(8,6))

    # # region sigma min
    # exp_name = 'sp_min0.75'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[190:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[190:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'sp_min0.25'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[190:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[190:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region
    # # end region

    # # region eval num
    # exp_name = 'sp_eval5'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[112:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[112:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'sp_eval1'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[570:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[570:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region
    # # end region

    # # region reverse eval num
    # exp_name = 'reverse_eval1_sp'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed1 = []
    # y_seed2 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed1.append(row[1])
    #         y_seed2.append(row[2])
    #         y_seed3.append(row[3])
    # x_step = x_step[1:]
    # y_seed1 = y_seed1[1:]
    # y_seed2 = y_seed2[1:]
    # y_seed3 = y_seed3[1:]
    # length = len(x_step)
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     y_seed1[i] = np.float(y_seed1[i])
    #     y_seed2[i] = np.float(y_seed2[i])
    #     y_seed3[i] = np.float(y_seed3[i])
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.array(x_step)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'reverse_eval3_sp'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed1 = []
    # y_seed2 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed1.append(row[1])
    #         y_seed2.append(row[2])
    #         y_seed3.append(row[3])
    # x_step = x_step[1:]
    # y_seed1 = y_seed1[1:]
    # y_seed2 = y_seed2[1:]
    # y_seed3 = y_seed3[1:]
    # length = len(x_step)
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     y_seed1[i] = np.float(y_seed1[i])
    #     y_seed2[i] = np.float(y_seed2[i])
    #     y_seed3[i] = np.float(y_seed3[i])
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.array(x_step)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region
    # # end region

    # region fraction
    # exp_name = '19'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_50_400_50/run1/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step1.append(env[1])
    #         y_seed1.append(env[2])
    # data_dir =  './' + exp_name + '_seed2' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_50_400_50/run2/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step2.append(env[1])
    #         y_seed2.append(env[2])
    # data_dir =  './' + exp_name + '_seed3' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_50_400_50/run3/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step3.append(env[1])
    #         y_seed3.append(env[2])
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for i in range(len(y_seed1)):
    #     y_seed1[i] = np.float(y_seed1[i])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    # for i in range(len(y_seed2)):
    #     y_seed2[i] = np.float(y_seed2[i])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for i in range(len(y_seed3)):
    #     y_seed3[i] = np.float(y_seed3[i])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[98:405]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)[98:405]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='brown')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = '12'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[98:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[98:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = '91'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_400_50_50/run1/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step1.append(env[1])
    #         y_seed1.append(env[2])
    # data_dir =  './' + exp_name + '_seed2' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_400_50_50/run2/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step2.append(env[1])
    #         y_seed2.append(env[2])
    # data_dir =  './' + exp_name + '_seed3' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/fraction_400_50_50/run3/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step3.append(env[1])
    #         y_seed3.append(env[2])
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for i in range(len(y_seed1)):
    #     y_seed1[i] = np.float(y_seed1[i])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    # for i in range(len(y_seed2)):
    #     y_seed2[i] = np.float(y_seed2[i])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for i in range(len(y_seed3)):
    #     y_seed3[i] = np.float(y_seed3[i])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[98:390]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)[98:390]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region
    # # end region

    # # region step size
    # exp_name = 'step001'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step001/run1/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step1.append(env[1])
    #         y_seed1.append(env[2])
    # data_dir =  './' + exp_name + '_seed2' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step001/run2/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step2.append(env[1])
    #         y_seed2.append(env[2])
    # data_dir =  './' + exp_name + '_seed3' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step001/run3/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step3.append(env[1])
    #         y_seed3.append(env[2])
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for i in range(len(y_seed1)):
    #     y_seed1[i] = np.float(y_seed1[i])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    # for i in range(len(y_seed2)):
    #     y_seed2[i] = np.float(y_seed2[i])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for i in range(len(y_seed3)):
    #     y_seed3[i] = np.float(y_seed3[i])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:385]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)[0:385]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='brown')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'step025'
    # # data_dir =  './' + exp_name + '.csv'
    # x_step1 = []
    # y_seed1 = []
    # x_step2 = []
    # y_seed2 = []
    # x_step3 = []
    # y_seed3 = []
    # # load data ranking by seed
    # data_dir =  './' + exp_name + '_seed1' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step025/run1/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step1.append(env[1])
    #         y_seed1.append(env[2])
    # data_dir =  './' + exp_name + '_seed2' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step025/run2/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step2.append(env[1])
    #         y_seed2.append(env[2])
    # data_dir =  './' + exp_name + '_seed3' + '.json'
    # with open(data_dir,'r') as csvfile:
    #     plots = json.load(csvfile)
    #     for env in plots['results/MPE/simple_spread/step025/run3/logs/agent/cover_rate_5step/cover_rate_5step']:
    #         x_step3.append(env[1])
    #         y_seed3.append(env[2])
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    # for i in range(len(y_seed1)):
    #     y_seed1[i] = np.float(y_seed1[i])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    # for i in range(len(y_seed2)):
    #     y_seed2[i] = np.float(y_seed2[i])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    # for i in range(len(y_seed3)):
    #     y_seed3[i] = np.float(y_seed3[i])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:385]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1)[0:385]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region
    # # end region

    # # region entity curriculum
    # exp_name = 'mix37_final_sp'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:136]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:136]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='mediumpurple')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='mediumpurple')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),50)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # exp_name = 'mixdecay_1to0_fre3'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:136]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:136]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),50)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='coral')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='coral')
    # # end region

    # # decay from 0.5
    # exp_name = 'mixdecay_sp_fre6'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:139]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:139]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),30)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'phase_sp_true'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:144]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:144]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='coral')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='coral')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),50)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region
    # end region

    # # region tech1
    # exp_name = 'reverse_eval1_sp'
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
    # # plt.plot(x_step,mean_seed,color='brown')
    # # region smooth
    # xnew = np.linspace(x_step.min(),x_step.max(),30)
    # smooth = make_interp_spline(x_step,mean_seed)(xnew)
    # plt.plot(xnew,smooth,color='brown')
    # # end region
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # exp_name = 'solved_eval1_sp'
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
    # # plt.plot(x_step,mean_seed,color='steelblue')
    # # region smooth
    # xnew = np.linspace(x_step.min(),x_step.max(),30)
    # smooth = make_interp_spline(x_step,mean_seed)(xnew)
    # plt.plot(xnew,smooth,color='mediumpurple')
    # # end region
    # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='mediumpurple')
    # # end region

    # # region main results
    # # ours
    # exp_name = 'mix_sp_train4eval8'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step4 = x_step1[0:202]
    # x_step5 = x_step2[0:241]
    # x_step6 = x_step3[0:207]
    # y_seed4 = y_seed1[0:202]
    # y_seed5 = y_seed2[0:241]
    # y_seed6 = y_seed3[0:207]
    # # concat
    # exp_name = 'mix37_final_sp'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i]) + x_step4[-1]
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i]) + x_step5[-1]
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i]) + x_step6[-1]
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step1 = x_step4 + x_step1
    # x_step2 = x_step5 + x_step2
    # x_step3 = x_step6 + x_step3
    # y_seed1 = y_seed4 + y_seed1
    # y_seed2 = y_seed5 + y_seed2
    # y_seed3 = y_seed6 + y_seed3
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:570]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:570]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,label='ACM')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='mediumpurple')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # # reverse 4agents
    # exp_name = 'reverse_sp_8agents'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1714]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1714]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,label='RCG')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='brown')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # # gan 4agents
    # exp_name = 'gan_sp'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:857]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:857]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,label='AGG')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region

    # # MAPPO
    # exp_name = 'sp_8agents'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1714]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1714]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,label='MAPPO')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='coral')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='coral')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='coral')
    # # end region

    # # PC MAPPO
    # exp_name = 'pc_sp'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # # length = min((len(x_step1),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1714]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1714]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,label='PC-MAPPO')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='dimgray')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='dimgray')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='dimgray')
    # # end region
    # # end region

    # # good results
    # exp_name = 'sp_eval3'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[190:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[190:]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # # region rebuttal
    # exp_name = 'sp_eval3'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:270]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:270]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # exp_name = 'amigo_cover_rate'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region
    # # end region


    # # region transfer
    # exp_name = 'transfer_init_optimizer'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown',label='transfer_init_optimizer')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')

    # exp_name = 'transfer_load_optimizer'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='dimgray',label='transfer_load_optimizer')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='dimgray')
    
    exp_name = 'transfer_warmup30iter'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
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
    plt.plot(xnew,mean_smooth,color='mediumpurple',label='transfer_warmup30iter')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # end region

    # exp_name = 'transfer_warmup150'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue',label='transfer_warmup150iter')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region

    # # region mix
    # exp_name = 'mix_ratio55_0.9off_initial_optimizer'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown',label='mix_init_optimizer')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')

    exp_name = 'mix_ratio55_0.9off_load_optimizer'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
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
    plt.plot(xnew,mean_smooth,color='dimgray',label='mix_load_optimizer')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='dimgray')
    
    # region decay
    # exp_name = 'decay_fre30_initial_optimizer'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='dimgray',label='decay_fre30_init_optimizer')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='dimgray')

    exp_name = 'decay_fre30_load_optimizer'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
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
    plt.plot(xnew,mean_smooth,color='brown',label='decay_fre30_load_optimizer')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')

    # exp_name = 'decay_fre150_load_optimizer'
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
    #         y_seed1.append(row[1:])
    # data_dir =  './' + exp_name + '_seed2' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step2.append(row[0])
    #         y_seed2.append(row[1:])
    # data_dir =  './' + exp_name + '_seed3' + '.csv'
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step3.append(row[0])
    #         y_seed3.append(row[1:])
    # x_step1 = x_step1[1:]
    # y_seed1 = y_seed1[1:]
    # x_step2 = x_step2[1:]
    # y_seed2 = y_seed2[1:]
    # x_step3 = x_step3[1:]
    # y_seed3 = y_seed3[1:]
    # length = min((len(x_step1),len(x_step2),len(x_step3)))
    # for i in range(len(x_step1)):
    #     x_step1[i] = np.float(x_step1[i])
    #     for j in range(len(y_seed1[i])):
    #         y_seed1[i][j] = np.float(y_seed1[i][j])
    # for i in range(len(x_step2)):
    #     x_step2[i] = np.float(x_step2[i])
    #     for j in range(len(y_seed2[i])):
    #         y_seed2[i][j] = np.float(y_seed2[i][j])
    # for i in range(len(x_step3)):
    #     x_step3[i] = np.float(x_step3[i])
    #     for j in range(len(y_seed3[i])):
    #         y_seed3[i][j] = np.float(y_seed3[i][j])
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='steelblue')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='steelblue')
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue',label='decay_fre150_load_optimizer')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')

    font = {
            'weight': 'normal',
            'size': 24,
            }
    plt.tick_params(labelsize=24)
    plt.xlabel('timesteps' + r'$(\times 10^{7})$',font)
    plt.ylabel('coverage rate',font)
    # plt.ylabel('average reward',font)
    plt.legend()
    plt.savefig(save_dir + save_name + '.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()