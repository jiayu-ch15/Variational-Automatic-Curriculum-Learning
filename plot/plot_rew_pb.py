import matplotlib.pyplot as plt
import matplotlib
import json
import pdb
import numpy as np
import os
import csv
import pdb
from scipy.interpolate import make_interp_spline

def main():
    scenario = 'push_ball'
    save_dir = './' + scenario + '/'
    save_name = 'main_results_pb'
    # pdb.set_trace()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use('ggplot')
    plt.figure(figsize=(8,6))

    # # region nips ablation
    # exp_name = 'pb_uniform_from_activeAndsolve'
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

    # exp_name = 'pb_wo_evaluation'
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

    # exp_name = 'pb_active_expansion'
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

    # # good results
    # exp_name = 'pb_eval3'
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
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # # region sgma min
    # exp_name = 'pb_min0.75'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[100:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[100:]
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

    # exp_name = 'pb_min0.25'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[100:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[100:]
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
    # exp_name = 'pb_eval5'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[56:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[56:]
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

    # exp_name = 'pb_eval1'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[285:]
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[285:]
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
    # exp_name = 'reverse_eval1_pb'
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

    # exp_name = 'reverse_eval3_pb'
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

    # # region reverse vs solved 
    # exp_name = 'reverse_eval1_pb'
    # data_dir =  './' + exp_name + '.csv'
    # x_step = []
    # y_seed = []
    # with open(data_dir,'r') as csvfile:
    #     plots = csv.reader(csvfile)
    #     for row in plots:
    #         x_step.append(row[0])
    #         y_seed.append(row[1:])
    # x_step = x_step[1:1350]
    # y_seed = y_seed[1:1350]
    # for i in range(len(x_step)):
    #     x_step[i] = np.float(x_step[i])
    #     for j in range(len(y_seed[i])):
    #         y_seed[i][j] = np.float(y_seed[i][j])
    # x_step = np.array(x_step)
    # y_seed = np.array(y_seed)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # # plt.plot(x_step,mean_seed,color='brown')
    # # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(x_step.min(),x_step.max(),20)
    # mean_smooth = make_interp_spline(x_step,mean_seed)(xnew)
    # std_smooth = make_interp_spline(x_step,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # exp_name = 'solved_eval1_pb'
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
    # # plt.fill_between(x_step,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(x_step.min(),x_step.max(),20)
    # mean_smooth = make_interp_spline(x_step,mean_seed)(xnew)
    # std_smooth = make_interp_spline(x_step,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region
    # end region

    # # region entity curriculum
    # # transfer
    # exp_name = 'transfer_pb'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1110]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1110]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='mediumpurple')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),40)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='brown')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # # end region

    # # decay
    # exp_name = 'decay_pb_iter300'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:700]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)[0:190]
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:700]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:190]
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='brown')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),40)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='mediumpurple')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # # end region

    # # phase_pb
    # exp_name = 'phase_pb'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1140]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1140]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='coral')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),40)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='steelblue')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='steelblue')
    # # end region

    # # mix37_new
    # exp_name = 'mix37_new'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1170]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1170]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='coral')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),40)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='gray')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='gray')
    # # end region

    # # mix28
    # exp_name = 'mix28'
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
    # x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:1170]
    # # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    # x_step = x_step-x_step[0] # 从0开始计数
    # y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:1170]
    # # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # mean_seed = np.mean(y_seed,axis=1)
    # std_seed = np.std(y_seed,axis=1)
    # timesteps = np.mean(x_step,axis=1)
    # # plt.plot(timesteps,mean_seed,color='coral')
    # # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)
    # # region smooth
    # xnew = np.linspace(timesteps.min(),timesteps.max(),40)
    # mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    # std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    # plt.plot(xnew,mean_smooth,color='gray')
    # plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='gray')
    # # end region
    # # end region

    # region main results
    # ours
    exp_name = 'mix_final_pb_train2eval4'
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
    x_step4 = x_step1[0:224]
    x_step5 = x_step2[0:339]
    x_step6 = x_step3[0:253]
    y_seed4 = y_seed1[0:224]
    y_seed5 = y_seed2[0:339]
    y_seed6 = y_seed3[0:253]
    # concat
    exp_name = 'mix_final_pb'
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
    # length = min((len(x_step1),len(x_step3)))
    for i in range(len(x_step1)):
        x_step1[i] = np.float(x_step1[i]) + x_step4[-1]
        for j in range(len(y_seed1[i])):
            y_seed1[i][j] = np.float(y_seed1[i][j])
    for i in range(len(x_step2)):
        x_step2[i] = np.float(x_step2[i]) + x_step5[-1]
        for j in range(len(y_seed2[i])):
            y_seed2[i][j] = np.float(y_seed2[i][j])
    for i in range(len(x_step3)):
        x_step3[i] = np.float(x_step3[i]) + x_step6[-1]
        for j in range(len(y_seed3[i])):
            y_seed3[i][j] = np.float(y_seed3[i][j])
    x_step1 = x_step4 + x_step1
    x_step2 = x_step5 + x_step2
    x_step3 = x_step6 + x_step3
    y_seed1 = y_seed4 + y_seed1
    y_seed2 = y_seed5 + y_seed2
    y_seed3 = y_seed6 + y_seed3
    length = min((len(x_step1),len(x_step2),len(x_step3)))
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='mediumpurple')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='mediumpurple')
    # end region

    # reverse 4agents
    exp_name = 'reverse_pb_4agents'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:6100]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:6100]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,color='brown')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='brown')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='brown')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='brown')
    # end region
    

    # gan 4agents
    exp_name = 'gan_pb'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:6200]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:6200]
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

    # MAPPO
    exp_name = 'pb_4agents'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)
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

   # PC MAPPO
    exp_name = 'pc_pb'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:6100]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:6100]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='PC-MAPPO')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='dimgray')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='dimgray')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='dimgray')
    # end region

    # amigo
    exp_name = 'amigo_pb'
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
    x_step = np.stack((x_step1[0:length],x_step2[0:length],x_step3[0:length]),axis=1)[0:6100]
    # x_step = np.stack((x_step1[0:length],x_step3[0:length]),axis=1)
    x_step = x_step-x_step[0] # 从0开始计数
    y_seed = np.stack((y_seed1[0:length],y_seed2[0:length],y_seed3[0:length]),axis=1).squeeze(2)[0:6100]
    # y_seed = np.stack((y_seed1[0:length],y_seed3[0:length]),axis=1).squeeze(2)
    mean_seed = np.mean(y_seed,axis=1)
    std_seed = np.std(y_seed,axis=1)
    timesteps = np.mean(x_step,axis=1)
    # plt.plot(timesteps,mean_seed,label='PC-MAPPO')
    # plt.fill_between(timesteps,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1,color='dimgray')
    # region smooth
    xnew = np.linspace(timesteps.min(),timesteps.max(),20)
    mean_smooth = make_interp_spline(timesteps,mean_seed)(xnew)
    std_smooth = make_interp_spline(timesteps,std_seed)(xnew)
    plt.plot(xnew,mean_smooth,color='green')
    plt.fill_between(xnew,mean_smooth-std_smooth,mean_smooth+std_smooth,alpha=0.1,color='green')
    # end region
    # end region

    font = {
            'weight': 'normal',
            'size': 24,
            }

    plt.xlabel('timesteps' + r'$(\times 10^{8})$',font)
    plt.ylabel('coverage rate',font)
    plt.tick_params(labelsize=24)
    # plt.legend()
    plt.savefig(save_dir + save_name + '.jpg',bbox_inches='tight')

if __name__ == "__main__":
    main()