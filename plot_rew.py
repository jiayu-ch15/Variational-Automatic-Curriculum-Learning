import matplotlib.pyplot as plt
import json
import pdb
import numpy as np
import os
def main():
    # with open('test.csv','r') as csvFile:
    # reader = csv.reader(csvFile)
    # for line in reader:
    #     print line
    load_dict = []
    scenario = 'simple_spread'

    # step=0.01*boundary
    dir_name = 'step001'
    save_dir = './plot/' + scenario + '/' + dir_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for run_id in range(3):
        model_dir = 'results/MPE/' + scenario + '/' + dir_name + '/run%i/logs'%(run_id+1)
        with open("./" + model_dir + "/summary.json",'r') as load_f:
            load_dict.append(json.load(load_f))
    x_step = []
    y_seed = []
    for run_id in range(3):
        key = 'results/MPE/'+scenario+'/'+dir_name+'/run%i/logs/agent/cover_rate_5step/cover_rate_5step'%(run_id+1)
        x_step.append(np.array(load_dict[run_id][key])[:,1])
        y_seed.append(np.array(load_dict[run_id][key])[:,2])
    x_step = np.array(x_step)
    y_seed = np.array(y_seed)
    timestep = np.mean(x_step,axis=0)
    mean_seed = np.mean(y_seed,axis=0)
    std_seed = np.var(y_seed,axis=0)
    plt.plot(timestep,mean_seed,label='step=0.01')
    plt.fill_between(timestep,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    # step=0.25*boundary
    dir_name = 'step025'
    save_dir = './plot/' + scenario + '/' + dir_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for run_id in range(3):
        model_dir = 'results/MPE/' + scenario + '/' + dir_name + '/run%i/logs'%(run_id+1)
        with open("./" + model_dir + "/summary.json",'r') as load_f:
            load_dict.append(json.load(load_f))
    x_step = []
    y_seed = []
    for run_id in range(3):
        key = 'results/MPE/'+scenario+'/'+dir_name+'/run%i/logs/agent/cover_rate_5step/cover_rate_5step'%(run_id+1)
        x_step.append(np.array(load_dict[run_id][key])[:,1])
        y_seed.append(np.array(load_dict[run_id][key])[:,2])
    x_step = np.array(x_step)
    y_seed = np.array(y_seed)
    timestep = np.mean(x_step,axis=0)
    mean_seed = np.mean(y_seed,axis=0)
    std_seed = np.var(y_seed,axis=0)
    plt.plot(timestep,mean_seed,label='step=0.01')
    plt.fill_between(timestep,mean_seed-std_seed,mean_seed+std_seed,alpha=0.1)

    plt.xlabel('timesteps')
    plt.ylabel('coverage rate')
    plt.legend()
    plt.savefig(save_dir+dir_name+'.jpg')

if __name__ == "__main__":
    main()