import matplotlib.pyplot as plt
import math
from matplotlib.ticker import StrMethodFormatter

def plot_metrics(result_df):
    plt.figure(figsize=(10,8)) #,layout='constrained'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'rmse', 'pearson','spearman'] ##
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(result_df.index, result_df[metric], color=colors[0], label='Train')
        plt.plot(result_df.index, result_df['dev_'+metric],
                    color=colors[3], linestyle="--", label='Dev')
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(name)



def plot_metrics2(result_df):
    plt.figure(figsize=(5,8)) #,layout='constrained'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'pear_corr'] ##
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,1,n+1)
        plt.plot(result_df.index, result_df['train_'+metric], color=colors[0], label='Train')
        plt.plot(result_df.index, result_df['dev_'+metric],
                    color=colors[3], linestyle="--", label='Dev')
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(name)



def plot_loss(result_df):
    # mpl.rcParams['figure.figsize'] = (8, 10)
    plt.figure(figsize=(5,5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metric = 'loss'
    plt.plot(range(1,len(result_df[metric])+1), result_df[metric], color=colors[0], label='Train')
    plt.plot(range(1,len(result_df[metric])+1), result_df['dev_'+metric],
                color=colors[3], linestyle="--", label='Dev')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.ylim(0, math.ceil(max(result_df[metric]+result_df['dev_'+metric]))) 
    plt.xlim(0, len(result_df[metric])+1) 
    plt.grid(True, linestyle='--')
    plt.tight_layout()
