import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_curve(iteration, loss):

    # Plot the a cross validation curve using Matplotlib

    fig = plt.figure(figsize=(4,0))
    plt.rc('font', weight='bold')
    plt.plot(iteration, loss, color = 'r', clip_on = False, label = 'loss')
    plt.legend()
    plt.ylabel('Loss', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Iteration', fontsize = 16, fontweight = 'bold')
    plt.xlim(iteration[0], iteration[-1])
    plt.show()
    #fig.savefig('RNNGAN.png', format = 'png', dpi = 600, bbox_inches = 'tight')
    #plt.close()

def main():

    df = pd.read_csv('train_log.csv', header = None)
    iteration = df.iloc[:, 0].as_matrix()
    gen_loss = df.iloc[:, 1].as_matrix()
    dis_loss = df.iloc[:, 2].as_matrix()
    loss = gen_loss+dis_loss
    plot_curve(iteration = iteration, loss = loss)

if __name__ == '__main__':
    main()
