import numpy as np

def simple_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -4
    else:
        return 1


def my_reward_closure(name):
    if name == 'magni':
        c0 = 3.35506
        c1 = 0.8353
        c2 = 3.7932
        baseline = 400
    elif name == 'kovatchev':
        c0 = 1.509
        c1 = 1.084
        c2 = 5.381
        baseline = 200
    
    def my_reward(BG_last_hour):
        risk = c0 * ((np.log(BG_last_hour[-1]) ** c1) - c2)
        risk = risk ** 2
        risk = 10 * risk
        # add baseline to keep reward positive
        return -risk + baseline
    
    return my_reward


# run this file to plot magni and kovatchev reward curves
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    bgs = np.arange(10, 601)
    my_reward = my_reward_closure('magni')
    mrs = np.array([my_reward([bg]) for bg in bgs])
    mr10 = f'Rew(BG = 10) = {mrs[0]:.4f}'
    mr600 = f'Rew(BG = 600) = {mrs[-1]:.4f}'
    mrmax = f'max(Rew) = {np.max(mrs):.4f}'
    margmax = f'argmax(Rew) = {np.argmax(mrs)}'
    plt.plot(mrs)
    plt.axvspan(70, 180, alpha=0.2)
    txt = '\n'.join([mr10, mr600, mrmax, margmax])
    plt.text(0.7, 0.1, txt, horizontalalignment='center', fontsize=10., transform=plt.gca().transAxes)
    plt.title('Magni Reward with baseline = 400')
    plt.xlabel('Blood Glucose (mg/dL)')
    plt.ylabel('Reward = -Risk + Baseline')
    plt.savefig('magni_rew_baseline400.png')
    plt.close()
    
    my_reward = my_reward_closure('kovatchev')
    krs = np.array([my_reward([bg]) for bg in bgs])
    kr10 = f'Rew(BG = 10) = {krs[0]:.4f}'
    kr600 = f'Rew(BG = 600) = {krs[-1]:.4f}'
    krmax = f'max(Rew) = {np.max(krs):.4f}'
    kargmax = f'argmax(Rew) = {np.argmax(krs)}'
    plt.plot(krs)
    plt.axvspan(70, 180, alpha=0.2)
    txt = '\n'.join([kr10, kr600, krmax, kargmax])
    plt.text(0.7, 0.1, txt, horizontalalignment='center', fontsize=10., transform=plt.gca().transAxes)
    plt.title('Kovatchev Reward with baseline = 200')
    plt.xlabel('Blood Glucose (mg/dL)')
    plt.ylabel('Reward = -Risk + Baseline')
    plt.savefig('kovatchev_rew_baseline200.png')
    plt.close()