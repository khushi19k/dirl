import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy.interpolate import interp1d
          
gtop = np.array([5.0, 10.0])

def trim_to_goal(traj, goal=gtop):
    xs, ys = traj[:, 0], traj[:, 1]
    cheby = np.maximum(np.abs(xs - goal[0]), np.abs(ys - goal[1]))
    idxs = np.nonzero(cheby <= 1.0)[0]
    return traj[:idxs[0]+1] if len(idxs) > 0 else traj

def preprocess_trajectory(traj, goal=gtop, step=2, final_length=10):
    sampled = traj[::step]
    if len(sampled) > final_length:
        sampled = sampled[:final_length]
    while len(sampled) < final_length:
        sampled = np.vstack([sampled, goal])
    return sampled

def load_trimmed_trajectory(start_pos,flag):
    if(flag==1):
        path = os.path.join("results","car2d",f"spec{0}","hierarchy",f"start_{start_pos}.0.0",f"iters_{50}",f"traj_start_{start_pos}.0_states.csv")
    else:
        path = os.path.join("results","car2d",f"spec{0}","hierarchy",f"start_{start_pos}.0",f"iters_{30}",f"traj_start_{start_pos}_states.csv")
    df = pd.read_csv(path)
    traj = df[['x', 'y']].to_numpy()
    traj_trim = trim_to_goal(traj)
    # return preprocess_trajectory(traj_trim, goal=gtop, step=3, final_length=10)
    t_old = np.linspace(0, 1, len(traj_trim))
    t_new = np.linspace(0, 1, 10)
    fx = interp1d(t_old, traj_trim[:, 0], kind='linear', fill_value="extrapolate")
    fy = interp1d(t_old, traj_trim[:, 1], kind='linear', fill_value="extrapolate")
    x_new = fx(t_new)
    y_new = fy(t_new)
    return np.stack([x_new, y_new], axis=1)

# build training set
trajectories = []
features = []
start_ids = []
train_start_pos = list(range(1,11))
for i in range(1, 11):
    traj = load_trimmed_trajectory(i,1)
    trajectories.append(traj.flatten())
    start_ids.append(i)
X = np.array(trajectories)

# build test set
test_start_pos=[1.9698142441959599,2.317819708796308,3.890075765713413,3.2583199488200365,6.012365139866942,6.357638596209332,6.706342335826177,8.873209749310195,8.971811977846885]
X_test = np.stack([load_trimmed_trajectory(tp,0).flatten() for tp in test_start_pos])

# Trying Fuzzy C-Means Clustering
# Transpose into (features, samples) for skfuzzy
data_train = X.T  
data_test  = X_test.T  
c = 2
# Train fuzzy c-means
cntr, u_train,  _, _, _, _, fpc = fuzz.cluster.cmeans(
    data_train, c=c, m=2.0, error=1e-5, maxiter=100
)
# Test fuzzy c-means
u_test, u0_test, d_test, jm_test, p_test, fpc_test = fuzz.cluster.cmeans_predict(
    data_test, cntr, m=2.0, error=1e-5, maxiter=100
)
print(fpc)
print("Training set data Probabilities:")
for i, s in enumerate(train_start_pos):
    probs = u_train[:, i]
    prob_str = ", ".join([f"C{j}:{probs[j]:.2f}" for j in range(c)])
    print(f" start={i:>2} : {prob_str}")

print("\Test set data Probabilities:")
for i, tp in enumerate(test_start_pos):
    probs = u_test[:, i]
    prob_str = ", ".join([f"C{j}:{probs[j]:.2f}" for j in range(c)])
    print(f" start={tp:.4f} : {prob_str}")

for i, tp in enumerate(test_start_pos):
    plt.figure()
    plt.bar(np.arange(c), u_test[:, i])
    plt.xticks(np.arange(c), [f"Cluster {j}" for j in range(c)])
    plt.ylim(0,1)
    plt.title(f"Start {tp:.4f}")
    plt.xlabel("Cluster")
    plt.ylabel("probability")
    plt.tight_layout()
plt.show()

ambig = []
for i, s in enumerate(train_start_pos):
    if u_train[0, i] >= 0.2 and u_train[1, i] >= 0.2:
        ambig.append(s)
print("Ambiguous training points:", ambig)

# same for test
ambig_test = []
for i, tp in enumerate(test_start_pos):
    if u_test[0, i] >= 0.2 and u_test[1, i] >= 0.2:
        ambig_test.append(tp)
print("Ambiguous test points:", ambig_test)
