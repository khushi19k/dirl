import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d
from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE

# Constants          
gtop = np.array([5.0, 10.0])

# Helper: Trim to goal
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

# path1 = os.path.join("results","car2d",f"spec{0}","hierarchy",f"start_{start_pos}.0.0",f"iters_{50}",f"traj_start_{start_pos}.0_states.csv")
# path2 = os.path.join("results","car2d",f"spec{0}","hierarchy",f"start_{start_pos}",f"iters_{50}",f"traj_start_{start_pos}.0_states.csv")

# Helper: Load, trim, and preprocess
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

# Load all
trajectories = []
features = []
start_ids = []
for i in range(1, 11):
    traj = load_trimmed_trajectory(i,1)
    trajectories.append(traj.flatten())
    start_ids.append(i)
X = np.array(trajectories)

# Try clustering for k = 2 to 6
scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append((k, score))
    print(f"k = {k}, silhouette = {score:.3f}")

# Final clustering with best k
best_k = max(scores, key=lambda score: score[1])[0]
kmeans = KMeans(n_clusters=best_k, random_state=0)
final_labels = kmeans.fit_predict(X)
sil_samples = silhouette_samples(X, final_labels)

# Identifying outlier
for i, score in enumerate(sil_samples):
    if score < 0.2:
        print(f"x0={start_ids[i]} has low silhouette: {score:.2f}")

test_start_pos=[1.9698142441959599,2.317819708796308,3.890075765713413,3.2583199488200365,6.012365139866942,6.357638596209332,6.706342335826177,8.873209749310195,8.971811977846885]
X_test   = np.stack([load_trimmed_trajectory(tp,0).flatten() for tp in test_start_pos])
test_labs = kmeans.predict(X_test)
test_sils = silhouette_samples(X_test, test_labs)

print("\nTest Trajectory Cluster Assignments:")
for tp, lbl, sil in zip(test_start_pos, test_labs, test_sils):
    print(f"  start={tp:.4f} → cluster={lbl}, silhouette={sil:.3f}")

# overall silhouette on test
print(f"\nMean silhouette on test set: {silhouette_score(X_test, test_labs):.4f}")

# Plot
colors = ['red','blue']
for i, label in enumerate(final_labels):
    traj_flat = X[i]
    traj_xy = traj_flat.reshape(10, 2)
    plt.plot(traj_xy[:, 0], traj_xy[:, 1], color=colors[label], label=f"x₀={start_ids[i]}", marker=".")
for tp, lab in zip(test_start_pos, test_labs):
    feat = X_test[test_start_pos.index(tp)]
    traj = feat.reshape(10,2)
    plt.plot(traj[:,0], traj[:,1], "--", c=colors[lab], lw=2, label=f"test {tp:.2f}")

# Goal + obstacle
plt.scatter(gtop[0], gtop[1], s=150, marker="s", color="purple", label="Goal")
plt.gca().add_patch(plt.Rectangle((4.0, 4.0), 2.0, 2.0, facecolor="lightblue", alpha=0.5, label="Obstacle"))
plt.title(f"Trajectory Clustering (k={best_k})")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# To convert 10*10D data to 2D data
X_2d = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X)

# Plot 2D points colored by cluster
for i in range(best_k):
    mask = final_labels == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {i}", alpha=0.7)
for i, label in enumerate(start_ids):
    plt.text(X_2d[i, 0], X_2d[i, 1], f"x₀={label}", fontsize=8)
plt.title(f"Trajectory Clusters (k={best_k})")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

