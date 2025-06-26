import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spectrl.envs.car2d import VC_Env  

if __name__ == "__main__":
    env = VC_Env(time_limit=50, std=0.2)
    state = env.reset()
    trajectory = [state.copy()]
    done = False

    while not done:
        try:
            action = env.sample_action()
        except AttributeError:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        trajectory.append(next_state.copy())
        if done:
            break

    trajectory = np.array(trajectory)
    plt.figure(figsize=(6, 6))
    rect = Rectangle(
        (obs[0], obs[1]),           
        obs[2] - obs[0],               
        obs[3] - obs[1],                
        fill=False,
        edgecolor='black',
        linewidth=1.5
    )
    plt.gca().add_patch(rect)
    goals = [
        np.array([5.0, 10.0]),
        np.array([5.0,  0.0]),
        np.array([10.0, 0.0]),
        np.array([10.0,10.0]),
        np.array([ 0.0,10.0])
    ]
    for g in goals:
        plt.scatter(g[0], g[1], marker='*', s=100, color='purple')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='o', s=80, color='green')
    plt.plot(trajectory[:, 0], trajectory[:, 1], color='orange', linewidth=2)

    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Example VC_Env Trajectory')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
