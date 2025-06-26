from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.monitor import Resource_Model
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams
from spectrl.envs.car2d import VC_Env
from numpy import linalg as LA

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # --- ADDITION: for plotting

num_iters = [30]

def resource_delta(sys_state, res_state, sys_action):
    return np.array([res_state[0] - 0.1 * abs(sys_state[0]) * LA.norm(sys_action)])


def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate

def avoid(obstacle):
    def predicate(sys_state, res_state):
        return 10 * max([obstacle[0] - sys_state[0],
                         obstacle[1] - sys_state[1],
                         sys_state[0] - obstacle[2],
                         sys_state[1] - obstacle[3]])
    return predicate


def have_fuel(sys_state, res_state):
    return res_state[0]


# Goals and obstacles
gtop = np.array([5.0, 10.0])
gbot = np.array([5.0, 0.0])
gright = np.array([10.0, 0.0])
gcorner = np.array([10.0, 10.0])
gcorner2 = np.array([0.0, 10.0])
origin = np.array([0.0, 0.0])
obs = np.array([4.0, 4.0, 6.0, 6.0])

# Specifications
spec1 = alw(avoid(obs), ev(reach(gtop, 1.0)))
spec2 = alw(avoid(obs), alw(have_fuel, ev(reach(gtop, 1.0))))
spec3 = seq(alw(avoid(obs), ev(reach(gtop, 1.0))),
            alw(avoid(obs), ev(reach(gbot, 1.0))))
spec4 = seq(choose(alw(avoid(obs), ev(reach(gtop, 1.0))), alw(avoid(obs), ev(reach(gright, 1.0)))),
            alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec5 = seq(spec3, alw(avoid(obs), ev(reach(gright, 1.0))))
spec6 = seq(spec5, alw(avoid(obs), ev(reach(gcorner, 1.0))))
spec7 = seq(spec6, alw(avoid(obs), ev(reach(origin, 1.0))))


# Examples: Choice but greedy doesn't work
gt1 = np.array([3.0, 4.0])
gt2 = np.array([6.0, 0.0])
gfinal = np.array([3.0, 7.0])
gfinal2 = np.array([7.0, 4.0])

spec8 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal, 0.5))))

spec9 = seq(choose(alw(avoid(obs), ev(reach(gt2, 0.5))), alw(avoid(obs), ev(reach(gt1, 0.5)))),
            alw(avoid(obs), ev(reach(gfinal2, 0.5))))

spec10 = choose(alw(avoid(obs), ev(reach(gt1, 0.5))),
                alw(avoid(obs), ev(reach(gt2, 0.5))))

specs = [spec1, spec2, spec3, spec4, spec5, spec6, spec7, spec8, spec9, spec10]

lb = [10., 20., 10., 10., 10., 9., 9., 9., 9., 9.]


def plot_trajectory(traj_xy, iters, folder, spec_num, goal):
    plt.figure(figsize=(8, 6))
    # obstacle
    rect = Rectangle((4.0, 4.0), 2.0, 2.0,facecolor="lightblue", edgecolor="navy",alpha=0.5, label="Obstacle")
    plt.gca().add_patch(rect)
    # goal
    plt.scatter(goal[0], goal[1], color="purple", s=150, marker="s", label="Goal")
    # Compute Chebyshev distance to goal at every step
    xs_full = traj_xy[:, 0]
    ys_full = traj_xy[:, 1]
    cheby = np.maximum(np.abs(xs_full - 5.0), np.abs(ys_full - 10.0))
    goal_idx = np.nonzero(cheby <= 1.0)[0]
    if len(goal_idx) > 0:
        end = goal_idx[0]
        xs = xs_full[: end + 1]
        ys = ys_full[: end + 1]
    else:
        xs = xs_full
        ys = ys_full
    plt.plot(xs, ys,marker="o", linestyle="-",color="tab:blue", alpha=0.8,label=f"Start x₀={start_pos}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Trajectory from x₀={start_pos} up to Goal ")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize="small")
    plt.tight_layout()

    # Save
    out_dir = os.path.join(folder, f"spec{spec_num}", "hierarchy",f"start_{start_pos}.0", f"iters_{iters}")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"traj_start_{start_pos}_plot.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    log_info = []

    test_start_positions = np.random.uniform(0.0, 10.0, size=10)  
    for start_pos in test_start_positions:
        for i in num_iters:
            hyperparams = HyperParams(30, i, 20, 8, 0.05, 1, 0.2)

            print(f'\n**** Learning Policy for Spec {spec_num} '
                    f'with {i} Iterations, start_x={start_pos} ****')

            # Step 1: initialize system environment
            system = VC_Env(500, start_pos=start_pos, std=0.05)

            # We will override after reset inside the wrapper—but set a default here:
            system.state = np.array([float(start_pos), 0.0], dtype=np.float32)
            system.time = 0

            # Step 2 (optional): construct resource model
            resource = Resource_Model(2, 2, 1, np.array([7.0]), resource_delta)

            # Step 3: construct abstract reachability graph
            _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
            print('\n**** Abstract Graph ****')
            abstract_reach.pretty_print()

            # Step 5: Learn policy
            abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
                system, hyperparams, res_model=resource, max_steps=20,
                neg_inf=-lb[spec_num], safety_penalty=-1, num_samples=500, render=render)

            # Test policy
            hierarchical_policy = HierarchicalPolicy(abstract_policy, nn_policies, abstract_reach.abstract_graph, 2)
            final_env = ConstrainedEnv(system, abstract_reach, abstract_policy, res_model=resource, max_steps=60)

            # Print statements
            _, prob = print_performance(final_env, hierarchical_policy, stateful_policy=True)
            print('\nTotal Sample Steps: {}'.format(stats[0]))
            print('Total Time Taken: {} mins'.format(stats[1]))
            print('Total Edges Learned: {}'.format(stats[2]))

            # Initialize and reset the environment
            sys_state, res_state, abs_state = final_env.reset()
            final_env.wrapped_env.state = np.array([float(start_pos), 0.0], dtype=np.float32)
            final_env.wrapped_env.time = 0
            final_env.res_state = final_env.res_model.res_init.copy()
            hierarchical_policy.reset()

            # Collect one rollout until goal is reached
            trajectory_xy = []
            trajectory_actions = []
            trajectory_rewards = []
            current_sys = final_env.wrapped_env.state
            current_res = final_env.res_state

            max_steps_rollout = 150
            for t in range(max_steps_rollout):
                obs_full = np.concatenate([current_sys, current_res])
                action = hierarchical_policy.get_action(obs_full)
                # Take a step in the env
                next_obs, reward, done, info = final_env.step(action)
                next_sys = next_obs[:2]
                next_res = next_obs[2:]

                # Log x,y, action
                trajectory_xy.append([float(next_sys[0]), float(next_sys[1])])
                trajectory_actions.append([float(action[0]), float(action[1])])

                # Update
                current_sys = next_sys.copy()
                current_res = next_res.copy()

            # Convert to NumPy arrays
            traj_xy_arr = np.array(trajectory_xy)   
            traj_act_arr = np.array(trajectory_actions)   

            rollout_folder = os.path.join(folder,f"spec{spec_num}","hierarchy",f"start_{start_pos}.0",f"iters_{i}")
            os.makedirs(rollout_folder, exist_ok=True)
            # states csv
            states_csv = os.path.join(rollout_folder, f"traj_start_{start_pos}_states.csv")
            with open(states_csv, "w", newline="") as f_states:
                writer = csv.writer(f_states)
                writer.writerow(["t", "x", "y"])
                for idx_step, (xx, yy) in enumerate(traj_xy_arr):
                    writer.writerow([idx_step, xx, yy])
            print(f"→ Saved STATES CSV  → {states_csv}")

            # actions csv
            actions_csv = os.path.join(rollout_folder, f"traj_start_{start_pos}_actions.csv")
            with open(actions_csv, "w", newline="") as f_actions:
                writer = csv.writer(f_actions)
                writer.writerow(["t", "u1", "u2"])
                for idx_step, (u1, u2) in enumerate((traj_act_arr)):
                    writer.writerow([idx_step, u1, u2])
            print(f"→ Saved ACTIONS CSV → {actions_csv}")

            # Plot the trajectory
            plot_trajectory(traj_xy_arr,  i, folder, spec_num,gtop)

logdir = os.path.join(folder, f"spec{spec_num}", "hierarchy")
if not os.path.exists(logdir):
    os.makedirs(logdir)
save_log_info(log_info, itno, logdir)
