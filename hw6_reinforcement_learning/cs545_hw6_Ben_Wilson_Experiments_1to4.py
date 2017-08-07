# # CS 545 - Machine Learning
# Homework #6
# Author - Ben Wilson
# Due Date - March 16, 2017
#
# The following code is meant to be run as is in a python interpreter.
# The first four experiments are listed sequentially at the bottom of
# this file and will all run at the same time. Each experiment generates
# a plot and returns the test average and test standard deviation.

import numpy as np
import matplotlib.pyplot as plt

class BotWorld(object):
    def __init__(self, height=10, width=10):
        ''' Create an enviroment for a robot '''
        # Save the dimensions of the bot world
        self.n_units = height * width
        self.n_rows = height
        self.n_cols = width
        
        # Create the grid and populate it with cans
        self.reset_world()
        return

    def reset_world(self, fill=0.5):
        ''' Repopulates the grid world with randomly placed cans '''
        # Create empty world
        self.grid = np.zeros(self.n_units)

        # Determine the number of cans to use
        self.n_cans = int(self.n_units * fill)

        # Randomly select placement of cans
        samples = np.random.choice(self.n_units, self.n_cans, replace=False)
        self.grid[samples] = 1

        # Reshape into the corect dimension
        self.grid = self.grid.reshape(self.n_rows, self.n_cols)
        return

    def disp(self):
        print(self.grid)
        return


class Robot(object):
    def __init__(self, world, eta=0.2, gamma=0.9, epsilon=1.0, tax=0, states=161, acts=5):
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.tax = tax
        self.n_states = states
        self.n_actions = acts
        self.reset_position(world)

        # Generate Q-matrix, initialized to zeros
        self.q_matrix = np.zeros((self.n_states, self.n_actions))
        return

    def reset_position(self, world):
        # Place robot on random grid square
        r = np.random.randint(world.n_rows)
        c = np.random.randint(world.n_cols)
        self.position = [r,c]
        return

    def get_state(self, world):
        ''' !!! THIS METHOD NEEDS TO BE UPDATED TO MAKE IT CLEANER !!! '''
        r,c = self.position     # Distance to North and West walls
        cp = world.n_cols-1 - c # Distance to the East wall
        rp = world.n_rows-1 - r # Distance to the South wall

        state = 0
        # Look at current position
        if world.grid[r,c]: # Can at current position
            state += 81

        # Look NORTH
        if r: # Not a wall
            if world.grid[r-1,c]: # Can north of postiion
                state += 27
        else: # Wall to the north
            state += 54

        # Look EAST
        if cp: # Not a wall
            if world.grid[r,c+1]: # Can east of postiion
                state += 9
        else: # Wall to the east
            state += 18

        # Look SOUTH
        if rp: # Not a wall
            if world.grid[r+1,c]: # Can south of postiion
                state += 3 
        else: # Wall to the south 
            state += 6

        # Look WEST 
        if c: # Not a wall
            if world.grid[r,c-1]: # Can west of postiion
                state += 1 
        else: # Wall to the west 
            state += 2
        return state

    def choose_action(self, state):
        exploit = np.random.uniform()
        if exploit > self.epsilon: # Exploit
            # Get the row from the q-table corresponding to the state
            q_row = self.q_matrix[state,:]

            # Find the best actions for that state
            best_act = np.argwhere(q_row == np.amax(q_row))

            # If there is a tie for best actions, break it at random
            if len(best_act) > 1:
                tie_break = np.random.randint(len(best_act))
                action = int(best_act[tie_break])
            else:
                action = int(best_act[0])
        else: # Explore
            action = np.random.randint(self.n_actions)
        return action

    def do_action(self, state, action, world):
        r,c = self.position
        reward = 0
        if action == 0: # Try to pick up a can
            if state > 80: # There is a can to pick up
                world.grid[r,c] = 0
                reward = 10
            else: # Tried to pick up on an empty location
                reward = -1
        elif action == 1: # Try to move NORTH
            if r:
                r -= 1
            else:
                reward = -5
        elif action == 2: # Try to move EAST
            if world.n_cols-1 - c:
                c += 1
            else:
                reward = -5
        elif action == 3: # Try to move SOUTH
            if world.n_rows-1 - r:
                r += 1
            else:
                reward = -5
        else: # Try to move WEST
            if c:
                c -= 1
            else:
                reward = -5

        reward -= self.tax
        self.position = [r,c]
        next_state = self.get_state(world)
        return reward, next_state

    def update_q(self, state, new_state, action, reward):
        next_best_act = np.max(self.q_matrix[new_state,:])
        current_q = self.q_matrix[state, action] 
        self.q_matrix[state, action] += self.eta*(float(reward) + self.gamma*next_best_act - float(current_q))
        return 

def train_bot(bot, world, fill=0.5, episode=5000, step=200, decay=True):
    # Create lists to hold plotting data
    x = [0]
    y = [0]

    for n in range(episode):
        # reset cans, randomly place robot, reset reward
        world.reset_world(fill=fill)
        robby.reset_position(world)
        tot_reward = 0

        for m in range(step):
            state = robby.get_state(world)
            action = robby.choose_action(state)
            reward, next_state = robby.do_action(state, action, world)
            robby.update_q(state, next_state, action, reward)
            tot_reward += reward

        if decay:
            # decrease the epsilon value by 0.01 every 50 episodes
            if (n+1) % 50 == 0 and robby.epsilon > 0.1:
                robby.epsilon -= 0.01

        # Save reward every 100 episodes
        if (n+1) % 100 == 0:
            x.append(n)
            y.append(tot_reward)
    return x, y

def test_bot(bot, world, fill=0.5, episode=5000, step=200):
    # Create an array to hold total reward data 
    y = np.zeros(episode) 

    for n in range(episode):
        # reset cans, randomly place robot, reset reward
        world.reset_world(fill=fill)
        robby.reset_position(world)
        tot_reward = 0

        for m in range(step):
            state = robby.get_state(world)
            action = robby.choose_action(state)
            reward, next_state = robby.do_action(state, action, world)
            robby.update_q(state, next_state, action, reward)
            tot_reward += reward

        y[n] = tot_reward

    test_avg = np.mean(y)
    test_std = np.std(y)
    return test_avg, test_std

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #1 -----------------------------
    # -------------------------------------------------------------------------
    # Create the grid world 
    world = BotWorld()

    # Add the robot to the grid world
    robby = Robot(world)

    # Train robot
    x,y = train_bot(robby, world, fill=0.5, episode=5000, step=200, decay=True)

    # Plot the results of the training session
    plt.plot(x, y, label='Training Session')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Experiment #1 - Training Reward Plot')
    # plt.show()
    file_name = 'experiment_1_training_reward.png'
    plt.savefig(file_name)
    plt.clf()

    # Keep epsilon constant at 0.1
    robby.epsilon = 0.1
    
    # Test robot
    mean, std = test_bot(robby, world, fill=0.5, episode=5000, step=200)
    print('mean = ' + str(mean))
    print('std  = ' + str(std))

    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #2 -----------------------------
    # -------------------------------------------------------------------------
    # Create the grid world 
    world = BotWorld()

    learn_rates = [0.05, 0.50, 0.75, 1.00]
    for eta in learn_rates:
        # Add the robot to the grid world
        robby = Robot(world)

        # Update the learning rate
        robby.eta = eta

        # Train robot
        x,y = train_bot(robby, world, fill=0.5, episode=5000, step=200, decay=True)

        # Plot the results of the training session
        plt.plot(x, y, label='Training Session')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Training Reward Plot\nExperiment #2 - eta = '+str(eta))
        # plt.show()
        file_name = 'experiment_2_training_reward' + str(eta) + '.png'
        plt.savefig(file_name)
        plt.clf()

        # Keep epsilon constant at 0.1
        robby.epsilon = 0.1
        
        # Test robot
        mean, std = test_bot(robby, world, fill=0.5, episode=5000, step=200)
        print('mean = ' + str(mean))
        print('std  = ' + str(std))

    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #3 -----------------------------
    # -------------------------------------------------------------------------
    # Create the grid world 
    world = BotWorld()

    # Add the robot to the grid world
    robby = Robot(world, epsilon=0.1)

    # Train robot
    x,y = train_bot(robby, world, fill=0.5, episode=5000, step=200, decay=False)

    # Plot the results of the training session
    plt.plot(x, y, label='Training Session')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Plot\nExperiment #3 - epsilon = '+str(robby.epsilon))
    # plt.show()
    file_name = 'experiment_3_training_reward' + str(robby.epsilon) + '.png'
    plt.savefig(file_name)
    plt.clf()

    # Test robot
    mean, std = test_bot(robby, world, fill=0.5, episode=5000, step=200)
    print('mean = ' + str(mean))
    print('std  = ' + str(std))

    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #4 -----------------------------
    # -------------------------------------------------------------------------
    # Create the grid world 
    world = BotWorld()

    # Add the robot to the grid world
    robby = Robot(world, tax=0.5)

    # Train robot
    x,y = train_bot(robby, world, fill=0.5, episode=5000, step=200, decay=True)

    # Plot the results of the training session
    plt.plot(x, y, label='Training Session')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Plot\nExperiment #4 - tax = -' + str(robby.tax))
    # plt.show()
    file_name = 'experiment_4_training_reward.png'
    plt.savefig(file_name)
    plt.clf()
    
    # Keep epsilon constant at 0.01
    robby.epsilon = 0.1

    # Test robot
    mean, std = test_bot(robby, world, fill=0.5, episode=5000, step=200)
    print('mean = ' + str(mean))
    print('std  = ' + str(std))

    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #5 -----------------------------
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # --------------------------- in a seperate file --------------------------
    # -------------------------------------------------------------------------
    
