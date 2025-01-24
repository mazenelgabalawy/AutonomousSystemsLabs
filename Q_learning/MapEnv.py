from matplotlib import pyplot as plt
import numpy as np

class MapEnv:
    def __init__(self, map, goal, max_steps):
        self.map = map
        self.current_state = None
        self.goal = goal.astype(np.int32)
        self.actions = 4
        self.steps = 0
        self.valid_rows ,self.valid_cols = np.where(self.map == 0) # Free Cells
        self.max_steps = max_steps
        if map[goal[0], goal[1]] != 0:
            raise ValueError("Goal position is an obstacle")

    def reset(self):
        # start the agent in a random position within the map and return agent state (cell in which it is)
        self.steps = 0
        random_idx = np.random.choice(len(self.valid_rows))
        self.current_state = np.array([self.valid_rows[random_idx], self.valid_cols[random_idx]])
        print(self.current_state)
        return self.current_state

    def step(self, action):
      # this function applies the action taken and returns the obtained state, a reward and a boolean that says if the episode has ended (max steps or goal reached) or not (any other case)
      # action: 0 = up, 1 = down, 2 = left, 3 = right

      new_state = self.current_state.copy() # Copy curren state

      # get new_state according to the required action
      match action:
        case 0:
          new_state[1] += 1
        case 1:
          new_state[1] -= 1
        case 2:
          new_state[0] -= 1
        case 3:
          new_state[0] += 1

      if self.map[new_state[0], new_state[1]] == 1:
        self.current_state = self.current_state # Don't move if will hit obstacle
        reward = -1
      else:
        self.current_state = new_state # Update state
        reward = -1

      done = False
      self.steps += 1 # increment steps

      if self.current_state.all() == self.goal.all():
        done = True
        reward = 1

      if self.steps >= self.max_steps:
        done = True
        reward = -1

      return self.current_state, reward, done

    def get_state(self):
      # returns current state
      return self.current_state

    def render(self, i=0):
        plt.matshow(self.map, cmap = "jet")
        plt.title('Map')
        plt.colorbar()
        plt.scatter(self.current_state[1], self.current_state[0], c = 'r')
        plt.scatter(self.goal[1], self.goal[0], c = 'g')
        plt.savefig("q_learning_{0:04}.png".format(i), dpi = 300)
        plt.show()