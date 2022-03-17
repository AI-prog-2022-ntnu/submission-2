import random


class NeuralNetworkConfig:
    def __init__(self,
                 episode_train_time_ms,
                 train_iterations,
                 batch_size,
                 lr,
                 data_passes=None,
                 use_iter=True
                 ):
        self.data_passes = data_passes
        self.use_iter = use_iter
        self.train_iterations = train_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.episode_train_time_ms = episode_train_time_ms


class EGreedy:
    def __init__(self,
                 init_val,
                 min_val,
                 rounds_to_min):
        self._init_epsilon = init_val
        self.epsilon = init_val
        self._epsilon_decay = (self.epsilon - min_val) / rounds_to_min
        self._epsilon_lb = min_val

    def round(self):
        self.epsilon = max(self.epsilon - self._epsilon_decay, self._epsilon_lb)

    def reset(self):
        """
        Resets the epsilon to the initial epsilon value.
        """
        self.epsilon = self._init_epsilon

    def should_pick_greedy(self, increment_round=False):
        """
        If the model should pick greedy or not.
        """
        ret = False
        if random.random() > self.epsilon:
            ret = True

        if increment_round:
            self.round()
        return ret


def generate_batch(x, y, batch_size):
    """
    Todo - Remove? Not used
    """
    batch_x = []
    batch_y = []
    while len(batch_x) < batch_size:
        idx = random.choice(range(len(x)))
        batch_x.append(x[idx])
        batch_y.append(y[idx])
    return batch_x, batch_y


def get_action_visit_map_as_target_vec(environment, action_visit_map: {}, invert=False):
    """
    Convert the vec to target distance.
    """
    possible_actions = environment.get_action_space()
    visit_sum = sum(action_visit_map.values())

    ret = []
    for action in possible_actions:
        if invert:
            value = action_visit_map.get((action[1], action[0]))
        else:
            value = action_visit_map.get(action)

        if value is None or visit_sum == 0:
            ret.append(0)
        else:
            ret.append(value / visit_sum)
    return ret
