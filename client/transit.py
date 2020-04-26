class Transition:
    def __init__(self, state, action, reward, next_state, value=0):
        self.state = state
        self.action = action
        self.reward = reward
        self.value = value
        self.next_state = next_state
        self.is_end = True if next_state is None else False

    def __repr__(self):
        return str(self.state) + ' with ' + str(self.action) + ' to ' + str(self.next_state) + ' r: ' + str(self.reward) + ' v: ' + str(self.value)

