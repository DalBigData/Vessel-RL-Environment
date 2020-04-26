class Message:
    def __init__(self):
        self.type = 'Message'
        pass

    def build(self):
        pass

    @staticmethod
    def parse(self):
        pass


class MessageClientConnectRequest(Message):
    def __init__(self, id_=0):
        super().__init__()
        self.type = "ClientConnectRequest"
        self.id = id_

    def build(self):
        msg = {"message_type": self.type, "value": {"id": self.id}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "ClientConnectRequest":
            message = MessageClientConnectRequest(msg['value']['id'])
            return True, message
        return False, None


class MessageClientConnectResponse(Message):
    def __init__(self, id_, vel, min_real_x, max_real_x, min_real_y, max_real_y):
        super().__init__()
        self.type = "MessageClientConnectResponse"
        self.id = id_
        self.vel = vel
        self.min_real_x = min_real_x
        self.max_real_x = max_real_x
        self.min_real_y = min_real_y
        self.max_real_y = max_real_y

    def build(self):
        msg = {"message_type": self.type,
               "value": {"id": self.id, "vel": self.vel, "min_real_x": self.min_real_x, "max_real_x": self.max_real_x,
                         "min_real_y": self.min_real_y, "max_real_y": self.max_real_y}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "MessageClientConnectResponse":
            message = MessageClientConnectResponse(msg['value']['id'], msg['value']['vel'], msg['value']['min_real_x'],
                                                   msg['value']['max_real_x'], msg['value']['min_real_y'],
                                                   msg['value']['max_real_y'])
            return True, message
        return False, None

    def __repr__(self):
        return str([self.id, self.vel, self.min_real_x, self.max_real_x, self.min_real_y, self.max_real_y])


class MessageClientDisconnect(Message):
    def __init__(self):
        super().__init__()
        self.type = "MessageClientDisconnect"

    def build(self):
        msg = {"message_type": self.type, "value": {}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "MessageClientDisconnect":
            message = MessageClientDisconnect()
            return True, message
        return False, None


class MessageClientWorld(Message):
    def __init__(self, cycle, world, reward, status, epoch_type):
        super().__init__()
        self.type = "MessageClientWorld"
        self.cycle = cycle
        self.world = world
        self.reward = reward
        self.status = status
        self.epoch_type = epoch_type

    def build(self):
        msg = {"message_type": self.type,
               "value": {"cycle": self.cycle, "reward": self.reward, "world": self.world, "status": self.status,
                         "epoch_type": self.epoch_type}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "MessageClientWorld":
            cycle = msg['value']['cycle']
            world = msg['value']['world']
            reward = msg['value']['reward']
            status = msg['value']['status']
            epoch_type = msg['value']['epoch_type']
            message = MessageClientWorld(cycle, world, reward, status, epoch_type)
            return True, message
        return False, None

    def __repr__(self):
        return self.type + ':' + str(self.cycle) + ',' + str(self.world) + ',' + str(self.reward) + ',' + str(
            self.status) + ',' + str(self.epoch_type)


class MessageEndPlan(Message):
    def __init__(self):
        super().__init__()
        self.type = "MessageEndPlan"

    def build(self):
        msg = {"message_type": self.type, "value": {}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "MessageEndPlan":
            message = MessageEndPlan()
            return True, message
        return False, None

    def __repr__(self):
        return self.type


class MessageClientAction(Message):
    def __init__(self, action=0):
        super().__init__()
        self.type = "MessageClientAction"
        self.action = action

    def build(self):
        msg = {"message_type": self.type, "value": {"action": self.action}}
        str_msg = str.encode(str(msg))
        return str_msg

    @staticmethod
    def parse(coded_msg):
        msg = eval(str(coded_msg.decode("utf-8")))
        if msg['message_type'] == "MessageClientAction":
            action = msg['value']['action']
            message = MessageClientAction(action)
            return True, message
        return False, None


def parse(coded_msg):
    ret = MessageClientConnectRequest.parse(coded_msg)
    if ret[0]:
        return ret[1]

    ret = MessageClientConnectResponse.parse(coded_msg)
    if ret[0]:
        return ret[1]

    ret = MessageClientDisconnect.parse(coded_msg)
    if ret[0]:
        return ret[1]

    ret = MessageClientAction.parse(coded_msg)
    if ret[0]:
        return ret[1]

    ret = MessageClientWorld.parse(coded_msg)
    if ret[0]:
        return ret[1]

    ret = MessageEndPlan.parse(coded_msg)
    if ret[0]:
        return ret[1]

    return Message()
