import numpy as np

class Environment():
    def __init__(self, dw_thread, serialChannel):
        self.dw_thread = dw_thread
        self.serial_channel = serialChannel

    def reset(self):
        PWM = 0
        MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
        self.serial_channel.serialConnection.write(MtrSpd.encode())

        return np.array([self.dw_thread.channel_data])

    def step(self, action):
        if action == 0:
            PWM = 0
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # stay

        elif action == 1:
            PWM = 150
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # go slow

        elif action == 2:
            PWM = 300
            MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
            self.serial_channel.serialConnection.write(MtrSpd.encode())
            # go fast

        angle = self.serial_channel.getSerialData()
        next_state = np.array([self.dw_thread.channel_data])
        print("state", next_state)

        reward = 0
        done = False

        if angle > 0:
            reward = angle
        if angle < -10:
            self.warning += 1
            done = False
            if self.warning >= 500:
                done = True
                PWM = 0
                MtrSpd = 'S' + str(PWM) + '%'  # '%' is our ending marker
                self.serial_channel.serialConnection.write(MtrSpd.encode())
                reward = -3000
                print('low angle')

        return next_state, reward, done