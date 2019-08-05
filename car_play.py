import gym

import cv2
import tensorflow as tf
from collections import deque
import numpy as np
import random
import statistics


class DQNAgent():

    def __init__(self, MEMORY_SIZE=200, INITIAL_EPSILON=1, FINAL_EPSILON=0.0001, EXPLORE=10000, GAMMA=0.99,
                 q_model=None, actionspace=2):

        # experience buffer
        self.memory = deque(maxlen=MEMORY_SIZE)
        # discount rate
        self.gamma = GAMMA
        self.batch_size = 32
        self.EXPLORE = EXPLORE
        self.FINAL_EPSILON = FINAL_EPSILON
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.epsilon = self.INITIAL_EPSILON
        self.epsilon_step = (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

        # Q Network for training
        self.q_model = q_model
        self.actionspace = actionspace

    # save Q Network params to a file
    def save_weights(self, filepath):
        self.q_model.save_weights(filepath, overwrite=True)

    def load(self, name):
        print("Loading...", name)
        self.q_model.load_weights(name)

    def predict(self, state):
        # select the action with max Q-value
        return np.argmax(self.q_model.predict(np.expand_dims(state, axis=0))[0])

    # eps-greedy policy
    def act(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.actionspace)

        # exploit
        q_values = self.q_model.predict(np.expand_dims(state, axis=0))
        # select the action with max Q-value
        return np.argmax(q_values[0])

    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    # compute Q_max
    # use of target Q Network solves the non-stationarity problem
    def get_target_q_value(self, next_state, rt_1):
        # Q_max = reward + gamma * Q_max
        q_value = rt_1 + self.gamma * np.amax(self.q_model.predict(np.expand_dims(next_state, axis=0))[0])
        return q_value

    def replay(self, decay=True):

        # sars = state, action, reward, state' (next_state)
        train_batch = random.sample(self.memory, self.batch_size)
        s_t_batch = [x[0] for x in train_batch]
        a_t_batch = [x[1] for x in train_batch]
        r_t_batch = [x[2] for x in train_batch]
        s_t_1_batch = [x[3] for x in train_batch]
        terminal_batch = [x[4] for x in train_batch]

        q_values_batch = self.q_model.predict(np.array(s_t_batch))

        for i in range(len(train_batch)):

            if terminal_batch[i]:
                q_values_batch[i][a_t_batch[i]] = r_t_batch[i]
            else:
                q_values_batch[i][a_t_batch[i]] = self.get_target_q_value(s_t_1_batch[i], r_t_batch[i])

        # train the Q-network
        self.q_model.fit(np.array(s_t_batch),
                         np.array(q_values_batch),
                         batch_size=self.batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        if decay and self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= self.epsilon_step


def build_model(input_shape,out_shape,LEARNING_RATE):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=input_shape))  # 80*80*4
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(out_shape))
    adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    return model




env = gym.make('CarRacing-v0')

action_buffer = np.array([
                    [0.0, 0.0, 0.0],     #Brake
                    [-0.6, 0.05, 0.0],   #Sharp left
                    [0.6, 0.05, 0.0],    #Sharp right
                    [0.0, 0.3, 0.0]] )   #Staight

q_model = build_model(input_shape=(80, 80, 4), out_shape=len(action_buffer), LEARNING_RATE=1e-5)
agent = DQNAgent(MEMORY_SIZE=25000, INITIAL_EPSILON=1, FINAL_EPSILON=0.1, EXPLORE=100000,actionspace=len(action_buffer), q_model=q_model)


#agent.load("CarRacing_95000.h5")
agent.load("CarRacing_770000.h5")



def process_img(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:80,:,1]
    image = cv2.resize(image, (80, 80))
    return image

t=0
max_step = 25

rewards = []

for ep in range(1, 101):

    st = process_img(env.reset())


    total_reward = 0

    done = False
    timesteps =0


    st = np.stack((st, st, st, st), axis=2)

    negative_reward = 0
    tiles = 0
    decay = True
    neg_reward = 0



    while not done:

        #env.render()


        st_1, rt, done, _ = env.step(action_buffer[agent.predict(st)])
        st_1 = np.append(np.expand_dims(process_img(st_1), axis=2), st[:, :, :3], axis=2)

        total_reward += rt

        st = st_1

        if rt <0:
            neg_reward += 1
            if neg_reward >=max_step:
                done= True

        else:
            neg_reward = 0
            tiles += 1

        if done:


            rewards.append(total_reward)
            print(f"Episode {ep}, Tiles: {tiles}, Reward :{total_reward} Mean Reward: {statistics.mean(rewards)}")



env.close()

max(rewards)
print(statistics.mean(rewards))

print(f"Max Reward {max(rewards)}, Min: {min(rewards)}, Average Reward: {statistics.mean(rewards)} Std :{statistics.stdev(rewards)}")