from os import error
import numpy as np
import tensorflow as tf
from robot_simulator import NYUFingerSimulator
import pinocchio as pin
from time import sleep

# DDPG Reinforcement Learning
class ReplayBuffer:
    """Defines the Buffer dataset from which the agent learns"""
    def __init__(self, max_size, input_shape, dim_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, dim_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr +=1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, _states, dones
    
class Critic(tf.keras.Model):
    """Defines a Critic Deep Learning Network"""
    def __init__(self, dim_actions, H1_dim=512, H2_dim=512, name='critic'):
        super(Critic, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name+'.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu') 
        self.Q = tf.keras.layers.Dense(1, activation=None)
        
    def call(self, state, action):
        action = self.H1(tf.concat([state,action], axis=1))
        action = self.H2(action)
        Q = self.Q(action)
        return Q
    
class Actor(tf.keras.Model):
    """Defines a Actor Deep Learning Network"""
    def __init__(self, dim_actions, H1_dim=512, H2_dim=512, name='actor'):
        super(Actor, self).__init__()
        self.H1_dim = H1_dim
        self.H2_dim = H2_dim
        self.dim_actions = dim_actions
        self.model_name = name
        self.checkpoint = self.model_name+'.h5'
        self.H1 = tf.keras.layers.Dense(self.H1_dim, activation='relu')
        self.H2 = tf.keras.layers.Dense(self.H2_dim, activation='relu') 
        self.mu = tf.keras.layers.Dense(self.dim_actions, activation='tanh')
        
    def call(self, state):
        action_prob = self.H1(state)
        action_prob = self.H2(action_prob)
        mu = self.mu(action_prob)
        return mu
    
class Agent:
    """Defines a RL Agent based on Actor-Critc method"""
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=4, max_size=1000000, tau=0.005,
                 H1=512, H2=256, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)
        self.batch_size = batch_size
        self.noise = noise
        self.max_action = 100
        self.min_action = -100
        
        self.actor  = Actor(self.n_actions, name='actor')
        self.critic = Critic(self.n_actions, name='critic')
        self.target_actor  = Actor(self.n_actions, name='target_actor')
        self.target_critic = Critic(self.n_actions, name='target_critic')
        
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.update_networks(tau=1)
        
    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau
            
        weights=[]
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights=[]
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        
    def recall(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def save_models(self):
        self.actor.save_weights('forge/data/DeepRL/TauPredictionHER/'+self.actor.checkpoint)
        #self.critic.save_weights('forge/data/DeepRL/TauPredictionHER'+self.critic.checkpoint)
        #self.target_actor.save_weights(self.target_actor.checkpoint)
        #self.target_critic.save_weights(self.target_critic.checkpoint)
        
    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint)
        self.critic.load_weights(self.critic.checkpoint)
        self.target_actor.load_weights(self.target_actor.checkpoint)
        self.target_critic.load_weights(self.target_critic.checkpoint)
        
    def choose_action(self, observation):
        evaluate=False
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        self.update_networks()


# Main
if __name__ == "__main__":
    robot = NYUFingerSimulator()
    q0, v0 = robot.get_state()
    robot.reset_state(q0, v0)
    
    agent = Agent(input_dims=6, n_actions=3)
    best_score = -np.inf

    n_games = 5000
    score_history = []
    AvgScore_history = []
    
    for i in range(n_games):
        done = False
        score = 0
        random_q = np.random.uniform(np.pi/2, -np.pi/2, (3,))
        
        robot.reset_state(random_q, np.zeros(3))

        target_q = np.random.uniform(np.pi/2, -np.pi/2, (3,))
        
        for j in range(500):
            q, _ = robot.get_state()
            
            tau = agent.choose_action(np.hstack((q,target_q))).numpy()
            
            robot.send_joint_torque(tau)
            robot.step()
            
            next_q, _ = robot.get_state()
            
            done = np.array_equal(next_q, target_q)
            
            if done:
                reward = 1
            else:
                reward = -1
            
            score += reward
            
    
            # Add Normal Experience to Replay Buffer
            agent.recall(np.hstack((q,target_q)), tau, reward, np.hstack((next_q,target_q)), done)
            
            # Add 'Hindsight' Experience to memory
            agent.recall(np.hstack((q, next_q)), tau, 1, np.hstack((next_q, next_q)), True)
            agent.learn()
            
            if done:
                break
        
        # Log the scores
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        AvgScore_history.append(avg_score)
        
        np.save('forge/data/DeepRL/TauPredictionHER/score_history', score_history, allow_pickle=False)
        np.save('forge/data/DeepRL/TauPredictionHER/avg_history', AvgScore_history, allow_pickle=False)
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'Episode: {i} \t Sum. Episodic Reward: {score:.4f}') 