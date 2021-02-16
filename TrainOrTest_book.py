#---------------------------------
# AUTHOR: KAUSHIK BALAKRISHNAN
#---------------------------------

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import sys

from replay_buffer_book import ReplayBuffer
from AandC_book import *


def trainDDPG(sess, env, args, actor, critic):

    sess.run(tf.global_variables_initializer())
 

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # start training on episodes 
    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim))) 
            
            t = env.time(j,int(args['max_episode_len']) )
            bs = env.befstate(s)
            
            s2, r, terminal, info = env.step(a[0])
            
            #print(j, s2, r, Q, a)<---

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding tuple to replay buffer until there are at least minibatch number of samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate target q; predicts for a state and an action
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = [] #predicted q value from the target network
                
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:             #si es terminal entonces solo la reward porque no habra mas
                        y_i.append(r_batch[k])
                    else:                       #si no es terminal entonces se suma el Q value futuro
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update critic
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1))) 

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r  

            if terminal:
                print('|Episode: {:d} | Reward:{:.4f} | Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))
                f = open("cstrmodel.txt", "a+")
                f.write(str(i) + " " + str(int(ep_reward)) + " " + str(ep_ave_max_q / float(j)) + '\n')  
                break
            
#-----------------------------------------------------------------------------

def testDDPG(sess, env, args, actor, critic):


    # test for max_episodes number of episodes
    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        
        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim))) 
            
            t = env.time(j,int(args['max_episode_len']) )
            bs = env.befstate(s)

            s2, r, terminal, info = env.step(a[0])

            s = s2
            ep_reward += r

            if terminal:
                print('| Episode: {:d} | Reward: {:.5f} |'.format(i, ep_reward))
                break
