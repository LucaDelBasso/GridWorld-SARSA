import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#* Written by Luca Del Basso for COM3240 Adaptive Intelligence
#* I used some aspects of the lab - the agent update code in particular


def safe_sigmoid(signal):
    '''a sigmoid function that prevents overflow or underflow'''
    sig = signal.copy()
    sig = np.clip( sig, -500, 500)
    return 1.0/(1+np.exp(-sig))

def epsilon_greedy(signal,eps):
    '''classic epsilon greedy
       inputs:
             signal: the array to be chosen from
             eps: the epsilon value '''
    p = np.random.rand()
    if p < eps: 
        return np.random.randint(0,len(signal))
    else:
        return np.argmax(signal)

def softmax(qs,ts,c):
    '''a softmax function turning q values into probability
       distributions
       inputs: 
            qs: the q(s,a) valeus
            ts: the current trial iteration
            c: a scaling factor used for experimentation -> default is 1'''
    q=qs.copy()
    temp = 1/ts * c
    temp = np.clip(temp,0,1)
    e = np.exp((q-np.max(q))/temp)
    if np.isnan((np.sum(e))):
        print("hi")
    k = e /np.sum(e)
    return k


def robot_ann(area_length,n_actions,trials,learning_rate,epsilon,gamma,
              termination,c=1,softm=False,e_trace=None,debug_str="",extrawalls=False):
    '''run a SARSA / SARSA(lambda) reinforcement learning algorithm on an agent in a
       square grid world.
       
       inputs:
             area_lenth (int): the width of the area
             n_actions (int): how many actions the agent can take
             trials (int): how many trials to iterate over
             learning rate(float): a hyper parameter of the SARSA algorithm
             epsilon (float): used for if using epsilon greedy
             gamma(float): the discount factor, a hyperparameter of the SARSA algorithm
             termination(int): number of steps to exit after if terminal state is not acheived
             c (float): a scaling parameter used for softmax action selection
             softm(bool): whether to use softmax or epsilon greedy
             e_trace(float): whether to use SARSA(lambda) (if not None) or SARSA (if none)
             debug_str(str): used to print iteration numbers etc
             externalwalls(bool): whether or not to include the walls for the last part of the assignment

       returns:
             a rewards matrix, a total steps matrix, an optimum steps matrix and the weights matrix
             '''
             
    n_states = area_length**2
    Lambda = e_trace
    #* Initalise set of S
    states = np.eye(n_states)
    n_t = np.zeros(n_states)
    #* N E S W 
    #took this from the lab, was too good not to
    action_row_change = np.array([-1,0,+1,0])               #number of cell shifted in vertical as a function of the action
    action_col_change = np.array([0,+1,0,-1]) 

    #* Initialse all Q(s,a) arbitrarily 
    weights = np.random.rand(n_actions,n_states) 
    E = np.zeros((n_actions,n_states))
    #* outputs to plot
    rewards = np.zeros((1,trials))
    steps = np.zeros((1,trials))
    opt_steps = np.zeros((1,trials))

    #* set Q(terminal-state, _ ) = 0 as per SARSA algorithm
    if extrawalls and n_states == 100:
        terminal_pos = np.array([2,2])
    else:
        terminal_pos = np.array([2,2])#np.random.randint(area_length,size=2)
    t_index = np.ravel_multi_index(terminal_pos,dims=(area_length,area_length),order='C')
    weights[:,t_index] = 0 
    
    walls =np.array([[9,3],[8,3],[7,3],[6,3],[5,3],[6,7],[6,8],[6,9],[0,5],[1,5],[2,5],[3,5]])

    #* Repeat for each episode ...
    for trial in range(trials):
        step = 0

        #* Initialse S -start state
        agent_pos = np.random.randint(area_length,size=2)

        if extrawalls and n_states == 100: #*prevent agent from starting "inside" a wall
            while agent_pos.tolist() in walls.tolist():
                agent_pos = np.random.randint(area_length,size=2)

        a_index = np.ravel_multi_index(agent_pos,dims=(area_length,area_length),order='C')
        min_steps = abs(agent_pos[0]-terminal_pos[0])+abs(agent_pos[1]-terminal_pos[1])
        opt_steps[0,trial] = min_steps

        n_t += states[:,a_index]
        
        state = states[:,a_index].reshape((n_states,1)) # input vector
        #* Choose A from S using policy epsilon
        if softm:
            qs = safe_sigmoid(weights.dot(state)) 
            action =  np.random.choice([0,1,2,3],1,p = softmax(qs,trial+1,c).reshape(n_actions))[0] # ! {0:N, 1:E, 2:S, 3:W}
        else: #epsilon greedy
            qs = safe_sigmoid(weights.dot(state))
            action = epsilon_greedy(qs,epsilon)
        Qold = qs[action]
        rewardOld = 0
        output = np.zeros((n_actions,1))

        #* Repeat (for each step of the episode):
        while not (np.array_equal(agent_pos, terminal_pos)) and step != termination:
            reward = 0
            steps[0,trial] = step
            step +=1
            state_new = np.array([0,0])
            #* take action A
            state_new[0] = agent_pos[0] + action_row_change[action]
            state_new[1] = agent_pos[1] + action_col_change[action]
            
            r_check = state_new.copy()#to compare later
            if extrawalls and n_states == 100:
                if state_new.tolist() in walls.tolist():
                    state_new = agent_pos.copy()
            #put the robot back in grid if it goes out. Consider also the option to give a negative reward
            #* Observe S'
            if state_new[0] < 0:             
                state_new[0] = 0             #* hit top wall
            if state_new[0] >= area_length:
                state_new[0] = area_length-1 #* hit bottom wall
            if state_new[1] < 0:
                state_new[1] = 0             #* hit left wall
            if state_new[1] >= area_length:
                state_new[1] = area_length-1 #* hit right wall
            agent_pos = state_new

            #* Observe R: bump into wall? r = 0.1 , else r = 0.0, includes outer boundaries and inner walls
            reward = -0.1 if not (np.array_equal(r_check,agent_pos)) else  0.0

            #* Choose A' from S' using policy
            a_index = np.ravel_multi_index(agent_pos,dims=(area_length,area_length),order='C')
            n_t += states[:,a_index]
            state_p = states[:,a_index].reshape((n_states,1)) # get new state
            if softm:
                qs_p = safe_sigmoid(weights.dot(state_p))
                action_p = np.random.choice([0,1,2,3],1,p = softmax(qs_p,trial+1,c).reshape(n_actions))[0] # ! {0:N, 1:E, 2:S, 3:W}
            else:
                qs_p =safe_sigmoid(weights.dot(state_p)) 
                action_p = epsilon_greedy(qs_p,epsilon)
            
            #* resize
            output = np.zeros((n_actions,1))
            output[action,0] = 1

            delta = reward  -( Qold - gamma*qs_p[action_p])
            
            if e_trace:#* eligibility trace
                k = output.dot(state.T)                         
                E += k
                weights += learning_rate*delta*E
                E = E*gamma * Lambda

            else:     #* Update Q value for single pre-post connection
                weights += learning_rate * delta * output.dot(state.T)

            #* S <- S' ; A <- A' etc
            state = state_p 
            action = action_p
            Qold = qs_p[action_p]
            rewardOld = reward
            rewards[0,trial] += rewardOld

        print("rep: "+debug_str +" trial: "+ str(trial) + " step: "+ str(step),end="\r")
        
        if np.array_equal(agent_pos, terminal_pos): #* if in terminal state
            reward = 1
        else:
            reward = -1
        rewards[0,trial] += reward
        opt_steps[0,trial] =min_steps
        #* different update when in final state
        if e_trace:
            k = output.dot(state.T)                         
            E += k
            weights += learning_rate*(reward-Qold)*E
        else:
            weights += learning_rate* (reward-Qold) * output.dot(state.T)
    return rewards, steps,opt_steps,weights

if __name__ == "__main__":
    reps,ntrials,termination = 100,200,100 # ! O(N_states x N_actions) for the reward information to propagate. 10x10 : ntrials = 400
    fontSize=18
    al = 10 #area length
    soft = True
  

    total_rewards = np.zeros((reps,ntrials))
    total_steps = np.zeros((reps,ntrials))
    optimum_steps = np.zeros((reps,ntrials))
    weights = np.zeros((reps,4,100))

    #add another for loop here if you want to iterate through a list of say, values for epsilon, and then pass it to the function
    #create some other matrices to store all these experiment values and then do something fancy with them
    for j in range(reps):
        total_rewards[j,:],total_steps[j,:],optimum_steps[j,:],weights[j] =robot_ann(area_length=al,n_actions=4,trials=ntrials,learning_rate=0.7,  
            epsilon=0,gamma=0.9,termination=termination,c=1,softm=soft,e_trace=0.9,debug_str = str(j),extrawalls=False)

 
    #! HEATMAP CODE
    # w = np.mean(weights,axis=0)
    # #mean values
    # policy_sarsa_values =  np.array([np.max(w[:,key]) for key in np.arange((al*al))]).reshape((al,al))
    # #dictionary for UTF-8 arrows
    # direction = {0:"\u2191",1:"\u2192",2:"\u2193",3:"\u2190"}
    # #argmaxs to use in direction dict
    # policy_sarsa_labels = np.array([direction[np.argmax(w[:,key])] for key in np.arange((al*al))]).reshape((al,al))

    # fig=plt.figure(figsize=(12,10))
    # ax = fig.add_subplot(111)
    # ax = sns.heatmap(policy_sarsa_values, annot = policy_sarsa_labels,cmap="magma",annot_kws={"size": 20}, fmt = '')
    # plt.title("Value Function Estimate using SARSA(λ) and softmax ASP",fontsize=18)
    # plt.show()