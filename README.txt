

SEE 'main' in ann.py for a start.

function 'robot_ann' performs one run for ntrials.
it returns:

    -reward received at the end of every trial
    -the total steps taken at the end of every trial
    -the optimal steps for each trial 
    -the weight matrix 


the main function currently will do 100 repitions, with 400 trials each.
It stores these values in total_rewards,total_steps,optimum_steps,weightsS etc.

some notes:

    -the parameter "extrawalls" will not run unless the area length is 10
    -the parameter value for 'epsilon' does not matter if soft=True as soft enables softmax 
        as the activation function.
    

    -to disable eligibility trace and run basic SARSA, set the parameter 'e_trace' to None.
        -any value other than None will be used in SARSA(Lambda) 


As an example:

    -my heatmap code is provided in main but is commented out initially, just as an example of
    reproducing figures.