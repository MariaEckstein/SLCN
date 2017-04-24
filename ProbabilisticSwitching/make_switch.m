function make_switch(trial)

global exp

exp.make_switch = exp.PROBSWITCHdata.key(last_switch_trial) == exp.PROBSWITCHdata.key(trial);
