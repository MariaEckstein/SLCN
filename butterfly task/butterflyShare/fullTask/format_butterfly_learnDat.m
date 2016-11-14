%manually input the output filenames from the 4 blocks in the order of their presentation
%concatenate them and make a new file with the 4 in one

%load in the first behav data from the first run
%rename all the varaibles that will need to be concatenated
%!!!run_info should be used from the first one only - rename it for the 
%first run for storing but not for the other runs

load s402_tb_l13_svlo-1_01Dec12_14-47.mat

category1=category;
delay1=delay;
delay_shown1=delay_shown;
new_negfb_image1=new_negfb_image;
new_posfb_image1=new_posfb_image;
old_negfb_image1=old_negfb_image;
old_posfb_image1=old_posfb_image;
optimal_corr1=optimal_corr;
outc_shown1=outc_shown;
outcome1=outcome;
resp1=resp;
rt1=rt;
run_info1=run_info;
shown_corr1=shown_corr;
stim1=stim;
stim_shown1=stim_shown;
stimtype1=stimtype;
stimtype_shown1=stimtype_shown;
trial_fb_image1=trial_fb_image;

%load in the second run and concatenate the data to the end of the files
%from the first run

load s402_tb_l14_svlo-1_01Dec12_14-54.mat

tmpblk=2*(ones(1,30));
category2=vertcat(category1,category);
delay2=vertcat(delay1,delay);
delay_shown2=horzcat(delay_shown1,delay_shown);
new_negfb_image2=horzcat(new_negfb_image1,new_negfb_image);
new_posfb_image2=horzcat(new_posfb_image1,new_posfb_image);
old_negfb_image2=horzcat(old_negfb_image1,old_negfb_image);
old_posfb_image2=horzcat(old_posfb_image1,old_posfb_image);
optimal_corr2=horzcat(optimal_corr1,optimal_corr);
outc_shown2=horzcat(outc_shown1,outc_shown);
outcome2=vertcat(outcome1,outcome);
resp2=horzcat(resp1,resp);
rt2=cat(2,rt1,rt);
shown_corr2=horzcat(shown_corr1,shown_corr);
stim2=vertcat(stim1,stim);
stim_shown2=horzcat(stim_shown1,stim_shown);
stimtype2=vertcat(stimtype1,stimtype);
stimtype_shown2=horzcat(stimtype_shown1,stimtype_shown);
trial_blk2=horzcat(trial_blk,tmpblk);
trial_fb_image2=horzcat(trial_fb_image1,trial_fb_image);

%load in the third run and concatenate the data to the end of the files
%from runs 1 & 2

load s402_tb_l11_svlo-1_01Dec12_15-02.mat

tmpblk=3*(ones(1,30));
category3=vertcat(category2,category);
delay3=vertcat(delay2,delay);
delay_shown3=horzcat(delay_shown2,delay_shown);
new_negfb_image3=horzcat(new_negfb_image2,new_negfb_image);
new_posfb_image3=horzcat(new_posfb_image2,new_posfb_image);
old_negfb_image3=horzcat(old_negfb_image2,old_negfb_image);
old_posfb_image3=horzcat(old_posfb_image2,old_posfb_image);
optimal_corr3=horzcat(optimal_corr2,optimal_corr);
outc_shown3=horzcat(outc_shown2,outc_shown);
outcome3=vertcat(outcome2,outcome);
resp3=horzcat(resp2,resp);
rt3=cat(2,rt2,rt);
shown_corr3=horzcat(shown_corr2,shown_corr);
stim3=vertcat(stim2,stim);
stim_shown3=horzcat(stim_shown2,stim_shown);
stimtype3=vertcat(stimtype2,stimtype);
stimtype_shown3=horzcat(stimtype_shown2,stimtype_shown);
trial_blk3=horzcat(trial_blk2,tmpblk);
trial_fb_image3=horzcat(trial_fb_image2,trial_fb_image);

%load in the fourth run and concatenate the data to the end of the files
%from runs 1,2, & 3

load s402_tb_l12_svlo-1_01Dec12_15-09.mat

tmpblk=4*(ones(1,30));
category4=vertcat(category3,category);
delay4=vertcat(delay3,delay);
delay_shown4=horzcat(delay_shown3,delay_shown);
new_negfb_image4=horzcat(new_negfb_image3,new_negfb_image);
new_posfb_image4=horzcat(new_posfb_image3,new_posfb_image);
old_negfb_image4=horzcat(old_negfb_image3,old_negfb_image);
old_posfb_image4=horzcat(old_posfb_image3,old_posfb_image);
optimal_corr4=horzcat(optimal_corr3,optimal_corr);
outc_shown4=horzcat(outc_shown3,outc_shown);
outcome4=vertcat(outcome3,outcome);
resp4=horzcat(resp3,resp);
rt4=cat(2,rt3,rt);
shown_corr4=horzcat(shown_corr3,shown_corr);
stim4=vertcat(stim3,stim);
stim_shown4=horzcat(stim_shown3,stim_shown);
stimtype4=vertcat(stimtype3,stimtype);
stimtype_shown4=horzcat(stimtype_shown3,stimtype_shown);
trial_blk4=horzcat(trial_blk3,tmpblk);
trial_fb_image4=horzcat(trial_fb_image3,trial_fb_image);

%save the 4-concatenated runs into the orignial variable names
category=category4;
delay=delay4;
delay_shown=delay_shown4;
new_negfb_image=new_negfb_image4;
new_posfb_image=new_posfb_image4;
old_negfb_image=old_negfb_image4;
old_posfb_image=old_posfb_image4;
optimal_corr=optimal_corr4;
outc_shown=outc_shown4;
outcome=outcome4;
resp=resp4;
rt=rt4;
run_info=run_info1;
shown_corr=shown_corr4;
stim=stim4;
stim_shown=stim_shown4;
stimtype=stimtype4;
stimtype_shown=stimtype_shown4;
trial_blk=trial_blk4;
trial_fb_image=trial_fb_image4;

%clean-up
clear *1
clear *2
clear *3
clear *4
clear tmp*

%saveit! put the order of the runs into the filename
save('s402_tb_l13312_svlo-1_01Dec12_14-47.mat');

clear all

%now do the same for the SM files for later SM test
	
load s402_SMlist_13svlo-1_01Dec12.mat
new_negfb_image1=new_negfb_image;
new_posfb_image1=new_posfb_image;
old_negfb_image1=old_negfb_image;
old_negfb_imagenum1=old_negfb_imagenum;
old_posfb_image1=old_posfb_image;
old_posfb_imagenum1=old_posfb_imagenum;
trial_blk1=trial_blk;

load s402_SMlist_14svlo-1_01Dec12.mat
tmpblk=2*(ones(1,30));
new_negfb_image2=horzcat(new_negfb_image1,new_negfb_image);
new_posfb_image2=horzcat(new_posfb_image1,new_posfb_image);
old_negfb_image2=horzcat(old_negfb_image1,old_negfb_image);
old_negfb_imagenum2=vertcat(old_negfb_imagenum1,old_negfb_imagenum);
old_posfb_image2=horzcat(old_posfb_image1,old_posfb_image);
old_posfb_imagenum2=vertcat(old_posfb_imagenum1,old_posfb_imagenum);
trial_blk2=horzcat(trial_blk1,tmpblk);

load s402_SMlist_11svlo-1_01Dec12.mat
tmpblk=3*(ones(1,30));
new_negfb_image3=horzcat(new_negfb_image2,new_negfb_image);
new_posfb_image3=horzcat(new_posfb_image2,new_posfb_image);
old_negfb_image3=horzcat(old_negfb_image2,old_negfb_image);
old_negfb_imagenum3=vertcat(old_negfb_imagenum2,old_negfb_imagenum);
old_posfb_image3=horzcat(old_posfb_image2,old_posfb_image);
old_posfb_imagenum3=vertcat(old_posfb_imagenum2,old_posfb_imagenum);
trial_blk3=horzcat(trial_blk2,tmpblk);

load s402_SMlist_12svlo-1_01Dec12.mat
tmpblk=4*(ones(1,30));
new_negfb_image4=horzcat(new_negfb_image3,new_negfb_image);
new_posfb_image4=horzcat(new_posfb_image3,new_posfb_image);
old_negfb_image4=horzcat(old_negfb_image3,old_negfb_image);
old_negfb_imagenum4=vertcat(old_negfb_imagenum3,old_negfb_imagenum);
old_posfb_image4=horzcat(old_posfb_image3,old_posfb_image);
old_posfb_imagenum4=vertcat(old_posfb_imagenum3,old_posfb_imagenum);
trial_blk4=horzcat(trial_blk3,tmpblk);

new_negfb_image=new_negfb_image4;
new_posfb_image=new_posfb_image4;
old_negfb_image=old_negfb_image4;
old_negfb_imagenum=old_negfb_imagenum4;
old_posfb_image=old_posfb_image4;
old_posfb_imagenum=old_posfb_imagenum4;
trial_blk=trial_blk4;

%clean-up
clear *1
clear *2
clear *3
clear *4
clear tmp*

%saveit! put the order of the runs into the filename
save('s402_SMlist_13412svlo-1_01Dec12.mat');

clear all