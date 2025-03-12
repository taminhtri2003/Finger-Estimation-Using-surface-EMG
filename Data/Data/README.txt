Data: 10 subjects <downsampled_filterd_emg(dsfilt_emg),  23-joint marker position data(finger_kinematics)>

Data File Extension: .mat 
Cell Variables:
   	* dsfilt_emg, 		<5x7 cell> 	<40000x8>
   	* finger_kinematics	<5x7 cell>    	<4000x69>  


Cell format:
row - correspond to the number of trials (total 5 trials)
column - correspond to the number of tasks (total 7 tasks)

7 sets of movement tasks:
(1-5) individual flexion and extension of each finger: thumb, index, middle, ring, little, in this order
(6)   simulteneous flexion and extension of all fingers 
(7)   random free finger movement, in no particular order, and only in the flexion and extension plane


Comments:
a.  dsfilt_emg data and muscle activation data consist of 8 column vectors represents the 8 forearm muscles in the following order 
<APL,FCR,FDS,FDP,ED,EI,ECU,ECR>

b. The finger_kinematics data consists of 69 column vectors. Each column vector is either the <x,y,z> joint positions
   of each marker. There are a total of 23 markers used.The marker assignment has been shown in <Marker_Position.png>.
   


Please check our paper for the additional detail regarding the data. Please cite our paper if you wish to use this dataset.
- S.K.Dwivedi, J. Ngeo, T.Shibata, Transaction of Biomedical Engineering, In Press, 
"Extraction of Nonlinear Synergies for Proportional and Simultaneous Estimation of Finger Kinematics."



Affiliation / Email  --- Graduate School of Life Sciences and System Engineering, 
                         Kyushu Institute of Technology, Kitakyushu, Japan
 			 (correspondence: tom@brain.kyutech.ac.jp) 

