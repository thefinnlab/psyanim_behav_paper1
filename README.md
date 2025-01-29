scripts_data_for_beh_paper contains all the data analysis scripts used in this paper.

'stimulus_generation_and_presentation_bundled_scripts' contains the stimulus generation and presentation scripts developed using the psyanim-jsPsych plugin (https://github.com/thefinnlab/psyanim-2).

The subfolders 'detection', 'discrimination' and 'mixed-task-design' contain scripts for the respective studies. For details on what the folders and sub-folders mean, check out our pre-print: https://www.biorxiv.org/content/10.1101/2025.01.19.633772v1

Stimulus generation: Within each experiment folder, the subfolder 'stimulus_generation' contains scripts used to generate these psyanim stimuli. During stimulus creation, the coordinates corresponding to the animations were saved in Google Firebase, and were invoked from Firebase when presenting the stimuli as animations. For convenience, all stimuli are also saved as media (.webm) files in 'stimuli'. mixed-task-design presented stimuli from the detection and discrimination tasks in interleaved blocks and we didn't generate new stimuli for this study.

'experiments' contain the bundled scripts contain an executable experiment. You can place it on any server and run the index file inside the 'dist' folder. The actual experiment was scripted in the src/index.js file within each folder. Details of the experiments are given in the preprint linked above, esp. Table 1. Both session 1 and session 2 of the experiments were identical, only that in session 2, participants viewed previously unseen animations.  'stimSet_selection_for_retest.ipynb' is the script used to ensure that only unseen stimuli are selected. The output files are not provided since they have identifiable information (prolific IDs).

Most of the generation and experiments were built on an older version of psyanim (in Nov 2023), so to run these scripts, you should move the 'node_modules' folder to each experiment's folder.

The trait questionnaires used for each experiment are in the subfolder '<experiment_name>/src/surveys
