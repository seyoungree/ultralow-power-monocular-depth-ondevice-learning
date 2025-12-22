# Launching a pre-made training receipt

To launch a training receipt, run the following commands:
~~~
conda activate <env_name>
source <receipt_name>.sh
~~~

Every receipt does the following:
1) Switches to the folder of the selected experiment
2) Creates 5 folders to run 5 different training/fine-tuning
3) Runs the 5 training/fine-tuning routines with the respective parameters 
4) Tests the 5 trained models
5) Moves all the 5 runs into "ultralow-power-monocular-depth-ondevice-learning/<experiment_name>/", where you find the results into "run<run_number>/" subfolders (<run_number> goes from 0 to 4)

There, you find the accuracy of every run in every "disp_log.txt" file, in terms of the set of accuracy metrics used in the [Depth Map Prediction from a Single Image
using a Multi-Scale Deep Network](https://proceedings.neurips.cc/paper_files/paper/2014/file/91c56ce4a249fae5419b90cba831e303-Paper.pdf) paper. 
