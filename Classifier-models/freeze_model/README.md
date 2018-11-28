# How to freeze the model

In order for the trainned model to run on carla, it needs to run on tensorflow 1.3. Fortunately, we can train using any tf version and we only need to use tf version 1.3 to create a saved model. 

This folder illustrates the steps that are required to achieve that. Assuming you have anaconda installed on your system, you can follow the steps below. 

```
git clone https://github.com/tensorflow/models.git
cd models/research
# In the models/research folder
conda create --name tf13 --file requirements.txt
source activate tf13

# below is the version that works with tf1.3
git checkout master
git checkout f87a58cd

python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/yourconfig.config --trained_checkpoint_prefix ./ckpt_dir/model.ckpt-<iternum> --output_directory output_dir

# In the above command, replace the config,  chkpt_dir, iternum and output_dir
# to appropriate values; For example, see below

# python object_detection/export_inference_graph.py --input_type image_tensor 
# --pipeline_config_path ./config/ssd_sim_5k.config --trained_checkpoint_prefix
# ./output_5k/model.ckpt-5000 --output_directory output_5k

# Finally, exit the anaconda environment
git checkout master
source deactivate

```

