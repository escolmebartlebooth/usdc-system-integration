# Instructions to train a model

In order to identify the traffic lights and to classify the state of the traffic lights, we will use tensorflow object detection system. (https://github.com/tensorflow/models)

While we can use any version of tensorflow to train, we **have** to use the version of the tensorflow 1.3 available in carla to freeze the trained model for inference. Thankfully, the freezing of the models is forward compatible. Hence, we used tensorflow 1.12 to train the model and used froze using tensorflow 1.3. The instructions to freeze the model and the corresponding environment are in freeze_model directory.

A lot of this work has been inspired from https://github.com/alex-lechner/Traffic-Light-Classification . We succintly summarize the relevant parts below, but curious reader can go to the source to get detailed information.  

# TF.Record creation
The tensorflow object detector takes input as record format as specified [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md). However, we were able to obtain pre-collected data from link provided above. The data was collected by [@alex-lechner](https://github.com/alex-lechner/Traffic-Light-Classification) & [@coldKnight](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset) 

We split the data for training & validation based on the user.
    
- sim_data.record  (Simulator Train)
- jpg_simulator_train.record (Simulator Test)
- real_data.record (Udacity Train)
- jpg_udacity_train.record (Udacity Test)





# Transfer Learning Starting Point

We want to do transfer learning to reduce the training time. Google helpfully provides several pretrained models on various datasets [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 

We looked at various options and ssd_inception_v2 trained on coco (ssd_inception_v2_coco_2018_01_28) provided a good starting point. We choose this because of the latency vs performance tradeoff presented in the table in that page. Furthermore, COCO had a class called traffic-light and it made sense to start with a model trained on that.


# Config file modification
To finetune the model, we start with the sample config file ```object_detection/samples/config/ssd_inception_v2_coco.config``` and make the following changes.

- Set the fine tuning starting point and enable fine tuning:
```
fine_tune_checkpoint: "ssd_inception_v2_coco_2018_01_28/model.ckpt"
from_detection_checkpoint: true
```
- set the number of classes to be detected to be 4: 
``` num_classes: 4```
- Change the number of steps to train. We arrived at 30k to be a good number: ```num_steps: 30000 ```
- we experimented with various data augmentation, but horizontal_flip and random_crop worked best.
- change the input path and label_map path. (for simulator data, we use sim_data, for real camera data we use real_data)
```
train_input_reader: {
tf_record_input_reader {
input_path: "sim_data.record"
}
label_map_path: "udacity_label_map.pbtxt"
}
```
- 


#  Detailed instructions for training

Download pretrainned data  from  
``` 
# checkout tensorflow models repository
git clone https://github.com/tensorflow/models.git

# checkout a known good working version as per @alex-lechner
git checkout f7e99c08B 

cd models/research
```

Follow the instructions for installation as mentioned [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) according to your platform to complete the instructions. 

```

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz

unzip ssd_inception_v2_coco_2018_01_28.tar.gz 

# place the config files from the config folder into this location.
python object_detection/train.py --pipeline_config_path=config/ssd_sim_30k.config --train_dir=sim_model_dir --num_clones=2 --ps_tasks=1

# If you only have a single GPU, ignore --num_clones and --ps_tasks options.


```

# Evaluation & Visualization

Once we have trained the model, we can use tensorboard to visualize the training process and eval results.
When the model is training, we can do the following to track training progress

```tensorboard --logdir=sim_model_dir```

After the training is done, we can evaluate the training model using the command below. 

```python object_detection/eval.py --pipeline_config_path=config/ssd_sim_30k.config --checkpoint_dir=sim_model_dir --eval_dir=eval/```

We also have a Jupyter notebook with the results of the trained model eval for your perusal.

# Freezing the model for inference
See freeze_model folder for details. 
