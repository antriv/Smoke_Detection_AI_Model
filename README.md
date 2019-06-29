# Microsoft_Smoke_Detection_AI_Model

Copyright (c) Microsoft Corporation. All rights reserved. 
Always credit "Anusua Trivedi, Microsoft AI for Good Team" if using this code.


## TO GET Tensorflow API:
1. Clone github: https://github.com/tensorflow/models
2. Use code here: https://github.com/tensorflow/models/tree/master/research/object_detection


## TO ADD TF API TO PATH:
export PYTHONPATH=$PYTHONPATH:/home/<user>/<base_dir>/models/research:/home/antriv/notebooks/models/research/slim

## TO TEST TF API: (run from the reseach folder)
python home/<user>/<base_dir>/models/research/object_detection/builders/model_builder_test.py

## TO PREPROCESS images/annotations:
1. python /home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/train_val_test_split.py
2. python /home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/generate_tfrecord.py --csv_input=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/train.csv --output_path=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/train.record --image_dir=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/PNGImages
3. python /home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/generate_tfrecord.py --csv_input=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/test.csv --output_path=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/test.record --image_dir=/home/<user>/<base_dir>/nvidia_smoke/data/smoke_annotations/PNGImages


## TO TRAIN USING TF API:
python /home/<user>/<base_dir>/models/research/object_detection/model_main.py \
    --pipeline_config_path=/home/<user>/<base_dir>/nvidia_smoke/ssd_mobilenet_v2_smoke/pipeline.config \
    --model_dir=/home/<user>/<base_dir>/nvidia_smoke/ssd_mobilenet_v2_smoke/smoke_model \
    --num_train_steps=10000 \
    --alsologtostderr

## TO FREEZE THE TRAINED TF MODEL:
python /home/<user>/<base_dir>/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /home/<user>/<base_dir>/nvidia_smoke/ssd_mobilenet_v2_smoke/pipeline.config --trained_checkpoint_prefix /home/<user>/<base_dir>/nvidia_smoke/ssd_mobilenet_v2_smoke/smoke_model/model.ckpt-10000 --output_directory /home/<user>/<base_dir>/nvidia_smoke/ssd_mobilenet_v2_smoke/smoke_model/frozen_graph

## TO TEST USING TF API:
python /home/<user>/<base_dir>/models/research/object_detection/nvidia_smoke_OD_inference.py
NOTE: You will need Xming server client to test locally
