#inference on train data
#python inference.py --output_file /home/fatemanoor_gmail_com/test_dir/temp_model/predictions.csv -input_data_pattern=/mnt/disks/disk-1/data/youtube_video/feature_label.tfrecord  -train_dir=/home/fatemanoor_gmail_com/train_dir/temp_model -feature_names=audio_embedding -feature_sizes=128 -frame_features -model=FrameLevelLogisticModel  -num_classes=2 -batch_size=512 -top_k=2

python inference.py --output_file /home/fatemanoor_gmail_com/test_dir/temp_model/predictions_eval.csv -input_data_pattern=/mnt/disks/disk-1/data/youtube_video/feature_label_eval.tfrecord  -train_dir=/home/fatemanoor_gmail_com/train_dir/temp_model -feature_names=audio_embedding -feature_sizes=128 -frame_features -model=FrameLevelLogisticModel  -num_classes=2 -batch_size=512 -top_k=2





