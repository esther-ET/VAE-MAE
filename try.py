'''
======
Loading NuScenes tables for version v1.0-mini...
23 category,
8 attribute,
4 visibility,
911 instance,
12 sensor,
120 calibrated_sensor,
31206 ego_pose,
8 log,
10 scene,
404 sample,
31206 sample_data,
18538 sample_annotation,
4 map,
Done loading in 0.2 seconds.
======
Reverse indexing ...
Done reverse indexing in 0.0 seconds.
======
total scene num: 10
exist scene num: 10
v1.0-mini: train scene(8), val scene(2)

'''
import pickle
with open('/home/ubuntu/SWW/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_train.pkl','rb') as f:
    data = pickle.load(f)
print(f"total is {len(data)}") # 323
# /home/ubuntu/SWW/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl # 81
