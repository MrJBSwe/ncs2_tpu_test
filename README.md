
**NCS2 model conversion**


ssd_mobilenet_v2_coco_2018_03_29

[OpenVino convert from tensorflow](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) 
[richardstechnotes ssd mobilenet v2 ](https://richardstechnotes.com/2018/12/01/ssd_mobilenet_v2_coco-running-on-the-intel-neural-compute-stick-2/) 

**Note**, there is no INT8 support on MYRIAD.

```
$python3 /opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v2_coco_2018_03_29.pb --tensorflow_use_custom_operations_config ./opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer/extensions/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config  ssd_mobilenet_v2_coco_2018_03_29_pipeline.config --data_type FP16
```

Test converted model
```
$cd ~/inference_engine_samples_build/intel64/Release/
$./object_detection_demo_ssd_async  -m ~/ncs2_tpu_test/ssd_mobilenet_v2_coco_2018_03_29.xml  -i  path_video -d MYRIAD
```


**Model conversion for Coral TPU**<br>
[TensorFlow frozen graph to a TensorFlow lite](https://medium.com/@teyou21/convert-a-tensorflow-frozen-graph-to-a-tflite-file-part-3-1ccdb3874c4a)  

**Note,** Coral TPU supports only TensorFlow Lite models that are fully 8-bit quantized and then compiled specifically for the Edge TPU.


```
$tflite_convert --output_file=output_tflite_graph.tflite --graph_def_file=tflite_graph.pb  --inference_type=QUANTIZED_UINT8  --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --mean_values=128  --std_dev_values=128  --input_shapes=1,300,300,3 --change_concat_input_ranges=false  --allow_nudging_weights_to_use_fast_gemm_kernel=true --allow_custom_ops
$edgetpu_compiler output_tflite_graph.tflite 
```
