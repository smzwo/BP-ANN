from tensorflow.python import pywrap_tensorflow

#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(r"D:\模式识别\2019\model\model.ckpt")

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

#最后，循环打印输出
for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))
