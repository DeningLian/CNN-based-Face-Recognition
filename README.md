# CNN-based Face Recognition
基于神经网络的人脸识别  

[展示时所用的 slides](https://docs.google.com/presentation/d/1WEyMGs2Ce05tGxz-uCoDWdpiwK7Yj4j0tCgpqLK3sPc/edit?usp=sharing)

### 运行指南  
1. 切换工作目录到 src/ 路径下  
2. python cnn.py  

### 参数修改指南  
涉及到需要修改的参数只在 cnn.py 文件下

1. `img_width = 16` 可能要改成更大一些的值  
2. `cnn.train(is_summary=False, summary_file='../tmp/test/' + dataset[0], epoch=5, is_load=False)` 这一行中的 `epoch` 为训练代树，需要调节；`is_load` 表示是否加载模型，根据情况修改。

在模型训练完毕以后，模型会保存到 `../models/` 目录下，加载时默认加载该目录下的模型。