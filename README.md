Cross-Age Face Recognition Using ResNet50 (unfinished)
====
Trainable ResNet50 using Python3.5 + Tensorflow <br>
DataSet: Cross-Age Celebrity Dataset[(CACD)](http://bcsiriuschen.github.io/CARC/)

Training Part
----
1. Run TrainResNet.py
2. Label and Image Name are loaded from "./label/label_1200.npy" and "./label/name_1200.npy"
3. Label is range from [1, LABELSNUM]
4. Set data_path to be None, if it is the frist time you Train. and set create_npy of load_all_image(nameList, h, w, c, parentPath, create_npy = False) to be True.
5. Set model_path to be None, if you train a network from scratch.
6. All trained model will be saved in ./model/XXX 

Extract Feature Part
----
1. Run TestResNet.py
2. Set data_path to be the model you use.
3. The feature will be saved as .mat
4. The "./label/label.npy" and "./label/name.npy" contain all 160,000+ images from 2000 identities.
5. LABELSNUM should be the same as training part, otherwise the Network cannot be correctly initialized.

