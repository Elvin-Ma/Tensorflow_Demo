
# 1. save tensorflow model for tensorboard https://www.cnblogs.com/ywheunji/p/12092115.html
# 2. strip training model https://blog.csdn.net/qq_17721239/article/details/89296911
# 3. freeze tensorflow model  https://blog.csdn.net/qq_17721239/article/details/89295840
# cd ../../src; python freeze_graph.py models/20180402-114759 frozen.pb
# 4. convert checkpoint to onnx 
# python -m tf2onnx.convert --checkpoint model_inference/facenet_stripped.ckpt.meta --output facenet_stripped_freezen.onnx --inputs input:0[1,160,160,3] --outputs InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1:0

# keras facent
# https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

import sys, os
from unicodedata import name
# from turtle import shape

# 打印pb模型参数及可视化结构
import tensorflow as tf
# from tensorflow.python.framework import graph_util


###### 
# import facenet network
# export PYTHONPATH=$PYTHONPATH:/home/gyf/pkg/xxgg/github/ai_app/cmcc/facenet
import inception_resnet_v1
import numpy as np
#import renovate
#from simplifier import Optimizer
import onnx

def save_for_tensorboard(input_graph):
    tf.reset_default_graph()  # 重置计算图

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()

        # 获得默认的图
        graph = tf.get_default_graph()
        with open(input_graph, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # 得到当前图有几个操作节点
            print("%d ops in the final graph." % len(output_graph_def.node))

            tensor_name = [tensor.name for tensor in output_graph_def.node]
            print(tensor_name)
            print('---------------------------')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            summaryWriter = tf.summary.FileWriter('log_pb/', graph)

            for op in graph.get_operations():
                # print出tensor的name和值
                print(op.name, op.values())

# input_graph = sys.argv[1]

def strip_model(training_ckpt, stripped_model):

    input_data = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 160, 160, 3])
    output, _ = inception_resnet_v1.inference(input_data, keep_probability=0.8, phase_train=False, bottleneck_layer_size=512)

    # label_batch = tf.identity(output, name = 'label_batch')
    embeddings = tf.identity(output, name = 'embeddings')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, training_ckpt)
        saved_path = saver.save(sess, stripped_model)
        print("Striped model saved in file: %s" % (saved_path))

# training_ckpt = sys.argv[1]
# stripped_model = sys.argv[2]
# strip_model(training_ckpt, stripped_model)

if __name__ == '__main__':
    src_ckpt_path = "/home/mtn/models/20180408-102900/model-20180408-102900.ckpt-90"
    strip_model(src_ckpt_path, 'facenet_striped/facenet.ckpt')
    # pip install -U tf2onnx
    os.system("python -m tf2onnx.convert --checkpoint facenet_striped/facenet.ckpt.meta --output facenet_stripped_freezen.onnx --inputs input:0[2,160,160,3] --outputs InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1:0")
    
    # onnx.utils.extract_model('facenet_stripped_freezen.onnx', 'facenet_fixed.onnx', ['input:0'], ['InceptionResnetV1/Bottleneck/BatchNorm/FusedBatchNorm:0'])
    
    #optimizer = Optimizer('facenet_fixed.onnx', 'facenet_ts_new.onnx')
    #optimizer.delete_node("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool__346")
    #optimizer.delete_node("InceptionResnetV1/Bottleneck/BatchNorm/Reshape")
    #optimizer.delete_node("InceptionResnetV1/Bottleneck/BatchNorm/FusedBatchNorm__350")
    #onnx.save(optimizer.onnx_model, optimizer.output_path)

    #input_data = np.random.rand(2,160,160,3).astype(np.float32)
    #facenet = renovate.Renovate(optimizer.output_path)
    #facenet.mma_bn()
    #facenet.conv_mul()
    #
    #d = facenet.check(input_data)
    #print("max diff is %f"%d)
    #inferred_onnx_model = onnx.shape_inference.infer_shapes(facenet.new_model)
    #onnx.save(inferred_onnx_model, "./facenet_tf_cb.onnx")
    #print("finish")


#dump train
    # os.system('python ../../freeze_graph.py ../20180408-102900 frozen.pb')
    # os.system("python -m tf2onnx.convert --input frozen.pb --output facenet_train_freezen.onnx --inputs input:0[2,160,160,3] --outputs InceptionResnetV1/Bottleneck/BatchNorm/Reshape_1:0")
