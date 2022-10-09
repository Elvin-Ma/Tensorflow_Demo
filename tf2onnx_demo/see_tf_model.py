import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

from tensorflow.python.platform import gfile

def print_pb(output_graph_path, meta_path):
    tf.reset_default_graph()  # 重置计算图
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        # 获得默认的图
        graph = tf.get_default_graph()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            # 得到当前图有几个操作节点
            print("%d ops in the final graph."% len(output_graph_def.node))

            tensor_name = [tensor.name for tensor in output_graph_def.node]
            print(tensor_name)
            print('---------------------------')
            # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
            summaryWriter = tf.summary.FileWriter('log_pb/', graph)

            for op in graph.get_operations():
                # print出tensor的name和值
                print(op.name, op.values())

        graphdef = graph.as_graph_def()

        _ = tf.train.import_meta_graph(meta_path)
        summary_write = tf.summary.FileWriter("./log_ck" , graph)

def print_ckpt(checkpoint_path):
    # 打印参数
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map=reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: ',key)

def check_data(path):
  with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(path,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
      graph_nodes=[n for n in graph_def.node]
  wts = [n for n in graph_nodes if n.op=='Const']
  from tensorflow.python.framework import tensor_util

  for n in wts:
      print("Name of the node - %s" % n.name)
      tensor_data=tensor_util.MakeNdarray(n.attr['value'].tensor)
      print("Value - ")
      # print(tensor_util.MakeNdarray(n.attr['value'].tensor))

if __name__ == "__main__":
    pb_path = "/home/mtn/models/20180408-102900/20180408-102900.pb"
    print_pb(pb_path, "/home/mtn/models/20180408-102900/model-20180408-102900.meta")
    # print_ckpt('/home/e00358/softwares/facenet/src/models/20180408-102900/model-20180408-102900.ckpt-90')
    # check_data(pb_path)
