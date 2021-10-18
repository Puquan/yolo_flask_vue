from easydict import EasyDict as edict
# 调用配置文件使用from config import cfg
__C                          = edict()
init                         = __C
__C.YOLO                     = edict()

# xml文件提取相关
__C.XML                      = edict()
__C.XML.LABELS               = ['football','basketball','donut','orange_juice','potato_chip']
__C.XML.VP                   = 0.2                                     #设置数据集划分比例
__C.XML.INPUT_DIR            = './data/supermarket/img'                #输入图片文件夹位置
__C.XML.XML_DIR              = './data/supermarket/outputs'            #xml文件夹位置
__C.XML.TRAIN_TXT            = './data/train.txt'                      #训练数据清单生成的位置
__C.XML.TEST_TXT             = './data/test.txt'                       #测试数据清单生成的位置

# 配置训练全局参数
__C.TRAIN                    = edict()
__C.TRAIN.NAME_PATH          = "./data/supermarket.name"  #标签配置文件
__C.TRAIN.NUM_CLASSES        = 5                                       #标签类别数
__C.TRAIN.TRAIN_ANNO_PATH    = "./data/train.txt"                      #加载训练数据清单
__C.TRAIN.TEST_ANNO_PATH     = "./data/test.txt"                       #加载测试数据清单
__C.TRAIN.YAML_PATH          = "./cfgs/yolov3_tiny.yaml"  #模型网络参数配置文件
__C.TRAIN.LEARN_RATE_INIT    = 1e-4                                    #初始学习率
__C.TRAIN.LEARN_RATE_END     = 1e-6                                    #最终学习率
__C.TRAIN.WARMUP_LEARN_RATE  = 1e-4                                    #预热最大学习率
__C.TRAIN.WARMUP_EPOCHS      = 1                                       #模型预热训练次数
__C.TRAIN.EPOCHS             = 20                                      #模型最大训练次数
__C.TRAIN.BATCH_SIZE         = 2                                       #设置批处理图片数量
__C.TRAIN.SAVE_NAME          = "mAP-{mAP:.4f}.h5"                      #模型权重文件保存命名格式
__C.TRAIN.SAVE_WEIGHT_PATH   = "./ckpts"                               #模型权重文件保存路径

# 预训练权重
__C.TRAIN.INIT_WEIGHT_PATH   = "./yolov3_tiny_mAP-0.7465.h5"  #载入预训练权重

# 模型评估参数
__C.EVAL                     = edict()
__C.EVAL.TEST_WEIGHT_PATH    = "./yolov3_tiny_mAP-0.7465.h5"  #用于评估权重地址，同时也是后续模型部署所使用的权重
__C.EVAL.IMG_PATH            = './data/supermarket/img'                #用于评估的图片/文件夹/视频地址
__C.EVAL.SAVE_PATH           = './data/supermarket/detection'          #预测结果保存位置


