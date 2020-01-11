### DCIC 智能盘点—钢筋数量AI识别 学习使我快乐 队伍代码说明：
### 1. 运行环境需求：
    操作系统：Ubuntu16.10
    语言：python3.6.5
    深度学习框架：pytorch0.4.1 cuda9.0 cudnn7
    GPU：1080Ti 11G显存
    相关的依赖python包：
        pyyaml
        opencv-python
        pandas
        numpy
        scipy
        matplotlib
        cython
        packaging
        pycocotools 
        tensorboardX

### 2. 运行说明：
    a. 编译CUDA 代码：
        cd lib  # please change to this directory
        sh make.sh

        然后就会开始编译

        如果您使用的是Volta架构的GPU，那么请将 lib/make.sh 中第14行后面加上反斜杠，并且打开下一行的注释来调整cuda编译时候的依赖版本
        
        一切顺利的话将完成对NMS, ROI_Pooing, ROI_Crop 以及 ROI_Align几个模块的编译工作

    b.数据准备：
        请将比赛所用的训练集所有图片放置于目录:
            data/gangjin/images/train
        请将比赛所用的测试集放所有图片置于目录：
            data/gangjin/images/test
        请从以下链接下载COCO预训练模型：
        https://dl.fbaipublicfiles.com/detectron/36761843/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_1x.yaml.06_35_59.RZotkLKI/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
        并且将其重命名为：COCO-mask-X-101-32x8d.pkl
        然后放置于 data/pretrained_model 目录下
    
    c. 训练：
        python tools/train_net_step.py --dataset gangjin --cfg configs/baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml --nw 6 --bs 1 --use_tfboard
    d. 预测：
        python tools/infer_csv.py --dataset gangjin --cfg configs/baselines/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml --load_ckpt Outputs/e2e_faster_rcnn_X-101-32x8d-FPN_1x/Deb16-01-20-13_gpuNode-6-4_step/ckpt/model_step109999.pth --image_dir data/gangjin/images/test/ 
        然后最终的提交文件会在submit文件夹下生成

最后衷心感谢比赛放提供的非常良心的标注数据！！是我参加几个比赛看到的最棒最良心的，谢谢！！
