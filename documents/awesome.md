# awesome-medical-universal-model [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

Curated list of awesome medical universal models (UMs), or medical foundation models.

**Universal models** (UMs), trained on large and diverse datasets, are capable of performing a wide range of tasks with little or no need for task-specific data [[1](https://www.nature.com/articles/s41586-023-05881-4)]. Despite encountering various challenges, such as conflicting class definitions [[2](https://arxiv.org/abs/2301.00785), [3](https://arxiv.org/abs/2203.02098), [4](https://arxiv.org/abs/2303.14444)], partial labeling [[2](https://arxiv.org/abs/2301.00785), [4](https://arxiv.org/abs/2303.14444)], GPU limitations [[3](https://arxiv.org/abs/2203.02098)], among others during training, UMs are able to produce impressive outputs, such as *generalizability* (performing well on new data across different hospitals) [[2](https://arxiv.org/abs/2301.00785), [3](https://arxiv.org/abs/2203.02098)], *transferability* (serving as a powerful pre-training model for other tasks) [[2](https://arxiv.org/abs/2301.00785), [4](https://arxiv.org/abs/2303.14444), [5](https://arxiv.org/abs/2304.06716), [6](https://arxiv.org/abs/2304.03493)], *expertise* (assisting medical experts) [[1](https://www.nature.com/articles/s41586-023-05881-4), [2](https://arxiv.org/abs/2301.00785)], etc. 

Some segmentation datasets and models designed for a large number of structures can also be considered as foundation models. For example, totaSegmentator [[7](https://arxiv.org/abs/2208.05868)] dataset defines 104 whole-body structures for CT segmentation, UNEST [[8](https://arxiv.org/abs/2209.14378)] covers 133 brain tissues for MRI whole-brain segmentation. 

[1] Moor, Michael, et al. "Foundation models for generalist medical artificial intelligence." Nature 616.7956 (2023): 259-265. \
[2] Liu, Jie, et al. "CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection." arXiv preprint arXiv:2301.00785 (2023). \
[3] Liu, Pengbo, et al. "Universal segmentation of 33 anatomies." arXiv preprint arXiv:2203.02098 (2022). \
[4] Ulrich, Constantin, et al. "MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation." arXiv preprint arXiv:2303.14444 (2023). \
[5] Huang, Ziyan, et al. "STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training." arXiv preprint arXiv:2304.06716 (2023). \
[6] Ye, Yiwen, et al. "UniSeg: A Prompt-driven Universal Segmentation Model as well as A Strong Representation Learner." arXiv preprint arXiv:2304.03493 (2023). \
[7] Wasserthal, et al. "TotalSegmentator: robust segmentation of 104 anatomical structures in CT images." arXiv preprint arXiv:2208.05868. (2023). \
[8] Yu, X. et al. "UNesT: Local Spatial Representation Learning with Hierarchical Transformer for Efficient Medical Segmentation." arXiv preprint arXiv:2209.14378.(2023).

😎 This is an active repository and your contributions are always welcome! **Feel free **to submit **[issues](https://github.com/ljwztc/CLIP-Driven-Universal-Model/issues/10)** for related**work and dataset!** Don't forget to star and fork!

# Contents
- [awesome-medical-universal-model](#awesome-medical-universal-model)
- [Contents](#contents)
- [Papers](#papers)
  - [Perspectives](#perspectives)
  - [Vision](#vision)
  - [NLP](#nlp)
  - [Multi-modality](#multi-modality)
  - [In-Context Learning](#in-context-learning)
  - [Few-shot Learning](#few-shot-learning)
  - [Continual Learning](#continual-learning)
  - [Segment Anything Model (SAM) related](#segment-anything-model-sam-related)
  - [Parameter-efficient Fine Tuning](#parameter-efficient-fine-tuning)
  - [Learning Utility](#learning-utility)
- [Datasets](#datasets)
  - [Abdomen](#abdomen)
  - [Vertebrae](#vertebrae)
  - [Total Body](#total-body)


# Papers

## Perspectives

**Foundation models for generalist medical artificial intelligence** \
*PMichael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M. Krumholz, Jure Leskovec, Eric J. Topol, Pranav Rajpurkar* \
[Apr. 12, 2023] [Nature, 2023] \
[[Paper](https://www.nature.com/articles/s41586-023-05881-4)] 

**Toward Foundational Deep Learning Models for Medical Imaging in the New Era of Transformer Networks** \
*Martin J. Willemink, Holger R. Roth, Veit Sandfort* \
[Nov. 2, 2022] [Radiology: Artificial Intelligence, 2022] \
[[Paper](https://pubs.rsna.org/doi/10.1148/ryai.210284)] 

**Foundation Models in Healthcare: Opportunities, Biases and Regulatory Prospects in Europe** \
*Malwina Anna Wójcik* \
[Jul. 29, 2022] [EGOVIS, 2022] \
[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-12673-4_3#Sec7)] 


## Vision

**Universal Segmentation of 33 Anatomies** \
*Pengbo Liu, Yang Deng, Ce Wang, Yuan Hui, Qian Li, Jun Li, Shiwei Luo, Mengke Sun, Quan Quan, Shuxin Yang, You Hao, Honghu Xiao, Chunpeng Zhao, Xinbao Wu, S. Kevin Zhou* \
[Mar. 04, 2022] [arXiv, 2022] \
[[Paper](https://arxiv.org/abs/2203.02098)]

**CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection** \
*Jie Liu, Yixiao Zhang, Jie-Neng Chen, Junfei Xiao, Yongyi Lu, Bennett A. Landman, Yixuan Yuan, Alan Yuille, Yucheng Tang, Zongwei Zhou* \
[Jan. 02, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2301.00785)] [[Code](https://github.com/ljwztc/CLIP-Driven-Universal-Model)] [![GitHub stars](https://img.shields.io/github/stars/ljwztc/CLIP-Driven-Universal-Model.svg?logo=github&label=Stars)](https://github.com/ljwztc/CLIP-Driven-Universal-Model)

**MultiTalent: A Multi-Dataset Approach to Medical Image Segmentation** \
*Constantin Ulrich, Fabian Isensee, Tassilo Wald, Maximilian Zenk, Michael Baumgartner, Klaus H. Maier-Hein* \
[Mar. 25, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2303.14444)]

**UniSeg: A Prompt-driven Universal Segmentation Model as well as A Strong Representation Learner** \
*Yiwen Ye, Yutong Xie, Jianpeng Zhang, Ziyang Chen, Yong Xia* \
[Apr. 07, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.03493)] [[Code](https://github.com/yeerwen/UniSeg)] [![GitHub stars](https://img.shields.io/github/stars/yeerwen/UniSeg.svg?logo=github&label=Stars)](https://github.com/yeerwen/UniSeg)

**STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training** \
*Ziyan Huang, Haoyu Wang, Zhongying Deng, Jin Ye, Yanzhou Su, Hui Sun, Junjun He, Yun Gu, Lixu Gu, Shaoting Zhang, Yu Qiao* \
[Apr. 13, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.06716)] [[Code](https://github.com/Ziyan-Huang/STU-Net)] [![GitHub stars](https://img.shields.io/github/stars/Ziyan-Huang/STU-Net.svg?logo=github&label=Stars)](https://github.com/Ziyan-Huang/STU-Net)


## NLP

**Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction** \
*Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao & Degui Zhi* \
[May 20, 2021] [npj digital medicine , 2021] \
[[Paper](https://www.nature.com/articles/s41746-021-00455-y)]


## Multi-modality

**MedCLIP: Contrastive Learning from Unpaired Medical Images and Text** \
*Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, Jimeng Sun* \
[Oct. 18, 2022] [EMNLP, 2022] \
[[Paper](https://arxiv.org/abs/2210.10163)] [[Code](https://github.com/RyanWangZf/MedCLIP)] [![GitHub stars](https://img.shields.io/github/stars/RyanWangZf/MedCLIP.svg?logo=github&label=Stars)](https://github.com/RyanWangZf/MedCLIP)


## In-Context Learning

**UniverSeg: Universal Medical Image Segmentation** \
*Victor Ion Butoi, Jose Javier Gonzalez Ortiz, Tianyu Ma, Mert R. Sabuncu, John Guttag, Adrian V. Dalca* \
[Apr. 12, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.06131)] [[Code](https://github.com/JJGO/UniverSeg)] [![GitHub stars](https://img.shields.io/github/stars/JJGO/UniverSeg.svg?logo=github&label=Stars)](https://github.com/JJGO/UniverSeg)

## Few-shot Learning

**Transductive few-shot adapters for medical image segmentation** \
*Julio Silva-Rodríguez, Jose Dolz, Ismail Ben Ayed* \
[Mar. 29, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.03493)] [[Code](https://github.com/jusiro/fewshot-finetuning)] [![GitHub stars](https://img.shields.io/github/stars/jusiro/fewshot-finetuning.svg?logo=github&label=Stars)](https://github.com/jusiro/fewshot-finetuning)

## Continual Learning

**Towards General Purpose Medical AI: Continual Learning Medical Foundation Model** \
*Huahui Yi, Ziyuan Qin, Qicheng Lao, Wei Xu, Zekun Jiang, Dequan Wang, Shaoting Zhang, Kang Li* \
[Apr. 12, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2303.06580)]


## Parameter-efficient Fine Tuning

**Medical Image Understanding with Pretrained Vision Language Models: A Comprehensive Study** \
*Ziyuan Qin, Huahui Yi, Qicheng Lao, Kang Li* \
[Fed. 07, 2023] [ICLR, 2023] \
[[Paper](https://arxiv.org/abs/2209.15517)] [[Code](https://github.com/MembrLab/MIU-VL)] [![GitHub stars](https://img.shields.io/github/stars/MembrLab/MIU-VL.svg?logo=github&label=Stars)](https://github.com/MembrLab/MIU-VL)

**Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts** \
*Zhihong Chen, Shizhe Diao, Benyou Wang, Guanbin Li, Xiang Wan*\
[Feb. 08, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2302.08958)]

**Segment Anything in Medical Images** \
*Jun Ma, Bo Wang* \
[Apr. 24, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.12306)] [[Code](https://github.com/bowang-lab/MedSAM)] [![GitHub stars](https://img.shields.io/github/stars/bowang-lab/MedSAM.svg?logo=github&label=Stars)](https://github.com/bowang-lab/MedSAM)

**Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation** \
*Junde Wu, Rao Fu, Huihui Fang, Yuanpei Liu, Zhaowei Wang, Yanwu Xu, Yueming Jin, Tal Arbel* \
[Apr. 25, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.12620)] [[Code](https://github.com/WuJunde/Medical-SAM-Adapter)] [![GitHub stars](https://img.shields.io/github/stars/WuJunde/Medical-SAM-Adapter.svg?logo=github&label=Stars)](https://github.com/WuJunde/Medical-SAM-Adapter)

**Med-Tuning: Exploring Parameter-Efficient Transfer Learning for Medical Volumetric Segmentation** \
*Wenxuan Wang, Jiachen Shen, Chen Chen, Jianbo Jiao, Yan Zhang, Shanshan Song, Jiangyun Li* \
[Apr. 21, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.10880)]

**Learnable Ophthalmology SAM** \
*Zhongxi Qiu, Yan Hu, Heng Li, Jiang Liu*\
[Apr. 26, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.13425)] [[Code](https://github.com/Qsingle/LearnablePromptSAM)] [![GitHub stars](https://img.shields.io/github/stars/Qsingle/LearnablePromptSAM.svg?logo=github&label=Stars)](https://github.com/Qsingle/LearnablePromptSAM)

**Customized Segment Anything Model for Medical Image Segmentation** \
*Kaidong Zhang, Dong Liu*\
[Apr. 26, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.13785)] [[Code](https://github.com/hitachinsk/SAMed)] [![GitHub stars](https://img.shields.io/github/stars/hitachinsk/SAMed.svg?logo=github&label=Stars)](https://github.com/hitachinsk/SAMed)

## Segment Anything Model (SAM) related

**Segment Anything Model (SAM) for Digital Pathology: Assess Zero-shot Segmentation on Whole Slide Imaging** \
*Ruining Deng, Can Cui, Quan Liu, Tianyuan Yao, Lucas W. Remedios, Shunxing Bao, Bennett A. Landman, Lee E. Wheless, Lori A. Coburn, Keith T. Wilson, Yaohong Wang, Shilin Zhao, Agnes B. Fogo, Haichun Yang, Yucheng Tang, Yuankai Huo* \
[Apr. 9, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.04155)]

**SAM vs BET: A Comparative Study for Brain Extraction and Segmentation of Magnetic Resonance Images using Deep Learning** \
*Sovesh Mohapatra, Advait Gosai, Gottfried Schlaug* \
[Apr. 10, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.04738)]

**SAM.MD: Zero-shot medical image segmentation capabilities of the Segment Anything Model** \
*Saikat Roy, Tassilo Wald, Gregor Koehler, Maximilian R. Rokuss, Nico Disch, Julius Holzschuh, David Zimmerer, Klaus H. Maier-Hein* \
[Apr. 10, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.05396)]

**SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM** \
*Yihao Liu, Jiaming Zhang, Zhangcong She, Amir Kheradmand, Mehran Armand* \
[Apr. 12, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.05622)] [[Code](https://github.com/bingogome/samm)] [![GitHub stars](https://img.shields.io/github/stars/bingogome/samm.svg?logo=github&label=Stars)](https://github.com/bingogome/samm)

**SAM Struggles in Concealed Scenes -- Empirical Study on "Segment Anything"** \
*Ge-Peng Ji, Deng-Ping Fan, Peng Xu, Ming-Ming Cheng, Bowen Zhou, Luc Van Gool* \
[Apr. 12, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.06022)]

**Can SAM Segment Polyps?** \
*Tao Zhou, Yizhe Zhang, Yi Zhou, Ye Wu, Chen Gong* \
[Apr. 15, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.07583)]

**The Segment Anything foundation model achieves favorable brain tumor autosegmentation accuracy on MRI to support radiotherapy treatment planning** \
*Florian Putz, Johanna Grigo, Thomas Weissmann, Philipp Schubert, Daniel Hoefler, Ahmed Gomaa, Hassen Ben Tkhayat, Amr Hagag, Sebastian Lettmaier, Benjamin Frey, Udo S. Gaipl, Luitpold V. Distel, Sabine Semrau, Christoph Bert, Rainer Fietkau, Yixing Huang* \
[Apr. 16, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.07875)]

**When SAM Meets Medical Images: An Investigation of Segment Anything Model (SAM) on Multi-phase Liver Tumor Segmentation** \
*Chuanfei Hu, Xinde Li* \
[Apr. 17, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.08506)]

**Accuracy of Segment-Anything Model (SAM) in medical image segmentation tasks** \
*Sheng He, Rina Bao, Jingpeng Li, P. Ellen Grant, Yangming Ou* \
[Apr. 18, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.09324)]

**Segment Anything Model for Medical Image Analysis: an Experimental Study** \
*Maciej A. Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, Nicholas Konz, Yixin Zhang* \
[Apr. 20, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.10517)]

**Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model** \
*Yizhe Zhang, Tao Zhou, Peixian Liang, Danny Z. Chen* \
[Apr. 22, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.11332)] [[Code](https://github.com/yizhezhang2000/SAMAug)] [![GitHub stars](https://img.shields.io/github/stars/yizhezhang2000/SAMAug.svg?logo=github&label=Stars)](https://github.com/yizhezhang2000/SAMAug)

**Segment Anything in Medical Images** \
*Jun Ma, Bo Wang* \
[Apr. 24, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.12306)] [[Code](https://github.com/bowang-lab/MedSAM)] [![GitHub stars](https://img.shields.io/github/stars/bowang-lab/MedSAM.svg?logo=github&label=Stars)](https://github.com/bowang-lab/MedSAM)

**Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation** \
*Junde Wu, Rao Fu, Huihui Fang, Yuanpei Liu, Zhaowei Wang, Yanwu Xu, Yueming Jin, Tal Arbel* \
[Apr. 25, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.12620)] [[Code](https://github.com/WuJunde/Medical-SAM-Adapter)] [![GitHub stars](https://img.shields.io/github/stars/WuJunde/Medical-SAM-Adapter.svg?logo=github&label=Stars)](https://github.com/WuJunde/Medical-SAM-Adapter)

**Generalist Vision Foundation Models for Medical Imaging: A Case Study of Segment Anything Model on Zero-Shot Medical Segmentation** \
*Peilun Shi, Jianing Qiu, Sai Mu Dalike Abaxi, Hao Wei, Frank P.-W. Lo, Wu Yuan* \
[Apr. 25, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.12637)]

**GazeSAM: What You See is What You Segment** \
*Bin Wang, Armstrong Aboah, Zheyuan Zhang, Ulas Bagci* \
[Apr. 26, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.13844)] [[Code](https://github.com/ukaukaaaa/GazeSAM)] [![GitHub stars](https://img.shields.io/github/stars/ukaukaaaa/GazeSAM.svg?logo=github&label=Stars)](https://github.com/ukaukaaaa/GazeSAM)

**Customized Segment Anything Model for Medical Image Segmentation** \
*Kaidong Zhang, Dong Liu*\
[Apr. 26, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.13785)] [[Code](https://github.com/hitachinsk/SAMed)] [![GitHub stars](https://img.shields.io/github/stars/hitachinsk/SAMed.svg?logo=github&label=Stars)](https://github.com/hitachinsk/SAMed)

**SkinSAM: Empowering Skin Cancer Segmentation with Segment Anything Model** \
*Mingzhe Hu, Yuheng Li, Xiaofeng Yang* \
[Apr. 27, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.13973)]

**SAM Meets Robotic Surgery: An Empirical Study in Robustness Perspective** \
*An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, Hongliang Ren* \
[Apr. 27, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.14674)]

**Segment Anything Model for Medical Images?** \
*Yuhao Huang, Xin Yang, Lian Liu, Han Zhou, Ao Chang, Xinrui Zhou, Rusi Chen, Junxuan Yu, Jiongquan Chen, Chaoyu Chen, Haozhe Chi, Xindi Hu, Deng-Ping Fan, Fajin Dong, Dong Ni* \
[Apr. 28, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2304.14660)] 👍

**Exploring the Zero-Shot Capabilities of the Segment Anything Model (SAM) in 2D Medical Imaging: A Comprehensive Evaluation and Practical Guideline** \
*Christian Mattjie, Luis Vinicius de Moura, Rafaela Cappelari Ravazio, Lucas Silveira Kupssinskü, Otávio Parraga, Marcelo Mussi Delucis, Rodrigo Coelho Barros* \
[Apr. 28, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.00109)]

**SAM on Medical Images: A Comprehensive Study on Three Prompt Modes** \
*Dongjie Cheng, Ziyuan Qin, Zekun Jiang, Shaoting Zhang, Qicheng Lao, Kang Li* \
[Apr. 28, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.00035)]

**Polyp-SAM: Transfer SAM for Polyp Segmentation** \
*Yuheng Li, Mingzhe Hu, Xiaofeng Yang* \
[Apr. 29, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.00293)] [[Code](https://github.com/ricklisz/Polyp-SAM)] [![GitHub stars](https://img.shields.io/github/stars/ricklisz/Polyp-SAM.svg?logo=github&label=Stars)](https://github.com/ricklisz/Polyp-SAM)

**How Segment Anything Model (SAM) Boost Medical Image Segmentation?** \
*Yichi Zhang, Rushi Jiao* \
[May 05, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.03678)] [[Code](https://github.com/YichiZhang98/SAM4MIS)] [![GitHub stars](https://img.shields.io/github/stars/YichiZhang98/SAM4MIS.svg?logo=github&label=Stars)](https://github.com/YichiZhang98/SAM4MIS)

**BreastSAM: A Study of Segment Anything Model for Breast Tumor Detection in Ultrasound Images** \
*Mingzhe Hu, Yuheng Li, Xiaofeng Yang* \
[May 21, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2305.12447)]

**DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation** \
*Yifan Gao, Wei Xia, Dingdu Hu, Xin Gao* \
[Jun 1, 2023] [arXiv, 2023] \
[[Paper](https://arxiv.org/abs/2306.00499)] [[Code](https://github.com/yifangao112/DeSAM)] [![GitHub stars](https://img.shields.io/github/stars/yifangao112/DeSAM.svg?logo=github&label=Stars)](https://github.com/yifangao112/DeSAM)


## Learning Utility

**MONAI: An open-source framework for deep learning in healthcare** \
*M. Jorge Cardoso, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot, Yiheng Wang, Benjamin Murrey, Andriy Myronenko, Can Zhao, Dong Yang, Vishwesh Nath, Yufan He, Ziyue Xu, Ali Hatamizadeh, Andriy Myronenko, Wentao Zhu, Yun Liu, Mingxin Zheng, Yucheng Tang, Isaac Yang, Michael Zephyr, Behrooz Hashemian, Sachidanand Alle, Mohammad Zalbagi Darestani, Charlie Budd, Marc Modat, Tom Vercauteren, Guotai Wang, Yiwen Li, Yipeng Hu, Yunguan Fu, Benjamin Gorman, Hans Johnson, Brad Genereaux, Barbaros S. Erdal, Vikash Gupta, Andres Diaz-Pinto, Andre Dourson, Lena Maier-Hein, Paul F. Jaeger, Michael Baumgartner, Jayashree Kalpathy-Cramer, Mona Flores, Justin Kirby, Lee A.D. Cooper, Holger R. Roth, Daguang Xu, David Bericat, Ralf Floca, S. Kevin Zhou, Haris Shuaib, Keyvan Farahani, Klaus H. Maier-Hein, Stephen Aylward, Prerna Dogra, Sebastien Ourselin, Andrew Feng* \
[Nov. 04, 2022] [arXiv, 2022] \
[[Paper](https://arxiv.org/abs/2211.02701)] [[Code](https://github.com/Project-MONAI)]

# Datasets
## Abdomen
 - [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) [CT]
 - [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) [CT]
 - [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/) [CT/MRI]
 - [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details) [CT]
 - [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block) [CT]
 - [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) [CT]
 - [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD) [CT]
 - [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K) [CT]
 - [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org) [CT/MRI]
 - [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) [CT]
 - [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890) [CT]
 

## Vertebrae
 - [Large Scale Vertebrae Segmentation Challenge (VerSe)](https://github.com/anjany/verse) [CT]
 - [CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K) [[Paper](https://arxiv.org/abs/2105.14711)] [CT]
 
## Total Body
 - [Totalsegmentator](https://zenodo.org/record/6802614) [CT]
 


