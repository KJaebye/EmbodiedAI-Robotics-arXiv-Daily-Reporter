# Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos 

**Title (ZH)**: 基于粒子-网格神经动力学的可变形物体模型学习方法（从RGB-D视频中学习） 

**Authors**: Kaifeng Zhang, Baoyu Li, Kris Hauser, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.15680)  

**Abstract**: Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at this https URL . 

**Abstract (ZH)**: 基于粒子-网格的动态建模框架：从稀疏视图RGB-D记录中学习变形物体的动力学并生成3D条件动作视频 

---
# MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System 

**Title (ZH)**: 多目全景对象SLAM系统 

**Authors**: Miaoxin Pan, Jinnan Li, Yaowen Zhang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2506.15402)  

**Abstract**: Object-level SLAM offers structured and semantically meaningful environment representations, making it more interpretable and suitable for high-level robotic tasks. However, most existing approaches rely on RGB-D sensors or monocular views, which suffer from narrow fields of view, occlusion sensitivity, and limited depth perception-especially in large-scale or outdoor environments. These limitations often restrict the system to observing only partial views of objects from limited perspectives, leading to inaccurate object modeling and unreliable data association. In this work, we propose MCOO-SLAM, a novel Multi-Camera Omnidirectional Object SLAM system that fully leverages surround-view camera configurations to achieve robust, consistent, and semantically enriched mapping in complex outdoor scenarios. Our approach integrates point features and object-level landmarks enhanced with open-vocabulary semantics. A semantic-geometric-temporal fusion strategy is introduced for robust object association across multiple views, leading to improved consistency and accurate object modeling, and an omnidirectional loop closure module is designed to enable viewpoint-invariant place recognition using scene-level descriptors. Furthermore, the constructed map is abstracted into a hierarchical 3D scene graph to support downstream reasoning tasks. Extensive experiments in real-world demonstrate that MCOO-SLAM achieves accurate localization and scalable object-level mapping with improved robustness to occlusion, pose variation, and environmental complexity. 

**Abstract (ZH)**: 基于多摄像头全景对象的SLAM系统：在复杂户外场景中实现稳健、一致和语义丰富的映射 

---
# RaCalNet: Radar Calibration Network for Sparse-Supervised Metric Depth Estimation 

**Title (ZH)**: RaCalNet：稀疏监督度量深度估计的雷达校准网络 

**Authors**: Xingrui Qin, Wentao Zhao, Chuan Cao, Yihe Niu, Houcheng Jiang, Jingchuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15560)  

**Abstract**: Dense metric depth estimation using millimeter-wave radar typically requires dense LiDAR supervision, generated via multi-frame projection and interpolation, to guide the learning of accurate depth from sparse radar measurements and RGB images. However, this paradigm is both costly and data-intensive. To address this, we propose RaCalNet, a novel framework that eliminates the need for dense supervision by using sparse LiDAR to supervise the learning of refined radar measurements, resulting in a supervision density of merely around 1% compared to dense-supervised methods. Unlike previous approaches that associate radar points with broad image regions and rely heavily on dense labels, RaCalNet first recalibrates and refines sparse radar points to construct accurate depth priors. These priors then serve as reliable anchors to guide monocular depth prediction, enabling metric-scale estimation without resorting to dense supervision. This design improves structural consistency and preserves fine details. Despite relying solely on sparse supervision, RaCalNet surpasses state-of-the-art dense-supervised methods, producing depth maps with clear object contours and fine-grained textures. Extensive experiments on the ZJU-4DRadarCam dataset and real-world deployment scenarios demonstrate its effectiveness, reducing RMSE by 35.30% and 34.89%, respectively. 

**Abstract (ZH)**: 基于毫米波雷达的密集度量深度估计通常需要通过多帧投影和插值生成的密集激光雷达监督，以指导从稀疏雷达测量和RGB图像中学习准确深度。为了解决这一问题，我们提出了一种新的框架RaCalNet，通过使用稀疏激光雷达监督稀疏雷达测量的学习，监督密度仅为密集监督方法的约1%。不同于以往方法将雷达点与广泛的图像区域关联并依赖密集标签，RaCalNet首先校准和细化稀疏雷达点以构建准确的深度先验，这些先验作为可靠的锚点引导单目深度预测，从而在无需密集监督的情况下实现度量级估计。这种设计增强了结构一致性并保留了精细细节。尽管仅依赖稀疏监督，RaCalNet仍超越了最新的密集监督方法，生成清晰的物体轮廓和细腻的纹理图。在ZJU-4DRadarCam数据集和实际部署场景中的广泛实验表明其有效性，分别将RMSE降低了35.30%和34.89%。 

---
# One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution 

**Title (ZH)**: 一步扩散生成细节丰富且时序连贯的视频超分辨率 

**Authors**: Yujing Sun, Lingchen Sun, Shuaizheng Liu, Rongyuan Wu, Zhengqiang Zhang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15591)  

**Abstract**: It is a challenging problem to reproduce rich spatial details while maintaining temporal consistency in real-world video super-resolution (Real-VSR), especially when we leverage pre-trained generative models such as stable diffusion (SD) for realistic details synthesis. Existing SD-based Real-VSR methods often compromise spatial details for temporal coherence, resulting in suboptimal visual quality. We argue that the key lies in how to effectively extract the degradation-robust temporal consistency priors from the low-quality (LQ) input video and enhance the video details while maintaining the extracted consistency priors. To achieve this, we propose a Dual LoRA Learning (DLoRAL) paradigm to train an effective SD-based one-step diffusion model, achieving realistic frame details and temporal consistency simultaneously. Specifically, we introduce a Cross-Frame Retrieval (CFR) module to aggregate complementary information across frames, and train a Consistency-LoRA (C-LoRA) to learn robust temporal representations from degraded inputs. After consistency learning, we fix the CFR and C-LoRA modules and train a Detail-LoRA (D-LoRA) to enhance spatial details while aligning with the temporal space defined by C-LoRA to keep temporal coherence. The two phases alternate iteratively for optimization, collaboratively delivering consistent and detail-rich outputs. During inference, the two LoRA branches are merged into the SD model, allowing efficient and high-quality video restoration in a single diffusion step. Experiments show that DLoRAL achieves strong performance in both accuracy and speed. Code and models are available at this https URL. 

**Abstract (ZH)**: 实时视频超分辨率中保留丰富空间细节和时间连贯性的再现是一个具有挑战性的问题，特别是在利用稳定扩散（SD）等预训练生成模型合成逼真细节时。现有的基于SD的实时视频超分辨率方法往往在空间细节和时间一致性之间权衡，导致视觉质量不佳。我们argue认为关键在于如何有效提取低质量输入视频中的去噪鲁棒时间一致性先验，并在保留提取的一致性先验的同时增强视频细节。为此，我们提出了一种双LoRA学习（DLoRAL）范式，训练一个有效的基于SD的一步扩散模型，同时实现逼真帧细节和时间一致性。具体来说，我们引入了一个跨帧检索（CFR）模块来聚合跨帧的互补信息，并训练一个一致性LoRA（C-LoRA）来从降质输入中学习鲁棒的时间表示。在一致性学习之后，我们固定CFR和C-LoRA模块，并训练一个细节LoRA（D-LoRA）来增强空间细节，同时与C-LoRA定义的时间空间对齐以保持时间连贯性。两个阶段交替迭代优化，协作生成一致性和细节丰富的输出。在推理过程中，两个LoRA分支合并到SD模型中，允许在单一扩散步骤中高效地实现高质量的视频恢复。实验表明，DLoRAL在准确性和速度上均表现出色。代码和模型可在以下链接获取。 

---
# CLAIM: Clinically-Guided LGE Augmentation for Realistic and Diverse Myocardial Scar Synthesis and Segmentation 

**Title (ZH)**: 临床指导的LGE增强以实现真实和多样的心肌疤痕合成与分割 

**Authors**: Farheen Ramzan, Yusuf Kiberu, Nikesh Jathanna, Shahnaz Jamil-Copley, Richard H. Clayton, Chen, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15549)  

**Abstract**: Deep learning-based myocardial scar segmentation from late gadolinium enhancement (LGE) cardiac MRI has shown great potential for accurate and timely diagnosis and treatment planning for structural cardiac diseases. However, the limited availability and variability of LGE images with high-quality scar labels restrict the development of robust segmentation models. To address this, we introduce CLAIM: \textbf{C}linically-Guided \textbf{L}GE \textbf{A}ugmentation for Real\textbf{i}stic and Diverse \textbf{M}yocardial Scar Synthesis and Segmentation framework, a framework for anatomically grounded scar generation and segmentation. At its core is the SMILE module (Scar Mask generation guided by cLinical knowledgE), which conditions a diffusion-based generator on the clinically adopted AHA 17-segment model to synthesize images with anatomically consistent and spatially diverse scar patterns. In addition, CLAIM employs a joint training strategy in which the scar segmentation network is optimized alongside the generator, aiming to enhance both the realism of synthesized scars and the accuracy of the scar segmentation performance. Experimental results show that CLAIM produces anatomically coherent scar patterns and achieves higher Dice similarity with real scar distributions compared to baseline models. Our approach enables controllable and realistic myocardial scar synthesis and has demonstrated utility for downstream medical imaging task. 

**Abstract (ZH)**: 基于深度学习的晚期钆增强（LGE）心脏MRI心肌疤痕分割在结构性心脏疾病的准确和及时诊断及治疗规划中展现了巨大的潜力。然而，高质量疤痕标签的LGE图像的有限可用性和变异性限制了稳健分割模型的发展。为了解决这一问题，我们引入了CLAIM：基于临床指导的LGE增强现实和多样心肌疤痕合成与分割框架，一种基于解剖学的心肌疤痕生成和分割框架。其核心是SMILE模块（由临床知识指导的心肌疤痕掩模生成），该模块以临床采用的AHA 17段模型为条件，以生成具有解剖一致性且空间多样性的心肌疤痕图样。此外，CLAIM采用了一种联合训练策略，使疤痕分割网络与生成器同步优化，旨在提高合成疤痕的现实性和疤痕分割性能的准确性。实验结果表明，CLAIM生成了解剖学上连贯的疤痕图样，并在与真实疤痕分布的Dice相似度方面超过了基线模型。我们的方法实现了可控且现实的心肌疤痕合成，并在下游医疗影像任务中展现了应用价值。 

---
# GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects 

**Title (ZH)**: GenHOI: 依赖文本泛化生成未知对象的4D人体-对象交互合成 

**Authors**: Shujia Li, Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2506.15483)  

**Abstract**: While diffusion models and large-scale motion datasets have advanced text-driven human motion synthesis, extending these advances to 4D human-object interaction (HOI) remains challenging, mainly due to the limited availability of large-scale 4D HOI datasets. In our study, we introduce GenHOI, a novel two-stage framework aimed at achieving two key objectives: 1) generalization to unseen objects and 2) the synthesis of high-fidelity 4D HOI sequences. In the initial stage of our framework, we employ an Object-AnchorNet to reconstruct sparse 3D HOI keyframes for unseen objects, learning solely from 3D HOI datasets, thereby mitigating the dependence on large-scale 4D HOI datasets. Subsequently, we introduce a Contact-Aware Diffusion Model (ContactDM) in the second stage to seamlessly interpolate sparse 3D HOI keyframes into densely temporally coherent 4D HOI sequences. To enhance the quality of generated 4D HOI sequences, we propose a novel Contact-Aware Encoder within ContactDM to extract human-object contact patterns and a novel Contact-Aware HOI Attention to effectively integrate the contact signals into diffusion models. Experimental results show that we achieve state-of-the-art results on the publicly available OMOMO and 3D-FUTURE datasets, demonstrating strong generalization abilities to unseen objects, while enabling high-fidelity 4D HOI generation. 

**Abstract (ZH)**: GenHOI：针对未见物体的高保真4D人体-物体交互合成 

---
# Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material 

**Title (ZH)**: 混沌3D 2.1：从图像到具有生产就绪PBR材质的高保真3D资产 

**Authors**: Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, Qingxiang Lin, Zeqiang Lai, Xianghui Yang, Huiwen Shi, Zibo Zhao, Bowen Zhang, Hongyu Yan, Lifu Wang, Sicong Liu, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Yulin Cai, Jiaao Yu, Yixuan Tang, Dongyuan Guo, Junlin Yu, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Shida Wei, Chao Zhang, Yonghao Tan, Yifu Sun, Lin Niu, Shirui Huang, Bojian Zheng, Shu Liu, Shilin Chen, Xiang Yuan, Xiaofeng Yang, Kai Liu, Jianchen Zhu, Peng Chen, Tian Liu, Di Wang, Yuhong Liu, Linus, Jie Jiang, Jingwei Huang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15442)  

**Abstract**: 3D AI-generated content (AIGC) is a passionate field that has significantly accelerated the creation of 3D models in gaming, film, and design. Despite the development of several groundbreaking models that have revolutionized 3D generation, the field remains largely accessible only to researchers, developers, and designers due to the complexities involved in collecting, processing, and training 3D models. To address these challenges, we introduce Hunyuan3D 2.1 as a case study in this tutorial. This tutorial offers a comprehensive, step-by-step guide on processing 3D data, training a 3D generative model, and evaluating its performance using Hunyuan3D 2.1, an advanced system for producing high-resolution, textured 3D assets. The system comprises two core components: the Hunyuan3D-DiT for shape generation and the Hunyuan3D-Paint for texture synthesis. We will explore the entire workflow, including data preparation, model architecture, training strategies, evaluation metrics, and deployment. By the conclusion of this tutorial, you will have the knowledge to finetune or develop a robust 3D generative model suitable for applications in gaming, virtual reality, and industrial design. 

**Abstract (ZH)**: 3D AI生成内容（AIGC）是一个充满激情的领域，极大地加速了游戏、电影和设计中的3D模型创建。尽管已经开发出多种革命性的模型来重塑3D生成，但由于收集、处理和训练3D模型的复杂性，该领域仍主要 доступ仅限于研究人员、开发人员和设计师。为应对这些挑战，本文通过Hunyuan3D 2.1作为案例研究介绍了这一教程。本教程提供了一个全面的、分步骤的指南，用于处理3D数据、训练3D生成模型以及使用Hunyuan3D 2.1（一个生产高分辨率、带纹理的3D资产的先进系统）评估其性能。该系统包括两个核心组件：Hunyuan3D-DiT用于形状生成和Hunyuan3D-Paint用于纹理合成。我们将探索整个工作流程，包括数据准备、模型架构、训练策略、评估指标和部署。通过完成本教程，您将具备调整或开发适用于游戏、虚拟现实和工业设计应用的稳健3D生成模型的知识。 

---
# A Real-time Endoscopic Image Denoising System 

**Title (ZH)**: 实时内窥镜图像去噪系统 

**Authors**: Yu Xing, Shishi Huang, Meng Lv, Guo Chen, Huailiang Wang, Lingzhi Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.15395)  

**Abstract**: Endoscopes featuring a miniaturized design have significantly enhanced operational flexibility, portability, and diagnostic capability while substantially reducing the invasiveness of medical procedures. Recently, single-use endoscopes equipped with an ultra-compact analogue image sensor measuring less than 1mm x 1mm bring revolutionary advancements to medical diagnosis. They reduce the structural redundancy and large capital expenditures associated with reusable devices, eliminate the risk of patient infections caused by inadequate disinfection, and alleviate patient suffering. However, the limited photosensitive area results in reduced photon capture per pixel, requiring higher photon sensitivity settings to maintain adequate brightness. In high-contrast medical imaging scenarios, the small-sized sensor exhibits a constrained dynamic range, making it difficult to simultaneously capture details in both highlights and shadows, and additional localized digital gain is required to compensate. Moreover, the simplified circuit design and analog signal transmission introduce additional noise sources. These factors collectively contribute to significant noise issues in processed endoscopic images. In this work, we developed a comprehensive noise model for analog image sensors in medical endoscopes, addressing three primary noise types: fixed-pattern noise, periodic banding noise, and mixed Poisson-Gaussian noise. Building on this analysis, we propose a hybrid denoising system that synergistically combines traditional image processing algorithms with advanced learning-based techniques for captured raw frames from sensors. Experiments demonstrate that our approach effectively reduces image noise without fine detail loss or color distortion, while achieving real-time performance on FPGA platforms and an average PSNR improvement from 21.16 to 33.05 on our test dataset. 

**Abstract (ZH)**: 带有微型化设计的内窥镜显著提升了操作灵活性、便携性和诊断能力，并大幅减少了医疗程序的侵入性。近期，配备超紧凑模拟图像传感器（尺寸小于1mm x 1mm）的一次性内窥镜为医疗诊断带来了革命性的进步。它们减少了与可重复使用设备相关的结构冗余和大规模资本支出，消除了因消毒不彻底导致的患者感染风险，并减轻了患者的痛苦。然而，有限的感光区域导致每个像素的光子捕获减少，需要设置更高的光子敏感度以维持足够的亮度。在高对比度的医疗成像场景中，小型传感器表现出受限的动态范围，难以同时捕捉高光和阴影中的细节，需要额外的局部数字增益来补偿。此外，简化的电路设计和模拟信号传输引入了额外的噪声源。这些因素共同导致了处理后的内窥镜图像中的显著噪声问题。在本文中，我们为医疗内窥镜中的模拟图像传感器开发了一个全面的噪声模型，解决了三种主要的噪声类型：固定模式噪声、周期性条纹噪声和混合泊松-高斯噪声。在此基础上，我们提出了一种结合传统图像处理算法和先进学习技术的混合去噪系统，用于传感器捕获的原始帧。实验结果表明，我们的方法在不牺牲细节和颜色保真度的情况下有效减少了图像噪声，同时在FPGA平台上实现了实时性能，并在我们的测试数据集上平均PSNR提高了从21.16到33.05。 

---
# Open-World Object Counting in Videos 

**Title (ZH)**: 开放世界视频中对象计数 

**Authors**: Niki Amini-Naieni, Andrew Zisserman  

**Link**: [PDF](https://arxiv.org/pdf/2506.15368)  

**Abstract**: We introduce a new task of open-world object counting in videos: given a text description, or an image example, that specifies the target object, the objective is to enumerate all the unique instances of the target objects in the video. This task is especially challenging in crowded scenes with occlusions and similar objects, where avoiding double counting and identifying reappearances is crucial. To this end, we make the following contributions: we introduce a model, CountVid, for this task. It leverages an image-based counting model, and a promptable video segmentation and tracking model to enable automated, open-world object counting across video frames. To evaluate its performance, we introduce VideoCount, a new dataset for our novel task built from the TAO and MOT20 tracking datasets, as well as from videos of penguins and metal alloy crystallization captured by x-rays. Using this dataset, we demonstrate that CountVid provides accurate object counts, and significantly outperforms strong baselines. The VideoCount dataset, the CountVid model, and all the code are available at this https URL. 

**Abstract (ZH)**: 开放世界视频中对象计数任务：给定文本描述或图像示例指定目标对象，目标是计算视频中目标对象的所有唯一实例。特别是在遮挡和相似对象众多的场景中，避免重复计数和识别再出现是尤其具有挑战性的任务。为此，我们做出了以下贡献：我们引入了一个名为CountVid的模型来解决这一任务。该模型结合了基于图像的计数模型和可调提示的视频分割与跟踪模型，能够在视频帧间实现自动的开放世界对象计数。为了评估其性能，我们构建了一个新的数据集VideoCount，该项目基于TAO和MOT20跟踪数据集，并包含了通过X射线拍摄的企鹅和金属合金结晶学视频。利用这个数据集，我们展示了CountVid提供了准确的对象计数，并显著优于强基线。VideoCount数据集、CountVid模型以及所有相关代码均可从此链接获得。 

---
# MapFM: Foundation Model-Driven HD Mapping with Multi-Task Contextual Learning 

**Title (ZH)**: MapFM：基于多任务上下文学习的基础模型驱动高精度地图构建 

**Authors**: Leonid Ivanov, Vasily Yuryev, Dmitry Yudin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15313)  

**Abstract**: In autonomous driving, high-definition (HD) maps and semantic maps in bird's-eye view (BEV) are essential for accurate localization, planning, and decision-making. This paper introduces an enhanced End-to-End model named MapFM for online vectorized HD map generation. We show significantly boost feature representation quality by incorporating powerful foundation model for encoding camera images. To further enrich the model's understanding of the environment and improve prediction quality, we integrate auxiliary prediction heads for semantic segmentation in the BEV representation. This multi-task learning approach provides richer contextual supervision, leading to a more comprehensive scene representation and ultimately resulting in higher accuracy and improved quality of the predicted vectorized HD maps. The source code is available at this https URL. 

**Abstract (ZH)**: 在自动驾驶中，鸟瞰视角（BEV）的高定义（HD）地图和语义地图对于准确的位置定位、规划和决策至关重要。本文介绍了一种增强的端到端模型MapFM，用于在线生成矢量化的HD地图。通过融入强大的基础模型来编码摄像头图像，显著提升了特征表示的质量。为进一步丰富模型对环境的理解并提高预测质量，我们将语义分割辅助预测头整合到了BEV表示中。这种多任务学习方法提供了更加丰富的上下文监督，导致更全面的场景表示，从而提高了矢量化HD地图预测的准确性和质量。源代码可在此处访问：this https URL。 

---
# Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models 

**Title (ZH)**: 基于衣物意识扩散模型的宽松稀疏惯性传感器人体运动捕捉 

**Authors**: Andela Ilic, Jiaxi Jiang, Paul Streli, Xintong Liu, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15290)  

**Abstract**: Motion capture using sparse inertial sensors has shown great promise due to its portability and lack of occlusion issues compared to camera-based tracking. Existing approaches typically assume that IMU sensors are tightly attached to the human body. However, this assumption often does not hold in real-world scenarios. In this paper, we present a new task of full-body human pose estimation using sparse, loosely attached IMU sensors. To solve this task, we simulate IMU recordings from an existing garment-aware human motion dataset. We developed transformer-based diffusion models to synthesize loose IMU data and estimate human poses based on this challenging loose IMU data. In addition, we show that incorporating garment-related parameters while training the model on simulated loose data effectively maintains expressiveness and enhances the ability to capture variations introduced by looser or tighter garments. Experiments show that our proposed diffusion methods trained on simulated and synthetic data outperformed the state-of-the-art methods quantitatively and qualitatively, opening up a promising direction for future research. 

**Abstract (ZH)**: 基于稀疏松挂惯性传感器的全身人体姿态估计 

---
# Domain Adaptation for Image Classification of Defects in Semiconductor Manufacturing 

**Title (ZH)**: 半导体制造中缺陷图像分类的领域适应方法 

**Authors**: Adrian Poniatowski, Natalie Gentner, Manuel Barusco, Davide Dalle Pezze, Samuele Salti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2506.15260)  

**Abstract**: In the semiconductor sector, due to high demand but also strong and increasing competition, time to market and quality are key factors in securing significant market share in various application areas. Thanks to the success of deep learning methods in recent years in the computer vision domain, Industry 4.0 and 5.0 applications, such as defect classification, have achieved remarkable success. In particular, Domain Adaptation (DA) has proven highly effective since it focuses on using the knowledge learned on a (source) domain to adapt and perform effectively on a different but related (target) domain. By improving robustness and scalability, DA minimizes the need for extensive manual re-labeling or re-training of models. This not only reduces computational and resource costs but also allows human experts to focus on high-value tasks. Therefore, we tested the efficacy of DA techniques in semi-supervised and unsupervised settings within the context of the semiconductor field. Moreover, we propose the DBACS approach, a CycleGAN-inspired model enhanced with additional loss terms to improve performance. All the approaches are studied and validated on real-world Electron Microscope images considering the unsupervised and semi-supervised settings, proving the usefulness of our method in advancing DA techniques for the semiconductor field. 

**Abstract (ZH)**: 在半导体领域，由于市场需求高且竞争激烈，上市时间和产品质量是确保在各类应用领域获得显著市场份额的关键因素。得益于近年来深度学习方法在计算机视觉领域、 Industry 4.0 和 Industry 5.0 应用（如缺陷分类）中的成功，域适应（DA）技术已 proven 高效有效。特别是，域适应专注于利用（源）域中学到的知识来适应和在不同的但相关（目标）域中表现良好，通过提高鲁棒性和可扩展性，DA 最小化了对大量手动重新标签或重新训练模型的需求。这不仅减少了计算和资源成本，还允许人类专家专注于高价值任务。因此，我们测试了域适应技术在半导体领域半监督和无监督设置中的有效性，并提出了一种受 CycleGAN 启发的 DBACS 方法，该方法通过增加额外的损失项以提高性能。所有方法均基于实际的电子显微镜图像在无监督和半监督设置下进行了研究和验证，证明了我们的方法在推进半导体领域的域适应技术方面的重要性。 

---
# SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning 

**Title (ZH)**: SonicVerse：基于音乐特征的多任务学习captioning 

**Authors**: Anuradha Chopra, Abhinaba Roy, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2506.15154)  

**Abstract**: Detailed captions that accurately reflect the characteristics of a music piece can enrich music databases and drive forward research in music AI. This paper introduces a multi-task music captioning model, SonicVerse, that integrates caption generation with auxiliary music feature detection tasks such as key detection, vocals detection, and more, so as to directly capture both low-level acoustic details as well as high-level musical attributes. The key contribution is a projection-based architecture that transforms audio input into language tokens, while simultaneously detecting music features through dedicated auxiliary heads. The outputs of these heads are also projected into language tokens, to enhance the captioning input. This framework not only produces rich, descriptive captions for short music fragments but also directly enables the generation of detailed time-informed descriptions for longer music pieces, by chaining the outputs using a large-language model. To train the model, we extended the MusicBench dataset by annotating it with music features using MIRFLEX, a modular music feature extractor, resulting in paired audio, captions and music feature data. Experimental results show that incorporating features in this way improves the quality and detail of the generated captions. 

**Abstract (ZH)**: 一种将音高检测、人声检测等辅助音乐特征检测任务与_caption生成_结合的多任务音乐 captioning 模型：SonicVerse及其应用 

---
# Insights Informed Generative AI for Design: Incorporating Real-world Data for Text-to-Image Output 

**Title (ZH)**: 基于现实数据的生成式AI设计洞察：文本转图像输出的实现 

**Authors**: Richa Gupta, Alexander Htet Kyaw  

**Link**: [PDF](https://arxiv.org/pdf/2506.15008)  

**Abstract**: Generative AI, specifically text-to-image models, have revolutionized interior architectural design by enabling the rapid translation of conceptual ideas into visual representations from simple text prompts. While generative AI can produce visually appealing images they often lack actionable data for designers In this work, we propose a novel pipeline that integrates DALL-E 3 with a materials dataset to enrich AI-generated designs with sustainability metrics and material usage insights. After the model generates an interior design image, a post-processing module identifies the top ten materials present and pairs them with carbon dioxide equivalent (CO2e) values from a general materials dictionary. This approach allows designers to immediately evaluate environmental impacts and refine prompts accordingly. We evaluate the system through three user tests: (1) no mention of sustainability to the user prior to the prompting process with generative AI, (2) sustainability goals communicated to the user before prompting, and (3) sustainability goals communicated along with quantitative CO2e data included in the generative AI outputs. Our qualitative and quantitative analyses reveal that the introduction of sustainability metrics in the third test leads to more informed design decisions, however, it can also trigger decision fatigue and lower overall satisfaction. Nevertheless, the majority of participants reported incorporating sustainability principles into their workflows in the third test, underscoring the potential of integrated metrics to guide more ecologically responsible practices. Our findings showcase the importance of balancing design freedom with practical constraints, offering a clear path toward holistic, data-driven solutions in AI-assisted architectural design. 

**Abstract (ZH)**: 生成式AI，特别是文本到图像模型，通过将简单的文本提示快速转换为视觉表示，彻底改变了室内建筑设计。尽管生成式AI能够生成视觉吸引力强的图像，但它们往往缺乏可操作的数据供设计师使用。在这项工作中，我们提出了一种新的流水线方法，将DALL-E 3与材料数据集集成，以在AI生成的设计中增加可持续性指标和材料使用洞察。在模型生成室内设计图像后，后处理模块识别出前十个材料，并与通用材料字典中的二氧化碳当量（CO2e）值配对。这种方法允许设计师立即评估环境影响并据此调整提示。我们通过三个用户测试评估了该系统：（1）在使用生成式AI之前不提及可持续性；（2）在使用生成式AI之前向用户传达可持续性目标；（3）在使用生成式AI生成输出时同时传达可持续性目标和定量的CO2e数据。我们的定性和定量分析表明，在第三次测试中引入可持续性指标会导致更具信息性的设计决策，但也可能引发决策疲劳并降低总体满意度。然而，大多数参与者在第三次测试中报告将可持续性原则纳入其工作流程，这突显了综合指标在引导更生态负责的做法方面的潜在价值。我们的研究结果展示了在保持设计自由的同时平衡实际限制的重要性，并为AI辅助建筑设计的全面、数据驱动解决方案提供了明确路径。 

---
# Improved Image Reconstruction and Diffusion Parameter Estimation Using a Temporal Convolutional Network Model of Gradient Trajectory Errors 

**Title (ZH)**: 基于梯度轨迹误差的时序卷积网络模型改进图像重建和扩散参数估计 

**Authors**: Jonathan B. Martin, Hannah E. Alderson, John C. Gore, Mark D. Does, Kevin D. Harkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.14995)  

**Abstract**: Summary: Errors in gradient trajectories introduce significant artifacts and distortions in magnetic resonance images, particularly in non-Cartesian imaging sequences, where imperfect gradient waveforms can greatly reduce image quality. Purpose: Our objective is to develop a general, nonlinear gradient system model that can accurately predict gradient distortions using convolutional networks. Methods: A set of training gradient waveforms were measured on a small animal imaging system, and used to train a temporal convolutional network to predict the gradient waveforms produced by the imaging system. Results: The trained network was able to accurately predict nonlinear distortions produced by the gradient system. Network prediction of gradient waveforms was incorporated into the image reconstruction pipeline and provided improvements in image quality and diffusion parameter mapping compared to both the nominal gradient waveform and the gradient impulse response function. Conclusion: Temporal convolutional networks can more accurately model gradient system behavior than existing linear methods and may be used to retrospectively correct gradient errors. 

**Abstract (ZH)**: 摘要：梯度轨迹中的误差会引入磁共振图像中的显著伪影和失真，特别是在非笛卡尔成像序列中，不完美的梯度波形可以显著降低图像质量。目的：我们的目标是开发一个通用的非线性梯度系统模型，使用卷积网络准确预测梯度失真。方法：在小型动物成像系统中测量了一组训练梯度波形，并使用这些波形训练时间卷积网络以预测成像系统产生的梯度波形。结果：训练后的网络能够准确预测梯度系统产生的非线性失真。将网络预测的梯度波形整合到图像重建管道中，与名义梯度波形和梯度冲击响应函数相比，提供了图像质量和扩散参数映射的改进。结论：时间卷积网络可以比现有线性方法更准确地建模梯度系统行为，并可能用于回顾性纠正梯度误差。 

---
# Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction 

**Title (ZH)**: 探索未知：基于神经不确定性图的主动视图选择在三维重建中的应用 

**Authors**: Zhengquan Zhang, Feng Xu, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14856)  

**Abstract**: Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training. 

**Abstract (ZH)**: 一些视角自然提供了更多的信息。如何使AI系统确定哪些视角能提供最 valuable insight 以实现准确且高效的3D物体重构？基于神经不确定性图的主动视角选择（AVS）在计算机视觉中仍然是一个基本挑战。目标是识别出能够实现最准确3D重构的最小视角集。不同于从当前观察中学习辐射场或3D高斯点云方法，我们提出了一种新的由轻量级前馈深度神经网络（UPNet）引导的AVS方法，并预测神经不确定性图。UPNet接收3D物体的一张输入图像，并输出预测的不确定性图，表示所有候选视角的不确定性值。通过利用从观察许多自然物体及其相关不确定性模式中得到的经验法则，我们训练UPNet学习从视角外观到底层体和平面表示中的不确定性之间的直接映射。然后，我们的方法聚合所有先前预测的神经不确定性图以抑制冗余的候选视角，并有效选择最信息量的视角。使用这些选定的视角，我们训练3D神经渲染模型，并将新型视角合成的质量与其他竞争性AVS方法进行评估。令人惊讶的是，尽管使用的视角数量仅为上限的一半，我们的方法仍能达到相似的重构准确度。此外，它显著降低了AVS过程中的计算开销，与基线方法相比，最多可实现400倍的速度提升，并且在CPU、RAM和GPU使用上降低超过50%。值得注意的是，我们的方法在不需要任何额外训练的情况下有效泛化到涉及新物体类别的AVS任务中。 

---
# Efficient Retail Video Annotation: A Robust Key Frame Generation Approach for Product and Customer Interaction Analysis 

**Title (ZH)**: 高效的零售视频标注：一种用于产品和顾客交互分析的稳健关键帧生成方法 

**Authors**: Varun Mannam, Zhenyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14854)  

**Abstract**: Accurate video annotation plays a vital role in modern retail applications, including customer behavior analysis, product interaction detection, and in-store activity recognition. However, conventional annotation methods heavily rely on time-consuming manual labeling by human annotators, introducing non-robust frame selection and increasing operational costs. To address these challenges in the retail domain, we propose a deep learning-based approach that automates key-frame identification in retail videos and provides automatic annotations of products and customers. Our method leverages deep neural networks to learn discriminative features by embedding video frames and incorporating object detection-based techniques tailored for retail environments. Experimental results showcase the superiority of our approach over traditional methods, achieving accuracy comparable to human annotator labeling while enhancing the overall efficiency of retail video annotation. Remarkably, our approach leads to an average of 2 times cost savings in video annotation. By allowing human annotators to verify/adjust less than 5% of detected frames in the video dataset, while automating the annotation process for the remaining frames without reducing annotation quality, retailers can significantly reduce operational costs. The automation of key-frame detection enables substantial time and effort savings in retail video labeling tasks, proving highly valuable for diverse retail applications such as shopper journey analysis, product interaction detection, and in-store security monitoring. 

**Abstract (ZH)**: 准确的视频标注在现代零售应用中发挥着重要作用，包括顾客行为分析、产品交互检测和店内活动识别。然而，传统的标注方法高度依赖耗时的手动标注，导致帧选择不够稳健并增加了运营成本。为了解决零售领域的这些挑战，我们提出了一种基于深度学习的方法，自动识别零售视频的关键帧，并提供产品和顾客的自动标注。该方法利用深度神经网络学习具有鉴别性的特征，通过嵌入视频帧并结合适用于零售环境的对象检测技术。实验结果展示了我们方法在准确性和传统方法相比的优势，同时提高了零售视频标注的整体效率。令人印象深刻的是，我们的方法在视频标注方面平均节省了20%的成本。通过让人工标注员验证/调整视频数据集中少于5%的检测帧，而自动化剩余帧的标注过程，零售商可以显著降低运营成本。关键帧检测的自动化在零售视频标签任务中节省了大量的时间和精力，对多种零售应用，如购物者旅程分析、产品交互检测和店内安全监控，都非常有价值。 

---
# Finding Optimal Kernel Size and Dimension in Convolutional Neural Networks An Architecture Optimization Approach 

**Title (ZH)**: 在卷积神经网络中寻找最优核大小和维度的架构优化方法 

**Authors**: Shreyas Rajeev, B Sathish Babu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14846)  

**Abstract**: Kernel size selection in Convolutional Neural Networks (CNNs) is a critical but often overlooked design decision that affects receptive field, feature extraction, computational cost, and model accuracy. This paper proposes the Best Kernel Size Estimation Function (BKSEF), a mathematically grounded and empirically validated framework for optimal, layer-wise kernel size determination. BKSEF balances information gain, computational efficiency, and accuracy improvements by integrating principles from information theory, signal processing, and learning theory. Extensive experiments on CIFAR-10, CIFAR-100, ImageNet-lite, ChestX-ray14, and GTSRB datasets demonstrate that BKSEF-guided architectures achieve up to 3.1 percent accuracy improvement and 42.8 percent reduction in FLOPs compared to traditional models using uniform 3x3 kernels. Two real-world case studies further validate the approach: one for medical image classification in a cloud-based setup, and another for traffic sign recognition on edge devices. The former achieved enhanced interpretability and accuracy, while the latter reduced latency and model size significantly, with minimal accuracy trade-off. These results show that kernel size can be an active, optimizable parameter rather than a fixed heuristic. BKSEF provides practical heuristics and theoretical support for researchers and developers seeking efficient and application-aware CNN designs. It is suitable for integration into neural architecture search pipelines and real-time systems, offering a new perspective on CNN optimization. 

**Abstract (ZH)**: 卷积神经网络（CNN）中的内核大小选择是一种关键但常常被忽视的设计决策，它影响着感受野、特征提取、计算成本和模型准确性。本文提出了最佳内核大小估计函数（BKSEF），这是一种基于数学原理并经过实验证明的框架，用于最优的逐层内核大小确定。BKSEF通过结合信息理论、信号处理和学习理论的原则，平衡信息增益、计算效率和精度提升。在CIFAR-10、CIFAR-100、ImageNet-lite、ChestX-ray14和GTSRB数据集上的广泛实验表明，BKSEF引导的架构相比使用统一3x3内核的传统模型，可以实现多达3.1个百分点的准确性提升和42.8个百分点的FLOPs减少。两个实际案例进一步验证了该方法：一个是在云环境中进行医学图像分类的应用，另一个是在边缘设备上进行交通标志识别的应用。前者提高了解释性和准确性，后者显著减少了延迟和模型大小，同时仅有轻微的准确性妥协。这些结果表明，内核大小可以是一个活跃的、可优化的参数，而不仅仅是一个固定的启发式参数。BKSEF为寻求高效和应用感知的CNN设计的研究人员和开发人员提供了实用的启发式方法和理论支持。它可以集成到神经架构搜索管道和实时系统中，为CNN优化提供了一个新的视角。 

---
# PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers 

**Title (ZH)**: PictSure: 预训练嵌入对于基于上下文学习图像分类器很重要 

**Authors**: Lukas Schiesser, Cornelius Wolff, Sophie Haas, Simon Pukrop  

**Link**: [PDF](https://arxiv.org/pdf/2506.14842)  

**Abstract**: Building image classification models remains cumbersome in data-scarce domains, where collecting large labeled datasets is impractical. In-context learning (ICL) has emerged as a promising paradigm for few-shot image classification (FSIC), enabling models to generalize across domains without gradient-based adaptation. However, prior work has largely overlooked a critical component of ICL-based FSIC pipelines: the role of image embeddings. In this work, we present PictSure, an ICL framework that places the embedding model -- its architecture, pretraining, and training dynamics -- at the center of analysis. We systematically examine the effects of different visual encoder types, pretraining objectives, and fine-tuning strategies on downstream FSIC performance. Our experiments show that the training success and the out-of-domain performance are highly dependent on how the embedding models are pretrained. Consequently, PictSure manages to outperform existing ICL-based FSIC models on out-of-domain benchmarks that differ significantly from the training distribution, while maintaining comparable results on in-domain tasks. Code can be found at this https URL. 

**Abstract (ZH)**: 在数据稀缺领域建立图像分类模型仍然较为繁琐，其中收集大型带标签数据集是不实际的。上下文学习（ICL）已成为少量样本图像分类（FSIC）的一种有前景的范式，使模型能够在无需基于梯度的适应情况下跨域泛化。然而，先前的工作大多忽视了ICL基FSIC管道中的一个关键组成部分：图像嵌入的作用。本文中，我们提出PictSure，这是一种将嵌入模型——其架构、预训练和训练动力学——放在核心分析位置的ICL框架。我们系统地研究了不同类型视觉编码器、预训练目标和微调策略对下游FSIC性能的影响。我们的实验表明，训练成功率和域外性能高度依赖于嵌入模型的预训练方式。因此，PictSure在训练分布与域外基准有显著差异的情况下取得了优于现有ICL基FSIC模型的表现，同时在域内任务上保持了相当的结果。代码可在此链接找到。 

---
# Deploying and Evaluating Multiple Deep Learning Models on Edge Devices for Diabetic Retinopathy Detection 

**Title (ZH)**: 在边缘设备上部署和评估多种深度学习模型以检测糖尿病视网膜病变 

**Authors**: Akwasi Asare, Dennis Agyemanh Nana Gookyi, Derrick Boateng, Fortunatus Aabangbio Wulnye  

**Link**: [PDF](https://arxiv.org/pdf/2506.14834)  

**Abstract**: Diabetic Retinopathy (DR), a leading cause of vision impairment in individuals with diabetes, affects approximately 34.6% of diabetes patients globally, with the number of cases projected to reach 242 million by 2045. Traditional DR diagnosis relies on the manual examination of retinal fundus images, which is both time-consuming and resource intensive. This study presents a novel solution using Edge Impulse to deploy multiple deep learning models for real-time DR detection on edge devices. A robust dataset of over 3,662 retinal fundus images, sourced from the Kaggle EyePACS dataset, was curated, and enhanced through preprocessing techniques, including augmentation and normalization. Using TensorFlow, various Convolutional Neural Networks (CNNs), such as MobileNet, ShuffleNet, SqueezeNet, and a custom Deep Neural Network (DNN), were designed, trained, and optimized for edge deployment. The models were converted to TensorFlowLite and quantized to 8-bit integers to reduce their size and enhance inference speed, with minimal trade-offs in accuracy. Performance evaluations across different edge hardware platforms, including smartphones and microcontrollers, highlighted key metrics such as inference speed, accuracy, precision, and resource utilization. MobileNet achieved an accuracy of 96.45%, while SqueezeNet demonstrated strong real-time performance with a small model size of 176 KB and latency of just 17 ms on GPU. ShuffleNet and the custom DNN achieved moderate accuracy but excelled in resource efficiency, making them suitable for lower-end devices. This integration of edge AI technology into healthcare presents a scalable, cost-effective solution for early DR detection, providing timely and accurate diagnosis, especially in resource-constrained and remote healthcare settings. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）：导致糖尿病患者视力损伤的主要原因，全球约影响34.6%的糖尿病患者，预计到2045年病例数将达到242 million。传统的DR诊断依赖于对眼底图像的手动检查，这既耗时又资源密集。本研究提出了一个使用Edge Impulse的新型解决方案，以在边缘设备上实现实时DR检测。一个包含超过3,662张眼底图像的稳健数据集，来源于Kaggle EyePACS数据集，并通过预处理技术，包括增强和归一化，进行了增强。使用TensorFlow设计并训练了多种卷积神经网络（CNNs），如MobileNet、ShuffleNet、SqueezeNet，以及一个自定义的深度神经网络（DNN），并进行了边端部署优化。将模型转换为TensorFlowLite并量化为8位整数，以减少模型大小和提高推理速度，同时保持较高的准确性。在不同的边缘硬件平台上，包括智能手机和微控制器，进行了性能评估，重点考察了推理速度、准确性、精度和资源利用率等关键指标。MobileNet的准确率为96.45%，而SqueezeNet展示了强大的实时性能，模型大小仅为176 KB，GPU上的延迟仅为17 ms。ShuffleNet和自定义DNN的准确率中等，但在资源效率方面表现出色，适用于低端设备。将边缘AI技术融入医疗保健为早期DR检测提供了一个可扩展且成本效益高的解决方案，尤其是在资源有限和偏远的医疗保健环境中提供了及时且准确的诊断。 

---
# DAVID-XR1: Detecting AI-Generated Videos with Explainable Reasoning 

**Title (ZH)**: DAVID-XR1: 可解释推理检测AI生成视频 

**Authors**: Yifeng Gao, Yifan Ding, Hongyu Su, Juncheng Li, Yunhan Zhao, Lin Luo, Zixing Chen, Li Wang, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14827)  

**Abstract**: As AI-generated video becomes increasingly pervasive across media platforms, the ability to reliably distinguish synthetic content from authentic footage has become both urgent and essential. Existing approaches have primarily treated this challenge as a binary classification task, offering limited insight into where or why a model identifies a video as AI-generated. However, the core challenge extends beyond simply detecting subtle artifacts; it requires providing fine-grained, persuasive evidence that can convince auditors and end-users alike. To address this critical gap, we introduce DAVID-X, the first dataset to pair AI-generated videos with detailed defect-level, temporal-spatial annotations and written rationales. Leveraging these rich annotations, we present DAVID-XR1, a video-language model designed to deliver an interpretable chain of visual reasoning-including defect categorization, temporal-spatial localization, and natural language explanations. This approach fundamentally transforms AI-generated video detection from an opaque black-box decision into a transparent and verifiable diagnostic process. We demonstrate that a general-purpose backbone, fine-tuned on our compact dataset and enhanced with chain-of-thought distillation, achieves strong generalization across a variety of generators and generation modes. Our results highlight the promise of explainable detection methods for trustworthy identification of AI-generated video content. 

**Abstract (ZH)**: 随着AI生成视频在各类媒体平台上的广泛应用，可靠地区分合成内容与真实 footage的能力变得至关重要且不可或缺。现有的方法主要将这一挑战视为二元分类任务，提供的洞察有限，难以解释模型为何将视频识别为AI生成。然而，核心挑战远不止于检测细微的瑕疵，还要求提供精细、有说服力的证据，以使审计人员和最终用户信服。为填补这一重要空白，我们引入了DAVID-X，这是首个将AI生成视频与详细的缺陷级别、时空注释及书面解释相匹配的数据集。利用这些丰富的注释，我们提出了DAVID-XR1，一种视频-语言模型，旨在提供可解释的视觉推理链，包括缺陷分类、时空定位和自然语言解释。这种方法从根本上将AI生成视频的检测从不透明的黑盒决策转变为透明且可验证的诊断过程。我们证明，一个通用的骨干网络，在我们的紧凑数据集上微调，并结合链式思维提炼，能够跨越多种生成器和生成模式实现强大的泛化能力。我们的结果展示了可解释检测方法在可靠识别AI生成视频内容方面的前景。 

---
# GraphGSOcc: Semantic and Geometric Graph Transformer for 3D Gaussian Splating-based Occupancy Prediction 

**Title (ZH)**: GraphGSOcc: 具有语义和几何图变换器的3D高斯分裂基于占用预测 

**Authors**: Ke Song, Yunhe Wu, Chunchit Siu, Huiyuan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14825)  

**Abstract**: Addressing the task of 3D semantic occupancy prediction for autonomous driving, we tackle two key issues in existing 3D Gaussian Splating (3DGS) methods: (1) unified feature aggregation neglecting semantic correlations among similar categories and across regions, and (2) boundary ambiguities caused by the lack of geometric constraints in MLP iterative optimization. We propose the GraphGSOcc model, a novel framework that combines semantic and geometric graph Transformer for 3D Gaussian Splating-based Occupancy Prediction. We propose the Dual Gaussians Graph Attenntion, which dynamically constructs dual graph structures: a geometric graph adaptively calculating KNN search radii based on Gaussian poses, enabling large-scale Gaussians to aggregate features from broader neighborhoods while compact Gaussians focus on local geometric consistency; a semantic graph retaining top-M highly correlated nodes via cosine similarity to explicitly encode semantic relationships within and across instances. Coupled with the Multi-scale Graph Attention framework, fine-grained attention at lower layers optimizes boundary details, while coarse-grained attention at higher layers models object-level topology. Experiments on the SurroundOcc dataset achieve an mIoU of 24.10%, reducing GPU memory to 6.1 GB, demonstrating a 1.97% mIoU improvement and 13.7% memory reduction compared to GaussianWorld 

**Abstract (ZH)**: 基于3D高斯分裂的方法解决自主驾驶中的3D语义占有预测任务，我们提出了GraphGSOcc模型，该模型结合语义和几何图变换器以解决现有3D高斯分裂方法中的两个关键问题：统一特征聚合忽视了相似类别内的语义关联和跨区域的语义关联，以及由于MLP迭代优化中缺乏几何约束导致的边界模糊。 

---
# ViLLa: A Neuro-Symbolic approach for Animal Monitoring 

**Title (ZH)**: ViLLa: 一种神经符号方法用于动物监测 

**Authors**: Harsha Koduri  

**Link**: [PDF](https://arxiv.org/pdf/2506.14823)  

**Abstract**: Monitoring animal populations in natural environments requires systems that can interpret both visual data and human language queries. This work introduces ViLLa (Vision-Language-Logic Approach), a neuro-symbolic framework designed for interpretable animal monitoring. ViLLa integrates three core components: a visual detection module for identifying animals and their spatial locations in images, a language parser for understanding natural language queries, and a symbolic reasoning layer that applies logic-based inference to answer those queries. Given an image and a question such as "How many dogs are in the scene?" or "Where is the buffalo?", the system grounds visual detections into symbolic facts and uses predefined rules to compute accurate answers related to count, presence, and location. Unlike end-to-end black-box models, ViLLa separates perception, understanding, and reasoning, offering modularity and transparency. The system was evaluated on a range of animal imagery tasks and demonstrates the ability to bridge visual content with structured, human-interpretable queries. 

**Abstract (ZH)**: 在自然环境中的动物种群监测需要能够解释视觉数据和人类语言查询的系统。本文介绍了ViLLa（视觉-语言-逻辑方法），这是一种用于可解释动物监测的神经符号框架。ViLLa 结合了三个核心组件：一个视觉检测模块，用于识别图像中的动物及其空间位置；一个语言解析器，用于理解自然语言查询；以及一个符号推理层，利用基于逻辑的推断来回答这些查询。给定一张图片和一个问题，比如“场景中有多少条狗？”或“水牛在哪里？”，该系统将视觉检测结果转化为符号事实，并使用预定义的规则来计算与数量、存在和位置相关的准确答案。与端到端的黑盒模型不同，ViLLa 将感知、理解和推理分离，提供了模块化和透明性。该系统在一系列动物图像任务上进行了评估，并展示了将视觉内容与结构化的人类可解释查询相结合的能力。 

---
