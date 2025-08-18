# Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks 

**Title (ZH)**: 视觉感知引擎：快速灵活的多头推理方法及其在机器人视觉任务中的应用 

**Authors**: Jakub Łucki, Jonathan Becktor, Georgios Georgakis, Robert Royce, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2508.11584)  

**Abstract**: Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capabilities through an example implementation using DINOv2 as the foundation model with multiple task (depth, object detection and semantic segmentation) heads, achieving up to 3x speedup compared to sequential execution. Building on CUDA Multi-Process Service (MPS), VPEngine offers efficient GPU utilization and maintains a constant memory footprint while allowing per-task inference frequencies to be adjusted dynamically during runtime. The framework is written in Python and is open source with ROS2 C++ (Humble) bindings for ease of use by the robotics community across diverse robotic platforms. Our example implementation demonstrates end-to-end real-time performance at $\geq$50 Hz on NVIDIA Jetson Orin AGX for TensorRT optimized models. 

**Abstract (ZH)**: 基于受限资源机器人平台的多机器学习模型部署常常导致冗余计算、大内存占用和复杂的集成挑战。为应对这一问题，本文提出Visual Perception Engine (VPEngine)，一个模块化框架，旨在实现视觉多任务处理的高效GPU使用，同时保持扩展性和开发者的易用性。该框架架构利用共享的基础模型骨干来提取图像表示，并在多个并行运行的专业任务特定模型头部之间高效共享，无需不必要的GPU-CPU内存传输。这种设计在部署传统顺序模型时消除了特征提取部分的冗余计算问题，并允许基于应用程序需求动态调整任务优先级。通过使用DINOv2作为基础模型和多个任务（深度、物体检测和语义分割）头部的实施示例，我们展示了高达3倍的提速效果。基于CUDA Multi-Process Service (MPS)，VPEngine实现了高效的GPU使用，保持恒定的内存占用，并允许在运行时根据任务调整推理频率。该框架用Python编写，并提供ROS2 C++（Humble）绑定，便于各类机器人平台的使用。我们的示例实现展示了在NVIDIA Jetson Orin AGX上针对TensorRT优化的模型实现端到端实时性能，频率≥50 Hz。 

---
# Relative Position Matters: Trajectory Prediction and Planning with Polar Representation 

**Title (ZH)**: 相对位置 Matters：基于极坐标表示的轨迹预测与规划 

**Authors**: Bozhou Zhang, Nan Song, Bingzhao Gao, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11492)  

**Abstract**: Trajectory prediction and planning in autonomous driving are highly challenging due to the complexity of predicting surrounding agents' movements and planning the ego agent's actions in dynamic environments. Existing methods encode map and agent positions and decode future trajectories in Cartesian coordinates. However, modeling the relationships between the ego vehicle and surrounding traffic elements in Cartesian space can be suboptimal, as it does not naturally capture the varying influence of different elements based on their relative distances and directions. To address this limitation, we adopt the Polar coordinate system, where positions are represented by radius and angle. This representation provides a more intuitive and effective way to model spatial changes and relative relationships, especially in terms of distance and directional influence. Based on this insight, we propose Polaris, a novel method that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches. By leveraging the Polar representation, this method explicitly models distance and direction variations and captures relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning. Extensive experiments on the challenging prediction (Argoverse 2) and planning benchmarks (nuPlan) demonstrate that Polaris achieves state-of-the-art performance. 

**Abstract (ZH)**: 自动驾驶中轨迹预测与规划由于周围代理动态运动的复杂性而极具挑战性，现有方法通过笛卡尔坐标编码地图和代理位置并解码未来的轨迹。为解决笛卡尔空间中 ego 车辆与周围交通元素之间关系建模不自然的问题，我们采用极坐标系，其中位置由半径和角度表示。这种表示方式提供了一种更直观且有效的方式来建模空间变化和相对关系，特别是在距离和方向影响方面。基于这一洞见，我们提出 Polaris，一种完全在极坐标系中运行的新方法，区别于传统的笛卡尔坐标系方法。通过利用极坐标表示，该方法明确建模了距离和方向的变化，并通过专用的编码和精炼模块捕捉相对关系，从而实现更结构化的、空间意识更强的轨迹预测与规划。在具有挑战性的预测（Argoverse 2）和规划基准测试（nuPlan）上的广泛实验表明，Polaris 达到了最先进的性能。 

---
# Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction 

**Title (ZH)**: 基于视觉的高斯散射协作语义占用预测 

**Authors**: Cheng Chen, Hao Huang, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10936)  

**Abstract**: Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets. 

**Abstract (ZH)**: 协作感知使连接车辆能够共享信息，克服遮挡并扩展单代理（非协作）系统固有的有限感知范围。为解决这些挑战，我们提出了首个利用稀疏3D语义高斯点阵进行协作3D语义占用预测的方法。通过共享和融合中间的高斯原始数据，我们的方法提供了三项益处：基于邻域的跨代理融合，去除重复并抑制噪声或不一致的高斯；每个原始数据中几何与语义的联合编码，减少对深度监督的依赖并允许简单的刚性对齐；稀疏、对象为中心的消息，保持结构性信息同时减少通信量。 extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. 当进一步减少传输的高斯数量时，即使使用仅为34.6%的通信量，我们的方法仍然实现了mIoU +1.9的提升，突出显示在有限通信预算下的稳健性能。 

---
# HQ-OV3D: A High Box Quality Open-World 3D Detection Framework based on Diffision Model 

**Title (ZH)**: HQ-OV33检测框架：基于扩散的高框33 world世界三维检测框架.Dんど 

**Authors**: Qi Liu, Yabei Li, Hongsong Wang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.10935)  

**Abstract**: Traditional closed-set 3D detection frameworks fail to meet the demands of open-world applications like autonomous driving. Existing open-vocabulary 3D detection methods typically adopt a two-stage pipeline consisting of pseudo-label generation followed by semantic alignment. While vision-language models (VLMs) recently have dramatically improved the semantic accuracy of pseudo-labels, their geometric quality, particularly bounding box precision, remains commonly this http URL address this issue, we propose a High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) framework, dedicated to generate and refine high-quality pseudo-labels for open-vocabulary classes. The framework comprises two key components: an Intra-Modality Cross-Validated (IMCV) Proposal Generator that utilizes cross-modality geometric consistency to generate high-quality initial 3D proposals, and an Annotated-Class Assisted (ACA) Denoiser that progressively refines 3D proposals by leveraging geometric priors from annotated categories through a DDIM-based denoising this http URL to the state-of-the-art method, training with pseudo-labels generated by our approach achieves a 7.37% improvement in mAP on novel classes, demonstrating the superior quality of the pseudo-labels produced by our framework. HQ-OV3D can serve not only as a strong standalone open-vocabulary 3D detector but also as a plug-in high-quality pseudo-label generator for existing open-vocabulary detection or annotation pipelines. 

**Abstract (ZH)**: 高品质开放词汇三维检测框架：High Box Quality Open-Vocabulary 3D Detection (HQ-OV3D) 

---
# ViPE: Video Pose Engine for 3D Geometric Perception 

**Title (ZH)**: ViPE: 视频姿态引擎 for 3D 几何感知 

**Authors**: Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, Sanja Fidler  

**Link**: [PDF](https://arxiv.org/pdf/2508.10934)  

**Abstract**: Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360° panoramas. We have benchmarked ViPE on multiple benchmarks. Notably, it outperforms existing uncalibrated pose estimation baselines by 18%/50% on TUM/KITTI sequences, and runs at 3-5FPS on a single GPU for standard input resolutions. We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open-source ViPE and the annotated dataset with the hope of accelerating the development of spatial AI systems. 

**Abstract (ZH)**: 准确的三维几何感知是广泛的空间AI系统的前提。尽管当前最先进的方法依赖大规模训练数据，但从野外视频中获取一致且精确的三维标注仍然是一项关键挑战。在本文中，我们介绍了ViPE，这是一种设计用于解决这一问题的手提式和多功能视频处理引擎。ViPE能够从不受约束的原始视频中高效地估计相机内参、相机运动以及稠密的近象限深度图。它能够应对多种场景，包括动态自拍视频、电影镜头或行车记录仪，并支持各种类型的相机模型，如针孔相机、广角相机和360°全景相机。我们在多个基准上测试了ViPE。值得注意的是，ViPE在TUM/KITTI序列上的表现优于现有未标定的姿态估计基线，分别高出18%/50%，并在单个GPU上以3-5FPS的速度运行，适用于标准输入分辨率。我们使用ViPE对大量视频进行标注，该集合包括约10万个真实世界的互联网视频、100万个高质量的人工智能生成视频以及2000个全景视频，总共约9600万帧，所有视频均附有准确的相机姿态和稠密深度图。我们将ViPE及其标注数据集开源，希望能够加速空间AI系统的开发。 

---
# Pretrained Conformers for Audio Fingerprinting and Retrieval 

**Title (ZH)**: 预训练Conformer模型在音频指纹提取与检索中的应用 

**Authors**: Kemal Altwlkany, Elmedin Selmanovic, Sead Delalic  

**Link**: [PDF](https://arxiv.org/pdf/2508.11609)  

**Abstract**: Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes. 

**Abstract (ZH)**: 基于自监督对比学习框架的协变器在语音处理中的应用：通过短音频片段生成独特嵌入并适应多种音频失真 

---
# Inside Knowledge: Graph-based Path Generation with Explainable Data Augmentation and Curriculum Learning for Visual Indoor Navigation 

**Title (ZH)**: 基于图的路径生成：具有可解释数据增强和渐增学习的视觉室内导航内知识方法 

**Authors**: Daniel Airinei, Elena Burceanu, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2508.11446)  

**Abstract**: Indoor navigation is a difficult task, as it generally comes with poor GPS access, forcing solutions to rely on other sources of information. While significant progress continues to be made in this area, deployment to production applications is still lacking, given the complexity and additional requirements of current solutions. Here, we introduce an efficient, real-time and easily deployable deep learning approach, based on visual input only, that can predict the direction towards a target from images captured by a mobile device. Our technical approach, based on a novel graph-based path generation method, combined with explainable data augmentation and curriculum learning, includes contributions that make the process of data collection, annotation and training, as automatic as possible, efficient and robust. On the practical side, we introduce a novel largescale dataset, with video footage inside a relatively large shopping mall, in which each frame is annotated with the correct next direction towards different specific target destinations. Different from current methods, ours relies solely on vision, avoiding the need of special sensors, additional markers placed along the path, knowledge of the scene map or internet access. We also created an easy to use application for Android, which we plan to make publicly available. We make all our data and code available along with visual demos on our project site 

**Abstract (ZH)**: 室内导航是一个具有挑战性的任务，通常受限于较差的GPS访问，因此解决方案往往会依赖其他信息源。尽管在这个领域持续取得了显著进展，但由于当前解决方案的复杂性和额外要求，将其部署到实际应用中仍然不足。在这里，我们提出了一种高效、实时且易于部署的基于视觉输入的深度学习方法，该方法能够仅从移动设备拍摄的图像中预测目标方向。我们的技术方法基于一种新颖的基于图的路径生成方法，结合可解释的数据增强和层次学习，包括使数据收集、标注和训练过程尽可能自动化、高效和稳健的贡献。在实际应用方面，我们引入了一个大规模的新型数据集，其中包含在相对大型购物商场内部的视频片段，每帧都标注了正确的下一个目标方向。与现有方法不同，我们的方法仅依赖视觉信息，避免了使用特殊传感器、路径上的额外标记、场景地图知识或互联网接入的需求。我们还创建了一个易于使用的Android应用程序，并计划将其公开发布。我们将在项目网站上提供所有数据、代码和视觉演示。 

---
# G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration 

**Title (ZH)**: G-CUT3R：带有相机和深度先验集成的引导三维重建 

**Authors**: Ramil Khafizov, Artem Komarichev, Ruslan Rakhimov, Peter Wonka, Evgeny Burnaev  

**Link**: [PDF](https://arxiv.org/pdf/2508.11379)  

**Abstract**: We introduce G-CUT3R, a novel feed-forward approach for guided 3D scene reconstruction that enhances the CUT3R model by integrating prior information. Unlike existing feed-forward methods that rely solely on input images, our method leverages auxiliary data, such as depth, camera calibrations, or camera positions, commonly available in real-world scenarios. We propose a lightweight modification to CUT3R, incorporating a dedicated encoder for each modality to extract features, which are fused with RGB image tokens via zero convolution. This flexible design enables seamless integration of any combination of prior information during inference. Evaluated across multiple benchmarks, including 3D reconstruction and other multi-view tasks, our approach demonstrates significant performance improvements, showing its ability to effectively utilize available priors while maintaining compatibility with varying input modalities. 

**Abstract (ZH)**: G-CUT3R：一种通过集成先验信息增强的新型前向指导三维场景重建方法 

---
# Does the Skeleton-Recall Loss Really Work? 

**Title (ZH)**: 骨架召回损失真的有效吗？ 

**Authors**: Devansh Arora, Nitin Kumar, Sukrit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.11374)  

**Abstract**: Image segmentation is an important and widely performed task in computer vision. Accomplishing effective image segmentation in diverse settings often requires custom model architectures and loss functions. A set of models that specialize in segmenting thin tubular structures are topology preservation-based loss functions. These models often utilize a pixel skeletonization process claimed to generate more precise segmentation masks of thin tubes and better capture the structures that other models often miss. One such model, Skeleton Recall Loss (SRL) proposed by Kirchhoff et al.~\cite {kirchhoff2024srl}, was stated to produce state-of-the-art results on benchmark tubular datasets. In this work, we performed a theoretical analysis of the gradients for the SRL loss. Upon comparing the performance of the proposed method on some of the tubular datasets (used in the original work, along with some additional datasets), we found that the performance of SRL-based segmentation models did not exceed traditional baseline models. By providing both a theoretical explanation and empirical evidence, this work critically evaluates the limitations of topology-based loss functions, offering valuable insights for researchers aiming to develop more effective segmentation models for complex tubular structures. 

**Abstract (ZH)**: 基于拓扑保护断细的图像分割：Skeleton Recall Loss (SRL)的理论分析与评估 

---
# Leveraging the RETFound foundation model for optic disc segmentation in retinal images 

**Title (ZH)**: 基于RETFound基础模型的眼底图像视神经盘分割方法 

**Authors**: Zhenyi Zhao, Muthu Rama Krishnan Mookiah, Emanuele Trucco  

**Link**: [PDF](https://arxiv.org/pdf/2508.11354)  

**Abstract**: RETFound is a well-known foundation model (FM) developed for fundus camera and optical coherence tomography images. It has shown promising performance across multiple datasets in diagnosing diseases, both eye-specific and systemic, from retinal images. However, to our best knowledge, it has not been used for other tasks. We present the first adaptation of RETFound for optic disc segmentation, a ubiquitous and foundational task in retinal image analysis. The resulting segmentation system outperforms state-of-the-art, segmentation-specific baseline networks after training a head with only a very modest number of task-specific examples. We report and discuss results with four public datasets, IDRID, Drishti-GS, RIM-ONE-r3, and REFUGE, and a private dataset, GoDARTS, achieving about 96% Dice consistently across all datasets. Overall, our method obtains excellent performance in internal verification, domain generalization and domain adaptation, and exceeds most of the state-of-the-art baseline results. We discuss the results in the framework of the debate about FMs as alternatives to task-specific architectures. The code is available at: [link to be added after the paper is accepted] 

**Abstract (ZH)**: RETFound是一种基于视网膜相机和光学相干断层成像图像开发的知名基础模型。它在诊断从视网膜图像中识别的眼部疾病和全身性疾病方面展示出了良好的性能。然而，据我们所知，它尚未被用于其他任务。我们首次将RETFound适应于视盘分割任务，这是一个在视网膜图像分析中普遍且基础的任务。训练仅需少量针对任务的示例后，生成的分割系统在四个公共数据集IDRID、Drishti-GS、RIM-ONE-r3和REFUGE以及一个私有数据集GoDARTS上实现了约96%的Dice一致性表现。总体而言，我们的方法在内部验证、领域泛化和领域适应方面取得了出色的表现，并超过了大多数现有的基线结果。我们将在框架内讨论FMs作为替代任务特定架构的讨论结果。代码将在论文被接受后提供链接。 

---
# Enhancing Supervised Composed Image Retrieval via Reasoning-Augmented Representation Engineering 

**Title (ZH)**: 通过推理增强的表示方式工程以增强监督指导下的图像检索 Closetitle: 通过推理增强的表示工程以增强监督指导下的图像检索 

**Authors**: Jun Li, Kai Li, Shaoguo Liu, Tingting Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.11272)  

**Abstract**: Composed Image Retrieval (CIR) presents a significant challenge as it requires jointly understanding a reference image and a modified textual instruction to find relevant target images. Some existing methods attempt to use a two-stage approach to further refine retrieval results. However, this often requires additional training of a ranking model. Despite the success of Chain-of-Thought (CoT) techniques in reducing training costs for language models, their application in CIR tasks remains limited -- compressing visual information into text or relying on elaborate prompt designs. Besides, existing works only utilize it for zero-shot CIR, as it is challenging to achieve satisfactory results in supervised CIR with a well-trained model. In this work, we proposed a framework that includes the Pyramid Matching Model with Training-Free Refinement (PMTFR) to address these challenges. Through a simple but effective module called Pyramid Patcher, we enhanced the Pyramid Matching Model's understanding of visual information at different granularities. Inspired by representation engineering, we extracted representations from COT data and injected them into the LVLMs. This approach allowed us to obtain refined retrieval scores in the Training-Free Refinement paradigm without relying on explicit textual reasoning, further enhancing performance. Extensive experiments on CIR benchmarks demonstrate that PMTFR surpasses state-of-the-art methods in supervised CIR tasks. The code will be made public. 

**Abstract (ZH)**: 组成的图像检索 (CIR) 提出了一个显著的挑战，因为它要求同时理解参考图像和修改后的文本指令以找到相关的目标图像。一些现有方法尝试使用两阶段方法进一步细化检索结果，但这通常需要对排名模型进行额外训练。尽管思维链 (Chain-of-Thought, CoT) 技术在减少语言模型训练成本方面取得了成功，但在CIR任务中的应用仍然有限——将其应用于视觉信息压缩或将依赖于精细的提示设计。此外，现有工作仅将其用于零样本CIR，因为即使使用训练良好的模型，在监督CIR任务中达到满意结果也是具有挑战性的。在本工作中，我们提出了一种包含无需训练的精炼 Pyramid 匹配模型 (PMTFR) 的框架来应对这些挑战。通过一个简单但有效的模块 Pyramid Patcher，我们增强了 Pyramid 匹配模型对不同粒度视觉信息的理解能力。受表示工程的启发，我们从思维链 (CoT) 数据中提取表示并注入到语言-视觉模型 LVLMs 中。这种方法使我们能够在无需依赖显式的文本推理的情况下获得精炼的检索分数，在 Training-Free Refinement 架构中进一步提高性能。在CIR基准上的广泛实验表明，PMTFR 在监督CIR任务中超过了现有最先进的方法。代码将公开。 

---
# Vision-Language Models display a strong gender bias 

**Title (ZH)**: Vision-Language模型表现出强烈的性别偏见 

**Authors**: Aiswarya Konavoor, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2508.11262)  

**Abstract**: Vision-language models (VLM) align images and text in a shared representation space that is useful for retrieval and zero-shot transfer. Yet, this alignment can encode and amplify social stereotypes in subtle ways that are not obvious from standard accuracy metrics. In this study, we test whether the contrastive vision-language encoder exhibits gender-linked associations when it places embeddings of face images near embeddings of short phrases that describe occupations and activities. We assemble a dataset of 220 face photographs split by perceived binary gender and a set of 150 unique statements distributed across six categories covering emotional labor, cognitive labor, domestic labor, technical labor, professional roles, and physical labor. We compute unit-norm image embeddings for every face and unit-norm text embeddings for every statement, then define a statement-level association score as the difference between the mean cosine similarity to the male set and the mean cosine similarity to the female set, where positive values indicate stronger association with the male set and negative values indicate stronger association with the female set. We attach bootstrap confidence intervals by resampling images within each gender group, aggregate by category with a separate bootstrap over statements, and run a label-swap null model that estimates the level of mean absolute association we would expect if no gender structure were present. The outcome is a statement-wise and category-wise map of gender associations in a contrastive vision-language space, accompanied by uncertainty, simple sanity checks, and a robust gender bias evaluation framework. 

**Abstract (ZH)**: 视觉语言模型中的性别关联研究：基于对比的视觉-语言编码在职业和活动中体现的性别偏见分析 

---
# Generalized Decoupled Learning for Enhancing Open-Vocabulary Dense Perception 

**Title (ZH)**: 增强开放式词汇密集感知的通用解耦学习 

**Authors**: Junjie Wang, Keyu Chen, Yulin Li, Bin Chen, Hengshuang Zhao, Xiaojuan Qi, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.11256)  

**Abstract**: Dense visual perception tasks have been constrained by their reliance on predefined categories, limiting their applicability in real-world scenarios where visual concepts are unbounded. While Vision-Language Models (VLMs) like CLIP have shown promise in open-vocabulary tasks, their direct application to dense perception often leads to suboptimal performance due to limitations in local feature representation. In this work, we present our observation that CLIP's image tokens struggle to effectively aggregate information from spatially or semantically related regions, resulting in features that lack local discriminability and spatial consistency. To address this issue, we propose DeCLIP, a novel framework that enhances CLIP by decoupling the self-attention module to obtain ``content'' and ``context'' features respectively. \revise{The context features are enhanced by jointly distilling semantic correlations from Vision Foundation Models (VFMs) and object integrity cues from diffusion models, thereby enhancing spatial consistency. In parallel, the content features are aligned with image crop representations and constrained by region correlations from VFMs to improve local discriminability. Extensive experiments demonstrate that DeCLIP establishes a solid foundation for open-vocabulary dense perception, consistently achieving state-of-the-art performance across a broad spectrum of tasks, including 2D detection and segmentation, 3D instance segmentation, video instance segmentation, and 6D object pose estimation.} Code is available at this https URL 

**Abstract (ZH)**: 密集视觉感知任务受制于其对预定义类别的依赖，限制了其在视觉概念不受限的现实场景中的应用。尽管Vision-Language模型（VLMs）如CLIP在开放词典任务中表现出潜力，但它们在直接应用于密集感知时由于局部特征表示的限制往往会导致性能不佳。在本文中，我们观察到CLIP的图像标记难以有效地聚集来自空间上或语义上相关区域的信息，导致生成的特征缺乏局部区分性和空间一致性。为了解决这一问题，我们提出了一种名为DeCLIP的新型框架，通过解耦自我注意力模块来分别提取“内容”和“上下文”特征。上下文特征通过联合从视觉基础模型（VFMs）中提取的语义相关性和从扩散模型中提取的对象完整性线索来增强，从而提高空间一致性。同时，内容特征与图像剪辑表示对齐，并受到VFMs区域相关性的约束以提高局部区分性。广泛的实验表明，DeCLIP为开放词典的密集感知奠定了坚实的基础，能够在包括二维检测和分割、三维实例分割、视频实例分割和六维物体姿态估计等多种任务中实现现有最佳性能。代码可在以下链接获取。 

---
# StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation 

**Title (ZH)**: StyleMM：基于文本驱动对齐图像转换的样式化3D可变形面部模型 

**Authors**: Seungmi Lee, Kwan Yun, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2508.11203)  

**Abstract**: We introduce StyleMM, a novel framework that can construct a stylized 3D Morphable Model (3DMM) based on user-defined text descriptions specifying a target style. Building upon a pre-trained mesh deformation network and a texture generator for original 3DMM-based realistic human faces, our approach fine-tunes these models using stylized facial images generated via text-guided image-to-image (i2i) translation with a diffusion model, which serve as stylization targets for the rendered mesh. To prevent undesired changes in identity, facial alignment, or expressions during i2i translation, we introduce a stylization method that explicitly preserves the facial attributes of the source image. By maintaining these critical attributes during image stylization, the proposed approach ensures consistent 3D style transfer across the 3DMM parameter space through image-based training. Once trained, StyleMM enables feed-forward generation of stylized face meshes with explicit control over shape, expression, and texture parameters, producing meshes with consistent vertex connectivity and animatability. Quantitative and qualitative evaluations demonstrate that our approach outperforms state-of-the-art methods in terms of identity-level facial diversity and stylization capability. The code and videos are available at [this http URL](this http URL). 

**Abstract (ZH)**: 我们介绍了一种新型框架StyleMM，可以根据用户定义的文本描述构建目标风格化的3D可变形模型（3DMM）。该方法基于预训练的网格变形网络和原始3DMM驱动的真实人类面部纹理生成器，通过文本引导的图像到图像（i2i）转换生成风格化面部图像调整这些模型，从而将目标风格化图像作为渲染网格的风格化目标。为了防止在i2i转换过程中发生不希望的身份、面部对齐或表情变化，我们引入了一种显式保留源图像面部属性的风格化方法。通过在图像基础上保持这些关键属性，所提出的方法确保在3DMM参数空间中实现一致的3D风格迁移。经过训练后，StyleMM可以以明确控制形状、表情和纹理参数的方式生成风格化的面部网格，生成具有一致顶点连接性和可动画性的网格。定量和定性的评估表明，与现有方法相比，我们的方法在身份级别面部多样性和风格化能力方面表现更优。代码和视频可在[该网址](该网址)获取。 

---
# LD-LAudio-V1: Video-to-Long-Form-Audio Generation Extension with Dual Lightweight Adapters 

**Title (ZH)**: LD-LAudio-V1: 基于双轻量级适配器的视频到长音频生成扩展 

**Authors**: Haomin Zhang, Kristin Qi, Shuxin Yang, Zihao Chen, Chaofan Ding, Xinhan Di  

**Link**: [PDF](https://arxiv.org/pdf/2508.11074)  

**Abstract**: Generating high-quality and temporally synchronized audio from video content is essential for video editing and post-production tasks, enabling the creation of semantically aligned audio for silent videos. However, most existing approaches focus on short-form audio generation for video segments under 10 seconds or rely on noisy datasets for long-form video-to-audio zsynthesis. To address these limitations, we introduce LD-LAudio-V1, an extension of state-of-the-art video-to-audio models and it incorporates dual lightweight adapters to enable long-form audio generation. In addition, we release a clean and human-annotated video-to-audio dataset that contains pure sound effects without noise or artifacts. Our method significantly reduces splicing artifacts and temporal inconsistencies while maintaining computational efficiency. Compared to direct fine-tuning with short training videos, LD-LAudio-V1 achieves significant improvements across multiple metrics: $FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%), $FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%), $FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%), $KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%), $KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%), $IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%), $IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%), $Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%), $Energy\Delta10\text{ms(this http URL)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%), and $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%). Our dataset aims to facilitate further research in long-form video-to-audio generation and is available at this https URL. 

**Abstract (ZH)**: 从视频生成高质量且时间上同步的音频对于视频编辑和后期制作任务至关重要，能够用于创建无声视频的语义对齐音频。然而，大多数现有方法专注于生成长度不超过10秒的视频片段中的短音频，或者依赖嘈杂的数据集进行长视频到音频的合成。为解决这些局限性，我们引入了LD-LAudio-V1，这是一种扩展现有的先进视频到音频模型的方法，并且它结合了双轻量级适配器以实现长视频音频的生成。此外，我们还发布了一个人工标注且纯净的视频到音频数据集，其中包含纯声音效果而无噪声或伪影。我们的方法在保持计算效率的同时显著减少了拼接伪影和时间不一致性。与直接使用短训练视频进行微调相比，LD-LAudio-V1 在多个指标上实现了显著改进：$FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%)，$FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%)，$FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%)，$KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%)，$KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%)，$IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%)，$IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%)，$Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%)，$Energy\Delta10\text{ms(this http URL)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%)，和 $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%)。我们的数据集旨在促进长视频到音频生成的研究，并可在以下链接获取：this https URL。 

---
# Deep Learning-Based Automated Segmentation of Uterine Myomas 

**Title (ZH)**: 基于深度学习的子宫肌瘤自动分割 

**Authors**: Tausifa Jan Saleem, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2508.11010)  

**Abstract**: Uterine fibroids (myomas) are the most common benign tumors of the female reproductive system, particularly among women of childbearing age. With a prevalence exceeding 70%, they pose a significant burden on female reproductive health. Clinical symptoms such as abnormal uterine bleeding, infertility, pelvic pain, and pressure-related discomfort play a crucial role in guiding treatment decisions, which are largely influenced by the size, number, and anatomical location of the fibroids. Magnetic Resonance Imaging (MRI) is a non-invasive and highly accurate imaging modality commonly used by clinicians for the diagnosis of uterine fibroids. Segmenting uterine fibroids requires a precise assessment of both the uterus and fibroids on MRI scans, including measurements of volume, shape, and spatial location. However, this process is labor intensive and time consuming and subjected to variability due to intra- and inter-expert differences at both pre- and post-treatment stages. As a result, there is a critical need for an accurate and automated segmentation method for uterine fibroids. In recent years, deep learning algorithms have shown re-markable improvements in medical image segmentation, outperforming traditional methods. These approaches offer the potential for fully automated segmentation. Several studies have explored the use of deep learning models to achieve automated segmentation of uterine fibroids. However, most of the previous work has been conducted using private datasets, which poses challenges for validation and comparison between studies. In this study, we leverage the publicly available Uterine Myoma MRI Dataset (UMD) to establish a baseline for automated segmentation of uterine fibroids, enabling standardized evaluation and facilitating future research in this domain. 

**Abstract (ZH)**: 子宫肌瘤（纤维瘤）是女性生殖系统中最常见的良性肿瘤，尤其在育龄妇女中更为常见。其发病率超过70%，对女性生殖健康构成了显著的负担。临床症状如异常子宫出血、不孕、盆腔疼痛和压迫感相关的不适，在指导治疗决策中起着关键作用，这些决策很大程度上受到肌瘤大小、数量和解剖位置的影响。磁共振成像（MRI）是一种常用的无创且高度准确的成像技术，用于子宫肌瘤的诊断。子宫肌瘤的分割需要对MRI扫描中的子宫和肌瘤进行精确的评估，包括体积、形状和空间位置的测量。然而，这一过程耗时且主观性高，受到预治疗和治疗后专家间差异的影响。因此，需要一种准确且自动化的分割方法来解决这些问题。近年来，深度学习算法在医学图像分割方面取得了显著进步，超越了传统方法。这些方法提供了实现完全自动化分割的潜力。多项研究探讨了使用深度学习模型实现子宫肌瘤自动化分割的可能性。然而，大多数先前的工作都基于私有数据集，这给跨研究的验证和比较带来了挑战。在本研究中，我们利用公开的子宫肌瘤MRI数据集（UMD）建立子宫肌瘤自动化分割的基线，以实现标准化评估并促进该领域的未来研究。 

---
# ORBIT: An Object Property Reasoning Benchmark for Visual Inference Tasks 

**Title (ZH)**: ORBIT: 一种用于视觉推理任务的对象属性推理基准 

**Authors**: Abhishek Kolari, Mohammadhossein Khojasteh, Yifan Jiang, Floris den Hengst, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2508.10956)  

**Abstract**: While vision-language models (VLMs) have made remarkable progress on many popular visual question answering (VQA) benchmarks, it remains unclear whether they abstract and reason over depicted objects. Inspired by human object categorisation, object property reasoning involves identifying and recognising low-level details and higher-level abstractions. While current VQA benchmarks consider a limited set of object property attributes like size, they typically blend perception and reasoning, and lack representativeness in terms of reasoning and image categories. To this end, we introduce a systematic evaluation framework with images of three representative types, three reasoning levels of increasing complexity, and four object property dimensions driven by prior work on commonsense reasoning. We develop a procedure to instantiate this benchmark into ORBIT, a multi-level reasoning VQA benchmark for object properties comprising 360 images paired with a total of 1,080 count-based questions. Experiments with 12 state-of-the-art VLMs in zero-shot settings reveal significant limitations compared to humans, with the best-performing model only reaching 40\% accuracy. VLMs struggle particularly with realistic (photographic) images, counterfactual reasoning about physical and functional properties, and higher counts. ORBIT points to the need to develop methods for scalable benchmarking, generalize annotation guidelines, and explore additional reasoning VLMs. We make the ORBIT benchmark and the experimental code available to support such endeavors. 

**Abstract (ZH)**: 基于视觉-语言模型的对象属性推理综合评估框架ORBIT 

---
