# A Training-Free Framework for Precise Mobile Manipulation of Small Everyday Objects 

**Title (ZH)**: 无需训练的精确移动小型日常物体框架 

**Authors**: Arjun Gupta, Rishik Sathua, Saurabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.13964)  

**Abstract**: Many everyday mobile manipulation tasks require precise interaction with small objects, such as grasping a knob to open a cabinet or pressing a light switch. In this paper, we develop Servoing with Vision Models (SVM), a closed-loop training-free framework that enables a mobile manipulator to tackle such precise tasks involving the manipulation of small objects. SVM employs an RGB-D wrist camera and uses visual servoing for control. Our novelty lies in the use of state-of-the-art vision models to reliably compute 3D targets from the wrist image for diverse tasks and under occlusion due to the end-effector. To mitigate occlusion artifacts, we employ vision models to out-paint the end-effector thereby significantly enhancing target localization. We demonstrate that aided by out-painting methods, open-vocabulary object detectors can serve as a drop-in module to identify semantic targets (e.g. knobs) and point tracking methods can reliably track interaction sites indicated by user clicks. This training-free method obtains an 85% zero-shot success rate on manipulating unseen objects in novel environments in the real world, outperforming an open-loop control method and an imitation learning baseline trained on 1000+ demonstrations by an absolute success rate of 50%. 

**Abstract (ZH)**: 基于视觉模型的伺服控制（Servoing with Vision Models）：一种无需训练的闭环框架，用于移动 manipulator 精准操作小型物体的任务 

---
# The NavINST Dataset for Multi-Sensor Autonomous Navigation 

**Title (ZH)**: NavINST 数据集：多传感器自主导航 

**Authors**: Paulo Ricardo Marques de Araujo, Eslam Mounier, Qamar Bader, Emma Dawson, Shaza I. Kaoud Abdelaziz, Ahmed Zekry, Mohamed Elhabiby, Aboelmagd Noureldin  

**Link**: [PDF](https://arxiv.org/pdf/2502.13863)  

**Abstract**: The NavINST Laboratory has developed a comprehensive multisensory dataset from various road-test trajectories in urban environments, featuring diverse lighting conditions, including indoor garage scenarios with dense 3D maps. This dataset includes multiple commercial-grade IMUs and a high-end tactical-grade IMU. Additionally, it contains a wide array of perception-based sensors, such as a solid-state LiDAR - making it one of the first datasets to do so - a mechanical LiDAR, four electronically scanning RADARs, a monocular camera, and two stereo cameras. The dataset also includes forward speed measurements derived from the vehicle's odometer, along with accurately post-processed high-end GNSS/IMU data, providing precise ground truth positioning and navigation information. The NavINST dataset is designed to support advanced research in high-precision positioning, navigation, mapping, computer vision, and multisensory fusion. It offers rich, multi-sensor data ideal for developing and validating robust algorithms for autonomous vehicles. Finally, it is fully integrated with the ROS, ensuring ease of use and accessibility for the research community. The complete dataset and development tools are available at this https URL. 

**Abstract (ZH)**: NavINST实验室开发了针对城市环境道路测试轨迹的综合性多传感数据集，涵盖了多种照明条件，包括具有密集3D地图的室内车库场景。该数据集包含多种商业级IMU和高端战术级IMU。此外，它还包含多种基于感知的传感器，如固态LiDAR（这是首次包含此类传感器的数据集）、机械LiDAR、四台电子扫描雷达、单目相机和两台立体相机。该数据集还包括来自车辆 odometer 的前向速度测量值，以及经过高精度后处理的GNSS/IMU数据，提供精确的姿态和导航信息。NavINST数据集旨在支持高精度定位、导航、制图、计算机视觉和多传感融合的高级研究。它提供了丰富的多传感器数据，适用于开发和验证适用于自动驾驶车辆的稳健算法。最后，该数据集与ROS完全集成，确保研究社区的易用性和访问性。完整数据集和开发工具可在此处访问：https://xxx.xxx.xxx 

---
# Improving Collision-Free Success Rate For Object Goal Visual Navigation Via Two-Stage Training With Collision Prediction 

**Title (ZH)**: 通过两阶段训练与碰撞预测改进物体目标视觉导航的无碰撞成功率 

**Authors**: Shiwei Lian, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13498)  

**Abstract**: The object goal visual navigation is the task of navigating to a specific target object using egocentric visual observations. Recent end-to-end navigation models based on deep reinforcement learning have achieved remarkable performance in finding and reaching target objects. However, the collision problem of these models during navigation remains unresolved, since the collision is typically neglected when evaluating the success. Although incorporating a negative reward for collision during training appears straightforward, it results in a more conservative policy, thereby limiting the agent's ability to reach targets. In addition, many of these models utilize only RGB observations, further increasing the difficulty of collision avoidance without depth information. To address these limitations, a new concept -- collision-free success is introduced to evaluate the ability of navigation models to find a collision-free path towards the target object. A two-stage training method with collision prediction is proposed to improve the collision-free success rate of the existing navigation models using RGB observations. In the first training stage, the collision prediction module supervises the agent's collision states during exploration to learn to predict the possible collision. In the second stage, leveraging the trained collision prediction, the agent learns to navigate to the target without collision. The experimental results in the AI2-THOR environment demonstrate that the proposed method greatly improves the collision-free success rate of different navigation models and outperforms other comparable collision-avoidance methods. 

**Abstract (ZH)**: 基于自我中心视觉观察的目标视觉导航是导航至特定目标物体的任务。基于深度强化学习的端到端导航模型在寻找和到达目标物体方面取得了显著性能。然而，这些模型在导航过程中回避碰撞的问题仍未解决，因为在评估成功时通常会忽略碰撞。尽管在训练中引入碰撞的负奖励看起来很简单，但实际上会导致更加保守的策略，从而限制了智能体到达目标的能力。此外，许多这些模型仅使用RGB观察，进一步增加了在缺乏深度信息的情况下避免碰撞的难度。为了解决这些局限性，引入了一种新的概念——碰撞-free成功，以评估导航模型在寻找通向目标物体的碰撞-free路径方面的能力。提出了一种两阶段训练方法，结合碰撞预测，以提高使用RGB观察的现有导航模型的碰撞-free成功率。在第一阶段训练中，碰撞预测模块监督智能体在探索过程中的碰撞状态，以学习预测可能的碰撞。在第二阶段，利用训练好的碰撞预测，智能体学习如何在无碰撞的情况下导航至目标。在AI2-THOR环境中的实验结果表明，所提出的方法显著提高了不同导航模型的碰撞-free成功率，并优于其他可比的碰撞避免方法。 

---
# Object-Pose Estimation With Neural Population Codes 

**Title (ZH)**: 基于神经群体码的物体姿态估计 

**Authors**: Heiko Hoffmann, Richard Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.13403)  

**Abstract**: Robotic assembly tasks require object-pose estimation, particularly for tasks that avoid costly mechanical constraints. Object symmetry complicates the direct mapping of sensory input to object rotation, as the rotation becomes ambiguous and lacks a unique training target. Some proposed solutions involve evaluating multiple pose hypotheses against the input or predicting a probability distribution, but these approaches suffer from significant computational overhead. Here, we show that representing object rotation with a neural population code overcomes these limitations, enabling a direct mapping to rotation and end-to-end learning. As a result, population codes facilitate fast and accurate pose estimation. On the T-LESS dataset, we achieve inference in 3.2 milliseconds on an Apple M1 CPU and a Maximum Symmetry-Aware Surface Distance accuracy of 84.7% using only gray-scale image input, compared to 69.7% accuracy when directly mapping to pose. 

**Abstract (ZH)**: 机器人装配任务需要物体姿态估计，特别是在避免昂贵的机械约束的情况下。物体的对称性使得直接将感觉输入映射到物体旋转变得复杂，因为旋转变得模糊不清，缺乏唯一的训练目标。一些拟议的解决方案涉及评估多个姿态假设与输入，或预测概率分布，但这些方法遭受显著的计算开销。在这里，我们展示了使用神经群体编码表示物体旋转克服了这些限制，实现了直接映射到旋转，并实现了端到端学习。因此，群体编码促进了快速准确的姿态估计。在T-LESS数据集上，我们仅使用灰度图像输入在Apple M1 CPU上实现3.2毫秒的推理，并且使用最大对称性意识表面距离准确率达到84.7%，而直接映射到姿态的准确率为69.7%。 

---
# 3D Gaussian Splatting aided Localization for Large and Complex Indoor-Environments 

**Title (ZH)**: 基于3D高斯点云辅助的大型复杂室内环境定位 

**Authors**: Vincent Ress, Jonas Meyer, Wei Zhang, David Skuddis, Uwe Soergel, Norbert Haala  

**Link**: [PDF](https://arxiv.org/pdf/2502.13803)  

**Abstract**: The field of visual localization has been researched for several decades and has meanwhile found many practical applications. Despite the strong progress in this field, there are still challenging situations in which established methods fail. We present an approach to significantly improve the accuracy and reliability of established visual localization methods by adding rendered images. In detail, we first use a modern visual SLAM approach that provides a 3D Gaussian Splatting (3DGS) based map to create reference data. We demonstrate that enriching reference data with images rendered from 3DGS at randomly sampled poses significantly improves the performance of both geometry-based visual localization and Scene Coordinate Regression (SCR) methods. Through comprehensive evaluation in a large industrial environment, we analyze the performance impact of incorporating these additional rendered views. 

**Abstract (ZH)**: 视觉定位领域的研究已有数十年的历史，并已在许多实际应用中找到应用。尽管该领域取得了显著进展，但仍存在某些情况下现有方法失效的挑战性场景。我们提出了一种通过添加渲染图像显著提高现有视觉定位方法的准确性和可靠性的方法。具体而言，我们首先使用一种现代的视觉SLAM方法生成基于3D高斯喷洒（3DGS）的地图以创建参考数据。我们证明，通过在随机抽取的姿态下，使用3DGS渲染图像来丰富参考数据，可以显著提高基于几何的视觉定位方法和场景坐标回归（SCR）方法的性能。通过在大型工业环境中进行全面评估，我们分析了引入这些额外渲染视图对性能的影响。 

---
# Continually Learning Structured Visual Representations via Network Refinement with Rerelation 

**Title (ZH)**: 持续学习结构化视觉表示通过关系网络精炼 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2502.13935)  

**Abstract**: Current machine learning paradigm relies on continuous representations like neural networks, which iteratively adjust parameters to approximate outcomes rather than directly learning the structure of problem. This spreads information across the network, causing issues like information loss and incomprehensibility Building on prior work in environment dynamics modeling, we propose a method that learns visual space in a structured, continual manner. Our approach refines networks to capture the core structure of objects while representing significant subvariants in structure efficiently. We demonstrate this with 2D shape detection, showing incremental learning on MNIST without overwriting knowledge and creating compact, comprehensible representations. These results offer a promising step toward a transparent, continually learning alternative to traditional neural networks for visual processing. 

**Abstract (ZH)**: 当前的机器学习范式依赖于神经网络等连续表示，通过迭代调整参数来逼近结果，而不是直接学习问题的结构。这种方法在网络中传播信息，导致信息丢失和不可理解性等问题。基于先前在环境动力学建模方面的研究，我们提出了一种在结构化、连续方式下学习视觉空间的方法。我们的方法 refinements 网络以捕捉对象的核心结构，同时有效地表示结构上的显著亚变体。我们通过2D形状检测展示了这一点，展示了在MNIST上进行增量学习而不覆盖知识，并生成紧凑且易于理解的表示。这些结果为透明的、连续学习的传统神经网络替代方案提供了有希望的步骤，用于视觉处理。 

---
# Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images 

**Title (ZH)**: 对称视觉对比优化：通过最少的对比图像对齐视觉语言模型 

**Authors**: Shengguang Wu, Fan-Yun Sun, Kaiyue Wen, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2502.13928)  

**Abstract**: Recent studies have shown that Large Vision-Language Models (VLMs) tend to neglect image content and over-rely on language-model priors, resulting in errors in visually grounded tasks and hallucinations. We hypothesize that this issue arises because existing VLMs are not explicitly trained to generate texts that are accurately grounded in fine-grained image details. To enhance visual feedback during VLM training, we propose S-VCO (Symmetrical Visual Contrastive Optimization), a novel finetuning objective that steers the model toward capturing important visual details and aligning them with corresponding text tokens. To further facilitate this detailed alignment, we introduce MVC, a paired image-text dataset built by automatically filtering and augmenting visual counterfactual data to challenge the model with hard contrastive cases involving Minimal Visual Contrasts. Experiments show that our method consistently improves VLM performance across diverse benchmarks covering various abilities and domains, achieving up to a 22% reduction in hallucinations, and significant gains in vision-centric and general tasks. Notably, these improvements become increasingly pronounced in benchmarks with higher visual dependency. In short, S-VCO offers a significant enhancement of VLM's visually-dependent task performance while retaining or even improving the model's general abilities. We opensource our code at this https URL 

**Abstract (ZH)**: Recent studies have shown that Large Vision-Language Models (VLMs) tend to neglect image content and over-rely on language-model priors, resulting in errors in visually grounded tasks and hallucinations. We hypothesize that this issue arises because existing VLMs are not explicitly trained to generate texts that are accurately grounded in fine-grained image details. To enhance visual feedback during VLM training, we propose S-VCO (Symmetrical Visual Contrastive Optimization), a novel finetuning objective that steers the model toward capturing important visual details and aligning them with corresponding text tokens. To further facilitate this detailed alignment, we introduce MVC, a paired image-text dataset built by automatically filtering and augmenting visual counterfactual data to challenge the model with hard contrastive cases involving Minimal Visual Contrasts. Experiments show that our method consistently improves VLM performance across diverse benchmarks covering various abilities and domains, achieving up to a 22% reduction in hallucinations, and significant gains in vision-centric and general tasks. Notably, these improvements become increasingly pronounced in benchmarks with higher visual dependency. In short, S-VCO offers a significant enhancement of VLM's visually-dependent task performance while retaining or even improving the model's general abilities. 我们开源了我们的代码，地址为：this https URL。 

---
# MEX: Memory-efficient Approach to Referring Multi-Object Tracking 

**Title (ZH)**: MEX: MEMORY-EFFICIENT APPROACH TO REFERRING MULTI-OBJECT TRACKING 

**Authors**: Huu-Thien Tran, Phuoc-Sang Pham, Thai-Son Tran, Khoa Luu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13875)  

**Abstract**: Referring Multi-Object Tracking (RMOT) is a relatively new concept that has rapidly gained traction as a promising research direction at the intersection of computer vision and natural language processing. Unlike traditional multi-object tracking, RMOT identifies and tracks objects and incorporates textual descriptions for object class names, making the approach more intuitive. Various techniques have been proposed to address this challenging problem; however, most require the training of the entire network due to their end-to-end nature. Among these methods, iKUN has emerged as a particularly promising solution. Therefore, we further explore its pipeline and enhance its performance. In this paper, we introduce a practical module dubbed Memory-Efficient Cross-modality -- MEX. This memory-efficient technique can be directly applied to off-the-shelf trackers like iKUN, resulting in significant architectural improvements. Our method proves effective during inference on a single GPU with 4 GB of memory. Among the various benchmarks, the Refer-KITTI dataset, which offers diverse autonomous driving scenes with relevant language expressions, is particularly useful for studying this problem. Empirically, our method demonstrates effectiveness and efficiency regarding HOTA tracking scores, substantially improving memory allocation and processing speed. 

**Abstract (ZH)**: 基于参考的多对象跟踪（RMOT）是一种相对较新的概念，作为计算机视觉和自然语言处理交叉领域的有前途的研究方向，正迅速获得关注。RMOT不同于传统的多对象跟踪，它能够识别和跟踪对象，并结合文本描述以提高对象类别名称的可理解性。为了解决这一具有挑战性的问题，提出了多种技术；然而，大多数方法由于是端到端的，需要重新训练整个网络。在这之中，iKUN 成为了一个特别有前景的解决方案。因此，我们进一步探索其管道并改进其性能。在本文中，我们引入了一个名为Memory-Efficient Cross-modality -- MEX的实用模块。这项内存效率高的技术可以直接应用于现成的跟踪器，如iKUN，从而带来架构上的显著改进。我们的方法在单个带有4 GB内存的GPU上进行推断时证明是有效的。特别是，Refer-KITTI数据集因其提供多种多样的自动驾驶场景及相关的语言表达，对于研究这一问题特别有用。我们的方法在HOTA跟踪评分上表现出有效性与效率，并显著改善了内存分配和处理速度。 

---
# An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice 

**Title (ZH)**: 实时分类与质量评估综合机制 for 米类 

**Authors**: Wanke Xia, Ruxin Peng, Haoqi Chu, Xinlei Zhu, Zhiyu Yang, Yaojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13764)  

**Abstract**: Rice is one of the most widely cultivated crops globally and has been developed into numerous varieties. The quality of rice during cultivation is primarily determined by its cultivar and characteristics. Traditionally, rice classification and quality assessment rely on manual visual inspection, a process that is both time-consuming and prone to errors. However, with advancements in machine vision technology, automating rice classification and quality evaluation based on its cultivar and characteristics has become increasingly feasible, enhancing both accuracy and efficiency. This study proposes a real-time evaluation mechanism for comprehensive rice grain assessment, integrating a one-stage object detection approach, a deep convolutional neural network, and traditional machine learning techniques. The proposed framework enables rice variety identification, grain completeness grading, and grain chalkiness evaluation. The rice grain dataset used in this study comprises approximately 20,000 images from six widely cultivated rice varieties in China. Experimental results demonstrate that the proposed mechanism achieves a mean average precision (mAP) of 99.14% in the object detection task and an accuracy of 97.89% in the classification task. Furthermore, the framework attains an average accuracy of 97.56% in grain completeness grading within the same rice variety, contributing to an effective quality evaluation system. 

**Abstract (ZH)**: 全球种植最为广泛的稻谷作物之一已发展出众多品种。稻谷在栽培期间的质量主要由其品种和特征决定。传统上，稻谷分类和质量评估依赖手工视觉检查，这一过程既耗时又容易出错。然而，随着机器视觉技术的进步，基于品种和特征自动进行稻谷分类和质量评价变得日益可行，提高了准确性和效率。本研究提出了一种实时的综合稻谷粒品质评价机制，结合了一阶段物体检测方法、深度卷积神经网络和传统的机器学习技术。所提出的框架能够实现稻谷品种识别、籽粒完整度分级和籽粒垩白评价。本研究使用的稻谷粒数据集包含来自中国广泛种植的六种水稻品种的约20,000张图像。实验结果表明，所提出机制在物体检测任务中的平均平均精确度（mAP）为99.14%，分类任务的准确率为97.89%。此外，该框架在同一水稻品种中的籽粒完整度分级的平均准确率为97.56%，有助于构建有效的质量评价系统。 

---
# MobileViM: A Light-weight and Dimension-independent Vision Mamba for 3D Medical Image Analysis 

**Title (ZH)**: MobileViM: 一种轻量级且维度无关的医学图像分析视觉章鱼 

**Authors**: Wei Dai, Steven Wang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13524)  

**Abstract**: Efficient evaluation of three-dimensional (3D) medical images is crucial for diagnostic and therapeutic practices in healthcare. Recent years have seen a substantial uptake in applying deep learning and computer vision to analyse and interpret medical images. Traditional approaches, such as convolutional neural networks (CNNs) and vision transformers (ViTs), face significant computational challenges, prompting the need for architectural advancements. Recent efforts have led to the introduction of novel architectures like the ``Mamba'' model as alternative solutions to traditional CNNs or ViTs. The Mamba model excels in the linear processing of one-dimensional data with low computational demands. However, Mamba's potential for 3D medical image analysis remains underexplored and could face significant computational challenges as the dimension increases. This manuscript presents MobileViM, a streamlined architecture for efficient segmentation of 3D medical images. In the MobileViM network, we invent a new dimension-independent mechanism and a dual-direction traversing approach to incorporate with a vision-Mamba-based framework. MobileViM also features a cross-scale bridging technique to improve efficiency and accuracy across various medical imaging modalities. With these enhancements, MobileViM achieves segmentation speeds exceeding 90 frames per second (FPS) on a single graphics processing unit (i.e., NVIDIA RTX 4090). This performance is over 24 FPS faster than the state-of-the-art deep learning models for processing 3D images with the same computational resources. In addition, experimental evaluations demonstrate that MobileViM delivers superior performance, with Dice similarity scores reaching 92.72%, 86.69%, 80.46%, and 77.43% for PENGWIN, BraTS2024, ATLAS, and Toothfairy2 datasets, respectively, which significantly surpasses existing models. 

**Abstract (ZH)**: 高效的三维医学图像评价对于医疗诊断和治疗至关重要。近年来，深度学习和计算机视觉在医学图像分析和解释中的应用取得了显著增长。传统的卷积神经网络(CNNs)和视觉变换器(ViTs)面临着显著的计算挑战，促使架构上的创新。最近的研究引入了新型架构“Mamba”模型作为传统CNNs或ViTs的替代方案。Mamba模型在处理一维数据方面表现出卓越的能力，并且具有较低的计算需求。然而，Mamba在三维医学图像分析中的潜力尚未被充分探索，随着维度的增加，可能会面临显著的计算挑战。本文提出了MobileViM，这是一种针对三维医学图像高效分割的精简架构。在MobileViM网络中，我们提出了一种新的维数独立机制和双向遍历方法，并结合了基于视觉-Mamba的框架。MobileViM还包含了一种跨尺度桥梁技术，以提高不同医学成像模态下的效率和准确性。通过这些增强，MobileViM在单个图形处理单元(NVIDIA RTX 4090)上实现了超过90帧每秒(FPS)的分割速度，比相同的计算资源下最先进的深度学习模型快24 FPS以上。此外，实验评估表明，MobileViM在PENGWIN、BraTS2024、ATLAS和Toothfairy2数据集中分别达到了92.72%、86.69%、80.46%和77.43%的Dice相似度分数，显著优于现有模型。 

---
# Semi-supervised classification of bird vocalizations 

**Title (ZH)**: 半监督鸟类鸣声分类 

**Authors**: Simen Hexeberg, Mandar Chitre, Matthias Hoffmann-Kuhnt, Bing Wen Low  

**Link**: [PDF](https://arxiv.org/pdf/2502.13440)  

**Abstract**: Changes in bird populations can indicate broader changes in ecosystems, making birds one of the most important animal groups to monitor. Combining machine learning and passive acoustics enables continuous monitoring over extended periods without direct human involvement. However, most existing techniques require extensive expert-labeled datasets for training and cannot easily detect time-overlapping calls in busy soundscapes. We propose a semi-supervised acoustic bird detector designed to allow both the detection of time-overlapping calls (when separated in frequency) and the use of few labeled training samples. The classifier is trained and evaluated on a combination of community-recorded open-source data and long-duration soundscape recordings from Singapore. It achieves a mean F0.5 score of 0.701 across 315 classes from 110 bird species on a hold-out test set, with an average of 11 labeled training samples per class. It outperforms the state-of-the-art BirdNET classifier on a test set of 103 bird species despite significantly fewer labeled training samples. The detector is further tested on 144 microphone-hours of continuous soundscape data. The rich soundscape in Singapore makes suppression of false positives a challenge on raw, continuous data streams. Nevertheless, we demonstrate that achieving high precision in such environments with minimal labeled training data is possible. 

**Abstract (ZH)**: 鸟类种群变化可以指示生态系统更为广泛的改变，使鸟类成为最重要的监测动物群之一。结合机器学习和被动声学能够在长时间内实现无需直接人类干预的连续监测。然而，大多数现有技术需要大量的专家标注数据集进行训练，并且难以在繁忙的声音景观中检测到时间重叠的声音。我们提出了一种半监督声学鸟类检测器，能够检测在频率上分离的时间重叠叫声，并且仅使用少量标注训练样本。分类器在组合了社区录制的开源数据和新加坡长达数小时的声音景观录音数据上进行训练和评估。该检测器在排除外部测试集的315个类别中（来自110种鸟类）实现了0.701的平均F0.5分数，平均每类有11个标注训练样本。尽管标注训练样本显著较少，但在103种鸟类的测试集上仍优于最先进的BirdNET分类器。该检测器进一步在连续声音景观数据上测试了144小时的麦克风录音。新加坡丰富的声音景观使得在原始连续数据流中抑制假阳性成为一个挑战。尽管如此，我们证明在这些环境中，使用最少的标注训练数据实现高精度是可能的。 

---
# MotionMatcher: Motion Customization of Text-to-Video Diffusion Models via Motion Feature Matching 

**Title (ZH)**: MotionMatcher：通过运动特征匹配的文本到视频扩散模型的运动自化 

**Authors**: Yen-Siang Wu, Chi-Pin Huang, Fu-En Yang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.13234)  

**Abstract**: Text-to-video (T2V) diffusion models have shown promising capabilities in synthesizing realistic videos from input text prompts. However, the input text description alone provides limited control over the precise objects movements and camera framing. In this work, we tackle the motion customization problem, where a reference video is provided as motion guidance. While most existing methods choose to fine-tune pre-trained diffusion models to reconstruct the frame differences of the reference video, we observe that such strategy suffer from content leakage from the reference video, and they cannot capture complex motion accurately. To address this issue, we propose MotionMatcher, a motion customization framework that fine-tunes the pre-trained T2V diffusion model at the feature level. Instead of using pixel-level objectives, MotionMatcher compares high-level, spatio-temporal motion features to fine-tune diffusion models, ensuring precise motion learning. For the sake of memory efficiency and accessibility, we utilize a pre-trained T2V diffusion model, which contains considerable prior knowledge about video motion, to compute these motion features. In our experiments, we demonstrate state-of-the-art motion customization performances, validating the design of our framework. 

**Abstract (ZH)**: 文本到视频（T2V）扩散模型展示了从输入文本提示合成逼真视频的潜力。然而，仅有的文本描述输入对精确对象运动和相机构图的控制有限。在本文中，我们解决了一种运动定制问题，提供了一个参考视频作为运动指导。尽管大多数现有方法选择微调预训练的扩散模型以重建参考视频的帧差异，但我们发现这种策略会从参考视频中泄露内容，并且无法准确捕捉复杂运动。为了解决这一问题，我们提出了一种运动匹配器（MotionMatcher）运动定制框架，该框架在特征级别微调预训练的T2V扩散模型。与其使用像素级目标，MotionMatcher比较高层次的空间-时间运动特征来微调扩散模型，确保精确的运动学习。为提高内存效率和可获得性，我们利用了一个含有大量视频运动先验知识的预训练T2V扩散模型来计算这些运动特征。在我们的实验中，我们展示了最先进的运动定制性能，验证了我们框架的设计。 

---
