# SparseLoc: Sparse Open-Set Landmark-based Global Localization for Autonomous Navigation 

**Title (ZH)**: SparseLoc: 稀疏开放集地标导向的全局定位方法用于自主导航 

**Authors**: Pranjal Paul, Vineeth Bhat, Tejas Salian, Mohammad Omama, Krishna Murthy Jatavallabhula, Naveen Arulselvan, K. Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2503.23465)  

**Abstract**: Global localization is a critical problem in autonomous navigation, enabling precise positioning without reliance on GPS. Modern global localization techniques often depend on dense LiDAR maps, which, while precise, require extensive storage and computational resources. Recent approaches have explored alternative methods, such as sparse maps and learned features, but they suffer from poor robustness and generalization. We propose SparseLoc, a global localization framework that leverages vision-language foundation models to generate sparse, semantic-topometric maps in a zero-shot manner. It combines this map representation with a Monte Carlo localization scheme enhanced by a novel late optimization strategy, ensuring improved pose estimation. By constructing compact yet highly discriminative maps and refining localization through a carefully designed optimization schedule, SparseLoc overcomes the limitations of existing techniques, offering a more efficient and robust solution for global localization. Our system achieves over a 5X improvement in localization accuracy compared to existing sparse mapping techniques. Despite utilizing only 1/500th of the points of dense mapping methods, it achieves comparable performance, maintaining an average global localization error below 5m and 2 degrees on KITTI sequences. 

**Abstract (ZH)**: 全球全局定位是自主导航中的一个关键问题，能够实现不依赖GPS的精确定位。现代全球全局定位技术通常依赖密集的激光雷达地图，虽然精确，但需要大量存储和计算资源。近年来的研究探索了替代方法，如稀疏地图和学习特征，但这些方法 robustness 和泛化能力较差。我们提出了 SparseLoc，这是一种利用视觉-语言基础模型生成稀疏语义-地形地图的零样本全局定位框架。该框架结合了增强的蒙特卡洛定位方案和一种新的后处理优化策略，确保了姿态估计的改进。通过构建紧凑且高度判别性的地图，并通过精心设计的优化计划进行定位细化，SparseLoc 克服了现有技术的局限性，提供了一种更具效率和 robust 性的全局定位解决方案。我们的系统在定位准确性方面相比现有稀疏映射技术提高了超过 5 倍。尽管仅使用密集映射方法的 1/500 个点，但在 KITTI 序列中实现了相近的性能，保持全局定位误差平均低于 5 米和 2 度。 

---
# Deep Visual Servoing of an Aerial Robot Using Keypoint Feature Extraction 

**Title (ZH)**: 使用关键点特征提取的空中机器人深度视觉伺服控制 

**Authors**: Shayan Sepahvand, Niloufar Amiri, Farrokh Janabi-Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23171)  

**Abstract**: The problem of image-based visual servoing (IBVS) of an aerial robot using deep-learning-based keypoint detection is addressed in this article. A monocular RGB camera mounted on the platform is utilized to collect the visual data. A convolutional neural network (CNN) is then employed to extract the features serving as the visual data for the servoing task. This paper contributes to the field by circumventing not only the challenge stemming from the need for man-made marker detection in conventional visual servoing techniques, but also enhancing the robustness against undesirable factors including occlusion, varying illumination, clutter, and background changes, thereby broadening the applicability of perception-guided motion control tasks in aerial robots. Additionally, extensive physics-based ROS Gazebo simulations are conducted to assess the effectiveness of this method, in contrast to many existing studies that rely solely on physics-less simulations. A demonstration video is available at this https URL. 

**Abstract (ZH)**: 基于深度学习关键点检测的 aerial 机器人图像视觉伺服问题研究 

---
# VLM-C4L: Continual Core Dataset Learning with Corner Case Optimization via Vision-Language Models for Autonomous Driving 

**Title (ZH)**: VLM-C4L：通过视觉语言模型在自动驾驶中基于边缘案例的持续核心数据集学习优化 

**Authors**: Haibo Hu, Jiacheng Zuo, Yang Lou, Yufei Cui, Jianping Wang, Nan Guan, Jin Wang, Yung-Hui Li, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.23046)  

**Abstract**: With the widespread adoption and deployment of autonomous driving, handling complex environments has become an unavoidable challenge. Due to the scarcity and diversity of extreme scenario datasets, current autonomous driving models struggle to effectively manage corner cases. This limitation poses a significant safety risk, according to the National Highway Traffic Safety Administration (NHTSA), autonomous vehicle systems have been involved in hundreds of reported crashes annually in the United States, occurred in corner cases like sun glare and fog, which caused a few fatal accident. Furthermore, in order to consistently maintain a robust and reliable autonomous driving system, it is essential for models not only to perform well on routine scenarios but also to adapt to newly emerging scenarios, especially those corner cases that deviate from the norm. This requires a learning mechanism that incrementally integrates new knowledge without degrading previously acquired capabilities. However, to the best of our knowledge, no existing continual learning methods have been proposed to ensure consistent and scalable corner case learning in autonomous driving. To address these limitations, we propose VLM-C4L, a continual learning framework that introduces Vision-Language Models (VLMs) to dynamically optimize and enhance corner case datasets, and VLM-C4L combines VLM-guided high-quality data extraction with a core data replay strategy, enabling the model to incrementally learn from diverse corner cases while preserving performance on previously routine scenarios, thus ensuring long-term stability and adaptability in real-world autonomous driving. We evaluate VLM-C4L on large-scale real-world autonomous driving datasets, including Waymo and the corner case dataset CODA. 

**Abstract (ZH)**: 基于视觉语言模型的持续学习框架VLM-C4L：面向自动驾驶的复杂corner case优化与适应 

---
# SR-LIO++: Efficient LiDAR-Inertial Odometry and Quantized Mapping with Sweep Reconstruction 

**Title (ZH)**: SR-LIO++: 高效的LiDAR-惯性里程计和基于扫掠重建的量化映射 

**Authors**: Zikang Yuan, Ruiye Ming, Chengwei Zhao, Yonghao Tan, Pingcheng Dong, Hongcheng Luo, Yuzhong Jiao, Xin Yang, Kwang-Ting Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22926)  

**Abstract**: Addressing the inherent low acquisition frequency limitation of 3D LiDAR to achieve high-frequency output has become a critical research focus in the LiDAR-Inertial Odometry (LIO) domain. To ensure real-time performance, frequency-enhanced LIO systems must process each sweep within significantly reduced timeframe, which presents substantial challenges for deployment on low-computational-power platforms. To address these limitations, we introduce SR-LIO++, an innovative LIO system capable of achieving doubled output frequency relative to input frequency on resource-constrained hardware platforms, including the Raspberry Pi 4B. Our system employs a sweep reconstruction methodology to enhance LiDAR sweep frequency, generating high-frequency reconstructed sweeps. Building upon this foundation, we propose a caching mechanism for intermediate results (i.e., surface parameters) of the most recent segments, effectively minimizing redundant processing of common segments in adjacent reconstructed sweeps. This method decouples processing time from the traditionally linear dependence on reconstructed sweep frequency. Furthermore, we present a quantized map point management based on index table mapping, significantly reducing memory usage by converting global 3D point storage from 64-bit double precision to 8-bit char representation. This method also converts the computationally intensive Euclidean distance calculations in nearest neighbor searches from 64-bit double precision to 16-bit short and 32-bit integer formats, significantly reducing both memory and computational cost. Extensive experimental evaluations across three distinct computing platforms and four public datasets demonstrate that SR-LIO++ maintains state-of-the-art accuracy while substantially enhancing efficiency. Notably, our system successfully achieves 20Hz state output on Raspberry Pi 4B hardware. 

**Abstract (ZH)**: 基于3D LiDAR固有的低获取频率限制，实现高频率输出已成为LiDAR-惯性里程计（LIO）领域的关键研究重点。为了确保实时性能，高频率增强的LIO系统必须在显著减少的时间框架内处理每个扫描，这对低计算能力平台上部署提出了巨大挑战。为了解决这些限制，我们引入了SR-LIO++，这是一种能够在资源受限硬件平台上（包括Raspberry Pi 4B）将输出频率相对输入频率提高一倍的创新LIO系统。我们的系统采用扫描重建方法来增强LiDAR扫描频率，生成高频率的重建扫描。在此基础上，我们提出了一种中间结果（即表面参数）缓存机制，有效减少了相邻重建扫描中常见段落的冗余处理，从而解耦处理时间与传统上与重建扫描频率呈线性关系的依赖性。此外，我们提出了一种基于索引表映射的姿态点管理量化方法，通过将全局三维点存储从64位双精度转换为8位字符表示，显著减少了内存使用量。这种方法还将最近邻搜索中的计算密集型欧几里得距离计算从64位双精度转换为16位短整数和32位整数格式，显著减少了内存和计算成本。通过对三个不同的计算平台和四个公开数据集进行广泛的实验评估，证明SR-LIO++在保持最先进的精度的同时显著提高了效率。值得注意的是，我们的系统在Raspberry Pi 4B硬件上成功实现了20Hz状态输出。 

---
# UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving 

**Title (ZH)**: UniOcc：自主驾驶中 occupancy 预测和估计统一基准 

**Authors**: Yuping Wang, Xiangyu Huang, Xiaokang Sun, Mingxuan Yan, Shuo Xing, Zhengzhong Tu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.24381)  

**Abstract**: We introduce UniOcc, a comprehensive, unified benchmark for occupancy forecasting (i.e., predicting future occupancies based on historical information) and current-frame occupancy prediction from camera images. UniOcc unifies data from multiple real-world datasets (i.e., nuScenes, Waymo) and high-fidelity driving simulators (i.e., CARLA, OpenCOOD), which provides 2D/3D occupancy labels with per-voxel flow annotations and support for cooperative autonomous driving. In terms of evaluation, unlike existing studies that rely on suboptimal pseudo labels for evaluation, UniOcc incorporates novel metrics that do not depend on ground-truth occupancy, enabling robust assessment of additional aspects of occupancy quality. Through extensive experiments on state-of-the-art models, we demonstrate that large-scale, diverse training data and explicit flow information significantly enhance occupancy prediction and forecasting performance. 

**Abstract (ZH)**: UniOcc：Occupancy 预测的综合统一基准 

---
# Learning 3D-Gaussian Simulators from RGB Videos 

**Title (ZH)**: 从RGB视频学习3D高斯模拟器 

**Authors**: Mikel Zhobro, Andreas René Geist, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.24009)  

**Abstract**: Learning physics simulations from video data requires maintaining spatial and temporal consistency, a challenge often addressed with strong inductive biases or ground-truth 3D information -- limiting scalability and generalization. We introduce 3DGSim, a 3D physics simulator that learns object dynamics end-to-end from multi-view RGB videos. It encodes images into a 3D Gaussian particle representation, propagates dynamics via a transformer, and renders frames using 3D Gaussian splatting. By jointly training inverse rendering with a dynamics transformer using a temporal encoding and merging layer, 3DGSimembeds physical properties into point-wise latent vectors without enforcing explicit connectivity constraints. This enables the model to capture diverse physical behaviors, from rigid to elastic and cloth-like interactions, along with realistic lighting effects that also generalize to unseen multi-body interactions and novel scene edits. 

**Abstract (ZH)**: 3DGSim：从多视角RGB视频中端到端学习物体动力学的3D物理模拟器 

---
# Video-based Traffic Light Recognition by Rockchip RV1126 for Autonomous Driving 

**Title (ZH)**: 基于Rockchip RV1126的视频交通灯识别技术及其在自动驾驶中的应用 

**Authors**: Miao Fan, Xuxu Kong, Shengtong Xu, Haoyi Xiong, Xiangzeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23965)  

**Abstract**: Real-time traffic light recognition is fundamental for autonomous driving safety and navigation in urban environments. While existing approaches rely on single-frame analysis from onboard cameras, they struggle with complex scenarios involving occlusions and adverse lighting conditions. We present \textit{ViTLR}, a novel video-based end-to-end neural network that processes multiple consecutive frames to achieve robust traffic light detection and state classification. The architecture leverages a transformer-like design with convolutional self-attention modules, which is optimized specifically for deployment on the Rockchip RV1126 embedded platform. Extensive evaluations on two real-world datasets demonstrate that \textit{ViTLR} achieves state-of-the-art performance while maintaining real-time processing capabilities (>25 FPS) on RV1126's NPU. The system shows superior robustness across temporal stability, varying target distances, and challenging environmental conditions compared to existing single-frame approaches. We have successfully integrated \textit{ViTLR} into an ego-lane traffic light recognition system using HD maps for autonomous driving applications. The complete implementation, including source code and datasets, is made publicly available to facilitate further research in this domain. 

**Abstract (ZH)**: 基于视频的实时交通灯识别：ViTLR在城市环境自主驾驶中的应用 

---
# A Benchmark for Vision-Centric HD Mapping by V2I Systems 

**Title (ZH)**: 基于V2I系统的视觉中心高精度地图基准 

**Authors**: Miao Fan, Shanshan Yu, Shengtong Xu, Kun Jiang, Haoyi Xiong, Xiangzeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23963)  

**Abstract**: Autonomous driving faces safety challenges due to a lack of global perspective and the semantic information of vectorized high-definition (HD) maps. Information from roadside cameras can greatly expand the map perception range through vehicle-to-infrastructure (V2I) communications. However, there is still no dataset from the real world available for the study on map vectorization onboard under the scenario of vehicle-infrastructure cooperation. To prosper the research on online HD mapping for Vehicle-Infrastructure Cooperative Autonomous Driving (VICAD), we release a real-world dataset, which contains collaborative camera frames from both vehicles and roadside infrastructures, and provides human annotations of HD map elements. We also present an end-to-end neural framework (i.e., V2I-HD) leveraging vision-centric V2I systems to construct vectorized maps. To reduce computation costs and further deploy V2I-HD on autonomous vehicles, we introduce a directionally decoupled self-attention mechanism to V2I-HD. Extensive experiments show that V2I-HD has superior performance in real-time inference speed, as tested by our real-world dataset. Abundant qualitative results also demonstrate stable and robust map construction quality with low cost in complex and various driving scenes. As a benchmark, both source codes and the dataset have been released at OneDrive for the purpose of further study. 

**Abstract (ZH)**: 自动驾驶面临由于缺乏全局视角和矢量化高分辨率（HD）地图的语义信息而带来的安全挑战。路边摄像头的信息可以通过车辆到基础设施（V2I）通信极大地扩展地图感知范围。然而，在车辆基础设施合作场景下进行车载地图矢量化研究尚无实际世界的数据集可用。为了促进车辆基础设施协同自动驾驶（VICAD）在线高分辨率地图绘制的研究，我们发布了一个真实世界的数据集，包含来自车辆和路边基础设施的协作摄像头帧，并提供了高分辨率地图元素的人工标注。我们还提出了一个端到端的神经框架（即V2I-HD），利用以视觉为中心的V2I系统构建矢量化地图。为了降低计算成本并在自动驾驶车辆上进一步部署V2I-HD，我们引入了方向解耦自注意力机制到V2I-HD中。大量的实验表明，V2I-HD在实时推断速度上表现出卓越的性能，经过我们真实世界的数据集测试。丰富的定性结果还展示了在复杂多样的驾驶场景中低成本的稳定和稳健的地图构建质量。作为基准，开源代码和数据集已在OneDrive上发布，以供进一步研究。 

---
# Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model 

**Title (ZH)**: 使用预训练深度基础模型增强 omnidirectional Stereo 匹配 

**Authors**: Jannik Endres, Oliver Hahn, Charles Corbière, Simone Schaub-Meyer, Stefan Roth, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23502)  

**Abstract**: Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360° field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method. 

**Abstract (ZH)**: 全景深度感知对于需要全方位360°视野场景理解的移动机器人应用至关重要。基于摄像头的设置通过使用立体深度估计生成稠密高分辨率深度图，提供了一种经济有效的方案，无需依赖昂贵的主动感知设备。然而，现有的全景立体配对方法在不同环境、深度范围和光照条件下仅能实现有限的深度精度，这是由于缺乏真实世界数据的支持。我们提出了一种新颖的全景立体配对方法DFI-OmniStereo，该方法结合了大规模预训练基础模型在迭代优化立体配对架构内的相对单目深度估计。我们引入了一种专门的两阶段训练策略，利用相对单目深度特征进行全景立体配对，并进行尺度不变的微调。DFI-OmniStereo在实际世界Helvipad数据集上取得了最先进的结果，相比之前的最佳全景立体配对方法，减少了约16%的视差MAE。 

---
# GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs 

**Title (ZH)**: GenVP：基于对比层次VAEs的视觉谜题生成 

**Authors**: Kalliopi Basioti, Pritish Sahu, Qingze Tony Liu, Zihao Xu, Hao Wang, Vladimir Pavlovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.23598)  

**Abstract**: Raven's Progressive Matrices (RPMs) is an established benchmark to examine the ability to perform high-level abstract visual reasoning (AVR). Despite the current success of algorithms that solve this task, humans can generalize beyond a given puzzle and create new puzzles given a set of rules, whereas machines remain locked in solving a fixed puzzle from a curated choice list. We propose Generative Visual Puzzles (GenVP), a framework to model the entire RPM generation process, a substantially more challenging task. Our model's capability spans from generating multiple solutions for one specific problem prompt to creating complete new puzzles out of the desired set of rules. Experiments on five different datasets indicate that GenVP achieves state-of-the-art (SOTA) performance both in puzzle-solving accuracy and out-of-distribution (OOD) generalization in 22 OOD scenarios. Compared to SOTA generative approaches, which struggle to solve RPMs when the feasible solution space increases, GenVP efficiently generalizes to these challenging setups. Moreover, our model demonstrates the ability to produce a wide range of complete RPMs given a set of abstract rules by effectively capturing the relationships between abstract rules and visual object properties. 

**Abstract (ZH)**: 生成视觉推理谜题（GenVP）：一种全新的高阶抽象视觉推理生成框架 

---
# AI Agents in Engineering Design: A Multi-Agent Framework for Aesthetic and Aerodynamic Car Design 

**Title (ZH)**: AI代理在工程设计中的应用：一种用于美学和空气动力学汽车设计的多代理框架 

**Authors**: Mohamed Elrefaie, Janet Qian, Raina Wu, Qian Chen, Angela Dai, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.23315)  

**Abstract**: We introduce the concept of "Design Agents" for engineering applications, particularly focusing on the automotive design process, while emphasizing that our approach can be readily extended to other engineering and design domains. Our framework integrates AI-driven design agents into the traditional engineering workflow, demonstrating how these specialized computational agents interact seamlessly with engineers and designers to augment creativity, enhance efficiency, and significantly accelerate the overall design cycle. By automating and streamlining tasks traditionally performed manually, such as conceptual sketching, styling enhancements, 3D shape retrieval and generative modeling, computational fluid dynamics (CFD) meshing, and aerodynamic simulations, our approach reduces certain aspects of the conventional workflow from weeks and days down to minutes. These agents leverage state-of-the-art vision-language models (VLMs), large language models (LLMs), and geometric deep learning techniques, providing rapid iteration and comprehensive design exploration capabilities. We ground our methodology in industry-standard benchmarks, encompassing a wide variety of conventional automotive designs, and utilize high-fidelity aerodynamic simulations to ensure practical and applicable outcomes. Furthermore, we present design agents that can swiftly and accurately predict simulation outcomes, empowering engineers and designers to engage in more informed design optimization and exploration. This research underscores the transformative potential of integrating advanced generative AI techniques into complex engineering tasks, paving the way for broader adoption and innovation across multiple engineering disciplines. 

**Abstract (ZH)**: 我们引入了“设计代理”概念，特别应用于汽车设计过程，并强调我们的方法可以轻易扩展到其他工程和设计领域。我们的框架将AI驱动的设计代理融入传统的工程工作流程中，展示了这些专门化的计算代理如何无缝地与工程师和设计师交互，增强创造力、提高效率，并显著加速整个设计周期。通过自动化和简化传统手工完成的任务，如概念草图、风格优化、3D形状检索和生成建模、计算流体动力学（CFD）网格划分和气动模拟，我们的方法将某些传统工作流程中的时间从几周或几天压缩到几分钟。这些代理利用最先进的视觉-语言模型（VLMs）、大规模语言模型（LLMs）和几何深度学习技术，提供快速迭代和全面的设计探索能力。我们基于行业标准基准，涵盖广泛的传统汽车设计，并利用高保真气动模拟以确保实用和适用的结果。此外，我们展示了可以快速准确预测模拟结果的设计代理，使工程师和设计师能够进行更加知情的设计优化和探索。本研究强调了将先进的生成AI技术集成到复杂工程任务中的变革潜力，为多个工程学科的更广泛采用和创新铺平了道路。 

---
# MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing 

**Title (ZH)**: MB-ORES: 一种多分支物体推理器用于遥感中的视觉grounding 

**Authors**: Karim Radouane, Hanane Azzag, Mustapha lebbah  

**Link**: [PDF](https://arxiv.org/pdf/2503.24219)  

**Abstract**: We propose a unified framework that integrates object detection (OD) and visual grounding (VG) for remote sensing (RS) imagery. To support conventional OD and establish an intuitive prior for VG task, we fine-tune an open-set object detector using referring expression data, framing it as a partially supervised OD task. In the first stage, we construct a graph representation of each image, comprising object queries, class embeddings, and proposal locations. Then, our task-aware architecture processes this graph to perform the VG task. The model consists of: (i) a multi-branch network that integrates spatial, visual, and categorical features to generate task-aware proposals, and (ii) an object reasoning network that assigns probabilities across proposals, followed by a soft selection mechanism for final referring object localization. Our model demonstrates superior performance on the OPT-RSVG and DIOR-RSVG datasets, achieving significant improvements over state-of-the-art methods while retaining classical OD capabilities. The code will be available in our repository: \url{this https URL}. 

**Abstract (ZH)**: 我们提出了一种统一框架，将目标检测（OD）和视觉定位（VG）集成到遥感（RS）图像中。为了支持传统的OD并为VG任务建立直观的先验知识，我们使用指示短语数据微调一个开放集目标检测器，将其视为半监督的目标检测任务。在第一阶段，我们为每张图像构建了一个图表示，包括对象查询、类别嵌入和建议位置。然后，我们的任务感知架构处理该图以执行VG任务。该模型由以下两部分组成：(i) 一个多分支网络，结合空间、视觉和类别特征生成任务感知的建议框；(ii) 一个对象推理网络，将概率分配给建议框，随后是软选择机制以实现最终的参照对象定位。我们的模型在OPT-RSVG和DIOR-RSVG数据集上展现出优越的性能，实现了与最先进的方法相比的显著改进，同时保留了传统的OD能力。代码将存放在我们的仓库中：\url{this https URL}。 

---
# DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting 

**Title (ZH)**: DiET-GS: 扩散先验和事件流辅助运动去模糊3D高斯点绘制 

**Authors**: Seungjun Lee, Gim Hee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.24210)  

**Abstract**: Reconstructing sharp 3D representations from blurry multi-view images are long-standing problem in computer vision. Recent works attempt to enhance high-quality novel view synthesis from the motion blur by leveraging event-based cameras, benefiting from high dynamic range and microsecond temporal resolution. However, they often reach sub-optimal visual quality in either restoring inaccurate color or losing fine-grained details. In this paper, we present DiET-GS, a diffusion prior and event stream-assisted motion deblurring 3DGS. Our framework effectively leverages both blur-free event streams and diffusion prior in a two-stage training strategy. Specifically, we introduce the novel framework to constraint 3DGS with event double integral, achieving both accurate color and well-defined details. Additionally, we propose a simple technique to leverage diffusion prior to further enhance the edge details. Qualitative and quantitative results on both synthetic and real-world data demonstrate that our DiET-GS is capable of producing significantly better quality of novel views compared to the existing baselines. Our project page is this https URL 

**Abstract (ZH)**: 从模糊多视角图像重建清晰的3D表示是计算机视觉中的长期问题。 recent works attempt to enhance high-quality novel view synthesis from motion blur by leveraging event-based cameras, benefiting from high dynamic range and microsecond temporal resolution. However, they often reach sub-optimal visual quality in either restoring inaccurate color or losing fine-grained details. In this paper, we present DiET-GS, a diffusion prior and event stream-assisted motion deblurring 3DGS. Our framework effectively leverages both blur-free event streams and diffusion prior in a two-stage training strategy. Specifically, we introduce the novel framework to constraint 3DGS with event double integral, achieving both accurate color and well-defined details. Additionally, we propose a simple technique to leverage diffusion prior to further enhance the edge details. Qualitative and quantitative results on both synthetic and real-world data demonstrate that our DiET-GS is capable of producing significantly better quality of novel views compared to the existing baselines. Our project page is this https URL。 

---
# PolypSegTrack: Unified Foundation Model for Colonoscopy Video Analysis 

**Title (ZH)**: 结肠镜视频分析的统一基础模型：PolypSegTrack 

**Authors**: Anwesa Choudhuri, Zhongpai Gao, Meng Zheng, Benjamin Planche, Terrence Chen, Ziyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24108)  

**Abstract**: Early detection, accurate segmentation, classification and tracking of polyps during colonoscopy are critical for preventing colorectal cancer. Many existing deep-learning-based methods for analyzing colonoscopic videos either require task-specific fine-tuning, lack tracking capabilities, or rely on domain-specific pre-training. In this paper, we introduce \textit{PolypSegTrack}, a novel foundation model that jointly addresses polyp detection, segmentation, classification and unsupervised tracking in colonoscopic videos. Our approach leverages a novel conditional mask loss, enabling flexible training across datasets with either pixel-level segmentation masks or bounding box annotations, allowing us to bypass task-specific fine-tuning. Our unsupervised tracking module reliably associates polyp instances across frames using object queries, without relying on any heuristics. We leverage a robust vision foundation model backbone that is pre-trained unsupervisedly on natural images, thereby removing the need for domain-specific pre-training. Extensive experiments on multiple polyp benchmarks demonstrate that our method significantly outperforms existing state-of-the-art approaches in detection, segmentation, classification, and tracking. 

**Abstract (ZH)**: 早期检测、精确分割、分类和跟踪内镜检查中的息肉对于预防结直肠癌至关重要。现有的许多基于深度学习的内镜视频分析方法要么需要特定任务的微调，要么缺乏跟踪能力，要么依赖于特定领域的预训练。本文介绍了一种新颖的基础模型 \textit{PolypSegTrack}，可以同时解决内镜视频中息肉的检测、分割、分类和无监督跟踪问题。我们的方法利用了一种新颖的条件掩码损失，从而使模型能够在具有像素级分割掩码或边界框注释的数据集上灵活训练，从而避免了特定任务的微调。我们的无监督跟踪模块可靠地在帧间关联息肉实例，无需依赖任何启发式方法。我们采用了一种在自然图像上进行无监督预训练的鲁棒视觉基础模型骨干网络，从而消除了特定领域的预训练需求。在多个息肉基准上的 extensive 实验表明，我们的方法在检测、分割、分类和跟踪方面显著优于现有最先进的方法。 

---
# DenseFormer: Learning Dense Depth Map from Sparse Depth and Image via Conditional Diffusion Model 

**Title (ZH)**: DenseFormer：通过条件扩散模型从稀疏深度图和图像学习密集深度图 

**Authors**: Ming Yuan, Sichao Wang, Chuang Zhang, Lei He, Qing Xu, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23993)  

**Abstract**: The depth completion task is a critical problem in autonomous driving, involving the generation of dense depth maps from sparse depth maps and RGB images. Most existing methods employ a spatial propagation network to iteratively refine the depth map after obtaining an initial dense depth. In this paper, we propose DenseFormer, a novel method that integrates the diffusion model into the depth completion task. By incorporating the denoising mechanism of the diffusion model, DenseFormer generates the dense depth map by progressively refining an initial random depth distribution through multiple iterations. We propose a feature extraction module that leverages a feature pyramid structure, along with multi-layer deformable attention, to effectively extract and integrate features from sparse depth maps and RGB images, which serve as the guiding condition for the diffusion process. Additionally, this paper presents a depth refinement module that applies multi-step iterative refinement across various ranges to the dense depth results generated by the diffusion process. The module utilizes image features enriched with multi-scale information and sparse depth input to further enhance the accuracy of the predicted depth map. Extensive experiments on the KITTI outdoor scene dataset demonstrate that DenseFormer outperforms classical depth completion methods. 

**Abstract (ZH)**: 深度完成任务是自主驾驶中的关键技术问题，涉及从稀疏深度图和RGB图像生成密集深度图。大多数现有方法使用空间传播网络在获得初始密集深度图后逐迭代地细化深度图。本文提出了一种新的方法DenseFormer，将扩散模型集成到深度完成任务中。通过结合扩散模型的去噪机制，DenseFormer通过多次迭代逐步细化初始随机深度分布生成密集深度图。我们提出了一种特征提取模块，利用特征金字塔结构和多层可变形注意力机制，有效提取并集成来自稀疏深度图和RGB图像的特征，这些特征作为扩散过程的引导条件。此外，本文还提出了一种深度细化模块，该模块对扩散过程生成的密集深度结果在不同范围内进行多步迭代细化，并利用多尺度信息丰富的图像特征和稀疏深度输入进一步增强预测深度图的准确性。在KITTI室外场景数据集上的大量实验表明，DenseFormer优于经典的深度完成方法。 

---
# Training-Free Text-Guided Image Editing with Visual Autoregressive Model 

**Title (ZH)**: 基于视觉自回归模型的无训练文本引导图像编辑 

**Authors**: Yufei Wang, Lanqing Guo, Zhihao Li, Jiaxing Huang, Pichao Wang, Bihan Wen, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23897)  

**Abstract**: Text-guided image editing is an essential task that enables users to modify images through natural language descriptions. Recent advances in diffusion models and rectified flows have significantly improved editing quality, primarily relying on inversion techniques to extract structured noise from input images. However, inaccuracies in inversion can propagate errors, leading to unintended modifications and compromising fidelity. Moreover, even with perfect inversion, the entanglement between textual prompts and image features often results in global changes when only local edits are intended. To address these challenges, we propose a novel text-guided image editing framework based on VAR (Visual AutoRegressive modeling), which eliminates the need for explicit inversion while ensuring precise and controlled modifications. Our method introduces a caching mechanism that stores token indices and probability distributions from the original image, capturing the relationship between the source prompt and the image. Using this cache, we design an adaptive fine-grained masking strategy that dynamically identifies and constrains modifications to relevant regions, preventing unintended changes. A token reassembling approach further refines the editing process, enhancing diversity, fidelity, and control. Our framework operates in a training-free manner and achieves high-fidelity editing with faster inference speeds, processing a 1K resolution image in as fast as 1.2 seconds. Extensive experiments demonstrate that our method achieves performance comparable to, or even surpassing, existing diffusion- and rectified flow-based approaches in both quantitative metrics and visual quality. The code will be released. 

**Abstract (ZH)**: 基于VAR的文本引导图像编辑框架 

---
# MuseFace: Text-driven Face Editing via Diffusion-based Mask Generation Approach 

**Title (ZH)**: MuseFace：基于扩散掩码生成方法的文本驱动面部编辑 

**Authors**: Xin Zhang, Siting Huang, Xiangyang Luo, Yifan Xie, Weijiang Yu, Heng Chang, Fei Ma, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23888)  

**Abstract**: Face editing modifies the appearance of face, which plays a key role in customization and enhancement of personal images. Although much work have achieved remarkable success in text-driven face editing, they still face significant challenges as none of them simultaneously fulfill the characteristics of diversity, controllability and flexibility. To address this challenge, we propose MuseFace, a text-driven face editing framework, which relies solely on text prompt to enable face editing. Specifically, MuseFace integrates a Text-to-Mask diffusion model and a semantic-aware face editing model, capable of directly generating fine-grained semantic masks from text and performing face editing. The Text-to-Mask diffusion model provides \textit{diversity} and \textit{flexibility} to the framework, while the semantic-aware face editing model ensures \textit{controllability} of the framework. Our framework can create fine-grained semantic masks, making precise face editing possible, and significantly enhancing the controllability and flexibility of face editing models. Extensive experiments demonstrate that MuseFace achieves superior high-fidelity performance. 

**Abstract (ZH)**: 文本驱动的面部编辑修改了面部的外观，在个性化和增强个人形象方面发挥着关键作用。尽管大量工作在文本驱动的面部编辑方面取得了显著成功，但它们仍然面临重大挑战，即没有任何方法能够同时具备多样性、可控性和灵活性的特点。为解决这一挑战，我们提出了一种文本驱动的面部编辑框架MuseFace，该框架仅依赖文本提示来实现面部编辑。具体而言，MuseFace 结合了文本到掩码的扩散模型和语义感知的面部编辑模型，能够直接从文本生成精细的语义掩码并执行面部编辑。文本到掩码的扩散模型为框架提供了多样性和灵活性，而语义感知的面部编辑模型则确保了框架的可控性。我们的框架能够生成精细的语义掩码，从而使精确的面部编辑成为可能，并显著提高了面部编辑模型的可控性和灵活性。广泛实验表明，MuseFace 达到了优越的高保真性能。 

---
# Learned Image Compression and Restoration for Digital Pathology 

**Title (ZH)**: 学习驱动的图像压缩与恢复在数字病理学中应用 

**Authors**: SeonYeong Lee, EonSeung Seong, DongEon Lee, SiYeoul Lee, Yubin Cho, Chunsu Park, Seonho Kim, MinKyoung Seo, YoungSin Ko, MinWoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.23862)  

**Abstract**: Digital pathology images play a crucial role in medical diagnostics, but their ultra-high resolution and large file sizes pose significant challenges for storage, transmission, and real-time visualization. To address these issues, we propose CLERIC, a novel deep learning-based image compression framework designed specifically for whole slide images (WSIs). CLERIC integrates a learnable lifting scheme and advanced convolutional techniques to enhance compression efficiency while preserving critical pathological details. Our framework employs a lifting-scheme transform in the analysis stage to decompose images into low- and high-frequency components, enabling more structured latent representations. These components are processed through parallel encoders incorporating Deformable Residual Blocks (DRB) and Recurrent Residual Blocks (R2B) to improve feature extraction and spatial adaptability. The synthesis stage applies an inverse lifting transform for effective image reconstruction, ensuring high-fidelity restoration of fine-grained tissue structures. We evaluate CLERIC on a digital pathology image dataset and compare its performance against state-of-the-art learned image compression (LIC) models. Experimental results demonstrate that CLERIC achieves superior rate-distortion (RD) performance, significantly reducing storage requirements while maintaining high diagnostic image quality. Our study highlights the potential of deep learning-based compression in digital pathology, facilitating efficient data management and long-term storage while ensuring seamless integration into clinical workflows and AI-assisted diagnostic systems. Code and models are available at: this https URL. 

**Abstract (ZH)**: 数字病理图像在医疗诊断中扮演着至关重要的角色，但其超高的分辨率和庞大的文件大小给存储、传输和实时可视化带来了重大挑战。为应对这些挑战，我们提出了一种名为CLERIC的新型基于深度学习的图像压缩框架，专门针对全切片图像（WSIs）。CLERIC结合了可学习提升方案和先进的卷积技术，以提高压缩效率同时保留关键的病理细节。该框架在分析阶段采用提升方案变换将图像分解为低频和高频组件，使其能够生成更具结构化的潜在表示。这些组件通过包含可变形残差块（DRB）和循环残差块（R2B）的并行编码器进行处理，以改善特征提取和空间适应性。合成功态应用逆提升变换进行有效的图像重建，确保细粒度组织结构的高保真恢复。我们在数字病理图像数据集上评估了CLERIC，并将其性能与最先进的学习图像压缩（LIC）模型进行比较。实验结果表明，CLERIC在率失真（RD）性能方面表现出色，显著减少了存储需求同时保持了高诊断图像质量。我们的研究突显了基于深度学习的压缩在数字病理学中的潜在价值，有助于实现高效的数据管理、长期存储以及与临床工作流程和AI辅助诊断系统的无缝集成。代码和模型可在以下链接获取：this https URL。 

---
# MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation 

**Title (ZH)**: MGD-SAM2：多视图引导细节增强的通用分割模型2for高分辨率无类别分割 

**Authors**: Haoran Shen, Peixian Zhuang, Jiahao Kou, Yuxin Zeng, Haoying Xu, Jiangyun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23786)  

**Abstract**: Segment Anything Models (SAMs), as vision foundation models, have demonstrated remarkable performance across various image analysis tasks. Despite their strong generalization capabilities, SAMs encounter challenges in fine-grained detail segmentation for high-resolution class-independent segmentation (HRCS), due to the limitations in the direct processing of high-resolution inputs and low-resolution mask predictions, and the reliance on accurate manual prompts. To address these limitations, we propose MGD-SAM2 which integrates SAM2 with multi-view feature interaction between a global image and local patches to achieve precise segmentation. MGD-SAM2 incorporates the pre-trained SAM2 with four novel modules: the Multi-view Perception Adapter (MPAdapter), the Multi-view Complementary Enhancement Module (MCEM), the Hierarchical Multi-view Interaction Module (HMIM), and the Detail Refinement Module (DRM). Specifically, we first introduce MPAdapter to adapt the SAM2 encoder for enhanced extraction of local details and global semantics in HRCS images. Then, MCEM and HMIM are proposed to further exploit local texture and global context by aggregating multi-view features within and across multi-scales. Finally, DRM is designed to generate gradually restored high-resolution mask predictions, compensating for the loss of fine-grained details resulting from directly upsampling the low-resolution prediction maps. Experimental results demonstrate the superior performance and strong generalization of our model on multiple high-resolution and normal-resolution datasets. Code will be available at this https URL. 

**Abstract (ZH)**: Segment Anything Models (SAMs) 在各种图像分析任务中展示了 remarkable 的表现。尽管 SAMs 具有强大的泛化能力，但在高分辨率类内分割（HRCS）任务中，它们在精细细节分割方面仍面临挑战，这主要是由于高分辨率输入的直接处理能力有限、低分辨率掩码预测的准确度不足以及对精确手动提示的依赖。为了解决这些局限性，我们提出了 MGD-SAM2，该模型将 SAM2 与全局图像和局部片段的多视图特征交互相结合，以实现精确分割。MGD-SAM2 结合预训练的 SAM2 和四个新颖模块：多视图感知适配器 (MPAdapter)、多视图互补增强模块 (MCEM)、分层多视图交互模块 (HMIM) 和细节精炼模块 (DRM)。具体而言，我们首先引入 MPAdapter 以增强 SAM2 编码器在 HRCS 图像中局部细节和全局语义的提取。然后，提出 MCEM 和 HMIM 通过在不同尺度内和跨尺度聚合多视图特征，进一步利用局部纹理和全局上下文。最后，设计 DRM 生成逐步恢复的高分辨率掩码预测，以补偿直接上采样低分辨率预测图造成的精细细节损失。实验结果表明，我们的模型在多个高分辨率和正常分辨率数据集上的性能和泛化能力优越。代码将在以下网址公开：this https URL。 

---
# WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation 

**Title (ZH)**: WaveFormer：一种基于小波驱动特征表示的高效医学图像分割3DTransformer 

**Authors**: Md Mahfuz Al Hasan, Mahdi Zaman, Abdul Jawad, Alberto Santamaria-Pang, Ho Hin Lee, Ivan Tarapov, Kyle See, Md Shah Imran, Antika Roy, Yaser Pourmohammadi Fallah, Navid Asadizanjani, Reza Forghani  

**Link**: [PDF](https://arxiv.org/pdf/2503.23764)  

**Abstract**: Transformer-based architectures have advanced medical image analysis by effectively modeling long-range dependencies, yet they often struggle in 3D settings due to substantial memory overhead and insufficient capture of fine-grained local features. We address these limi- tations with WaveFormer, a novel 3D-transformer that: i) leverages the fundamental frequency-domain properties of features for contextual rep- resentation, and ii) is inspired by the top-down mechanism of the human visual recognition system, making it a biologically motivated architec- ture. By employing discrete wavelet transformations (DWT) at multiple scales, WaveFormer preserves both global context and high-frequency de- tails while replacing heavy upsampling layers with efficient wavelet-based summarization and reconstruction. This significantly reduces the number of parameters, which is critical for real-world deployment where compu- tational resources and training times are constrained. Furthermore, the model is generic and easily adaptable to diverse applications. Evaluations on BraTS2023, FLARE2021, and KiTS2023 demonstrate performance on par with state-of-the-art methods while offering substantially lower computational complexity. 

**Abstract (ZH)**: 基于变换器的架构通过有效地建模长距离依赖性，促进了医学图像分析，但往往在3D环境中由于内存开销巨大且难以捕捉细致的局部特征而受限。我们通过提出一种新型的3D变换器WaveFormer来克服这些限制：i) 利用特征的基本频域特性进行上下文表示；ii) 受人类视觉识别系统自上而下机制的启发，使其成为一种生物动机型架构。通过在多个尺度上采用离散小波变换（DWT），WaveFormer既能保持全局上下文又能捕捉高频细节，同时用高效的小波基总结和重建替代了复杂的上采样层，极大地减少了参数数量，这对于计算资源和训练时间受限的实际部署至关重要。此外，该模型是通用的且易于适应各种应用。在BraTS2023、FLARE2021和KiTS2023上的评估显示，其性能与最先进的方法相当，而计算复杂度大幅降低。 

---
# Investigation of intelligent barbell squat coaching system based on computer vision and machine learning 

**Title (ZH)**: 基于计算机视觉和机器学习的智能杠铃深蹲指导系统研究 

**Authors**: Yinq-Rong Chern, Yuhao Lee, Hsiao-Ching Lin, Guan-Ting Chen, Ying-Hsien Chen, Fu-Sung Lin, Chih-Yao Chuang, Jenn-Jier James Lien, Chih-Hsien Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23731)  

**Abstract**: Purpose: Research has revealed that strength training can reduce the incidence of chronic diseases and physical deterioration at any age. Therefore, having a movement diagnostic system is crucial for training alone. Hence, this study developed an artificial intelligence and computer vision-based barbell squat coaching system with a real-time mode that immediately diagnoses the issue and provides feedback after each squat. In addition, a replay mode allows users to examine their previous squats and check their comments. Initially, four primary characteristics of the barbell squat were identified: body joint angles, dorsiflexion, the ratio of knee-to-hip movement, and barbell stability. Methods: We collect 8,151 squats from 77 participants, categorizing them as good squats and six issues. Then, we trained the diagnosis models with three machine-learning architectures. Furthermore, this research applied the SHapley Additive exPlanations (SHAP) method to enhance the accuracy of issue prediction and reduce the computation time by feature selection. Results: The F1 score of the six issues reached 86.86%, 69.01%, 77.42%, 90.74%, 95.83%, and 100%. Each squat diagnosis took less than 0.5 seconds. Finally, this study examined the efficacy of the proposed system with two groups of participants trained with and without the system. Subsequently, participants trained with the system exhibited substantial improvements in their squat technique, as assessed both by the system itself and by a professional weightlifting coach. Conclusion: This is a comprehensive study that integrates artificial intelligence, computer vision and multivariable processing technologies, aimed at building a real-time, user-friendly barbell squat feedback and training system. 

**Abstract (ZH)**: 目的: 研究表明，力量训练可以减少任何年龄段慢性疾病和身体退化的发生率。因此，拥有一个运动诊断系统对于训练至关重要。故此研究开发了一个基于人工 intelligence 和计算机视觉的杠铃深蹲指导系统，该系统具有实时模式，能够在每次深蹲后立即诊断问题并提供反馈。此外，录制模式允许用户回顾之前的深蹲并检查评论。最初，确定了杠铃深蹲的四个主要特征：身体关节角、 dorsiflexion、膝髋运动比例以及杠铃稳定性。方法: 收集了来自77名参与者的8,151次深蹲动作，并将其分类为良好深蹲和六个问题。然后，用三种机器学习架构训练诊断模型。此外，本研究使用 SHapley Additive exPlanations (SHAP) 方法以通过特征选择提高问题预测的准确性并减少计算时间。结果: 六个问题的 F1 分数分别为 86.86%、69.01%、77.42%、90.74%、95.83% 和 100%。每次深蹲诊断时间少于 0.5 秒。最后，本研究通过两组使用系统和未使用系统的参与者来检验所提系统的有效性。结果显示，使用系统的参与者在深蹲技术上表现出显著的进步，无论是系统的评估还是专业举重教练的评估都如此。结论: 本研究是一个综合性的研究，集成了人工智能、计算机视觉和多变量处理技术，旨在建立一个实时、用户友好的杠铃深蹲反馈与训练系统。 

---
# KOFFVQA: An Objectively Evaluated Free-form VQA Benchmark for Large Vision-Language Models in the Korean Language 

**Title (ZH)**: KOFFVQA：韩语文本的大规模视觉语言模型客观评价自由形式问答基准 

**Authors**: Yoonshik Kim, Jaeyoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.23730)  

**Abstract**: The recent emergence of Large Vision-Language Models(VLMs) has resulted in a variety of different benchmarks for evaluating such models. Despite this, we observe that most existing evaluation methods suffer from the fact that they either require the model to choose from pre-determined responses, sacrificing open-endedness, or evaluate responses using a judge model, resulting in subjective and unreliable evaluation. In addition, we observe a lack of benchmarks for VLMs in the Korean language, which are necessary as a separate metric from more common English language benchmarks, as the performance of generative language models can differ significantly based on the language being used. Therefore, we present KOFFVQA, a general-purpose free-form visual question answering benchmark in the Korean language for the evaluation of VLMs. Our benchmark consists of 275 carefully crafted questions each paired with an image and grading criteria covering 10 different aspects of VLM performance. The grading criteria eliminate the problem of unreliability by allowing the judge model to grade each response based on a pre-determined set of rules. By defining the evaluation criteria in an objective manner, even a small open-source model can be used to evaluate models on our benchmark reliably. In addition to evaluating a large number of existing VLMs on our benchmark, we also experimentally verify that our method of using pre-existing grading criteria for evaluation is much more reliable than existing methods. Our evaluation code is available at this https URL 

**Abstract (ZH)**: Recent 出现的大规模视觉-语言模型(VLMs)为评估这类模型带来了多种不同的基准。尽管如此，我们观察到大多数现有的评估方法存在以下问题：要么要求模型从预定义的回答中选择，牺牲了开放性；要么使用裁判模型评估回答，导致主观且不可靠的评估。此外，我们还观察到关于韩语的大规模视觉-语言模型(VLMs)基准不足，作为与更常见的英语基准分离的指标是必要的，因为生成语言模型的性能会根据使用的语言有很大差异。因此，我们提出了KOFFVQA，一种用于大规模视觉-语言模型评估的韩语通用开放式视觉问答基准。我们的基准包括275个精心设计的问题，每个问题配有一张图片和涵盖10个方面的大规模视觉-语言模型性能评估标准。这些评估标准通过允许裁判模型根据预定义的规则对每个回答进行评分来解决不可靠性问题。通过以客观的方式定义评估标准，即使是小型开源模型也可以可靠地在我们的基准上评估模型。除了在我们的基准上评估大量现有的大规模视觉-语言模型外，我们还实验证明，我们使用现成的评分标准进行评估的方法比现有方法要可靠得多。我们的评估代码可在以下网址获得。 

---
# GMapLatent: Geometric Mapping in Latent Space 

**Title (ZH)**: GMapLatent：潜在空间中的几何映射 

**Authors**: Wei Zeng, Xuebin Chang, Jianghao Su, Xiang Gu, Jian Sun, Zongben Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23407)  

**Abstract**: Cross-domain generative models based on encoder-decoder AI architectures have attracted much attention in generating realistic images, where domain alignment is crucial for generation accuracy. Domain alignment methods usually deal directly with the initial distribution; however, mismatched or mixed clusters can lead to mode collapse and mixture problems in the decoder, compromising model generalization capabilities. In this work, we innovate a cross-domain alignment and generation model that introduces a canonical latent space representation based on geometric mapping to align the cross-domain latent spaces in a rigorous and precise manner, thus avoiding mode collapse and mixture in the encoder-decoder generation architectures. We name this model GMapLatent. The core of the method is to seamlessly align latent spaces with strict cluster correspondence constraints using the canonical parameterizations of cluster-decorated latent spaces. We first (1) transform the latent space to a canonical parameter domain by composing barycenter translation, optimal transport merging and constrained harmonic mapping, and then (2) compute geometric registration with cluster constraints over the canonical parameter domains. This process realizes a bijective (one-to-one and onto) mapping between newly transformed latent spaces and generates a precise alignment of cluster pairs. Cross-domain generation is then achieved through the aligned latent spaces embedded in the encoder-decoder pipeline. Experiments on gray-scale and color images validate the efficiency, efficacy and applicability of GMapLatent, and demonstrate that the proposed model has superior performance over existing models. 

**Abstract (ZH)**: 基于编码器-解码器架构的跨域生成模型：通过几何映射实现精确的跨域潜空间对齐 

---
# Towards Physically Plausible Video Generation via VLM Planning 

**Title (ZH)**: 基于VLM规划的物理合理视频生成 

**Authors**: Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, Xu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.23368)  

**Abstract**: Video diffusion models (VDMs) have advanced significantly in recent years, enabling the generation of highly realistic videos and drawing the attention of the community in their potential as world simulators. However, despite their capabilities, VDMs often fail to produce physically plausible videos due to an inherent lack of understanding of physics, resulting in incorrect dynamics and event sequences. To address this limitation, we propose a novel two-stage image-to-video generation framework that explicitly incorporates physics. In the first stage, we employ a Vision Language Model (VLM) as a coarse-grained motion planner, integrating chain-of-thought and physics-aware reasoning to predict a rough motion trajectories/changes that approximate real-world physical dynamics while ensuring the inter-frame consistency. In the second stage, we use the predicted motion trajectories/changes to guide the video generation of a VDM. As the predicted motion trajectories/changes are rough, noise is added during inference to provide freedom to the VDM in generating motion with more fine details. Extensive experimental results demonstrate that our framework can produce physically plausible motion, and comparative evaluations highlight the notable superiority of our approach over existing methods. More video results are available on our Project Page: this https URL. 

**Abstract (ZH)**: 视频扩散模型（VDMs）近年来取得了显著进展，使其能够生成高度逼真的视频，并引起了社区对它们作为世界模拟器潜力的关注。然而，尽管具有这些能力，VDMs往往由于缺乏对物理原理的理解而无法生成物理上合理的视频，导致错误的动力学和事件序列。为解决这一局限性，我们提出了一种新的两阶段图像到视频生成框架，明确地融合了物理原理。在第一阶段，我们采用视觉语言模型（VLM）作为粗粒度的运动规划器，结合链式思考和物理意识推理来预测近似真实世界物理动态的粗略运动轨迹/变化，同时确保帧间一致性。在第二阶段，我们利用预测的运动轨迹/变化来指导VDM的视频生成。由于预测的运动轨迹/变化较为粗糙，推理过程中会添加噪声以赋予VDM更多细节运动的自由度。大量实验结果表明，我们的框架可以生成物理上合理的运动，而且与现有方法相比，我们的方法具有显著优势。更多视频结果请参见我们的项目页面：this https URL。 

---
# Improved Ear Verification with Vision Transformers and Overlapping Patches 

**Title (ZH)**: 基于视觉变压器和重叠patches的改进耳验证方法 

**Authors**: Deeksha Arun, Kagan Ozturk, Kevin W. Bowyer, Patrick Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2503.23275)  

**Abstract**: Ear recognition has emerged as a promising biometric modality due to the relative stability in appearance during adulthood. Although Vision Transformers (ViTs) have been widely used in image recognition tasks, their efficiency in ear recognition has been hampered by a lack of attention to overlapping patches, which is crucial for capturing intricate ear features. In this study, we evaluate ViT-Tiny (ViT-T), ViT-Small (ViT-S), ViT-Base (ViT-B) and ViT-Large (ViT-L) configurations on a diverse set of datasets (OPIB, AWE, WPUT, and EarVN1.0), using an overlapping patch selection strategy. Results demonstrate the critical importance of overlapping patches, yielding superior performance in 44 of 48 experiments in a structured study. Moreover, upon comparing the results of the overlapping patches with the non-overlapping configurations, the increase is significant, reaching up to 10% for the EarVN1.0 dataset. In terms of model performance, the ViT-T model consistently outperformed the ViT-S, ViT-B, and ViT-L models on the AWE, WPUT, and EarVN1.0 datasets. The highest scores were achieved in a configuration with a patch size of 28x28 and a stride of 14 pixels. This patch-stride configuration represents 25% of the normalized image area (112x112 pixels) for the patch size and 12.5% of the row or column size for the stride. This study confirms that transformer architectures with overlapping patch selection can serve as an efficient and high-performing option for ear-based biometric recognition tasks in verification scenarios. 

**Abstract (ZH)**: 基于重叠Patch选择的变压器架构在耳纹识别中的应用研究 

---
# FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation 

**Title (ZH)**: FIESTA：基于 Fisher 信息的有效选择性测试时适应算法 

**Authors**: Mohammadmahdi Honarmand, Onur Cezmi Mutlu, Parnian Azizian, Saimourya Surabhi, Dennis P. Wall  

**Link**: [PDF](https://arxiv.org/pdf/2503.23257)  

**Abstract**: Robust facial expression recognition in unconstrained, "in-the-wild" environments remains challenging due to significant domain shifts between training and testing distributions. Test-time adaptation (TTA) offers a promising solution by adapting pre-trained models during inference without requiring labeled test data. However, existing TTA approaches typically rely on manually selecting which parameters to update, potentially leading to suboptimal adaptation and high computational costs. This paper introduces a novel Fisher-driven selective adaptation framework that dynamically identifies and updates only the most critical model parameters based on their importance as quantified by Fisher information. By integrating this principled parameter selection approach with temporal consistency constraints, our method enables efficient and effective adaptation specifically tailored for video-based facial expression recognition. Experiments on the challenging AffWild2 benchmark demonstrate that our approach significantly outperforms existing TTA methods, achieving a 7.7% improvement in F1 score over the base model while adapting only 22,000 parameters-more than 20 times fewer than comparable methods. Our ablation studies further reveal that parameter importance can be effectively estimated from minimal data, with sampling just 1-3 frames sufficient for substantial performance gains. The proposed approach not only enhances recognition accuracy but also dramatically reduces computational overhead, making test-time adaptation more practical for real-world affective computing applications. 

**Abstract (ZH)**: 不受约束环境下鲁棒面部表情识别仍具有挑战性，因为训练和测试分布之间存在显著的变化。测试时自适应（TTA）通过在推理时调整预训练模型来提供一种有前景的解决方案，无需使用标记的测试数据。然而，现有的TTA方法通常需要手动选择要更新的参数，这可能导致次优的自适应并增加计算成本。本文提出了一种基于Fishere信息的选择性自适应框架，该框架能够动态地识别和仅更新最关键模型参数。通过将这种原理性的参数选择方法与时间一致性约束相结合，我们的方法能够针对基于视频的面部表情识别进行高效的自适应调整。在具有挑战性的AffWild2基准测试上的实验表明，我们的方法显著优于现有的TTA方法，在仅调整22,000个参数（比同类方法少20多倍）的情况下，F1分数提高了7.7%。进一步的消融研究表明，参数重要性可以从少量数据中有效地估计，仅采样1-3帧即可实现显著的性能提升。提出的这种方法不仅提高了识别准确性，还大幅减少了计算开销，使测试时自适应在实际情感计算应用中更具实用价值。 

---
# Synthetic Art Generation and DeepFake Detection A Study on Jamini Roy Inspired Dataset 

**Title (ZH)**: 合成艺术生成与DeepFake检测：基于Jamini Roy风格数据集的研究 

**Authors**: Kushal Agrawal, Romi Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23226)  

**Abstract**: The intersection of generative AI and art is a fascinating area that brings both exciting opportunities and significant challenges, especially when it comes to identifying synthetic artworks. This study takes a unique approach by examining diffusion-based generative models in the context of Indian art, specifically focusing on the distinctive style of Jamini Roy. To explore this, we fine-tuned Stable Diffusion 3 and used techniques like ControlNet and IPAdapter to generate realistic images. This allowed us to create a new dataset that includes both real and AI-generated artworks, which is essential for a detailed analysis of what these models can produce. We employed various qualitative and quantitative methods, such as Fourier domain assessments and autocorrelation metrics, to uncover subtle differences between synthetic images and authentic pieces. A key takeaway from recent research is that existing methods for detecting deepfakes face considerable challenges, especially when the deepfakes are of high quality and tailored to specific cultural contexts. This highlights a critical gap in current detection technologies, particularly in light of the challenges identified above, where high-quality and culturally specific deepfakes are difficult to detect. This work not only sheds light on the increasing complexity of generative models but also sets a crucial foundation for future research aimed at effective detection of synthetic art. 

**Abstract (ZH)**: 生成式AI与艺术的交集是一个令人着迷的领域，带来了激动人心的机会和重大挑战，尤其是在识别合成艺术品方面。本研究采取独特的视角，着眼于基于扩散的生成模型在印度艺术中的应用，特别是聚焦于贾米尼·罗伊的特色风格。为探索这一领域，我们对Stable Diffusion 3进行了微调，并使用ControlNet和IPAdapter等技术生成了逼真的图像，从而创建了一个包含真实和AI生成作品的新数据集，这对于详细分析这些模型的能力至关重要。我们采用了四域分析和自相关度量等多种定性与定量方法，以揭示合成图像与真迹之间的细微差异。近年来的研究显示，现有的检测深度伪造的方法面临重大挑战，尤其是在高质量且针对特定文化语境的深度伪造方面。这凸显了当前检测技术中的关键差距，尤其是在上述挑战所指出的地方，高质量且具有文化特异性的深度伪造难以检测。本研究不仅揭示了生成模型不断增加的复杂性，也为未来旨在有效检测合成艺术的研究奠定了重要基础。 

---
# Evaluating Compositional Scene Understanding in Multimodal Generative Models 

**Title (ZH)**: 多模态生成模型中的组成场景理解评估 

**Authors**: Shuhao Fu, Andrew Jun Lee, Anna Wang, Ida Momennejad, Trevor Bihl, Hongjing Lu, Taylor W. Webb  

**Link**: [PDF](https://arxiv.org/pdf/2503.23125)  

**Abstract**: The visual world is fundamentally compositional. Visual scenes are defined by the composition of objects and their relations. Hence, it is essential for computer vision systems to reflect and exploit this compositionality to achieve robust and generalizable scene understanding. While major strides have been made toward the development of general-purpose, multimodal generative models, including both text-to-image models and multimodal vision-language models, it remains unclear whether these systems are capable of accurately generating and interpreting scenes involving the composition of multiple objects and relations. In this work, we present an evaluation of the compositional visual processing capabilities in the current generation of text-to-image (DALL-E 3) and multimodal vision-language models (GPT-4V, GPT-4o, Claude Sonnet 3.5, QWEN2-VL-72B, and InternVL2.5-38B), and compare the performance of these systems to human participants. The results suggest that these systems display some ability to solve compositional and relational tasks, showing notable improvements over the previous generation of multimodal models, but with performance nevertheless well below the level of human participants, particularly for more complex scenes involving many ($>5$) objects and multiple relations. These results highlight the need for further progress toward compositional understanding of visual scenes. 

**Abstract (ZH)**: 视觉世界本质上是组合性的。视觉场景由对象及其关系的组合定义。因此，对于实现鲁棒性和广泛适用性的场景理解而言，计算机视觉系统必须反映并利用这种组合性。尽管已经取得了相当大的进展，开发出了多种通用的多模态生成模型，包括文本到图像模型和多模态视觉语言模型，但对于这些系统是否能够准确生成和解释涉及多个对象及其关系的场景仍不清楚。在这项工作中，我们评估了当前一代文本到图像（DALL-E 3）和多模态视觉语言模型（GPT-4V、GPT-4o、Claude Sonnet 3.5、QWEN2-VL-72B 和 InternVL2.5-38B）的组合视觉处理能力，并将这些系统的性能与人类参与者进行比较。结果显示，这些系统在解决组合性和关系任务方面展示了一定的能力，相比上一代多模态模型表现出了显著的改进，但在性能上仍然远低于人类参与者的水平，尤其是在涉及多个（>5）对象和多种关系的复杂场景中。这些结果强调了进一步推进对视觉场景组合理解的必要性。 

---
# Efficient Adaptation For Remote Sensing Visual Grounding 

**Title (ZH)**: 远程 sensing 视觉定位的高效适应 

**Authors**: Hasan Moughnieh, Mohamad Chalhoub, Hasan Nasrallah, Cristiano Nattero, Paolo Campanella, Ali J. Ghandour  

**Link**: [PDF](https://arxiv.org/pdf/2503.23083)  

**Abstract**: Foundation models have revolutionized artificial intelligence (AI), offering remarkable capabilities across multi-modal domains. Their ability to precisely locate objects in complex aerial and satellite images, using rich contextual information and detailed object descriptions, is essential for remote sensing (RS). These models can associate textual descriptions with object positions through the Visual Grounding (VG) task, but due to domain-specific challenges, their direct application to RS produces sub-optimal results. To address this, we applied Parameter Efficient Fine Tuning (PEFT) techniques to adapt these models for RS-specific VG tasks. Specifically, we evaluated LoRA placement across different modules in Grounding DINO and used BitFit and adapters to fine-tune the OFA foundation model pre-trained on general-purpose VG datasets. This approach achieved performance comparable to or surpassing current State Of The Art (SOTA) models while significantly reducing computational costs. This study highlights the potential of PEFT techniques to advance efficient and precise multi-modal analysis in RS, offering a practical and cost-effective alternative to full model training. 

**Abstract (ZH)**: 基于参数高效微调的技术在遥感特定视觉 grounding 任务中的应用：推进多模态分析的高效与精准 

---
# STSA: Spatial-Temporal Semantic Alignment for Visual Dubbing 

**Title (ZH)**: STSA: 空间- temporal 语义对齐在视觉配音中的应用 

**Authors**: Zijun Ding, Mingdie Xiong, Congcong Zhu, Jingrun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23039)  

**Abstract**: Existing audio-driven visual dubbing methods have achieved great success. Despite this, we observe that the semantic ambiguity between spatial and temporal domains significantly degrades the synthesis stability for the dynamic faces. We argue that aligning the semantic features from spatial and temporal domains is a promising approach to stabilizing facial motion. To achieve this, we propose a Spatial-Temporal Semantic Alignment (STSA) method, which introduces a dual-path alignment mechanism and a differentiable semantic representation. The former leverages a Consistent Information Learning (CIL) module to maximize the mutual information at multiple scales, thereby reducing the manifold differences between spatial and temporal domains. The latter utilizes probabilistic heatmap as ambiguity-tolerant guidance to avoid the abnormal dynamics of the synthesized faces caused by slight semantic jittering. Extensive experimental results demonstrate the superiority of the proposed STSA, especially in terms of image quality and synthesis stability. Pre-trained weights and inference code are available at this https URL. 

**Abstract (ZH)**: 现有的基于音频的视觉配音方法取得了巨大成功。尽管如此，我们观察到空域和时域之间的语义不确定性显著降低了动态面部的合成稳定性。我们主张，对空域和时域的语义特征进行对齐是一种稳定面部运动的有前途的方法。为此，我们提出了一种空时语义对齐（STSA）方法，该方法引入了一条双路径对齐机制和可微分的语义表示。前者利用一种一致信息学习（CIL）模块，在多个尺度上最大化互信息，从而减少空域和时域之间的流形差异。后者利用概率热图作为容忍不确定性的引导，避免由细微的语义抖动导致的合成面部的异常动态。大量实验结果表明，所提出的STSA在图像质量和合成稳定性方面具有优越性。预训练权重和推理代码可在此处访问：this https URL。 

---
# On Geometrical Properties of Text Token Embeddings for Strong Semantic Binding in Text-to-Image Generation 

**Title (ZH)**: 文本令牌嵌入的几何性质在文本到图像生成中的强语义绑定 

**Authors**: Hoigi Seo, Junseo Bang, Haechang Lee, Joohoon Lee, Byung Hyun Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2503.23011)  

**Abstract**: Text-to-Image (T2I) models often suffer from text-image misalignment in complex scenes involving multiple objects and attributes. Semantic binding aims to mitigate this issue by accurately associating the generated attributes and objects with their corresponding noun phrases (NPs). Existing methods rely on text or latent optimizations, yet the factors influencing semantic binding remain underexplored. Here we investigate the geometrical properties of text token embeddings and their cross-attention (CA) maps. We empirically and theoretically analyze that the geometrical properties of token embeddings, specifically both angular distances and norms, play a crucial role in CA map differentiation. Then, we propose \textbf{TeeMo}, a training-free text embedding-aware T2I framework with strong semantic binding. TeeMo consists of Causality-Aware Projection-Out (CAPO) for distinct inter-NP CA maps and Adaptive Token Mixing (ATM) with our loss to enhance inter-NP separation while maintaining intra-NP cohesion in CA maps. Extensive experiments confirm TeeMo consistently outperforms prior arts across diverse baselines and datasets. 

**Abstract (ZH)**: 基于文本的图像生成（T2I）模型在涉及多个对象和属性的复杂场景中常常存在文本与图像不匹配的问题。语义绑定旨在通过准确地将生成的属性和对象与其相应的名词短语（NPs）关联来缓解这一问题。现有方法依赖于文本或潜在优化，但影响语义绑定的因素仍需进一步探索。我们研究了文本标记嵌入的几何性质及其交叉注意（CA）图。我们实证和理论分析表明，标记嵌入的几何性质，特别是角度距离和范数，对CA图的区分起着关键作用。然后，我们提出了一个无需训练的文本嵌入感知T2I框架TeeMo，具有强大的语义绑定能力。TeeMo包括因果 Aware 投影去除（CAPO）以实现不同的跨名词短语CA图，以及增强跨名词短语分离同时保持名词短语内部联合性的自适应标记混合（ATM）。广泛实验表明，TeeMo在多种基准和数据集上一致优于现有方法。 

---
# Enhancing DeepLabV3+ to Fuse Aerial and Satellite Images for Semantic Segmentation 

**Title (ZH)**: 增强DeepLabV3+以融合航空和卫星图像进行语义分割 

**Authors**: Anas Berka, Mohamed El Hajji, Raphael Canals, Youssef Es-saady, Adel Hafiane  

**Link**: [PDF](https://arxiv.org/pdf/2503.22909)  

**Abstract**: Aerial and satellite imagery are inherently complementary remote sensing sources, offering high-resolution detail alongside expansive spatial coverage. However, the use of these sources for land cover segmentation introduces several challenges, prompting the development of a variety of segmentation methods. Among these approaches, the DeepLabV3+ architecture is considered as a promising approach in the field of single-source image segmentation. However, despite its reliable results for segmentation, there is still a need to increase its robustness and improve its performance. This is particularly crucial for multimodal image segmentation, where the fusion of diverse types of information is essential.
An interesting approach involves enhancing this architectural framework through the integration of novel components and the modification of certain internal processes.
In this paper, we enhance the DeepLabV3+ architecture by introducing a new transposed conventional layers block for upsampling a second entry to fuse it with high level features. This block is designed to amplify and integrate information from satellite images, thereby enriching the segmentation process through fusion with aerial images.
For experiments, we used the this http URL (Land Cover from Aerial Imagery) dataset for aerial images, alongside the corresponding dataset sourced from Sentinel 2 data.
Through the fusion of both sources, the mean Intersection over Union (mIoU) achieved a total mIoU of 84.91% without data augmentation. 

**Abstract (ZH)**: 基于航天航空影像的DeepLabV3+架构改进及其在多模态影像分割中的应用 

---
# Pairwise Matching of Intermediate Representations for Fine-grained Explainability 

**Title (ZH)**: 中间表示的成对匹配以实现细粒度解释性 

**Authors**: Lauren Shrack, Timm Haucke, Antoine Salaün, Arjun Subramonian, Sara Beery  

**Link**: [PDF](https://arxiv.org/pdf/2503.22881)  

**Abstract**: The differences between images belonging to fine-grained categories are often subtle and highly localized, and existing explainability techniques for deep learning models are often too diffuse to provide useful and interpretable explanations. We propose a new explainability method (PAIR-X) that leverages both intermediate model activations and backpropagated relevance scores to generate fine-grained, highly-localized pairwise visual explanations. We use animal and building re-identification (re-ID) as a primary case study of our method, and we demonstrate qualitatively improved results over a diverse set of explainability baselines on 35 public re-ID datasets. In interviews, animal re-ID experts were in unanimous agreement that PAIR-X was an improvement over existing baselines for deep model explainability, and suggested that its visualizations would be directly applicable to their work. We also propose a novel quantitative evaluation metric for our method, and demonstrate that PAIR-X visualizations appear more plausible for correct image matches than incorrect ones even when the model similarity score for the pairs is the same. By improving interpretability, PAIR-X enables humans to better distinguish correct and incorrect matches. Our code is available at: this https URL 

**Abstract (ZH)**: 细粒度类别图像之间的差异往往微妙且高度局部化，现有的深度学习模型解释技术往往过于模糊，无法提供有用和可解释的解释。我们提出了一种新的解释方法（PAIR-X），该方法结合了中间模型激活和反向传播的相关得分，以生成细粒度和高度局部化的成对视觉解释。我们将动物和建筑再识别（re-ID）作为我们方法的主要案例研究，并在35个公开的re-ID数据集上展示了比多种解释基线方法有质的改进。在访谈中，动物re-ID专家一致认为PAIR-X比现有基线更适合深度模型解释，并建议其可视化可以直接应用于他们的工作中。我们还提出了对我们方法的一种新的定量评价指标，并且证明即使模型对成对图像相似性评分相同，PAIR-X的可视化对于正确图像匹配看起来更可信。通过提高可解释性，PAIR-X使人类能够更好地区分正确的和错误的匹配。我们的代码可在以下链接获取：this https URL 

---
# Patronus: Bringing Transparency to Diffusion Models with Prototypes 

**Title (ZH)**: Patronus: 通过原型提升扩散模型的透明度 

**Authors**: Nina Weng, Aasa Feragen, Siavash Bigdeli  

**Link**: [PDF](https://arxiv.org/pdf/2503.22782)  

**Abstract**: Diffusion-based generative models, such as Denoising Diffusion Probabilistic Models (DDPMs), have achieved remarkable success in image generation, but their step-by-step denoising process remains opaque, leaving critical aspects of the generation mechanism unexplained. To address this, we introduce \emph{Patronus}, an interpretable diffusion model inspired by ProtoPNet. Patronus integrates a prototypical network into DDPMs, enabling the extraction of prototypes and conditioning of the generation process on their prototype activation vector. This design enhances interpretability by showing the learned prototypes and how they influence the generation process. Additionally, the model supports downstream tasks like image manipulation, enabling more transparent and controlled modifications. Moreover, Patronus could reveal shortcut learning in the generation process by detecting unwanted correlations between learned prototypes. Notably, Patronus operates entirely without any annotations or text prompts. This work opens new avenues for understanding and controlling diffusion models through prototype-based interpretability. Our code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于扩散的生成模型，如去噪扩散概率模型（DDPMs），已经在图像生成方面取得了显著成功，但其逐步去噪过程仍然不够透明，留下了生成机制中的关键方面未解释的问题。为解决这一问题，我们提出了一种名为Patronus的可解释扩散模型，该模型受到ProtoPNet的启发。Patronus将原型网络整合到DDPMs中，能够提取原型并根据其原型激活向量条件化生成过程。该设计通过展示学习到的原型及其对生成过程的影响来提升可解释性。此外，该模型支持图像操作等下游任务，使透明和可控的修改成为可能。同时，Patronus可以通过检测学习到的原型之间的不良相关性揭示生成过程中的捷径学习。值得注意的是，Patronus完全无需任何标注或文本提示。这项工作为通过基于原型的可解释性理解和控制扩散模型开辟了新的途径。我们的代码可在此处获得：\href{this https URL}{this https URL}。 

---
# Ancestral Mamba: Enhancing Selective Discriminant Space Model with Online Visual Prototype Learning for Efficient and Robust Discriminant Approach 

**Title (ZH)**: 祖先环蛇：结合在线视觉原型学习以增强选择性 discriminant 空间模型，实现高效和稳健的鉴别方法 

**Authors**: Jiahao Qin, Feng Liu, Lu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22729)  

**Abstract**: In the realm of computer graphics, the ability to learn continuously from non-stationary data streams while adapting to new visual patterns and mitigating catastrophic forgetting is of paramount importance. Existing approaches often struggle to capture and represent the essential characteristics of evolving visual concepts, hindering their applicability to dynamic graphics tasks. In this paper, we propose Ancestral Mamba, a novel approach that integrates online prototype learning into a selective discriminant space model for efficient and robust online continual learning. The key components of our approach include Ancestral Prototype Adaptation (APA), which continuously refines and builds upon learned visual prototypes, and Mamba Feedback (MF), which provides targeted feedback to adapt to challenging visual patterns. APA enables the model to continuously adapt its prototypes, building upon ancestral knowledge to tackle new challenges, while MF acts as a targeted feedback mechanism, focusing on challenging classes and refining their representations. Extensive experiments on graphics-oriented datasets, such as CIFAR-10 and CIFAR-100, demonstrate the superior performance of Ancestral Mamba compared to state-of-the-art baselines, achieving significant improvements in accuracy and forgetting mitigation. 

**Abstract (ZH)**: 在计算机图形学领域，能够在非平稳数据流中持续学习、适应新视觉模式并缓解灾难性遗忘的能力至关重要。现有方法往往难以捕捉并表示不断演化的视觉概念的本质特征，限制了其在动态图形任务中的应用。本文提出了一种名为Ancestral Mamba的新方法，该方法将在线原型学习集成到选择性判别空间模型中，以实现高效而稳健的在线持续学习。该方法的关键组成部分包括祖先原型适应（APA），它连续 refining 并建立已学视觉原型，以及Mamba 反馈（MF），它提供针对性反馈以适应具有挑战性的视觉模式。APA 使模型能够连续适应其原型，基于祖先知识应对新挑战，而 MF 作为针对性反馈机制，专注于具有挑战性的类别并改进其表示。在以CIFAR-10和CIFAR-100为代表的图形导向数据集上的大量实验表明，Ancestral Mamba 在准确性和遗忘缓解方面显著优于现有最先进的基线方法。 

---
# From Eye to Mind: brain2text Decoding Reveals the Neural Mechanisms of Visual Semantic Processing 

**Title (ZH)**: 从眼及至脑：brain2text解码揭示视觉语义处理的神经机制 

**Authors**: Feihan Feng, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.22697)  

**Abstract**: Deciphering the neural mechanisms that transform sensory experiences into meaningful semantic representations is a fundamental challenge in cognitive neuroscience. While neuroimaging has mapped a distributed semantic network, the format and neural code of semantic content remain elusive, particularly for complex, naturalistic stimuli. Traditional brain decoding, focused on visual reconstruction, primarily captures low-level perceptual features, missing the deeper semantic essence guiding human cognition. Here, we introduce a paradigm shift by directly decoding fMRI signals into textual descriptions of viewed natural images. Our novel deep learning model, trained without visual input, achieves state-of-the-art semantic decoding performance, generating meaningful captions that capture the core semantic content of complex scenes. Neuroanatomical analysis reveals the critical role of higher-level visual regions, including MT+, ventral stream visual cortex, and inferior parietal cortex, in this semantic transformation. Category-specific decoding further demonstrates nuanced neural representations for semantic dimensions like animacy and motion. This text-based decoding approach provides a more direct and interpretable window into the brain's semantic encoding than visual reconstruction, offering a powerful new methodology for probing the neural basis of complex semantic processing, refining our understanding of the distributed semantic network, and potentially inspiring brain-inspired language models. 

**Abstract (ZH)**: 解读将感官体验转化为有意义语义表示的神经机制是认知神经科学中的一个基本挑战。虽然神经成像已映射出一个分布式的语义网络，但语义内容的形式及其神经编码方式仍然模糊，尤其是在复杂自然刺激方面。传统的大脑解码主要侧重于视觉重建，主要捕捉低层级知觉特征，而忽略了指导人类认知的深层语义本质。在此，我们通过直接将fMRI信号解码为所观看自然图像的文本描述，引入了一种范式转变。我们的新型深度学习模型在未使用视觉输入的情况下，实现了最先进的语义解码性能，生成能够捕捉复杂场景核心语义内容的有意义描述。神经解剖分析揭示了较高层级的视觉区域，包括MT+、腹侧视皮层和背侧下顶叶皮层，在这一语义转变中的关键作用。类别特异性的解码进一步证明了语义维度（如有生命性和运动性）的精巧神经表征。基于文本的解码方法提供了比视觉重建更直接和可解释的窗口，以观察大脑的语义编码，并提供了一种探索复杂语义处理神经基础的强大新方法，有助于细化分散的语义网络的理解，并可能启发神经启发的语言模型。 

---
