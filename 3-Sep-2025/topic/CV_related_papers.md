# Classification of Vision-Based Tactile Sensors: A Review 

**Title (ZH)**: 基于视觉的触觉传感器分类：一个综述 

**Authors**: Haoran Li, Yijiong Lin, Chenghua Lu, Max Yang, Efi Psomopoulou, Nathan F Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2509.02478)  

**Abstract**: Vision-based tactile sensors (VBTS) have gained widespread application in robotic hands, grippers and prosthetics due to their high spatial resolution, low manufacturing costs, and ease of customization. While VBTSs have common design features, such as a camera module, they can differ in a rich diversity of sensing principles, material compositions, multimodal approaches, and data interpretation methods. Here, we propose a novel classification of VBTS that categorizes the technology into two primary sensing principles based on the underlying transduction of contact into a tactile image: the Marker-Based Transduction Principle and the Intensity-Based Transduction Principle. Marker-Based Transduction interprets tactile information by detecting marker displacement and changes in marker density. In contrast, Intensity-Based Transduction maps external disturbances with variations in pixel values. Depending on the design of the contact module, Marker-Based Transduction can be further divided into two subtypes: Simple Marker-Based (SMB) and Morphological Marker-Based (MMB) mechanisms. Similarly, the Intensity-Based Transduction Principle encompasses the Reflective Layer-based (RLB) and Transparent Layer-Based (TLB) mechanisms. This paper provides a comparative study of the hardware characteristics of these four types of sensors including various combination types, and discusses the commonly used methods for interpreting tactile information. This~comparison reveals some current challenges faced by VBTS technology and directions for future research. 

**Abstract (ZH)**: 基于视觉的触觉传感器（Vision-based Tactile Sensors, VBTS）在机器人手、夹持器和假肢中的广泛应用得益于其高空间分辨率、低制造成本和易定制性。尽管VBTSs具有共同的设计特征，如相机模块，但它们在传感原理、材料组成、多模态方法和数据解释方法上呈现丰富的多样性。在这里，我们提出了一种新颖的VBTS分类方法，根据接触转换为触觉图像的基本原理，将技术分为两大类：标志基转换原理和强度基转换原理。标志基转换通过检测标志物的位移和标志密度的变化来解释触觉信息。相比之下，强度基转换通过像素值的变化映射外部扰动。根据接触模块的设计，标志基转换可以进一步分为简单标志基（SMB）机制和形态学标志基（MMB）机制。同样，强度基转换原理包括反射层基（RLB）机制和透明层基（TLB）机制。本文对这四种类型传感器的硬件特性及其各种组合类型进行了比较研究，并讨论了常用的数据解释方法。比较揭示了VBTS技术目前面临的某些挑战及未来研究的方向。 

---
# Physics-Informed Machine Learning with Adaptive Grids for Optical Microrobot Depth Estimation 

**Title (ZH)**: 基于物理 informant 的自适应网格机器学习方法用于光学微机器人深度估计 

**Authors**: Lan Wei, Lou Genoud, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02343)  

**Abstract**: Optical microrobots actuated by optical tweezers (OT) offer great potential for biomedical applications such as cell manipulation and microscale assembly. These tasks demand accurate three-dimensional perception to ensure precise control in complex and dynamic biological environments. However, the transparent nature of microrobots and low-contrast microscopic imaging challenge conventional deep learning methods, which also require large annotated datasets that are costly to obtain. To address these challenges, we propose a physics-informed, data-efficient framework for depth estimation of optical microrobots. Our method augments convolutional feature extraction with physics-based focus metrics, such as entropy, Laplacian of Gaussian, and gradient sharpness, calculated using an adaptive grid strategy. This approach allocates finer grids over microrobot regions and coarser grids over background areas, enhancing depth sensitivity while reducing computational complexity. We evaluate our framework on multiple microrobot types and demonstrate significant improvements over baseline models. Specifically, our approach reduces mean squared error (MSE) by over 60% and improves the coefficient of determination (R^2) across all test cases. Notably, even when trained on only 20% of the available data, our model outperforms ResNet50 trained on the full dataset, highlighting its robustness under limited data conditions. Our code is available at: this https URL. 

**Abstract (ZH)**: 光学镊子驱动的光学微机器人三维深度估计的物理导向高效框架 

---
# Sem-RaDiff: Diffusion-Based 3D Radar Semantic Perception in Cluttered Agricultural Environments 

**Title (ZH)**: Sem-RaDiff: 基于扩散的复杂农业环境三维雷达语义感知 

**Authors**: Ruibin Zhang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02283)  

**Abstract**: Accurate and robust environmental perception is crucial for robot autonomous navigation. While current methods typically adopt optical sensors (e.g., camera, LiDAR) as primary sensing modalities, their susceptibility to visual occlusion often leads to degraded performance or complete system failure. In this paper, we focus on agricultural scenarios where robots are exposed to the risk of onboard sensor contamination. Leveraging radar's strong penetration capability, we introduce a radar-based 3D environmental perception framework as a viable alternative. It comprises three core modules designed for dense and accurate semantic perception: 1) Parallel frame accumulation to enhance signal-to-noise ratio of radar raw data. 2) A diffusion model-based hierarchical learning framework that first filters radar sidelobe artifacts then generates fine-grained 3D semantic point clouds. 3) A specifically designed sparse 3D network optimized for processing large-scale radar raw data. We conducted extensive benchmark comparisons and experimental evaluations on a self-built dataset collected in real-world agricultural field scenes. Results demonstrate that our method achieves superior structural and semantic prediction performance compared to existing methods, while simultaneously reducing computational and memory costs by 51.3% and 27.5%, respectively. Furthermore, our approach achieves complete reconstruction and accurate classification of thin structures such as poles and wires-which existing methods struggle to perceive-highlighting its potential for dense and accurate 3D radar perception. 

**Abstract (ZH)**: 精准且稳健的环境感知对于机器人自主导航至关重要。现有方法通常采用光学传感器（如摄像头、激光雷达）作为主要传感模态，但其对视觉遮挡的敏感性往往会降低性能或导致系统完全失效。本文着重于农业场景，其中机器人面临车载传感器污染的风险。借助雷达强大的穿透能力，本文介绍了一种基于雷达的3D环境感知框架作为可行替代方案。该框架包含三个核心模块，用于进行密集且精确的语义感知：1）并行帧累积以增强雷达原始数据的信噪比。2）基于扩散模型的分层学习框架，首先过滤雷达旁瓣伪影，然后生成精细粒度的3D语义点云。3）专门设计的稀疏3D网络，优化大型雷达原始数据处理。我们在实际农业场景自建数据集上进行了广泛的基准比较和实验评估。结果表明，我们的方法在结构和语义预测性能上优于现有方法，同时分别减少了51.3%和27.5%的计算和内存成本。此外，我们的方法能够实现对诸如杆和电线等薄结构的完整重建和准确分类，而现有方法难以感知，突显了其在密集且精确3D雷达感知方面的潜力。 

---
# AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving 

**Title (ZH)**: AutoDrive-R$^2$: 激励VLA模型在自动驾驶中进行推理和自我反思能力 

**Authors**: Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01944)  

**Abstract**: Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method. 

**Abstract (ZH)**: 基于视觉-语言-行动（VLA）模型的自动驾驶系统通过多模态感知与决策能力的结合展现出变革性的潜力，然而其决策过程的可解释性和行动序列的连贯性及合理性仍需进一步探索。为解决这些问题，我们提出了一种名为AutoDrive-R$^2$的新型VLA框架，通过链式思考（CoT）处理和强化学习（RL）增强自动驾驶系统的推理和自我反思能力。具体地，我们首先提出了一种创新的CoT数据集nuScenesR$^2$-6K，用于监督微调，该数据集通过四步逻辑链和自我反思有效地建立了输入信息与输出轨迹之间的认知桥梁。此外，为了在RL阶段最大化推理和自我反思能力，我们进一步采用基于物理的奖励框架中的Group Relative Policy Optimization（GRPO）算法，该框架整合了空间对齐、车辆动力学和时间连贯性的标准，以确保可靠的和现实的轨迹规划。我们的方法在nuScenes和Waymo数据集上的广泛评估结果展示了其领先性能和强大的泛化能力。 

---
# AI-Driven Marine Robotics: Emerging Trends in Underwater Perception and Ecosystem Monitoring 

**Title (ZH)**: AI驱动的海洋机器人：水上探测与生态系统监测的新兴趋势 

**Authors**: Scarlett Raine, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.01878)  

**Abstract**: Marine ecosystems face increasing pressure due to climate change, driving the need for scalable, AI-powered monitoring solutions. This paper examines the rapid emergence of underwater AI as a major research frontier and analyzes the factors that have transformed marine perception from a niche application into a catalyst for AI innovation. We identify three convergent drivers: environmental necessity for ecosystem-scale monitoring, democratization of underwater datasets through citizen science platforms, and researcher migration from saturated terrestrial computer vision domains. Our analysis reveals how unique underwater challenges - turbidity, cryptic species detection, expert annotation bottlenecks, and cross-ecosystem generalization - are driving fundamental advances in weakly supervised learning, open-set recognition, and robust perception under degraded conditions. We survey emerging trends in datasets, scene understanding and 3D reconstruction, highlighting the paradigm shift from passive observation toward AI-driven, targeted intervention capabilities. The paper demonstrates how underwater constraints are pushing the boundaries of foundation models, self-supervised learning, and perception, with methodological innovations that extend far beyond marine applications to benefit general computer vision, robotics, and environmental monitoring. 

**Abstract (ZH)**: 海洋生态系统由于气候变化面临不断增加的压力，推动了可扩展的AI驱动监测解决方案的需求。本文探讨了水下AI作为重要研究前沿的迅猛发展，并分析了将其从 niche 应用转变为AI创新催化剂的因素。我们确定了三种相互作用的驱动力：环境需求促使生态系统规模的监测，通过公民科学平台普及水下数据集，以及研究人员从饱和的陆地计算机视觉领域转向。我们的分析揭示了水下独特挑战——浑浊度、隐匿物种检测、专家注释瓶颈以及跨生态系统的一般化——如何推动了弱监督学习、开放集识别和退化条件下鲁棒感知的基石性进步。我们概述了新兴趋势，包括数据集、场景理解和3D重建，展示了从被动观察向AI驱动的目标干预能力的范式转变。本文展示了水下约束如何推动基础模型、自监督学习和感知的边界，其中方法上的创新远不止于海洋应用，还能惠及通用计算机视觉、机器人技术和环境监测。 

---
# Articulated Object Estimation in the Wild 

**Title (ZH)**: 野外 articulated 对象估计 

**Authors**: Abdelrhman Werby, Martin Büchner, Adrian Röfer, Chenguang Huang, Wolfram Burgard, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.01708)  

**Abstract**: Understanding the 3D motion of articulated objects is essential in robotic scene understanding, mobile manipulation, and motion planning. Prior methods for articulation estimation have primarily focused on controlled settings, assuming either fixed camera viewpoints or direct observations of various object states, which tend to fail in more realistic unconstrained environments. In contrast, humans effortlessly infer articulation by watching others manipulate objects. Inspired by this, we introduce ArtiPoint, a novel estimation framework that can infer articulated object models under dynamic camera motion and partial observability. By combining deep point tracking with a factor graph optimization framework, ArtiPoint robustly estimates articulated part trajectories and articulation axes directly from raw RGB-D videos. To foster future research in this domain, we introduce Arti4D, the first ego-centric in-the-wild dataset that captures articulated object interactions at a scene level, accompanied by articulation labels and ground-truth camera poses. We benchmark ArtiPoint against a range of classical and learning-based baselines, demonstrating its superior performance on Arti4D. We make code and Arti4D publicly available at this https URL. 

**Abstract (ZH)**: 理解 articulated 对象的 3D 运动在机器人场景理解、移动操作和运动规划中至关重要。以往的articulation 估计方法主要集中在受控环境中，假设要么是固定的相机视角，要么是对各种对象状态的直接观察，这些方法在更具现实感的非受限环境中往往失效。与此相反，人类通过观看他人操作对象而轻松推断出articulation。受此启发，我们提出了一种新的估计框架 ArtiPoint，该框架能够在动态相机运动和部分可观测性条件下推断 articulated 对象模型。通过将深度点跟踪与因子图优化框架相结合，ArtiPoint 直接从原始 RGB-D 视频中稳健地估计出 articulated 部分轨迹和 articulation 轴。为了促进该领域的未来研究，我们引入了 Arti4D，这是第一个以第一人称视角捕捉场景级别 articulated 对象交互的野外数据集，并附带articulation 标签和真实相机姿态。我们在 Arti4D 上对 ArtiPoint 进行基准测试，展示了其优于多种经典和学习基线的方法性能。我们已在以下网址公开发布代码和 Arti4D：this https URL。 

---
# Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation 

**Title (ZH)**: 基于AI的6D物体姿态预测器的 aleatoric 不确定性在物体相对状态估计中的应用 

**Authors**: Thomas Jantos, Stephan Weiss, Jan Steinbrener  

**Link**: [PDF](https://arxiv.org/pdf/2509.01583)  

**Abstract**: Deep Learning (DL) has become essential in various robotics applications due to excelling at processing raw sensory data to extract task specific information from semantic objects. For example, vision-based object-relative navigation relies on a DL-based 6D object pose predictor to provide the relative pose between the object and the robot as measurements to the robot's state estimator. Accurately knowing the uncertainty inherent in such Deep Neural Network (DNN) based measurements is essential for probabilistic state estimators subsequently guiding the robot's tasks. Thus, in this letter, we show that we can extend any existing DL-based object-relative pose predictor for aleatoric uncertainty inference simply by including two multi-layer perceptrons detached from the translational and rotational part of the DL predictor. This allows for efficient training while freezing the existing pre-trained predictor. We then use the inferred 6D pose and its uncertainty as a measurement and corresponding noise covariance matrix in an extended Kalman filter (EKF). Our approach induces minimal computational overhead such that the state estimator can be deployed on edge devices while benefiting from the dynamically inferred measurement uncertainty. This increases the performance of the object-relative state estimation task compared to a fix-covariance approach. We conduct evaluations on synthetic data and real-world data to underline the benefits of aleatoric uncertainty inference for the object-relative state estimation task. 

**Abstract (ZH)**: 基于深度学习的对象相对姿态预测中 aleatoric 不确定性推断在机器人状态估计中的应用 

---
# DyPho-SLAM : Real-time Photorealistic SLAM in Dynamic Environments 

**Title (ZH)**: DyPho-SLAM : 实时高保真动态环境SLAM 

**Authors**: Yi Liu, Keyu Fan, Bin Lan, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00741)  

**Abstract**: Visual SLAM algorithms have been enhanced through the exploration of Gaussian Splatting representations, particularly in generating high-fidelity dense maps. While existing methods perform reliably in static environments, they often encounter camera tracking drift and fuzzy mapping when dealing with the disturbances caused by moving objects. This paper presents DyPho-SLAM, a real-time, resource-efficient visual SLAM system designed to address the challenges of localization and photorealistic mapping in environments with dynamic objects. Specifically, the proposed system integrates prior image information to generate refined masks, effectively minimizing noise from mask misjudgment. Additionally, to enhance constraints for optimization after removing dynamic obstacles, we devise adaptive feature extraction strategies significantly improving the system's resilience. Experiments conducted on publicly dynamic RGB-D datasets demonstrate that the proposed system achieves state-of-the-art performance in camera pose estimation and dense map reconstruction, while operating in real-time in dynamic scenes. 

**Abstract (ZH)**: 视觉SLAM算法通过探索高保真密集地图生成的Gaussian Splatting表示得到了增强，特别是DyPho-SLAM：一种针对动态环境的实时高效视觉SLAM系统 

---
# Autonomous Aggregate Sorting in Construction and Mining via Computer Vision-Aided Robotic Arm Systems 

**Title (ZH)**: 基于计算机视觉辅助 robotic arm 系统的建筑与矿业自动分选 

**Authors**: Md. Taherul Islam Shawon, Yuan Li, Yincai Cai, Junjie Niu, Ting Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00339)  

**Abstract**: Traditional aggregate sorting methods, whether manual or mechanical, often suffer from low precision, limited flexibility, and poor adaptability to diverse material properties such as size, shape, and lithology. To address these limitations, this study presents a computer vision-aided robotic arm system designed for autonomous aggregate sorting in construction and mining applications. The system integrates a six-degree-of-freedom robotic arm, a binocular stereo camera for 3D perception, and a ROS-based control framework. Core techniques include an attention-augmented YOLOv8 model for aggregate detection, stereo matching for 3D localization, Denavit-Hartenberg kinematic modeling for arm motion control, minimum enclosing rectangle analysis for size estimation, and hand-eye calibration for precise coordinate alignment. Experimental validation with four aggregate types achieved an average grasping and sorting success rate of 97.5%, with comparable classification accuracy. Remaining challenges include the reliable handling of small aggregates and texture-based misclassification. Overall, the proposed system demonstrates significant potential to enhance productivity, reduce operational costs, and improve safety in aggregate handling, while providing a scalable framework for advancing smart automation in construction, mining, and recycling industries. 

**Abstract (ZH)**: 传统的集料分拣方法，无论是人工的还是机械的，往往精度较低、灵活性有限，并且难以适应不同材料的性质，如尺寸、形状和地层。为了解决这些限制，本研究提出了一种计算机视觉辅助机械臂系统，用于建筑和采矿应用中的自主集料分拣。该系统集成了六自由度机械臂、双目立体相机进行三维感知以及基于ROS的控制框架。核心技术包括注意力增强的YOLOv8模型进行集料检测、立体匹配进行三维定位、Denavit-Hartenberg运动学模型进行机械臂运动控制、最小包围矩形分析进行尺寸估计以及手眼标定进行精确坐标对齐。使用四种集料类型进行的实验验证实现了平均抓取和分拣成功率97.5%，分类准确性相当。剩余的挑战包括可靠处理小集料和基于纹理的分类错误。总体而言，所提出的系统显示出显著的潜力，可以通过提高集料处理的生产率、降低操作成本和提高安全性来提升施工、采矿和回收行业的智能自动化水平，同时提供一个可扩展的框架以推进这些行业的智能自动化。 

---
# Ensemble-Based Event Camera Place Recognition Under Varying Illumination 

**Title (ZH)**: 基于事件相机的自适应光照条件下的场所识别集成方法 

**Authors**: Therese Joseph, Tobias Fischer, Michael Milford  

**Link**: [PDF](https://arxiv.org/pdf/2509.01968)  

**Abstract**: Compared to conventional cameras, event cameras provide a high dynamic range and low latency, offering greater robustness to rapid motion and challenging lighting conditions. Although the potential of event cameras for visual place recognition (VPR) has been established, developing robust VPR frameworks under severe illumination changes remains an open research problem. In this paper, we introduce an ensemble-based approach to event camera place recognition that combines sequence-matched results from multiple event-to-frame reconstructions, VPR feature extractors, and temporal resolutions. Unlike previous event-based ensemble methods, which only utilise temporal resolution, our broader fusion strategy delivers significantly improved robustness under varied lighting conditions (e.g., afternoon, sunset, night), achieving a 57% relative improvement in Recall@1 across day-night transitions. We evaluate our approach on two long-term driving datasets (with 8 km per traverse) without metric subsampling, thereby preserving natural variations in speed and stop duration that influence event density. We also conduct a comprehensive analysis of key design choices, including binning strategies, polarity handling, reconstruction methods, and feature extractors, to identify the most critical components for robust performance. Additionally, we propose a modification to the standard sequence matching framework that enhances performance at longer sequence lengths. To facilitate future research, we will release our codebase and benchmarking framework. 

**Abstract (ZH)**: 事件 cameras 在视觉局部场景识别中的基于集成的方法：在极端光照变化下的鲁棒性增强 

---
# TransForSeg: A Multitask Stereo ViT for Joint Stereo Segmentation and 3D Force Estimation in Catheterization 

**Title (ZH)**: TransForSeg: 一种用于共聚焦导管 stereo 分割和三维力估计的多任务 Stereo ViT 

**Authors**: Pedram Fekri, Mehrdad Zadeh, Javad Dargahi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01605)  

**Abstract**: Recently, the emergence of multitask deep learning models has enhanced catheterization procedures by providing tactile and visual perception data through an end-to-end architec- ture. This information is derived from a segmentation and force estimation head, which localizes the catheter in X-ray images and estimates the applied pressure based on its deflection within the image. These stereo vision architectures incorporate a CNN- based encoder-decoder that captures the dependencies between X-ray images from two viewpoints, enabling simultaneous 3D force estimation and stereo segmentation of the catheter. With these tasks in mind, this work approaches the problem from a new perspective. We propose a novel encoder-decoder Vision Transformer model that processes two input X-ray images as separate sequences. Given sequences of X-ray patches from two perspectives, the transformer captures long-range dependencies without the need to gradually expand the receptive field for either image. The embeddings generated by both the encoder and decoder are fed into two shared segmentation heads, while a regression head employs the fused information from the decoder for 3D force estimation. The proposed model is a stereo Vision Transformer capable of simultaneously segmenting the catheter from two angles while estimating the generated forces at its tip in 3D. This model has undergone extensive experiments on synthetic X-ray images with various noise levels and has been compared against state-of-the-art pure segmentation models, vision-based catheter force estimation methods, and a multitask catheter segmentation and force estimation approach. It outperforms existing models, setting a new state-of-the-art in both catheter segmentation and force estimation. 

**Abstract (ZH)**: 最近，多任务深度学习模型的出现通过端到端架构提供了触觉和视觉感知数据，增强了导管操作程序。这些信息来源于一个分割和力估计头，该头在X射线图像中定位导管并根据其在图像中的弯曲程度估计施加的压力。这些立体视觉架构结合了一个基于CNN的编码器-解码器，捕获来自两个视角的X射线图像之间的依赖关系，从而实现同时的3D力估计和立体分割。基于此，本文从一个新的视角来解决这个问题。我们提出了一种新型的编码器-解码器视觉变换器模型，将两个输入的X射线图像作为单独的序列进行处理。给定两个视角的X射线补丁序列，变压器捕获长距离依赖关系，而无需逐步扩展任一图像的感受野。编码器和解码器生成的嵌入被输入到两个共享分割头，而回归头则利用解码器中的融合信息进行3D力估计。所提出模型是一种同时从两个角度分割导管并估计其尖端产生的3D力的立体视觉变换器。该模型在具有不同噪声水平的合成X射线图像上进行了广泛试验，并与最先进的纯分割模型、基于视觉的导管力估计方法以及多任务导管分割和力估计方法进行了比较。该模型在导管分割和力估计方面均优于现有模型，建立了新的最先进的标准。 

---
# AI-driven Dispensing of Coral Reseeding Devices for Broad-scale Restoration of the Great Barrier Reef 

**Title (ZH)**: AI驱动的珊瑚重播设备分发以实现大堡礁的大规模修复 

**Authors**: Scarlett Raine, Benjamin Moshirian, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.01019)  

**Abstract**: Coral reefs are on the brink of collapse, with climate change, ocean acidification, and pollution leading to a projected 70-90% loss of coral species within the next decade. Restoration efforts are crucial, but their success hinges on introducing automation to upscale efforts. We present automated deployment of coral re-seeding devices powered by artificial intelligence, computer vision, and robotics. Specifically, we perform automated substrate classification, enabling detection of areas of the seafloor suitable for coral growth, thus significantly reducing reliance on human experts and increasing the range and efficiency of restoration. Real-world testing of the algorithms on the Great Barrier Reef leads to deployment accuracy of 77.8%, sub-image patch classification of 89.1%, and real-time model inference at 5.5 frames per second. Further, we present and publicly contribute a large collection of annotated substrate image data to foster future research in this area. 

**Abstract (ZH)**: 珊瑚礁正处于崩溃的边缘，气候变化、海洋酸化和污染导致未来十年珊瑚物种可能会减少70-90%。恢复工作至关重要，但其成功取决于引入自动化技术以扩大努力规模。我们提出了一种由人工智能、计算机视觉和机器人技术驱动的自动投放珊瑚重新播种设备的方法。具体而言，我们实现了自动底质分类，能够检测适合珊瑚生长的海底区域，从而大大减少了对人类专家的依赖，并提高了恢复工作的范围和效率。在大堡礁的实际测试中，算法的部署准确率为77.8%，子图像patch分类准确率为89.1%，实时模型推理速率为每秒5.5帧。此外，我们还提供并公开贡献了大量的标注底质图像数据，以促进该领域的未来研究。 

---
# MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation 

**Title (ZH)**: MV-SSM：多视图状态空间建模在3D人体姿态估计中的应用 

**Authors**: Aviral Chharia, Wenbo Gou, Haoye Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00649)  

**Abstract**: While significant progress has been made in single-view 3D human pose estimation, multi-view 3D human pose estimation remains challenging, particularly in terms of generalizing to new camera configurations. Existing attention-based transformers often struggle to accurately model the spatial arrangement of keypoints, especially in occluded scenarios. Additionally, they tend to overfit specific camera arrangements and visual scenes from training data, resulting in substantial performance drops in new settings. In this study, we introduce a novel Multi-View State Space Modeling framework, named MV-SSM, for robustly estimating 3D human keypoints. We explicitly model the joint spatial sequence at two distinct levels: the feature level from multi-view images and the person keypoint level. We propose a Projective State Space (PSS) block to learn a generalized representation of joint spatial arrangements using state space modeling. Moreover, we modify Mamba's traditional scanning into an effective Grid Token-guided Bidirectional Scanning (GTBS), which is integral to the PSS block. Multiple experiments demonstrate that MV-SSM achieves strong generalization, outperforming state-of-the-art methods: +10.8 on AP25 (+24%) on the challenging three-camera setting in CMU Panoptic, +7.0 on AP25 (+13%) on varying camera arrangements, and +15.3 PCP (+38%) on Campus A1 in cross-dataset evaluations. Project Website: this https URL 

**Abstract (ZH)**: 多视角状态空间建模：一种稳健的3D人体关键点估计方法 

---
# AGS: Accelerating 3D Gaussian Splatting SLAM via CODEC-Assisted Frame Covisibility Detection 

**Title (ZH)**: AGS: 通过CODEC辅助帧共视性检测加速3D高斯体绘制SLAM 

**Authors**: Houshu He, Naifeng Jing, Li Jiang, Xiaoyao Liang, Zhuoran Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.00433)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) is a critical task that enables autonomous vehicles to construct maps and localize themselves in unknown environments. Recent breakthroughs combine SLAM with 3D Gaussian Splatting (3DGS) to achieve exceptional reconstruction fidelity. However, existing 3DGS-SLAM systems provide insufficient throughput due to the need for multiple training iterations per frame and the vast number of Gaussians.
In this paper, we propose AGS, an algorithm-hardware co-design framework to boost the efficiency of 3DGS-SLAM based on the intuition that SLAM systems process frames in a streaming manner, where adjacent frames exhibit high similarity that can be utilized for acceleration. On the software level: 1) We propose a coarse-then-fine-grained pose tracking method with respect to the robot's movement. 2) We avoid redundant computations of Gaussians by sharing their contribution information across frames. On the hardware level, we propose a frame covisibility detection engine to extract intermediate data from the video CODEC. We also implement a pose tracking engine and a mapping engine with workload schedulers to efficiently deploy the AGS algorithm. Our evaluation shows that AGS achieves up to $17.12\times$, $6.71\times$, and $5.41\times$ speedups against the mobile and high-end GPUs, and a state-of-the-art 3DGS accelerator, GSCore. 

**Abstract (ZH)**: 三维Gauss散点表示同步定位与Mapping (AGS):一种算法-硬件协同设计框架 

---
# Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation 

**Title (ZH)**: 基于领域适应的跨模态知识精炼方法及其在3D语义分割中的应用 

**Authors**: Jialiang Kang, Jiawen Wang, Dingsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00379)  

**Abstract**: Semantic segmentation of 3D LiDAR data plays a pivotal role in autonomous driving. Traditional approaches rely on extensive annotated data for point cloud analysis, incurring high costs and time investments. In contrast, realworld image datasets offer abundant availability and substantial scale. To mitigate the burden of annotating 3D LiDAR point clouds, we propose two crossmodal knowledge distillation methods: Unsupervised Domain Adaptation Knowledge Distillation (UDAKD) and Feature and Semantic-based Knowledge Distillation (FSKD). Leveraging readily available spatio-temporally synchronized data from cameras and LiDARs in autonomous driving scenarios, we directly apply a pretrained 2D image model to unlabeled 2D data. Through crossmodal knowledge distillation with known 2D-3D correspondence, we actively align the output of the 3D network with the corresponding points of the 2D network, thereby obviating the necessity for 3D annotations. Our focus is on preserving modality-general information while filtering out modality-specific details during crossmodal distillation. To achieve this, we deploy self-calibrated convolution on 3D point clouds as the foundation of our domain adaptation module. Rigorous experimentation validates the effectiveness of our proposed methods, consistently surpassing the performance of state-of-the-art approaches in the field. 

**Abstract (ZH)**: 基于跨模态知识蒸馏的3D LiDAR语义分割方法 

---
# Ordinal Adaptive Correction: A Data-Centric Approach to Ordinal Image Classification with Noisy Labels 

**Title (ZH)**: 序数自适应校正：一种基于数据的序数图像分类方法以处理噪声标签 

**Authors**: Alireza Sedighi Moghaddam, Mohammad Reza Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.02351)  

**Abstract**: Labeled data is a fundamental component in training supervised deep learning models for computer vision tasks. However, the labeling process, especially for ordinal image classification where class boundaries are often ambiguous, is prone to error and noise. Such label noise can significantly degrade the performance and reliability of machine learning models. This paper addresses the problem of detecting and correcting label noise in ordinal image classification tasks. To this end, a novel data-centric method called ORDinal Adaptive Correction (ORDAC) is proposed for adaptive correction of noisy labels. The proposed approach leverages the capabilities of Label Distribution Learning (LDL) to model the inherent ambiguity and uncertainty present in ordinal labels. During training, ORDAC dynamically adjusts the mean and standard deviation of the label distribution for each sample. Rather than discarding potentially noisy samples, this approach aims to correct them and make optimal use of the entire training dataset. The effectiveness of the proposed method is evaluated on benchmark datasets for age estimation (Adience) and disease severity detection (Diabetic Retinopathy) under various asymmetric Gaussian noise scenarios. Results show that ORDAC and its extended versions (ORDAC_C and ORDAC_R) lead to significant improvements in model performance. For instance, on the Adience dataset with 40% noise, ORDAC_R reduced the mean absolute error from 0.86 to 0.62 and increased the recall metric from 0.37 to 0.49. The method also demonstrated its effectiveness in correcting intrinsic noise present in the original datasets. This research indicates that adaptive label correction using label distributions is an effective strategy to enhance the robustness and accuracy of ordinal classification models in the presence of noisy data. 

**Abstract (ZH)**: 标签数据是训练计算机视觉任务监督深度学习模型的基本成分。然而，标签过程，尤其是在类别边界经常模糊的序数图像分类中，容易出错和引入噪声。这样的标签噪声会显著降低机器学习模型的性能和可靠性。本文旨在解决序数图像分类任务中检测和纠正标签噪声的问题。为此，提出了一种名为ORDinal Adaptive Correction (ORDAC)的新数据导向方法，用于噪声标签的自适应修正。该方法利用标签分布学习（LDL）的能力来建模序数标签中存在的内在模糊性和不确定性。在训练过程中，ORDAC 动态调整每个样本的标签分布的均值和标准差。该方法通过纠正潜在的噪声样本，而非简单丢弃，来优化整个训练数据集的使用。该方法在年龄估计（Adience）和糖尿病视网膜病变严重程度检测（Diabetic Retinopathy）的标准数据集上，在不同非对称高斯噪声条件下评估了其有效性。结果表明，ORDAC 及其扩展版本（ORDAC_C 和 ORDAC_R）在模型性能上取得了显著提升。例如，在Adience数据集含有40%噪声的情况下，ORDAC_R将平均绝对误差从0.86降至0.62，同时将召回率从0.37提升至0.49。此外，该方法还展示了其在原始数据集中内在噪声矫正方面的有效性。研究结果表明，使用标签分布进行自适应标签修正是一种在噪声数据存在的情况下增强序数分类模型稳健性和准确性的有效策略。 

---
# Understanding Space Is Rocket Science - Only Top Reasoning Models Can Solve Spatial Understanding Tasks 

**Title (ZH)**: 理解空间关系是火箭科学 - 只有顶级推理模型能解决空间理解任务 

**Authors**: Nils Hoehing, Mayug Maniparambil, Ellen Rushe, Noel E. O'Connor, Anthony Ventresque  

**Link**: [PDF](https://arxiv.org/pdf/2509.02175)  

**Abstract**: We propose RocketScience, an open-source contrastive VLM benchmark that tests for spatial relation understanding. It is comprised of entirely new real-world image-text pairs covering mostly relative spatial understanding and the order of objects. The benchmark is designed
to be very easy for humans and hard for the current generation of VLMs, and this is empirically verified. Our results show a striking lack of spatial relation understanding in open source and frontier commercial VLMs and a surprisingly high performance of reasoning models. Additionally, we perform a disentanglement analysis to separate the contributions of object localization and spatial reasoning in chain-of-thought-based models and find that the performance on the benchmark is bottlenecked by spatial reasoning and not object localization capabilities.
We release the dataset with a CC-BY-4.0 license and make the evaluation code available at: this https URL 

**Abstract (ZH)**: 我们提出RocketScience，这是一个开源对比VLM基准，用于测试空间关系理解能力。该基准由全新的真实世界图像-文本对组成，主要涵盖相对空间理解及物体顺序。该基准设计旨在对人类来说非常简单，但对当前的VLM来说却非常困难，这得到了实证验证。我们的结果表明，开源和前沿商用VLM在空间关系理解方面存在显著不足，而推理模型的表现却意外地高。此外，我们进行了互信息分析，以分离链式推理模型中物体定位能力和空间推理能力的贡献，并发现该基准的性能瓶颈在于空间推理能力而非物体定位能力。我们以CC-BY-4.0许可发布数据集，并在如下链接提供评估代码：this https URL。 

---
# Synesthesia of Machines (SoM)-Based Task-Driven MIMO System for Image Transmission 

**Title (ZH)**: 基于机器联觉的面向任务的MIMO系统用于图像传输 

**Authors**: Sijiang Li, Rongqing Zhang, Xiang Cheng, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02031)  

**Abstract**: To support cooperative perception (CP) of networked mobile agents in dynamic scenarios, the efficient and robust transmission of sensory data is a critical challenge. Deep learning-based joint source-channel coding (JSCC) has demonstrated promising results for image transmission under adverse channel conditions, outperforming traditional rule-based codecs. While recent works have explored to combine JSCC with the widely adopted multiple-input multiple-output (MIMO) technology, these approaches are still limited to the discrete-time analog transmission (DTAT) model and simple tasks. Given the limited performance of existing MIMO JSCC schemes in supporting complex CP tasks for networked mobile agents with digital MIMO communication systems, this paper presents a Synesthesia of Machines (SoM)-based task-driven MIMO system for image transmission, referred to as SoM-MIMO. By leveraging the structural properties of the feature pyramid for perceptual tasks and the channel properties of the closed-loop MIMO communication system, SoM-MIMO enables efficient and robust digital MIMO transmission of images. Experimental results have shown that compared with two JSCC baseline schemes, our approach achieves average mAP improvements of 6.30 and 10.48 across all SNR levels, while maintaining identical communication overhead. 

**Abstract (ZH)**: 基于Synesthesia of Machines的任务驱动MIMO图像传输系统 

---
# Fake & Square: Training Self-Supervised Vision Transformers with Synthetic Data and Synthetic Hard Negatives 

**Title (ZH)**: 假象与方块：使用合成数据和合成硬负样本训练自我监督视觉变换器 

**Authors**: Nikolaos Giakoumoglou, Andreas Floros, Kleanthis Marios Papadopoulos, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.02029)  

**Abstract**: This paper does not introduce a new method per se. Instead, we build on existing self-supervised learning approaches for vision, drawing inspiration from the adage "fake it till you make it". While contrastive self-supervised learning has achieved remarkable success, it typically relies on vast amounts of real-world data and carefully curated hard negatives. To explore alternatives to these requirements, we investigate two forms of "faking it" in vision transformers. First, we study the potential of generative models for unsupervised representation learning, leveraging synthetic data to augment sample diversity. Second, we examine the feasibility of generating synthetic hard negatives in the representation space, creating diverse and challenging contrasts. Our framework - dubbed Syn2Co - combines both approaches and evaluates whether synthetically enhanced training can lead to more robust and transferable visual representations on DeiT-S and Swin-T architectures. Our findings highlight the promise and limitations of synthetic data in self-supervised learning, offering insights for future work in this direction. 

**Abstract (ZH)**: 本文并非引入新的方法，而是基于现有的视觉自监督学习方法进行构建，受到“装样子，再变成样”的谚语的启发。尽管对比自监督学习取得了显著的成果，但它通常依赖大量的真实数据和精心筛选的负样本。为探索这些要求的替代方案，我们研究了视觉变换器中“装样子”的两种形式。首先，我们考察了生成模型在无监督表示学习中的潜力，利用合成数据来增加样本多样性。其次，我们探讨了在表示空间生成合成负样本的可能性，以创建多样的挑战性对比。我们的框架——名为Syn2Co——结合了这两种方法，并评估了增强训练是否能在DeiT-S和Swin-T架构上导致更 robust 和可迁移的视觉表示。我们的发现强调了合成数据在自监督学习中的潜力和限制，为进一步研究提供了见解。 

---
# Unsupervised Training of Vision Transformers with Synthetic Negatives 

**Title (ZH)**: 使用合成负例子进行自监督视Transformer训练 

**Authors**: Nikolaos Giakoumoglou, Andreas Floros, Kleanthis Marios Papadopoulos, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2509.02024)  

**Abstract**: This paper does not introduce a novel method per se. Instead, we address the neglected potential of hard negative samples in self-supervised learning. Previous works explored synthetic hard negatives but rarely in the context of vision transformers. We build on this observation and integrate synthetic hard negatives to improve vision transformer representation learning. This simple yet effective technique notably improves the discriminative power of learned representations. Our experiments show performance improvements for both DeiT-S and Swin-T architectures. 

**Abstract (ZH)**: 本文未引入全新的方法，而是关注自监督学习中被忽视的硬负样本潜在价值。先前工作探索了合成硬负样本，但在视觉变压器的背景下却鲜有涉及。基于此观察，我们在视觉变压器的表示学习中集成合成硬负样本，简单而有效地提高了学习表示的鉴别能力。实验结果表明，该方法在DeiT-S和Swin-T架构上均取得了性能提升。 

---
# 2D Gaussian Splatting with Semantic Alignment for Image Inpainting 

**Title (ZH)**: 基于语义对齐的2D高斯散列图像修复 

**Authors**: Hongyu Li, Chaofeng Chen, Xiaoming Li, Guangming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01964)  

**Abstract**: Gaussian Splatting (GS), a recent technique for converting discrete points into continuous spatial representations, has shown promising results in 3D scene modeling and 2D image super-resolution. In this paper, we explore its untapped potential for image inpainting, which demands both locally coherent pixel synthesis and globally consistent semantic restoration. We propose the first image inpainting framework based on 2D Gaussian Splatting, which encodes incomplete images into a continuous field of 2D Gaussian splat coefficients and reconstructs the final image via a differentiable rasterization process. The continuous rendering paradigm of GS inherently promotes pixel-level coherence in the inpainted results. To improve efficiency and scalability, we introduce a patch-wise rasterization strategy that reduces memory overhead and accelerates inference. For global semantic consistency, we incorporate features from a pretrained DINO model. We observe that DINO's global features are naturally robust to small missing regions and can be effectively adapted to guide semantic alignment in large-mask scenarios, ensuring that the inpainted content remains contextually consistent with the surrounding scene. Extensive experiments on standard benchmarks demonstrate that our method achieves competitive performance in both quantitative metrics and perceptual quality, establishing a new direction for applying Gaussian Splatting to 2D image processing. 

**Abstract (ZH)**: 基于2D高斯散射的图像 inpainting 方法 

---
# Towards Interpretable Geo-localization: a Concept-Aware Global Image-GPS Alignment Framework 

**Title (ZH)**: 面向可解释地理定位的的概念aware全局图像-GPS对齐框架 

**Authors**: Furong Jia, Lanxin Liu, Ce Hou, Fan Zhang, Xinyan Liu, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01910)  

**Abstract**: Worldwide geo-localization involves determining the exact geographic location of images captured globally, typically guided by geographic cues such as climate, landmarks, and architectural styles. Despite advancements in geo-localization models like GeoCLIP, which leverages images and location alignment via contrastive learning for accurate predictions, the interpretability of these models remains insufficiently explored. Current concept-based interpretability methods fail to align effectively with Geo-alignment image-location embedding objectives, resulting in suboptimal interpretability and performance. To address this gap, we propose a novel framework integrating global geo-localization with concept bottlenecks. Our method inserts a Concept-Aware Alignment Module that jointly projects image and location embeddings onto a shared bank of geographic concepts (e.g., tropical climate, mountain, cathedral) and minimizes a concept-level loss, enhancing alignment in a concept-specific subspace and enabling robust interpretability. To our knowledge, this is the first work to introduce interpretability into geo-localization. Extensive experiments demonstrate that our approach surpasses GeoCLIP in geo-localization accuracy and boosts performance across diverse geospatial prediction tasks, revealing richer semantic insights into geographic decision-making processes. 

**Abstract (ZH)**: 全球地理定位涉及确定全球拍摄图像的精确地理位置，通常由气候、地标和建筑风格等地理线索引导。尽管像GeoCLIP这样的地理定位模型通过对比学习利用图像和位置对齐实现了准确的预测，但这些模型的可解释性仍备受忽略。当前的概念基础可解释性方法未能有效地与Geo-对齐图像-位置嵌入目标相匹配，导致可解释性和性能不佳。为解决这一问题，我们提出了一种结合全球地理定位和概念瓶颈的新框架。我们的方法插入了一个概念感知对齐模块，该模块联合将图像和位置嵌入投影到共享的概念库（如热带气候、山脉、大教堂）上，并最小化概念层次的损失，增强在概念特定子空间中的对齐能力，从而实现稳健的可解释性。据我们所知，这是首次将可解释性引入地理定位的研究。广泛的经验表明，我们的方法在地理定位准确性上超越了GeoCLIP，并且在多种地理空间预测任务中提高了性能，揭示了更丰富的地理决策过程的语义洞察。 

---
# HydroVision: Predicting Optically Active Parameters in Surface Water Using Computer Vision 

**Title (ZH)**: HydroVision: 使用计算机视觉预测地表水的光学活性参数 

**Authors**: Shubham Laxmikant Deshmukh, Matthew Wilchek, Feras A. Batarseh  

**Link**: [PDF](https://arxiv.org/pdf/2509.01882)  

**Abstract**: Ongoing advancements in computer vision, particularly in pattern recognition and scene classification, have enabled new applications in environmental monitoring. Deep learning now offers non-contact methods for assessing water quality and detecting contamination, both critical for disaster response and public health protection. This work introduces HydroVision, a deep learning-based scene classification framework that estimates optically active water quality parameters including Chlorophyll-Alpha, Chlorophylls, Colored Dissolved Organic Matter (CDOM), Phycocyanins, Suspended Sediments, and Turbidity from standard Red-Green-Blue (RGB) images of surface water. HydroVision supports early detection of contamination trends and strengthens monitoring by regulatory agencies during external environmental stressors, industrial activities, and force majeure events. The model is trained on more than 500,000 seasonally varied images collected from the United States Geological Survey Hydrologic Imagery Visualization and Information System between 2022 and 2024. This approach leverages widely available RGB imagery as a scalable, cost-effective alternative to traditional multispectral and hyperspectral remote sensing. Four state-of-the-art convolutional neural networks (VGG-16, ResNet50, MobileNetV2, DenseNet121) and a Vision Transformer are evaluated through transfer learning to identify the best-performing architecture. DenseNet121 achieves the highest validation performance, with an R2 score of 0.89 in predicting CDOM, demonstrating the framework's promise for real-world water quality monitoring across diverse conditions. While the current model is optimized for well-lit imagery, future work will focus on improving robustness under low-light and obstructed scenarios to expand its operational utility. 

**Abstract (ZH)**: 基于深度学习的HydroVision场景分类框架：从标准RGB图像估计水体光学活性水质参数 

---
# Doctoral Thesis: Geometric Deep Learning For Camera Pose Prediction, Registration, Depth Estimation, and 3D Reconstruction 

**Title (ZH)**: 博士学位论文：几何深度学习在相机姿态预测、配准、深度估计及3D重建中的应用 

**Authors**: Xueyang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01873)  

**Abstract**: Modern deep learning developments create new opportunities for 3D mapping technology, scene reconstruction pipelines, and virtual reality development. Despite advances in 3D deep learning technology, direct training of deep learning models on 3D data faces challenges due to the high dimensionality inherent in 3D data and the scarcity of labeled datasets. Structure-from-motion (SfM) and Simultaneous Localization and Mapping (SLAM) exhibit robust performance when applied to structured indoor environments but often struggle with ambiguous features in unstructured environments. These techniques often struggle to generate detailed geometric representations effective for downstream tasks such as rendering and semantic analysis. Current limitations require the development of 3D representation methods that combine traditional geometric techniques with deep learning capabilities to generate robust geometry-aware deep learning models.
The dissertation provides solutions to the fundamental challenges in 3D vision by developing geometric deep learning methods tailored for essential tasks such as camera pose estimation, point cloud registration, depth prediction, and 3D reconstruction. The integration of geometric priors or constraints, such as including depth information, surface normals, and equivariance into deep learning models, enhances both the accuracy and robustness of geometric representations. This study systematically investigates key components of 3D vision, including camera pose estimation, point cloud registration, depth estimation, and high-fidelity 3D reconstruction, demonstrating their effectiveness across real-world applications such as digital cultural heritage preservation and immersive VR/AR environments. 

**Abstract (ZH)**: 现代深度学习的发展为三维测绘技术、场景重建管道和虚拟现实开发创造了新的机会。尽管在三维深度学习技术方面取得了进展，但直接在三维数据上训练深度学习模型仍面临挑战，这主要是由于三维数据的高维度特性以及标注数据集的稀缺性。结构从运动（SfM）和即时定位与地图构建（SLAM）在处理结构化室内环境时表现出色，但在处理无结构环境时往往难以应对模糊的特征。这些技术通常难以生成有效用于下游任务如渲染和语义分析的详细几何表示。当前的限制要求开发将传统几何技术与深度学习能力相结合的三维表示方法，以生成稳健的几何感知深度学习模型。

该论文通过开发针对关键任务如相机姿态估计、点云配准、深度预测和三维重建的几何深度学习方法，提供了解决三维视觉根本挑战的解决方案。将几何先验或约束，如深度信息、表面法线和同变性，集成到深度学习模型中，可以提升几何表示的准确性和鲁棒性。本研究系统地探讨了三维视觉的关键组件，包括相机姿态估计、点云配准、深度估计和高保真三维重建，并展示了其在数字文化遗产保护和沉浸式VR/AR环境等实际应用中的有效性。 

---
# Deep Learning-Based Rock Particulate Classification Using Attention-Enhanced ConvNeXt 

**Title (ZH)**: 基于注意力增强ConvNeXt的深学习岩屑颗粒分类 

**Authors**: Anthony Amankwah, Chris Aldrich  

**Link**: [PDF](https://arxiv.org/pdf/2509.01704)  

**Abstract**: Accurate classification of rock sizes is a vital component in geotechnical engineering, mining, and resource management, where precise estimation influences operational efficiency and safety. In this paper, we propose an enhanced deep learning model based on the ConvNeXt architecture, augmented with both self-attention and channel attention mechanisms. Building upon the foundation of ConvNext, our proposed model, termed CNSCA, introduces self-attention to capture long-range spatial dependencies and channel attention to emphasize informative feature channels. This hybrid design enables the model to effectively capture both fine-grained local patterns and broader contextual relationships within rock imagery, leading to improved classification accuracy and robustness. We evaluate our model on a rock size classification dataset and compare it against three strong baseline. The results demonstrate that the incorporation of attention mechanisms significantly enhances the models capability for fine-grained classification tasks involving natural textures like rocks. 

**Abstract (ZH)**: 基于ConvNeXt架构结合自注意力和通道注意力机制的岩石粒度增强分类模型 

---
# Q-Sched: Pushing the Boundaries of Few-Step Diffusion Models with Quantization-Aware Scheduling 

**Title (ZH)**: Q-Sched：基于量化感知调度的Few-Step扩散模型边界探索 

**Authors**: Natalia Frumkin, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01624)  

**Abstract**: Text-to-image diffusion models are computationally intensive, often requiring dozens of forward passes through large transformer backbones. For instance, Stable Diffusion XL generates high-quality images with 50 evaluations of a 2.6B-parameter model, an expensive process even for a single batch. Few-step diffusion models reduce this cost to 2-8 denoising steps but still depend on large, uncompressed U-Net or diffusion transformer backbones, which are often too costly for full-precision inference without datacenter GPUs. These requirements also limit existing post-training quantization methods that rely on full-precision calibration. We introduce Q-Sched, a new paradigm for post-training quantization that modifies the diffusion model scheduler rather than model weights. By adjusting the few-step sampling trajectory, Q-Sched achieves full-precision accuracy with a 4x reduction in model size. To learn quantization-aware pre-conditioning coefficients, we propose the JAQ loss, which combines text-image compatibility with an image quality metric for fine-grained optimization. JAQ is reference-free and requires only a handful of calibration prompts, avoiding full-precision inference during calibration. Q-Sched delivers substantial gains: a 15.5% FID improvement over the FP16 4-step Latent Consistency Model and a 16.6% improvement over the FP16 8-step Phased Consistency Model, showing that quantization and few-step distillation are complementary for high-fidelity generation. A large-scale user study with more than 80,000 annotations further confirms Q-Sched's effectiveness on both FLUX.1[schnell] and SDXL-Turbo. 

**Abstract (ZH)**: Text-to-image扩散模型计算密集型，通常需要多次通过大型变压器骨干网络。通过少量步骤扩散模型可将此成本减少到2-8个去噪步骤，但仍依赖于大型未压缩的U-Net或扩散变压器骨干网络，这在没有数据中心GPU的情况下进行全精度推理时往往成本过高。这些要求也限制了现有的依赖于全精度校准的后训练量化方法。我们引入了Q-Sched，这是一种新的后训练量化范式，修改扩散模型调度器而非模型权重。通过调整少量步骤采样轨迹，Q-Sched实现了4倍于模型大小的全精度精度。为了学习量化感知预条件系数，我们提出了JAQ损失，这是一种结合文本-图像兼容性和图像质量度量的损失函数，以实现精细优化。JAQ是免参考的，并且只需要少量的校准提示，避免在校准过程中进行全精度推理。Q-Sched带来了显著的收益：相较于FP16 4步潜空间一致性模型，FID改进了15.5%，相较于FP16 8步分阶段一致性模型，FID改进了16.6%，表明量化和少量步骤蒸馏对高保真生成是互补的。大规模用户研究（超过80,000个注解）进一步证实了Q-Sched在FLUX.1[schnell]和SDXL-Turbo中的有效性。 

---
# O-DisCo-Edit: Object Distortion Control for Unified Realistic Video Editing 

**Title (ZH)**: O-DisCo-Edit: 对象扭曲控制以实现统一的现实视频编辑 

**Authors**: Yuqing Chen, Junjie Wang, Lin Liu, Ruihang Chu, Xiaopeng Zhang, Qi Tian, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01596)  

**Abstract**: Diffusion models have recently advanced video editing, yet controllable editing remains challenging due to the need for precise manipulation of diverse object properties. Current methods require different control signal for diverse editing tasks, which complicates model design and demands significant training resources. To address this, we propose O-DisCo-Edit, a unified framework that incorporates a novel object distortion control (O-DisCo). This signal, based on random and adaptive noise, flexibly encapsulates a wide range of editing cues within a single representation. Paired with a "copy-form" preservation module for preserving non-edited regions, O-DisCo-Edit enables efficient, high-fidelity editing through an effective training paradigm. Extensive experiments and comprehensive human evaluations consistently demonstrate that O-DisCo-Edit surpasses both specialized and multitask state-of-the-art methods across various video editing tasks. this https URL 

**Abstract (ZH)**: 基于对象 distortion 控制的统一视频编辑框架 O-DisCo-Edit 

---
# Unified Supervision For Vision-Language Modeling in 3D Computed Tomography 

**Title (ZH)**: 3D计算机断层成像中的统一监督的视觉-语言建模 

**Authors**: Hao-Chih Lee, Zelong Liu, Hamza Ahmed, Spencer Kim, Sean Huver, Vishwesh Nath, Zahi A. Fayad, Timothy Deyer, Xueyan Mei  

**Link**: [PDF](https://arxiv.org/pdf/2509.01554)  

**Abstract**: General-purpose vision-language models (VLMs) have emerged as promising tools in radiology, offering zero-shot capabilities that mitigate the need for large labeled datasets. However, in high-stakes domains like diagnostic radiology, these models often lack the discriminative precision required for reliable clinical use. This challenge is compounded by the scarcity and heterogeneity of publicly available volumetric CT datasets, which vary widely in annotation formats and granularity. To address these limitations, we introduce Uniferum, a volumetric VLM that unifies diverse supervision signals, encoded in classification labels and segmentation masks, into a single training framework. By harmonizing three public 3D CT datasets with distinct annotations, Uniferum achieves state-of-the-art performance, improving AUROC on the CT-RATE benchmark by 7% compared to CLIP-based and conventional multi-label convolutional models. The model demonstrates robust out-of-distribution generalization, with observed evidence of unexpected zero-shot performance on the RAD-CHEST and INSPECT datasets. Our results highlight the effectiveness of integrating heterogeneous annotations and body segmentation to enhance model performance, setting a new direction for clinically reliable, data-efficient VLMs in 3D medical imaging. 

**Abstract (ZH)**: 通用视觉-语言模型（VLMs）在放射学中 emerged 作为有前景的工具，提供零 shot 能力以减轻对大量标记数据集的需求。然而，在如诊断放射学这样的高风险领域，这些模型往往缺乏可靠临床应用所需的区分精度。这一挑战因可用的公开巣体积CT数据集稀缺且格式和粒度各异而加剧。为了应对这些限制，我们介绍了 Uniferum，这是一种统一多样指导信号的巺体积 VLM，这些指导信号以分类标签和分割掩码的形式编码，统一到一个训练框架中。通过将三个具有不同注释的公开 3D CT 数据集协调一致，Uniferum 达到了最先进的性能，比基于 CLIP 的和传统的多标签卷积模型在 CT-RATE 基准上的 AUROC 提高了 7%。该模型展示了稳健的离群分布泛化能力，并在 RAD-CHEST 和 INSPECT 数据集中观察到意外的零 shot 表现。我们的结果突显了集成异质注释和身体分割以增强模型性能的有效性，为 3D 医学影像中临床可靠、数据高效 VLM 设定了新方向。 

---
# SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization 

**Title (ZH)**: SoccerHigh：自动足球视频摘要生成的标准数据集 

**Authors**: Artur Díaz-Juan, Coloma Ballester, Gloria Haro  

**Link**: [PDF](https://arxiv.org/pdf/2509.01439)  

**Abstract**: Video summarization aims to extract key shots from longer videos to produce concise and informative summaries. One of its most common applications is in sports, where highlight reels capture the most important moments of a game, along with notable reactions and specific contextual events. Automatic summary generation can support video editors in the sports media industry by reducing the time and effort required to identify key segments. However, the lack of publicly available datasets poses a challenge in developing robust models for sports highlight generation. In this paper, we address this gap by introducing a curated dataset for soccer video summarization, designed to serve as a benchmark for the task. The dataset includes shot boundaries for 237 matches from the Spanish, French, and Italian leagues, using broadcast footage sourced from the SoccerNet dataset. Alongside the dataset, we propose a baseline model specifically designed for this task, which achieves an F1 score of 0.3956 in the test set. Furthermore, we propose a new metric constrained by the length of each target summary, enabling a more objective evaluation of the generated content. The dataset and code are available at this https URL. 

**Abstract (ZH)**: 体育视频摘要旨在从较长的视频中提取关键片段，生成简洁而富有信息性的概要。其最常见的应用之一是在体育领域，重点捕捉比赛中的关键时刻以及显著的反应和特定的上下文事件。自动摘要生成可以支持体育媒体行业的视频编辑，减少识别关键段落所需的时间和精力。然而，缺乏公开可用的数据集为开发稳健的体育亮点生成模型带来了挑战。在本文中，我们通过引入一个精心策划的足球视频摘要数据集来解决这一问题，该数据集旨在作为该任务的基准。数据集包括从SoccerNet数据集中广播的西班牙、法国和意大利联赛的237场比赛的镜头边界。此外，我们还提出了一种针对该任务的基线模型，在测试集上达到0.3956的F1分数。我们还提出了一种新的评价指标，该指标受每个目标摘要长度的约束，以便更客观地评估生成的内容。数据集和代码可在以下链接获取： this https URL。 

---
# Uirapuru: Timely Video Analytics for High-Resolution Steerable Cameras on Edge Devices 

**Title (ZH)**: uirapuru：边缘设备上及时分析高分辨率可调摄像头视频 

**Authors**: Guilherme H. Apostolo, Pablo Bauszat, Vinod Nigade, Henri E. Bal, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01371)  

**Abstract**: Real-time video analytics on high-resolution cameras has become a popular technology for various intelligent services like traffic control and crowd monitoring. While extensive work has been done on improving analytics accuracy with timing guarantees, virtually all of them target static viewpoint cameras. In this paper, we present Uirapuru, a novel framework for real-time, edge-based video analytics on high-resolution steerable cameras. The actuation performed by those cameras brings significant dynamism to the scene, presenting a critical challenge to existing popular approaches such as frame tiling. To address this problem, Uirapuru incorporates a comprehensive understanding of camera actuation into the system design paired with fast adaptive tiling at a per-frame level. We evaluate Uirapuru on a high-resolution video dataset, augmented by pan-tilt-zoom (PTZ) movements typical for steerable cameras and on real-world videos collected from an actual PTZ camera. Our experimental results show that Uirapuru provides up to 1.45x improvement in accuracy while respecting specified latency budgets or reaches up to 4.53x inference speedup with on-par accuracy compared to state-of-the-art static camera approaches. 

**Abstract (ZH)**: 高分辨率可变视角摄像头上的实时边缘视频分析框架：Uirapuru 

---
# Generalizable Self-supervised Monocular Depth Estimation with Mixture of Low-Rank Experts for Diverse Endoscopic Scenes 

**Title (ZH)**: 适用于多样化内窥镜场景的混合低秩专家通用自监督单目深度估计 

**Authors**: Liangjing Shao, Benshuang Chen, Chenkang Du, Xueli Liu, Xinrong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01206)  

**Abstract**: Self-supervised monocular depth estimation is a significant task for low-cost and efficient three-dimensional scene perception in endoscopy. The variety of illumination conditions and scene features is still the primary challenge for generalizable depth estimation in endoscopic scenes. In this work, a self-supervised framework is proposed for monocular depth estimation in various endoscopy. Firstly, due to various features in endoscopic scenes with different tissues, a novel block-wise mixture of dynamic low-rank experts is proposed to efficiently finetuning the foundation model for endoscopic depth estimation. In the proposed module, based on the input feature, different experts with a small amount of trainable parameters are adaptively selected for weighted inference, from various mixture of low-rank experts which are allocated based on the training quality of each block. Moreover, a novel self-supervised training framework is proposed to jointly cope with the inconsistency of brightness and reflectance. The proposed method outperform state-of-the-art works on both realistic and simulated endoscopic datasets. Furthermore, the proposed network also achieves the best generalization based on zero-shot depth estimation on diverse endoscopic scenes. The proposed method could contribute to accurate endoscopic perception for minimally invasive measurement and surgery. The code will be released upon acceptance, while the demo video can be found on here: this https URL. 

**Abstract (ZH)**: 自监督单目深度估计在内窥镜三维场景感知中的应用：一种针对不同照明条件和场景特征的可推广深度估计方法 

---
# FocusDPO: Dynamic Preference Optimization for Multi-Subject Personalized Image Generation via Adaptive Focus 

**Title (ZH)**: FocusDPO：多主题个性化图像生成的自适应聚焦动态偏好优化 

**Authors**: Qiaoqiao Jin, Siming Fu, Dong She, Weinan Jia, Hualiang Wang, Mu Liu, Jidong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01181)  

**Abstract**: Multi-subject personalized image generation aims to synthesize customized images containing multiple specified subjects without requiring test-time optimization. However, achieving fine-grained independent control over multiple subjects remains challenging due to difficulties in preserving subject fidelity and preventing cross-subject attribute leakage. We present FocusDPO, a framework that adaptively identifies focus regions based on dynamic semantic correspondence and supervision image complexity. During training, our method progressively adjusts these focal areas across noise timesteps, implementing a weighted strategy that rewards information-rich patches while penalizing regions with low prediction confidence. The framework dynamically adjusts focus allocation during the DPO process according to the semantic complexity of reference images and establishes robust correspondence mappings between generated and reference subjects. Extensive experiments demonstrate that our method substantially enhances the performance of existing pre-trained personalized generation models, achieving state-of-the-art results on both single-subject and multi-subject personalized image synthesis benchmarks. Our method effectively mitigates attribute leakage while preserving superior subject fidelity across diverse generation scenarios, advancing the frontier of controllable multi-subject image synthesis. 

**Abstract (ZH)**: 多主题个性化图像生成旨在合成包含多个指定主题的定制图像，而不需在测试时进行优化。然而，由于保持主题保真度和防止跨主题属性泄漏的困难，实现多个主题的精细独立控制仍然具有挑战性。我们提出了FocusDPO框架，该框架基于动态语义对应和监督图像复杂度自适应地识别焦点区域。在训练过程中，我们的方法逐步调整这些焦点区域，实施一种加权策略，奖励信息丰富的斑块，同时惩罚低预测置信度的区域。该框架在DPO过程中根据参考图像的语义复杂度动态调整焦点分配，并在生成和参考主题之间建立稳健的对应映射。 extensive实验表明，我们的方法显著增强了现有预训练个性化生成模型的性能，在单主题和多主题个性化图像合成基准测试中取得了最先进的成果。我们的方法有效减轻了属性泄漏，同时在多种生成场景中保持了出色的主题保真度，推动了可控多主题图像合成的前沿。 

---
# Seeing through Unclear Glass: Occlusion Removal with One Shot 

**Title (ZH)**: 透过模糊玻璃：一瞥去除遮挡 

**Authors**: Qiang Li, Yuanming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.01033)  

**Abstract**: Images taken through window glass are often degraded by contaminants adhered to the glass surfaces. Such contaminants cause occlusions that attenuate the incoming light and scatter stray light towards the camera. Most of existing deep learning methods for neutralizing the effects of contaminated glasses relied on synthetic training data. Few researchers used real degraded and clean image pairs, but they only considered removing or alleviating the effects of rain drops on glasses. This paper is concerned with the more challenging task of learning the restoration of images taken through glasses contaminated by a wide range of occluders, including muddy water, dirt and other small foreign particles found in reality. To facilitate the learning task we have gone to a great length to acquire real paired images with and without glass contaminants. More importantly, we propose an all-in-one model to neutralize contaminants of different types by utilizing the one-shot test-time adaptation mechanism. It involves a self-supervised auxiliary learning task to update the trained model for the unique occlusion type of each test image. Experimental results show that the proposed method outperforms the state-of-the-art methods quantitatively and qualitatively in cleaning realistic contaminated images, especially the unseen ones. 

**Abstract (ZH)**: 通过窗玻璃拍摄的图像通常会受到粘附在玻璃表面的污染物的降解。这些污染物会导致遮挡，减弱入射光并散射杂散光至相机。大多数现有的深度学习方法依赖于合成训练数据来消除污染玻璃的影响。少数研究者使用了真实降级和干净图像对，但他们仅考虑去除或减轻雨滴对玻璃的影响。本文关注更具挑战性的任务，即学习消除各种遮挡物（包括泥土水、污渍和其他现实中的小型外来颗粒）污染玻璃后拍摄图像的恢复。为了便于学习任务，我们极力获取了带有和不带玻璃污染物的真实成对图像。更重要的是，我们提出了一种全能模型，通过利用单次测试时自适应机制来消除不同类型污染物的影响。该模型利用自监督辅助学习任务来更新训练模型，以适应每张测试图像的独特遮挡类型。实验结果表明，所提出的方法在清洁真实污染图像（尤其是未见过的图像）方面在定量和定性上均优于现有最先进的方法。 

---
# Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion 

**Title (ZH)**: 超越常规：基于全景和视频扩散的两阶段场景视图生成 

**Authors**: Xueyang Kang, Zhengkang Xiang, Zezheng Zhang, Kourosh Khoshelham  

**Link**: [PDF](https://arxiv.org/pdf/2509.00843)  

**Abstract**: Novel view synthesis (NVS) from a single image is highly ill-posed due to large unobserved regions, especially for views that deviate significantly from the input. While existing methods focus on consistency between the source and generated views, they often fail to maintain coherence and correct view alignment across long-range or looped trajectories. We propose a model that addresses this by decomposing single-view NVS into a 360-degree scene extrapolation followed by novel view interpolation. This design ensures long-term view and scene consistency by conditioning on keyframes extracted and warped from a generated panoramic representation. In the first stage, a panorama diffusion model learns the scene prior from the input perspective image. Perspective keyframes are then sampled and warped from the panorama and used as anchor frames in a pre-trained video diffusion model, which generates novel views through a proposed spatial noise diffusion process. Compared to prior work, our method produces globally consistent novel views -- even in loop closure scenarios -- while enabling flexible camera control. Experiments on diverse scene datasets demonstrate that our approach outperforms existing methods in generating coherent views along user-defined trajectories. Our implementation is available at this https URL. 

**Abstract (ZH)**: 单图像新型视图合成（NVS）由于未观察区域较大而高度病态，尤其是在输入视图差异较大的情况下。现有方法往往侧重于源视图与生成视图之间的一致性，但在长距离或循环轨迹中难以保持视图一致性和正确的视图对齐。我们提出了一种通过将单视角NVS分解为全景场景外推和新型视图插值的设计，以此确保基于从生成全景表示中提取和变形的关键帧条件，实现长期视图和场景一致性。在第一阶段，全景扩散模型从输入视角图像中学习场景先验。然后从全景中采样和变形视角关键帧，并作为预训练的视频扩散模型中的锚帧，通过提出的空间噪声扩散过程生成新型视图。与先前方法相比，我们的方法在环回闭合场景中也能生成全局一致的新型视图，并支持灵活的相机控制。实验表明，我们的方法在生成沿用户定义轨迹的一致视图方面优于现有方法。我们的实现可在以下链接获取。 

---
# Adaptive Vehicle Speed Classification via BMCNN with Reinforcement Learning-Enhanced Acoustic Processing 

**Title (ZH)**: 基于强化学习增强声学处理的BMCNN自适应车辆速度分类 

**Authors**: Yuli Zhang, Pengfei Fan, Ruiyuan Jiang, Hankang Gu, Dongyao Jia, Xinheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00839)  

**Abstract**: Traffic congestion remains a pressing urban challenge, requiring intelligent transportation systems for real-time management. We present a hybrid framework that combines deep learning and reinforcement learning for acoustic vehicle speed classification. A dual-branch BMCNN processes MFCC and wavelet features to capture complementary frequency patterns. An attention-enhanced DQN adaptively selects the minimal number of audio frames and triggers early decisions once confidence thresholds are reached. Evaluations on IDMT-Traffic and our SZUR-Acoustic (Suzhou) datasets show 95.99% and 92.3% accuracy, with up to 1.63x faster average processing via early termination. Compared with A3C, DDDQN, SA2C, PPO, and TD3, the method provides a superior accuracy-efficiency trade-off and is suitable for real-time ITS deployment in heterogeneous urban environments. 

**Abstract (ZH)**: 交通拥堵仍然是一个紧迫的城市挑战，需要智能交通系统进行实时管理。我们提出一种结合深度学习和强化学习的混合框架，用于声学车辆速度分类。双分支BMCNN处理MFCC和小波特征以捕捉互补的频率模式。注意力增强的DQN自适应选择最少的音频帧数，并在置信度阈值达到时触发早期决策。在IDMT-Traffic和我们的SZUR-Acoustic（苏州）数据集上的评估显示95.99%和92.3%的准确率，并通过早期终止实现平均处理速度最多提升1.63倍。与A3C、DDDQN、SA2C、PPO和TD3相比，该方法提供更优的准确率-效率Trade-off，并适合部署在异质城市环境中。 

---
# Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization 

**Title (ZH)**: 多阶段优化生成对抗样本的序列差异最大化 

**Authors**: Xinlei Liu, Tao Hu, Peng Yi, Weitao Han, Jichao Xie, Baolin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00826)  

**Abstract**: Efficient adversarial attack methods are critical for assessing the robustness of computer vision models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing the difference between the non-true labels' probability upper bound and the true label's probability," and propose a gradient-based attack method termed Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the negative probability of the true label is used as the loss function to compress the solution space; in subsequent stages, we introduce the Directional Probability Difference Ratio (DPDR) loss function to gradually increase the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to enhance their defensive effects. The code is available at this https URL. 

**Abstract (ZH)**: 高效的对抗攻击方法对于评估计算机视觉模型的鲁棒性至关重要。本文重构了生成对抗样本的优化目标为“最大化非真实标签概率上界与真实标签概率之间的差异”，并提出了一种基于梯度的攻击方法，称为Sequential Difference Maximization (SDM)。SDM建立了一个三层优化框架“循环-步骤-阶段”。循环之间和迭代步骤之间的过程分别相同，而优化阶段在损失函数方面不同：在初始阶段，使用真实标签的负概率作为损失函数以压缩解空间；在后续阶段，引入方向概率差异比(DPDR)损失函数以逐步通过压缩无关标签的概率来增加非真实标签概率的上界。实验表明，与之前的SOTA方法相比，SDM不仅具有更强的攻击性能，而且还实现了更高的攻击成本效益。此外，SDM可以与对抗训练方法结合使用，以增强其防御效果。代码可在以下链接获取：this https URL。 

---
# Adaptive Contrast Adjustment Module: A Clinically-Inspired Plug-and-Play Approach for Enhanced Fetal Plane Classification 

**Title (ZH)**: 自适应对比度调整模块：一种临床启发的即插即用方法，用于增强胎儿切面分类 

**Authors**: Yang Chen, Sanglin Zhao, Baoyu Chen, Mans Gustaf  

**Link**: [PDF](https://arxiv.org/pdf/2509.00808)  

**Abstract**: Fetal ultrasound standard plane classification is essential for reliable prenatal diagnosis but faces inherent challenges, including low tissue contrast, boundary ambiguity, and operator-dependent image quality variations. To overcome these limitations, we propose a plug-and-play adaptive contrast adjustment module (ACAM), whose core design is inspired by the clinical practice of doctors adjusting image contrast to obtain clearer and more discriminative structural information. The module employs a shallow texture-sensitive network to predict clinically plausible contrast parameters, transforms input images into multiple contrast-enhanced views through differentiable mapping, and fuses them within downstream classifiers. Validated on a multi-center dataset of 12,400 images across six anatomical categories, the module consistently improves performance across diverse models, with accuracy of lightweight models increasing by 2.02 percent, accuracy of traditional models increasing by 1.29 percent, and accuracy of state-of-the-art models increasing by 1.15 percent. The innovation of the module lies in its content-aware adaptation capability, replacing random preprocessing with physics-informed transformations that align with sonographer workflows while improving robustness to imaging heterogeneity through multi-view fusion. This approach effectively bridges low-level image features with high-level semantics, establishing a new paradigm for medical image analysis under real-world image quality variations. 

**Abstract (ZH)**: 胎儿超声标准切面分类对于可靠的产前诊断至关重要，但面临固有的挑战，包括低组织对比度、边界模糊以及操作者依赖的图像质量变化。为克服这些局限，我们提出了一种可插拔自适应对比度调整模块（ACAM），其核心设计灵感来源于医生通过调整图像对比度以获得更清晰、更具区分性的结构信息的临床实践。该模块采用浅层纹理敏感网络来预测临床合理的对比参数，通过可微映射将输入图像转换为多种对比增强视图，并在下游分类器中融合这些视图。该模块在涵盖六大解剖类别共12,400张图像的多中心数据集上验证，能够跨多种模型一致地提高性能，轻量模型的准确率提高2.02个百分点，传统模型的准确率提高1.29个百分点，先进模型的准确率提高1.15个百分点。模块的创新点在于其内容感知自适应能力，用基于物理的变换替代随机预处理，这些变换与超声技师的工作流程一致，并通过多视图融合提升对成像异质性的鲁棒性。这种方法有效地将低级图像特征与高级语义相结合，建立了在实际图像质量变化条件下医疗图像分析的新范式。 

---
# Causal Interpretation of Sparse Autoencoder Features in Vision 

**Title (ZH)**: 视觉中稀疏自动编码器特征的因果解释 

**Authors**: Sangyu Han, Yearim Kim, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2509.00749)  

**Abstract**: Understanding what sparse auto-encoder (SAE) features in vision transformers truly represent is usually done by inspecting the patches where a feature's activation is highest. However, self-attention mixes information across the entire image, so an activated patch often co-occurs with-but does not cause-the feature's firing. We propose Causal Feature Explanation (CaFE), which leverages Effective Receptive Field (ERF). We consider each activation of an SAE feature to be a target and apply input-attribution methods to identify the image patches that causally drive that activation. Across CLIP-ViT features, ERF maps frequently diverge from naive activation maps, revealing hidden context dependencies (e.g., a "roaring face" feature that requires the co-occurrence of eyes and nose, rather than merely an open mouth). Patch insertion tests confirm that CaFE more effectively recovers or suppresses feature activations than activation-ranked patches. Our results show that CaFE yields more faithful and semantically precise explanations of vision-SAE features, highlighting the risk of misinterpretation when relying solely on activation location. 

**Abstract (ZH)**: 探讨视觉变换器中稀疏自编码器（SAE）特征的实际含义通常通过检查特征激活最高的patches来进行。然而，自注意力机制在整个图像中混合信息，因此激活的patch往往与特征的激活关联出现但并不导致特征的激活。我们提出因果特征解释（CaFE），利用有效感受野（ERF）。我们将每个SAE特征的激活视为目标，并应用输入归因方法来识别因果驱动该激活的图像patches。在CLIP-ViT特征中，ERF映射经常与简单的激活映射存在差异，揭示了隐藏的上下文依赖性（例如，一个“咆哮的脸”特征需要眼睛和鼻子的共现，而不仅仅是张开的嘴巴）。补丁插入测试证实，CaFE比按激活排序的补丁更有效地恢复或抑制特征激活。我们的结果表明，CaFE提供了更加忠实且语义精确的视觉-SAE特征解释，突显了仅依赖激活位置可能导致误解的风险。 

---
# LatentEdit: Adaptive Latent Control for Consistent Semantic Editing 

**Title (ZH)**: LatentEdit：自适应潜在控制以实现一致的语义编辑 

**Authors**: Siyi Liu, Weiming Chen, Yushun Tang, Zhihai He  

**Link**: [PDF](https://arxiv.org/pdf/2509.00541)  

**Abstract**: Diffusion-based Image Editing has achieved significant success in recent years. However, it remains challenging to achieve high-quality image editing while maintaining the background similarity without sacrificing speed or memory efficiency. In this work, we introduce LatentEdit, an adaptive latent fusion framework that dynamically combines the current latent code with a reference latent code inverted from the source image. By selectively preserving source features in high-similarity, semantically important regions while generating target content in other regions guided by the target prompt, LatentEdit enables fine-grained, controllable editing. Critically, the method requires no internal model modifications or complex attention mechanisms, offering a lightweight, plug-and-play solution compatible with both UNet-based and DiT-based architectures. Extensive experiments on the PIE-Bench dataset demonstrate that our proposed LatentEdit achieves an optimal balance between fidelity and editability, outperforming the state-of-the-art method even in 8-15 steps. Additionally, its inversion-free variant further halves the number of neural function evaluations and eliminates the need for storing any intermediate variables, substantially enhancing real-time deployment efficiency. 

**Abstract (ZH)**: 基于扩散的图像编辑已在近年来取得了显著成功。然而，在保持背景相似性的同时不牺牲速度或内存效率进行高质量图像编辑仍具有挑战性。本文介绍了LatentEdit，这是一个自适应的潜在融合框架，可以通过动态结合当前的潜在代码和从源图像反向推理出的参考潜在代码实现。通过在高相似性和语义重要区域保留源特征，在其他区域根据目标提示生成目标内容，LatentEdit实现了细致可控的编辑。关键的是，该方法不需要内部模型修改或复杂的注意力机制，提供了一个轻量级且即插即用的解决方案，兼容基于UNet和DiT的架构。在PIE-Bench数据集上的大量实验表明，我们提出的LatentEdit在保真度和编辑性之间达到了最优平衡，即使在8-15步中也超越了最先进的方法。此外，其无反演变体进一步减少了神经网络函数评估次数，并消除了存储中间变量的需求，显著提升了实时部署效率。 

---
# Multi-Focused Video Group Activities Hashing 

**Title (ZH)**: 多焦点视频组活动哈希 

**Authors**: Zhongmiao Qi, Yan Jiang, Bolin Zhang, Lijun Guo, Chong Wang, Qiangbo Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.00490)  

**Abstract**: With the explosive growth of video data in various complex scenarios, quickly retrieving group activities has become an urgent problem. However, many tasks can only retrieve videos focusing on an entire video, not the activity granularity. To solve this problem, we propose a new STVH (spatiotemporal interleaved video hashing) technique for the first time. Through a unified framework, the STVH simultaneously models individual object dynamics and group interactions, capturing the spatiotemporal evolution on both group visual features and positional features. Moreover, in real-life video retrieval scenarios, it may sometimes require activity features, while at other times, it may require visual features of objects. We then further propose a novel M-STVH (multi-focused spatiotemporal video hashing) as an enhanced version to handle this difficult task. The advanced method incorporates hierarchical feature integration through multi-focused representation learning, allowing the model to jointly focus on activity semantics features and object visual features. We conducted comparative experiments on publicly available datasets, and both STVH and M-STVH can achieve excellent results. 

**Abstract (ZH)**: 随着复杂场景下视频数据的爆炸式增长，快速检索群体活动成为一个迫切的问题。然而，许多任务只能检索整个视频，而不是活动颗粒度。为了解决这一问题，我们首次提出了一种新的STVH（时空交错视频哈希）技术。通过一个统一框架，STVH 同时建模individual对象动力学和群体互动，捕获群体视觉特征和位置特征的时空演化。此外，在实际的视频检索情景中，有时需要活动特征，有时又需要对象的视觉特征。为此，我们进一步提出了一种新的M-STVH（多聚焦时空视频哈希）作为增强版本来处理这一难题。该高级方法通过多聚焦表示学习整合层级特征，使模型能够同时关注活动语义特征和对象视觉特征。我们在公开可用的数据集上进行了比较实验，STVH 和 M-STVH 均取得了优异的结果。 

---
# DAOVI: Distortion-Aware Omnidirectional Video Inpainting 

**Title (ZH)**: DAOVI：失真感知全景视频修复 

**Authors**: Ryosuke Seshimo, Mariko Isogawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00396)  

**Abstract**: Omnidirectional videos that capture the entire surroundings are employed in a variety of fields such as VR applications and remote sensing. However, their wide field of view often causes unwanted objects to appear in the videos. This problem can be addressed by video inpainting, which enables the natural removal of such objects while preserving both spatial and temporal consistency. Nevertheless, most existing methods assume processing ordinary videos with a narrow field of view and do not tackle the distortion in equirectangular projection of omnidirectional videos. To address this issue, this paper proposes a novel deep learning model for omnidirectional video inpainting, called Distortion-Aware Omnidirectional Video Inpainting (DAOVI). DAOVI introduces a module that evaluates temporal motion information in the image space considering geodesic distance, as well as a depth-aware feature propagation module in the feature space that is designed to address the geometric distortion inherent to omnidirectional videos. The experimental results demonstrate that our proposed method outperforms existing methods both quantitatively and qualitatively. 

**Abstract (ZH)**: 全景视频 inpainting 技术在虚拟现实应用和遥感等领域中捕获全视角环境，然而其宽广视野往往会导致视频中出现不必要的物体。这可以通过视频 inpainting 来解决，进而实现这些物体的自然移除，同时保持空间和时间的一致性。然而，现有的大多数方法假设处理具有狭窄视野的普通视频，并未解决全景视频 equirectangular 投影中的失真问题。为了解决这一问题，本文提出了一种用于全景视频 inpainting 的新型深度学习模型，称为感知失真全景视频 inpainting（DAOVI）。DAOVI 引入了一个模块，该模块在图像空间中考虑测地距离来评估时间运动信息，并设计了一个深度感知特征传播模块，以应对全景视频固有的几何失真问题。实验结果表明，所提出的方法在定量和定性方面均优于现有方法。 

---
# Activation Steering Meets Preference Optimization: Defense Against Jailbreaks in Vision Language Models 

**Title (ZH)**: 激活调控结合偏好优化：视觉语言模型抵御逃逸攻击的方法 

**Authors**: Sihao Wu, Gaojie Jin, Wei Huang, Jianhong Wang, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00373)  

**Abstract**: Vision Language Models (VLMs) have demonstrated impressive capabilities in integrating visual and textual information for understanding and reasoning, but remain highly vulnerable to adversarial attacks. While activation steering has emerged as a promising defence, existing approaches often rely on task-specific contrastive prompts to extract harmful directions, which exhibit suboptimal performance and can degrade visual grounding performance. To address these limitations, we propose \textit{Sequence-Level Preference Optimization} for VLM (\textit{SPO-VLM}), a novel two-stage defense framework that combines activation-level intervention with policy-level optimization to enhance model robustness. In \textit{Stage I}, we compute adaptive layer-specific steering vectors from diverse data sources, enabling generalized suppression of harmful behaviors during inference. In \textit{Stage II}, we refine these steering vectors through a sequence-level preference optimization process. This stage integrates automated toxicity assessment, as well as visual-consistency rewards based on caption-image alignment, to achieve safe and semantically grounded text generation. The two-stage structure of SPO-VLM balances efficiency and effectiveness by combining a lightweight mitigation foundation in Stage I with deeper policy refinement in Stage II. Extensive experiments shown SPO-VLM enhances safety against attacks via activation steering and preference optimization, while maintaining strong performance on benign tasks without compromising visual understanding capabilities. We will release our code, model weights, and evaluation toolkit to support reproducibility and future research. \textcolor{red}{Warning: This paper may contain examples of offensive or harmful text and images.} 

**Abstract (ZH)**: Vision Language模型（VLMs）在整合视觉和文本信息以实现理解和推理方面展现了令人印象深刻的潜力，但仍然高度易受对抗攻击的影响。虽然激活导向已成为一种有潜力的防御方法，现有的方法通常依赖于特定任务对比提示来提取有害方向，这表现出次优性能并可能降低视觉接地性能。为了解决这些局限性，我们提出了一种新的两阶段防御框架——基于序列级偏好优化的Vision Language模型（SPO-VLM），该框架结合了激活级别干预与策略级别优化，以增强模型的鲁棒性。在第一阶段，我们从多种数据源中计算自适应的分层特定导向向量，使在推理过程中能够泛化抑制有害行为。在第二阶段，我们通过序列级偏好优化过程进一步细化这些导向向量。该阶段结合了自动毒性评估和基于描述图匹配的视觉一致性奖励，以实现安全和语义上适当的文本生成。SPO-VLM的两阶段结构通过在第一阶段引入轻量级缓解基础与在第二阶段进行深入策略优化相结合，平衡了效率与效果。大量实验表明，SPO-VLM通过激活导向和偏好优化增强了模型的安全性，同时在保持良性任务高性能的同时不牺牲视觉理解能力。我们将发布我们的代码、模型权重和评估工具包以支持可重复性和未来的研究。请注意，本论文可能包含冒犯性或有害的文本和图像示例。 

---
# Generative AI for Industrial Contour Detection: A Language-Guided Vision System 

**Title (ZH)**: 生成式AI在工业轮廓检测中的应用：一种语言引导的视觉系统 

**Authors**: Liang Gong, Tommy, Wang, Sara Chaker, Yanchen Dong, Fouad Bousetouane, Brenden Morton, Mark Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2509.00284)  

**Abstract**: Industrial computer vision systems often struggle with noise, material variability, and uncontrolled imaging conditions, limiting the effectiveness of classical edge detectors and handcrafted pipelines. In this work, we present a language-guided generative vision system for remnant contour detection in manufacturing, designed to achieve CAD-level precision. The system is organized into three stages: data acquisition and preprocessing, contour generation using a conditional GAN, and multimodal contour refinement through vision-language modeling, where standardized prompts are crafted in a human-in-the-loop process and applied through image-text guided synthesis. On proprietary FabTrack datasets, the proposed system improved contour fidelity, enhancing edge continuity and geometric alignment while reducing manual tracing. For the refinement stage, we benchmarked several vision-language models, including Google's Gemini 2.0 Flash, OpenAI's GPT-image-1 integrated within a VLM-guided workflow, and open-source baselines. Under standardized conditions, GPT-image-1 consistently outperformed Gemini 2.0 Flash in both structural accuracy and perceptual quality. These findings demonstrate the promise of VLM-guided generative workflows for advancing industrial computer vision beyond the limitations of classical pipelines. 

**Abstract (ZH)**: 工业计算机视觉系统常常受到噪声、材料变异性以及无法控制的成像条件的限制，这限制了经典边缘检测器和手工制作管道的有效性。本文提出了一种语言引导的生成性视觉系统，用于制造中的残余轮廓检测，旨在实现CAD级精度。该系统分为三个阶段：数据采集和预处理、使用条件GAN的轮廓生成，以及通过视觉-语言模型进行多模态轮廓细化，在这个过程中通过人工在环过程制作标准化提示并通过图像-文本引导合成应用。在专属的FabTrack数据集中，所提出的系统提高了轮廓保真度，增强了边缘连续性和几何对齐，并减少了手工绘图的工作量。在细化阶段，我们benchmark了几种视觉-语言模型，包括Google的Gemini 2.0 Flash、OpenAI的GPT-image-1嵌入在VLM引导的工作流中以及开源基准。在标准化条件下，GPT-image-1在结构准确性和感知质量方面始终优于Gemini 2.0 Flash。这些发现证明了视觉-语言模型引导的生成性工作流在超越传统管道限制的工业计算机视觉方面的潜力。 

---
# Amplifying Emotional Signals: Data-Efficient Deep Learning for Robust Speech Emotion Recognition 

**Title (ZH)**: 增强情感信号：高效数据深度学习在稳健语音情感识别中的应用 

**Authors**: Tai Vu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00077)  

**Abstract**: Speech Emotion Recognition (SER) presents a significant yet persistent challenge in human-computer interaction. While deep learning has advanced spoken language processing, achieving high performance on limited datasets remains a critical hurdle. This paper confronts this issue by developing and evaluating a suite of machine learning models, including Support Vector Machines (SVMs), Long Short-Term Memory networks (LSTMs), and Convolutional Neural Networks (CNNs), for automated emotion classification in human speech. We demonstrate that by strategically employing transfer learning and innovative data augmentation techniques, our models can achieve impressive performance despite the constraints of a relatively small dataset. Our most effective model, a ResNet34 architecture, establishes a new performance benchmark on the combined RAVDESS and SAVEE datasets, attaining an accuracy of 66.7% and an F1 score of 0.631. These results underscore the substantial benefits of leveraging pre-trained models and data augmentation to overcome data scarcity, thereby paving the way for more robust and generalizable SER systems. 

**Abstract (ZH)**: 语音情感识别（SER）在人机交互中 presents a significant yet persistent challenge. While deep learning has advanced spoken language processing, achieving high performance on limited datasets remains a critical hurdle. This paper confronts this issue by developing and evaluating a suite of machine learning models, including Support Vector Machines (SVMs), Long Short-Term Memory networks (LSTMs), and Convolutional Neural Networks (CNNs), for automated emotion classification in human speech. We demonstrate that by strategically employing transfer learning and innovative data augmentation techniques, our models can achieve impressive performance despite the constraints of a relatively small dataset. Our most effective model, a ResNet34 architecture, establishes a new performance benchmark on the combined RAVDESS and SAVEE datasets, attaining an accuracy of 66.7% and an F1 score of 0.631. These results underscore the substantial benefits of leveraging pre-trained models and data augmentation to overcome data scarcity, thereby paving the way for more robust and generalizable SER systems。 

---
# Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion 

**Title (ZH)**: 支架扩散：基于离散扩散的稀疏多类体素结构生成 

**Authors**: Justin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.00062)  

**Abstract**: Generating realistic sparse multi-category 3D voxel structures is difficult due to the cubic memory scaling of voxel structures and moreover the significant class imbalance caused by sparsity. We introduce Scaffold Diffusion, a generative model designed for sparse multi-category 3D voxel structures. By treating voxels as tokens, Scaffold Diffusion uses a discrete diffusion language model to generate 3D voxel structures. We show that discrete diffusion language models can be extended beyond inherently sequential domains such as text to generate spatially coherent 3D structures. We evaluate on Minecraft house structures from the 3D-Craft dataset and demonstrate that, unlike prior baselines and an auto-regressive formulation, Scaffold Diffusion produces realistic and coherent structures even when trained on data with over 98% sparsity. We provide an interactive viewer where readers can visualize generated samples and the generation process. Our results highlight discrete diffusion as a promising framework for 3D sparse voxel generative modeling. 

**Abstract (ZH)**: 生成真实的稀疏多分类3D体素结构因体素结构的立方体内存缩放以及由此导致的重大类别不平衡而具有挑战性。我们引入了Scaffold Diffusion，这是一种针对稀疏多分类3D体素结构的生成模型。通过将体素视为标记，Scaffold Diffusion 使用离散扩散语言模型来生成3D体素结构。我们展示了离散扩散语言模型可以扩展到诸如文本等固有的序列领域之外，以生成空间上连贯的3D结构。我们在来自3D-Craft数据集的Minecraft房屋结构上进行评估，并证明与之前的基线方法和自回归表述相比，Scaffold Diffusion 即使在训练数据中存在超过98%的稀疏性时，也能生成真实且连贯的结构。我们提供了一个交互式查看器，读者可以在此查看生成的样本和生成过程。我们的结果突显了离散扩散作为一种有 promise 的3D 稀疏体素生成建模框架。 

---
# Lightning Fast Caching-based Parallel Denoising Prediction for Accelerating Talking Head Generation 

**Title (ZH)**: 基于缓存的快速并行去噪预测加速说话头部生成 

**Authors**: Jianzhi Long, Wenhao Sun, Rongcheng Tu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00052)  

**Abstract**: Diffusion-based talking head models generate high-quality, photorealistic videos but suffer from slow inference, limiting practical applications. Existing acceleration methods for general diffusion models fail to exploit the temporal and spatial redundancies unique to talking head generation. In this paper, we propose a task-specific framework addressing these inefficiencies through two key innovations. First, we introduce Lightning-fast Caching-based Parallel denoising prediction (LightningCP), caching static features to bypass most model layers in inference time. We also enable parallel prediction using cached features and estimated noisy latents as inputs, efficiently bypassing sequential sampling. Second, we propose Decoupled Foreground Attention (DFA) to further accelerate attention computations, exploiting the spatial decoupling in talking head videos to restrict attention to dynamic foreground regions. Additionally, we remove reference features in certain layers to bring extra speedup. Extensive experiments demonstrate that our framework significantly improves inference speed while preserving video quality. 

**Abstract (ZH)**: 基于扩散的头部动画模型生成高质量的 PHOTO-REALISTIC 视频，但在推理速度上存在局限，影响其实用性。现有的通用扩散模型加速方法未能充分利用头部动画生成特有的时域和空域冗余性。本文提出一种针对任务的框架，通过两项关键创新解决这些问题。首先，提出 Lightning-fast Caching-based Parallel Denoising Prediction (LightningCP)，在推理时缓存静态特征以跳过大部分模型层。我们还利用缓存特征和估计的噪声潜变量进行并行预测，高效地避免了顺序采样。其次，提出解耦前景注意（DFA）以进一步加速注意力计算，利用头部动画视频的空域解耦特性，将注意力限制在动态前景区域。此外，某些层中移除参考特征以带来额外的加速。大量实验证明，我们的框架在保持视频质量的同时显著提高了推理速度。 

---
# From Sound to Sight: Towards AI-authored Music Videos 

**Title (ZH)**: 从声音到视觉： towards AI作曲的音乐视频 

**Authors**: Leo Vitasovic, Stella Graßhof, Agnes Mercedes Kloft, Ville V. Lehtola, Martin Cunneen, Justyna Starostka, Glenn McGarry, Kun Li, Sami S. Brandt  

**Link**: [PDF](https://arxiv.org/pdf/2509.00029)  

**Abstract**: Conventional music visualisation systems rely on handcrafted ad hoc transformations of shapes and colours that offer only limited expressiveness. We propose two novel pipelines for automatically generating music videos from any user-specified, vocal or instrumental song using off-the-shelf deep learning models. Inspired by the manual workflows of music video producers, we experiment on how well latent feature-based techniques can analyse audio to detect musical qualities, such as emotional cues and instrumental patterns, and distil them into textual scene descriptions using a language model. Next, we employ a generative model to produce the corresponding video clips. To assess the generated videos, we identify several critical aspects and design and conduct a preliminary user evaluation that demonstrates storytelling potential, visual coherency and emotional alignment with the music. Our findings underscore the potential of latent feature techniques and deep generative models to expand music visualisation beyond traditional approaches. 

**Abstract (ZH)**: 基于自动生成模型的音乐视频制作新pipeline：从乐谱到故事的情感连贯性探索 

---
# IPG: Incremental Patch Generation for Generalized Adversarial Patch Training 

**Title (ZH)**: 基于增量片段生成的通用对抗片段训练方法 

**Authors**: Wonho Lee, Hyunsik Na, Jisu Lee, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10946)  

**Abstract**: The advent of adversarial patches poses a significant challenge to the robustness of AI models, particularly in the domain of computer vision tasks such as object detection. In contradistinction to traditional adversarial examples, these patches target specific regions of an image, resulting in the malfunction of AI models. This paper proposes Incremental Patch Generation (IPG), a method that generates adversarial patches up to 11.1 times more efficiently than existing approaches while maintaining comparable attack performance. The efficacy of IPG is demonstrated by experiments and ablation studies including YOLO's feature distribution visualization and adversarial training results, which show that it produces well-generalized patches that effectively cover a broader range of model vulnerabilities. Furthermore, IPG-generated datasets can serve as a robust knowledge foundation for constructing a robust model, enabling structured representation, advanced reasoning, and proactive defenses in AI security ecosystems. The findings of this study suggest that IPG has considerable potential for future utilization not only in adversarial patch defense but also in real-world applications such as autonomous vehicles, security systems, and medical imaging, where AI models must remain resilient to adversarial attacks in dynamic and high-stakes environments. 

**Abstract (ZH)**: adversarial 贴的出现对人工智能模型的稳健性构成了重大挑战，特别是在物体检测等计算机视觉任务领域。与传统的对抗性样本不同，这些贴片针对图像的特定区域，导致人工智能模型失效。本文提出了一种增量贴片生成（IPG）方法，该方法在保持攻击性能相近的情况下，相比现有方法更高效地生成对抗性贴片，最高可达11.1倍。通过包括YOLO特征分布可视化和对抗性训练结果在内的实验证明和消融研究，展示了IPG生成的有效且具有良好泛化能力的贴片，能够覆盖更广泛的模型漏洞。此外，IPG生成的数据集可以作为构建稳健模型的坚实知识基础，促进人工智能安全生态系统中的结构化表示、高级推理和积极防御。研究发现表明，IPG在对抗性贴片防御以及自动驾驶车辆、安全系统和医疗成像等实际应用中抵御动态和高风险环境中对抗性攻击方面具有很大的潜在利用价值。 

---
