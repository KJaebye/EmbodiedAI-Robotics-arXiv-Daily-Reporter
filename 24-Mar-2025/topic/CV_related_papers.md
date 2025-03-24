# Deep Learning for Human Locomotion Analysis in Lower-Limb Exoskeletons: A Comparative Study 

**Title (ZH)**: 深度学习在下肢外骨骼中的人体运动分析：一项比较研究 

**Authors**: Omar Coser, Christian Tamantini, Matteo Tortora, Leonardo Furia, Rosa Sicilia, Loredana Zollo, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2503.16904)  

**Abstract**: Wearable robotics for lower-limb assistance have become a pivotal area of research, aiming to enhance mobility for individuals with physical impairments or augment the performance of able-bodied users. Accurate and adaptive control systems are essential to ensure seamless interaction between the wearer and the robotic device, particularly when navigating diverse and dynamic terrains. Despite the recent advances in neural networks for time series analysis, no attempts have been directed towards the classification of ground conditions, categorized into five classes and subsequently determining the ramp's slope and stair's height. In this respect, this paper presents an experimental comparison between eight deep neural network backbones to predict high-level locomotion parameters across diverse terrains.
All the models are trained on the publicly available CAMARGO 2021 dataset. IMU-only data equally or outperformed IMU+EMG inputs, promoting a cost-effective and efficient design. Indeeds, using three IMU sensors, the LSTM achieved high terrain classification accuracy (0.94 +- 0.04) and precise ramp slope (1.95 +- 0.58°) and the CNN-LSTM a stair height (15.65 +- 7.40 mm) estimations. As a further contribution, SHAP analysis justified sensor reduction without performance loss, ensuring a lightweight setup. The system operates with ~2 ms inference time, supporting real-time applications. The code is code available at this https URL. 

**Abstract (ZH)**: 穿戴式机器人在下肢辅助领域的研究已成为关键研究方向，旨在提升身体残疾个体的移动性或增强健全个体的表现。准确且适应性强的控制系统对于确保穿戴者与机器人设备之间无缝交互至关重要，尤其是在穿越多样且动态地形时。尽管在时间序列分析的神经网络方面取得了最近的进步，但仍没有尝试对地面条件进行分类，将其分为五类，并据此确定斜坡的坡度和阶梯的高度。就此而言，本文对比了八种深度神经网络骨干模型在不同地形下预测高级步行参数的实验效果。所有模型均在公开可用的CAMARGO 2021数据集上进行训练。仅使用IMU数据或IMU+EMG输入方式可实现同等甚至更优的效果，促进成本效益高且高效的系统设计。确实，使用三个IMU传感器时，LSTM 在地形分类准确性上达到0.94 ± 0.04，并精确估计斜坡坡度为1.95 ± 0.58°，而CNN-LSTM 估计阶梯高度为15.65 ± 7.40 mm。另外，SHAP 分析证明了在不损失性能的情况下减少传感器数量的有效性，确保轻量化配置。该系统具有约2 ms的推理时间，支持实时应用。代码可通过此链接访问。 

---
# Informative Path Planning to Explore and Map Unknown Planetary Surfaces with Gaussian Processes 

**Title (ZH)**: 基于高斯过程的信息性路径规划以探索和绘制未知行星表面 

**Authors**: Ashten Akemoto, Frances Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16613)  

**Abstract**: Many environments, such as unvisited planetary surfaces and oceanic regions, remain unexplored due to a lack of prior knowledge. Autonomous vehicles must sample upon arrival, process data, and either transmit findings to a teleoperator or decide where to explore next. Teleoperation is suboptimal, as human intuition lacks mathematical guarantees for optimality. This study evaluates an informative path planning algorithm for mapping a scalar variable distribution while minimizing travel distance and ensuring model convergence. We compare traditional open loop coverage methods (e.g., Boustrophedon, Spiral) with information-theoretic approaches using Gaussian processes, which update models iteratively with confidence metrics. The algorithm's performance is tested on three surfaces, a parabola, Townsend function, and lunar crater hydration map, to assess noise, convexity, and function behavior. Results demonstrate that information-driven methods significantly outperform naive exploration in reducing model error and travel distance while improving convergence potential. 

**Abstract (ZH)**: 基于信息的路径规划算法在探索未知环境中的应用与评估 

---
# GAA-TSO: Geometry-Aware Assisted Depth Completion for Transparent and Specular Objects 

**Title (ZH)**: 几何感知辅助深度完成：透明和镜面对象的深度补全 

**Authors**: Yizhe Liu, Tong Jia, Da Cai, Hao Wang, Dongyue Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17106)  

**Abstract**: Transparent and specular objects are frequently encountered in daily life, factories, and laboratories. However, due to the unique optical properties, the depth information on these objects is usually incomplete and inaccurate, which poses significant challenges for downstream robotics tasks. Therefore, it is crucial to accurately restore the depth information of transparent and specular objects. Previous depth completion methods for these objects usually use RGB information as an additional channel of the depth image to perform depth prediction. Due to the poor-texture characteristics of transparent and specular objects, these methods that rely heavily on color information tend to generate structure-less depth predictions. Moreover, these 2D methods cannot effectively explore the 3D structure hidden in the depth channel, resulting in depth ambiguity. To this end, we propose a geometry-aware assisted depth completion method for transparent and specular objects, which focuses on exploring the 3D structural cues of the scene. Specifically, besides extracting 2D features from RGB-D input, we back-project the input depth to a point cloud and build the 3D branch to extract hierarchical scene-level 3D structural features. To exploit 3D geometric information, we design several gated cross-modal fusion modules to effectively propagate multi-level 3D geometric features to the image branch. In addition, we propose an adaptive correlation aggregation strategy to appropriately assign 3D features to the corresponding 2D features. Extensive experiments on ClearGrasp, OOD, TransCG, and STD datasets show that our method outperforms other state-of-the-art methods. We further demonstrate that our method significantly enhances the performance of downstream robotic grasping tasks. 

**Abstract (ZH)**: 透明和镜面物体在日常生活中、工厂和实验室中经常遇到。然而，由于这些物体独特的光学性质，通常很难获取其完整准确的深度信息，这对下游机器人任务构成了巨大挑战。因此，准确恢复透明和镜面物体的深度信息至关重要。针对这些物体的先前深度完成方法通常使用RGB信息作为深度图像的附加通道来进行深度预测。但由于透明和镜面物体具有较差的纹理特征，依赖于色彩信息的方法往往会生成缺乏结构的深度预测。此外，这些2D方法无法有效探索隐藏在深度通道中的3D结构，导致深度信息的不确定性。为此，我们提出了一种几何感知辅助的深度完成方法，专注于探索场景的3D结构线索。具体来说，除了从RGB-D输入中提取2D特征外，我们还将输入深度反投影至点云，并构建3D分支以提取分层场景级的3D结构特征。为了利用3D几何信息，我们设计了几种门控跨模态融合模块，以有效传播多级3D几何特征到图像分支。此外，我们提出了一种自适应相关聚合策略，以适当分配3D特征到相应的2D特征。在ClearGrasp、OOD、TransCG和STD数据集上的广泛实验表明，我们的方法优于其他最先进的方法。我们进一步证明，我们的方法显著提升了下游机器人抓取任务的性能。 

---
# SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion 

**Title (ZH)**: SGFormer: 卫星-地面融合用于3D语义场景完成 

**Authors**: Xiyue Guo, Jiarui Hu, Junjie Hu, Hujun Bao, Guofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16825)  

**Abstract**: Recently, camera-based solutions have been extensively explored for scene semantic completion (SSC). Despite their success in visible areas, existing methods struggle to capture complete scene semantics due to frequent visual occlusions. To address this limitation, this paper presents the first satellite-ground cooperative SSC framework, i.e., SGFormer, exploring the potential of satellite-ground image pairs in the SSC task. Specifically, we propose a dual-branch architecture that encodes orthogonal satellite and ground views in parallel, unifying them into a common domain. Additionally, we design a ground-view guidance strategy that corrects satellite image biases during feature encoding, addressing misalignment between satellite and ground views. Moreover, we develop an adaptive weighting strategy that balances contributions from satellite and ground views. Experiments demonstrate that SGFormer outperforms the state of the art on SemanticKITTI and SSCBench-KITTI-360 datasets. Our code is available on this https URL. 

**Abstract (ZH)**: 基于卫星-地面协同的场景语义完成框架：SGFormer探索图像对在场景语义完成任务中的潜力 

---
# A Digital Twin Simulator of a Pastillation Process with Applications to Automatic Control based on Computer Vision 

**Title (ZH)**: 基于计算机视觉的 Pastillation 过程数字孪生仿真器及其自动控制应用 

**Authors**: Leonardo D. González, Joshua L. Pulsipher, Shengli Jiang, Tyler Soderstrom, Victor M. Zavala  

**Link**: [PDF](https://arxiv.org/pdf/2503.16539)  

**Abstract**: We present a digital-twin simulator for a pastillation process. The simulation framework produces realistic thermal image data of the process that is used to train computer vision-based soft sensors based on convolutional neural networks (CNNs); the soft sensors produce output signals for temperature and product flow rate that enable real-time monitoring and feedback control. Pastillation technologies are high-throughput devices that are used in a broad range of industries; these processes face operational challenges such as real-time identification of clog locations (faults) in the rotating shell and the automatic, real-time adjustment of conveyor belt speed and operating conditions to stabilize output. The proposed simulator is able to capture this behavior and generates realistic data that can be used to benchmark different algorithms for image processing and different control architectures. We present a case study to illustrate the capabilities; the study explores behavior over a range of equipment sizes, clog locations, and clog duration. A feedback controller (tuned using Bayesian optimization) is used to adjust the conveyor belt speed based on the CNN output signal to achieve the desired process outputs. 

**Abstract (ZH)**: 我们提出了一种用于模拟结晶过程的数字孪生模拟器。该模拟框架生成了过程的现实热图像数据，用于基于卷积神经网络（CNN）的软传感器训练；软传感器生成温度和产品流率的输出信号，实现实时监控和反馈控制。结晶技术是高通量设备，广泛应用于多个行业；这些过程面临实时识别旋转壳体内堵塞位置（故障）以及自动、实时调整输送带速度和操作条件以稳定输出的操作挑战。所提出的模拟器能够捕捉这一行为，并生成可用于验证不同图像处理算法和不同控制架构的现实数据。我们提供了一个案例研究以展示其能力；该研究探讨了不同设备尺寸、堵塞位置和堵塞持续时间下的行为。使用基于贝叶斯优化调谐的反馈控制器根据CNN输出信号调整输送带速度，以实现所需的工艺输出。 

---
# Inclusive STEAM Education: A Framework for Teaching Cod-2 ing and Robotics to Students with Visually Impairment Using 3 Advanced Computer Vision 

**Title (ZH)**: 包容性STEAM教育：面向视觉 impairment 学生的高级计算机视觉技术下的编程与机器人教学框架 

**Authors**: Mahmoud Hamash, Md Raqib Khan, Peter Tiernan  

**Link**: [PDF](https://arxiv.org/pdf/2503.16482)  

**Abstract**: STEAM education integrates Science, Technology, Engineering, Arts, and Mathematics to foster creativity and problem-solving. However, students with visual impairments (VI) encounter significant challenges in programming and robotics, particularly in tracking robot movements and developing spatial awareness. This paper presents a framework that leverages pre-constructed robots and algorithms, such as maze-solving techniques, within an accessible learning environment. The proposed system employs Contrastive Language-Image Pre-training (CLIP) to process global camera-captured maze layouts, converting visual data into textual descriptions that generate spatial audio prompts in an Audio Virtual Reality (AVR) system. Students issue verbal commands, which are refined through CLIP, while robot-mounted stereo cameras provide real-time data processed via Simultaneous Localization and Mapping (SLAM) for continuous feedback. By integrating these technologies, the framework empowers VI students to develop coding skills and engage in complex problem-solving tasks. Beyond maze-solving applications, this approach demonstrates the broader potential of computer vision in special education, contributing to improved accessibility and learning experiences in STEAM disciplines. 

**Abstract (ZH)**: STEAM教育融合科学、技术、工程、艺术和数学以促进创造力和解决问题的能力，但视觉障碍（VI）学生在编程和机器人技术方面面临重大挑战，特别是在追踪机器人运动和培养空间意识方面。本文提出了一种框架，利用预制机器人和算法，如迷宫求解技术，在无障碍学习环境中运用。所提议的系统采用对比语言-图像预训练（CLIP）处理全局摄像头捕获的迷宫布局，将视觉数据转换成文本描述，在音频虚拟现实（AVR）系统中生成空间音频提示。学生发出口头命令，通过CLIP进行优化，同时，安装在机器人上的立体摄像头提供通过即时定位与地图构建（SLAM）处理的实时数据，以实现持续反馈。通过整合这些技术，该框架使视觉障碍学生能够发展编程技能并参与复杂的解决问题任务。除了迷宫求解应用，该方法展示了计算机视觉在特殊教育领域的更广泛潜力，有助于提高STEAM学科中的无障碍性和学习体验。 

---
# Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation 

**Title (ZH)**: 调整你的节奏：基于门控增强节奏感知特征表示的高对齐舞蹈姿态生成 

**Authors**: Congyi Fan, Jian Guan, Xuanjia Zhao, Dongli Xu, Youtian Lin, Tong Ye, Pengming Feng, Haiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17340)  

**Abstract**: Automatically generating natural, diverse and rhythmic human dance movements driven by music is vital for virtual reality and film industries. However, generating dance that naturally follows music remains a challenge, as existing methods lack proper beat alignment and exhibit unnatural motion dynamics. In this paper, we propose Danceba, a novel framework that leverages gating mechanism to enhance rhythm-aware feature representation for music-driven dance generation, which achieves highly aligned dance poses with enhanced rhythmic sensitivity. Specifically, we introduce Phase-Based Rhythm Extraction (PRE) to precisely extract rhythmic information from musical phase data, capitalizing on the intrinsic periodicity and temporal structures of music. Additionally, we propose Temporal-Gated Causal Attention (TGCA) to focus on global rhythmic features, ensuring that dance movements closely follow the musical rhythm. We also introduce Parallel Mamba Motion Modeling (PMMM) architecture to separately model upper and lower body motions along with musical features, thereby improving the naturalness and diversity of generated dance movements. Extensive experiments confirm that Danceba outperforms state-of-the-art methods, achieving significantly better rhythmic alignment and motion diversity. Project page: this https URL . 

**Abstract (ZH)**: 自动生成与音乐节奏自然契合、多样且富有韵律的人类舞蹈动作对于虚拟现实和电影行业至关重要。然而，生成能够自然跟随音乐的舞蹈仍然存在挑战，因为现有方法缺乏恰当的节奏对齐，表现出不自然的运动动态。在本文中，我们提出Danceba，一个新颖的框架，利用门控机制增强音乐感知特征表示，以实现高度对齐且增强节奏敏感性的舞蹈姿态。具体来说，我们引入基于相位的节奏提取（PRE）以精确提取音乐相位数据中的节奏信息，充分利用音乐的固有周期性和时间结构。此外，我们提出时间门控因果注意力（TGCA）以聚焦全局节奏特征，确保舞蹈动作紧密跟随音乐节奏。我们还引入并行马amba运动建模（PMMM）架构，分别建模上身和下身运动及其与音乐特征的关系，从而提高生成舞蹈动作的自然性和多样性。广泛实验结果显示，Danceba在节奏对齐和运动多样性方面显著优于现有方法。项目页面：this https URL。 

---
# Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID 

**Title (ZH)**: 强 baseline：基于 YOLOv12 和 BoT-SORT-ReID 的多无人机跟踪 

**Authors**: Yu-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17237)  

**Abstract**: Detecting and tracking multiple unmanned aerial vehicles (UAVs) in thermal infrared video is inherently challenging due to low contrast, environmental noise, and small target sizes. This paper provides a straightforward approach to address multi-UAV tracking in thermal infrared video, leveraging recent advances in detection and tracking. Instead of relying on the YOLOv5 with the DeepSORT pipeline, we present a tracking framework built on YOLOv12 and BoT-SORT, enhanced with tailored training and inference strategies. We evaluate our approach following the metrics from the 4th Anti-UAV Challenge and demonstrate competitive performance. Notably, we achieve strong results without using contrast enhancement or temporal information fusion to enrich UAV features, highlighting our approach as a "Strong Baseline" for the multi-UAV tracking task. We provide implementation details, in-depth experimental analysis, and a discussion of potential improvements. The code is available at this https URL . 

**Abstract (ZH)**: 基于热红外视频的多无人机检测与跟踪：一种简单有效的框架 

---
# D2Fusion: Dual-domain Fusion with Feature Superposition for Deepfake Detection 

**Title (ZH)**: D2Fusion: 双 domain 融合与特征叠加方法在深度假信息检测中的应用 

**Authors**: Xueqi Qiu, Xingyu Miao, Fan Wan, Haoran Duan, Tejal Shah, Varun Ojhab, Yang Longa, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17184)  

**Abstract**: Deepfake detection is crucial for curbing the harm it causes to society. However, current Deepfake detection methods fail to thoroughly explore artifact information across different domains due to insufficient intrinsic interactions. These interactions refer to the fusion and coordination after feature extraction processes across different domains, which are crucial for recognizing complex forgery clues. Focusing on more generalized Deepfake detection, in this work, we introduce a novel bi-directional attention module to capture the local positional information of artifact clues from the spatial domain. This enables accurate artifact localization, thus addressing the coarse processing with artifact features. To further address the limitation that the proposed bi-directional attention module may not well capture global subtle forgery information in the artifact feature (e.g., textures or edges), we employ a fine-grained frequency attention module in the frequency domain. By doing so, we can obtain high-frequency information in the fine-grained features, which contains the global and subtle forgery information. Although these features from the diverse domains can be effectively and independently improved, fusing them directly does not effectively improve the detection performance. Therefore, we propose a feature superposition strategy that complements information from spatial and frequency domains. This strategy turns the feature components into the form of wave-like tokens, which are updated based on their phase, such that the distinctions between authentic and artifact features can be amplified. Our method demonstrates significant improvements over state-of-the-art (SOTA) methods on five public Deepfake datasets in capturing abnormalities across different manipulated operations and real-life. 

**Abstract (ZH)**: 深度生成伪造检测对于遏制其对社会的危害至关重要。然而，当前的深度生成伪造检测方法由于内在交互不足，未能充分探索不同域之间的艺术信息。这些交互是指不同域在特征提取过程后的融合和协调，对于识别复杂的伪造线索至关重要。为了进行更通用的深度生成伪造检测，本工作引入了一个新颖的双向注意力模块，从空间域捕捉艺术线索的局部位置信息，从而实现准确的艺术品局部化，解决粗略处理的艺术特征问题。为了进一步解决所提议的双向注意力模块可能难以在艺术特征中很好地捕捉全局微细伪造信息（如纹理或边缘）的问题，我们在频率域中采用了细粒度频率注意力模块。通过这样做，我们可以获取细粒度特征中的高频信息，这些信息包含全局和微细的伪造信息。尽管来自不同领域的这些特征可以有效且独立地提高，直接融合它们并不能显著提高检测性能。因此，我们提出了一种特征叠加策略，以补充空间域和频率域的信息。该策略将特征组件转换为波形标记的形式，并基于其相位更新，从而放大了真实和艺术特征之间的差异。我们的方法在五个公开的深度生成伪造数据集上，在不同操作和现实生活中的异常检测方面，显著优于现有最佳方法。 

---
# Temporal-Guided Spiking Neural Networks for Event-Based Human Action Recognition 

**Title (ZH)**: 基于时间引导的事件驱动人体动作识别Spiking神经网络 

**Authors**: Siyuan Yang, Shilin Lu, Shizheng Wang, Meng Hwa Er, Zengwei Zheng, Alex C. Kot  

**Link**: [PDF](https://arxiv.org/pdf/2503.17132)  

**Abstract**: This paper explores the promising interplay between spiking neural networks (SNNs) and event-based cameras for privacy-preserving human action recognition (HAR). The unique feature of event cameras in capturing only the outlines of motion, combined with SNNs' proficiency in processing spatiotemporal data through spikes, establishes a highly synergistic compatibility for event-based HAR. Previous studies, however, have been limited by SNNs' ability to process long-term temporal information, essential for precise HAR. In this paper, we introduce two novel frameworks to address this: temporal segment-based SNN (\textit{TS-SNN}) and 3D convolutional SNN (\textit{3D-SNN}). The \textit{TS-SNN} extracts long-term temporal information by dividing actions into shorter segments, while the \textit{3D-SNN} replaces 2D spatial elements with 3D components to facilitate the transmission of temporal information. To promote further research in event-based HAR, we create a dataset, \textit{FallingDetection-CeleX}, collected using the high-resolution CeleX-V event camera $(1280 \times 800)$, comprising 7 distinct actions. Extensive experimental results show that our proposed frameworks surpass state-of-the-art SNN methods on our newly collected dataset and three other neuromorphic datasets, showcasing their effectiveness in handling long-range temporal information for event-based HAR. 

**Abstract (ZH)**: 基于事件的相机和脉冲神经网络在隐私保护的人体动作识别中的前景探索：TS-SNN和3D-SNN框架 

---
# FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields 

**Title (ZH)**: FFaceNeRF：神经辐射场中的少量样本面部编辑 

**Authors**: Kwan Yun, Chaelin Kim, Hangyeul Shin, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.17095)  

**Abstract**: Recent 3D face editing methods using masks have produced high-quality edited images by leveraging Neural Radiance Fields (NeRF). Despite their impressive performance, existing methods often provide limited user control due to the use of pre-trained segmentation masks. To utilize masks with a desired layout, an extensive training dataset is required, which is challenging to gather. We present FFaceNeRF, a NeRF-based face editing technique that can overcome the challenge of limited user control due to the use of fixed mask layouts. Our method employs a geometry adapter with feature injection, allowing for effective manipulation of geometry attributes. Additionally, we adopt latent mixing for tri-plane augmentation, which enables training with a few samples. This facilitates rapid model adaptation to desired mask layouts, crucial for applications in fields like personalized medical imaging or creative face editing. Our comparative evaluations demonstrate that FFaceNeRF surpasses existing mask based face editing methods in terms of flexibility, control, and generated image quality, paving the way for future advancements in customized and high-fidelity 3D face editing. The code is available on the {\href{this https URL}{project-page}}. 

**Abstract (ZH)**: Recent 3D人脸编辑方法利用掩模并通过神经辐射场（NeRF）产生了高质量的编辑图像。尽管这些方法表现令人印象深刻，但现有方法往往因使用预训练分割掩模而提供了有限的用户控制。为了利用具有期望布局的掩模，需要一个庞大的训练数据集，这具有挑战性。我们提出了FFaceNeRF，一种基于NeRF的人脸编辑技术，可以克服由于固定掩模布局导致的有限用户控制问题。该方法采用带特征注入的几何适配器，能够有效操纵几何属性。此外，我们采用潜在掺混进行三平面增强，这使得使用少量样本进行训练成为可能。这促进了模型对期望掩模布局的快速适应，对于个性化医疗成像或创意人脸编辑等领域至关重要。我们的比较评估表明，FFaceNeRF 在灵活性、控制能力和生成图像质量方面超越了现有的基于掩模的人脸编辑方法，为定制和高保真3D人脸编辑的未来进展铺平了道路。源代码可在项目页面（this https URL）获取。 

---
# Exploring the Efficacy of Partial Denoising Using Bit Plane Slicing for Enhanced Fracture Identification: A Comparative Study of Deep Learning-Based Approaches and Handcrafted Feature Extraction Techniques 

**Title (ZH)**: 基于位平面分割的局部去噪方法增强骨折识别效果探索：深度学习方法与手工特征提取技术的比较研究 

**Authors**: Snigdha Paul, Sambit Mallick, Anindya Sen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17030)  

**Abstract**: Computer vision has transformed medical diagnosis, treatment, and research through advanced image processing and machine learning techniques. Fracture classification, a critical area in healthcare, has greatly benefited from these advancements, yet accurate detection is challenged by complex patterns and image noise. Bit plane slicing enhances medical images by reducing noise interference and extracting informative features. This research explores partial denoising techniques to provide practical solutions for improved fracture analysis, ultimately enhancing patient care. The study explores deep learning model DenseNet and handcrafted feature extraction. Decision Tree and Random Forest, were employed to train and evaluate distinct image representations. These include the original image, the concatenation of the four bit planes from the LSB as well as MSB, the fully denoised image, and an image consisting of 6 bit planes from MSB and 2 denoised bit planes from LSB. The purpose of forming these diverse image representations is to analyze SNR as well as classification accuracy and identify the bit planes that contain the most informative features. Moreover, the study delves into the significance of partial denoising techniques in preserving crucial features, leading to improvements in classification results. Notably, this study shows that employing the Random Forest classifier, the partially denoised image representation exhibited a testing accuracy of 95.61% surpassing the performance of other image representations. The outcomes of this research provide valuable insights into the development of efficient preprocessing, feature extraction and classification approaches for fracture identification. By enhancing diagnostic accuracy, these advancements hold the potential to positively impact patient care and overall medical outcomes. 

**Abstract (ZH)**: 计算机视觉通过对先进图像处理和机器学习技术的应用，已革新了医疗诊断、治疗和研究。骨折分类这一医疗领域关键环节极大地受益于这些进步，但准确检测仍受复杂模式和图像噪声的挑战。位平面切片通过减少噪声干扰并提取有用特征，增强了医疗图像。本研究探讨部分去噪技术，以提供提高骨折分析的实用解决方案，最终提升患者护理质量。研究探讨了深度学习模型DenseNet和手工特征提取，并使用决策树和随机森林对不同的图像表示进行训练和评估。这些表示包括原始图像、最下位平面(LSB)和最上位平面(MSB)的四位位平面拼接、完全去噪图像，以及由MSB的6位平面和LSB的2个去噪位平面组成的图像。形成这些多样化的图像表示旨在分析信噪比和分类准确性，并确定包含最有用特征的位平面。此外，研究深入探讨了部分去噪技术在保持重要特征方面的意义，从而改善分类结果。值得注意的是，本研究显示，使用随机森林分类器的部分去噪图像表示在测试中的准确率为95.61%，超过了其他图像表示的性能。本研究的结果为骨折识别的高效预处理、特征提取和分类方法的发展提供了宝贵的见解。通过提升诊断准确性，这些进步有望对患者护理和整体医疗结果产生积极影响。 

---
# Enabling Versatile Controls for Video Diffusion Models 

**Title (ZH)**: 支持视频扩散模型的多功能控制 

**Authors**: Xu Zhang, Hao Zhou, Haoming Qin, Xiaobin Lu, Jiaxing Yan, Guanzhong Wang, Zeyu Chen, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16983)  

**Abstract**: Despite substantial progress in text-to-video generation, achieving precise and flexible control over fine-grained spatiotemporal attributes remains a significant unresolved challenge in video generation research. To address these limitations, we introduce VCtrl (also termed PP-VCtrl), a novel framework designed to enable fine-grained control over pre-trained video diffusion models in a unified manner. VCtrl integrates diverse user-specified control signals-such as Canny edges, segmentation masks, and human keypoints-into pretrained video diffusion models via a generalizable conditional module capable of uniformly encoding multiple types of auxiliary signals without modifying the underlying generator. Additionally, we design a unified control signal encoding pipeline and a sparse residual connection mechanism to efficiently incorporate control representations. Comprehensive experiments and human evaluations demonstrate that VCtrl effectively enhances controllability and generation quality. The source code and pre-trained models are publicly available and implemented using the PaddlePaddle framework at this http URL. 

**Abstract (ZH)**: 尽管在文本到视频生成方面取得了显著进展，但在视频生成研究中实现精确灵活的细粒度时空属性控制仍然是一个重要的未解决问题。为了解决这些局限性，我们引入了VCtrl（也称为PP-VCtrl）这一新型框架，旨在以统一的方式对预训练的视频扩散模型进行细粒度控制。VCtrl通过一个通用的条件模块将用户指定的控制信号（如Canny边缘、分割掩码和人体关键点）整合到预训练的视频扩散模型中，该模块能够均匀编码多种类型的辅助信号而无需修改底层生成器。此外，我们设计了一种统一的控制信号编码管道和一种稀疏残差连接机制，以高效地整合控制表示。全面的实验和人工评估表明，VCtrl显著提高了可控性和生成质量。源代码和预训练模型已公开，并使用PaddlePaddle框架在以下网址实现：this http URL。 

---
# ARFlow: Human Action-Reaction Flow Matching with Physical Guidance 

**Title (ZH)**: ARFlow: 基于物理指导的人类动作-反应流匹配 

**Authors**: Wentao Jiang, Jingya Wang, Haotao Lu, Kaiyang Ji, Baoxiong Jia, Siyuan Huang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.16973)  

**Abstract**: Human action-reaction synthesis, a fundamental challenge in modeling causal human interactions, plays a critical role in applications ranging from virtual reality to social robotics. While diffusion-based models have demonstrated promising performance, they exhibit two key limitations for interaction synthesis: reliance on complex noise-to-reaction generators with intricate conditional mechanisms, and frequent physical violations in generated motions. To address these issues, we propose Action-Reaction Flow Matching (ARFlow), a novel framework that establishes direct action-to-reaction mappings, eliminating the need for complex conditional mechanisms. Our approach introduces two key innovations: an x1-prediction method that directly outputs human motions instead of velocity fields, enabling explicit constraint enforcement; and a training-free, gradient-based physical guidance mechanism that effectively prevents body penetration artifacts during sampling. Extensive experiments on NTU120 and Chi3D datasets demonstrate that ARFlow not only outperforms existing methods in terms of Fréchet Inception Distance and motion diversity but also significantly reduces body collisions, as measured by our new Intersection Volume and Intersection Frequency metrics. 

**Abstract (ZH)**: 基于人类动作-反作用合成的人类因果交互建模是一项基本挑战，对于从虚拟现实到社会机器人应用等领域起着关键作用。尽管基于扩散的方法已经显示出有希望的性能，但它们在交互合成方面存在两个关键局限性：对复杂噪声到反作用生成器的依赖性及生成动作中的频繁物理违反现象。为了解决这些问题，我们提出了一种新的框架，称为动作-反作用流动匹配（ARFlow），该框架直接建立了动作到反作用的映射，消除了复杂条件机制的需要。我们的方法引入了两个关键创新：一种x1预测方法，直接输出人类动作而不是速度场，从而允许显式约束的施加；以及一种无需训练的梯度基于物理引导机制，在采样过程中有效防止身体穿插现象。在NTU120和Chi3D数据集上的广泛实验表明，ARFlow不仅在弗雷歇入学距离和动作多样性方面优于现有方法，还通过我们提出的新交集体积和交集频率度量显著减少了身体碰撞。 

---
# From Faces to Voices: Learning Hierarchical Representations for High-quality Video-to-Speech 

**Title (ZH)**: 从面部到声音：学习多层次表示以实现高质量视频到语音转换 

**Authors**: Ji-Hoon Kim, Jeongsoo Choi, Jaehun Kim, Chaeyoung Jung, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2503.16956)  

**Abstract**: The objective of this study is to generate high-quality speech from silent talking face videos, a task also known as video-to-speech synthesis. A significant challenge in video-to-speech synthesis lies in the substantial modality gap between silent video and multi-faceted speech. In this paper, we propose a novel video-to-speech system that effectively bridges this modality gap, significantly enhancing the quality of synthesized speech. This is achieved by learning of hierarchical representations from video to speech. Specifically, we gradually transform silent video into acoustic feature spaces through three sequential stages -- content, timbre, and prosody modeling. In each stage, we align visual factors -- lip movements, face identity, and facial expressions -- with corresponding acoustic counterparts to ensure the seamless transformation. Additionally, to generate realistic and coherent speech from the visual representations, we employ a flow matching model that estimates direct trajectories from a simple prior distribution to the target speech distribution. Extensive experiments demonstrate that our method achieves exceptional generation quality comparable to real utterances, outperforming existing methods by a significant margin. 

**Abstract (ZH)**: 本研究的目标是从静音说话人脸视频中生成高质量语音，这一任务也被称为视频到语音合成。视频到语音合成中的一个重大挑战在于静音视频与多维语音之间巨大的模态差距。本文提出了一种新颖的视频到语音系统，有效地弥合了这一模态差距，大幅提升了合成语音的质量。这一目标通过对视频到语音进行层次化表示学习实现。具体而言，我们通过三个连续阶段逐渐将静音视频转换为声学特征空间——内容建模、音色建模和语调建模。在每个阶段中，我们通过将视觉因素——唇部运动、面部身份和面部表情——与相应的声学对应物对齐，确保无缝转换。此外，为了从视觉表示生成现实且连贯的语音，我们采用了流匹配模型，该模型从简单的先验分布直接估计到目标语音分布的轨迹。广泛实验表明，本方法的生成质量出众，可与真实语音媲美，显著优于现有方法。 

---
# Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification 

**Title (ZH)**: 基于分类器引导的CLIP知识蒸馏用于无监督多标签分类 

**Authors**: Dongseob Kim, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16873)  

**Abstract**: Multi-label classification is crucial for comprehensive image understanding, yet acquiring accurate annotations is challenging and costly. To address this, a recent study suggests exploiting unsupervised multi-label classification leveraging CLIP, a powerful vision-language model. Despite CLIP's proficiency, it suffers from view-dependent predictions and inherent bias, limiting its effectiveness. We propose a novel method that addresses these issues by leveraging multiple views near target objects, guided by Class Activation Mapping (CAM) of the classifier, and debiasing pseudo-labels derived from CLIP predictions. Our Classifier-guided CLIP Distillation (CCD) enables selecting multiple local views without extra labels and debiasing predictions to enhance classification performance. Experimental results validate our method's superiority over existing techniques across diverse datasets. The code is available at this https URL. 

**Abstract (ZH)**: 利用CLIP进行去偏见的多标签分类：基于分类器引导的CLIP蒸馏（Classifier-guided CLIP Distillation） 

---
# Auto-Regressive Diffusion for Generating 3D Human-Object Interactions 

**Title (ZH)**: 自回归扩散生成3D人体物交互 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Saeed Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.16801)  

**Abstract**: Text-driven Human-Object Interaction (Text-to-HOI) generation is an emerging field with applications in animation, video games, virtual reality, and robotics. A key challenge in HOI generation is maintaining interaction consistency in long sequences. Existing Text-to-Motion-based approaches, such as discrete motion tokenization, cannot be directly applied to HOI generation due to limited data in this domain and the complexity of the modality. To address the problem of interaction consistency in long sequences, we propose an autoregressive diffusion model (ARDHOI) that predicts the next continuous token. Specifically, we introduce a Contrastive Variational Autoencoder (cVAE) to learn a physically plausible space of continuous HOI tokens, thereby ensuring that generated human-object motions are realistic and natural. For generating sequences autoregressively, we develop a Mamba-based context encoder to capture and maintain consistent sequential actions. Additionally, we implement an MLP-based denoiser to generate the subsequent token conditioned on the encoded context. Our model has been evaluated on the OMOMO and BEHAVE datasets, where it outperforms existing state-of-the-art methods in terms of both performance and inference speed. This makes ARDHOI a robust and efficient solution for text-driven HOI tasks 

**Abstract (ZH)**: 基于文本的人机物交互（Text-to-HOI）生成是动画、电子游戏、虚拟现实和机器人领域的新兴领域。长序列中交互一致性保持是HOI生成的关键挑战。现有的基于文本到运动的方法，如离散运动token化，由于该领域数据有限且模态复杂性高，无法直接应用于HOI生成。为了解决长序列中交互一致性的问题，我们提出了一种自回归扩散模型（ARDHOI），用于预测下一个连续token。具体地，我们引入了一种对比变分自编码器（cVAE）以学习物理上合理的连续HOI token空间，从而保证生成的人机物运动具有现实感和自然性。为了自回归地生成序列，我们开发了一种基于Mamba的上下文编码器，以捕获和保持一致的序列动作。此外，我们实现了一种基于MLP的去噪器，以在编码的上下文条件下生成后续token。我们在OMOMO和BEHAVE数据集上对模型进行了评估，结果显示在性能和推理速度上均优于现有最先进的方法。这使得ARDHOI成为文本驱动HOI任务的 robust 和高效解决方案。 

---
# Learning Part Knowledge to Facilitate Category Understanding for Fine-Grained Generalized Category Discovery 

**Title (ZH)**: 学习部分知识以促进细粒度泛化类别的理解与发现 

**Authors**: Enguang Wang, Zhimao Peng, Zhengyuan Xie, Haori Lu, Fei Yang, Xialei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16782)  

**Abstract**: Generalized Category Discovery (GCD) aims to classify unlabeled data containing both seen and novel categories. Although existing methods perform well on generic datasets, they struggle in fine-grained scenarios. We attribute this difficulty to their reliance on contrastive learning over global image features to automatically capture discriminative cues, which fails to capture the subtle local differences essential for distinguishing fine-grained categories. Therefore, in this paper, we propose incorporating part knowledge to address fine-grained GCD, which introduces two key challenges: the absence of annotations for novel classes complicates the extraction of the part features, and global contrastive learning prioritizes holistic feature invariance, inadvertently suppressing discriminative local part patterns. To address these challenges, we propose PartGCD, including 1) Adaptive Part Decomposition, which automatically extracts class-specific semantic parts via Gaussian Mixture Models, and 2) Part Discrepancy Regularization, enforcing explicit separation between part features to amplify fine-grained local part distinctions.
Experiments demonstrate state-of-the-art performance across multiple fine-grained benchmarks while maintaining competitiveness on generic datasets, validating the effectiveness and robustness of our approach. 

**Abstract (ZH)**: 广义类别发现（GCD）旨在对包含已见类别和新颖类别的未标记数据进行分类。尽管现有方法在通用数据集中表现良好，但在细粒度场景中却面临挑战。我们将这一困难归因于它们依赖于全局图像特征的对比学习自动捕捉区分性线索的方法，这种方法未能捕捉到区分细粒度类别所必需的微妙局部差异。因此，在本文中，我们提出将部分知识融入细粒度GCD中，这引入了两个关键挑战：新颖类别的标注缺失使得部分特征的提取变得复杂，而全局对比学习 prioritizes 整体特征不变性，无意中抑制了区分性局部部分模式。为了解决这些挑战，我们提出了PartGCD，包括1）自适应部分分解，通过高斯混合模型自动提取类特定的语义部分，以及2）部分差异正则化，强制部分特征之间的明确分离以放大细粒度局部部分的区别。实验结果在多个细粒度基准上展示了最先进的性能，同时在通用数据集上保持竞争力，验证了我们方法的有效性和鲁棒性。 

---
# Dynamic Attention Mechanism in Spatiotemporal Memory Networks for Object Tracking 

**Title (ZH)**: 时空记忆网络中动态注意力机制的研究 

**Authors**: Meng Zhou, Jiadong Xie, Mingsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16768)  

**Abstract**: Mainstream visual object tracking frameworks predominantly rely on template matching paradigms. Their performance heavily depends on the quality of template features, which becomes increasingly challenging to maintain in complex scenarios involving target deformation, occlusion, and background clutter. While existing spatiotemporal memory-based trackers emphasize memory capacity expansion, they lack effective mechanisms for dynamic feature selection and adaptive fusion. To address this gap, we propose a Dynamic Attention Mechanism in Spatiotemporal Memory Network (DASTM) with two key innovations: 1) A differentiable dynamic attention mechanism that adaptively adjusts channel-spatial attention weights by analyzing spatiotemporal correlations between the templates and memory features; 2) A lightweight gating network that autonomously allocates computational resources based on target motion states, prioritizing high-discriminability features in challenging scenarios. Extensive evaluations on OTB-2015, VOT 2018, LaSOT, and GOT-10K benchmarks demonstrate our DASTM's superiority, achieving state-of-the-art performance in success rate, robustness, and real-time efficiency, thereby offering a novel solution for real-time tracking in complex environments. 

**Abstract (ZH)**: 主流的视觉对象跟踪框架主要依赖模板匹配 paradigm。它们的性能高度依赖于模板特征的质量，在涉及目标变形、遮挡和背景杂乱的复杂场景中，这一依赖关系变得越来越具挑战性。虽然现有的基于时空记忆的跟踪器侧重于扩大记忆容量，但缺乏有效的动态特征选择和自适应融合机制。为解决这一问题，我们提出了一种时空记忆网络中的动态注意力机制 (DASTM)，其中包括两项关键创新：1) 一个可微分的动态注意力机制，通过分析模板与记忆特征之间的时空相关性，自适应调整通道-空间注意力权重；2) 一个轻量级门控网络，能够根据目标运动状态自主分配计算资源，在挑战性场景中优先选择高可区分特征。在 OTB-2015、VOT 2018、LaSOT 和 GOT-10K 基准上的广泛评估表明，我们的 DASTM 超越了现有方法，在成功率、鲁棒性和实时效率方面达到最佳性能，从而为复杂环境下的实时跟踪提供了一种新的解决方案。 

---
# QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge 

**Title (ZH)**: QuartDepth：边缘实时深度估计的后训练量化 

**Authors**: Xuan Shen, Weize Ma, Jing Liu, Changdi Yang, Rui Ding, Quanyi Wang, Henghui Ding, Wei Niu, Yanzhi Wang, Pu Zhao, Jun Lin, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16709)  

**Abstract**: Monocular Depth Estimation (MDE) has emerged as a pivotal task in computer vision, supporting numerous real-world applications. However, deploying accurate depth estimation models on resource-limited edge devices, especially Application-Specific Integrated Circuits (ASICs), is challenging due to the high computational and memory demands. Recent advancements in foundational depth estimation deliver impressive results but further amplify the difficulty of deployment on ASICs. To address this, we propose QuartDepth which adopts post-training quantization to quantize MDE models with hardware accelerations for ASICs. Our approach involves quantizing both weights and activations to 4-bit precision, reducing the model size and computation cost. To mitigate the performance degradation, we introduce activation polishing and compensation algorithm applied before and after activation quantization, as well as a weight reconstruction method for minimizing errors in weight quantization. Furthermore, we design a flexible and programmable hardware accelerator by supporting kernel fusion and customized instruction programmability, enhancing throughput and efficiency. Experimental results demonstrate that our framework achieves competitive accuracy while enabling fast inference and higher energy efficiency on ASICs, bridging the gap between high-performance depth estimation and practical edge-device applicability. Code: this https URL 

**Abstract (ZH)**: 单目深度估计(MDE)已成为计算机视觉中的一个重要任务，支持众多实际应用。然而，在资源受限的边缘设备，尤其是应用特定集成电路(ASICs)上部署准确的深度估计模型颇具挑战性，因为这需要高计算能力和内存需求。近期基础深度估计的发展虽然取得了显著成果，但进一步加剧了其在ASICs上的部署难度。为解决这一问题，我们提出了QuartDepth，采用后训练量化技术结合硬件加速器为ASICs量化MDE模型。我们的方法将权重和激活量化为4位精度，减小模型大小和计算成本。为了缓解性能下降，我们引入了在激活量化前后应用的激活优化和补偿算法，以及一种权重重建方法以最小化权重量化误差。此外，我们设计了一种灵活且可编程的硬件加速器，支持内核融合和自定义指令编程，提高吞吐量和效率。实验结果表明，我们的框架在ASICs上实现了 competitive准确度，支持快速推理和更高能效，填补了高性能深度估计与实际边缘设备应用之间的差距。代码：this https URL。 

---
# MobilePlantViT: A Mobile-friendly Hybrid ViT for Generalized Plant Disease Image Classification 

**Title (ZH)**: MobilePlantViT：一种适用于通用植物病害图像分类的移动友好型混合ViT 

**Authors**: Moshiur Rahman Tonmoy, Md. Mithun Hossain, Nilanjan Dey, M. F. Mridha  

**Link**: [PDF](https://arxiv.org/pdf/2503.16628)  

**Abstract**: Plant diseases significantly threaten global food security by reducing crop yields and undermining agricultural sustainability. AI-driven automated classification has emerged as a promising solution, with deep learning models demonstrating impressive performance in plant disease identification. However, deploying these models on mobile and edge devices remains challenging due to high computational demands and resource constraints, highlighting the need for lightweight, accurate solutions for accessible smart agriculture systems. To address this, we propose MobilePlantViT, a novel hybrid Vision Transformer (ViT) architecture designed for generalized plant disease classification, which optimizes resource efficiency while maintaining high performance. Extensive experiments across diverse plant disease datasets of varying scales show our model's effectiveness and strong generalizability, achieving test accuracies ranging from 80% to over 99%. Notably, with only 0.69 million parameters, our architecture outperforms the smallest versions of MobileViTv1 and MobileViTv2, despite their higher parameter counts. These results underscore the potential of our approach for real-world, AI-powered automated plant disease classification in sustainable and resource-efficient smart agriculture systems. All codes will be available in the GitHub repository: this https URL 

**Abstract (ZH)**: 植物疾病严重威胁全球粮食安全，通过降低作物产量和削弱农业可持续性。基于AI的自动化分类方法 emerge 作为一项有前景的解决方案，深度学习模型在植物病害识别方面展示了出色的性能。然而，将这些模型部署在移动和边缘设备上仍然面临挑战，由于计算需求高和资源限制，突显了轻量级、准确的解决方案对于可访问的智能农业系统的需求。为了解决这一问题，我们提出了MobilePlantViT，这是一种新型混合Vision Transformer (ViT) 架构，旨在实现泛化的植物病害分类，该架构优化了资源效率，同时保持高性能。跨不同规模的多种植物病害数据集的大量实验显示了我们模型的有效性和强大的泛化能力，其测试准确率从80%到超过99%不等。值得注意的是，尽管MobileViTv1和MobileViTv2的参数量更高，我们的架构仅包含0.69百万个参数，但仍表现出色。这些结果强调了我们方法在可持续和资源高效的智能农业系统中实现AI驱动的自动化植物病害分类的潜力。所有代码将在GitHub仓库中提供：this https URL 

---
# A Recipe for Generating 3D Worlds From a Single Image 

**Title (ZH)**: 从单张图片生成3D世界的 recipe 

**Authors**: Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2503.16611)  

**Abstract**: We introduce a recipe for generating immersive 3D worlds from a single image by framing the task as an in-context learning problem for 2D inpainting models. This approach requires minimal training and uses existing generative models. Our process involves two steps: generating coherent panoramas using a pre-trained diffusion model and lifting these into 3D with a metric depth estimator. We then fill unobserved regions by conditioning the inpainting model on rendered point clouds, requiring minimal fine-tuning. Tested on both synthetic and real images, our method produces high-quality 3D environments suitable for VR display. By explicitly modeling the 3D structure of the generated environment from the start, our approach consistently outperforms state-of-the-art, video synthesis-based methods along multiple quantitative image quality metrics. Project Page: this https URL 

**Abstract (ZH)**: 我们提出了一种生成单张图像的沉浸式3D世界的食谱，将任务重新定义为针对2D修复模型的上下文学习问题。该方法需要少量训练，并利用现有的生成模型。我们的过程分为两步：使用预训练的扩散模型生成连贯的全景图，并用度量深度估计器将其提升到3D。然后，我们通过将修复模型条件化于渲染的点云来填充未观察到的区域，只需少量微调。在合成和真实图像上测试后，我们的方法生成适用于VR显示的高度高质量的3D环境。从一开始就明确建模生成环境的3D结构，我们的方法在多个定量图像质量指标上始终优于基于视频合成的方法。项目页面：这个链接。 

---
# Reliable Radiologic Skeletal Muscle Area Assessment -- A Biomarker for Cancer Cachexia Diagnosis 

**Title (ZH)**: 可靠的放射学骨骼肌面积评估——癌症恶病质诊断的生物标志物 

**Authors**: Sabeen Ahmed, Nathan Parker, Margaret Park, Daniel Jeong, Lauren Peres, Evan W. Davis, Jennifer B. Permuth, Erin Siegel, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2503.16556)  

**Abstract**: Cancer cachexia is a common metabolic disorder characterized by severe muscle atrophy which is associated with poor prognosis and quality of life. Monitoring skeletal muscle area (SMA) longitudinally through computed tomography (CT) scans, an imaging modality routinely acquired in cancer care, is an effective way to identify and track this condition. However, existing tools often lack full automation and exhibit inconsistent accuracy, limiting their potential for integration into clinical workflows. To address these challenges, we developed SMAART-AI (Skeletal Muscle Assessment-Automated and Reliable Tool-based on AI), an end-to-end automated pipeline powered by deep learning models (nnU-Net 2D) trained on mid-third lumbar level CT images with 5-fold cross-validation, ensuring generalizability and robustness. SMAART-AI incorporates an uncertainty-based mechanism to flag high-error SMA predictions for expert review, enhancing reliability. We combined the SMA, skeletal muscle index, BMI, and clinical data to train a multi-layer perceptron (MLP) model designed to predict cachexia at the time of cancer diagnosis. Tested on the gastroesophageal cancer dataset, SMAART-AI achieved a Dice score of 97.80% +/- 0.93%, with SMA estimated across all four datasets in this study at a median absolute error of 2.48% compared to manual annotations with SliceOmatic. Uncertainty metrics-variance, entropy, and coefficient of variation-strongly correlated with SMA prediction errors (0.83, 0.76, and 0.73 respectively). The MLP model predicts cachexia with 79% precision, providing clinicians with a reliable tool for early diagnosis and intervention. By combining automation, accuracy, and uncertainty awareness, SMAART-AI bridges the gap between research and clinical application, offering a transformative approach to managing cancer cachexia. 

**Abstract (ZH)**: 癌症恶病质是一种常见的代谢障碍，以严重的肌肉萎缩为特征，与不良的预后和生活质量相关。通过计算机断层扫描（CT）扫描纵向监测骨骼肌面积（SMA）是识别和追踪这种状况的有效方法。然而，现有工具往往缺乏完全自动化并且表现出不一致的准确性，限制了其在临床工作流程中的集成。为解决这些挑战，我们开发了SMAART-AI（基于AI的骨骼肌评估自动化可靠工具），该工具基于深度学习模型（nnU-Net 2D），经过5折交叉验证训练，以确保泛化能力和鲁棒性。SMAART-AI Incorporates一种基于不确定性的机制，对高误差SMA预测进行专家审查，提高可靠性。我们结合SMA、骨骼肌指数、BMI和临床数据，训练了一个多层感知机（MLP）模型，用于预测癌症诊断时的恶病质。在胃食管癌数据集上测试，SMAART-AI实现了97.80%±0.93%的Dice分数，与使用SliceOmatic的手动注释相比，在所有四个数据集中的SMA估算中位绝对误差为2.48%。不确定性度量——方差、熵和变异系数——与SMA预测误差强烈相关（分别为0.83、0.76和0.73）。MLP模型以79%的精度预测恶病质，为临床早期诊断和干预提供了一个可靠工具。通过结合自动化、准确性和不确定性意识，SMAART-AI弥合了科研与临床应用之间的差距，提供了一种管理癌症恶病质的变革性方法。 

---
# A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Applications, and Emerging Research Directions 

**Title (ZH)**: 深度CNN架构进展综述：挑战、应用及新兴研究方向 

**Authors**: Saddam Hussain Khan, Rashid Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2503.16546)  

**Abstract**: Deep Convolutional Neural Networks (CNNs) have significantly advanced deep learning, driving breakthroughs in computer vision, natural language processing, medical diagnosis, object detection, and speech recognition. Architectural innovations including 1D, 2D, and 3D convolutional models, dilated and grouped convolutions, depthwise separable convolutions, and attention mechanisms address domain-specific challenges and enhance feature representation and computational efficiency. Structural refinements such as spatial-channel exploitation, multi-path design, and feature-map enhancement contribute to robust hierarchical feature extraction and improved generalization, particularly through transfer learning. Efficient preprocessing strategies, including Fourier transforms, structured transforms, low-precision computation, and weight compression, optimize inference speed and facilitate deployment in resource-constrained environments. This survey presents a unified taxonomy that classifies CNN architectures based on spatial exploitation, multi-path structures, depth, width, dimensionality expansion, channel boosting, and attention mechanisms. It systematically reviews CNN applications in face recognition, pose estimation, action recognition, text classification, statistical language modeling, disease diagnosis, radiological analysis, cryptocurrency sentiment prediction, 1D data processing, video analysis, and speech recognition. In addition to consolidating architectural advancements, the review highlights emerging learning paradigms such as few-shot, zero-shot, weakly supervised, federated learning frameworks and future research directions include hybrid CNN-transformer models, vision-language integration, generative learning, etc. This review provides a comprehensive perspective on CNN's evolution from 2015 to 2025, outlining key innovations, challenges, and opportunities. 

**Abstract (ZH)**: 深度卷积神经网络（CNNs）极大地推动了深度学习的发展，促进了计算机视觉、自然语言处理、医学诊断、对象检测和语音识别等领域的突破。包括1D、2D和3D卷积模型、膨胀卷积、分组卷积、深度可分离卷积和注意力机制在内的架构创新解决了领域特定的挑战，提升了特征表示和计算效率。结构精炼如空域-通道利用、多路径设计和特征图增强促进了鲁棒的层级特征提取和泛化能力，特别是在迁移学习方面。高效的预处理策略，包括傅里叶变换、结构化变换、低精度计算和权重压缩，优化了推理速度并促进了资源受限环境下的部署。本文综述提出了一个统一的分类体系，根据空域利用、多路径结构、深度、宽度、维度扩展、通道增强和注意力机制对CNN架构进行分类。系统回顾了CNN在面部识别、姿态估计、动作识别、文本分类、统计语言建模、疾病诊断、影像分析、加密货币情绪预测、1D数据处理、视频分析和语音识别等方面的应用。此外，综述还强调了 emerging learning paradigms 如少样本学习、零样本学习、弱监督学习及联邦学习框架，并指出了未来的研究方向，包括混合CNN-Transformer模型、视觉-语言集成、生成学习等。本文对2015年至2025年间CNN的发展进行了综合概述，概述了关键创新、挑战和机遇。 

---
# From Voices to Worlds: Developing an AI-Powered Framework for 3D Object Generation in Augmented Reality 

**Title (ZH)**: 从声音到世界：开发一种基于人工智能的增强现实三维对象生成框架 

**Authors**: Majid Behravan, Denis Gracanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16474)  

**Abstract**: This paper presents Matrix, an advanced AI-powered framework designed for real-time 3D object generation in Augmented Reality (AR) environments. By integrating a cutting-edge text-to-3D generative AI model, multilingual speech-to-text translation, and large language models (LLMs), the system enables seamless user interactions through spoken commands. The framework processes speech inputs, generates 3D objects, and provides object recommendations based on contextual understanding, enhancing AR experiences. A key feature of this framework is its ability to optimize 3D models by reducing mesh complexity, resulting in significantly smaller file sizes and faster processing on resource-constrained AR devices. Our approach addresses the challenges of high GPU usage, large model output sizes, and real-time system responsiveness, ensuring a smoother user experience. Moreover, the system is equipped with a pre-generated object repository, further reducing GPU load and improving efficiency. We demonstrate the practical applications of this framework in various fields such as education, design, and accessibility, and discuss future enhancements including image-to-3D conversion, environmental object detection, and multimodal support. The open-source nature of the framework promotes ongoing innovation and its utility across diverse industries. 

**Abstract (ZH)**: 这篇论文介绍了一种名为Matrix的先进AI驱动框架，用于 augmented reality (AR) 环境中的实时3D物体生成。通过整合前沿的文字到3D生成AI模型、多语言语音到文本翻译以及大语言模型（LLMs），该系统能够通过语音命令实现无缝用户交互。该框架处理语音输入，生成3D物体，并基于上下文理解提供物体推荐，从而增强AR体验。该框架的一个关键功能是通过减少网格复杂性优化3D模型，从而在资源受限的AR设备上显著减小文件大小并加快处理速度。我们的方法解决了一系列挑战，包括高GPU使用率、大的模型输出尺寸以及实时系统的响应性，从而确保更流畅的用户体验。此外，系统配备有预生成的物体库，进一步减轻GPU负载并提高效率。我们展示了该框架在教育、设计和无障碍等多种领域的实际应用，并探讨了未来增强的功能，包括图像到3D的转换、环境物体检测以及多模态支持。该框架的开源性质促进了持续创新并使其在各个行业中具有广泛应用价值。 

---
# Rank-O-ToM: Unlocking Emotional Nuance Ranking to Enhance Affective Theory-of-Mind 

**Title (ZH)**: Rank-O-ToM: 解锁情感细微差别的排名以增强情感共情理论 

**Authors**: JiHyun Kim, JuneHyoung Kwon, MiHyeon Kim, Eunju Lee, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.16461)  

**Abstract**: Facial Expression Recognition (FER) plays a foundational role in enabling AI systems to interpret emotional nuances, a critical aspect of affective Theory of Mind (ToM). However, existing models often struggle with poor calibration and a limited capacity to capture emotional intensity and complexity. To address this, we propose Ranking the Emotional Nuance for Theory of Mind (Rank-O-ToM), a framework that leverages ordinal ranking to align confidence levels with the emotional spectrum. By incorporating synthetic samples reflecting diverse affective complexities, Rank-O-ToM enhances the nuanced understanding of emotions, advancing AI's ability to reason about affective states. 

**Abstract (ZH)**: 情绪精细排序用于理论心智（Rank-O-ToM）：一种利用序数排序提升情绪理解的框架 

---
# CLIP-PING: Boosting Lightweight Vision-Language Models with Proximus Intrinsic Neighbors Guidance 

**Title (ZH)**: CLIP-PING：使用Proximus内在邻域指导增强轻量级视觉语言模型 

**Authors**: Chu Myaet Thwal, Ye Lin Tun, Minh N. H. Nguyen, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.03871)  

**Abstract**: Beyond the success of Contrastive Language-Image Pre-training (CLIP), recent trends mark a shift toward exploring the applicability of lightweight vision-language models for resource-constrained scenarios. These models often deliver suboptimal performance when relying solely on a single image-text contrastive learning objective, spotlighting the need for more effective training mechanisms that guarantee robust cross-modal feature alignment. In this work, we propose CLIP-PING: Contrastive Language-Image Pre-training with Proximus Intrinsic Neighbors Guidance, a novel yet simple and efficient training paradigm designed to boost the performance of lightweight vision-language models with minimal computational overhead and lower data demands. CLIP-PING bootstraps unimodal features extracted from arbitrary pre-trained encoders to obtain intrinsic guidance of proximus neighbor samples, i.e., nearest-neighbor (NN) and cross nearest-neighbor (XNN). We find that extra contrastive supervision from these neighbors substantially boosts cross-modal alignment, enabling lightweight models to learn more generic features with rich semantic diversity. Extensive experiments reveal that CLIP-PING notably surpasses its peers in zero-shot generalization and cross-modal retrieval tasks. Specifically, a 5.5% gain on zero-shot ImageNet1K classification with 10.7% (I2T) and 5.7% (T2I) on Flickr30K retrieval, compared to the original CLIP when using ViT-XS image encoder trained on 3 million (image, text) pairs. Moreover, CLIP-PING showcases a strong transferability under the linear evaluation protocol across several downstream tasks. 

**Abstract (ZH)**: 超越CLIP的成功：面向资源受限场景的轻量级跨模态模型训练新趋势——CLIP-PING：基于近邻样本内在指导的对比语言-图像预训练 

---
