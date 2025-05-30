# Improving Trajectory Stitching with Flow Models 

**Title (ZH)**: 基于流模型改进轨迹拼接 

**Authors**: Reece O'Mahoney, Wanming Yu, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2505.07802)  

**Abstract**: Generative models have shown great promise as trajectory planners, given their affinity to modeling complex distributions and guidable inference process. Previous works have successfully applied these in the context of robotic manipulation but perform poorly when the required solution does not exist as a complete trajectory within the training set. We identify that this is a result of being unable to plan via stitching, and subsequently address the architectural and dataset choices needed to remedy this. On top of this, we propose a novel addition to the training and inference procedures to both stabilize and enhance these capabilities. We demonstrate the efficacy of our approach by generating plans with out of distribution boundary conditions and performing obstacle avoidance on the Franka Panda in simulation and on real hardware. In both of these tasks our method performs significantly better than the baselines and is able to avoid obstacles up to four times as large. 

**Abstract (ZH)**: 生成模型在轨迹规划中的应用展示了巨大潜力，这得益于其对复杂分布建模的亲和力和可指导的推断过程。以往的工作在机器人的操作场景中取得了成功，但在要求的解在训练集中不存在完整轨迹时表现不佳。我们发现这是由于无法通过拼接来规划轨迹，因此我们针对这一问题改进了架构和数据集的选择。在此基础上，我们提出了一种新的训练和推理程序中的方法，以稳定并增强这些能力。通过在仿真和实际硬件上生成超出分布边界的计划并进行避障实验，展示了该方法的有效性。在这些任务中，我们的方法显著优于基线方法，并且能避开大小达到四倍的障碍物。 

---
# Privacy Risks of Robot Vision: A User Study on Image Modalities and Resolution 

**Title (ZH)**: 机器人视觉的隐私风险：一项关于图像模态和分辨率的用户研究 

**Authors**: Xuying Huang, Sicong Pan, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07766)  

**Abstract**: User privacy is a crucial concern in robotic applications, especially when mobile service robots are deployed in personal or sensitive environments. However, many robotic downstream tasks require the use of cameras, which may raise privacy risks. To better understand user perceptions of privacy in relation to visual data, we conducted a user study investigating how different image modalities and image resolutions affect users' privacy concerns. The results show that depth images are broadly viewed as privacy-safe, and a similarly high proportion of respondents feel the same about semantic segmentation images. Additionally, the majority of participants consider 32*32 resolution RGB images to be almost sufficiently privacy-preserving, while most believe that 16*16 resolution can fully guarantee privacy protection. 

**Abstract (ZH)**: 移动服务机器人在个人或敏感环境中部署时，用户隐私是机器人应用中的一个重要关切。然而，许多下游任务需要使用摄像头，这可能会引发隐私风险。为了更好地了解视觉数据与用户隐私感知之间的关系，我们开展了一项用户研究，调查不同的图像模态和分辨率如何影响用户的隐私担忧。研究结果表明，_DEPTH IMAGES ARE WIDELY CONSIDERED AS PRIVACY-SAFE_，并且有相似比例的受访者对语义分割图像也持同样看法。此外，大多数参与者认为32*32分辨率的RGB图像几乎足以保护隐私，而大多数人认为16*16分辨率可以完全确保隐私保护。 

---
# Beyond Static Perception: Integrating Temporal Context into VLMs for Cloth Folding 

**Title (ZH)**: 超越静态感知：将时间上下文集成到VLMs中的衣物折叠研究 

**Authors**: Oriol Barbany, Adrià Colomé, Carme Torras  

**Link**: [PDF](https://arxiv.org/pdf/2505.07600)  

**Abstract**: Manipulating clothes is challenging due to their complex dynamics, high deformability, and frequent self-occlusions. Garments exhibit a nearly infinite number of configurations, making explicit state representations difficult to define. In this paper, we analyze BiFold, a model that predicts language-conditioned pick-and-place actions from visual observations, while implicitly encoding garment state through end-to-end learning. To address scenarios such as crumpled garments or recovery from failed manipulations, BiFold leverages temporal context to improve state estimation. We examine the internal representations of the model and present evidence that its fine-tuning and temporal context enable effective alignment between text and image regions, as well as temporal consistency. 

**Abstract (ZH)**: 操纵衣物具有挑战性，因为衣物动态复杂、高度可变形且频繁自遮挡。衣物展现出几乎无限的配置方式，使得显式状态表示难以定义。本文分析了BiFold模型，该模型可以从视觉观察中预测语言条件下的拿取和放置动作，并通过端到端学习隐式编码衣物状态。为了应对褶皱衣物或失败操纵后的恢复等场景，BiFold利用时间上下文提高状态估计。我们研究了模型的内部表示，并展示了其微调和时间上下文能够有效实现文本区域和图像区域的对齐以及时间一致性。 

---
# VALISENS: A Validated Innovative Multi-Sensor System for Cooperative Automated Driving 

**Title (ZH)**: VALISENS：一个验证的创新多传感器系统用于协同自动驾驶 

**Authors**: Lei Wan, Prabesh Gupta, Andreas Eich, Marcel Kettelgerdes, Hannan Ejaz Keen, Michael Klöppel-Gersdorf, Alexey Vinel  

**Link**: [PDF](https://arxiv.org/pdf/2505.06980)  

**Abstract**: Perception is a core capability of automated vehicles and has been significantly advanced through modern sensor technologies and artificial intelligence. However, perception systems still face challenges in complex real-world scenarios. To improve robustness against various external factors, multi-sensor fusion techniques are essential, combining the strengths of different sensor modalities. With recent developments in Vehicle-to-Everything (V2X communication, sensor fusion can now extend beyond a single vehicle to a cooperative multi-agent system involving Connected Automated Vehicle (CAV) and intelligent infrastructure. This paper presents VALISENS, an innovative multi-sensor system distributed across multiple agents. It integrates onboard and roadside LiDARs, radars, thermal cameras, and RGB cameras to enhance situational awareness and support cooperative automated driving. The thermal camera adds critical redundancy for perceiving Vulnerable Road User (VRU), while fusion with roadside sensors mitigates visual occlusions and extends the perception range beyond the limits of individual vehicles. We introduce the corresponding perception module built on this sensor system, which includes object detection, tracking, motion forecasting, and high-level data fusion. The proposed system demonstrates the potential of cooperative perception in real-world test environments and lays the groundwork for future Cooperative Intelligent Transport Systems (C-ITS) applications. 

**Abstract (ZH)**: 多代理分布式多传感器系统VALISENS：面向智能互联车辆的协同感知 

---
# M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark 

**Title (ZH)**: M3CAD: 向通用协作自动驾驶基准迈进 

**Authors**: Morui Zhu, Yongqi Zhu, Yihao Zhu, Qi Chen, Deyuan Qu, Song Fu, Qing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06746)  

**Abstract**: We introduce M$^3$CAD, a novel benchmark designed to advance research in generic cooperative autonomous driving. M$^3$CAD comprises 204 sequences with 30k frames, spanning a diverse range of cooperative driving scenarios. Each sequence includes multiple vehicles and sensing modalities, e.g., LiDAR point clouds, RGB images, and GPS/IMU, supporting a variety of autonomous driving tasks, including object detection and tracking, mapping, motion forecasting, occupancy prediction, and path planning. This rich multimodal setup enables M$^3$CAD to support both single-vehicle and multi-vehicle autonomous driving research, significantly broadening the scope of research in the field. To our knowledge, M$^3$CAD is the most comprehensive benchmark specifically tailored for cooperative multi-task autonomous driving research. We evaluate the state-of-the-art end-to-end solution on M$^3$CAD to establish baseline performance. To foster cooperative autonomous driving research, we also propose E2EC, a simple yet effective framework for cooperative driving solution that leverages inter-vehicle shared information for improved path planning. We release M$^3$CAD, along with our baseline models and evaluation results, to support the development of robust cooperative autonomous driving systems. All resources will be made publicly available on this https URL 

**Abstract (ZH)**: M$^3$CAD：一种促进通用协同自动驾驶研究的新基准 

---
# Boosting Cross-spectral Unsupervised Domain Adaptation for Thermal Semantic Segmentation 

**Title (ZH)**: 跨谱域无监督领域适应的增强式热语义分割 

**Authors**: Seokjun Kwon, Jeongmin Shin, Namil Kim, Soonmin Hwang, Yukyung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06951)  

**Abstract**: In autonomous driving, thermal image semantic segmentation has emerged as a critical research area, owing to its ability to provide robust scene understanding under adverse visual conditions. In particular, unsupervised domain adaptation (UDA) for thermal image segmentation can be an efficient solution to address the lack of labeled thermal datasets. Nevertheless, since these methods do not effectively utilize the complementary information between RGB and thermal images, they significantly decrease performance during domain adaptation. In this paper, we present a comprehensive study on cross-spectral UDA for thermal image semantic segmentation. We first propose a novel masked mutual learning strategy that promotes complementary information exchange by selectively transferring results between each spectral model while masking out uncertain regions. Additionally, we introduce a novel prototypical self-supervised loss designed to enhance the performance of the thermal segmentation model in nighttime scenarios. This approach addresses the limitations of RGB pre-trained networks, which cannot effectively transfer knowledge under low illumination due to the inherent constraints of RGB sensors. In experiments, our method achieves higher performance over previous UDA methods and comparable performance to state-of-the-art supervised methods. 

**Abstract (ZH)**: 自主驾驶中跨光谱领域适应的热图像语义分割研究 

---
# Edge-Enabled VIO with Long-Tracked Features for High-Accuracy Low-Altitude IoT Navigation 

**Title (ZH)**: 基于边缘计算的长跟踪特征高精度低altitude物联网导航视觉惯性导航 

**Authors**: Xiaohong Huang, Cui Yang, Miaowen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06517)  

**Abstract**: This paper presents a visual-inertial odometry (VIO) method using long-tracked features. Long-tracked features can constrain more visual frames, reducing localization drift. However, they may also lead to accumulated matching errors and drift in feature tracking. Current VIO methods adjust observation weights based on re-projection errors, yet this approach has flaws. Re-projection errors depend on estimated camera poses and map points, so increased errors might come from estimation inaccuracies, not actual feature tracking errors. This can mislead the optimization process and make long-tracked features ineffective for suppressing localization drift. Furthermore, long-tracked features constrain a larger number of frames, which poses a significant challenge to real-time performance of the system. To tackle these issues, we propose an active decoupling mechanism for accumulated errors in long-tracked feature utilization. We introduce a visual reference frame reset strategy to eliminate accumulated tracking errors and a depth prediction strategy to leverage the long-term constraint. To ensure real time preformane, we implement three strategies for efficient system state estimation: a parallel elimination strategy based on predefined elimination order, an inverse-depth elimination simplification strategy, and an elimination skipping strategy. Experiments on various datasets show that our method offers higher positioning accuracy with relatively short consumption time, making it more suitable for edge-enabled low-altitude IoT navigation, where high-accuracy positioning and real-time operation on edge device are required. The code will be published at github. 

**Abstract (ZH)**: 基于长踪迹特征的视觉惯性里程计方法 

---
# Hybrid Spiking Vision Transformer for Object Detection with Event Cameras 

**Title (ZH)**: 基于事件相机的混合脉冲视觉变换器目标检测方法 

**Authors**: Qi Xu, Jie Deng, Jiangrong Shen, Biwu Chen, Huajin Tang, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07715)  

**Abstract**: Event-based object detection has gained increasing attention due to its advantages such as high temporal resolution, wide dynamic range, and asynchronous address-event representation. Leveraging these advantages, Spiking Neural Networks (SNNs) have emerged as a promising approach, offering low energy consumption and rich spatiotemporal dynamics. To further enhance the performance of event-based object detection, this study proposes a novel hybrid spike vision Transformer (HsVT) model. The HsVT model integrates a spatial feature extraction module to capture local and global features, and a temporal feature extraction module to model time dependencies and long-term patterns in event sequences. This combination enables HsVT to capture spatiotemporal features, improving its capability to handle complex event-based object detection tasks. To support research in this area, we developed and publicly released The Fall Detection Dataset as a benchmark for event-based object detection tasks. This dataset, captured using an event-based camera, ensures facial privacy protection and reduces memory usage due to the event representation format. We evaluated the HsVT model on GEN1 and Fall Detection datasets across various model sizes. Experimental results demonstrate that HsVT achieves significant performance improvements in event detection with fewer parameters. 

**Abstract (ZH)**: 基于事件的对象检测因其高时间分辨率、宽动态范围和异步地址事件表示等优势引起了越来越多的关注。利用这些优势，契神经网络（SNNs） emerged as a promising approach，提供低能耗和丰富的时空动态。为了进一步提高基于事件的对象检测性能，本研究提出了一种新的混合契视觉变换器（HsVT）模型。HsVT模型结合了空间特征提取模块以捕获局部和全局特征，以及时间特征提取模块以建模事件序列中的时间依赖性和长期模式。这种结合使HsVT能够捕获时空特征，从而提高其处理复杂基于事件的对象检测任务的能力。为了支持该领域的研究，我们开发并公开发布了跌倒检测数据集作为基于事件的对象检测任务的基准。该数据集由基于事件的相机捕获，确保面部隐私保护并因事件表示格式减少内存使用。我们在不同模型大小的GEN1和跌倒检测数据集上评估了HsVT模型。实验结果表明，HsVT以较少的参数实现了显著的性能提升。 

---
# Evaluating Modern Visual Anomaly Detection Approaches in Semiconductor Manufacturing: A Comparative Study 

**Title (ZH)**: 现代视觉异常检测方法在半导体制造中的评估：一项比较研究 

**Authors**: Manuel Barusco, Francesco Borsatti, Youssef Ben Khalifa, Davide Dalle Pezze, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07576)  

**Abstract**: Semiconductor manufacturing is a complex, multistage process. Automated visual inspection of Scanning Electron Microscope (SEM) images is indispensable for minimizing equipment downtime and containing costs. Most previous research considers supervised approaches, assuming a sufficient number of anomalously labeled samples. On the contrary, Visual Anomaly Detection (VAD), an emerging research domain, focuses on unsupervised learning, avoiding the costly defect collection phase while providing explanations of the predictions. We introduce a benchmark for VAD in the semiconductor domain by leveraging the MIIC dataset. Our results demonstrate the efficacy of modern VAD approaches in this field. 

**Abstract (ZH)**: 半导体制造是一个复杂的多阶段过程。扫描电子显微镜（SEM）图像的自动化视觉检测对于减少设备停机时间和控制成本至关重要。大多数先前的研究考虑了监督方法，假设有足够的异常标记样本。相反，视觉异常检测（VAD）这一新兴研究领域集中于无监督学习，避免了昂贵的缺陷收集阶段，同时提供预测解释。我们通过利用MIIC数据集，引入了半导体领域的VAD基准。我们的结果证明了现代VAD方法在这一领域的有效性。 

---
# Robust Kidney Abnormality Segmentation: A Validation Study of an AI-Based Framework 

**Title (ZH)**: 基于AI的框架的肾异常分割鲁棒性验证研究 

**Authors**: Sarah de Boer, Hartmut Häntze, Kiran Vaidhya Venkadesh, Myrthe A. D. Buser, Gabriel E. Humpire Mamani, Lina Xu, Lisa C. Adams, Jawed Nawabi, Keno K. Bressem, Bram van Ginneken, Mathias Prokop, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2505.07573)  

**Abstract**: Kidney abnormality segmentation has important potential to enhance the clinical workflow, especially in settings requiring quantitative assessments. Kidney volume could serve as an important biomarker for renal diseases, with changes in volume correlating directly with kidney function. Currently, clinical practice often relies on subjective visual assessment for evaluating kidney size and abnormalities, including tumors and cysts, which are typically staged based on diameter, volume, and anatomical location. To support a more objective and reproducible approach, this research aims to develop a robust, thoroughly validated kidney abnormality segmentation algorithm, made publicly available for clinical and research use. We employ publicly available training datasets and leverage the state-of-the-art medical image segmentation framework nnU-Net. Validation is conducted using both proprietary and public test datasets, with segmentation performance quantified by Dice coefficient and the 95th percentile Hausdorff distance. Furthermore, we analyze robustness across subgroups based on patient sex, age, CT contrast phases, and tumor histologic subtypes. Our findings demonstrate that our segmentation algorithm, trained exclusively on publicly available data, generalizes effectively to external test sets and outperforms existing state-of-the-art models across all tested datasets. Subgroup analyses reveal consistent high performance, indicating strong robustness and reliability. The developed algorithm and associated code are publicly accessible at this https URL. 

**Abstract (ZH)**: 肾脏异常分割具有增强临床工作流程的重要潜力，特别是在需要定量评估的环境中。肾脏体积可以作为肾疾病的重要生物标志物，体积的变化与肾功能成直接相关。目前，临床实践中常常依赖主观视觉评估来评价肾脏大小和异常情况，包括肿瘤和囊肿，通常根据直径、体积和解剖位置进行分期。为了支持更加客观和可重复的方法，本研究旨在开发一个稳健且完全验证的肾脏异常分割算法，并向临床和研究公众开放。我们使用公开可用的训练数据集，并利用最先进的医疗图像分割框架nnU-Net进行分割。验证使用了私人和公开的测试数据集，并通过Dice系数和95百分位Hausdorff距离量化分割性能。此外，我们基于患者性别、年龄、CT对比期以及肿瘤组织学亚型分析了分割算法的稳健性。研究结果表明，该分割算法仅在公开数据上训练后，能够有效泛化到外部测试集，并在所有测试数据集中优于现有的最先进的模型。子组分析显示出一致的高性能，表明其具有很强的稳健性和可靠性。所开发的算法及其相关代码在以下网址公开：这个 https URL。 

---
# Automated Visual Attention Detection using Mobile Eye Tracking in Behavioral Classroom Studies 

**Title (ZH)**: 基于移动眼动追踪的行为课堂研究中自动视觉注意检测 

**Authors**: Efe Bozkir, Christian Kosel, Tina Seidel, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07552)  

**Abstract**: Teachers' visual attention and its distribution across the students in classrooms can constitute important implications for student engagement, achievement, and professional teacher training. Despite that, inferring the information about where and which student teachers focus on is not trivial. Mobile eye tracking can provide vital help to solve this issue; however, the use of mobile eye tracking alone requires a significant amount of manual annotations. To address this limitation, we present an automated processing pipeline concept that requires minimal manually annotated data to recognize which student the teachers focus on. To this end, we utilize state-of-the-art face detection models and face recognition feature embeddings to train face recognition models with transfer learning in the classroom context and combine these models with the teachers' gaze from mobile eye trackers. We evaluated our approach with data collected from four different classrooms, and our results show that while it is possible to estimate the visually focused students with reasonable performance in all of our classroom setups, U-shaped and small classrooms led to the best results with accuracies of approximately 0.7 and 0.9, respectively. While we did not evaluate our method for teacher-student interactions and focused on the validity of the technical approach, as our methodology does not require a vast amount of manually annotated data and offers a non-intrusive way of handling teachers' visual attention, it could help improve instructional strategies, enhance classroom management, and provide feedback for professional teacher development. 

**Abstract (ZH)**: 教室中教师的视觉注意及其在学生之间的分布对学生活动参与、学业成就及专业教师培训具有重要意义。然而，推断教师关注的学生位置和对象并不容易。移动眼动追踪可以提供重要帮助，但单独使用移动眼动追踪需要大量手动注释。为解决这一局限，我们提出了一种自动化处理管道概念，以最小的手动标注数据来识别教师关注的学生。为此，我们利用最新的面部检测模型和面部识别特征嵌入，在教室背景下进行迁移学习训练面部识别模型，并将这些模型与移动眼动追踪的教师凝视相结合。我们使用来自四间不同教室的数据评估了我们的方法，并结果显示，在所有教室设置中，均可以以合理性能估计视觉关注的学生，U形和小型教室的结果分别为约0.7和0.9。尽管我们未评估教师与学生之间的互动，并专注于技术方法的有效性，但鉴于我们的方法不需要大量手动标注数据且能够非侵入性地处理教师的视觉注意，它可以帮助改善教学策略、增强课堂管理，并为专业教师发展提供反馈。 

---
# MAIS: Memory-Attention for Interactive Segmentation 

**Title (ZH)**: MAIS: 记忆注意力机制用于交互式分割 

**Authors**: Mauricio Orbes-Arteaga, Oeslle Lucena, Sabastien Ourselin, M. Jorge Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.07511)  

**Abstract**: Interactive medical segmentation reduces annotation effort by refining predictions through user feedback. Vision Transformer (ViT)-based models, such as the Segment Anything Model (SAM), achieve state-of-the-art performance using user clicks and prior masks as prompts. However, existing methods treat interactions as independent events, leading to redundant corrections and limited refinement gains. We address this by introducing MAIS, a Memory-Attention mechanism for Interactive Segmentation that stores past user inputs and segmentation states, enabling temporal context integration. Our approach enhances ViT-based segmentation across diverse imaging modalities, achieving more efficient and accurate refinements. 

**Abstract (ZH)**: 交互式医学分割通过用户反馈 refinement 预测从而减少标注努力。通过引入基于记忆-注意机制的 MAIS，利用过往用户输入和分割状态实现时间上下文集成，我们提升了 Vision Transformer (ViT) 基模型在多种成像模态下的分割性能，实现了更高效和准确的 refinement。 

---
# Few-shot Semantic Encoding and Decoding for Video Surveillance 

**Title (ZH)**: Few-shot语义编码与解码在视频監控中的应用 

**Authors**: Baoping Cheng, Yukun Zhang, Liming Wang, Xiaoyan Xie, Tao Fu, Dongkun Wang, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07381)  

**Abstract**: With the continuous increase in the number and resolution of video surveillance cameras, the burden of transmitting and storing surveillance video is growing. Traditional communication methods based on Shannon's theory are facing optimization bottlenecks. Semantic communication, as an emerging communication method, is expected to break through this bottleneck and reduce the storage and transmission consumption of video. Existing semantic decoding methods often require many samples to train the neural network for each scene, which is time-consuming and labor-intensive. In this study, a semantic encoding and decoding method for surveillance video is proposed. First, the sketch was extracted as semantic information, and a sketch compression method was proposed to reduce the bit rate of semantic information. Then, an image translation network was proposed to translate the sketch into a video frame with a reference frame. Finally, a few-shot sketch decoding network was proposed to reconstruct video from sketch. Experimental results showed that the proposed method achieved significantly better video reconstruction performance than baseline methods. The sketch compression method could effectively reduce the storage and transmission consumption of semantic information with little compromise on video quality. The proposed method provides a novel semantic encoding and decoding method that only needs a few training samples for each surveillance scene, thus improving the practicality of the semantic communication system. 

**Abstract (ZH)**: 基于视频监控的语义编码与解码方法 

---
# GAN-based synthetic FDG PET images from T1 brain MRI can serve to improve performance of deep unsupervised anomaly detection models 

**Title (ZH)**: 基于GAN的合成FDG PET图像可以从T1脑MRI中获得，并可改善深度无监督异常检测模型的性能 

**Authors**: Daria Zotova, Nicolas Pinon, Robin Trombetta, Romain Bouet, Julien Jung, Carole Lartizien  

**Link**: [PDF](https://arxiv.org/pdf/2505.07364)  

**Abstract**: Background and Objective. Research in the cross-modal medical image translation domain has been very productive over the past few years in tackling the scarce availability of large curated multimodality datasets with the promising performance of GAN-based architectures. However, only a few of these studies assessed task-based related performance of these synthetic data, especially for the training of deep models. Method. We design and compare different GAN-based frameworks for generating synthetic brain [18F]fluorodeoxyglucose (FDG) PET images from T1 weighted MRI data. We first perform standard qualitative and quantitative visual quality evaluation. Then, we explore further impact of using these fake PET data in the training of a deep unsupervised anomaly detection (UAD) model designed to detect subtle epilepsy lesions in T1 MRI and FDG PET images. We introduce novel diagnostic task-oriented quality metrics of the synthetic FDG PET data tailored to our unsupervised detection task, then use these fake data to train a use case UAD model combining a deep representation learning based on siamese autoencoders with a OC-SVM density support estimation model. This model is trained on normal subjects only and allows the detection of any variation from the pattern of the normal population. We compare the detection performance of models trained on 35 paired real MR T1 of normal subjects paired either on 35 true PET images or on 35 synthetic PET images generated from the best performing generative models. Performance analysis is conducted on 17 exams of epilepsy patients undergoing surgery. Results. The best performing GAN-based models allow generating realistic fake PET images of control subject with SSIM and PSNR values around 0.9 and 23.8, respectively and in distribution (ID) with regard to the true control dataset. The best UAD model trained on these synthetic normative PET data allows reaching 74% sensitivity. Conclusion. Our results confirm that GAN-based models are the best suited for MR T1 to FDG PET translation, outperforming transformer or diffusion models. We also demonstrate the diagnostic value of these synthetic data for the training of UAD models and evaluation on clinical exams of epilepsy patients. Our code and the normative image dataset are available. 

**Abstract (ZH)**: 背景与目的. 近几年，跨模态医学图像转换领域的研究在应对稀缺的大型多模态数据集方面取得了丰硕成果，并且基于生成对抗网络（GAN）的架构表现出令人promise的效果。然而，其中只有少数研究评估了这些合成数据的任务相关性能，尤其是用于训练深度模型。方法. 我们设计并比较了不同的基于GAN的框架，用于从T1加权MRI数据生成合成的[18F]氟脱氧葡萄糖（FDG）PET图像。我们首先进行标准的定性和定量视觉质量评估，然后进一步探索使用这些假PET数据训练用于检测T1 MRI和FDG PET图像中细微癫痫病灶的半监督异常检测（UAD）模型的影响。我们引入了针对我们无监督检测任务量身定制的新诊断任务导向的质量评估指标，然后使用这些假数据训练一个结合基于Siamese自动编码器的深度表征学习模型和OC-SVM密度支持估计模型的使用案例UAD模型。该模型仅使用正常受试者的数据进行训练，并能够检测任何与正常人群模式的偏离。我们将用35对真实T1 MRI图像正常受试者数据训练的模型与用35对由表现最佳的生成模型生成的合成PET图像训练的模型进行检测性能比较。性能分析是在17例癫痫患者手术过程中进行的。结果. 表现最佳的GAN模型能够生成与对照受试者真实PET图像在结构相似性（SSIM）值约为0.9和峰值信噪比（PSNR）值约为23.8方面高度真实的假PET图像，并且在分布上与真实对照数据集一致。使用这些合成的正常PET数据训练的最佳UAD模型可实现74%的灵敏度。结论. 我们的成果表明，基于GAN的模型最适合用于从T1 MRI到FDG PET的转换，优于变压器或扩散模型。我们还证明了这些合成数据在UAD模型训练和在癫痫患者临床检查评估中的诊断价值。我们的代码和正常图像数据集已公开。 

---
# Generative Pre-trained Autoregressive Diffusion Transformer 

**Title (ZH)**: 预训练自回归扩散变换器生成模型 

**Authors**: Yuan Zhang, Jiacheng Jiang, Guoqing Ma, Zhiying Lu, Haoyang Huang, Jianlong Yuan, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07344)  

**Abstract**: In this work, we present GPDiT, a Generative Pre-trained Autoregressive Diffusion Transformer that unifies the strengths of diffusion and autoregressive modeling for long-range video synthesis, within a continuous latent space. Instead of predicting discrete tokens, GPDiT autoregressively predicts future latent frames using a diffusion loss, enabling natural modeling of motion dynamics and semantic consistency across frames. This continuous autoregressive framework not only enhances generation quality but also endows the model with representation capabilities. Additionally, we introduce a lightweight causal attention variant and a parameter-free rotation-based time-conditioning mechanism, improving both the training and inference efficiency. Extensive experiments demonstrate that GPDiT achieves strong performance in video generation quality, video representation ability, and few-shot learning tasks, highlighting its potential as an effective framework for video modeling in continuous space. 

**Abstract (ZH)**: GPDiT：统一扩散与自回归建模长时序视频合成的生成预训练自回归扩散变换器 

---
# Towards Scalable IoT Deployment for Visual Anomaly Detection via Efficient Compression 

**Title (ZH)**: 面向视觉异常检测的高效压缩驱动可扩展物联网部署 

**Authors**: Arianna Stropeni, Francesco Borsatti, Manuel Barusco, Davide Dalle Pezze, Marco Fabris, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07119)  

**Abstract**: Visual Anomaly Detection (VAD) is a key task in industrial settings, where minimizing waste and operational costs is essential. Deploying deep learning models within Internet of Things (IoT) environments introduces specific challenges due to the limited computational power and bandwidth of edge devices. This study investigates how to perform VAD effectively under such constraints by leveraging compact and efficient processing strategies. We evaluate several data compression techniques, examining the trade-off between system latency and detection accuracy. Experiments on the MVTec AD benchmark demonstrate that significant compression can be achieved with minimal loss in anomaly detection performance compared to uncompressed data. 

**Abstract (ZH)**: 视觉异常检测（VAD）是工业环境中的一项关键任务，其中减少浪费和运营成本至关重要。在物联网（IoT）环境中部署深度学习模型由于边缘设备的计算能力和带宽有限而引入了特定的挑战。本研究探讨如何在这种约束条件下有效进行VAD，通过利用紧凑且高效的处理策略。我们评估了几种数据压缩技术，研究了系统延迟和检测准确性之间的权衡。在MVTec AD基准测试上的实验表明，与未压缩数据相比，可以实现显著的压缩且异常检测性能的下降可以忽略不计。 

---
# Efficient and Robust Multidimensional Attention in Remote Physiological Sensing through Target Signal Constrained Factorization 

**Title (ZH)**: 远程生理传感中基于目标信号约束因子分解的高效稳健多维注意力机制 

**Authors**: Jitesh Joshi, Youngjun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.07013)  

**Abstract**: Remote physiological sensing using camera-based technologies offers transformative potential for non-invasive vital sign monitoring across healthcare and human-computer interaction domains. Although deep learning approaches have advanced the extraction of physiological signals from video data, existing methods have not been sufficiently assessed for their robustness to domain shifts. These shifts in remote physiological sensing include variations in ambient conditions, camera specifications, head movements, facial poses, and physiological states which often impact real-world performance significantly. Cross-dataset evaluation provides an objective measure to assess generalization capabilities across these domain shifts. We introduce Target Signal Constrained Factorization module (TSFM), a novel multidimensional attention mechanism that explicitly incorporates physiological signal characteristics as factorization constraints, allowing more precise feature extraction. Building on this innovation, we present MMRPhys, an efficient dual-branch 3D-CNN architecture designed for simultaneous multitask estimation of photoplethysmography (rPPG) and respiratory (rRSP) signals from multimodal RGB and thermal video inputs. Through comprehensive cross-dataset evaluation on five benchmark datasets, we demonstrate that MMRPhys with TSFM significantly outperforms state-of-the-art methods in generalization across domain shifts for rPPG and rRSP estimation, while maintaining a minimal inference latency suitable for real-time applications. Our approach establishes new benchmarks for robust multitask and multimodal physiological sensing and offers a computationally efficient framework for practical deployment in unconstrained environments. The web browser-based application featuring on-device real-time inference of MMRPhys model is available at this https URL 

**Abstract (ZH)**: 基于摄像头的远程生理传感技术在医疗保健和人机交互领域提供了非侵入性生命体征监测的变革潜力。尽管深度学习方法在从视频数据中提取生理信号方面取得了进展，但现有方法在面对域转移时的稳健性评估尚不满意。远程生理传感中的这些域转移包括环境条件变化、摄像头规格差异、头部移动、面部姿态和生理状态的变化，这些因素往往在实际应用中显著影响性能。跨数据集评估提供了一种客观的手段来衡量在这些域转移下的泛化能力。我们引入了目标信号约束分解模块（TSFM），这是一种新型多维注意力机制，明确地将生理信号特征作为分解约束，从而实现更精确的特征提取。在此基础上，我们提出了MMRPhys，一种高效的双分支3D-CNN架构，旨在同时从多模态RGB和热视频输入中估计光体积描记图（rPPG）和呼吸（rRSP）信号。通过在五个基准数据集上的全面跨数据集评估，我们证明了带有TSFM的MMRPhys在rPPG和rRSP估计的域转移泛化能力上显著优于现有方法，同时保持了适合实时应用的最小推理延迟。我们的方法确立了鲁棒多任务和多模态生理传感的新基准，并提供了一种在不受约束环境中进行现实部署的高效计算框架。基于Web浏览器的应用程序可以在以下链接中实现MMRPhys模型的设备上实时推理功能：[链接]。 

---
# NeuGen: Amplifying the 'Neural' in Neural Radiance Fields for Domain Generalization 

**Title (ZH)**: NeuGen: 强化神经辐射场中的“神经”元素以实现领域泛化 

**Authors**: Ahmed Qazi, Abdul Basit, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06894)  

**Abstract**: Neural Radiance Fields (NeRF) have significantly advanced the field of novel view synthesis, yet their generalization across diverse scenes and conditions remains challenging. Addressing this, we propose the integration of a novel brain-inspired normalization technique Neural Generalization (NeuGen) into leading NeRF architectures which include MVSNeRF and GeoNeRF. NeuGen extracts the domain-invariant features, thereby enhancing the models' generalization capabilities. It can be seamlessly integrated into NeRF architectures and cultivates a comprehensive feature set that significantly improves accuracy and robustness in image rendering. Through this integration, NeuGen shows improved performance on benchmarks on diverse datasets across state-of-the-art NeRF architectures, enabling them to generalize better across varied scenes. Our comprehensive evaluations, both quantitative and qualitative, confirm that our approach not only surpasses existing models in generalizability but also markedly improves rendering quality. Our work exemplifies the potential of merging neuroscientific principles with deep learning frameworks, setting a new precedent for enhanced generalizability and efficiency in novel view synthesis. A demo of our study is available at this https URL. 

**Abstract (ZH)**: 基于神经辐射场的新型视图合成中，神经泛化（NeuGen）的脑启发规范化技术在多样化场景和条件下的泛化能力提升 

---
# Symbolic Rule Extraction from Attention-Guided Sparse Representations in Vision Transformers 

**Title (ZH)**: 基于注意力引导稀疏表示的视觉变换器中符号规则提取 

**Authors**: Parth Padalkar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06745)  

**Abstract**: Recent neuro-symbolic approaches have successfully extracted symbolic rule-sets from CNN-based models to enhance interpretability. However, applying similar techniques to Vision Transformers (ViTs) remains challenging due to their lack of modular concept detectors and reliance on global self-attention mechanisms. We propose a framework for symbolic rule extraction from ViTs by introducing a sparse concept layer inspired by Sparse Autoencoders (SAEs). This linear layer operates on attention-weighted patch representations and learns a disentangled, binarized representation in which individual neurons activate for high-level visual concepts. To encourage interpretability, we apply a combination of L1 sparsity, entropy minimization, and supervised contrastive loss. These binarized concept activations are used as input to the FOLD-SE-M algorithm, which generates a rule-set in the form of logic programs. Our method achieves a 5.14% better classification accuracy than the standard ViT while enabling symbolic reasoning. Crucially, the extracted rule-set is not merely post-hoc but acts as a logic-based decision layer that operates directly on the sparse concept representations. The resulting programs are concise and semantically meaningful. This work is the first to extract executable logic programs from ViTs using sparse symbolic representations. It bridges the gap between transformer-based vision models and symbolic logic programming, providing a step forward in interpretable and verifiable neuro-symbolic AI. 

**Abstract (ZH)**: Recent神经符号方法已成功从基于CNN的模型中提取符号规则集以增强可解释性。但由于ViTs缺乏模块化概念检测器且依赖全局自注意力机制，将其上类似的技术应用仍然具有挑战性。我们提出了一种从ViTs提取符号规则的新框架，通过引入灵感源于稀疏自动编码器（SAEs）的稀疏概念层。该线性层作用于注意力加权片段表示，并学习一个彼此独立的二值化表示，其中单个神经元为高级视觉概念激活。为促进可解释性，我们应用L1稀疏性、熵最小化和监督对比丢失的组合。这些二值化概念激活被用作FOLD-SE-M算法的输入，该算法生成逻辑程序形式的规则集。该方法在标准ViT的基础上提高了5.14%的分类精度同时支持符号推理。至关重要的是，提取的规则集不仅仅是事后解释性的，而是作为基于逻辑的决策层直接作用于稀疏概念表示上。生成的程序简洁且语义有意义。这项工作首次使用稀疏符号表示从ViTs中提取可执行的逻辑程序，填补了基于变压器的视觉模型与符号逻辑编程之间的空白，为可解释和可验证的神经符号AI迈出了重要一步。 

---
# Underwater object detection in sonar imagery with detection transformer and Zero-shot neural architecture search 

**Title (ZH)**: 基于检测变换器和零样本神经架构搜索的声纳图像 underwater目标检测 

**Authors**: XiaoTong Gu, Shengyu Tang, Yiming Cao, Changdong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06694)  

**Abstract**: Underwater object detection using sonar imagery has become a critical and rapidly evolving research domain within marine technology. However, sonar images are characterized by lower resolution and sparser features compared to optical images, which seriously degrades the performance of object this http URL address these challenges, we specifically propose a Detection Transformer (DETR) architecture optimized with a Neural Architecture Search (NAS) approach called NAS-DETR for object detection in sonar images. First, an improved Zero-shot Neural Architecture Search (NAS) method based on the maximum entropy principle is proposed to identify a real-time, high-representational-capacity CNN-Transformer backbone for sonar image detection. This method enables the efficient discovery of high-performance network architectures with low computational and time overhead. Subsequently, the backbone is combined with a Feature Pyramid Network (FPN) and a deformable attention-based Transformer decoder to construct a complete network architecture. This architecture integrates various advanced components and training schemes to enhance overall performance. Extensive experiments demonstrate that this architecture achieves state-of-the-art performance on two Representative datasets, while maintaining minimal overhead in real-time efficiency and computational complexity. Furthermore, correlation analysis between the key parameters and differential entropy-based fitness function is performed to enhance the interpretability of the proposed framework. To the best of our knowledge, this is the first work in the field of sonar object detection to integrate the DETR architecture with a NAS search mechanism. 

**Abstract (ZH)**: 基于声呐图像的目标检测：NAS-DETR在marine技术中的应用 

---
# ProFashion: Prototype-guided Fashion Video Generation with Multiple Reference Images 

**Title (ZH)**: 基于原型指导的多参考图像时尚视频生成 

**Authors**: Xianghao Kong, Qiaosong Qi, Yuanbin Wang, Anyi Rao, Biaolong Chen, Aixi Zhang, Si Liu, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06537)  

**Abstract**: Fashion video generation aims to synthesize temporally consistent videos from reference images of a designated character. Despite significant progress, existing diffusion-based methods only support a single reference image as input, severely limiting their capability to generate view-consistent fashion videos, especially when there are different patterns on the clothes from different perspectives. Moreover, the widely adopted motion module does not sufficiently model human body movement, leading to sub-optimal spatiotemporal consistency. To address these issues, we propose ProFashion, a fashion video generation framework leveraging multiple reference images to achieve improved view consistency and temporal coherency. To effectively leverage features from multiple reference images while maintaining a reasonable computational cost, we devise a Pose-aware Prototype Aggregator, which selects and aggregates global and fine-grained reference features according to pose information to form frame-wise prototypes, which serve as guidance in the denoising process. To further enhance motion consistency, we introduce a Flow-enhanced Prototype Instantiator, which exploits the human keypoint motion flow to guide an extra spatiotemporal attention process in the denoiser. To demonstrate the effectiveness of ProFashion, we extensively evaluate our method on the MRFashion-7K dataset we collected from the Internet. ProFashion also outperforms previous methods on the UBC Fashion dataset. 

**Abstract (ZH)**: 基于多参考图像的时尚视频生成：实现增强的视角一致性和时间连贯性 

---
# My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing 

**Title (ZH)**: 你在脸上的情绪：面部关键点检测在潜在空间编辑中保留情绪的应用 

**Authors**: Jingrui He, Andrew Stephen McGough  

**Link**: [PDF](https://arxiv.org/pdf/2505.06436)  

**Abstract**: Generative Adversarial Network approaches such as StyleGAN/2 provide two key benefits: the ability to generate photo-realistic face images and possessing a semantically structured latent space from which these images are created. Many approaches have emerged for editing images derived from vectors in the latent space of a pre-trained StyleGAN/2 models by identifying semantically meaningful directions (e.g., gender or age) in the latent space. By moving the vector in a specific direction, the ideal result would only change the target feature while preserving all the other features. Providing an ideal data augmentation approach for gesture research as it could be used to generate numerous image variations whilst keeping the facial expressions intact. However, entanglement issues, where changing one feature inevitably affects other features, impacts the ability to preserve facial expressions. To address this, we propose the use of an addition to the loss function of a Facial Keypoint Detection model to restrict changes to the facial expressions. Building on top of an existing model, adding the proposed Human Face Landmark Detection (HFLD) loss, provided by a pre-trained Facial Keypoint Detection model, to the original loss function. We quantitatively and qualitatively evaluate the existing and our extended model, showing the effectiveness of our approach in addressing the entanglement issue and maintaining the facial expression. Our approach achieves up to 49% reduction in the change of emotion in our experiments. Moreover, we show the benefit of our approach by comparing with state-of-the-art models. By increasing the ability to preserve the facial gesture and expression during facial transformation, we present a way to create human face images with fixed expression but different appearances, making it a reliable data augmentation approach for Facial Gesture and Expression research. 

**Abstract (ZH)**: 生成对抗网络方法，如StyleGAN/2，提供两项关键优势：生成照片级真实的人脸图像和拥有一个语义上结构化的潜在空间，从该空间中生成这些图像。通过在预训练StyleGAN/2模型的潜在空间中的向量中识别语义上有意义的方向（例如性别或年龄），已出现了许多图像编辑方法。通过在特定方向上移动向量，理想的结果是仅改变目标特征同时保留所有其他特征。这为手势研究提供了一种理想的數據增强方法，因为可以生成大量图像变体同时保持面部表情不变。然而，特征纠缠问题（改变一个特征不可避免地会影响其他特征）影响了保持面部表情的能力。为此，我们提出在面部关键点检测模型的损失函数中添加一项，以限制对面部表情的更改。基于现有模型，将预训练面部关键点检测模型提供的提议的人脸地标检测（HFLD）损失添加到原始损失函数中。我们从定量和定性两方面评估现有和扩展后的模型，展示了我们的方法在解决特征纠缠问题和保持面部表情方面的有效性。在我们的实验中，我们的方法实现了最多49%的情绪变化减少。此外，我们通过与最先进的模型进行对比，展示了我们方法的优势。通过增强在面部变换过程中保持面部手势和表情的能力，我们提出了一种方法，可以创建具有固定表情但不同外观的人脸图像，使之成为面部手势和表情研究的可靠数据增强方法。 

---
# Attonsecond Streaking Phase Retrieval Via Deep Learning Methods 

**Title (ZH)**: 亚飞秒级 streaking 相位恢复基于深度学习方法 

**Authors**: Yuzhou Zhu, Zheng Zhang, Ruyi Zhang, Liang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06275)  

**Abstract**: Attosecond streaking phase retrieval is essential for resolving electron dynamics on sub-femtosecond time scales yet traditional algorithms rely on iterative minimization and central momentum approximations that degrade accuracy for broadband pulses. In this work phase retrieval is reformulated as a supervised computer-vision problem and four neural architectures are systematically compared. A convolutional network demonstrates strong sensitivity to local streak edges but lacks global context; a vision transformer captures long-range delay-energy correlations at the expense of local inductive bias; a hybrid CNN-ViT model unites local feature extraction and full-graph attention; and a capsule network further enforces spatial pose agreement through dynamic routing. A theoretical analysis introduces local, global and positional sensitivity measures and derives surrogate error bounds that predict the strict ordering $CNN<ViT<Hybrid<Capsule$. Controlled experiments on synthetic streaking spectrograms confirm this hierarchy, with the capsule network achieving the highest retrieval fidelity. Looking forward, embedding the strong-field integral into physics-informed neural networks and exploring photonic hardware implementations promise pathways toward real-time attosecond pulse characterization under demanding experimental conditions. 

**Abstract (ZH)**: 阿斯皮秒级相位检索对于亚飞秒时间尺度的电子动力学解析至关重要，但传统算法依赖于迭代最小化和质心动量近似，这会降低宽带脉冲的准确性。本工作中，相位检索被重新表述为监督计算机视觉问题，并系统对比了四种神经网络架构。卷积网络对局部条形边缘表现出强烈的敏感性，但缺乏全局上下文；视觉变换器捕捉长程延迟-能量相关性，但以牺牲局部归纳偏见为代价；混合CNN-ViT模型结合了局部特征提取和全图注意力；胶囊网络进一步通过动态路由确保空间姿态的一致性。理论分析引入了局部、全局和位置敏感性度量，并推导出预测严格排序的替代误差界：$CNN<ViT<Hybrid<Capsule$。受控实验在合成条形光谱图上确认了该层级关系，胶囊网络实现了最高的检索保真度。展望未来，将强场积分嵌入物理感知神经网络并在光子硬件实现中的探索为在苛刻实验条件下实时阿斯皮秒脉冲表征指明了路径。 

---
