# Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition 

**Title (ZH)**: 端到端车对外协作自主驾驶竞赛中的研究挑战与进展 

**Authors**: Ruiyang Hao, Haibao Yu, Jiaru Zhong, Chuanye Wang, Jiahao Wang, Yiming Kan, Wenxian Yang, Siqi Fan, Huilin Yin, Jianing Qiu, Yao Mu, Jiankai Sun, Li Chen, Walter Zimmer, Dandan Zhang, Shanghang Zhang, Mac Schwager, Wei Huang, Xiaobo Zhang, Ping Luo, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2507.21610)  

**Abstract**: With the rapid advancement of autonomous driving technology, vehicle-to-everything (V2X) communication has emerged as a key enabler for extending perception range and enhancing driving safety by providing visibility beyond the line of sight. However, integrating multi-source sensor data from both ego-vehicles and infrastructure under real-world constraints, such as limited communication bandwidth and dynamic environments, presents significant technical challenges. To facilitate research in this area, we organized the End-to-End Autonomous Driving through V2X Cooperation Challenge, which features two tracks: cooperative temporal perception and cooperative end-to-end planning. Built on the UniV2X framework and the V2X-Seq-SPD dataset, the challenge attracted participation from over 30 teams worldwide and established a unified benchmark for evaluating cooperative driving systems. This paper describes the design and outcomes of the challenge, highlights key research problems including bandwidth-aware fusion, robust multi-agent planning, and heterogeneous sensor integration, and analyzes emerging technical trends among top-performing solutions. By addressing practical constraints in communication and data fusion, the challenge contributes to the development of scalable and reliable V2X-cooperative autonomous driving systems. 

**Abstract (ZH)**: 基于V2X合作的端到端自主驾驶挑战赛：设计与成果 

---
# Diffusion Denoiser-Aided Gyrocompassing 

**Title (ZH)**: 扩散去噪辅助速率陀螺仪定向 

**Authors**: Gershy Ben-Arie, Daniel Engelsman, Rotem Dror, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2507.21245)  

**Abstract**: An accurate initial heading angle is essential for efficient and safe navigation across diverse domains. Unlike magnetometers, gyroscopes can provide accurate heading reference independent of the magnetic disturbances in a process known as gyrocompassing. Yet, accurate and timely gyrocompassing, using low-cost gyroscopes, remains a significant challenge in scenarios where external navigation aids are unavailable. Such challenges are commonly addressed in real-world applications such as autonomous vehicles, where size, weight, and power limitations restrict sensor quality, and noisy measurements severely degrade gyrocompassing performance. To cope with this challenge, we propose a novel diffusion denoiser-aided gyrocompass approach. It integrates a diffusion-based denoising framework with an enhanced learning-based heading estimation model. The diffusion denoiser processes raw inertial sensor signals before input to the deep learning model, resulting in accurate gyrocompassing. Experiments using both simulated and real sensor data demonstrate that our proposed approach improves gyrocompassing accuracy by 26% compared to model-based gyrocompassing and by 15% compared to other learning-driven approaches. This advancement holds particular significance for ensuring accurate and robust navigation in autonomous platforms that incorporate low-cost gyroscopes within their navigation systems. 

**Abstract (ZH)**: 一种新型扩散去噪辅助的陀螺罗经方法对于跨不同领域的高效和安全导航至关重要 

---
# MapDiffusion: Generative Diffusion for Vectorized Online HD Map Construction and Uncertainty Estimation in Autonomous Driving 

**Title (ZH)**: MapDiffusion: 生成扩散在自主驾驶中基于向量的在线高精度地图构建及不确定性估计 

**Authors**: Thomas Monninger, Zihan Zhang, Zhipeng Mo, Md Zafar Anwar, Steffen Staab, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.21423)  

**Abstract**: Autonomous driving requires an understanding of the static environment from sensor data. Learned Bird's-Eye View (BEV) encoders are commonly used to fuse multiple inputs, and a vector decoder predicts a vectorized map representation from the latent BEV grid. However, traditional map construction models provide deterministic point estimates, failing to capture uncertainty and the inherent ambiguities of real-world environments, such as occlusions and missing lane markings. We propose MapDiffusion, a novel generative approach that leverages the diffusion paradigm to learn the full distribution of possible vectorized maps. Instead of predicting a single deterministic output from learned queries, MapDiffusion iteratively refines randomly initialized queries, conditioned on a BEV latent grid, to generate multiple plausible map samples. This allows aggregating samples to improve prediction accuracy and deriving uncertainty estimates that directly correlate with scene ambiguity. Extensive experiments on the nuScenes dataset demonstrate that MapDiffusion achieves state-of-the-art performance in online map construction, surpassing the baseline by 5% in single-sample performance. We further show that aggregating multiple samples consistently improves performance along the ROC curve, validating the benefit of distribution modeling. Additionally, our uncertainty estimates are significantly higher in occluded areas, reinforcing their value in identifying regions with ambiguous sensor input. By modeling the full map distribution, MapDiffusion enhances the robustness and reliability of online vectorized HD map construction, enabling uncertainty-aware decision-making for autonomous vehicles in complex environments. 

**Abstract (ZH)**: 自主驾驶需要从传感器数据中理解静态环境。Learned鸟瞰图（BEV）编码器常用于融合多个输入，并且向量解码器从潜在BEV网格中预测向量化的地图表示。然而，传统的地图构建模型仅提供确定性的点估计，无法捕捉现实环境中固有的不确定性和模糊性，例如遮挡和缺失的车道标记。我们提出MapDiffusion，这是一种新颖的生成性方法，利用扩散范式学习可能的向量化地图的完整分布。MapDiffusion通过迭代改进依据BEV潜在网格随机初始化的查询，而不是从学习的查询中预测单一的确定性输出，来生成多个可信的地图样本。这允许通过聚合样本来提高预测准确性，并直接从场景模糊性中推导出不确定性估计。在nuScenes数据集上的实验表明，MapDiffusion在在线地图构建方面的表现达到最新水平，在单样本性能上超过了基线5%。此外，我们展示了聚合多个样本沿着ROC曲线一致地提高性能，验证了分布建模的益处。此外，我们的不确定性估计在遮挡区域显著较高，强化了它们在识别模糊传感器输入区域的价值。通过建模完整地图分布，MapDiffusion增强了在线向量化高精度地图构建的稳健性和可靠性，从而在复杂环境中为自主车辆提供基于不确定性的决策支持。 

---
# Snap, Segment, Deploy: A Visual Data and Detection Pipeline for Wearable Industrial Assistants 

**Title (ZH)**: 快照、分割、部署：可穿戴工业助手的视觉数据和检测管道 

**Authors**: Di Wen, Junwei Zheng, Ruiping Liu, Yi Xu, Kunyu Peng, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21072)  

**Abstract**: Industrial assembly tasks increasingly demand rapid adaptation to complex procedures and varied components, yet are often conducted in environments with limited computing, connectivity, and strict privacy requirements. These constraints make conventional cloud-based or fully autonomous solutions impractical for factory deployment. This paper introduces a mobile-device-based assistant system for industrial training and operational support, enabling real-time, semi-hands-free interaction through on-device perception and voice interfaces. The system integrates lightweight object detection, speech recognition, and Retrieval-Augmented Generation (RAG) into a modular on-device pipeline that operates entirely on-device, enabling intuitive support for part handling and procedure understanding without relying on manual supervision or cloud services. To enable scalable training, we adopt an automated data construction pipeline and introduce a two-stage refinement strategy to improve visual robustness under domain shift. Experiments on our generated dataset, i.e., Gear8, demonstrate improved robustness to domain shift and common visual corruptions. A structured user study further confirms its practical viability, with positive user feedback on the clarity of the guidance and the quality of the interaction. These results indicate that our framework offers a deployable solution for real-time, privacy-preserving smart assistance in industrial environments. We will release the Gear8 dataset and source code upon acceptance. 

**Abstract (ZH)**: 基于移动设备的工业培训与操作支持助手系统：面向现实环境的实时半自助交互 

---
# SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation 

**Title (ZH)**: SafeDriveRAG：基于知识图谱检索增强生成 toward 安全自动驾驶 

**Authors**: Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.21585)  

**Abstract**: In this work, we study how vision-language models (VLMs) can be utilized to enhance the safety for the autonomous driving system, including perception, situational understanding, and path planning. However, existing research has largely overlooked the evaluation of these models in traffic safety-critical driving scenarios. To bridge this gap, we create the benchmark (SafeDrive228K) and propose a new baseline based on VLM with knowledge graph-based retrieval-augmented generation (SafeDriveRAG) for visual question answering (VQA). Specifically, we introduce SafeDrive228K, the first large-scale multimodal question-answering benchmark comprising 228K examples across 18 sub-tasks. This benchmark encompasses a diverse range of traffic safety queries, from traffic accidents and corner cases to common safety knowledge, enabling a thorough assessment of the comprehension and reasoning abilities of the models. Furthermore, we propose a plug-and-play multimodal knowledge graph-based retrieval-augmented generation approach that employs a novel multi-scale subgraph retrieval algorithm for efficient information retrieval. By incorporating traffic safety guidelines collected from the Internet, this framework further enhances the model's capacity to handle safety-critical situations. Finally, we conduct comprehensive evaluations on five mainstream VLMs to assess their reliability in safety-sensitive driving tasks. Experimental results demonstrate that integrating RAG significantly improves performance, achieving a +4.73% gain in Traffic Accidents tasks, +8.79% in Corner Cases tasks and +14.57% in Traffic Safety Commonsense across five mainstream VLMs, underscoring the potential of our proposed benchmark and methodology for advancing research in traffic safety. Our source code and data are available at this https URL. 

**Abstract (ZH)**: 在本工作中，我们研究了视觉语言模型（VLMs）如何被利用以增强自主驾驶系统的安全性，包括感知、情景理解以及路径规划。然而，现有的研究大多忽视了在交通安全关键驾驶场景中评估这些模型的表现。为了填补这一空白，我们创建了基准（SafeDrive228K），并基于知识图谱检索增强生成（SafeDriveRAG）提出了一个视觉问答（VQA）的新基线。具体地，我们引入了SafeDrive228K，这是第一个大规模多模态问答基准，包含228K个跨18个亚任务的样本。该基准涵盖了从交通事故和极端情况到普通安全知识的各种交通安全查询，使模型的理解和推理能力得到全面评估。此外，我们提出了一个插即用的多模态知识图谱检索增强生成方法，使用了一种新颖的多尺度子图检索算法，以实现高效的信息检索。通过集成从互联网收集的交通安全指南，该框架进一步增强了模型在处理安全关键情况时的能力。最后，我们在五种主流VLM上进行了全面评估，以评估其在安全敏感驾驶任务中的可靠性。实验结果表明，结合RAG显著提高了性能，在交通事故任务中提高了4.73%，在极端情况任务中提高了8.79%，在交通安全性常识中提高了14.57%，突显了我们提出基准和方法在交通安全性研究中的潜力。我们的源代码和数据可在以下链接获取。 

---
# ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports 

**Title (ZH)**: ReXGroundingCT：一种用于自由文本报告中发现分割的3D胸腔CT数据集 

**Authors**: Mohammed Baharoon, Luyang Luo, Michael Moritz, Abhinav Kumar, Sung Eun Kim, Xiaoman Zhang, Miao Zhu, Mahmoud Hussain Alabbad, Maha Sbayel Alhazmi, Neel P. Mistry, Kent Ryan Kleinschmidt, Brady Chrisler, Sathvik Suryadevara, Sri Sai Dinesh Jaliparthi, Noah Michael Prudlo, Mark David Marino, Jeremy Palacio, Rithvik Akula, Hong-Yu Zhou, Ibrahim Ethem Hamamci, Scott J. Adams, Hassan Rayhan AlOmaish, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.22030)  

**Abstract**: We present ReXGroundingCT, the first publicly available dataset to link free-text radiology findings with pixel-level segmentations in 3D chest CT scans that is manually annotated. While prior datasets have relied on structured labels or predefined categories, ReXGroundingCT captures the full expressiveness of clinical language represented in free text and grounds it to spatially localized 3D segmentation annotations in volumetric imaging. This addresses a critical gap in medical AI: the ability to connect complex, descriptive text, such as "3 mm nodule in the left lower lobe", to its precise anatomical location in three-dimensional space, a capability essential for grounded radiology report generation systems. The dataset comprises 3,142 non-contrast chest CT scans paired with standardized radiology reports from the CT-RATE dataset. Using a systematic three-stage pipeline, GPT-4 was used to extract positive lung and pleural findings, which were then manually segmented by expert annotators. A total of 8,028 findings across 16,301 entities were annotated, with quality control performed by board-certified radiologists. Approximately 79% of findings are focal abnormalities, while 21% are non-focal. The training set includes up to three representative segmentations per finding, while the validation and test sets contain exhaustive labels for each finding entity. ReXGroundingCT establishes a new benchmark for developing and evaluating sentence-level grounding and free-text medical segmentation models in chest CT. The dataset can be accessed at this https URL. 

**Abstract (ZH)**: ReXGroundingCT：第一个将自由文本放射学发现与3D胸CT扫描的像素级分割关联起来的手动标注数据集 

---
# Staining and locking computer vision models without retraining 

**Title (ZH)**: 不重新训练即染色和锁定计算机视觉模型 

**Authors**: Oliver J. Sutton, Qinghua Zhou, George Leete, Alexander N. Gorban, Ivan Y. Tyukin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22000)  

**Abstract**: We introduce new methods of staining and locking computer vision models, to protect their owners' intellectual property. Staining, also known as watermarking, embeds secret behaviour into a model which can later be used to identify it, while locking aims to make a model unusable unless a secret trigger is inserted into input images. Unlike existing methods, our algorithms can be used to stain and lock pre-trained models without requiring fine-tuning or retraining, and come with provable, computable guarantees bounding their worst-case false positive rates. The stain and lock are implemented by directly modifying a small number of the model's weights and have minimal impact on the (unlocked) model's performance. Locked models are unlocked by inserting a small `trigger patch' into the corner of the input image. We present experimental results showing the efficacy of our methods and demonstrating their practical performance on a variety of computer vision models. 

**Abstract (ZH)**: 我们介绍了新的染色和锁定方法，以保护计算机视觉模型所有者的知识产权。染色，也称为水印技术，将秘密行为嵌入到模型中，以后可以用于识别该模型，而锁定的目的是除非在输入图像中插入秘密触发器否则模型无法使用。与现有方法不同，我们的算法可以在不需要微调或重新训练的情况下对预训练模型进行染色和锁定，并且提供了计算可证明的最坏情况的假阳性率界。染色和锁定通过直接修改模型的少量权重实现，并对（解锁状态下的）模型性能影响极小。通过在输入图像的角落插入小型“触发补丁”可以解锁模型。我们展示了这些方法的有效性并演示了它们在多种计算机视觉模型上的实际性能。 

---
# Contrast-Prior Enhanced Duality for Mask-Free Shadow Removal 

**Title (ZH)**: 对比先验增强对偶性在无遮罩阴影去除中的应用 

**Authors**: Jiyu Wu, Yifan Liu, Jiancheng Huang, Mingfu Yan, Shifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21949)  

**Abstract**: Existing shadow removal methods often rely on shadow masks, which are challenging to acquire in real-world scenarios. Exploring intrinsic image cues, such as local contrast information, presents a potential alternative for guiding shadow removal in the absence of explicit masks. However, the cue's inherent ambiguity becomes a critical limitation in complex scenes, where it can fail to distinguish true shadows from low-reflectance objects and intricate background textures. To address this motivation, we propose the Adaptive Gated Dual-Branch Attention (AGBA) mechanism. AGBA dynamically filters and re-weighs the contrast prior to effectively disentangle shadow features from confounding visual elements. Furthermore, to tackle the persistent challenge of restoring soft shadow boundaries and fine-grained details, we introduce a diffusion-based Frequency-Contrast Fusion Network (FCFN) that leverages high-frequency and contrast cues to guide the generative process. Extensive experiments demonstrate that our method achieves state-of-the-art results among mask-free approaches while maintaining competitive performance relative to mask-based methods. 

**Abstract (ZH)**: 现有的阴影去除方法通常依赖于阴影掩模，但在实际场景中获取这些掩模具有挑战性。探索内在图像线索，如局部对比度信息，提供了在无明确掩模情况下指导阴影去除的潜在替代方案。然而，在复杂场景中，线索的固有模糊性成为了关键限制，可能导致无法将真实阴影与低反射物体和复杂的背景纹理区分开来。为了解决这一问题，我们提出了一种自适应门控双分支注意机制（AGBA）。AGBA动态筛选并重新加权对比度信息，以有效地分离出阴影特征与混淆的视觉元素。此外，为了解决恢复软阴影边界和细粒度细节的持续挑战，我们引入了一种基于扩散的频率-对比度融合网络（FCFN），利用高频和对比度线索来引导生成过程。广泛实验证明，我们的方法在无需掩模的方案中达到了最先进的性能，并且在与基于掩模的方法相比时，保持了竞争力。 

---
# SwinECAT: A Transformer-based fundus disease classification model with Shifted Window Attention and Efficient Channel Attention 

**Title (ZH)**: SwinECAT：一种基于变换器的 fundus 疾病分类模型，融合了移窗注意力和高效通道注意力 

**Authors**: Peiran Gu, Teng Yao, Mengshen He, Fuhao Duan, Feiyan Liu, RenYuan Peng, Bao Ge  

**Link**: [PDF](https://arxiv.org/pdf/2507.21922)  

**Abstract**: In recent years, artificial intelligence has been increasingly applied in the field of medical imaging. Among these applications, fundus image analysis presents special challenges, including small lesion areas in certain fundus diseases and subtle inter-disease differences, which can lead to reduced prediction accuracy and overfitting in the models. To address these challenges, this paper proposes the Transformer-based model SwinECAT, which combines the Shifted Window (Swin) Attention with the Efficient Channel Attention (ECA) Attention. SwinECAT leverages the Swin Attention mechanism in the Swin Transformer backbone to effectively capture local spatial structures and long-range dependencies within fundus images. The lightweight ECA mechanism is incorporated to guide the SwinECAT's attention toward critical feature channels, enabling more discriminative feature representation. In contrast to previous studies that typically classify fundus images into 4 to 6 categories, this work expands fundus disease classification to 9 distinct types, thereby enhancing the granularity of diagnosis. We evaluate our method on the Eye Disease Image Dataset (EDID) containing 16,140 fundus images for 9-category classification. Experimental results demonstrate that SwinECAT achieves 88.29\% accuracy, with weighted F1-score of 0.88 and macro F1-score of 0.90. The classification results of our proposed model SwinECAT significantly outperform the baseline Swin Transformer and multiple compared baseline models. To our knowledge, this represents the highest reported performance for 9-category classification on this public dataset. 

**Abstract (ZH)**: 基于变换器的SwinECAT模型在视网膜图像分析中的应用 

---
# APT: Improving Diffusion Models for High Resolution Image Generation with Adaptive Path Tracing 

**Title (ZH)**: APT：通过自适应路径跟踪提高高分辨率图像生成的扩散模型性能 

**Authors**: Sangmin Han, Jinho Jeong, Jinwoo Kim, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.21690)  

**Abstract**: Latent Diffusion Models (LDMs) are generally trained at fixed resolutions, limiting their capability when scaling up to high-resolution images. While training-based approaches address this limitation by training on high-resolution datasets, they require large amounts of data and considerable computational resources, making them less practical. Consequently, training-free methods, particularly patch-based approaches, have become a popular alternative. These methods divide an image into patches and fuse the denoising paths of each patch, showing strong performance on high-resolution generation. However, we observe two critical issues for patch-based approaches, which we call ``patch-level distribution shift" and ``increased patch monotonicity." To address these issues, we propose Adaptive Path Tracing (APT), a framework that combines Statistical Matching to ensure patch distributions remain consistent in upsampled latents and Scale-aware Scheduling to deal with the patch monotonicity. As a result, APT produces clearer and more refined details in high-resolution images. In addition, APT enables a shortcut denoising process, resulting in faster sampling with minimal quality degradation. Our experimental results confirm that APT produces more detailed outputs with improved inference speed, providing a practical approach to high-resolution image generation. 

**Abstract (ZH)**: 潜扩散模型（LDMs）通常在固定分辨率下训练，这限制了它们在扩大到高分辨率图像时的能力。虽然基于训练的方法通过在高分辨率数据集上进行训练来解决这一限制，但它们需要大量的数据和大量的计算资源，使其不够实用。因此，无训练方法，特别是基于补丁的方法，已成为一种流行的替代方案。这些方法将图像划分为补丁，并融合每个补丁的去噪路径，展现出在高分辨率生成方面的强大性能。然而，我们观察到补丁方法存在两个关键问题，称为“补丁级别分布偏移”和“增加的补丁单调性”。为了解决这些问题，我们提出了一种结合统计匹配和尺度感知调度的自适应路径追踪（APT）框架，以确保上采样潜在变量中的补丁分布保持一致，并处理补丁单调性问题。结果，APT 能够生成更清晰和更精细的高分辨率图像细节。此外，APT 允许一个捷径去噪过程，从而实现快速采样，且质量下降最小。我们的实验结果证实，APT 能够生成更详细且推理速度更快的输出，提供了一种高分辨率图像生成的实用方法。 

---
# Evaluating Deep Learning Models for African Wildlife Image Classification: From DenseNet to Vision Transformers 

**Title (ZH)**: 评估适用于非洲野生动物图像分类的深度学习模型：从DenseNet到视觉变换器 

**Authors**: Lukman Jibril Aliyu, Umar Sani Muhammad, Bilqisu Ismail, Nasiru Muhammad, Almustapha A Wakili, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad, Mustapha Abdullahi  

**Link**: [PDF](https://arxiv.org/pdf/2507.21364)  

**Abstract**: Wildlife populations in Africa face severe threats, with vertebrate numbers declining by over 65% in the past five decades. In response, image classification using deep learning has emerged as a promising tool for biodiversity monitoring and conservation. This paper presents a comparative study of deep learning models for automatically classifying African wildlife images, focusing on transfer learning with frozen feature extractors. Using a public dataset of four species: buffalo, elephant, rhinoceros, and zebra; we evaluate the performance of DenseNet-201, ResNet-152, EfficientNet-B4, and Vision Transformer ViT-H/14. DenseNet-201 achieved the best performance among convolutional networks (67% accuracy), while ViT-H/14 achieved the highest overall accuracy (99%), but with significantly higher computational cost, raising deployment concerns. Our experiments highlight the trade-offs between accuracy, resource requirements, and deployability. The best-performing CNN (DenseNet-201) was integrated into a Hugging Face Gradio Space for real-time field use, demonstrating the feasibility of deploying lightweight models in conservation settings. This work contributes to African-grounded AI research by offering practical insights into model selection, dataset preparation, and responsible deployment of deep learning tools for wildlife conservation. 

**Abstract (ZH)**: 非洲野生动物种群面临严重威胁，有脊椎动物数量在过去五十年中下降了超过65%。为应对这一挑战，基于深度学习的图像分类已成为生物多样性和保护监测的有前途的工具。本文对自动分类非洲野生动物图像的深度学习模型进行了比较研究，着重于冻结特征提取器的迁移学习。使用包含四种物种（水牛、大象、犀牛和斑马）的公共数据集，评估了DenseNet-201、ResNet-152、EfficientNet-B4和Vision Transformer ViT-H/14的性能。DenseNet-201在卷积网络中表现最佳（准确率为67%），而ViT-H/14实现了最高的总体准确率（99%），但计算成本显著更高，引发了部署方面的担忧。本文的研究结果突显了准确率、资源需求和可部署性之间的权衡。表现最佳的CNN（DenseNet-201）被集成到Hugging Face Gradio Space中，以实现实地实时使用，展示了如何在保护环境中部署轻量级模型的可行性。本文通过提供关于模型选择、数据集准备和负责任部署深度学习工具以促进野生动物保护的实际见解，为非洲本地化的人工智能研究做出了贡献。 

---
# Learning Simulatable Models of Cloth with Spatially-varying Constitutive Properties 

**Title (ZH)**: 学习具有空间变分本构性质的可模拟布料模型 

**Authors**: Guanxiong Chen, Shashwat Suri, Yuhao Wu, Etienne Voulga, David I.W. Levin, Dinesh Pai  

**Link**: [PDF](https://arxiv.org/pdf/2507.21288)  

**Abstract**: Materials used in real clothing exhibit remarkable complexity and spatial variation due to common processes such as stitching, hemming, dyeing, printing, padding, and bonding. Simulating these materials, for instance using finite element methods, is often computationally demanding and slow. Worse, such methods can suffer from numerical artifacts called ``membrane locking'' that makes cloth appear artificially stiff. Here we propose a general framework, called Mass-Spring Net, for learning a simple yet efficient surrogate model that captures the effects of these complex materials using only motion observations. The cloth is discretized into a mass-spring network with unknown material parameters that are learned directly from the motion data, using a novel force-and-impulse loss function. Our approach demonstrates the ability to accurately model spatially varying material properties from a variety of data sources, and immunity to membrane locking which plagues FEM-based simulations. Compared to graph-based networks and neural ODE-based architectures, our method achieves significantly faster training times, higher reconstruction accuracy, and improved generalization to novel dynamic scenarios. 

**Abstract (ZH)**: 用于学习从运动观测中捕获复杂材料效应的简单高效代理模型的质点弹簧网络框架 

---
# Generating Adversarial Point Clouds Using Diffusion Model 

**Title (ZH)**: 使用扩散模型生成对抗点云 

**Authors**: Ruiyang Zhao, Bingbing Zhu, Chuxuan Tong, Xiaoyi Zhou, Xi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.21163)  

**Abstract**: Adversarial attack methods for 3D point cloud classification reveal the vulnerabilities of point cloud recognition models. This vulnerability could lead to safety risks in critical applications that use deep learning models, such as autonomous vehicles. To uncover the deficiencies of these models, researchers can evaluate their security through adversarial attacks. However, most existing adversarial attack methods are based on white-box attacks. While these methods achieve high attack success rates and imperceptibility, their applicability in real-world scenarios is limited. Black-box attacks, which are more meaningful in real-world scenarios, often yield poor results. This paper proposes a novel black-box adversarial example generation method that utilizes a diffusion model to improve the attack success rate and imperceptibility in the black-box setting, without relying on the internal information of the point cloud classification model to generate adversarial samples. We use a 3D diffusion model to use the compressed features of the point cloud as prior knowledge to guide the reverse diffusion process to add adversarial points to clean examples. Subsequently, its reverse process is employed to transform the distribution of other categories into adversarial points, which are then added to the point cloud. 

**Abstract (ZH)**: 基于对抗攻击方法的3D点云分类揭示了点云识别模型的脆弱性：一种在黑盒设置中利用扩散模型生成对抗样本的新方法及其在自主车辆等关键应用中的安全性风险 

---
