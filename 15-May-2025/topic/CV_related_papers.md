# Learning Long-Context Diffusion Policies via Past-Token Prediction 

**Title (ZH)**: 基于过去词元预测的长上下文扩散策略学习 

**Authors**: Marcel Torne, Andy Tang, Yuejiang Liu, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2505.09561)  

**Abstract**: Reasoning over long sequences of observations and actions is essential for many robotic tasks. Yet, learning effective long-context policies from demonstrations remains challenging. As context length increases, training becomes increasingly expensive due to rising memory demands, and policy performance often degrades as a result of spurious correlations. Recent methods typically sidestep these issues by truncating context length, discarding historical information that may be critical for subsequent decisions. In this paper, we propose an alternative approach that explicitly regularizes the retention of past information. We first revisit the copycat problem in imitation learning and identify an opposite challenge in recent diffusion policies: rather than over-relying on prior actions, they often fail to capture essential dependencies between past and future actions. To address this, we introduce Past-Token Prediction (PTP), an auxiliary task in which the policy learns to predict past action tokens alongside future ones. This regularization significantly improves temporal modeling in the policy head, with minimal reliance on visual representations. Building on this observation, we further introduce a multistage training strategy: pre-train the visual encoder with short contexts, and fine-tune the policy head using cached long-context embeddings. This strategy preserves the benefits of PTP while greatly reducing memory and computational overhead. Finally, we extend PTP into a self-verification mechanism at test time, enabling the policy to score and select candidates consistent with past actions during inference. Experiments across four real-world and six simulated tasks demonstrate that our proposed method improves the performance of long-context diffusion policies by 3x and accelerates policy training by more than 10x. 

**Abstract (ZH)**: 长序列观测与动作推理是许多机器人任务的关键。然而，从演示中学习有效的长期上下文策略仍然具有挑战性。随着上下文长度的增加，由于内存需求的上升，训练变得越来越昂贵，政策性能往往会因虚假相关性而下降。最近的方法通常通过截断上下文长度来回避这些问题，从而忽略了对后续决策可能至关重要的历史信息。本文提出了一种替代方法，明确正则化保留过去信息。我们首先重新审视了模仿学习中的“复制猫”问题，并识别出最近扩散策略中的一个相反挑战：与过度依赖于先前的动作相比，它们往往未能捕捉到过去和未来动作之间的关键依赖性。为此，我们引入了过去动作令牌预测（PTP），这是一种辅助任务，在该任务中，策略学会同时预测过去的动作令牌和未来的动作令牌。这种正则化显著改进了策略头中的时间建模，同时减少了对视觉表示的依赖。在此基础上，我们进一步引入了一种多阶段训练策略：使用短上下文预训练视觉编码器，并使用缓存的长上下文嵌入微调策略头。这种策略保留了PTP的益处，同时大大降低了内存和计算开销。最后，我们在测试时将PTP扩展为一种自我验证机制，使政策能够在推理过程中为一致于过去动作的候选者评分并选择。在四个真实世界和六个模拟任务上的实验表明，我们提出的方法将长期上下文扩散策略的性能提升了3倍，并将策略训练速度加快了超过10倍。 

---
# Train a Multi-Task Diffusion Policy on RLBench-18 in One Day with One GPU 

**Title (ZH)**: 在一天内使用一块GPU训练一个针对RLBench-18的多任务扩散策略 

**Authors**: Yutong Hu, Pinhao Song, Kehan Wen, Renaud Detry  

**Link**: [PDF](https://arxiv.org/pdf/2505.09430)  

**Abstract**: We present a method for training multi-task vision-language robotic diffusion policies that reduces training time and memory usage by an order of magnitude. This improvement arises from a previously underexplored distinction between action diffusion and the image diffusion techniques that inspired it: image generation targets are high-dimensional, while robot actions lie in a much lower-dimensional space. Meanwhile, the vision-language conditions for action generation remain high-dimensional. Our approach, Mini-Diffuser, exploits this asymmetry by introducing Level-2 minibatching, which pairs multiple noised action samples with each vision-language condition, instead of the conventional one-to-one sampling strategy. To support this batching scheme, we introduce architectural adaptations to the diffusion transformer that prevent information leakage across samples while maintaining full conditioning access. In RLBench simulations, Mini-Diffuser achieves 95\% of the performance of state-of-the-art multi-task diffusion policies, while using only 5\% of the training time and 7\% of the memory. Real-world experiments further validate that Mini-Diffuser preserves the key strengths of diffusion-based policies, including the ability to model multimodal action distributions and produce behavior conditioned on diverse perceptual inputs. Code available at this http URL. 

**Abstract (ZH)**: 一种通过利用动作扩散和图像扩散之间维度差异来减少训练时间和内存使用量的多任务vision-language机器人扩散策略训练方法：Mini-Diffuser 

---
# APR-Transformer: Initial Pose Estimation for Localization in Complex Environments through Absolute Pose Regression 

**Title (ZH)**: APR-Transformer: 通过绝对位姿回归在复杂环境中的初始姿态估计定位 

**Authors**: Srinivas Ravuri, Yuan Xu, Martin Ludwig Zehetner, Ketan Motlag, Sahin Albayrak  

**Link**: [PDF](https://arxiv.org/pdf/2505.09356)  

**Abstract**: Precise initialization plays a critical role in the performance of localization algorithms, especially in the context of robotics, autonomous driving, and computer vision. Poor localization accuracy is often a consequence of inaccurate initial poses, particularly noticeable in GNSS-denied environments where GPS signals are primarily relied upon for initialization. Recent advances in leveraging deep neural networks for pose regression have led to significant improvements in both accuracy and robustness, especially in estimating complex spatial relationships and orientations. In this paper, we introduce APR-Transformer, a model architecture inspired by state-of-the-art methods, which predicts absolute pose (3D position and 3D orientation) using either image or LiDAR data. We demonstrate that our proposed method achieves state-of-the-art performance on established benchmark datasets such as the Radar Oxford Robot-Car and DeepLoc datasets. Furthermore, we extend our experiments to include our custom complex APR-BeIntelli dataset. Additionally, we validate the reliability of our approach in GNSS-denied environments by deploying the model in real-time on an autonomous test vehicle. This showcases the practical feasibility and effectiveness of our approach. The source code is available at:this https URL. 

**Abstract (ZH)**: 精确初始化在定位算法中的性能中扮演着至关重要的角色，特别是在机器人技术、自动驾驶和计算机视觉的背景下。在GNSS受限环境中，主要依赖GPS信号进行初始化时，初始姿态的不准确性通常会导致定位精度低下。近年来，利用深度神经网络进行姿态回归的进步显著提高了准确性和鲁棒性，尤其是在估计复杂的空间关系和方向方面。在本文中，我们提出了一种受最新方法启发的APR-Transformer模型架构，该模型使用图像或LiDAR数据预测绝对姿态（3D位置和3D方向）。我们证明，我们提出的方法在雷达牛津自动驾驶汽车和DeepLoc数据集等建立基准的数据集上实现了最先进的性能。此外，我们将实验扩展到包括我们自定义的复杂APR-BeIntelli数据集。我们还在实际环境中部署该模型于一辆自主测试车辆上，以验证其在GNSS受限环境中的可靠性。这展示了我们方法的实际可行性和有效性。源代码可在以下链接获取：this https URL。 

---
# FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis 

**Title (ZH)**: FoldNet：通过关键点驱动的资产和演示合成学习通用闭环折衣策略 

**Authors**: Yuxing Chen, Bowen Xiao, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09109)  

**Abstract**: Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 由于服装的可变形性，为机器人服装操作任务生成大量高质量数据极具挑战性。本文提出一个用于机器人服装折叠的合成服装数据集。我们首先基于关键点构建几何服装模板，并应用生成模型生成逼真的纹理图案。利用这些关键点标注，我们在仿真中生成折叠示范，并通过闭环模仿学习训练折叠策略。为了提高鲁棒性，我们提出了KG-DAgger方法，该方法使用关键点为基础的策略生成从失败中恢复的示范数据。KG-DAgger显著提高了模型性能，将实际成功率提升了25%。经过15K轨迹（约200万张图像-动作对）的训练，模型在实际环境中达到了75%的成功率。实验在仿真和实际环境设置中均验证了我们提出框架的有效性。 

---
# Parameter-Efficient Fine-Tuning of Vision Foundation Model for Forest Floor Segmentation from UAV Imagery 

**Title (ZH)**: 基于无人机imagery的森林地面分割的参数高效微调视觉基础模型 

**Authors**: Mohammad Wasil, Ahmad Drak, Brennan Penfold, Ludovico Scarton, Maximilian Johenneken, Alexander Asteroth, Sebastian Houben  

**Link**: [PDF](https://arxiv.org/pdf/2505.08932)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly used for reforestation and forest monitoring, including seed dispersal in hard-to-reach terrains. However, a detailed understanding of the forest floor remains a challenge due to high natural variability, quickly changing environmental parameters, and ambiguous annotations due to unclear definitions. To address this issue, we adapt the Segment Anything Model (SAM), a vision foundation model with strong generalization capabilities, to segment forest floor objects such as tree stumps, vegetation, and woody debris. To this end, we employ parameter-efficient fine-tuning (PEFT) to fine-tune a small subset of additional model parameters while keeping the original weights fixed. We adjust SAM's mask decoder to generate masks corresponding to our dataset categories, allowing for automatic segmentation without manual prompting. Our results show that the adapter-based PEFT method achieves the highest mean intersection over union (mIoU), while Low-rank Adaptation (LoRA), with fewer parameters, offers a lightweight alternative for resource-constrained UAV platforms. 

**Abstract (ZH)**: 无人 aerial 车辆 (UAVs) 越来越多地被用于植树造林和森林监测，包括在难以到达的地形中进行种子散布。然而，对森林地表的详细理解仍然面临挑战，原因包括高自然变异性、快速变化的环境参数以及由于定义不清而产生的模糊标注。为解决这一问题，我们适应了具有强大泛化能力的 Segment Anything 模型 (SAM)，以分割森林地表对象，如树桩、植被和木质残骸。为此，我们采用参数高效微调 (PEFT) 方法，对少量附加模型参数进行微调，同时固定原始权重。我们调整 SAM 的掩码解码器，使其生成与我们数据集类别对应的掩码，从而实现自动分割而无需手动提示。我们的结果显示，基于适配器的 PEFT 方法取得了最高的平均交并比 (mIoU)，而具有较少参数的低秩适应 (LoRA) 则为资源受限的 UAV 平台提供了轻量级替代方案。 

---
# OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions 

**Title (ZH)**: OpenLKA: 现代汽车车型在实际驾驶条件下的一种车道保持辅助数据集 

**Authors**: Yuhang Wang, Abdulaziz Alhuraish, Shengming Yuan, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.09092)  

**Abstract**: Lane Keeping Assist (LKA) is widely adopted in modern vehicles, yet its real-world performance remains underexplored due to proprietary systems and limited data access. This paper presents OpenLKA, the first open, large-scale dataset for LKA evaluation and improvement. It includes 400 hours of driving data from 50+ production vehicle models, collected through extensive road testing in Tampa, Florida and global contributions from the this http URL driving community. The dataset spans a wide range of challenging scenarios, including complex road geometries, degraded lane markings, adverse weather, lighting conditions and surrounding traffic. The dataset is multimodal, comprising: i) full CAN bus streams, decoded using custom reverse-engineered DBC files to extract key LKA events (e.g., system disengagements, lane detection failures); ii) synchronized high-resolution dash-cam video; iii) real-time outputs from Openpilot, providing accurate estimates of road curvature and lane positioning; iv) enhanced scene annotations generated by Vision Language Models, describing lane visibility, pavement quality, weather, lighting, and traffic conditions. By integrating vehicle-internal signals with high-fidelity perception and rich semantic context, OpenLKA provides a comprehensive platform for benchmarking the real-world performance of production LKA systems, identifying safety-critical operational scenarios, and assessing the readiness of current road infrastructure for autonomous driving. The dataset is publicly available at: this https URL. 

**Abstract (ZH)**: 开放路径保持辅助（OpenLKA）：首个开放的大规模路径保持辅助数据集 

---
# TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian 

**Title (ZH)**: TUGS: 基于物理的 underwater 场景张量化高斯紧凑表示 

**Authors**: Shijie Lian, Ziyi Zhang, Laurence Tianruo Yang and, Mengyu Ren, Debin Liu, Hua Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08811)  

**Abstract**: Underwater 3D scene reconstruction is crucial for undewater robotic perception and navigation. However, the task is significantly challenged by the complex interplay between light propagation, water medium, and object surfaces, with existing methods unable to model their interactions accurately. Additionally, expensive training and rendering costs limit their practical application in underwater robotic systems. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), which can effectively solve the modeling challenges of the complex interactions between object geometries and water media while achieving significant parameter reduction. TUGS employs lightweight tensorized higher-order Gaussians with a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments. Compared to other NeRF-based and GS-based methods designed for underwater, TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters, making it particularly suitable for memory-constrained underwater UAV applications 

**Abstract (ZH)**: 水下3D场景重建对于水下机器人感知和导航至关重要。然而，任务受到光线传播、水中介质和物体表面之间复杂相互作用的显著挑战，现有方法难以准确建模这些交互作用。此外，高昂的训练和渲染成本限制了其在水下机器人系统中的实际应用。因此，我们提出了张量水下高斯点云表示（Tensorized Underwater Gaussian Splatting，TUGS），它可以有效解决物体几何与水介质之间复杂交互的建模挑战，同时实现显著的参数减少。TUGS 使用基于物理的水下自适应介质估计（AME）模块和轻量级张量高阶高斯函数，能够准确模拟水下环境中的光线衰减和后向散射效应。与为水下设计的其他基于NeRF和GS的方法相比，TUGS 能以更快的渲染速度和更少的内存使用渲染高质量的水下图像。在真实世界水下数据集上的大量实验表明，TUGS 能够使用有限的参数高效地实现卓越的重建质量，使其特别适合于内存受限的水下无人机应用。 

---
# Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput 

**Title (ZH)**: Flash-VL 2B: 优化超低延迟和高吞吐量的视觉语言模型性能 

**Authors**: Bo Zhang, Shuo Li, Runhe Tian, Yang Yang, Jixin Tang, Jinhao Zhou, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.09498)  

**Abstract**: In this paper, we introduce Flash-VL 2B, a novel approach to optimizing Vision-Language Models (VLMs) for real-time applications, targeting ultra-low latency and high throughput without sacrificing accuracy. Leveraging advanced architectural enhancements and efficient computational strategies, Flash-VL 2B is designed to maximize throughput by reducing processing time while maintaining competitive performance across multiple vision-language benchmarks. Our approach includes tailored architectural choices, token compression mechanisms, data curation, training schemes, and a novel image processing technique called implicit semantic stitching that effectively balances computational load and model performance. Through extensive evaluations on 11 standard VLM benchmarks, we demonstrate that Flash-VL 2B achieves state-of-the-art results in both speed and accuracy, making it a promising solution for deployment in resource-constrained environments and large-scale real-time applications. 

**Abstract (ZH)**: 在本文中，我们介绍了Flash-VL 2B，这是一种针对实时应用优化视觉-语言模型的新方法，旨在实现超低延迟和高吞吐量而不牺牲准确性。通过利用先进的架构增强和高效的计算策略，Flash-VL 2B 设计旨在通过减少处理时间来最大化吞吐量，同时在多个视觉-语言基准上保持竞争力。我们的方法包括定制的架构选择、标记压缩机制、数据收集、训练方案以及一种名为隐式语义缝合的新型图像处理技术，以有效地平衡计算负载和模型性能。通过在11个标准视觉-语言模型基准上的广泛评估，我们证明了Flash-VL 2B 在速度和准确性上均达到业内领先水平，使其成为资源受限环境和大规模实时应用程序部署的有前途的解决方案。 

---
# A 2D Semantic-Aware Position Encoding for Vision Transformers 

**Title (ZH)**: 二维语义感知位置编码用于视觉变换器 

**Authors**: Xi Chen, Shiyang Zhou, Muqi Huang, Jiaxu Feng, Yun Xiong, Kun Zhou, Biao Yang, Yuhui Zhang, Huishuai Bao, Sijia Peng, Chuan Li, Feng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09466)  

**Abstract**: Vision transformers have demonstrated significant advantages in computer vision tasks due to their ability to capture long-range dependencies and contextual relationships through self-attention. However, existing position encoding techniques, which are largely borrowed from natural language processing, fail to effectively capture semantic-aware positional relationships between image patches. Traditional approaches like absolute position encoding and relative position encoding primarily focus on 1D linear position relationship, often neglecting the semantic similarity between distant yet contextually related patches. These limitations hinder model generalization, translation equivariance, and the ability to effectively handle repetitive or structured patterns in images. In this paper, we propose 2-Dimensional Semantic-Aware Position Encoding ($\text{SaPE}^2$), a novel position encoding method with semantic awareness that dynamically adapts position representations by leveraging local content instead of fixed linear position relationship or spatial coordinates. Our method enhances the model's ability to generalize across varying image resolutions and scales, improves translation equivariance, and better aggregates features for visually similar but spatially distant patches. By integrating $\text{SaPE}^2$ into vision transformers, we bridge the gap between position encoding and perceptual similarity, thereby improving performance on computer vision tasks. 

**Abstract (ZH)**: Vision Transformners的2维语义感知位置编码(SaPE²)：通过利用局部内容动态适应位置表示以增强模型在计算机视觉任务中的性能 

---
# Endo-CLIP: Progressive Self-Supervised Pre-training on Raw Colonoscopy Records 

**Title (ZH)**: Endo-CLIP: 基于原始肠镜记录的分阶段自我监督预训练 

**Authors**: Yili He, Yan Zhu, Peiyao Fu, Ruijie Yang, Tianyi Chen, Zhihua Wang, Quanlin Li, Pinghong Zhou, Xian Yang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09435)  

**Abstract**: Pre-training on image-text colonoscopy records offers substantial potential for improving endoscopic image analysis, but faces challenges including non-informative background images, complex medical terminology, and ambiguous multi-lesion descriptions. We introduce Endo-CLIP, a novel self-supervised framework that enhances Contrastive Language-Image Pre-training (CLIP) for this domain. Endo-CLIP's three-stage framework--cleansing, attunement, and unification--addresses these challenges by (1) removing background frames, (2) leveraging large language models to extract clinical attributes for fine-grained contrastive learning, and (3) employing patient-level cross-attention to resolve multi-polyp ambiguities. Extensive experiments demonstrate that Endo-CLIP significantly outperforms state-of-the-art pre-training methods in zero-shot and few-shot polyp detection and classification, paving the way for more accurate and clinically relevant endoscopic analysis. 

**Abstract (ZH)**: 基于内镜图像-文本结肠镜检查记录的预训练在内镜图像分析中具有巨大潜力，但面临背景图像无信息性、复杂医学术语和多病灶描述模糊等挑战。我们介绍了Endo-CLIP，这是一种新颖的自监督框架，旨在增强该领域的对比语言-图像预训练（CLIP）。Endo-CLIP 的三阶段框架——净化、调谐和统一——通过（1）去除背景帧，（2）利用大规模语言模型提取临床属性以实现精细对比学习，以及（3）采用病患级别的跨注意力解决多腺瘤模糊性，来应对这些挑战。广泛实验表明，Endo-CLIP 在零样本和少样本息肉检测与分类中显著优于现有预训练方法，为更准确和临床相关性更强的内镜分析铺平了道路。 

---
# FedSaaS: Class-Consistency Federated Semantic Segmentation via Global Prototype Supervision and Local Adversarial Harmonization 

**Title (ZH)**: FedSaaS: 基于全局原型监督和局部对抗 harmonization 的类一致联邦语义分割 

**Authors**: Xiaoyang Yu, Xiaoming Wu, Xin Wang, Dongrun Li, Ming Yang, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.09385)  

**Abstract**: Federated semantic segmentation enables pixel-level classification in images through collaborative learning while maintaining data privacy. However, existing research commonly overlooks the fine-grained class relationships within the semantic space when addressing heterogeneous problems, particularly domain shift. This oversight results in ambiguities between class representation. To overcome this challenge, we propose a novel federated segmentation framework that strikes class consistency, termed FedSaaS. Specifically, we introduce class exemplars as a criterion for both local- and global-level class representations. On the server side, the uploaded class exemplars are leveraged to model class prototypes, which supervise global branch of clients, ensuring alignment with global-level representation. On the client side, we incorporate an adversarial mechanism to harmonize contributions of global and local branches, leading to consistent output. Moreover, multilevel contrastive losses are employed on both sides to enforce consistency between two-level representations in the same semantic space. Extensive experiments on several driving scene segmentation datasets demonstrate that our framework outperforms state-of-the-art methods, significantly improving average segmentation accuracy and effectively addressing the class-consistency representation problem. 

**Abstract (ZH)**: 联邦语义分割通过协作学习在保持数据隐私的同时实现像素级分类，但现有研究在处理异构问题时，尤其是领域转移问题时，通常忽略了语义空间内的细粒度类关系，导致类表示之间的模糊性。为解决这一挑战，我们提出了一种新的联邦分割框架FedSaaS，该框架确保类的一致性。具体而言，我们引入类示例作为局部和全局类表示的标准。在服务器端，上传的类示例用于建模类原型，监督客户端的全局分支，确保与全局水平表示的一致性。在客户端，我们引入对抗机制以协调全局和局部分支的贡献，从而实现一致的输出。此外，我们还在双方使用多级对比损失，以确保相同语义空间中两层表示的一致性。在几个驾驶场景分割数据集上的 extensive 实验表明，我们的框架优于现有方法，在平均分割准确性和有效地解决类一致表示问题方面表现出显著提升。 

---
# BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models for Biomedical Image Analysis 

**Title (ZH)**: BioVFM-21M：自我监督视觉基础模型在生物医学图像分析中的基准测试与扩展研究 

**Authors**: Jiarun Liu, Hong-Yu Zhou, Weijian Huang, Hao Yang, Dongning Song, Tao Tan, Yong Liang, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09329)  

**Abstract**: Scaling up model and data size have demonstrated impressive performance improvement over a wide range of tasks. Despite extensive studies on scaling behaviors for general-purpose tasks, medical images exhibit substantial differences from natural data. It remains unclear the key factors in developing medical vision foundation models at scale due to the absence of an extensive understanding of scaling behavior in the medical domain. In this paper, we explored the scaling behavior across model sizes, training algorithms, data sizes, and imaging modalities in developing scalable medical vision foundation models by self-supervised learning. To support scalable pretraining, we introduce BioVFM-21M, a large-scale biomedical image dataset encompassing a wide range of biomedical image modalities and anatomies. We observed that scaling up does provide benefits but varies across tasks. Additional analysis reveals several factors correlated with scaling benefits. Finally, we propose BioVFM, a large-scale medical vision foundation model pretrained on 21 million biomedical images, which outperforms the previous state-of-the-art foundation models across 12 medical benchmarks. Our results highlight that while scaling up is beneficial for pursuing better performance, task characteristics, data diversity, pretraining methods, and computational efficiency remain critical considerations for developing scalable medical foundation models. 

**Abstract (ZH)**: 扩展模型和数据规模在广泛的任务中展现了显著的性能提升。尽管已经对通用任务的扩展行为进行了大量研究，但医学图像与自然数据之间存在显著差异。由于对医学领域中扩展行为缺乏广泛的理解，开发大规模医学视觉基础模型的关键因素尚不清楚。在本文中，我们通过自监督学习探索了在模型规模、训练算法、数据规模和成像模态方面开发可扩展的医学视觉基础模型的扩展行为。为了支持可扩展的预训练，我们引入了BioVFM-21M，这是一个大规模的生物医学图像数据集，涵盖了多种生物医学图像模态和解剖结构。我们观察到，扩展确实提供了益处，但这些益处在不同任务中有所不同。进一步的分析揭示了一些与扩展益处相关的因素。最后，我们提出了BioVFM，这是一个在2100万生物医学图像上预训练的大规模医学视觉基础模型，它在12项医学基准测试中优于之前的最佳基础模型。我们的结果表明，虽然扩展有助于提高性能，但任务特征、数据多样性、预训练方法和计算效率仍然是开发可扩展的医学基础模型的关键考虑因素。 

---
# Neural Video Compression using 2D Gaussian Splatting 

**Title (ZH)**: 基于2D高斯点绘制的神经视频压缩 

**Authors**: Lakshya Gupta, Imran N. Junejo  

**Link**: [PDF](https://arxiv.org/pdf/2505.09324)  

**Abstract**: The computer vision and image processing research community has been involved in standardizing video data communications for the past many decades, leading to standards such as AVC, HEVC, VVC, AV1, AV2, etc. However, recent groundbreaking works have focused on employing deep learning-based techniques to replace the traditional video codec pipeline to a greater affect. Neural video codecs (NVC) create an end-to-end ML-based solution that does not rely on any handcrafted features (motion or edge-based) and have the ability to learn content-aware compression strategies, offering better adaptability and higher compression efficiency than traditional methods. This holds a great potential not only for hardware design, but also for various video streaming platforms and applications, especially video conferencing applications such as MS-Teams or Zoom that have found extensive usage in classrooms and workplaces. However, their high computational demands currently limit their use in real-time applications like video conferencing. To address this, we propose a region-of-interest (ROI) based neural video compression model that leverages 2D Gaussian Splatting. Unlike traditional codecs, 2D Gaussian Splatting is capable of real-time decoding and can be optimized using fewer data points, requiring only thousands of Gaussians for decent quality outputs as opposed to millions in 3D scenes. In this work, we designed a video pipeline that speeds up the encoding time of the previous Gaussian splatting-based image codec by 88% by using a content-aware initialization strategy paired with a novel Gaussian inter-frame redundancy-reduction mechanism, enabling Gaussian splatting to be used for a video-codec solution, the first of its kind solution in this neural video codec space. 

**Abstract (ZH)**: 基于区域兴趣的2D高斯斑点神经视频压缩模型 

---
# MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning 

**Title (ZH)**: MetaUAS: 通用异常分割的一提示元学习 

**Authors**: Bin-Bin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09265)  

**Abstract**: Zero- and few-shot visual anomaly segmentation relies on powerful vision-language models that detect unseen anomalies using manually designed textual prompts. However, visual representations are inherently independent of language. In this paper, we explore the potential of a pure visual foundation model as an alternative to widely used vision-language models for universal visual anomaly segmentation. We present a novel paradigm that unifies anomaly segmentation into change segmentation. This paradigm enables us to leverage large-scale synthetic image pairs, featuring object-level and local region changes, derived from existing image datasets, which are independent of target anomaly datasets. We propose a one-prompt Meta-learning framework for Universal Anomaly Segmentation (MetaUAS) that is trained on this synthetic dataset and then generalizes well to segment any novel or unseen visual anomalies in the real world. To handle geometrical variations between prompt and query images, we propose a soft feature alignment module that bridges paired-image change perception and single-image semantic segmentation. This is the first work to achieve universal anomaly segmentation using a pure vision model without relying on special anomaly detection datasets and pre-trained visual-language models. Our method effectively and efficiently segments any anomalies with only one normal image prompt and enjoys training-free without guidance from language. Our MetaUAS significantly outperforms previous zero-shot, few-shot, and even full-shot anomaly segmentation methods. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 纯视觉基础模型赋能通用视觉异常分割：无需依赖特殊异常检测数据集和预训练视觉语言模型的通用异常分割新范式 

---
# Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt 

**Title (ZH)**: 仅凭一张正常图像提示学习检测多类异常 

**Authors**: Bin-Bin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09264)  

**Abstract**: Unsupervised reconstruction networks using self-attention transformers have achieved state-of-the-art performance for multi-class (unified) anomaly detection with a single model. However, these self-attention reconstruction models primarily operate on target features, which may result in perfect reconstruction for both normal and anomaly features due to high consistency with context, leading to failure in detecting anomalies. Additionally, these models often produce inaccurate anomaly segmentation due to performing reconstruction in a low spatial resolution latent space. To enable reconstruction models enjoying high efficiency while enhancing their generalization for unified anomaly detection, we propose a simple yet effective method that reconstructs normal features and restores anomaly features with just One Normal Image Prompt (OneNIP). In contrast to previous work, OneNIP allows for the first time to reconstruct or restore anomalies with just one normal image prompt, effectively boosting unified anomaly detection performance. Furthermore, we propose a supervised refiner that regresses reconstruction errors by using both real normal and synthesized anomalous images, which significantly improves pixel-level anomaly segmentation. OneNIP outperforms previous methods on three industry anomaly detection benchmarks: MVTec, BTAD, and VisA. The code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 使用自注意变换器的无监督重建网络已实现单一模型在多类综合异常检测中的最先进性能。然而，这些自注意重建模型主要处理目标特征，由于上下文一致性高，可能导致正常和异常特征的完美重建，从而导致异常检测失败。此外，这些模型因在低空间分辨率的潜在空间中执行重建而经常产生不准确的异常分割。为使重建模型既保持高效率又能增强其统一异常检测的泛化能力，我们提出了一种简单有效的“一个正常图像提示”(OneNIP) 方法，该方法仅通过一个正常图像提示重建正常特征并恢复异常特征。与先前的工作不同，OneNIP 允许以史无前例的方式仅通过一个正常图像提示重建或恢复异常，显著提升了统一异常检测性能。此外，我们提出了一种监督修整器，通过同时使用真实正常图像和合成异常图像回归重建误差，显著提高了像素级异常分割。OneNIP 在 MVTec、BTAD 和 VisA 三个工业异常检测基准数据集上表现出色。相关代码和预训练模型可在以下链接获得。 

---
# DRRNet: Macro-Micro Feature Fusion and Dual Reverse Refinement for Camouflaged Object Detection 

**Title (ZH)**: DRRNet：宏观-微观特征融合与双重反向精炼在伪装目标检测中的应用 

**Authors**: Jianlin Sun, Xiaolin Fang, Juwei Guan, Dongdong Gui, Teqi Wang, Tongxin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09168)  

**Abstract**: The core challenge in Camouflage Object Detection (COD) lies in the indistinguishable similarity between targets and backgrounds in terms of color, texture, and shape. This causes existing methods to either lose edge details (such as hair-like fine structures) due to over-reliance on global semantic information or be disturbed by similar backgrounds (such as vegetation patterns) when relying solely on local features. We propose DRRNet, a four-stage architecture characterized by a "context-detail-fusion-refinement" pipeline to address these issues. Specifically, we introduce an Omni-Context Feature Extraction Module to capture global camouflage patterns and a Local Detail Extraction Module to supplement microstructural information for the full-scene context module. We then design a module for forming dual representations of scene understanding and structural awareness, which fuses panoramic features and local features across various scales. In the decoder, we also introduce a reverse refinement module that leverages spatial edge priors and frequency-domain noise suppression to perform a two-stage inverse refinement of the output. By applying two successive rounds of inverse refinement, the model effectively suppresses background interference and enhances the continuity of object boundaries. Experimental results demonstrate that DRRNet significantly outperforms state-of-the-art methods on benchmark datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 伪装目标检测（COD）的核心挑战在于目标与背景在颜色、纹理和形状方面的难以区分的相似性。这导致现有方法要么由于过度依赖全局语义信息而丢失边缘细节（如发丝般的微细结构），要么由于仅依赖局部特征而受到类似背景（如植被模式）的干扰。我们提出了一种名为DRRNet的四阶段架构，该架构通过“上下文-细节-融合-精炼”管道来解决这些问题。具体而言，我们引入了一种全方位上下文特征提取模块以捕获全局伪装模式，并引入了一种局部细节提取模块以补充全场景上下文模块的微细结构信息。我们还设计了一种模块以形成场景理解和结构意识的双表示，并融合不同尺度的全景特征和局部特征。在解码器中，我们还引入了一种逆精炼模块，利用空间边缘先验和频域噪声抑制来执行输出的两级逆精炼。通过应用两轮逆精炼，模型有效地抑制了背景干扰并增强了对象边界的连续性。实验结果表明，DRRNet在基准数据集上显著优于现有方法。我们的代码可在以下链接获取。 

---
# Template-Guided Reconstruction of Pulmonary Segments with Neural Implicit Functions 

**Title (ZH)**: 基于模板引导的肺段重建：神经隐式函数方法 

**Authors**: Kangxian Xie, Yufei Zhu, Kaiming Kuang, Li Zhang, Hongwei Bran Li, Mingchen Gao, Jiancheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08919)  

**Abstract**: High-quality 3D reconstruction of pulmonary segments plays a crucial role in segmentectomy and surgical treatment planning for lung cancer. Due to the resolution requirement of the target reconstruction, conventional deep learning-based methods often suffer from computational resource constraints or limited granularity. Conversely, implicit modeling is favored due to its computational efficiency and continuous representation at any resolution. We propose a neural implicit function-based method to learn a 3D surface to achieve anatomy-aware, precise pulmonary segment reconstruction, represented as a shape by deforming a learnable template. Additionally, we introduce two clinically relevant evaluation metrics to assess the reconstruction comprehensively. Further, due to the absence of publicly available shape datasets to benchmark reconstruction algorithms, we developed a shape dataset named Lung3D, including the 3D models of 800 labeled pulmonary segments and the corresponding airways, arteries, veins, and intersegmental veins. We demonstrate that the proposed approach outperforms existing methods, providing a new perspective for pulmonary segment reconstruction. Code and data will be available at this https URL. 

**Abstract (ZH)**: 高质量的肺段三维重建在肺段切除及肺癌手术治疗规划中发挥着关键作用。由于重建所需的分辨率要求，传统的基于深度学习的方法往往受到计算资源限制或粒度有限的约束。相比之下，隐式建模由于其计算效率和任意分辨率下的连续表示而受到青睐。我们提出了一种基于神经隐式函数的方法，通过变形可学习模板来学习一个3D表面，以实现解剖感知的、精确的肺段重建。此外，我们引入了两个临床相关的评估指标来全面评估重建效果。由于缺乏用于基准测试重建算法的公开形状数据集，我们开发了一个名为Lung3D的数据集，包含800个标注的肺段3D模型及其对应的气道、动脉、静脉和段间静脉。我们证明了所提出的方法优于现有方法，为肺段重建提供了新的视角。代码和数据可在以下链接获取。 

---
# Crowd Scene Analysis using Deep Learning Techniques 

**Title (ZH)**: 基于深度学习技术的 crowd 场景分析 

**Authors**: Muhammad Junaid Asif  

**Link**: [PDF](https://arxiv.org/pdf/2505.08834)  

**Abstract**: Our research is focused on two main applications of crowd scene analysis crowd counting and anomaly detection In recent years a large number of researches have been presented in the domain of crowd counting We addressed two main challenges in this domain 1 Deep learning models are datahungry paradigms and always need a large amount of annotated data for the training of algorithm It is timeconsuming and costly task to annotate such large amount of data Selfsupervised training is proposed to deal with this challenge 2 MCNN consists of multicolumns of CNN with different sizes of filters by presenting a novel approach based on a combination of selfsupervised training and MultiColumn CNN This enables the model to learn features at different levels and makes it effective in dealing with challenges of occluded scenes nonuniform density complex backgrounds and scale invariation The proposed model was evaluated on publicly available data sets such as ShanghaiTech and UCFQNRF by means of MAE and MSE A spatiotemporal model based on VGG19 is proposed for crowd anomaly detection addressing challenges like lighting environmental conditions unexpected objects and scalability The model extracts spatial and temporal features allowing it to be generalized to realworld scenes Spatial features are learned using CNN while temporal features are learned using LSTM blocks The model works on binary classification and can detect normal or abnormal behavior The models performance is improved by replacing fully connected layers with dense residual blocks Experiments on the Hockey Fight dataset and SCVD dataset show our models outperform other stateoftheart approaches 

**Abstract (ZH)**: crowd场景分析中的 crowd计数和异常检测研究：基于自监督训练和多柱卷积神经网络的方法

基于自监督训练和多柱卷积神经网络的 crowd计数研究

基于自监督训练和多柱卷积神经网络的 crowd计数和异常检测研究 

---
# In-Context Learning for Label-Efficient Cancer Image Classification in Oncology 

**Title (ZH)**: 基于上下文的学习在肿瘤学中的标签高效癌症图像分类 

**Authors**: Mobina Shrestha, Bishwas Mandal, Vishal Mandal, Asis Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2505.08798)  

**Abstract**: The application of AI in oncology has been limited by its reliance on large, annotated datasets and the need for retraining models for domain-specific diagnostic tasks. Taking heed of these limitations, we investigated in-context learning as a pragmatic alternative to model retraining by allowing models to adapt to new diagnostic tasks using only a few labeled examples at inference, without the need for retraining. Using four vision-language models (VLMs)-Paligemma, CLIP, ALIGN and GPT-4o, we evaluated the performance across three oncology datasets: MHIST, PatchCamelyon and HAM10000. To the best of our knowledge, this is the first study to compare the performance of multiple VLMs on different oncology classification tasks. Without any parameter updates, all models showed significant gains with few-shot prompting, with GPT-4o reaching an F1 score of 0.81 in binary classification and 0.60 in multi-class classification settings. While these results remain below the ceiling of fully fine-tuned systems, they highlight the potential of ICL to approximate task-specific behavior using only a handful of examples, reflecting how clinicians often reason from prior cases. Notably, open-source models like Paligemma and CLIP demonstrated competitive gains despite their smaller size, suggesting feasibility for deployment in computing constrained clinical environments. Overall, these findings highlight the potential of ICL as a practical solution in oncology, particularly for rare cancers and resource-limited contexts where fine-tuning is infeasible and annotated data is difficult to obtain. 

**Abstract (ZH)**: AI在肿瘤学中的应用受限于其对大规模标注数据的依赖以及需要为领域特定诊断任务重新训练模型。鉴于这些限制，我们研究了上下文学习作为一种实用的替代方案，允许模型仅通过少量标注示例在推理时适应新的诊断任务，而无需重新训练。我们使用四种视觉-语言模型（VLMs）——Paligemma、CLIP、ALIGN和GPT-4o，在三个肿瘤学数据集（MHIST、PatchCamelyon和HAM10000）上评估了性能。据我们所知，这是首次将多种VLMs在不同的肿瘤分类任务上进行性能比较的研究。无需任何参数更新，所有模型在少样本提示下均显示出显著提升，其中GPT-4o在二分类和多分类设置下的F1分数分别为0.81和0.60。虽然这些结果仍低于完全微调系统的天花板，但它们突显了上下文学习通过少量示例逼近任务特定行为的潜力，反映出临床医生通常如何从以往病例中推理。值得注意的是，开源模型Paligemma和CLIP尽管规模较小，但仍然表现出竞争力的提升，表明其在计算资源受限的临床环境中部署的可行性。总体而言，这些发现强调了上下文学习作为肿瘤学中实用解决方案的潜力，特别是在肿瘤罕见且资源有限的情况下，微调不可行且标注数据难以获取的背景下。 

---
