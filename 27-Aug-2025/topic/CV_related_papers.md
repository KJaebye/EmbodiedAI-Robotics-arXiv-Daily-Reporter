# VibES: Induced Vibration for Persistent Event-Based Sensing 

**Title (ZH)**: VibES: 引起振动以实现持久的事件驱动传感 

**Authors**: Vincenzo Polizzi, Stephen Yang, Quentin Clark, Jonathan Kelly, Igor Gilitschenski, David B. Lindell  

**Link**: [PDF](https://arxiv.org/pdf/2508.19094)  

**Abstract**: Event cameras are a bio-inspired class of sensors that asynchronously measure per-pixel intensity changes. Under fixed illumination conditions in static or low-motion scenes, rigidly mounted event cameras are unable to generate any events, becoming unsuitable for most computer vision tasks. To address this limitation, recent work has investigated motion-induced event stimulation that often requires complex hardware or additional optical components. In contrast, we introduce a lightweight approach to sustain persistent event generation by employing a simple rotating unbalanced mass to induce periodic vibrational motion. This is combined with a motion-compensation pipeline that removes the injected motion and yields clean, motion-corrected events for downstream perception tasks. We demonstrate our approach with a hardware prototype and evaluate it on real-world captured datasets. Our method reliably recovers motion parameters and improves both image reconstruction and edge detection over event-based sensing without motion induction. 

**Abstract (ZH)**: 基于运动诱导的事件生成轻量级方法及其实验验证：改善事件感知中的运动参数恢复、图像重建和边缘检测 

---
# PseudoMapTrainer: Learning Online Mapping without HD Maps 

**Title (ZH)**: 伪地图训练器：基于在线映射的高精度地图学习 

**Authors**: Christian Löwens, Thorben Funke, Jingchao Xie, Alexandru Paul Condurache  

**Link**: [PDF](https://arxiv.org/pdf/2508.18788)  

**Abstract**: Online mapping models show remarkable results in predicting vectorized maps from multi-view camera images only. However, all existing approaches still rely on ground-truth high-definition maps during training, which are expensive to obtain and often not geographically diverse enough for reliable generalization. In this work, we propose PseudoMapTrainer, a novel approach to online mapping that uses pseudo-labels generated from unlabeled sensor data. We derive those pseudo-labels by reconstructing the road surface from multi-camera imagery using Gaussian splatting and semantics of a pre-trained 2D segmentation network. In addition, we introduce a mask-aware assignment algorithm and loss function to handle partially masked pseudo-labels, allowing for the first time the training of online mapping models without any ground-truth maps. Furthermore, our pseudo-labels can be effectively used to pre-train an online model in a semi-supervised manner to leverage large-scale unlabeled crowdsourced data. The code is available at this http URL. 

**Abstract (ZH)**: 基于伪标签的在线地图训练：无需地理标记高清地图的多视图相机图像矢量化地图预测 

---
# Towards Training-Free Underwater 3D Object Detection from Sonar Point Clouds: A Comparison of Traditional and Deep Learning Approaches 

**Title (ZH)**: 无需训练的水下3D物体检测从声呐点云：传统方法与深度学习方法的比较 

**Authors**: M. Salman Shaukat, Yannik Käckenmeister, Sebastian Bader, Thomas Kirste  

**Link**: [PDF](https://arxiv.org/pdf/2508.18293)  

**Abstract**: Underwater 3D object detection remains one of the most challenging frontiers in computer vision, where traditional approaches struggle with the harsh acoustic environment and scarcity of training data. While deep learning has revolutionized terrestrial 3D detection, its application underwater faces a critical bottleneck: obtaining sufficient annotated sonar data is prohibitively expensive and logistically complex, often requiring specialized vessels, expert surveyors, and favorable weather conditions. This work addresses a fundamental question: Can we achieve reliable underwater 3D object detection without real-world training data? We tackle this challenge by developing and comparing two paradigms for training-free detection of artificial structures in multibeam echo-sounder point clouds. Our dual approach combines a physics-based sonar simulation pipeline that generates synthetic training data for state-of-the-art neural networks, with a robust model-based template matching system that leverages geometric priors of target objects. Evaluation on real bathymetry surveys from the Baltic Sea reveals surprising insights: while neural networks trained on synthetic data achieve 98% mean Average Precision (mAP) on simulated scenes, they drop to 40% mAP on real sonar data due to domain shift. Conversely, our template matching approach maintains 83% mAP on real data without requiring any training, demonstrating remarkable robustness to acoustic noise and environmental variations. Our findings challenge conventional wisdom about data-hungry deep learning in underwater domains and establish the first large-scale benchmark for training-free underwater 3D detection. This work opens new possibilities for autonomous underwater vehicle navigation, marine archaeology, and offshore infrastructure monitoring in data-scarce environments where traditional machine learning approaches fail. 

**Abstract (ZH)**: 水下3D物体检测仍然是计算机视觉中最具挑战性的前沿领域之一，其中传统方法在恶劣的声学环境中和训练数据稀缺的情况下难以应对。尽管深度学习已 revolutionized 地面3D检测，但其在水下应用面临的关键瓶颈在于：获取足够的标注声纳数据极其昂贵且操作复杂，往往需要专门的船只、专家测绘和理想的天气条件。本研究探讨了一个基本问题：我们能否在没有真实世界训练数据的情况下实现可靠的水下3D物体检测？我们通过开发和比较两种无监督检测人工结构的框架来应对这一挑战，在多波束回声测深点云中。我们的双管齐下方法结合了一个基于物理的声纳仿真管道，用于为最新神经网络生成合成训练数据，以及一个利用目标物体几何先验的鲁棒模型导向模板匹配系统。在波罗的海实际水深调查中的评估揭示了令人惊讶的见解：虽然在合成数据上训练的神经网络在仿真场景中达到98%的平均精确度（mAP），但在实际声纳数据中降至40%的mAP，原因是领域转移。相反，我们的模板匹配方法在无需任何训练的情况下在实际数据中保持83%的mAP，显示出对声学噪声和环境变化的非凡鲁棒性。本研究挑战了水下领域数据饥渴型深度学习的 convention，确立了第一个大规模的无监督水下3D检测基准。这项工作为在数据稀缺环境中水下自主航行器导航、水下考古学和海上基础设施监测开辟了新可能性，这些环境是传统机器学习方法无法胜任的。 

---
# VibeVoice Technical Report 

**Title (ZH)**: VibeVoice 技术报告 

**Authors**: Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, Shaohan Huang, Yan Xia, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.19205)  

**Abstract**: This report presents VibeVoice, a novel model designed to synthesize long-form speech with multiple speakers by employing next-token diffusion, which is a unified method for modeling continuous data by autoregressively generating latent vectors via diffusion. To enable this, we introduce a novel continuous speech tokenizer that, when compared to the popular Encodec model, improves data compression by 80 times while maintaining comparable performance. The tokenizer effectively preserves audio fidelity while significantly boosting computational efficiency for processing long sequences. Thus, VibeVoice can synthesize long-form speech for up to 90 minutes (in a 64K context window length) with a maximum of 4 speakers, capturing the authentic conversational ``vibe'' and surpassing open-source and proprietary dialogue models. 

**Abstract (ZH)**: 本报告介绍了VibeVoice，这是一种新型模型，通过运用下一词扩散技术合成长篇多说话者语音，该技术是一种通过自回归生成潜变量来 Modeling 统一方法连续数据的方法。为此，我们引入了一种新颖的连续语音分词器，与流行的Encodec模型相比，分词器在保持相当性能的同时，提高了数据压缩率80倍。分词器在有效保持音频保真度的同时，显著提升了处理长序列的计算效率。因此，VibeVoice 可以在64K上下文窗口长度的情况下合成长达90分钟的语音（最多4位说话者），并捕捉到真实的对话“氛围”，超越了开源和专有对话模型。 

---
# LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding 

**Title (ZH)**: LSD-3D：基于几何约束的大规模3D驾驶场景生成 

**Authors**: Julian Ost, Andrea Ramazzina, Amogh Joshi, Maximilian Bömer, Mario Bijelic, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2508.19204)  

**Abstract**: Large-scale scene data is essential for training and testing in robot learning. Neural reconstruction methods have promised the capability of reconstructing large physically-grounded outdoor scenes from captured sensor data. However, these methods have baked-in static environments and only allow for limited scene control -- they are functionally constrained in scene and trajectory diversity by the captures from which they are reconstructed. In contrast, generating driving data with recent image or video diffusion models offers control, however, at the cost of geometry grounding and causality. In this work, we aim to bridge this gap and present a method that directly generates large-scale 3D driving scenes with accurate geometry, allowing for causal novel view synthesis with object permanence and explicit 3D geometry estimation. The proposed method combines the generation of a proxy geometry and environment representation with score distillation from learned 2D image priors. We find that this approach allows for high controllability, enabling the prompt-guided geometry and high-fidelity texture and structure that can be conditioned on map layouts -- producing realistic and geometrically consistent 3D generations of complex driving scenes. 

**Abstract (ZH)**: 大规模场景数据对于机器人学习中的训练和测试至关重要。尽管神经重建方法承诺能够从捕获的传感器数据中重建大规模的物理 grounding 户外场景，但这些方法存在固化的静态环境限制，并且只能实现有限的场景控制——它们的功能受制于重建它们的捕获数据。相比之下，利用 recent 图像或视频扩散模型生成驾驶数据提供了控制能力，但代价是几何 grounding 和因果性的损失。本文旨在弥合这一差距，并提出了一种直接生成具有准确几何结构的大规模 3D 驱动场景的方法，该方法允许因果新型视图合成，并明确估计 3D 几何结构。所提方法将代理几何和环境表示的生成与从学习到的 2D 图像先验中提取的分数蒸馏相结合。我们发现，这种方法允许高度可控性，使得通过提示引导几何结构和高保真度的纹理与结构可以根据地图布局进行条件化——产生真实的、几何上一致的复杂驾驶场景的 3D 生成。 

---
# Few-Shot Connectivity-Aware Text Line Segmentation in Historical Documents 

**Title (ZH)**: 面向历史文档的少样本连接意识文本行分割 

**Authors**: Rafael Sterzinger, Tingyu Lin, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2508.19162)  

**Abstract**: A foundational task for the digital analysis of documents is text line segmentation. However, automating this process with deep learning models is challenging because it requires large, annotated datasets that are often unavailable for historical documents. Additionally, the annotation process is a labor- and cost-intensive task that requires expert knowledge, which makes few-shot learning a promising direction for reducing data requirements. In this work, we demonstrate that small and simple architectures, coupled with a topology-aware loss function, are more accurate and data-efficient than more complex alternatives. We pair a lightweight UNet++ with a connectivity-aware loss, initially developed for neuron morphology, which explicitly penalizes structural errors like line fragmentation and unintended line merges. To increase our limited data, we train on small patches extracted from a mere three annotated pages per manuscript. Our methodology significantly improves upon the current state-of-the-art on the U-DIADS-TL dataset, with a 200% increase in Recognition Accuracy and a 75% increase in Line Intersection over Union. Our method also achieves an F-Measure score on par with or even exceeding that of the competition winner of the DIVA-HisDB baseline detection task, all while requiring only three annotated pages, exemplifying the efficacy of our approach. Our implementation is publicly available at: this https URL. 

**Abstract (ZH)**: 数字文档分析中的基础任务是文本行分割。然而，利用深度学习模型自动化这一过程具有挑战性，因为这需要大量标注数据，而历史文档中这类数据通常不可用。此外，标注过程是劳动和成本密集型的，需要专家知识，这使得少量样本学习成为减少数据需求的有前途的方向。在此项工作中，我们证明了小型且简单的架构与拓扑感知损失函数相结合比更复杂的替代方案更准确、更数据高效。我们使用轻量级的UNet++并搭配一种感知连接性的损失函数，该损失函数最初是为神经形态学开发的，它明确地惩罚像行分割和意外行合并这样的结构错误。为增加有限的数据，我们从每部手稿仅三个标注页面的小尺寸片段中进行训练。我们的方法在U-DIADS-TL数据集上显著改进了当前的最先进水平，识字准确率提高了200%，行交并比提高了75%。我们的方法还实现了与或超过DIVA-HisDB基线检测任务竞赛获胜者F-度量评分，同时仅需三个标注页面，证明了我们方法的有效性。我们的实现可以在以下地址公开获得：this https URL。 

---
# RDDM: Practicing RAW Domain Diffusion Model for Real-world Image Restoration 

**Title (ZH)**: RDDM: 实践RAW域扩散模型以实现现实世界图像修复 

**Authors**: Yan Chen, Yi Wen, Wei Li, Junchao Liu, Yong Guo, Jie Hu, Xinghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.19154)  

**Abstract**: We present the RAW domain diffusion model (RDDM), an end-to-end diffusion model that restores photo-realistic images directly from the sensor RAW data. While recent sRGB-domain diffusion methods achieve impressive results, they are caught in a dilemma between high fidelity and realistic generation. As these models process lossy sRGB inputs and neglect the accessibility of the sensor RAW images in many scenarios, e.g., in image and video capturing in edge devices, resulting in sub-optimal performance. RDDM bypasses this limitation by directly restoring images in the RAW domain, replacing the conventional two-stage image signal processing (ISP) + IR pipeline. However, a simple adaptation of pre-trained diffusion models to the RAW domain confronts the out-of-distribution (OOD) issues. To this end, we propose: (1) a RAW-domain VAE (RVAE) learning optimal latent representations, (2) a differentiable Post Tone Processing (PTP) module enabling joint RAW and sRGB space optimization. To compensate for the deficiency in the dataset, we develop a scalable degradation pipeline synthesizing RAW LQ-HQ pairs from existing sRGB datasets for large-scale training. Furthermore, we devise a configurable multi-bayer (CMB) LoRA module handling diverse RAW patterns such as RGGB, BGGR, etc. Extensive experiments demonstrate RDDM's superiority over state-of-the-art sRGB diffusion methods, yielding higher fidelity results with fewer artifacts. 

**Abstract (ZH)**: 我们提出了RAW域扩散模型（RDDM），这是一种端到端的扩散模型，可以直接从传感器RAW数据恢复出照片真实的图像。虽然最近的sRGB域扩散方法取得了令人 impressive 的成果，但在高保真度和现实生成之间陷入了困境。由于这些模型处理的是失真的sRGB输入，并且忽视了许多场景下传感器RAW图像的可用性，例如在边缘设备中的图像和视频捕捉，导致了次优的性能。RDDM通过直接在RAW域恢复图像跳过了这一限制，替代了传统的ISP+IR两阶段图像信号处理管道。然而，将预训练的扩散模型简单地适应到RAW域面临着分布外（OOD）问题。为此，我们提出了：(1) 一种RAW域VAE（RVAE）学习最优的潜在表示，(2) 一个可微后调色处理（PTP）模块，使得RAW和sRGB空间可以联合优化。为弥补数据集的不足，我们开发了一种可扩展的降质流水线，从现有的sRGB数据集中合成RAW低质-高质（LQ-HQ）对，以支持大规模训练。此外，我们设计了一种可配置的多层板（CMB）LoRA模块，以处理各种RAW模式，如RGGB、BGGR等。广泛的实验表明，RDDM在与最先进的sRGB扩散方法的对比中具有优越性，能够以较少的伪影获得更高的保真度结果。 

---
# No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes 

**Title (ZH)**: 无标签被遗漏：面向所有监督制度的统一表面缺陷检测模型 

**Authors**: Blaž Rolih, Matic Fučka, Danijel Skočaj  

**Link**: [PDF](https://arxiv.org/pdf/2508.19060)  

**Abstract**: Surface defect detection is a critical task across numerous industries, aimed at efficiently identifying and localising imperfections or irregularities on manufactured components. While numerous methods have been proposed, many fail to meet industrial demands for high performance, efficiency, and adaptability. Existing approaches are often constrained to specific supervision scenarios and struggle to adapt to the diverse data annotations encountered in real-world manufacturing processes, such as unsupervised, weakly supervised, mixed supervision, and fully supervised settings. To address these challenges, we propose SuperSimpleNet, a highly efficient and adaptable discriminative model built on the foundation of SimpleNet. SuperSimpleNet incorporates a novel synthetic anomaly generation process, an enhanced classification head, and an improved learning procedure, enabling efficient training in all four supervision scenarios, making it the first model capable of fully leveraging all available data annotations. SuperSimpleNet sets a new standard for performance across all scenarios, as demonstrated by its results on four challenging benchmark datasets. Beyond accuracy, it is very fast, achieving an inference time below 10 ms. With its ability to unify diverse supervision paradigms while maintaining outstanding speed and reliability, SuperSimpleNet represents a promising step forward in addressing real-world manufacturing challenges and bridging the gap between academic research and industrial applications. Code: this https URL 

**Abstract (ZH)**: 表面缺陷检测是众多行业中的一项关键任务，旨在高效地识别和定位制造部件上的瑕疵或不规则性。尽管提出了许多方法，但许多方法无法满足工业对高性能、高效性和适应性的需求。现有方法往往受限于特定的监督场景，并且难以适应现实制造过程中遇到的各种数据标注，如无监督、弱监督、混合监督和全监督设置。为解决这些挑战，我们提出了SuperSimpleNet，这是一种基于SimpleNet构建的高度高效和适应性强的判别模型。SuperSimpleNet结合了新颖的合成异常生成过程、增强的分类头以及改进的学习过程，使其能够在所有四种监督场景下实现高效训练，成为首个能够充分利用所有可用数据标注的模型。SuperSimpleNet在所有场景中均设定了新的性能标准，其结果在四个具有挑战性的基准数据集上得到了验证。除了准确性，它是非常快速的，推理时间低于10毫秒。凭借其统一各种监督范式的能力，同时保持出色的效率和可靠性，SuperSimpleNet代表了在解决现实制造挑战和弥合学术研究与工业应用之间差距方面的一个有前景的进步。代码：this https URL。 

---
# RoofSeg: An edge-aware transformer-based network for end-to-end roof plane segmentation 

**Title (ZH)**: RoofSeg：一种边缘意识的基于变换器的端到端屋面平面分割网络 

**Authors**: Siyuan You, Guozheng Xu, Pengwei Zhou, Qiwen Jin, Jian Yao, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19003)  

**Abstract**: Roof plane segmentation is one of the key procedures for reconstructing three-dimensional (3D) building models at levels of detail (LoD) 2 and 3 from airborne light detection and ranging (LiDAR) point clouds. The majority of current approaches for roof plane segmentation rely on the manually designed or learned features followed by some specifically designed geometric clustering strategies. Because the learned features are more powerful than the manually designed features, the deep learning-based approaches usually perform better than the traditional approaches. However, the current deep learning-based approaches have three unsolved problems. The first is that most of them are not truly end-to-end, the plane segmentation results may be not optimal. The second is that the point feature discriminability near the edges is relatively low, leading to inaccurate planar edges. The third is that the planar geometric characteristics are not sufficiently considered to constrain the network training. To solve these issues, a novel edge-aware transformer-based network, named RoofSeg, is developed for segmenting roof planes from LiDAR point clouds in a truly end-to-end manner. In the RoofSeg, we leverage a transformer encoder-decoder-based framework to hierarchically predict the plane instance masks with the use of a set of learnable plane queries. To further improve the segmentation accuracy of edge regions, we also design an Edge-Aware Mask Module (EAMM) that sufficiently incorporates planar geometric prior of edges to enhance its discriminability for plane instance mask refinement. In addition, we propose an adaptive weighting strategy in the mask loss to reduce the influence of misclassified points, and also propose a new plane geometric loss to constrain the network training. 

**Abstract (ZH)**: 基于变压器的边缘感知屋顶面分割网络：RoofSeg 

---
# HOTSPOT-YOLO: A Lightweight Deep Learning Attention-Driven Model for Detecting Thermal Anomalies in Drone-Based Solar Photovoltaic Inspections 

**Title (ZH)**: HOTSPOT-YOLO：一种基于无人机光伏检测的轻量级深度学习注意力驱动模型 

**Authors**: Mahmoud Dhimish  

**Link**: [PDF](https://arxiv.org/pdf/2508.18912)  

**Abstract**: Thermal anomaly detection in solar photovoltaic (PV) systems is essential for ensuring operational efficiency and reducing maintenance costs. In this study, we developed and named HOTSPOT-YOLO, a lightweight artificial intelligence (AI) model that integrates an efficient convolutional neural network backbone and attention mechanisms to improve object detection. This model is specifically designed for drone-based thermal inspections of PV systems, addressing the unique challenges of detecting small and subtle thermal anomalies, such as hotspots and defective modules, while maintaining real-time performance. Experimental results demonstrate a mean average precision of 90.8%, reflecting a significant improvement over baseline object detection models. With a reduced computational load and robustness under diverse environmental conditions, HOTSPOT-YOLO offers a scalable and reliable solution for large-scale PV inspections. This work highlights the integration of advanced AI techniques with practical engineering applications, revolutionizing automated fault detection in renewable energy systems. 

**Abstract (ZH)**: 太阳能光伏（PV）系统的热异常检测对于确保运行效率和降低维护成本至关重要。本研究开发了HOTSPOT-YOLO，这是一种轻量级的人工智能模型，结合了高效卷积神经网络骨干和注意力机制以改进目标检测。该模型专门设计用于无人机对PV系统的热检查，以应对检测小而微妙的热异常（如热点和损坏模块）的独特挑战，同时保持实时性能。实验结果表明，平均精确度为90.8%，显著优于基准目标检测模型。HOTSPOT-YOLO具有降低的计算负载和在不同环境条件下的鲁棒性，为大规模PV检查提供可扩展且可靠的解决方案。本研究强调了将先进技术与工程应用相结合的重要性，革新了可再生能源系统自动故障检测的方法。 

---
# Clustering-based Feature Representation Learning for Oracle Bone Inscriptions Detection 

**Title (ZH)**: 基于聚类的特征表示学习在甲骨文检测中的应用 

**Authors**: Ye Tao, Xinran Fu, Honglin Pang, Xi Yang, Chuntao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18641)  

**Abstract**: Oracle Bone Inscriptions (OBIs), play a crucial role in understanding ancient Chinese civilization. The automated detection of OBIs from rubbing images represents a fundamental yet challenging task in digital archaeology, primarily due to various degradation factors including noise and cracks that limit the effectiveness of conventional detection networks. To address these challenges, we propose a novel clustering-based feature space representation learning method. Our approach uniquely leverages the Oracle Bones Character (OBC) font library dataset as prior knowledge to enhance feature extraction in the detection network through clustering-based representation learning. The method incorporates a specialized loss function derived from clustering results to optimize feature representation, which is then integrated into the total network loss. We validate the effectiveness of our method by conducting experiments on two OBIs detection dataset using three mainstream detection frameworks: Faster R-CNN, DETR, and Sparse R-CNN. Through extensive experimentation, all frameworks demonstrate significant performance improvements. 

**Abstract (ZH)**: 甲骨文（OBIs）在理解古代中国文化中发挥着关键作用。从拓片图像中自动检测甲骨文代表了数字考古学中一个基础但具挑战性的任务，主要由于包括噪声和裂痕在内的各种退化因素限制了传统检测网络的有效性。为了解决这些挑战，我们提出了一种新颖的基于聚类的特征空间表示学习方法。该方法通过基于聚类的表示学习充分利用了甲骨文字符（OBC）字体库数据集的先验知识，以增强检测网络中的特征提取。该方法结合了源自聚类结果的特殊损失函数来优化特征表示，该表示随后被集成到总网络损失中。通过在两个甲骨文检测数据集上使用三种主流检测框架（Faster R-CNN、DETR和Sparse R-CNN）进行实验验证了该方法的有效性。广泛实验表明，所有框架均实现了显著的性能提升。 

---
# ROSE: Remove Objects with Side Effects in Videos 

**Title (ZH)**: ROSE: 删除视频中具有副作用的对象 

**Authors**: Chenxuan Miao, Yutong Feng, Jianshu Zeng, Zixiang Gao, Hantang Liu, Yunfeng Yan, Donglian Qi, Xi Chen, Bin Wang, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18633)  

**Abstract**: Video object removal has achieved advanced performance due to the recent success of video generative models. However, when addressing the side effects of objects, e.g., their shadows and reflections, existing works struggle to eliminate these effects for the scarcity of paired video data as supervision. This paper presents ROSE, termed Remove Objects with Side Effects, a framework that systematically studies the object's effects on environment, which can be categorized into five common cases: shadows, reflections, light, translucency and mirror. Given the challenges of curating paired videos exhibiting the aforementioned effects, we leverage a 3D rendering engine for synthetic data generation. We carefully construct a fully-automatic pipeline for data preparation, which simulates a large-scale paired dataset with diverse scenes, objects, shooting angles, and camera trajectories. ROSE is implemented as an video inpainting model built on diffusion transformer. To localize all object-correlated areas, the entire video is fed into the model for reference-based erasing. Moreover, additional supervision is introduced to explicitly predict the areas affected by side effects, which can be revealed through the differential mask between the paired videos. To fully investigate the model performance on various side effect removal, we presents a new benchmark, dubbed ROSE-Bench, incorporating both common scenarios and the five special side effects for comprehensive evaluation. Experimental results demonstrate that ROSE achieves superior performance compared to existing video object erasing models and generalizes well to real-world video scenarios. The project page is this https URL. 

**Abstract (ZH)**: 去除具有副作用的对象以实现视频中对象的高级移除：ROSE，一种系统研究对象对环境影响的框架 

---
# SAT-SKYLINES: 3D Building Generation from Satellite Imagery and Coarse Geometric Priors 

**Title (ZH)**: 基于卫星图像和粗略几何先验的3D建筑生成 

**Authors**: Zhangyu Jin, Andrew Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18531)  

**Abstract**: We present SatSkylines, a 3D building generation approach that takes satellite imagery and coarse geometric priors. Without proper geometric guidance, existing image-based 3D generation methods struggle to recover accurate building structures from the top-down views of satellite images alone. On the other hand, 3D detailization methods tend to rely heavily on highly detailed voxel inputs and fail to produce satisfying results from simple priors such as cuboids. To address these issues, our key idea is to model the transformation from interpolated noisy coarse priors to detailed geometries, enabling flexible geometric control without additional computational cost. We have further developed Skylines-50K, a large-scale dataset of over 50,000 unique and stylized 3D building assets in order to support the generations of detailed building models. Extensive evaluations indicate the effectiveness of our model and strong generalization ability. 

**Abstract (ZH)**: SatSkylines：一种基于卫星图像和粗略几何先验的3D建筑生成方法 

---
# A Deep Learning Application for Psoriasis Detection 

**Title (ZH)**: 基于深度学习的银屑病检测应用 

**Authors**: Anna Milani, Fábio S. da Silva, Elloá B. Guedes, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2508.18528)  

**Abstract**: In this paper a comparative study of the performance of three Convolutional Neural Network models, ResNet50, Inception v3 and VGG19 for classification of skin images with lesions affected by psoriasis is presented. The images used for training and validation of the models were obtained from specialized platforms. Some techniques were used to adjust the evaluation metrics of the neural networks. The results found suggest the model Inception v3 as a valuable tool for supporting the diagnosis of psoriasis. This is due to its satisfactory performance with respect to accuracy and F1-Score (97.5% ${\pm}$ 0.2). 

**Abstract (ZH)**: 本文呈现了对用于斑块影响的皮肤图像分类的三种卷积神经网络模型——ResNet50、Inception v3和VGG19——性能的比较研究。研究表明，Inception v3模型因其在准确率和F1分数（97.5% ± 0.2）方面的满意表现，是支持银屑病诊断的一个有价值的工具。 

---
# Automated Landfill Detection Using Deep Learning: A Comparative Study of Lightweight and Custom Architectures with the AerialWaste Dataset 

**Title (ZH)**: 基于深度学习的垃圾填埋自动检测：AerialWaste数据集中轻量级和定制架构的对比研究 

**Authors**: Nowshin Sharmily, Rusab Sarmun, Muhammad E. H. Chowdhury, Mir Hamidul Hussain, Saad Bin Abul Kashem, Molla E Majid, Amith Khandakar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18315)  

**Abstract**: Illegal landfills are posing as a hazardous threat to people all over the world. Due to the arduous nature of manually identifying the location of landfill, many landfills go unnoticed by authorities and later cause dangerous harm to people and environment. Deep learning can play a significant role in identifying these landfills while saving valuable time, manpower and resources. Despite being a burning concern, good quality publicly released datasets for illegal landfill detection are hard to find due to security concerns. However, AerialWaste Dataset is a large collection of 10434 images of Lombardy region of Italy. The images are of varying qualities, collected from three different sources: AGEA Orthophotos, WorldView-3, and Google Earth. The dataset contains professionally curated, diverse and high-quality images which makes it particularly suitable for scalable and impactful research. As we trained several models to compare results, we found complex and heavy models to be prone to overfitting and memorizing training data instead of learning patterns. Therefore, we chose lightweight simpler models which could leverage general features from the dataset. In this study, Mobilenetv2, Googlenet, Densenet, MobileVit and other lightweight deep learning models were used to train and validate the dataset as they achieved significant success with less overfitting. As we saw substantial improvement in the performance using some of these models, we combined the best performing models and came up with an ensemble model. With the help of ensemble and fusion technique, binary classification could be performed on this dataset with 92.33% accuracy, 92.67% precision, 92.33% sensitivity, 92.41% F1 score and 92.71% specificity. 

**Abstract (ZH)**: 非法填埋场对全球人口构成严重威胁。由于人工识别填埋场位置的艰巨性，许多填埋场未被当局发现，进而对人员和环境造成危险损害。深度学习在识别这些填埋场方面可以发挥重要作用，同时节省宝贵的时间、人力和资源。尽管这是一个紧迫的问题，但由于安全原因，高质量的公开数据集很难找到用于非法填埋场检测。然而，AerialWaste数据集是意大利伦巴第地区10434张图像的大规模集合。这些图像来自三个不同的来源：AGEA正射影像、WorldView-3和Google Earth，并且包含专业的、多样化的和高质量的图像，特别适合进行规模化和有影响力的研究所用。在训练多种模型以进行比较后，我们发现复杂的和重型模型容易过拟合并记住训练数据而非学习模式。因此，我们选择了轻量级的简单模型，这些模型可以从数据集中利用通用特征。在这项研究中，我们使用了Mobilenetv2、Googlenet、Densenet、MobileVit以及其他轻量级的深度学习模型进行数据集的训练和验证，因为这些模型在减少过拟合的情况下取得了显著的成效。由于我们在这些模型中的一些模型上看到了显著性能提升，我们结合了表现最佳的模型并提出了一个集成模型。借助集成和融合技术，该数据集在二元分类上的准确率为92.33%，精确率为92.67%，敏感性为92.33%，F1分数为92.41%，特异性为92.71%。 

---
# MobileDenseAttn:A Dual-Stream Architecture for Accurate and Interpretable Brain Tumor Detection 

**Title (ZH)**: MobileDenseAttn：一种用于准确可解释脑肿瘤检测的双流架构 

**Authors**: Shudipta Banik, Muna Das, Trapa Banik, Md. Ehsanul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2508.18294)  

**Abstract**: The detection of brain tumor in MRI is an important aspect of ensuring timely diagnostics and treatment; however, manual analysis is commonly long and error-prone. Current approaches are not universal because they have limited generalization to heterogeneous tumors, are computationally inefficient, are not interpretable, and lack transparency, thus limiting trustworthiness. To overcome these issues, we introduce MobileDenseAttn, a fusion model of dual streams of MobileNetV2 and DenseNet201 that can help gradually improve the feature representation scale, computing efficiency, and visual explanations via GradCAM. Our model uses feature level fusion and is trained on an augmented dataset of 6,020 MRI scans representing glioma, meningioma, pituitary tumors, and normal samples. Measured under strict 5-fold cross-validation protocols, MobileDenseAttn provides a training accuracy of 99.75%, a testing accuracy of 98.35%, and a stable F1 score of 0.9835 (95% CI: 0.9743 to 0.9920). The extensive validation shows the stability of the model, and the comparative analysis proves that it is a great advancement over the baseline models (VGG19, DenseNet201, MobileNetV2) with a +3.67% accuracy increase and a 39.3% decrease in training time compared to VGG19. The GradCAM heatmaps clearly show tumor-affected areas, offering clinically significant localization and improving interpretability. These findings position MobileDenseAttn as an efficient, high performance, interpretable model with a high probability of becoming a clinically practical tool in identifying brain tumors in the real world. 

**Abstract (ZH)**: 基于MobileNetV2和DenseNet201的MobileDenseAttn在MRI脑肿瘤检测中的应用研究 

---
