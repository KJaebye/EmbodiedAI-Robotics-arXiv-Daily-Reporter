# VTire: A Bimodal Visuotactile Tire with High-Resolution Sensing Capability 

**Title (ZH)**: VTire：一种高分辨率感测能力的双模态视触轮胎 

**Authors**: Shoujie Li, Jianle Xu, Tong Wu, Yang Yang, Yanbo Chen, Xueqian Wang, Wenbo Ding, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19194)  

**Abstract**: Developing smart tires with high sensing capability is significant for improving the moving stability and environmental adaptability of wheeled robots and vehicles. However, due to the classical manufacturing design, it is always challenging for tires to infer external information precisely. To this end, this paper introduces a bimodal sensing tire, which can simultaneously capture tactile and visual data. By leveraging the emerging visuotactile techniques, the proposed smart tire can realize various functions, including terrain recognition, ground crack detection, load sensing, and tire damage detection. Besides, we optimize the material and structure of the tire to ensure its outstanding elasticity, toughness, hardness, and transparency. In terms of algorithms, a transformer-based multimodal classification algorithm, a load detection method based on finite element analysis, and a contact segmentation algorithm have been developed. Furthermore, we construct an intelligent mobile platform to validate the system's effectiveness and develop visual and tactile datasets in complex terrains. The experimental results show that our multimodal terrain sensing algorithm can achieve a classification accuracy of 99.2\%, a tire damage detection accuracy of 97\%, a 98\% success rate in object search, and the ability to withstand tire loading weights exceeding 35 kg. In addition, we open-source our algorithms, hardware, and datasets at this https URL. 

**Abstract (ZH)**: 开发高感知能力的智能轮胎对于提高轮式机器人和车辆的移动稳定性和环境适应性具有重要意义。然而，由于传统的制造设计，轮胎精确推断外部信息一直颇具挑战。为此，本文介绍了一种双模态感知轮胎，可以同时获取触觉和视觉数据。通过利用新兴的视触觉技术，所提出的智能轮胎可以实现地形识别、地面裂缝检测、负载感测和轮胎损伤检测等多种功能。此外，我们优化了轮胎的材料和结构，确保其具备出色的弹性、韧性、硬度和透明度。在算法方面，开发了一种基于变换器的多模态分类算法、基于有限元分析的负载检测方法和接触分割算法。我们还构建了一个智能移动平台来验证系统的有效性，并在复杂地形中开发了视觉和触觉数据集。实验结果表明，我们的多模态地形感知算法可以达到99.2%的分类准确率、97%的轮胎损伤检测准确率、98%的目标搜索成功率，并能够承受超过35公斤的轮胎负载。此外，我们在此httpsURL开放了我们的算法、硬件和数据集。 

---
# M2R2: MulitModal Robotic Representation for Temporal Action Segmentation 

**Title (ZH)**: M2R2: 多模态机器人表示用于时间动作分割 

**Authors**: Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.18662)  

**Abstract**: Temporal action segmentation (TAS) has long been a key area of research in both robotics and computer vision. In robotics, algorithms have primarily focused on leveraging proprioceptive information to determine skill boundaries, with recent approaches in surgical robotics incorporating vision. In contrast, computer vision typically relies on exteroceptive sensors, such as cameras. Existing multimodal TAS models in robotics integrate feature fusion within the model, making it difficult to reuse learned features across different models. Meanwhile, pretrained vision-only feature extractors commonly used in computer vision struggle in scenarios with limited object visibility. In this work, we address these challenges by proposing M2R2, a multimodal feature extractor tailored for TAS, which combines information from both proprioceptive and exteroceptive sensors. We introduce a novel pretraining strategy that enables the reuse of learned features across multiple TAS models. Our method achieves state-of-the-art performance on the REASSEMBLE dataset, a challenging multimodal robotic assembly dataset, outperforming existing robotic action segmentation models by 46.6%. Additionally, we conduct an extensive ablation study to evaluate the contribution of different modalities in robotic TAS tasks. 

**Abstract (ZH)**: 多模态动作分割（TAS）一直是机器人技术和计算机视觉研究中的关键领域。在机器人技术中，算法主要侧重于利用本体感受信息来确定技能边界，近期的手术机器人研究则开始结合视觉信息。相比之下，计算机视觉通常依赖于外部传感器，如摄像头。现有的机器人多模态TAS模型在模型内部进行特征融合，使得跨不同模型重用学习到的特征变得困难。同时，计算机视觉中常用的预训练视觉特征提取器在物体视线受限的场景中表现不佳。在这项工作中，我们通过提出一种针对TAS的多模态特征提取器M2R2来应对这些挑战，该提取器结合了本体感受和外部传感器信息。我们介绍了一种新的预训练策略，使得学习到的特征在多个TAS模型中能够重用。我们的方法在REASSEMBLE数据集上取得了最先进的性能，该数据集是一个具有挑战性的多模态机器人装配数据集，性能超越现有机器人动作分割模型46.6%。此外，我们还进行了广泛的消融研究，以评估不同模态在机器人TAS任务中的贡献。 

---
# DriVerse: Navigation World Model for Driving Simulation via Multimodal Trajectory Prompting and Motion Alignment 

**Title (ZH)**: DriVerse: 驾驶模拟的多模态轨迹提示与运动对齐导航世界模型 

**Authors**: Xiaofan Li, Chenming Wu, Zhao Yang, Zhihao Xu, Dingkang Liang, Yumeng Zhang, Ji Wan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18576)  

**Abstract**: This paper presents DriVerse, a generative model for simulating navigation-driven driving scenes from a single image and a future trajectory. Previous autonomous driving world models either directly feed the trajectory or discrete control signals into the generation pipeline, leading to poor alignment between the control inputs and the implicit features of the 2D base generative model, which results in low-fidelity video outputs. Some methods use coarse textual commands or discrete vehicle control signals, which lack the precision to guide fine-grained, trajectory-specific video generation, making them unsuitable for evaluating actual autonomous driving algorithms. DriVerse introduces explicit trajectory guidance in two complementary forms: it tokenizes trajectories into textual prompts using a predefined trend vocabulary for seamless language integration, and converts 3D trajectories into 2D spatial motion priors to enhance control over static content within the driving scene. To better handle dynamic objects, we further introduce a lightweight motion alignment module, which focuses on the inter-frame consistency of dynamic pixels, significantly enhancing the temporal coherence of moving elements over long sequences. With minimal training and no need for additional data, DriVerse outperforms specialized models on future video generation tasks across both the nuScenes and Waymo datasets. The code and models will be released to the public. 

**Abstract (ZH)**: 本文介绍了DriVerse，一种从单张图像和未来轨迹生成由导航驱动的驾驶场景的生成模型。之前的自主驾驶世界模型要么直接将轨迹或离散控制信号输入生成管道，要么使用粗糙的文本命令或离散的车辆控制信号，这导致控制输入与2D基础生成模型的隐式特征对齐较差，从而产生低保真度的视频输出。一些方法使用粗略的文本命令或离散的车辆控制信号，缺乏对细粒度、轨迹特定视频生成的精度指导，使其不适合评估实际的自主驾驶算法。DriVerse通过两种互补的方式引入了显式的轨迹指导：使用预定义的趋势词汇对轨迹进行标记化，以便无缝的语言整合，并将3D轨迹转换为2D空间运动先验，以增强对驾驶场景中静态内容的控制。为了更好地处理动态对象，我们进一步引入了一个轻量级的运动对齐模块，该模块专注于帧间动态像素的连续性，显著增强了长时间序列中移动元素的时序一致性。DriVerse在nuScenes和Waymo数据集上的未来视频生成任务中表现出色，训练量小且无需额外数据，代码和模型将对公众开放。 

---
# Towards AI-Driven Policing: Interdisciplinary Knowledge Discovery from Police Body-Worn Camera Footage 

**Title (ZH)**: 面向人工智能驱动的警务：从警察执法记录仪视频中跨学科知识发现 

**Authors**: Anita Srbinovska, Angela Srbinovska, Vivek Senthil, Adrian Martin, John McCluskey, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2504.20007)  

**Abstract**: This paper proposes a novel interdisciplinary framework for analyzing police body-worn camera (BWC) footage from the Rochester Police Department (RPD) using advanced artificial intelligence (AI) and statistical machine learning (ML) techniques. Our goal is to detect, classify, and analyze patterns of interaction between police officers and civilians to identify key behavioral dynamics, such as respect, disrespect, escalation, and de-escalation. We apply multimodal data analysis by integrating video, audio, and natural language processing (NLP) techniques to extract meaningful insights from BWC footage. We present our methodology, computational techniques, and findings, outlining a practical approach for law enforcement while advancing the frontiers of knowledge discovery from police BWC data. 

**Abstract (ZH)**: 本文提出了一种新的跨学科框架，利用先进的人工智能（AI）和统计机器学习（ML）技术分析罗彻斯特警察部门（RPD）的警察佩戴摄像机（BWC）录像，旨在检测、分类和分析警察与平民互动的模式，识别关键行为动态，如尊重、不尊重、升级和降级。我们通过将视频、音频和自然语言处理（NLP）技术融为一体，进行多模态数据分析，从中提取有意义的见解。我们介绍了我们的方法论、计算技术及研究成果，提出了适用于执法机构的实际方法，并推进了从警察BWC数据中发现知识前沿。 

---
# Enhancing Surgical Documentation through Multimodal Visual-Temporal Transformers and Generative AI 

**Title (ZH)**: 通过多模态视觉-时间变换器和生成式AI增强手术记录 

**Authors**: Hugo Georgenthum, Cristian Cosentino, Fabrizio Marozzo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.19918)  

**Abstract**: The automatic summarization of surgical videos is essential for enhancing procedural documentation, supporting surgical training, and facilitating post-operative analysis. This paper presents a novel method at the intersection of artificial intelligence and medicine, aiming to develop machine learning models with direct real-world applications in surgical contexts. We propose a multi-modal framework that leverages recent advancements in computer vision and large language models to generate comprehensive video summaries. %
The approach is structured in three key stages. First, surgical videos are divided into clips, and visual features are extracted at the frame level using visual transformers. This step focuses on detecting tools, tissues, organs, and surgical actions. Second, the extracted features are transformed into frame-level captions via large language models. These are then combined with temporal features, captured using a ViViT-based encoder, to produce clip-level summaries that reflect the broader context of each video segment. Finally, the clip-level descriptions are aggregated into a full surgical report using a dedicated LLM tailored for the summarization task. %
We evaluate our method on the CholecT50 dataset, using instrument and action annotations from 50 laparoscopic videos. The results show strong performance, achieving 96\% precision in tool detection and a BERT score of 0.74 for temporal context summarization. This work contributes to the advancement of AI-assisted tools for surgical reporting, offering a step toward more intelligent and reliable clinical documentation. 

**Abstract (ZH)**: 自动手术视频摘要对于提高手术文档质量、支持手术培训以及促进术后分析至关重要。本文提出了一种结合人工智能和医学的创新方法，旨在开发直接应用于手术环境的机器学习模型。我们提出了一种多模态框架，利用计算机视觉和大型语言模型的最新进展生成全面的视频摘要。%
该方法分为三个关键阶段。首先，将手术视频分为片段，并使用视觉变换器在帧级提取视觉特征。此步骤专注于检测工具、组织、器官和手术操作。其次，提取的特征通过大型语言模型转换为帧级描述。这些描述与使用ViViT基编码器捕获的时间特征相结合，生成反映每个视频片段广泛背景的片段级摘要。最后，使用专为摘要任务设计的LLM将片段级描述汇总为完整的手术报告。%
我们在CholecT50数据集上评估了我们的方法，使用了50个腹腔镜视频的仪器和动作注释。结果表明，该方法具有很强的表现力，在工具检测上达到了96%的准确率，并且在时间上下文摘要中获得了0.74的BERT分数。这项工作为AI辅助的手术报告工具的进步做出了贡献，朝着更智能和可靠的临床文档方向迈出了一步。 

---
# CLIP-KOA: Enhancing Knee Osteoarthritis Diagnosis with Multi-Modal Learning and Symmetry-Aware Loss Functions 

**Title (ZH)**: CLIP-KOA：基于多模态学习和对称感知损失函数的膝关节骨关节炎诊断增强方法 

**Authors**: Yejin Jeong, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.19443)  

**Abstract**: Knee osteoarthritis (KOA) is a universal chronic musculoskeletal disorders worldwide, making early diagnosis crucial. Currently, the Kellgren and Lawrence (KL) grading system is widely used to assess KOA severity. However, its high inter-observer variability and subjectivity hinder diagnostic consistency. To address these limitations, automated diagnostic techniques using deep learning have been actively explored in recent years. In this study, we propose a CLIP-based framework (CLIP-KOA) to enhance the consistency and reliability of KOA grade prediction. To achieve this, we introduce a learning approach that integrates image and text information and incorporate Symmetry Loss and Consistency Loss to ensure prediction consistency between the original and flipped images. CLIP-KOA achieves state-of-the-art accuracy of 71.86\% on KOA severity prediction task, and ablation studies show that CLIP-KOA has 2.36\% improvement in accuracy over the standard CLIP model due to our contribution. This study shows a novel direction for data-driven medical prediction not only to improve reliability of fine-grained diagnosis and but also to explore multimodal methods for medical image analysis. Our code is available at this https URL. 

**Abstract (ZH)**: 基于CLIP的KOA分级框架：提高膝关节骨性关节炎严重程度预测的一致性和可靠性 

---
# Platonic Grounding for Efficient Multimodal Language Models 

**Title (ZH)**: 柏拉图式的多模态语言模型接地方法 

**Authors**: Moulik Choraria, Xinbo Wu, Akhil Bhimaraju, Nitesh Sekhar, Yue Wu, Xu Zhang, Prateek Singhal, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2504.19327)  

**Abstract**: The hyperscaling of data and parameter count in Transformer-based models is yielding diminishing performance improvement, especially when weighed against training costs. Such plateauing indicates the importance of methods for more efficient finetuning and inference, while retaining similar performance. This is especially relevant for multimodal learning paradigms, where inference costs of processing multimodal tokens can determine the model's practical viability. At the same time, research on representations and mechanistic interpretability has improved our understanding of the inner workings of Transformer-based models; one such line of work reveals an implicit alignment in the deeper layers of pretrained models, across modalities. Taking inspiration from this, we motivate and propose a simple modification to existing multimodal frameworks that rely on aligning pretrained models. We demonstrate that our approach maintains and, in some cases, even improves performance of baseline methods while achieving significant gains in both training and inference-time compute. Our work also has implications for combining pretrained models into larger systems efficiently. 

**Abstract (ZH)**: 基于Transformer模型的数据和参数规模超标律导致性能改进递减，尤其是在与训练成本相比时更为明显。这种增长 plateau 表明了更高效微调和推理方法的重要性，同时保持类似性能。这对于多模态学习范式尤其 relevant，因为处理多模态令牌的推理成本可能决定模型的实际可行性。同时，关于表示和机械可解释性的研究提升了我们对基于Transformer模型内部机制的理解；其中一项研究揭示了预训练模型深层层间跨模态的隐式对齐。受到这一发现的启发，我们动机并提出了一种对现有依赖于预训练模型对齐的多模态框架的简单修改。我们证明，我们的方法不仅能保持，而且在某些情况下甚至能改进基准方法的性能，在训练时间和推理时间的计算上取得了显著的增益。我们的工作还对高效地将预训练模型组合到更大系统产生了影响。 

---
# VIST-GPT: Ushering in the Era of Visual Storytelling with LLMs? 

**Title (ZH)**: VIST-GPT: 用大型语言模型 usher 哪里视觉叙事的时代？ 

**Authors**: Mohamed Gado, Towhid Taliee, Muhammad Memon, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.19267)  

**Abstract**: Visual storytelling is an interdisciplinary field combining computer vision and natural language processing to generate cohesive narratives from sequences of images. This paper presents a novel approach that leverages recent advancements in multimodal models, specifically adapting transformer-based architectures and large multimodal models, for the visual storytelling task. Leveraging the large-scale Visual Storytelling (VIST) dataset, our VIST-GPT model produces visually grounded, contextually appropriate narratives. We address the limitations of traditional evaluation metrics, such as BLEU, METEOR, ROUGE, and CIDEr, which are not suitable for this task. Instead, we utilize RoViST and GROOVIST, novel reference-free metrics designed to assess visual storytelling, focusing on visual grounding, coherence, and non-redundancy. These metrics provide a more nuanced evaluation of narrative quality, aligning closely with human judgment. 

**Abstract (ZH)**: 视觉 storytelling 是一个将计算机视觉和自然语言处理相结合的跨学科领域，旨在从图像序列中生成连贯的故事叙述。本文提出了一种新方法，利用近年来多模态模型的最新进展，特别是适应基于变换器的架构和大型多模态模型，用于视觉 storytelling 任务。利用大规模 Visual Storytelling (VIST) 数据集，我们的 VIST-GPT 模型生成了与视觉内容紧密结合且上下文适当的叙述。我们针对传统评价指标（如 BLEU、METEOR、ROUGE 和 CIDEr）的局限性，这些指标不适用于此任务，而是采用了 RoViST 和 GROOVIST 这两类新颖的无需参考的评价指标，专门用于评估视觉 storytelling，重点关注视觉接地、连贯性和非冗余性。这些指标能够更细致地评估叙述质量，与人类判断更为一致。 

---
# CapsFake: A Multimodal Capsule Network for Detecting Instruction-Guided Deepfakes 

**Title (ZH)**: CapsFake：一种多模态胶囊网络用于检测指令引导的深度假信息 

**Authors**: Tuan Nguyen, Naseem Khan, Issa Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2504.19212)  

**Abstract**: The rapid evolution of deepfake technology, particularly in instruction-guided image editing, threatens the integrity of digital images by enabling subtle, context-aware manipulations. Generated conditionally from real images and textual prompts, these edits are often imperceptible to both humans and existing detection systems, revealing significant limitations in current defenses. We propose a novel multimodal capsule network, CapsFake, designed to detect such deepfake image edits by integrating low-level capsules from visual, textual, and frequency-domain modalities. High-level capsules, predicted through a competitive routing mechanism, dynamically aggregate local features to identify manipulated regions with precision. Evaluated on diverse datasets, including MagicBrush, Unsplash Edits, Open Images Edits, and Multi-turn Edits, CapsFake outperforms state-of-the-art methods by up to 20% in detection accuracy. Ablation studies validate its robustness, achieving detection rates above 94% under natural perturbations and 96% against adversarial attacks, with excellent generalization to unseen editing scenarios. This approach establishes a powerful framework for countering sophisticated image manipulations. 

**Abstract (ZH)**: 深度生成虚假图像技术的快速演进，特别是在指令引导的图像编辑中，威胁着数字图像的完整性，通过实现情境感知的微妙篡改。生成这些编辑既来自于真实图像又来源于文本提示，常常无法被人类和现有的检测系统察觉，揭示了当前防御措施的重要局限性。我们提出了一种新型的多模态胶囊网络CapsFake，通过结合视觉、文本和频域模态的低层胶囊来检测这些深度生成虚假图像编辑。通过竞争路由机制预测的高层胶囊动态聚合局部特征，以精确识别篡改区域。CapsFake在包括MagicBrush、Unsplash Edits、Open Images Edits和Multi-turn Edits等多种数据集上的检测准确性比最先进的方法高出最高达20%。消融研究验证了其鲁棒性，在天然扰动下实现超过94%的检测率，并在对抗性攻击下实现超过96%的检测率，表现出色地适用于未见过的编辑场景。这种方法建立了对抗复杂图像篡改的强大框架。 

---
# Audio-Driven Talking Face Video Generation with Joint Uncertainty Learning 

**Title (ZH)**: 基于音频驱动的联合不确定性学习面部视频生成 

**Authors**: Yifan Xie, Fei Ma, Yi Bin, Ying He, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18810)  

**Abstract**: Talking face video generation with arbitrary speech audio is a significant challenge within the realm of digital human technology. The previous studies have emphasized the significance of audio-lip synchronization and visual quality. Currently, limited attention has been given to the learning of visual uncertainty, which creates several issues in existing systems, including inconsistent visual quality and unreliable performance across different input conditions. To address the problem, we propose a Joint Uncertainty Learning Network (JULNet) for high-quality talking face video generation, which incorporates a representation of uncertainty that is directly related to visual error. Specifically, we first design an uncertainty module to individually predict the error map and uncertainty map after obtaining the generated image. The error map represents the difference between the generated image and the ground truth image, while the uncertainty map is used to predict the probability of incorrect estimates. Furthermore, to match the uncertainty distribution with the error distribution through a KL divergence term, we introduce a histogram technique to approximate the distributions. By jointly optimizing error and uncertainty, the performance and robustness of our model can be enhanced. Extensive experiments demonstrate that our method achieves superior high-fidelity and audio-lip synchronization in talking face video generation compared to previous methods. 

**Abstract (ZH)**: 具有任意语音音频的对话视频生成中的视听说不确定性的联合学习是一项数字人类技术中的显著挑战。现有的研究表明，音频-唇部同步和视觉质量的重要性。当前，对视觉不确定性的学习关注不足，这在现有系统中引发了视觉质量不一致和不同输入条件下的不可靠性能等问题。为了解决这一问题，我们提出了一种联合不确定性学习网络（JULNet），用于高质量的对话视频生成，该网络包含一个与视觉误差直接相关的不确定性表示。具体地，我们首先设计了一个不确定性模块，在生成图像后分别预测误差图和不确定性图。误差图表示生成图像与真实图像之间的差异，而不确定性图用于预测不正确估计的概率。此外，为了通过KL散度项将不确定性分布与误差分布匹配，我们引入了一种直方图技术来近似分布。通过联合优化误差和不确定性，可以提高我们模型的性能和鲁棒性。广泛的实验表明，与先前的方法相比，我们的方法在对话视频生成中实现了更高的保真度和音频-唇部同步。 

---
