# Simultaneous Localization and 3D-Semi Dense Mapping for Micro Drones Using Monocular Camera and Inertial Sensors 

**Title (ZH)**: 使用单目摄像头和惯性传感器进行微无人机的同时定位与三维半稠密建图 

**Authors**: Jeryes Danial, Yosi Ben Asher, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2511.14335)  

**Abstract**: Monocular simultaneous localization and mapping (SLAM) algorithms estimate drone poses and build a 3D map using a single camera. Current algorithms include sparse methods that lack detailed geometry, while learning-driven approaches produce dense maps but are computationally intensive. Monocular SLAM also faces scale ambiguities, which affect its accuracy. To address these challenges, we propose an edge-aware lightweight monocular SLAM system combining sparse keypoint-based pose estimation with dense edge reconstruction. Our method employs deep learning-based depth prediction and edge detection, followed by optimization to refine keypoints and edges for geometric consistency, without relying on global loop closure or heavy neural computations. We fuse inertial data with vision by using an extended Kalman filter to resolve scale ambiguity and improve accuracy. The system operates in real time on low-power platforms, as demonstrated on a DJI Tello drone with a monocular camera and inertial sensors. In addition, we demonstrate robust autonomous navigation and obstacle avoidance in indoor corridors and on the TUM RGBD dataset. Our approach offers an effective, practical solution to real-time mapping and navigation in resource-constrained environments. 

**Abstract (ZH)**: 单目同时定位与建图（SLAM）算法使用单个摄像头估计无人机姿势并构建3D地图。现有算法包括缺乏详细几何信息的稀疏方法，而基于学习的方法则生成密集地图但计算量大。单目SLAM还面临尺度模糊问题，影响其精度。为应对这些挑战，我们提出了一种结合基于稀疏关键点的姿势估计和密集边缘重建的边缘意识轻量级单目SLAM系统。该方法利用基于深度学习的深度预测和边缘检测，并通过优化来细化关键点和边缘以确保几何一致性，不依赖全局回环闭合或复杂的神经计算。我们利用扩展卡尔曼滤波融合惯性数据与视觉信息以解决尺度模糊问题并提高精度。该系统能够在低功率平台上实时运行，如在使用单目摄像头和惯性传感器的DJI Tello无人机上验证。此外，我们在室内走廊和TUM RGBD数据集上展示了稳健的自主导航和避障能力。我们的方法为资源受限环境下的实时建图与导航提供了有效解决方案。 

---
# Co-Me: Confidence-Guided Token Merging for Visual Geometric Transformers 

**Title (ZH)**: Co-Me: 信心引导的令牌合并方法用于视觉几何变换器 

**Authors**: Yutian Chen, Yuheng Qiu, Ruogu Li, Ali Agha, Shayegan Omidshafiei, Jay Patrikar, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2511.14751)  

**Abstract**: We propose Confidence-Guided Token Merging (Co-Me), an acceleration mechanism for visual geometric transformers without retraining or finetuning the base model. Co-Me distilled a light-weight confidence predictor to rank tokens by uncertainty and selectively merge low-confidence ones, effectively reducing computation while maintaining spatial coverage. Compared to similarity-based merging or pruning, the confidence signal in Co-Me reliably indicates regions emphasized by the transformer, enabling substantial acceleration without degrading performance. Co-Me applies seamlessly to various multi-view and streaming visual geometric transformers, achieving speedups that scale with sequence length. When applied to VGGT and MapAnything, Co-Me achieves up to $11.3\times$ and $7.2\times$ speedup, making visual geometric transformers practical for real-time 3D perception and reconstruction. 

**Abstract (ZH)**: 我们提出了一种加速视觉几何变换器的机制Confidence-Guided Token Merging (Co-Me)，无需重新训练或微调基础模型便能有效加速。Co-Me 提炼出一个轻量级的置信度预测器，根据不确定性对令牌进行排名，并选择性地合并低置信度的令牌，从而有效地减少计算量同时保持空间覆盖率。与基于相似性的合并或剪枝相比，Co-Me 中的置信度信号可靠地指示了变换器所强调的区域，能够在不降低性能的情况下实现显著加速。Co-Me 可无缝应用于各种多视图和流式视觉几何变换器，并实现了随序列长度而扩展的加速效果。当应用于VGGT和MapAnything时，Co-Me 分别实现了高达11.3倍和7.2倍的加速，使视觉几何变换器适用于实时三维感知和重构。 

---
# A Trajectory-free Crash Detection Framework with Generative Approach and Segment Map Diffusion 

**Title (ZH)**: 基于生成方法和段图扩散的无轨迹碰撞检测框架 

**Authors**: Weiying Shen, Hao Yu, Yu Dong, Pan Liu, Yu Han, Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13795)  

**Abstract**: Real-time crash detection is essential for developing proactive safety management strategy and enhancing overall traffic efficiency. To address the limitations associated with trajectory acquisition and vehicle tracking, road segment maps recording the individual-level traffic dynamic data were directly served in crash detection. A novel two-stage trajectory-free crash detection framework, was present to generate the rational future road segment map and identify crashes. The first-stage diffusion-based segment map generation model, Mapfusion, conducts a noisy-to-normal process that progressively adds noise to the road segment map until the map is corrupted to pure Gaussian noise. The denoising process is guided by sequential embedding components capturing the temporal dynamics of segment map sequences. Furthermore, the generation model is designed to incorporate background context through ControlNet to enhance generation control. Crash detection is achieved by comparing the monitored segment map with the generations from diffusion model in second stage. Trained on non-crash vehicle motion data, Mapfusion successfully generates realistic road segment evolution maps based on learned motion patterns and remains robust across different sampling intervals. Experiments on real-world crashes indicate the effectiveness of the proposed two-stage method in accurately detecting crashes. 

**Abstract (ZH)**: 实时碰撞检测是开发前瞻性安全管理策略和提升整体交通效率的关键。为了解决轨迹获取和车辆跟踪的局限性，采用记录个体级交通动态数据的道路段落地图直接用于碰撞检测。提出了一种新型两阶段无轨迹碰撞检测框架，以生成合理的未来道路段落地图并识别碰撞。第一阶段基于扩散的段落地图生成模型Mapfusion，通过逐步向道路段落地图添加噪音，直到地图变为纯高斯噪音，实现去噪过程。去噪过程由捕捉段落地图序列的时间动态的顺序嵌入组件引导。此外，生成模型通过ControlNet整合背景上下文，以增强生成控制。碰撞检测通过将监测的道路段落地图与扩散模型生成的结果进行对比实现。Mapfusion基于非碰撞车辆运动数据训练，成功生成基于学习到的运动模式的现实道路段落演化地图，并且在不同的采样间隔下保持稳健。实验表明，所提出的两阶段方法在准确检测碰撞方面是有效的。 

---
# nuCarla: A nuScenes-Style Bird's-Eye View Perception Dataset for CARLA Simulation 

**Title (ZH)**: nuCarla：CARLA仿真的nuScenes样式 bird's-eye view 视角感知数据集 

**Authors**: Zhijie Qiao, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13744)  

**Abstract**: End-to-end (E2E) autonomous driving heavily relies on closed-loop simulation, where perception, planning, and control are jointly trained and evaluated in interactive environments. Yet, most existing datasets are collected from the real world under non-interactive conditions, primarily supporting open-loop learning while offering limited value for closed-loop testing. Due to the lack of standardized, large-scale, and thoroughly verified datasets to facilitate learning of meaningful intermediate representations, such as bird's-eye-view (BEV) features, closed-loop E2E models remain far behind even simple rule-based baselines. To address this challenge, we introduce nuCarla, a large-scale, nuScenes-style BEV perception dataset built within the CARLA simulator. nuCarla features (1) full compatibility with the nuScenes format, enabling seamless transfer of real-world perception models; (2) a dataset scale comparable to nuScenes, but with more balanced class distributions; (3) direct usability for closed-loop simulation deployment; and (4) high-performance BEV backbones that achieve state-of-the-art detection results. By providing both data and models as open benchmarks, nuCarla substantially accelerates closed-loop E2E development, paving the way toward reliable and safety-aware research in autonomous driving. 

**Abstract (ZH)**: 端到端自主驾驶高度依赖闭环仿真，其中感知、规划和控制在交互环境中联合训练和评估。然而，现有的大多数数据集是在非交互条件下从现实世界收集的，主要支持开环学习，对闭环测试提供的价值有限。由于缺乏标准化、大规模且充分验证的数据集来促进有意义中间表示的学习，如鸟瞰图（BEV）特征，闭环端到端模型仍然落后于简单的规则基线。为了解决这一挑战，我们介绍了nuCarla，一种在CARLA仿真器中构建的、风格类似于nuScenes的BEV感知数据集。nuCarla具有以下特点：（1）完全兼容nuScenes格式，使现实世界的感知模型无缝转移；（2）数据集规模与nuScenes相当，但类别分布更加平衡；（3）直接适用于闭环仿真部署；（4）高性能BEV骨干网络，实现最先进检测结果。通过提供开放基准的数据和模型，nuCarla显著加速了闭环端到端开发，为自主驾驶的研究铺平了可靠性和安全意识的道路。 

---
# Artificial Intelligence Agents in Music Analysis: An Integrative Perspective Based on Two Use Cases 

**Title (ZH)**: 音乐分析中的人工智能代理：基于两个案例研究的综合视角 

**Authors**: Antonio Manuel Martínez-Heredia, Dolores Godrid Rodríguez, Andrés Ortiz García  

**Link**: [PDF](https://arxiv.org/pdf/2511.13987)  

**Abstract**: This paper presents an integrative review and experimental validation of artificial intelligence (AI) agents applied to music analysis and education. We synthesize the historical evolution from rule-based models to contemporary approaches involving deep learning, multi-agent architectures, and retrieval-augmented generation (RAG) frameworks. The pedagogical implications are evaluated through a dual-case methodology: (1) the use of generative AI platforms in secondary education to foster analytical and creative skills; (2) the design of a multiagent system for symbolic music analysis, enabling modular, scalable, and explainable workflows.
Experimental results demonstrate that AI agents effectively enhance musical pattern recognition, compositional parameterization, and educational feedback, outperforming traditional automated methods in terms of interpretability and adaptability. The findings highlight key challenges concerning transparency, cultural bias, and the definition of hybrid evaluation metrics, emphasizing the need for responsible deployment of AI in educational environments.
This research contributes to a unified framework that bridges technical, pedagogical, and ethical considerations, offering evidence-based guidance for the design and application of intelligent agents in computational musicology and music education. 

**Abstract (ZH)**: 本研究综述并实验验证了应用于音乐分析和教育的人工智能（AI）代理。我们从基于规则的模型的历史演变到现今涉及深度学习、多代理架构和检索增强生成（RAG）框架的当代方法进行了综合分析。通过双重案例研究方法评估了教育意义：（1）在中等教育中使用生成型AI平台以培养分析和创造技能；（2）设计一种符号音乐分析的多代理系统，实现模块化、可扩展且可解释的工作流程。

实验结果表明，AI代理有效增强了音乐模式识别、作曲参数化和教育反馈，从可解释性和适应性方面超越了传统的自动化方法。研究结果突出了透明度、文化偏见以及混合评价指标定义的关键挑战，强调了在教育环境中负责任部署AI的必要性。

本研究为统合技术、教育和伦理考量的框架做出了贡献，提供了基于证据的设计和应用智能代理在计算音乐学和音乐教育中的指导。 

---
# Scene Graph-Guided Generative AI Framework for Synthesizing and Evaluating Industrial Hazard Scenarios 

**Title (ZH)**: 基于场景图引导的生成式AI框架：工业 hazard 情景合成与评估 

**Authors**: Sanjay Acharjee, Abir Khan Ratul, Diego Patino, Md Nazmus Sakib  

**Link**: [PDF](https://arxiv.org/pdf/2511.13970)  

**Abstract**: Training vision models to detect workplace hazards accurately requires realistic images of unsafe conditions that could lead to accidents. However, acquiring such datasets is difficult because capturing accident-triggering scenarios as they occur is nearly impossible. To overcome this limitation, this study presents a novel scene graph-guided generative AI framework that synthesizes photorealistic images of hazardous scenarios grounded in historical Occupational Safety and Health Administration (OSHA) accident reports. OSHA narratives are analyzed using GPT-4o to extract structured hazard reasoning, which is converted into object-level scene graphs capturing spatial and contextual relationships essential for understanding risk. These graphs guide a text-to-image diffusion model to generate compositionally accurate hazard scenes. To evaluate the realism and semantic fidelity of the generated data, a visual question answering (VQA) framework is introduced. Across four state-of-the-art generative models, the proposed VQA Graph Score outperforms CLIP and BLIP metrics based on entropy-based validation, confirming its higher discriminative sensitivity. 

**Abstract (ZH)**: 使用场景图引导的生成AI框架合成基于历史OSHA事故报告的现实主义危险场景图像以准确检测职场危害 

---
# KANGURA: Kolmogorov-Arnold Network-Based Geometry-Aware Learning with Unified Representation Attention for 3D Modeling of Complex Structures 

**Title (ZH)**: KANGURA：基于柯尔莫戈洛夫-阿诺尔德网络的统一表示注意力几何感知学习用于复杂结构的3D建模 

**Authors**: Mohammad Reza Shafie, Morteza Hajiabadi, Hamed Khosravi, Mobina Noori, Imtiaz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2511.13798)  

**Abstract**: Microbial Fuel Cells (MFCs) offer a promising pathway for sustainable energy generation by converting organic matter into electricity through microbial processes. A key factor influencing MFC performance is the anode structure, where design and material properties play a crucial role. Existing predictive models struggle to capture the complex geometric dependencies necessary to optimize these structures. To solve this problem, we propose KANGURA: Kolmogorov-Arnold Network-Based Geometry-Aware Learning with Unified Representation Attention. KANGURA introduces a new approach to three-dimensional (3D) machine learning modeling. It formulates prediction as a function decomposition problem, where Kolmogorov-Arnold Network (KAN)- based representation learning reconstructs geometric relationships without a conventional multi- layer perceptron (MLP). To refine spatial understanding, geometry-disentangled representation learning separates structural variations into interpretable components, while unified attention mechanisms dynamically enhance critical geometric regions. Experimental results demonstrate that KANGURA outperforms over 15 state-of-the-art (SOTA) models on the ModelNet40 benchmark dataset, achieving 92.7% accuracy, and excels in a real-world MFC anode structure problem with 97% accuracy. This establishes KANGURA as a robust framework for 3D geometric modeling, unlocking new possibilities for optimizing complex structures in advanced manufacturing and quality-driven engineering applications. 

**Abstract (ZH)**: KANGURA: Kolmogorov-Arnold Network-Based Geometry-Aware Learning with Unified Representation Attention 

---
# ARC Is a Vision Problem! 

**Title (ZH)**: ARC 是一个视觉问题！ 

**Authors**: Keya Hu, Ali Cy, Linlu Qiu, Xiaoman Delores Ding, Runqian Wang, Yeyin Eva Zhu, Jacob Andreas, Kaiming He  

**Link**: [PDF](https://arxiv.org/pdf/2511.14761)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) is designed to promote research on abstract reasoning, a fundamental aspect of human intelligence. Common approaches to ARC treat it as a language-oriented problem, addressed by large language models (LLMs) or recurrent reasoning models. However, although the puzzle-like tasks in ARC are inherently visual, existing research has rarely approached the problem from a vision-centric perspective. In this work, we formulate ARC within a vision paradigm, framing it as an image-to-image translation problem. To incorporate visual priors, we represent the inputs on a "canvas" that can be processed like natural images. It is then natural for us to apply standard vision architectures, such as a vanilla Vision Transformer (ViT), to perform image-to-image mapping. Our model is trained from scratch solely on ARC data and generalizes to unseen tasks through test-time training. Our framework, termed Vision ARC (VARC), achieves 60.4% accuracy on the ARC-1 benchmark, substantially outperforming existing methods that are also trained from scratch. Our results are competitive with those of leading LLMs and close the gap to average human performance. 

**Abstract (ZH)**: 视觉抽象推理语料库（Vision-based Abstraction and Reasoning Corpus, VARC）的设计及其应用 

---
# Zero-shot Synthetic Video Realism Enhancement via Structure-aware Denoising 

**Title (ZH)**: 结构感知去噪驱动的零样本合成视频真实感增强 

**Authors**: Yifan Wang, Liya Ji, Zhanghan Ke, Harry Yang, Ser-Nam Lim, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14719)  

**Abstract**: We propose an approach to enhancing synthetic video realism, which can re-render synthetic videos from a simulator in photorealistic fashion. Our realism enhancement approach is a zero-shot framework that focuses on preserving the multi-level structures from synthetic videos into the enhanced one in both spatial and temporal domains, built upon a diffusion video foundational model without further fine-tuning. Specifically, we incorporate an effective modification to have the generation/denoising process conditioned on estimated structure-aware information from the synthetic video, such as depth maps, semantic maps, and edge maps, by an auxiliary model, rather than extracting the information from a simulator. This guidance ensures that the enhanced videos are consistent with the original synthetic video at both the structural and semantic levels. Our approach is a simple yet general and powerful approach to enhancing synthetic video realism: we show that our approach outperforms existing baselines in structural consistency with the original video while maintaining state-of-the-art photorealism quality in our experiments. 

**Abstract (ZH)**: 我们提出了一种增强合成视频真实感的方法，可以从模拟器中以照片级真实感方式重新渲染合成视频。我们的真实感增强方法是一个零样本框架，专注于在空间和时间域内保留合成视频中的多级结构到增强视频中，该方法基于一个扩散视频基础模型，无需进一步微调。具体而言，我们通过辅助模型集成了一种有效的修改，使生成/去噪过程依赖于从合成视频中估算的结构感知信息，如深度图、语义图和边缘图，而不是从模拟器中提取信息。这种指导确保增强视频在结构和语义层面与原始合成视频一致。我们的方法是一种简单的、通用且强大的合成视频真实感增强方法：我们的实验结果表明，该方法在结构一致性方面优于现有基线，同时保持了最先进的照片级真实感质量。 

---
# Impact of Image Resolution on Age Estimation with DeepFace and InsightFace 

**Title (ZH)**: 基于DeepFace和InsightFace的图像分辨率对年龄估计影响的研究 

**Authors**: Shiyar Jamo  

**Link**: [PDF](https://arxiv.org/pdf/2511.14689)  

**Abstract**: Automatic age estimation is widely used for age verification, where input images often vary considerably in resolution. This study evaluates the effect of image resolution on age estimation accuracy using DeepFace and InsightFace. A total of 1000 images from the IMDB-Clean dataset were processed in seven resolutions, resulting in 7000 test samples. Performance was evaluated using Mean Absolute Error (MAE), Standard Deviation (SD), and Median Absolute Error (MedAE). Based on this study, we conclude that input image resolution has a clear and consistent impact on the accuracy of age estimation in both DeepFace and InsightFace. Both frameworks achieve optimal performance at 224x224 pixels, with an MAE of 10.83 years (DeepFace) and 7.46 years (InsightFace). At low resolutions, MAE increases substantially, while very high resolutions also degrade accuracy. InsightFace is consistently faster than DeepFace across all resolutions. 

**Abstract (ZH)**: 图像分辨率对DeepFace和InsightFace年龄估计准确性的影响研究 

---
# Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms 

**Title (ZH)**: 使用心脏信号改进 Doppler 全息图中视网膜动脉和静脉的分割 

**Authors**: Marius Dubosc, Yann Fischer, Zacharie Auray, Nicolas Boutry, Edwin Carlinet, Michael Atlan, Thierry Geraud  

**Link**: [PDF](https://arxiv.org/pdf/2511.14654)  

**Abstract**: Doppler holography is an emerging retinal imaging technique that captures the dynamic behavior of blood flow with high temporal resolution, enabling quantitative assessment of retinal hemodynamics. This requires accurate segmentation of retinal arteries and veins, but traditional segmentation methods focus solely on spatial information and overlook the temporal richness of holographic data. In this work, we propose a simple yet effective approach for artery-vein segmentation in temporal Doppler holograms using standard segmentation architectures. By incorporating features derived from a dedicated pulse analysis pipeline, our method allows conventional U-Nets to exploit temporal dynamics and achieve performance comparable to more complex attention- or iteration-based models. These findings demonstrate that time-resolved preprocessing can unlock the full potential of deep learning for Doppler holography, opening new perspectives for quantitative exploration of retinal hemodynamics. The dataset is publicly available at this https URL 

**Abstract (ZH)**: 频域光学全息术是一种新兴的视网膜成像技术，能够以高时间分辨率捕获血液流动的动力学行为，从而实现视网膜血液动力学的定量评估。这需要对视网膜动脉和静脉进行准确分割，但传统分割方法仅专注于空间信息，而忽略了光学全息数据的时间丰富性。在本工作中，我们提出了一种简单而有效的方法，利用标准分割架构对时间域频域光学全息图像中的动脉和静脉进行分割，通过结合来自专门脉冲分析管道提取的特征，我们的方法使常规U-Net能够利用时间动态性，并实现与更复杂注意力机制或迭代模型相当的性能。这些发现表明，时间分辨预处理可以解锁深度学习在频域光学全息术中的全部潜力，为定量探索视网膜血液动力学提供了新的视角。数据集可从此链接获取：this https URL 

---
# SweeperBot: Making 3D Browsing Accessible through View Analysis and Visual Question Answering 

**Title (ZH)**: SweeperBot: 通过视角分析和视觉问答实现3D浏览的无障碍访问 

**Authors**: Chen Chen, Cuong Nguyen, Alexa Siu, Dingzeyu Li, Nadir Weibel  

**Link**: [PDF](https://arxiv.org/pdf/2511.14567)  

**Abstract**: Accessing 3D models remains challenging for Screen Reader (SR) users. While some existing 3D viewers allow creators to provide alternative text, they often lack sufficient detail about the 3D models. Grounded on a formative study, this paper introduces SweeperBot, a system that enables SR users to leverage visual question answering to explore and compare 3D models. SweeperBot answers SR users' visual questions by combining an optimal view selection technique with the strength of generative- and recognition-based foundation models. An expert review with 10 Blind and Low-Vision (BLV) users with SR experience demonstrated the feasibility of using SweeperBot to assist BLV users in exploring and comparing 3D models. The quality of the descriptions generated by SweeperBot was validated by a second survey study with 30 sighted participants. 

**Abstract (ZH)**: Screen Reader 用户访问 3D 模型仍然具有挑战性。基于形成性研究，本文介绍了 SweeperBot 系统，该系统使 Screen Reader 用户能够利用视觉问答技术探索和比较 3D 模型。SweeperBot 通过结合最优视图选择技术和生成式及识别基础模型的优势，回答 SR 用户的视觉问题。10 名有 Screen Reader 经验的盲人和低视力（BLV）用户进行的专家审查证明了使用 SweeperBot 辅助 BLV 用户探索和比较 3D 模型的可能性。通过另一项涉及 30 名视力正常参与者的研究进一步验证了 SweeperBot 生成的描述质量。 

---
# Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models 

**Title (ZH)**: Apo2Mol：基于动态口袋意识扩散模型的三维分子生成 

**Authors**: Xinzhe Zheng, Shiyu Jiang, Gustavo Seabra, Chenglong Li, Yanjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14559)  

**Abstract**: Deep generative models are rapidly advancing structure-based drug design, offering substantial promise for generating small molecule ligands that bind to specific protein targets. However, most current approaches assume a rigid protein binding pocket, neglecting the intrinsic flexibility of proteins and the conformational rearrangements induced by ligand binding, limiting their applicability in practical drug discovery. Here, we propose Apo2Mol, a diffusion-based generative framework for 3D molecule design that explicitly accounts for conformational flexibility in protein binding pockets. To support this, we curate a dataset of over 24,000 experimentally resolved apo-holo structure pairs from the Protein Data Bank, enabling the characterization of protein structure changes associated with ligand binding. Apo2Mol employs a full-atom hierarchical graph-based diffusion model that simultaneously generates 3D ligand molecules and their corresponding holo pocket conformations from input apo states. Empirical studies demonstrate that Apo2Mol can achieve state-of-the-art performance in generating high-affinity ligands and accurately capture realistic protein pocket conformational changes. 

**Abstract (ZH)**: 基于扩散的分子生成框架Apo2Mol在考虑蛋白质结合口袋构象柔性的三维分子设计中的应用 

---
# Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding 

**Title (ZH)**: 代理视频智能：一种灵活的高级视频探索与理解框架 

**Authors**: Hong Gao, Yiming Bao, Xuezhen Tu, Yutong Xu, Yue Jin, Yiyang Mu, Bin Zhong, Linan Yue, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14446)  

**Abstract**: Video understanding requires not only visual recognition but also complex reasoning. While Vision-Language Models (VLMs) demonstrate impressive capabilities, they typically process videos largely in a single-pass manner with limited support for evidence revisit and iterative refinement. While recently emerging agent-based methods enable long-horizon reasoning, they either depend heavily on expensive proprietary models or require extensive agentic RL training. To overcome these limitations, we propose Agentic Video Intelligence (AVI), a flexible and training-free framework that can mirror human video comprehension through system-level design and optimization. AVI introduces three key innovations: (1) a human-inspired three-phase reasoning process (Retrieve-Perceive-Review) that ensures both sufficient global exploration and focused local analysis, (2) a structured video knowledge base organized through entity graphs, along with multi-granularity integrated tools, constituting the agent's interaction environment, and (3) an open-source model ensemble combining reasoning LLMs with lightweight base CV models and VLM, eliminating dependence on proprietary APIs or RL training. Experiments on LVBench, VideoMME-Long, LongVideoBench, and Charades-STA demonstrate that AVI achieves competitive performance while offering superior interpretability. 

**Abstract (ZH)**: 视频理解不仅需要视觉识别还要求复杂的推理。为此，我们提出了基于代理的视频智能（AVI），这是一种灵活且无需训练的框架，通过系统级设计和优化，可以模拟人类的视频理解过程。AVI 引入了三项关键创新：（1）受人类启发的三阶段推理过程（检索-感知-复审），确保了充分的全局探索和聚焦的局部分析；（2）以实体图组织的结构化视频知识库，以及多层次集成工具，构成代理的交互环境；（3）开源模型集合，结合推理语言模型与轻量级基础计算机视觉模型和视频语言模型，消除对专有API或强化学习训练的依赖。在 LVBench、VideoMME-Long、LongVideoBench 和 Charades-STA 上的实验表明，AVI 获得了竞争力的表现并提供了更好的可解释性。 

---
# Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving 

**Title (ZH)**: 全尺度范围内的立体匹配作弊：针对自动驾驶中双目深度估计的物理对抗攻击 

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.14386)  

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information. 

**Abstract (ZH)**: 虽然用于实现自动驾驶感知的深度神经网络模型已被证明容易受到对抗样本的影响，已知的攻击通常利用二维patches并主要针对单目感知。因此，基于立体匹配的双目深度估计中物理对抗样本（PAEs）的有效性仍 largely unexplored. 为此，我们提出了第一个针对自动驾驶背景下立体匹配模型的纹理启用物理对抗攻击。我们的方法采用全局迷彩纹理的3D PAE，而非局部基于2D patches的PAE，确保了不同视点间视觉一致性和攻击有效性。为应对这些相机的视差效应，我们还提出了一种新的3D立体匹配渲染模块，使得PAE能够在双目视觉中与真实世界的位置和方向对齐。此外，我们提出了一个新的融合攻击，通过细腻的PAE优化无缝地将目标融入环境。该攻击在现有无法无缝融入背景的隐藏攻击中显著增强了隐身性和致命性。广泛评估表明，我们的PAE能够成功使立体模型产生错误的深度信息。 

---
# LSP-YOLO: A Lightweight Single-Stage Network for Sitting Posture Recognition on Embedded Devices 

**Title (ZH)**: LSP-YOLO：嵌入式设备上基于轻量级单阶段网络的坐姿识别 

**Authors**: Nanjun Li, Ziyue Hao, Quanqiang Wang, Xuanyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14322)  

**Abstract**: With the rise in sedentary behavior, health problems caused by poor sitting posture have drawn increasing attention. Most existing methods, whether using invasive sensors or computer vision, rely on two-stage pipelines, which result in high intrusiveness, intensive computation, and poor real-time performance on embedded edge devices. Inspired by YOLOv11-Pose, a lightweight single-stage network for sitting posture recognition on embedded edge devices termed LSP-YOLO was proposed. By integrating partial convolution(PConv) and Similarity-Aware Activation Module(SimAM), a lightweight module, Light-C3k2, was designed to reduce computational cost while maintaining feature extraction capability. In the recognition head, keypoints were directly mapped to posture classes through pointwise convolution, and intermediate supervision was employed to enable efficient fusion of pose estimation and classification. Furthermore, a dataset containing 5,000 images across six posture categories was constructed for model training and testing. The smallest trained model, LSP-YOLO-n, achieved 94.2% accuracy and 251 Fps on personal computer(PC) with a model size of only 1.9 MB. Meanwhile, real-time and high-accuracy inference under constrained computational resources was demonstrated on the SV830C + GC030A platform. The proposed approach is characterized by high efficiency, lightweight design and deployability, making it suitable for smart classrooms, rehabilitation, and human-computer interaction applications. 

**Abstract (ZH)**: 基于嵌入式边缘设备的轻量化单阶段坐姿识别方法 

---
# Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation 

**Title (ZH)**: Few-Shot精确事件检测 via 统一多实体图和知识蒸馏 

**Authors**: Zhaoyu Liu, Kan Jiang, Murong Ma, Zhe Hou, Yun Lin, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.14186)  

**Abstract**: Precise event spotting (PES) aims to recognize fine-grained events at exact moments and has become a key component of sports analytics. This task is particularly challenging due to rapid succession, motion blur, and subtle visual differences. Consequently, most existing methods rely on domain-specific, end-to-end training with large labeled datasets and often struggle in few-shot conditions due to their dependence on pixel- or pose-based inputs alone. However, obtaining large labeled datasets is practically hard. We propose a Unified Multi-Entity Graph Network (UMEG-Net) for few-shot PES. UMEG-Net integrates human skeletons and sport-specific object keypoints into a unified graph and features an efficient spatio-temporal extraction module based on advanced GCN and multi-scale temporal shift. To further enhance performance, we employ multimodal distillation to transfer knowledge from keypoint-based graphs to visual representations. Our approach achieves robust performance with limited labeled data and significantly outperforms baseline models in few-shot settings, providing a scalable and effective solution for few-shot PES. Code is publicly available at this https URL. 

**Abstract (ZH)**: 精确事件定位（PES）旨在识别事件在精确时刻的细粒度事件，并已成为体育分析中的关键组件。由于事件的快速连续发生、运动模糊以及细微的视觉差异，这一任务尤为具有挑战性。因此，大多数现有方法依赖于特定领域的端到端训练和大型标注数据集，但往往在少量标注数据条件下表现不佳，因为它们很大程度上依赖于基于像素或姿态的输入。然而，获取大型标注数据集在实践中是困难的。我们提出了一种统一多实体图网络（UMEG-Net）以应对少量标注数据条件下的PES任务。UMEG-Net将人体骨骼和特定运动对象的关键点整合到一个统一的图中，并基于先进GCN的时空提取模块和多尺度时空移位。为了进一步提升性能，我们采用多模态蒸馏将基于关键点的图的知识转移到视觉表示中。我们的方法在少量标注数据条件下表现稳健，并在少量标注数据设置中显著优于基线模型，提供了一个可扩展且有效的少量标注数据条件下的PES解决方案。代码已在该网址公开。 

---
# CascadedViT: Cascaded Chunk-FeedForward and Cascaded Group Attention Vision Transformer 

**Title (ZH)**: 级联ViT：级联块前馈和级联组注意力视觉变换器 

**Authors**: Srivathsan Sivakumar, Faisal Z. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2511.14111)  

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance across a range of computer vision tasks; however, their high computational, memory, and energy demands hinder deployment on resource-constrained platforms. In this paper, we propose \emph{Cascaded-ViT (CViT)}, a lightweight and compute-efficient vision transformer architecture featuring a novel feedforward network design called \emph{Cascaded-Chunk Feed Forward Network (CCFFN)}. By splitting input features, CCFFN improves parameter and FLOP efficiency without sacrificing accuracy. Experiments on ImageNet-1K show that our \emph{CViT-XL} model achieves 75.5\% Top-1 accuracy while reducing FLOPs by 15\% and energy consumption by 3.3\% compared to EfficientViT-M5. Across various model sizes, the CViT family consistently exhibits the lowest energy consumption, making it suitable for deployment on battery-constrained devices such as mobile phones and drones. Furthermore, when evaluated using a new metric called \emph{Accuracy-Per-FLOP (APF)}, which quantifies compute efficiency relative to accuracy, CViT models consistently achieve top-ranking efficiency. Particularly, CViT-L is 2.2\% more accurate than EfficientViT-M2 while having comparable APF scores. 

**Abstract (ZH)**: Vision Transformers (ViTs)在计算机视觉任务中展现了卓越的性能；然而，其高的计算、内存和能源需求限制了其在资源受限平台上的部署。本文提出了一种轻量级且计算高效的Vision Transformer架构——Cascaded-ViT (CViT)，并提出了一种新的前馈网络设计，称为Cascaded-Chunk Feed Forward Network (CCFFN)。通过输入特征分割，CCFFN在不牺牲准确性的前提下提高了参数和FLOP效率。实验结果表明，我们的CViT-XL模型在ImageNet-1K数据集上达到了75.5%的Top-1准确率，同时FLOPs减少了15%，能量消耗减少了3.3%，相比EfficientViT-M5。在各种模型规模下，CViT家族始终表现出最低的能量消耗，使其适用于智能手机和无人机等电池受限设备的部署。此外，当使用新的评估指标Accuracy-Per-FLOP (APF)进行评估时（该指标量化了计算效率相对于准确性的性能），CViT模型始终表现出最高的效率。特别是，CViT-L的准确率比EfficientViT-M2高2.2%，且APF分数相当。 

---
# GCA-ResUNet:Image segmentation in medical images using grouped coordinate attention 

**Title (ZH)**: GCA-ResUNet：在医疗图像中使用分组坐标注意力进行图像分割 

**Authors**: Jun Ding, Shang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14087)  

**Abstract**: Medical image segmentation underpins computer-aided diagnosis and therapy by supporting clinical diagnosis, preoperative planning, and disease monitoring. While U-Net style convolutional neural networks perform well due to their encoder-decoder structures with skip connections, they struggle to capture long-range dependencies. Transformer-based variants address global context but often require heavy computation and large training datasets. This paper proposes GCA-ResUNet, an efficient segmentation network that integrates Grouped Coordinate Attention (GCA) into ResNet-50 residual blocks. GCA uses grouped coordinate modeling to jointly encode global dependencies across channels and spatial locations, strengthening feature representation and boundary delineation while adding minimal parameter and FLOP overhead compared with self-attention. On the Synapse dataset, GCA-ResUNet achieves a Dice score of 86.11%, and on the ACDC dataset, it reaches 92.64%, surpassing several state-of-the-art baselines while maintaining fast inference and favorable computational efficiency. These results indicate that GCA offers a practical way to enhance convolutional architectures with global modeling capability, enabling high-accuracy and resource-efficient medical image segmentation. 

**Abstract (ZH)**: 基于组坐标注意力的ResUNet在医学图像分割中的应用：支持临床诊断、术前规划和疾病监测的高效网络 

---
# CFG-EC: Error Correction Classifier-Free Guidance 

**Title (ZH)**: CFG-EC: Error Correction Classifier-Free Guidance 

**Authors**: Nakkyu Yang, Yechan Lee, SooJean Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.14075)  

**Abstract**: Classifier-Free Guidance (CFG) has become a mainstream approach for simultaneously improving prompt fidelity and generation quality in conditional generative models. During training, CFG stochastically alternates between conditional and null prompts to enable both conditional and unconditional generation. However, during sampling, CFG outputs both null and conditional prompts simultaneously, leading to inconsistent noise estimates between the training and sampling processes. To reduce this error, we propose CFG-EC, a versatile correction scheme augmentable to any CFG-based method by refining the unconditional noise predictions. CFG-EC actively realigns the unconditional noise error component to be orthogonal to the conditional error component. This corrective maneuver prevents interference between the two guidance components, thereby constraining the sampling error's upper bound and establishing more reliable guidance trajectories for high-fidelity image generation. Our numerical experiments show that CFG-EC handles the unconditional component more effectively than CFG and CFG++, delivering a marked performance increase in the low guidance sampling regime and consistently higher prompt alignment across the board. 

**Abstract (ZH)**: Classifier-Free Guidance (CFG)已成为同时提高条件生成模型的提示忠实度和生成质量的主流方法。在训练过程中，CFG通过在条件提示和空提示之间随机交替来同时启用有条件和无条件生成。然而，在采样过程中，CFG同时输出空和条件提示，导致训练和采样过程之间的噪声估计不一致。为了减少这种误差，我们提出了一种可与任何基于CFG的方法兼容的CFG-EC通用校正方案，通过细化无条件噪声预测来提高无条件噪声预测的准确性。CFG-EC积极将无条件噪声误差分量与条件误差分量正交化，从而防止两个指导分量之间的干扰，限制采样误差的上限，并为高保真图像生成建立更可靠的指导轨迹。我们的数值实验表明，与CFG和CFG++相比，CFG-EC更有效地处理无条件分量，在低指导采样区段表现出显著的性能提升，并且在所有情况下的一致性提示对齐也更高。 

---
# Hybrid Convolution Neural Network Integrated with Pseudo-Newton Boosting for Lumbar Spine Degeneration Detection 

**Title (ZH)**: 融合伪牛顿增强的混合卷积神经网络在腰椎间盘退变检测中的应用 

**Authors**: Pandiyaraju V, Abishek Karthik, Jaspin K, Kannan A, Jaime Lloret  

**Link**: [PDF](https://arxiv.org/pdf/2511.13877)  

**Abstract**: This paper proposes a new enhanced model architecture to perform classification of lumbar spine degeneration with DICOM images while using a hybrid approach, integrating EfficientNet and VGG19 together with custom-designed components. The proposed model is differentiated from traditional transfer learning methods as it incorporates a Pseudo-Newton Boosting layer along with a Sparsity-Induced Feature Reduction Layer that forms a multi-tiered framework, further improving feature selection and representation. The Pseudo-Newton Boosting layer makes smart variations of feature weights, with more detailed anatomical features, which are mostly left out in a transfer learning setup. In addition, the Sparsity-Induced Layer removes redundancy for learned features, producing lean yet robust representations for pathology in the lumbar spine. This architecture is novel as it overcomes the constraints in the traditional transfer learning approach, especially in the high-dimensional context of medical images, and achieves a significant performance boost, reaching a precision of 0.9, recall of 0.861, F1 score of 0.88, loss of 0.18, and an accuracy of 88.1%, compared to the baseline model, EfficientNet. This work will present the architectures, preprocessing pipeline, and experimental results. The results contribute to the development of automated diagnostic tools for medical images. 

**Abstract (ZH)**: 本文提出了一种新的增强模型架构，结合Hybrid方法、EfficientNet和VGG19，并加入自定义组件，用于利用DICOM图像进行腰椎退化分类。所提出的模型与传统的迁移学习方法不同，因为它引入了Pseudo-Newton Boosting层和Sparsity-Induced Feature Reduction层，形成一个多级框架，进一步提高特征选择和表示能力。Pseudo-Newton Boosting层对特征权重进行智能调整，强调更详细的解剖特征，这些特征在迁移学习设置中往往被忽略。此外，Sparsity-Induced层消除了学习特征的冗余性，生成精炼且稳健的病理表示。该架构新颖之处在于它克服了传统迁移学习方法在医学图像高维情境下的局限性，实现了显著的性能提升，相比基线模型EfficientNet，达到了精度0.9、召回率0.861、F1分数0.88、损失0.18和准确率88.1%。本文将介绍该架构、预处理管道和实验结果，这些结果有助于开发自动医学图像诊断工具。 

---
# H-CNN-ViT: A Hierarchical Gated Attention Multi-Branch Model for Bladder Cancer Recurrence Prediction 

**Title (ZH)**: H-CNN-ViT：一种分层门控注意力多分支模型用于膀胱癌复发预测 

**Authors**: Xueyang Li, Zongren Wang, Yuliang Zhang, Zixuan Pan, Yu-Jen Chen, Nishchal Sapkota, Gelei Xu, Danny Z. Chen, Yiyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.13869)  

**Abstract**: Bladder cancer is one of the most prevalent malignancies worldwide, with a recurrence rate of up to 78%, necessitating accurate post-operative monitoring for effective patient management. Multi-sequence contrast-enhanced MRI is commonly used for recurrence detection; however, interpreting these scans remains challenging, even for experienced radiologists, due to post-surgical alterations such as scarring, swelling, and tissue remodeling. AI-assisted diagnostic tools have shown promise in improving bladder cancer recurrence prediction, yet progress in this field is hindered by the lack of dedicated multi-sequence MRI datasets for recurrence assessment study. In this work, we first introduce a curated multi-sequence, multi-modal MRI dataset specifically designed for bladder cancer recurrence prediction, establishing a valuable benchmark for future research. We then propose H-CNN-ViT, a new Hierarchical Gated Attention Multi-Branch model that enables selective weighting of features from the global (ViT) and local (CNN) paths based on contextual demands, achieving a balanced and targeted feature fusion. Our multi-branch architecture processes each modality independently, ensuring that the unique properties of each imaging channel are optimally captured and integrated. Evaluated on our dataset, H-CNN-ViT achieves an AUC of 78.6%, surpassing state-of-the-art models. Our model is publicly available at this https URL}. 

**Abstract (ZH)**: 膀胱癌是全球最常见的恶性肿瘤之一，复发率高达78%，需要准确的术后监测以有效管理患者。多序列对比增强MRI常用于检测复发；然而，即使是经验丰富的放射科医生也因手术后改变（如瘢痕、肿胀和组织重塑）而难以解读这些扫描图像。AI辅助诊断工具在改善膀胱癌复发预测方面显示出潜力，但该领域的进展受限于缺乏专门用于复发评估的多序列MRI数据集。在本工作中，我们首先引入了一个专门设计用于膀胱癌复发预测的多序列多模态MRI数据集，为未来的研究建立了有价值的基准。然后，我们提出了H-CNN-ViT，一种新的分层门控注意多分支模型，能够在基于上下文需求的全球（ViT）和局部（CNN）路径之间选择性地加权特征，实现平衡和有针对性的特征融合。我们的多分支架构独立处理每个模态，确保每个成像通道的独特属性被最优地捕获和集成。在我们的数据集上评估，H-CNN-ViT 达到了78.6%的AUC，超越了现有最先进的模型。我们的模型可在以下链接获得：this https URL。 

---
# Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection 

**Title (ZH)**: 时空对象意识视觉变换器在少样本视频对象检测中的应用 

**Authors**: Yogesh Kumar, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2511.13784)  

**Abstract**: Few-shot Video Object Detection (FSVOD) addresses the challenge of detecting novel objects in videos with limited labeled examples, overcoming the constraints of traditional detection methods that require extensive training data. This task presents key challenges, including maintaining temporal consistency across frames affected by occlusion and appearance variations, and achieving novel object generalization without relying on complex region proposals, which are often computationally expensive and require task-specific training. Our novel object-aware temporal modeling approach addresses these challenges by incorporating a filtering mechanism that selectively propagates high-confidence object features across frames. This enables efficient feature progression, reduces noise accumulation, and enhances detection accuracy in a few-shot setting. By utilizing few-shot trained detection and classification heads with focused feature propagation, we achieve robust temporal consistency without depending on explicit object tube proposals. Our approach achieves performance gains, with AP improvements of 3.7% (FSVOD-500), 5.3% (FSYTV-40), 4.3% (VidOR), and 4.5 (VidVRD) in the 5-shot setting. Further results demonstrate improvements in 1-shot, 3-shot, and 10-shot configurations. We make the code public at: this https URL 

**Abstract (ZH)**: 少量标注样本视频对象检测（Few-shot Video Object Detection, FSVOD）解决了有限标注样本下视频中新型对象检测的挑战，克服了传统检测方法对大量训练数据的依赖。该任务面临的关键挑战包括保持受遮挡和外观变化影响的帧间的一致性，以及在不依赖复杂区域提议的情况下实现新型对象的泛化。我们提出了一种新型的aware时空建模方法，通过引入一个筛选机制，选择性地传播高置信度的对象特征，从而实现高效的特征演进，减少噪声积累，并在少量标注样本设置中提升检测准确性。通过使用少量标注样本训练的检测和分类头，结合聚焦的特征传播，我们在不依赖显式的对象管预测的情况下实现了时空一致性。在5-shot设置下，我们的方法分别在FSVOD-500、FSYTV-40、VidOR和VidVRD上取得了3.7%、5.3%、4.3%和4.5%的mAP提升。进一步的结果表明，在1-shot、3-shot和10-shot配置下也取得了改进。我们已开源代码：this https URL。 

---
# VitalBench: A Rigorous Multi-Center Benchmark for Long-Term Vital Sign Prediction in Intraoperative Care 

**Title (ZH)**: VitalBench: 一项严格的多中心长期内镜护理生命体征预测基准 

**Authors**: Xiuding Cai, Xueyao Wang, Sen Wang, Yaoyao Zhu, Jiao Chen, Yu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.13757)  

**Abstract**: Intraoperative monitoring and prediction of vital signs are critical for ensuring patient safety and improving surgical outcomes. Despite recent advances in deep learning models for medical time-series forecasting, several challenges persist, including the lack of standardized benchmarks, incomplete data, and limited cross-center validation. To address these challenges, we introduce VitalBench, a novel benchmark specifically designed for intraoperative vital sign prediction. VitalBench includes data from over 4,000 surgeries across two independent medical centers, offering three evaluation tracks: complete data, incomplete data, and cross-center generalization. This framework reflects the real-world complexities of clinical practice, minimizing reliance on extensive preprocessing and incorporating masked loss techniques for robust and unbiased model evaluation. By providing a standardized and unified platform for model development and comparison, VitalBench enables researchers to focus on architectural innovation while ensuring consistency in data handling. This work lays the foundation for advancing predictive models for intraoperative vital sign forecasting, ensuring that these models are not only accurate but also robust and adaptable across diverse clinical environments. Our code and data are available at this https URL. 

**Abstract (ZH)**: intraoperative 监测和预测生命体征对于确保患者安全和提高手术效果至关重要。尽管最近在医疗时间序列预测的深度学习模型方面取得了进展，但仍存在标准化基准缺乏、数据不完整和中心间验证有限等挑战。为解决这些挑战，我们引入了 VitalBench，这是一种专门针对手术中生命体征预测的新基准。VitalBench 包括两个独立医疗中心超过 4,000 例手术的数据，提供三种评估轨道：完整数据、不完整数据和中心间泛化。该框架反映了临床实践中的现实复杂性，减少了对大量预处理的依赖，并结合了掩码损失技术以实现稳健和无偏的模型评估。通过提供一个标准化和统一的平台进行模型开发和比较，VitalBench 使研究人员能够专注于架构创新，同时确保数据处理的一致性。这项工作为推进手术中生命体征预测的预测模型奠定了基础，确保这些模型不仅准确，还能在多种临床环境中保持稳健和适应性。我们的代码和数据可在以下链接获取。 

---
