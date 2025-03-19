# Foundation Feature-Driven Online End-Effector Pose Estimation: A Marker-Free and Learning-Free Approach 

**Title (ZH)**: 基于基础特征的在线末端执行器姿态估计：一种无标志物且无学习的 Approach 

**Authors**: Tianshu Wu, Jiyao Zhang, Shiqian Liang, Zhengxiao Han, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.14051)  

**Abstract**: Accurate transformation estimation between camera space and robot space is essential. Traditional methods using markers for hand-eye calibration require offline image collection, limiting their suitability for online self-calibration. Recent learning-based robot pose estimation methods, while advancing online calibration, struggle with cross-robot generalization and require the robot to be fully visible. This work proposes a Foundation feature-driven online End-Effector Pose Estimation (FEEPE) algorithm, characterized by its training-free and cross end-effector generalization capabilities. Inspired by the zero-shot generalization capabilities of foundation models, FEEPE leverages pre-trained visual features to estimate 2D-3D correspondences derived from the CAD model and target image, enabling 6D pose estimation via the PnP algorithm. To resolve ambiguities from partial observations and symmetry, a multi-historical key frame enhanced pose optimization algorithm is introduced, utilizing temporal information for improved accuracy. Compared to traditional hand-eye calibration, FEEPE enables marker-free online calibration. Unlike robot pose estimation, it generalizes across robots and end-effectors in a training-free manner. Extensive experiments demonstrate its superior flexibility, generalization, and performance. 

**Abstract (ZH)**: 基于基础特征的在线末端执行器姿态估计（FEEPE）算法 

---
# Evaluating Global Geo-alignment for Precision Learned Autonomous Vehicle Localization using Aerial Data 

**Title (ZH)**: 基于航空数据的全球地理对齐方法在精准学习自动驾驶车辆定位中的评估 

**Authors**: Yi Yang, Xuran Zhao, H. Charles Zhao, Shumin Yuan, Samuel M. Bateman, Tiffany A. Huang, Chris Beall, Will Maddern  

**Link**: [PDF](https://arxiv.org/pdf/2503.13896)  

**Abstract**: Recently there has been growing interest in the use of aerial and satellite map data for autonomous vehicles, primarily due to its potential for significant cost reduction and enhanced scalability. Despite the advantages, aerial data also comes with challenges such as a sensor-modality gap and a viewpoint difference gap. Learned localization methods have shown promise for overcoming these challenges to provide precise metric localization for autonomous vehicles. Most learned localization methods rely on coarsely aligned ground truth, or implicit consistency-based methods to learn the localization task -- however, in this paper we find that improving the alignment between aerial data and autonomous vehicle sensor data at training time is critical to the performance of a learning-based localization system. We compare two data alignment methods using a factor graph framework and, using these methods, we then evaluate the effects of closely aligned ground truth on learned localization accuracy through ablation studies. Finally, we evaluate a learned localization system using the data alignment methods on a comprehensive (1600km) autonomous vehicle dataset and demonstrate localization error below 0.3m and 0.5$^{\circ}$ sufficient for autonomous vehicle applications. 

**Abstract (ZH)**: 近年来，人们越来越关注使用航空和卫星地图数据在自主车辆中的应用，主要得益于其在成本降低和扩展性增强方面的潜力。尽管有利之处众多，航空数据也带来了一些挑战，如传感器模态差距和视点差异差距。学习定位方法显示出克服这些挑战、为自主车辆提供精确的度量级定位的潜力。大多数学习定位方法依赖于粗略对齐的地面真实数据，或基于隐式一致性的方法来学习定位任务——然而，在本文中我们发现，在训练时提高航空数据与自主车辆传感器数据的对齐程度对于基于学习的定位系统性能至关重要。我们使用因子图框架比较了两种数据对齐方法，并通过消融研究评估紧密对齐的地面真实数据对学习定位准确性的影响。最后，我们在一个全面的（1600公里）自主车辆数据集上评估了使用数据对齐方法的学习定位系统，并展示了低于0.3米和0.5°的定位误差，足以满足自主车辆应用需求。 

---
# State Space Model Meets Transformer: A New Paradigm for 3D Object Detection 

**Title (ZH)**: 状态空间模型遇上变压器：三维物体检测的新范式 

**Authors**: Chuxin Wang, Wenfei Yang, Xiang Liu, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14493)  

**Abstract**: DETR-based methods, which use multi-layer transformer decoders to refine object queries iteratively, have shown promising performance in 3D indoor object detection. However, the scene point features in the transformer decoder remain fixed, leading to minimal contributions from later decoder layers, thereby limiting performance improvement. Recently, State Space Models (SSM) have shown efficient context modeling ability with linear complexity through iterative interactions between system states and inputs. Inspired by SSMs, we propose a new 3D object DEtection paradigm with an interactive STate space model (DEST). In the interactive SSM, we design a novel state-dependent SSM parameterization method that enables system states to effectively serve as queries in 3D indoor detection tasks. In addition, we introduce four key designs tailored to the characteristics of point cloud and SSM: The serialization and bidirectional scanning strategies enable bidirectional feature interaction among scene points within the SSM. The inter-state attention mechanism models the relationships between state points, while the gated feed-forward network enhances inter-channel correlations. To the best of our knowledge, this is the first method to model queries as system states and scene points as system inputs, which can simultaneously update scene point features and query features with linear complexity. Extensive experiments on two challenging datasets demonstrate the effectiveness of our DEST-based method. Our method improves the GroupFree baseline in terms of AP50 on ScanNet V2 (+5.3) and SUN RGB-D (+3.2) datasets. Based on the VDETR baseline, Our method sets a new SOTA on the ScanNetV2 and SUN RGB-D datasets. 

**Abstract (ZH)**: 基于DETR的方法通过迭代 refinement 对象查询，展示了在3D室内对象检测中的 promising 性能。然而，transformer 解码器中的场景点特征保持固定，导致后续解码层的贡献最小，从而限制了性能改进。受到State Space Models (SSM) 的启发，我们提出了一种新的交互式State Space Model (DEST) 基于的3D对象检测范式。在交互式SSM中，我们设计了一种新型的状态依赖SSM参数化方法，使得系统状态能够有效作为3D室内检测任务中的 queries。此外，我们提出了四种针对点云和SSM特性的关键设计：序列化和双向扫描策略在SSM中实现场景点间的双向特征交互。状态间注意力机制建模了状态点之间的关系，而门控前馈网络增强了通道间的相关性。据我们所知，这是首次将queries表示为系统状态并将场景点表示为系统输入的方法，同时以线性复杂度更新场景点特征和query特征。在两个具有挑战性的数据集上的广泛实验表明了我们DEST方法的有效性。我们的方法在ScanNet V2和SUN RGB-D数据集上相对于GroupFree基线方法在AP50上分别提高了5.3和3.2，基于VDETR基线，我们的方法在ScanNetV2和SUN RGB-D数据集上取得了新的SOTA结果。 

---
# DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers 

**Title (ZH)**: DiffMoE: 动态token选择以实现可扩展的扩散变换器 

**Authors**: Minglei Shi, Ziyang Yuan, Haotian Yang, Xintao Wang, Mingwu Zheng, Xin Tao, Wenliang Zhao, Wenzhao Zheng, Jie Zhou, Jiwen Lu, Pengfei Wan, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2503.14487)  

**Abstract**: Diffusion models have demonstrated remarkable success in various image generation tasks, but their performance is often limited by the uniform processing of inputs across varying conditions and noise levels. To address this limitation, we propose a novel approach that leverages the inherent heterogeneity of the diffusion process. Our method, DiffMoE, introduces a batch-level global token pool that enables experts to access global token distributions during training, promoting specialized expert behavior. To unleash the full potential of the diffusion process, DiffMoE incorporates a capacity predictor that dynamically allocates computational resources based on noise levels and sample complexity. Through comprehensive evaluation, DiffMoE achieves state-of-the-art performance among diffusion models on ImageNet benchmark, substantially outperforming both dense architectures with 3x activated parameters and existing MoE approaches while maintaining 1x activated parameters. The effectiveness of our approach extends beyond class-conditional generation to more challenging tasks such as text-to-image generation, demonstrating its broad applicability across different diffusion model applications. Project Page: this https URL 

**Abstract (ZH)**: 扩散模型在各种图像生成任务中展现了显著的成功，但其性能往往受限于在不同条件和噪声水平下输入的均匀处理。为应对这一局限，我们提出了一种新颖的方法，利用扩散过程内在的异质性。我们的方法，DiffMoE，引入了批次级全局令牌池，使专家能够在训练过程中访问全局令牌分布，促进专家的专业化行为。为了充分发挥扩散过程的潜力，DiffMoE 结合了容量预测器，根据噪声水平和样本复杂性动态分配计算资源。通过全面评估，DiffMoE 在 ImageNet 基准测试中达到了扩散模型的最新性能，与具有 3 倍激活参数的密集架构相比，显著超越了现有 MoE 方法，同时保持相同的激活参数。我们的方法的有效性不仅限于类别条件生成，还扩展到更具挑战性的任务如文本到图像生成，展示了其在不同扩散模型应用中的广泛适用性。项目页面：https://this-url 

---
# MagicComp: Training-free Dual-Phase Refinement for Compositional Video Generation 

**Title (ZH)**: MagicComp：无需训练的双阶段精炼方法用于组合视频生成 

**Authors**: Hongyu Zhang, Yufan Deng, Shenghai Yuan, Peng Jin, Zesen Cheng, Yian Zhao, Chang Liu, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.14428)  

**Abstract**: Text-to-video (T2V) generation has made significant strides with diffusion models. However, existing methods still struggle with accurately binding attributes, determining spatial relationships, and capturing complex action interactions between multiple subjects. To address these limitations, we propose MagicComp, a training-free method that enhances compositional T2V generation through dual-phase refinement. Specifically, (1) During the Conditioning Stage: We introduce the Semantic Anchor Disambiguation to reinforces subject-specific semantics and resolve inter-subject ambiguity by progressively injecting the directional vectors of semantic anchors into original text embedding; (2) During the Denoising Stage: We propose Dynamic Layout Fusion Attention, which integrates grounding priors and model-adaptive spatial perception to flexibly bind subjects to their spatiotemporal regions through masked attention modulation. Furthermore, MagicComp is a model-agnostic and versatile approach, which can be seamlessly integrated into existing T2V architectures. Extensive experiments on T2V-CompBench and VBench demonstrate that MagicComp outperforms state-of-the-art methods, highlighting its potential for applications such as complex prompt-based and trajectory-controllable video generation. Project page: this https URL. 

**Abstract (ZH)**: 无需生成标题，以下是翻译的内容：

Text-to-video (T2V) 生成借助扩散模型取得了显著进展。然而，现有方法仍然在准确绑定属性、确定空间关系以及捕捉多个主体之间的复杂动作交互方面面临挑战。为了解决这些局限，我们提出了一种名为 MagicComp 的无训练方法，通过双阶段细化来增强组合式的 T2V 生成。具体来说，（1）在条件化阶段：我们引入语义锚点去模糊处理，通过逐步注入语义锚点的方向向量到原始文本嵌入中，以强化主体特定的语义并解决跨主体的模糊性；（2）在除噪阶段：我们提出动态布局融合注意力，结合语义先验和模型自适应的空间感知，通过掩模注意力调节灵活绑定主体到其时空区域。此外，MagicComp 是一个模型无关且通用的方法，可以无缝集成到现有的 T2V 架构中。在 T2V-CompBench 和 VBench 的广泛实验中，MagicComp 显示出优于现有最佳方法的性能，突显其在复杂提示驱动的和轨迹可控的视频生成等应用中的潜力。项目页面：这个 https URL。 

---
# ExDDV: A New Dataset for Explainable Deepfake Detection in Video 

**Title (ZH)**: ExDDV：一个用于可解释的视频深fake检测的新数据集 

**Authors**: Vlad Hondru, Eduard Hogea, Darian Onchis, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14421)  

**Abstract**: The ever growing realism and quality of generated videos makes it increasingly harder for humans to spot deepfake content, who need to rely more and more on automatic deepfake detectors. However, deepfake detectors are also prone to errors, and their decisions are not explainable, leaving humans vulnerable to deepfake-based fraud and misinformation. To this end, we introduce ExDDV, the first dataset and benchmark for Explainable Deepfake Detection in Video. ExDDV comprises around 5.4K real and deepfake videos that are manually annotated with text descriptions (to explain the artifacts) and clicks (to point out the artifacts). We evaluate a number of vision-language models on ExDDV, performing experiments with various fine-tuning and in-context learning strategies. Our results show that text and click supervision are both required to develop robust explainable models for deepfake videos, which are able to localize and describe the observed artifacts. Our novel dataset and code to reproduce the results are available at this https URL. 

**Abstract (ZH)**: 随生成视频的不断增多，其真实度和质量不断提高，使得人类越来越难以辨别深度伪造内容，从而更多地依赖于自动深度伪造检测器。然而，深度伪造检测器也容易出错，其决策过程不可解释，使得人类容易受到深度伪造欺诈和 misinformation 的影响。为此，我们引入了 ExDDV，这是首个用于视频可解释深度伪造检测的数据集和基准。ExDDV 包含约 5.4K 条真实和深度伪造视频，并手工标注了文本描述（解释伪影）和点击（指出伪影）。我们评估了多种视觉-语言模型在 ExDDV 上的表现，进行了各种微调和上下文学习策略的实验。我们的结果表明，文本和点击监督对于开发适用于深度伪造视频的 robust 可解释模型都是必要的，能够定位并描述观察到的伪影。我们的新型数据集及复现结果的代码可在以下链接获取。 

---
# VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation 

**Title (ZH)**: VEGGIE: 基于语义生成的地基编辑与推理视频概念 

**Authors**: Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.14350)  

**Abstract**: Recent video diffusion models have enhanced video editing, but it remains challenging to handle instructional editing and diverse tasks (e.g., adding, removing, changing) within a unified framework. In this paper, we introduce VEGGIE, a Video Editor with Grounded Generation from Instructions, a simple end-to-end framework that unifies video concept editing, grounding, and reasoning based on diverse user instructions. Specifically, given a video and text query, VEGGIE first utilizes an MLLM to interpret user intentions in instructions and ground them to the video contexts, generating frame-specific grounded task queries for pixel-space responses. A diffusion model then renders these plans and generates edited videos that align with user intent. To support diverse tasks and complex instructions, we employ a curriculum learning strategy: first aligning the MLLM and video diffusion model with large-scale instructional image editing data, followed by end-to-end fine-tuning on high-quality multitask video data. Additionally, we introduce a novel data synthesis pipeline to generate paired instructional video editing data for model training. It transforms static image data into diverse, high-quality video editing samples by leveraging Image-to-Video models to inject dynamics. VEGGIE shows strong performance in instructional video editing with different editing skills, outperforming the best instructional baseline as a versatile model, while other models struggle with multi-tasking. VEGGIE also excels in video object grounding and reasoning segmentation, where other baselines fail. We further reveal how the multiple tasks help each other and highlight promising applications like zero-shot multimodal instructional and in-context video editing. 

**Abstract (ZH)**: Recent视频扩散模型提升了视频编辑能力，但如何在统一框架内处理指导性编辑和多样化任务（例如添加、删除、修改）依然是一个挑战。本文介绍了一种新的Video Editor with Grounded Generation from Instructions（VEGGIE），它提供了一个简单的一站式框架，结合视频概念编辑、语义接地和基于多样用户指令的推理。具体来说，给定一个视频和文本查询，VEGGIE首先利用一个MLLM解析指令中的用户意图，并将其与视频内容相关联，生成帧特定的语义接地任务查询以供像素级响应。随后，扩散模型根据这些计划生成符合用户意图的编辑视频。为了支持多样化的任务和复杂的指令，我们采用了渐进式学习策略：首先用大规模的指导性图像编辑数据优化MLLM和视频扩散模型，然后使用高质量的多任务视频数据进行端到端微调。此外，我们还引入了一种新的数据合成流程来生成用于模型训练的配对指导性视频编辑数据。该流程通过利用图像到视频模型引入动态性，将静态图像数据转换为多样化的高质量视频编辑样本。VEGGIE在不同编辑技能的指导性视频编辑任务中表现出强大性能，作为一款多功能模型，其性能超过了现有的最佳指导性基线，而其他模型在多任务处理上则存在困难。VEGGIE在视频对象语义接地和推理分割方面也表现出色，而其他基线模型在此任务上失效。我们进一步揭示了多种任务之间的相互帮助关系，并强调了零样本多模态指导性编辑和上下文相关视频编辑等有前途的应用。 

---
# PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation 

**Title (ZH)**: PC-Talk: 音频驱动的精确面部动画控制与生成 

**Authors**: Baiqin Wang, Xiangyu Zhu, Fan Shen, Hao Xu, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.14295)  

**Abstract**: Recent advancements in audio-driven talking face generation have made great progress in lip synchronization. However, current methods often lack sufficient control over facial animation such as speaking style and emotional expression, resulting in uniform outputs. In this paper, we focus on improving two key factors: lip-audio alignment and emotion control, to enhance the diversity and user-friendliness of talking videos. Lip-audio alignment control focuses on elements like speaking style and the scale of lip movements, whereas emotion control is centered on generating realistic emotional expressions, allowing for modifications in multiple attributes such as intensity. To achieve precise control of facial animation, we propose a novel framework, PC-Talk, which enables lip-audio alignment and emotion control through implicit keypoint deformations. First, our lip-audio alignment control module facilitates precise editing of speaking styles at the word level and adjusts lip movement scales to simulate varying vocal loudness levels, maintaining lip synchronization with the audio. Second, our emotion control module generates vivid emotional facial features with pure emotional deformation. This module also enables the fine modification of intensity and the combination of multiple emotions across different facial regions. Our method demonstrates outstanding control capabilities and achieves state-of-the-art performance on both HDTF and MEAD datasets in extensive experiments. 

**Abstract (ZH)**: 基于音频驱动的说话人脸生成 recent 进展在唇部同步方面取得了显著进步。然而，当前方法在面部动画控制，如发音风格和情感表达方面往往不够充分，导致输出结果不够多样。本文旨在通过改进唇音对齐和情感控制两个关键因素，提高说话视频的多样性和用户友好性。唇音对齐控制侧重于发音风格和唇部动作的规模等元素，而情感控制则侧重于生成逼真的情感表达，并能在多个属性如强度上进行修改。为实现对面部动画的精确控制，我们提出了一种新型框架 PC-Talk，通过隐式关键点变形实现唇音对齐和情感控制。首先，我们的唇音对齐控制模块在词级上实现精细的发音风格编辑，并调整唇部动作规模以模拟不同音量水平，保持唇部与音频同步。其次，我们的情感控制模块通过纯情感变形生成生动的情感面部特征，该模块还能够在不同面部区域实现多种情感的微调与组合。实验结果表明，我们的方法具有出色的控制能力和在 HDTF 和 MEAD 数据集上实现了最先进的性能。 

---
# Manual Labelling Artificially Inflates Deep Learning-Based Segmentation Performance on Closed Canopy: Validation Using TLS 

**Title (ZH)**: 人工标注 Artificially Inflates 深学习基础的闭合冠层分割性能：基于 TLS 的验证 

**Authors**: Matthew J. Allen, Harry J. F. Owen, Stuart W. D. Grieve, Emily R. Lines  

**Link**: [PDF](https://arxiv.org/pdf/2503.14273)  

**Abstract**: Monitoring forest dynamics at an individual tree scale is essential for accurately assessing ecosystem responses to climate change, yet traditional methods relying on field-based forest inventories are labor-intensive and limited in spatial coverage. Advances in remote sensing using drone-acquired RGB imagery combined with deep learning models have promised precise individual tree crown (ITC) segmentation; however, existing methods are frequently validated against human-annotated images, lacking rigorous independent ground truth. In this study, we generate high-fidelity validation labels from co-located Terrestrial Laser Scanning (TLS) data for drone imagery of mixed unmanaged boreal and Mediterranean forests. We evaluate the performance of two widely used deep learning ITC segmentation models - DeepForest (RetinaNet) and Detectree2 (Mask R-CNN) - on these data, and compare to performance on further Mediterranean forest data labelled manually. When validated against TLS-derived ground truth from Mediterranean forests, model performance decreased significantly compared to assessment based on hand-labelled from an ecologically similar site (AP50: 0.094 vs. 0.670). Restricting evaluation to only canopy trees shrank this gap considerably (Canopy AP50: 0.365), although performance was still far lower than on similar hand-labelled data. Models also performed poorly on boreal forest data (AP50: 0.142), although again increasing when evaluated on canopy trees only (Canopy AP50: 0.308). Both models showed very poor localisation accuracy at stricter IoU thresholds, even when restricted to canopy trees (Max AP75: 0.051). Similar results have been observed in studies using aerial LiDAR data, suggesting fundamental limitations in aerial-based segmentation approaches in closed canopy forests. 

**Abstract (ZH)**: 利用无人机RGB影像和深度学习模型从混合未管理的 boreal 和地中海森林无人机影像中生成高度忠实的验证标签，评估 DeepForest (RetinaNet) 和 Detectree2 (Mask R-CNN) 的单木冠层分割性能 

---
# Panoramic Distortion-Aware Tokenization for Person Detection and Localization Using Transformers in Overhead Fisheye Images 

**Title (ZH)**: 全景畸变感知的token化方法在Overhead鱼缸镜头图像中的人体检测与定位中应用Transformer技术 

**Authors**: Nobuhiko Wakai, Satoshi Sato, Yasunori Ishii, Takayoshi Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2503.14228)  

**Abstract**: Person detection methods are used widely in applications including visual surveillance, pedestrian detection, and robotics. However, accurate detection of persons from overhead fisheye images remains an open challenge because of factors including person rotation and small-sized persons. To address the person rotation problem, we convert the fisheye images into panoramic images. For smaller people, we focused on the geometry of the panoramas. Conventional detection methods tend to focus on larger people because these larger people yield large significant areas for feature maps. In equirectangular panoramic images, we find that a person's height decreases linearly near the top of the images. Using this finding, we leverage the significance values and aggregate tokens that are sorted based on these values to balance the significant areas. In this leveraging process, we introduce panoramic distortion-aware tokenization. This tokenization procedure divides a panoramic image using self-similarity figures that enable determination of optimal divisions without gaps, and we leverage the maximum significant values in each tile of token groups to preserve the significant areas of smaller people. To achieve higher detection accuracy, we propose a person detection and localization method that combines panoramic-image remapping and the tokenization procedure. Extensive experiments demonstrated that our method outperforms conventional methods when applied to large-scale datasets. 

**Abstract (ZH)**: 基于全景图的人体检测方法在视屏监控、行人检测和机器人等领域得到广泛应用，但由于因素包括人体旋转和小尺寸人体的影响，从鱼眼图像中准确检测人体仍是一个开放挑战。为解决人体旋转问题，我们将鱼眼图像转换为全景图像。对于小尺寸的人体，我们关注全景图的几何特性。传统的检测方法通常关注较大的人体，因为较大的人体在特征图上提供了较大的显著区域。在等角正圆柱投影全景图像中，我们发现人体的高度在图像顶部附近呈线性减小。利用这一发现，我们利用显著值并基于这些值对排序的标记进行聚合，以平衡显著区域。在这一过程中，我们引入了全景图畸变感知标记化方法。这种方法使用自我相似图形分割全景图像，以确定没有间隙的最佳分割，并利用每个标记组中的最大显著值来保留小尺寸人体的显著区域。为提高检测准确性，我们提出了一种结合全景图重新映射和标记化过程的人体检测与定位方法。广泛实验表明，当应用于大规模数据集时，我们的方法优于传统方法。 

---
# Concat-ID: Towards Universal Identity-Preserving Video Synthesis 

**Title (ZH)**: Concat-ID: 向量通用身份保留视频合成 

**Authors**: Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.14151)  

**Abstract**: We present Concat-ID, a unified framework for identity-preserving video generation. Concat-ID employs Variational Autoencoders to extract image features, which are concatenated with video latents along the sequence dimension, leveraging solely 3D self-attention mechanisms without the need for additional modules. A novel cross-video pairing strategy and a multi-stage training regimen are introduced to balance identity consistency and facial editability while enhancing video naturalness. Extensive experiments demonstrate Concat-ID's superiority over existing methods in both single and multi-identity generation, as well as its seamless scalability to multi-subject scenarios, including virtual try-on and background-controllable generation. Concat-ID establishes a new benchmark for identity-preserving video synthesis, providing a versatile and scalable solution for a wide range of applications. 

**Abstract (ZH)**: 我们提出Concat-ID，一种统一的身份保留视频生成框架。Concat-ID 使用变分自编码器提取图像特征，将这些特征与视频潜在变量沿序列维度拼接，仅依靠3D 自注意力机制，无需额外模块。引入了一种新颖的跨视频配对策略和多阶段训练 regimen，以平衡身份一致性与面部可编辑性，同时提升视频的自然度。广泛的实验表明，Concat-ID 在单身份和多身份生成方面均优于现有方法，并且能够无缝扩展到多主体场景，包括虚拟试穿和背景可控生成。Concat-ID 建立了身份保留视频合成的新基准，提供了适用于广泛应用场景的灵活且可扩展的解决方案。 

---
# Exploring Disparity-Accuracy Trade-offs in Face Recognition Systems: The Role of Datasets, Architectures, and Loss Functions 

**Title (ZH)**: 探索面部识别系统中准确率与差距之间的权衡关系：数据集、架构和损失函数的作用 

**Authors**: Siddharth D Jaiswal, Sagnik Basu, Sandipan Sikdar, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.14138)  

**Abstract**: Automated Face Recognition Systems (FRSs), developed using deep learning models, are deployed worldwide for identity verification and facial attribute analysis. The performance of these models is determined by a complex interdependence among the model architecture, optimization/loss function and datasets. Although FRSs have surpassed human-level accuracy, they continue to be disparate against certain demographics. Due to the ubiquity of applications, it is extremely important to understand the impact of the three components -- model architecture, loss function and face image dataset on the accuracy-disparity trade-off to design better, unbiased platforms. In this work, we perform an in-depth analysis of three FRSs for the task of gender prediction, with various architectural modifications resulting in ten deep-learning models coupled with four loss functions and benchmark them on seven face datasets across 266 evaluation configurations. Our results show that all three components have an individual as well as a combined impact on both accuracy and disparity. We identify that datasets have an inherent property that causes them to perform similarly across models, independent of the choice of loss functions. Moreover, the choice of dataset determines the model's perceived bias -- the same model reports bias in opposite directions for three gender-balanced datasets of ``in-the-wild'' face images of popular individuals. Studying the facial embeddings shows that the models are unable to generalize a uniform definition of what constitutes a ``female face'' as opposed to a ``male face'', due to dataset diversity. We provide recommendations to model developers on using our study as a blueprint for model development and subsequent deployment. 

**Abstract (ZH)**: 基于深度学习的自动面部识别系统（FRSs）已在全球范围内用于身份验证和面部属性分析。这些模型的表现受模型架构、优化/损失函数和数据集之间复杂相互依赖关系的影响。尽管FRSs已超越了人类水平的准确性，但仍呔在某些 demographic 上存在差异。由于应用的普遍性，了解模型架构、损失函数和面部图像数据集对准确性和差异性权衡的影响至关重要，以便设计更好且无偏见的平台。在本工作中，我们对三个用于性别预测任务的FRSs进行了深入分析，通过各种架构修改，得到十种深度学习模型并结合四种损失函数，在七个面部数据集上的266种评估配置上进行基准测试。我们的结果表明，所有三个组件分别及联合地影响准确性和差异性。我们发现数据集具有固有的特性，使其在不同模型中表现相似，独立于所选择的损失函数。此外，数据集的选择决定了模型感知的偏见——对于三个性别平衡的数据集，同一模型在“在野”名人面部图像中报告的偏见方向相反。通过对面部嵌入的分析，我们发现模型无法普遍定义何为“女性面孔”或“男性面孔”，这是由于数据集多样性。我们为模型开发人员提供了建议，利用我们的研究作为模型开发和后续部署的蓝图。 

---
# Fast Autoregressive Video Generation with Diagonal Decoding 

**Title (ZH)**: 快速自回归视频生成：对角解码方法 

**Authors**: Yang Ye, Junliang Guo, Haoyu Wu, Tianyu He, Tim Pearce, Tabish Rashid, Katja Hofmann, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2503.14070)  

**Abstract**: Autoregressive Transformer models have demonstrated impressive performance in video generation, but their sequential token-by-token decoding process poses a major bottleneck, particularly for long videos represented by tens of thousands of tokens. In this paper, we propose Diagonal Decoding (DiagD), a training-free inference acceleration algorithm for autoregressively pre-trained models that exploits spatial and temporal correlations in videos. Our method generates tokens along diagonal paths in the spatial-temporal token grid, enabling parallel decoding within each frame as well as partially overlapping across consecutive frames. The proposed algorithm is versatile and adaptive to various generative models and tasks, while providing flexible control over the trade-off between inference speed and visual quality. Furthermore, we propose a cost-effective finetuning strategy that aligns the attention patterns of the model with our decoding order, further mitigating the training-inference gap on small-scale models. Experiments on multiple autoregressive video generation models and datasets demonstrate that DiagD achieves up to $10\times$ speedup compared to naive sequential decoding, while maintaining comparable visual fidelity. 

**Abstract (ZH)**: 自回归Transformer模型在视频生成中展现了 impressive 的性能，但其依次解码词元的过程成为主要瓶颈，特别是对于由数万个词元表示的长视频。本文提出了一种名为 Diagonal Decoding (DiagD) 的无需训练的推理加速算法，该算法利用视频中的空间和时间相关性。该方法沿着空间-时间词元网格的对角路径生成词元，允许每帧内并行解码以及连续帧之间的部分重叠解码。所提算法具有灵活性和适应性，适用于各种生成模型和任务，同时提供了在推理速度与视觉质量之间进行灵活权衡的控制。此外，本文提出了一种经济有效的微调策略，以使模型的注意力模式与解码顺序对齐，进一步缓解小型模型上的训练-推理差距。在多种自回归视频生成模型和数据集上的实验表明，DiagD 相较于简单的依次解码可实现高达 $10\times$ 的加速，同时保持相当的视觉保真度。 

---
# Beyond holography: the entropic quantum gravity foundations of image processing 

**Title (ZH)**: 超越全息图：图像处理的熵量子引力基础 

**Authors**: Ginestra Bianconi  

**Link**: [PDF](https://arxiv.org/pdf/2503.14048)  

**Abstract**: Recently, thanks to the development of artificial intelligence (AI) there is increasing scientific attention to establishing the connections between theoretical physics and AI. Traditionally, these connections have been focusing mostly on the relation between string theory and image processing and involve important theoretical paradigms such as holography. Recently G. Bianconi has proposed the entropic quantum gravity approach that proposes an action for gravity given by the quantum relative entropy between the metrics associated to a manifold. Here it is demonstrated that the famous Perona-Malik algorithm for image processing is the gradient flow of the entropic quantum gravity action. These results provide the geometrical and information theory foundations for the Perona-Malik algorithm and open new avenues for establishing fundamental relations between brain research, machine learning and entropic quantum gravity. 

**Abstract (ZH)**: 近期，由于人工智能（AI）的发展，越来越多的科学关注点转向建立理论物理与AI之间的联系。传统上，这些联系主要集中在弦理论与图像处理之间的关系上，并涉及重要的理论范式如反面光学。最近，G. Bianconi 提出了熵量子引力方法，该方法给出了由流形相关度规的量子相对熵组成的引力作用。研究表明，著名的Perona-Malik图像处理算法是熵量子引力作用的梯度流。这些结果为Perona-Malik算法提供了几何和信息论基础，并开启了在脑科学研究、机器学习和熵量子引力之间建立根本关系的新途径。 

---
# Boosting Semi-Supervised Medical Image Segmentation via Masked Image Consistency and Discrepancy Learning 

**Title (ZH)**: 基于掩码图像一致性与差异性学习的增强半监督医疗图像分割 

**Authors**: Pengcheng Zhou, Lantian Zhang, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.14013)  

**Abstract**: Semi-supervised learning is of great significance in medical image segmentation by exploiting unlabeled data. Among its strategies, the co-training framework is prominent. However, previous co-training studies predominantly concentrate on network initialization variances and pseudo-label generation, while overlooking the equilibrium between information interchange and model diversity preservation. In this paper, we propose the Masked Image Consistency and Discrepancy Learning (MICD) framework with three key modules. The Masked Cross Pseudo Consistency (MCPC) module enriches context perception and small sample learning via pseudo-labeling across masked-input branches. The Cross Feature Consistency (CFC) module fortifies information exchange and model robustness by ensuring decoder feature consistency. The Cross Model Discrepancy (CMD) module utilizes EMA teacher networks to oversee outputs and preserve branch diversity. Together, these modules address existing limitations by focusing on fine-grained local information and maintaining diversity in a heterogeneous framework. Experiments on two public medical image datasets, AMOS and Synapse, demonstrate that our approach outperforms state-of-the-art methods. 

**Abstract (ZH)**: 半监督学习在医学图像分割中具有重要意义，通过利用未标记数据。其中，共训练框架尤为重要。然而，以往共训练研究主要集中在网络初始化差异和伪标签生成上，忽视了信息交互与模型多样性保持之间的平衡。本文提出了一种带有三个关键模块的Masked Image Consistency and Discrepancy Learning (MICD)框架。Masked Cross Pseudo Consistency (MCPC)模块通过跨掩码输入分支的伪标签增强上下文感知和小样本学习。Cross Feature Consistency (CFC)模块通过确保解码器特征一致性来增强信息交互和模型鲁棒性。Cross Model Discrepancy (CMD)模块利用EMA教师网络监管输出并保持分支多样性。这些模块共同解决了现有局限性，特别是在细粒度局部信息和异构框架中保持多样性方面。在两个公开的医学图像数据集AMOS和Synapse上的实验表明，我们的方法优于现有最佳方法。 

---
# MeshFleet: Filtered and Annotated 3D Vehicle Dataset for Domain Specific Generative Modeling 

**Title (ZH)**: MeshFleet: 经过筛选和注释的3D车辆数据集，用于领域特定生成模型 

**Authors**: Damian Boborzi, Phillip Mueller, Jonas Emrich, Dominik Schmid, Sebastian Mueller, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2503.14002)  

**Abstract**: Generative models have recently made remarkable progress in the field of 3D objects. However, their practical application in fields like engineering remains limited since they fail to deliver the accuracy, quality, and controllability needed for domain-specific tasks. Fine-tuning large generative models is a promising perspective for making these models available in these fields. Creating high-quality, domain-specific 3D datasets is crucial for fine-tuning large generative models, yet the data filtering and annotation process remains a significant bottleneck. We present MeshFleet, a filtered and annotated 3D vehicle dataset extracted from Objaverse-XL, the most extensive publicly available collection of 3D objects. Our approach proposes a pipeline for automated data filtering based on a quality classifier. This classifier is trained on a manually labeled subset of Objaverse, incorporating DINOv2 and SigLIP embeddings, refined through caption-based analysis and uncertainty estimation. We demonstrate the efficacy of our filtering method through a comparative analysis against caption and image aesthetic score-based techniques and fine-tuning experiments with SV3D, highlighting the importance of targeted data selection for domain-specific 3D generative modeling. 

**Abstract (ZH)**: 生成模型在3D物体领域的 Recent 进展已取得显著成果，但在如工程等领域的实际应用仍受到限制，因为它们无法满足特定领域任务所需的准确度、质量和可控性。针对大型生成模型的微调是使这些模型在这些领域可用的一个有前景的方向。创建高质量的专业特定3D数据集对于微调大型生成模型至关重要，但数据过滤和标注过程仍然是一个重要的瓶颈。我们提出MeshFleet，这是一个从Objaverse-XL提取的过滤和标注的3D车辆数据集，Objaverse-XL是目前已知最大的公开3D对象集合。我们的方法提出了一种基于质量分类器的自动化数据过滤管道。该分类器在手工标注的Objaverse子集上训练，结合了DINOv2和SigLIP嵌入，并通过基于图 caption 的分析和不确定性估计进行了细化。我们通过与基于图 caption 和图像美学评分的技术的比较分析以及SV3D的微调实验，展示了我们过滤方法的有效性，突出了针对特定领域3D生成建模的重要性。 

---
# GraphTEN: Graph Enhanced Texture Encoding Network 

**Title (ZH)**: 图增强纹理编码网络 

**Authors**: Bo Peng, Jintao Chen, Mufeng Yao, Chenhao Zhang, Jianghui Zhang, Mingmin Chi, Jiang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.13991)  

**Abstract**: Texture recognition is a fundamental problem in computer vision and pattern recognition. Recent progress leverages feature aggregation into discriminative descriptions based on convolutional neural networks (CNNs). However, modeling non-local context relations through visual primitives remains challenging due to the variability and randomness of texture primitives in spatial distributions. In this paper, we propose a graph-enhanced texture encoding network (GraphTEN) designed to capture both local and global features of texture primitives. GraphTEN models global associations through fully connected graphs and captures cross-scale dependencies of texture primitives via bipartite graphs. Additionally, we introduce a patch encoding module that utilizes a codebook to achieve an orderless representation of texture by encoding multi-scale patch features into a unified feature space. The proposed GraphTEN achieves superior performance compared to state-of-the-art methods across five publicly available datasets. 

**Abstract (ZH)**: 纹理识别是计算机视觉和模式识别中的一个基础问题。Recent progress leverages feature aggregation into discriminative descriptions based on convolutional neural networks (CNNs)。一种图增强的纹理编码网络（GraphTEN）被提出，旨在捕捉纹理 primitives 的局部和全局特征。GraphTEN 通过全连接图建模全局关联，并通过二分图捕捉纹理 primitives 的跨尺度依赖关系。此外，我们引入了一个补丁编码模块，利用码本将多尺度补丁特征编码到统一的特征空间中，实现无序的纹理表示。实验结果表明，所提出的 GraphTEN 在五个公开数据集上优于现有方法。 

---
# DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection 

**Title (ZH)**: DefectFill：基于 inpainting 扩散模型的视觉检测真实缺陷生成 

**Authors**: Jaewoo Song, Daemin Park, Kanghyun Baek, Sangyub Lee, Jooyoung Choi, Eunji Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2503.13985)  

**Abstract**: Developing effective visual inspection models remains challenging due to the scarcity of defect data. While image generation models have been used to synthesize defect images, producing highly realistic defects remains difficult. We propose DefectFill, a novel method for realistic defect generation that requires only a few reference defect images. It leverages a fine-tuned inpainting diffusion model, optimized with our custom loss functions incorporating defect, object, and attention terms. It enables precise capture of detailed, localized defect features and their seamless integration into defect-free objects. Additionally, our Low-Fidelity Selection method further enhances the defect sample quality. Experiments show that DefectFill generates high-quality defect images, enabling visual inspection models to achieve state-of-the-art performance on the MVTec AD dataset. 

**Abstract (ZH)**: 由于缺陷数据稀缺，开发有效的视觉检测模型仍然具有挑战性。尽管使用图像生成模型可以合成缺陷图像，但生成高度真实的缺陷仍然困难重重。我们提出了一种名为DefectFill的新方法，仅需少量参考缺陷图像即可实现逼真的缺陷生成。该方法利用了经过微调的 inpainting 扩散模型，并结合了我们自定义的包含缺陷、对象和注意力项的损失函数。它能够精确捕捉详细的局部缺陷特征，并使这些特征无缝地集成到无缺陷的对象中。此外，我们的低保真度选择方法进一步提高了缺陷样本的质量。实验结果显示，DefectFill 生成了高质量的缺陷图像，使视觉检测模型在 MVTec AD 数据集中达到了最先进的性能。 

---
# FrustumFusionNets: A Three-Dimensional Object Detection Network Based on Tractor Road Scene 

**Title (ZH)**: 视场融合网络：基于拖拉机道路场景的三维物体检测网络 

**Authors**: Lili Yang, Mengshuai Chang, Xiao Guo, Yuxin Feng, Yiwen Mei, Caicong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13951)  

**Abstract**: To address the issues of the existing frustum-based methods' underutilization of image information in road three-dimensional object detection as well as the lack of research on agricultural scenes, we constructed an object detection dataset using an 80-line Light Detection And Ranging (LiDAR) and a camera in a complex tractor road scene and proposed a new network called FrustumFusionNets (FFNets). Initially, we utilize the results of image-based two-dimensional object detection to narrow down the search region in the three-dimensional space of the point cloud. Next, we introduce a Gaussian mask to enhance the point cloud information. Then, we extract the features from the frustum point cloud and the crop image using the point cloud feature extraction pipeline and the image feature extraction pipeline, respectively. Finally, we concatenate and fuse the data features from both modalities to achieve three-dimensional object detection. Experiments demonstrate that on the constructed test set of tractor road data, the FrustumFusionNetv2 achieves 82.28% and 95.68% accuracy in the three-dimensional object detection of the two main road objects, cars and people, respectively. This performance is 1.83% and 2.33% better than the original model. It offers a hybrid fusion-based multi-object, high-precision, real-time three-dimensional object detection technique for unmanned agricultural machines in tractor road scenarios. On the Karlsruhe Institute of Technology and Toyota Technological Institute (KITTI) Benchmark Suite validation set, the FrustumFusionNetv2 also demonstrates significant superiority in detecting road pedestrian objects compared with other frustum-based three-dimensional object detection methods. 

**Abstract (ZH)**: 基于锥体融合网络的农业场景道路三维物体检测方法 

---
# ChatBEV: A Visual Language Model that Understands BEV Maps 

**Title (ZH)**: ChatBEV: 一种理解鸟瞰图的视觉语言模型 

**Authors**: Qingyao Xu, Siheng Chen, Guang Chen, Yanfeng Wang, Ya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13938)  

**Abstract**: Traffic scene understanding is essential for intelligent transportation systems and autonomous driving, ensuring safe and efficient vehicle operation. While recent advancements in VLMs have shown promise for holistic scene understanding, the application of VLMs to traffic scenarios, particularly using BEV maps, remains under explored. Existing methods often suffer from limited task design and narrow data amount, hindering comprehensive scene understanding. To address these challenges, we introduce ChatBEV-QA, a novel BEV VQA benchmark contains over 137k questions, designed to encompass a wide range of scene understanding tasks, including global scene understanding, vehicle-lane interactions, and vehicle-vehicle interactions. This benchmark is constructed using an novel data collection pipeline that generates scalable and informative VQA data for BEV maps. We further fine-tune a specialized vision-language model ChatBEV, enabling it to interpret diverse question prompts and extract relevant context-aware information from BEV maps. Additionally, we propose a language-driven traffic scene generation pipeline, where ChatBEV facilitates map understanding and text-aligned navigation guidance, significantly enhancing the generation of realistic and consistent traffic scenarios. The dataset, code and the fine-tuned model will be released. 

**Abstract (ZH)**: 交通场景理解对于智能交通系统和自动驾驶至关重要，确保车辆安全高效运行。尽管近期基于VLMs的整体场景理解显示出潜力，但VLMs在交通场景中的应用，特别是使用BEV地图的应用，仍鲜有探索。现有方法往往受限于任务设计有限和数据量狭窄，阻碍了全面的场景理解。为了应对这些挑战，我们引入了ChatBEV-QA这一新的BEV VQA基准，包含超过137k个问题，旨在涵盖广泛的场景理解任务，包括全局场景理解、车辆-车道交互和车辆-车辆交互。该基准通过一种新型数据收集流程构建，生成适用于BEV地图的大规模和信息丰富的VQA数据。我们进一步对专门的视觉-语言模型ChatBEV进行微调，使其能够解读多样化的提问并从BEV地图中提取相关上下文信息。此外，我们提出了一种以语言驱动的交通场景生成流水线，ChatBEV促进地图理解和文本对齐的导航指导，显著提高了真实一致的交通场景生成能力。该数据集、代码和微调模型将公开发布。 

---
# HSOD-BIT-V2: A New Challenging Benchmarkfor Hyperspectral Salient Object Detection 

**Title (ZH)**: HSOD-BIT-V2：新的高光谱显著目标检测挑战基准 

**Authors**: Yuhao Qiu, Shuyan Bai, Tingfa Xu, Peifu Liu, Haolin Qin, Jianan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.13906)  

**Abstract**: Salient Object Detection (SOD) is crucial in computer vision, yet RGB-based methods face limitations in challenging scenes, such as small objects and similar color features. Hyperspectral images provide a promising solution for more accurate Hyperspectral Salient Object Detection (HSOD) by abundant spectral information, while HSOD methods are hindered by the lack of extensive and available datasets. In this context, we introduce HSOD-BIT-V2, the largest and most challenging HSOD benchmark dataset to date. Five distinct challenges focusing on small objects and foreground-background similarity are designed to emphasize spectral advantages and real-world complexity. To tackle these challenges, we propose Hyper-HRNet, a high-resolution HSOD network. Hyper-HRNet effectively extracts, integrates, and preserves effective spectral information while reducing dimensionality by capturing the self-similar spectral features. Additionally, it conveys fine details and precisely locates object contours by incorporating comprehensive global information and detailed object saliency representations. Experimental analysis demonstrates that Hyper-HRNet outperforms existing models, especially in challenging scenarios. 

**Abstract (ZH)**: 高光谱显著目标检测(HSOD)在计算机视觉中至关重要，基于RGB的方法在小对象和相似颜色特征等挑战性场景中面临局限。高光谱图像通过丰富的光谱信息提供了更准确的HSOD的有希望的解决方案，而HSOD方法受限于缺乏广泛和可用的数据集。在此背景下，我们引入了HSOD-BIT-V2，截至目前为止最大的和最具挑战性的HSOD基准数据集。设计了五个针对小对象和前景背景相似性的独特挑战，以强调光谱优势和现实世界的复杂性。为了应对这些挑战，我们提出了Hyper-HRNet，一种高分辨率HSOD网络。Hyper-HRNet有效地提取、集成并保持有效的光谱信息，通过捕捉自我相似的光谱特征来降低维度。此外，它通过整合全面的全局信息和详细的对象显著性表示来传递细节点和精确定位对象轮廓。实验分析表明，Hyper-HRNet 在挑战性场景中优于现有模型。 

---
# TGBFormer: Transformer-GraphFormer Blender Network for Video Object Detection 

**Title (ZH)**: TGBFormer: Transformer-GraphFormer 混合网络及其在视频对象检测中的应用 

**Authors**: Qiang Qi, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13903)  

**Abstract**: Video object detection has made significant progress in recent years thanks to convolutional neural networks (CNNs) and vision transformers (ViTs). Typically, CNNs excel at capturing local features but struggle to model global representations. Conversely, ViTs are adept at capturing long-range global features but face challenges in representing local feature details. Off-the-shelf video object detection methods solely rely on CNNs or ViTs to conduct feature aggregation, which hampers their capability to simultaneously leverage global and local information, thereby resulting in limited detection performance. In this paper, we propose a Transformer-GraphFormer Blender Network (TGBFormer) for video object detection, with three key technical improvements to fully exploit the advantages of transformers and graph convolutional networks while compensating for their limitations. First, we develop a spatial-temporal transformer module to aggregate global contextual information, constituting global representations with long-range feature dependencies. Second, we introduce a spatial-temporal GraphFormer module that utilizes local spatial and temporal relationships to aggregate features, generating new local representations that are complementary to the transformer outputs. Third, we design a global-local feature blender module to adaptively couple transformer-based global representations and GraphFormer-based local representations. Extensive experiments demonstrate that our TGBFormer establishes new state-of-the-art results on the ImageNet VID dataset. Particularly, our TGBFormer achieves 86.5% mAP while running at around 41.0 FPS on a single Tesla A100 GPU. 

**Abstract (ZH)**: 基于Transformer和GraphFormer的时空融合网络在视频对象检测中的应用 

---
# Disentangling Fine-Tuning from Pre-Training in Visual Captioning with Hybrid Markov Logic 

**Title (ZH)**: 细解视觉_captioning中预训练与微调的分离研究：基于混合马尔可夫逻辑的方法 

**Authors**: Monika Shah, Somdeb Sarkhel, Deepak Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2503.13847)  

**Abstract**: Multimodal systems have highly complex processing pipelines and are pretrained over large datasets before being fine-tuned for specific tasks such as visual captioning. However, it becomes hard to disentangle what the model learns during the fine-tuning process from what it already knows due to its pretraining. In this work, we learn a probabilistic model using Hybrid Markov Logic Networks (HMLNs) over the training examples by relating symbolic knowledge (extracted from the caption) with visual features (extracted from the image). For a generated caption, we quantify the influence of training examples based on the HMLN distribution using probabilistic inference. We evaluate two types of inference procedures on the MSCOCO dataset for different types of captioning models. Our results show that for BLIP2 (a model that uses a LLM), the fine-tuning may have smaller influence on the knowledge the model has acquired since it may have more general knowledge to perform visual captioning as compared to models that do not use a LLM 

**Abstract (ZH)**: 多模态系统具有高度复杂的处理管道，并在大规模数据集上进行预训练，然后针对特定任务（如视觉字幕生成）进行微调。但由于预训练的原因，在微调过程中模型学到的内容与已知内容交织在一起，难以区分。在本文中，我们通过将符号知识（从字幕中提取）与视觉特征（从图像中提取）关联，使用混合马尔可夫逻辑网络（HMLNs）在训练样本上学习一个概率模型。对于生成的字幕，我们基于HMLN分布使用概率推理度量训练样本的影响。我们在MSCOCO数据集上对两种类型的推理程序对不同类型的字幕生成模型进行评估。结果显示，对于BLIP2（一种使用LLM的模型），微调对模型已获得的知识的影响可能较小，因为它可能具有更普遍的知识来进行视觉字幕生成， compared to models that do not use a LLM。 

---
# SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing 

**Title (ZH)**: SALAD: 骨骼感知的潜在扩散模型用于文本驱动的运动生成与编辑 

**Authors**: Seokhyeon Hong, Chaelin Kim, Serin Yoon, Junghyun Nam, Sihun Cha, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.13836)  

**Abstract**: Text-driven motion generation has advanced significantly with the rise of denoising diffusion models. However, previous methods often oversimplify representations for the skeletal joints, temporal frames, and textual words, limiting their ability to fully capture the information within each modality and their interactions. Moreover, when using pre-trained models for downstream tasks, such as editing, they typically require additional efforts, including manual interventions, optimization, or fine-tuning. In this paper, we introduce a skeleton-aware latent diffusion (SALAD), a model that explicitly captures the intricate inter-relationships between joints, frames, and words. Furthermore, by leveraging cross-attention maps produced during the generation process, we enable attention-based zero-shot text-driven motion editing using a pre-trained SALAD model, requiring no additional user input beyond text prompts. Our approach significantly outperforms previous methods in terms of text-motion alignment without compromising generation quality, and demonstrates practical versatility by providing diverse editing capabilities beyond generation. Code is available at project page. 

**Abstract (ZH)**: 基于文本的运动生成随着去噪扩散模型的兴起取得了显著进展。然而，先前的方法往往简化了对骨骼关节、时间帧和文本词的表示，限制了它们全面捕捉每种模态内及其之间信息的能力。此外，在使用预训练模型进行下游任务（如编辑）时，通常需要进行额外的努力，包括手动干预、优化或微调。本文引入了一种关节感知的潜在扩散（SALAD）模型，该模型明确捕捉关节、帧和词之间的复杂相互关系。此外，通过利用生成过程中产生的跨注意力图，我们使用预训练的SALAD模型实现了基于注意力的零样本文本驱动运动编辑，无需额外的用户输入，只需文本提示即可。我们的方法在文本-运动对齐方面显著优于先前的方法，同时在不牺牲生成质量的情况下展示了广泛的编辑能力。代码可在项目页面获取。 

---
# Organ-aware Multi-scale Medical Image Segmentation Using Text Prompt Engineering 

**Title (ZH)**: 基于文本提示工程的器官aware多尺度医学图像分割 

**Authors**: Wenjie Zhang, Ziyang Zhang, Mengnan He, Jiancheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13806)  

**Abstract**: Accurate segmentation is essential for effective treatment planning and disease monitoring. Existing medical image segmentation methods predominantly rely on uni-modal visual inputs, such as images or videos, requiring labor-intensive manual annotations. Additionally, medical imaging techniques capture multiple intertwined organs within a single scan, further complicating segmentation accuracy. To address these challenges, MedSAM, a large-scale medical segmentation model based on the Segment Anything Model (SAM), was developed to enhance segmentation accuracy by integrating image features with user-provided prompts. While MedSAM has demonstrated strong performance across various medical segmentation tasks, it primarily relies on geometric prompts (e.g., points and bounding boxes) and lacks support for text-based prompts, which could help specify subtle or ambiguous anatomical structures. To overcome these limitations, we propose the Organ-aware Multi-scale Text-guided Medical Image Segmentation Model (OMT-SAM) for multi-organ segmentation. Our approach introduces CLIP encoders as a novel image-text prompt encoder, operating with the geometric prompt encoder to provide informative contextual guidance. We pair descriptive textual prompts with corresponding images, processing them through pre-trained CLIP encoders and a cross-attention mechanism to generate fused image-text embeddings. Additionally, we extract multi-scale visual features from MedSAM, capturing fine-grained anatomical details at different levels of granularity. We evaluate OMT-SAM on the FLARE 2021 dataset, benchmarking its performance against existing segmentation methods. Empirical results demonstrate that OMT-SAM achieves a mean Dice Similarity Coefficient of 0.937, outperforming MedSAM (0.893) and other segmentation models, highlighting its superior capability in handling complex medical image segmentation tasks. 

**Abstract (ZH)**: 基于多尺度文本引导的器官aware医学图像分割模型（OMT-SAM） 

---
# Using 3D reconstruction from image motion to predict total leaf area in dwarf tomato plants 

**Title (ZH)**: 基于图像运动的3D重建预测矮番茄总叶面积 

**Authors**: Dmitrii Usenko, David Helman, Chen Giladi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13778)  

**Abstract**: Accurate estimation of total leaf area (TLA) is crucial for evaluating plant growth, photosynthetic activity, and transpiration. However, it remains challenging for bushy plants like dwarf tomatoes due to their complex canopies. Traditional methods are often labor-intensive, damaging to plants, or limited in capturing canopy complexity. This study evaluated a non-destructive method combining sequential 3D reconstructions from RGB images and machine learning to estimate TLA for three dwarf tomato cultivars: Mohamed, Hahms Gelbe Topftomate, and Red Robin -- grown under controlled greenhouse conditions. Two experiments (spring-summer and autumn-winter) included 73 plants, yielding 418 TLA measurements via an "onion" approach. High-resolution videos were recorded, and 500 frames per plant were used for 3D reconstruction. Point clouds were processed using four algorithms (Alpha Shape, Marching Cubes, Poisson's, Ball Pivoting), and meshes were evaluated with seven regression models: Multivariable Linear Regression, Lasso Regression, Ridge Regression, Elastic Net Regression, Random Forest, Extreme Gradient Boosting, and Multilayer Perceptron. The Alpha Shape reconstruction ($\alpha = 3$) with Extreme Gradient Boosting achieved the best performance ($R^2 = 0.80$, $MAE = 489 cm^2$). Cross-experiment validation showed robust results ($R^2 = 0.56$, $MAE = 579 cm^2$). Feature importance analysis identified height, width, and surface area as key predictors. This scalable, automated TLA estimation method is suited for urban farming and precision agriculture, offering applications in automated pruning, resource efficiency, and sustainable food production. The approach demonstrated robustness across variable environmental conditions and canopy structures. 

**Abstract (ZH)**: 准确估计矮番茄的总叶片面积对于评估植物生长、光合活性和蒸腾作用至关重要。然而，对于如矮番茄这样的丛生植物而言，由于其复杂的树冠结构，这仍然是一个挑战。传统方法往往耗时费力、对植物有害，或者难以捕捉树冠的复杂性。本研究评估了一种结合RGB图像序列3D重建和机器学习的非破坏性方法，以估计三种矮番茄品种（Mohamed、Hahms Gelbe Topftomate和Red Robin）在受控温室条件下的树冠总叶片面积。两个实验（春季-夏季和秋季-冬季）共包括73株植物，通过“洋葱”方法获得418个树冠总叶片面积测量值。记录了高分辨率视频，并为3D重建使用了每株植物500帧。点云数据使用四种算法（Alpha Shape、Marching Cubes、Poisson、Ball Pivoting）进行处理，网状结构使用七种回归模型进行评估：多元线性回归、套索回归、岭回归、弹性网回归、随机森林、极 Gradient Boosting和支持向量机（Multilayer Perceptron）。Alpha Shape重建（$\alpha = 3$）与极Gradient Boosting相结合取得了最佳性能（$R^2 = 0.80$，$MAE = 489 cm^2$）。跨实验验证显示了稳健的结果（$R^2 = 0.56$，$MAE = 579 cm^2$）。特征重要性分析确定了高度、宽度和表面积为关键预测因素。该可扩展、自动化的总叶片面积估计方法适用于城市农业和精准农业，其应用包括自动化修剪、资源效率和可持续食品生产。该方法展示了在不同环境条件和树冠结构下的稳健性。 

---
# ASMR: Adaptive Skeleton-Mesh Rigging and Skinning via 2D Generative Prior 

**Title (ZH)**: 自适应骨架网格绑定与皮肤权重分配的2D生成先验方法 

**Authors**: Seokhyeon Hong, Soojin Choi, Chaelin Kim, Sihun Cha, Junyong Noh  

**Link**: [PDF](https://arxiv.org/pdf/2503.13579)  

**Abstract**: Despite the growing accessibility of skeletal motion data, integrating it for animating character meshes remains challenging due to diverse configurations of both skeletons and meshes. Specifically, the body scale and bone lengths of the skeleton should be adjusted in accordance with the size and proportions of the mesh, ensuring that all joints are accurately positioned within the character mesh. Furthermore, defining skinning weights is complicated by variations in skeletal configurations, such as the number of joints and their hierarchy, as well as differences in mesh configurations, including their connectivity and shapes. While existing approaches have made efforts to automate this process, they hardly address the variations in both skeletal and mesh configurations. In this paper, we present a novel method for the automatic rigging and skinning of character meshes using skeletal motion data, accommodating arbitrary configurations of both meshes and skeletons. The proposed method predicts the optimal skeleton aligned with the size and proportion of the mesh as well as defines skinning weights for various mesh-skeleton configurations, without requiring explicit supervision tailored to each of them. By incorporating Diffusion 3D Features (Diff3F) as semantic descriptors of character meshes, our method achieves robust generalization across different configurations. To assess the performance of our method in comparison to existing approaches, we conducted comprehensive evaluations encompassing both quantitative and qualitative analyses, specifically examining the predicted skeletons, skinning weights, and deformation quality. 

**Abstract (ZH)**: 尽管骨骼运动数据的可获得性不断提高，但由于骨骼和网格的多样配置，将它们整合用于动画角色网格仍具有挑战性。具体而言，需要根据网格的大小和比例调整骨骼的身体尺度和骨长，确保所有关节准确地定位在角色网格中。此外，由骨骼配置的差异（如关节数量和层次结构）以及网格配置的差异（包括拓扑和形状）引起的变化，使得定义蒙皮权重变得复杂。虽然现有方法已试图自动化这一过程，但它们难以解决骨骼和网格配置的差异。在本文中，我们提出了一种利用骨骼运动数据自动设置角色网格的方法，能够适应骨骼和网格的任意配置。所提出的方法预测与网格大小和比例相匹配的最佳骨骼，并为各种网格-骨骼配置定义蒙皮权重，无需针对每种配置进行显式监督。通过将Diffusion 3D Features (Diff3F) 作为角色网格的语义描述符，我们的方法实现了跨不同配置的稳健泛化。为了评估我们的方法与现有方法的性能差异，我们进行了综合评估，包括定量和定性的分析，具体评估了预测的骨骼、蒙皮权重和变形质量。 

---
# Long-horizon Visual Instruction Generation with Logic and Attribute Self-reflection 

**Title (ZH)**: 长时视角视觉指令生成：逻辑与属性自省 

**Authors**: Yucheng Suo, Fan Ma, Kaixin Shen, Linchao Zhu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13500)  

**Abstract**: Visual instructions for long-horizon tasks are crucial as they intuitively clarify complex concepts and enhance retention across extended steps. Directly generating a series of images using text-to-image models without considering the context of previous steps results in inconsistent images, increasing cognitive load. Additionally, the generated images often miss objects or the attributes such as color, shape, and state of the objects are inaccurate. To address these challenges, we propose LIGER, the first training-free framework for Long-horizon Instruction GEneration with logic and attribute self-Reflection. LIGER first generates a draft image for each step with the historical prompt and visual memory of previous steps. This step-by-step generation approach maintains consistency between images in long-horizon tasks. Moreover, LIGER utilizes various image editing tools to rectify errors including wrong attributes, logic errors, object redundancy, and identity inconsistency in the draft images. Through this self-reflection mechanism, LIGER improves the logic and object attribute correctness of the images. To verify whether the generated images assist human understanding, we manually curated a new benchmark consisting of various long-horizon tasks. Human-annotated ground truth expressions reflect the human-defined criteria for how an image should appear to be illustrative. Experiments demonstrate the visual instructions generated by LIGER are more comprehensive compared with baseline methods. 

**Abstract (ZH)**: 长时间任务的视觉指导至关重要，因为它们能直观地阐明复杂概念并提高长时间步骤的记忆保留。直接使用文本到图像模型生成一系列图像而不考虑先前步骤的上下文会导致图像不一致，增加认知负担。此外，生成的图像往往遗漏物体，或物体的颜色、形状和状态不准确。为了解决这些问题，我们提出了LIGER，这是一种不需要训练的长期指令生成框架，结合了逻辑和属性自我反思。LIGER首先使用历史提示和先前步骤的视觉记忆为每个步骤生成一个草图图像。这种逐步生成的方法在长时间任务中保持了图像的一致性。此外，LIGER利用各种图像编辑工具修正草图图像中的错误，包括错误的属性、逻辑错误、对象冗余和身份不一致性。通过这种自我反思机制，LIGER提高了图像的逻辑和物体属性的准确性。为了验证生成的图像是否有助于人类理解，我们手动构建了一个包含多种长时间任务的新基准。基于人类注释的真实表达反映了人类定义的图像应该如何直观呈现的标准。实验结果表明，LIGER生成的视觉指令比基线方法更为全面。 

---
# Onboard Terrain Classification via Stacked Intelligent Metasurface-Diffractive Deep Neural Networks from SAR Level-0 Raw Data 

**Title (ZH)**: 基于SAR原始零级数据的堆叠智能超表面-衍射深度神经网络的地表分类 

**Authors**: Mengbing Liu, Xin Li, Jiancheng An, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13488)  

**Abstract**: This paper introduces a novel approach for real-time onboard terrain classification from Sentinel-1 (S1) level-0 raw In-phase/Quadrature (IQ) data, leveraging a Stacked Intelligent Metasurface (SIM) to perform inference directly in the analog wave domain. Unlike conventional digital deep neural networks, the proposed multi-layer Diffractive Deep Neural Network (D$^2$NN) setup implements automatic feature extraction as electromagnetic waves propagate through stacked metasurface layers. This design not only reduces reliance on expensive downlink bandwidth and high-power computing at terrestrial stations but also achieves performance levels around 90\% directly from the real raw IQ data, in terms of accuracy, precision, recall, and F1 Score. Our method therefore helps bridge the gap between next-generation remote sensing tasks and in-orbit processing needs, paving the way for computationally efficient remote sensing applications. 

**Abstract (ZH)**: 本文介绍了一种利用堆叠智能介质（SIM）直接在模拟波域进行推理的新型方法，以实现Sentinel-1（S1）级0级原始同相/正交（IQ）数据的实时在轨地形分类。所提出的多层衍射深度神经网络（D$^2$NN）设置在电磁波通过堆叠介质层传播时实现了自动特征提取。与传统的数字深度神经网络不同，该设计不仅减少了对昂贵的下行链路带宽和高功率地面站计算资源的依赖，还在准确度、精确度、召回率和F1分数方面直接从实际原始IQ数据中实现了约90%的性能水平。因此，该方法有助于弥合下一代遥感任务与在轨处理需求之间的差距，为计算高效的遥感应用铺平了道路。 

---
# Robust Detection of Extremely Thin Lines Using 0.2mm Piano Wire 

**Title (ZH)**: 使用0.2mm钢琴线进行稳健的极细线检测 

**Authors**: Jisoo Hong, Youngjin Jung, Jihwan Bae, Seungho Song, Sung-Woo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13473)  

**Abstract**: This study developed an algorithm capable of detecting a reference line (a 0.2 mm thick piano wire) to accurately determine the position of an automated installation robot within an elevator shaft. A total of 3,245 images were collected from the experimental tower of H Company, the leading elevator manufacturer in South Korea, and the detection performance was evaluated using four experimental approaches (GCH, GSCH, GECH, FCH). During the initial image processing stage, Gaussian blurring, sharpening filter, embossing filter, and Fourier Transform were applied, followed by Canny Edge Detection and Hough Transform. Notably, the method was developed to accurately extract the reference line by averaging the x-coordinates of the lines detected through the Hough Transform. This approach enabled the detection of the 0.2 mm thick piano wire with high accuracy, even in the presence of noise and other interfering factors (e.g., concrete cracks inside the elevator shaft or safety bars for filming equipment). The experimental results showed that Experiment 4 (FCH), which utilized Fourier Transform in the preprocessing stage, achieved the highest detection rate for the LtoL, LtoR, and RtoL datasets. Experiment 2(GSCH), which applied Gaussian blurring and a sharpening filter, demonstrated superior detection performance on the RtoR dataset. This study proposes a reference line detection algorithm that enables precise position calculation and control of automated robots in elevator shaft installation. Moreover, the developed method shows potential for applicability even in confined working spaces. Future work aims to develop a line detection algorithm equipped with machine learning-based hyperparameter tuning capabilities. 

**Abstract (ZH)**: 本研究开发了一种能够检测参考线（0.2 mm厚的钢琴线）的算法，以精确确定安装机器人在电梯井中的位置。共收集了来自韩国领先电梯制造商H公司的实验塔的3,245张图像，并使用四种实验方法（GCH、GSCH、GECH、FCH）评估了检测性能。在初始图像处理阶段，应用了高斯模糊、锐化滤波、浮雕滤波和傅里叶变换，随后进行了Canny边缘检测和霍夫变换。值得注意的是，该方法通过霍夫变换检测的线的x坐标求平均值的方式，精确提取了参考线。这一方法即使在存在噪声和其他干扰因素（如电梯井内的混凝土裂缝或拍摄设备的安全杆）的情况下，也能高精度地检测到0.2 mm厚的钢琴线。实验结果显示，使用傅里叶变换预处理的实验4（FCH）在LtoL、LtoR和RtoL数据集上的检测率最高。应用了高斯模糊和锐化滤波的实验2（GSCH）在RtoR数据集上表现出色。本研究提出了一种参考线检测算法，该算法能够精确计算和控制电梯井安装中自动化机器人的位置。此外，所开发的方法在狭窄的工作空间内也具有潜在的应用价值。未来工作将致力于开发具有基于机器学习的超参数调整能力的线检测算法。 

---
# Efficient Domain Augmentation for Autonomous Driving Testing Using Diffusion Models 

**Title (ZH)**: 使用扩散模型的高效领域扩充方法用于自动驾驶测试 

**Authors**: Luciano Baresi, Davide Yi Xian Hu, Andrea Stocco, Paolo Tonella  

**Link**: [PDF](https://arxiv.org/pdf/2409.13661)  

**Abstract**: Simulation-based testing is widely used to assess the reliability of Autonomous Driving Systems (ADS), but its effectiveness is limited by the operational design domain (ODD) conditions available in such simulators. To address this limitation, in this work, we explore the integration of generative artificial intelligence techniques with physics-based simulators to enhance ADS system-level testing. Our study evaluates the effectiveness and computational overhead of three generative strategies based on diffusion models, namely instruction-editing, inpainting, and inpainting with refinement. Specifically, we assess these techniques' capabilities to produce augmented simulator-generated images of driving scenarios representing new ODDs. We employ a novel automated detector for invalid inputs based on semantic segmentation to ensure semantic preservation and realism of the neural generated images. We then perform system-level testing to evaluate the ADS's generalization ability to newly synthesized ODDs. Our findings show that diffusion models help increase the ODD coverage for system-level testing of ADS. Our automated semantic validator achieved a percentage of false positives as low as 3%, retaining the correctness and quality of the generated images for testing. Our approach successfully identified new ADS system failures before real-world testing. 

**Abstract (ZH)**: 基于生成人工智能技术与物理仿真结合的自动驾驶系统测试方法研究 

---
