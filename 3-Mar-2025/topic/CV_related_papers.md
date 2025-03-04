# Towards Semantic 3D Hand-Object Interaction Generation via Functional Text Guidance 

**Title (ZH)**: 基于功能性文本指导的语义三维手物交互生成 

**Authors**: Yongqi Tian, Xueyu Sun, Haoyuan He, Linji Hao, Ning Ding, Caigui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20805)  

**Abstract**: Hand-object interaction(HOI) is the fundamental link between human and environment, yet its dexterous and complex pose significantly challenges for gesture control. Despite significant advances in AI and robotics, enabling machines to understand and simulate hand-object interactions, capturing the semantics of functional grasping tasks remains a considerable challenge. While previous work can generate stable and correct 3D grasps, they are still far from achieving functional grasps due to unconsidered grasp semantics. To address this challenge, we propose an innovative two-stage framework, Functional Grasp Synthesis Net (FGS-Net), for generating 3D HOI driven by functional text. This framework consists of a text-guided 3D model generator, Functional Grasp Generator (FGG), and a pose optimization strategy, Functional Grasp Refiner (FGR). FGG generates 3D models of hands and objects based on text input, while FGR fine-tunes the poses using Object Pose Approximator and energy functions to ensure the relative position between the hand and object aligns with human intent and remains physically plausible. Extensive experiments demonstrate that our approach achieves precise and high-quality HOI generation without requiring additional 3D annotation data. 

**Abstract (ZH)**: 功能化抓取合成网络（FGS-Net）：基于功能文本驱动的3D手物互动生成 

---
# L-Lipschitz Gershgorin ResNet Network 

**Title (ZH)**: L-Lipschitz Gershgorin ResNet 网络 

**Authors**: Marius F. R. Juston, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21279)  

**Abstract**: Deep residual networks (ResNets) have demonstrated outstanding success in computer vision tasks, attributed to their ability to maintain gradient flow through deep architectures. Simultaneously, controlling the Lipschitz bound in neural networks has emerged as an essential area of research for enhancing adversarial robustness and network certifiability. This paper uses a rigorous approach to design $\mathcal{L}$-Lipschitz deep residual networks using a Linear Matrix Inequality (LMI) framework. The ResNet architecture was reformulated as a pseudo-tri-diagonal LMI with off-diagonal elements and derived closed-form constraints on network parameters to ensure $\mathcal{L}$-Lipschitz continuity. To address the lack of explicit eigenvalue computations for such matrix structures, the Gershgorin circle theorem was employed to approximate eigenvalue locations, guaranteeing the LMI's negative semi-definiteness. Our contributions include a provable parameterization methodology for constructing Lipschitz-constrained networks and a compositional framework for managing recursive systems within hierarchical architectures. These findings enable robust network designs applicable to adversarial robustness, certified training, and control systems. However, a limitation was identified in the Gershgorin-based approximations, which over-constrain the system, suppressing non-linear dynamics and diminishing the network's expressive capacity. 

**Abstract (ZH)**: 基于Lipschitz约束的线性矩阵不等式框架下的深度残差网络设计 

---
# The PanAf-FGBG Dataset: Understanding the Impact of Backgrounds in Wildlife Behaviour Recognition 

**Title (ZH)**: PanAf-FGBG数据集：了解背景对野生动物行为识别的影响 

**Authors**: Otto Brookes, Maksim Kukushkin, Majid Mirmehdi, Colleen Stephens, Paula Dieguez, Thurston C. Hicks, Sorrel Jones, Kevin Lee, Maureen S. McCarthy, Amelia Meier, Emmanuelle Normand, Erin G. Wessling, Roman M.Wittig, Kevin Langergraber, Klaus Zuberbühler, Lukas Boesch, Thomas Schmid, Mimi Arandjelovic, Hjalmar Kühl, Tilo Burghardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.21201)  

**Abstract**: Computer vision analysis of camera trap video footage is essential for wildlife conservation, as captured behaviours offer some of the earliest indicators of changes in population health. Recently, several high-impact animal behaviour datasets and methods have been introduced to encourage their use; however, the role of behaviour-correlated background information and its significant effect on out-of-distribution generalisation remain unexplored. In response, we present the PanAf-FGBG dataset, featuring 20 hours of wild chimpanzee behaviours, recorded at over 350 individual camera locations. Uniquely, it pairs every video with a chimpanzee (referred to as a foreground video) with a corresponding background video (with no chimpanzee) from the same camera location. We present two views of the dataset: one with overlapping camera locations and one with disjoint locations. This setup enables, for the first time, direct evaluation of in-distribution and out-of-distribution conditions, and for the impact of backgrounds on behaviour recognition models to be quantified. All clips come with rich behavioural annotations and metadata including unique camera IDs and detailed textual scene descriptions. Additionally, we establish several baselines and present a highly effective latent-space normalisation technique that boosts out-of-distribution performance by +5.42% mAP for convolutional and +3.75% mAP for transformer-based models. Finally, we provide an in-depth analysis on the role of backgrounds in out-of-distribution behaviour recognition, including the so far unexplored impact of background durations (i.e., the count of background frames within foreground videos). 

**Abstract (ZH)**: 计算机视觉分析的摄像机陷阱视频对于野生动物保护至关重要，因为捕获的行为提供了有关种群健康变化的最早指标。最近，一些高影响力的动物行为数据集和方法被引入以鼓励其使用；然而，与行为相关的背景信息的作用及其对外分布泛化的显著影响尚未被探索。为应对这一挑战，我们提出了PanAf-FGBG数据集，包含20小时野外黑猩猩行为录像，记录在超过350个独立的摄像头位置。该数据集独特的特点是，每个视频都被配对为一个包含黑猩猩的前景视频（来自同一摄像头位置）和一个没有黑猩猩的背景视频。我们展示了数据集的两种视图：一种是重叠摄像头位置的视图，另一种是非重叠位置的视图。这一设置使得首次能够直接评估内部分布和外部分布条件，并量化背景对行为识别模型的影响。所有剪辑都附有丰富的行为注释和元数据，包括独特的摄像头ID和详细的文本场景描述。此外，我们建立了几个基线，并提出了一种高效的空间隐状态归一化技术，该技术分别提高了卷积模型和基于变压器模型的 +5.42% 和 +3.75% 的mAP性能。最后，我们深入分析了背景在外分布行为识别中的作用，包括到目前为止尚未探索的背景持续时间（即前景视频中的背景帧计数）的影响。 

---
# Synthesizing Individualized Aging Brains in Health and Disease with Generative Models and Parallel Transport 

**Title (ZH)**: 使用生成模型和平行传输合成健康与疾病中的个性化衰老大脑 

**Authors**: Jingru Fu, Yuqi Zheng, Neel Dey, Daniel Ferreira, Rodrigo Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2502.21049)  

**Abstract**: Simulating prospective magnetic resonance imaging (MRI) scans from a given individual brain image is challenging, as it requires accounting for canonical changes in aging and/or disease progression while also considering the individual brain's current status and unique characteristics. While current deep generative models can produce high-resolution anatomically accurate templates for population-wide studies, their ability to predict future aging trajectories for individuals remains limited, particularly in capturing subject-specific neuroanatomical variations over time. In this study, we introduce Individualized Brain Synthesis (InBrainSyn), a framework for synthesizing high-resolution subject-specific longitudinal MRI scans that simulate neurodegeneration in both Alzheimer's disease (AD) and normal aging. InBrainSyn uses a parallel transport algorithm to adapt the population-level aging trajectories learned by a generative deep template network, enabling individualized aging synthesis. As InBrainSyn uses diffeomorphic transformations to simulate aging, the synthesized images are topologically consistent with the original anatomy by design. We evaluated InBrainSyn both quantitatively and qualitatively on AD and healthy control cohorts from the Open Access Series of Imaging Studies - version 3 dataset. Experimentally, InBrainSyn can also model neuroanatomical transitions between normal aging and AD. An evaluation of an external set supports its generalizability. Overall, with only a single baseline scan, InBrainSyn synthesizes realistic 3D spatiotemporal T1w MRI scans, producing personalized longitudinal aging trajectories. The code for InBrainSyn is available at: this https URL. 

**Abstract (ZH)**: 个体化脑图像合成：从给定个体脑图像模拟前瞻性磁共振成像扫描是一项挑战，需要考虑衰老和/或疾病进展的规范变化，同时还要考虑个体脑的当前状态和独特特征。当前的深度生成模型可以为大规模研究生成高分辨率的解剖学准确模板，但在预测个体的未来衰老轨迹方面仍然有限，特别是在捕捉随时间变化的个体神经解剖学变异方面。在本研究中，我们引入了一种个体化脑合成（InBrainSyn）框架，用于合成高分辨率的个体特异性纵向磁共振成像扫描，以模拟阿尔茨海默病（AD）和正常衰老中的神经退行性变化。InBrainSyn 使用并行传输算法来适应生成性深度模板网络学习到的群体水平的衰老轨迹，从而实现个体化的衰老合成。由于 InBrainSyn 使用辛 diffeomorphic 变换来模拟衰老，因此合成图像在设计上与原始解剖结构保持拓扑一致性。我们在使用开放访问成像系列 - 第三版数据集中 AD 和健康对照组进行了定量和定性的评估。实验中，InBrainSyn 还可以模拟正常衰老和 AD 之间的神经解剖学过渡。外部数据集的评估支持其普适性。总体而言，仅使用一个基线扫描，InBrainSyn 合成真实的世界 3D 脑 T1w 磁共振成像扫描，生成个性化的纵向衰老轨迹。InBrainSyn 的代码可在以下链接获取：this https URL。 

---
# LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging 

**Title (ZH)**: LesionLocator: 零样本全身三维肿瘤分割与跟踪 

**Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2502.20985)  

**Abstract**: In this work, we present LesionLocator, a framework for zero-shot longitudinal lesion tracking and segmentation in 3D medical imaging, establishing the first end-to-end model capable of 4D tracking with dense spatial prompts. Our model leverages an extensive dataset of 23,262 annotated medical scans, as well as synthesized longitudinal data across diverse lesion types. The diversity and scale of our dataset significantly enhances model generalizability to real-world medical imaging challenges and addresses key limitations in longitudinal data availability. LesionLocator outperforms all existing promptable models in lesion segmentation by nearly 10 dice points, reaching human-level performance, and achieves state-of-the-art results in lesion tracking, with superior lesion retrieval and segmentation accuracy. LesionLocator not only sets a new benchmark in universal promptable lesion segmentation and automated longitudinal lesion tracking but also provides the first open-access solution of its kind, releasing our synthetic 4D dataset and model to the community, empowering future advancements in medical imaging. Code is available at: this http URL 

**Abstract (ZH)**: 本研究介绍了LesionLocator框架，该框架用于三维医学成像中的零样本纵向病灶跟踪和分割，建立了首个能够进行4D跟踪的端到端模型，使用密集的空间提示。我们的模型利用了包含23,262份标注医学扫描数据的大型数据集，并跨多种病灶类型合成了纵向数据。我们的数据集的多样性和规模显著增强了模型对实际医学成像挑战的通用性，并解决了纵向数据可用性的关键限制。LesionLocator在病灶分割方面优于所有现有的提示可调模型，达到了接近人类水平的性能，并在病灶跟踪方面取得了最先进的结果，具有更高的病灶检索和分割准确性。LesionLocator不仅在通用提示可调病灶分割和自动化纵向病灶跟踪方面设定了新的标准，还提供了同类首个开源解决方案，发布了我们的合成4D数据集和模型，推动未来医学成像的发展。代码可在以下链接获取：this http URL。 

---
# Fine-Grained Retrieval-Augmented Generation for Visual Question Answering 

**Title (ZH)**: 细粒度检索增强生成在视觉问答中的应用 

**Authors**: Zhengxuan Zhang, Yin Wu, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20964)  

**Abstract**: Visual Question Answering (VQA) focuses on providing answers to natural language questions by utilizing information from images. Although cutting-edge multimodal large language models (MLLMs) such as GPT-4o achieve strong performance on VQA tasks, they frequently fall short in accessing domain-specific or the latest knowledge. To mitigate this issue, retrieval-augmented generation (RAG) leveraging external knowledge bases (KBs), referred to as KB-VQA, emerges as a promising approach. Nevertheless, conventional unimodal retrieval techniques, which translate images into textual descriptions, often result in the loss of critical visual details. This study presents fine-grained knowledge units, which merge textual snippets with entity images stored in vector databases. Furthermore, we introduce a knowledge unit retrieval-augmented generation framework (KU-RAG) that integrates fine-grained retrieval with MLLMs. The proposed KU-RAG framework ensures precise retrieval of relevant knowledge and enhances reasoning capabilities through a knowledge correction chain. Experimental findings demonstrate that our approach significantly boosts the performance of leading KB-VQA methods, achieving improvements of up to 10%. 

**Abstract (ZH)**: 视觉问答（VQA）专注于通过利用图像信息来回答自然语言问题。尽管最先进的多模态大型语言模型（MLLMs），如GPT-4o，在VQA任务中表现出色，但它们在获取领域特定或最新知识方面经常不足。为解决这个问题，利用外部知识库（KBs）的检索增强生成（RAG）方法，即KB-VQA，成为一种有前景的方法。然而，传统的单模态检索技术将图像转换为文本描述，往往会损失关键的视觉细节。本研究提出细粒度知识单元，将文本片段与存储在向量数据库中的实体图像结合。此外，我们引入了一种细粒度检索增强生成框架（KU-RAG），将细粒度检索与MLLMs集成。提出的KU-RAG框架确保精准检索相关知识，并通过知识校正链增强推理能力。实验结果表明，我们的方法显著提升了领先KB-VQA方法的性能，最高提高了10%。 

---
# Less is More? Revisiting the Importance of Frame Rate in Real-Time Zero-Shot Surgical Video Segmentation 

**Title (ZH)**: 少即是多？重新审视实时零样本手术视频分割中的帧率的重要性 

**Authors**: Utku Ozbulak, Seyed Amir Mousavi, Francesca Tozzi, Nikdokht Rashidian, Wouter Willaert, Wesley De Neve, Joris Vankerschaver  

**Link**: [PDF](https://arxiv.org/pdf/2502.20934)  

**Abstract**: Real-time video segmentation is a promising feature for AI-assisted surgery, providing intraoperative guidance by identifying surgical tools and anatomical structures. However, deploying state-of-the-art segmentation models, such as SAM2, in real-time settings is computationally demanding, which makes it essential to balance frame rate and segmentation performance. In this study, we investigate the impact of frame rate on zero-shot surgical video segmentation, evaluating SAM2's effectiveness across multiple frame sampling rates for cholecystectomy procedures. Surprisingly, our findings indicate that in conventional evaluation settings, frame rates as low as a single frame per second can outperform 25 FPS, as fewer frames smooth out segmentation inconsistencies. However, when assessed in a real-time streaming scenario, higher frame rates yield superior temporal coherence and stability, particularly for dynamic objects such as surgical graspers. Finally, we investigate human perception of real-time surgical video segmentation among professionals who work closely with such data and find that respondents consistently prefer high FPS segmentation mask overlays, reinforcing the importance of real-time evaluation in AI-assisted surgery. 

**Abstract (ZH)**: 实时视频分割是AI辅助手术中的一项有前途的功能，通过识别手术工具和解剖结构提供术中指导。然而，在实际应用中部署先进的分割模型，如SAM2，需要较高的计算能力，因此需要在帧率和分割性能之间进行权衡。在本研究中，我们探讨了帧率对零样本手术视频分割的影响，评估了SAM2在胆囊切除手术过程中多种帧采样率下的有效性。令人惊讶的是，我们的发现表明，在常规评估设置中，低至每秒一帧的帧率可以优于25 FPS，因为较少的帧可以平滑分割不一致。然而，在实时流媒体场景中评估时，较高的帧率提供了更好的时序一致性和稳定性，尤其是在对手术钳这类动态对象的分割上。最后，我们考察了专业人士对手术视频实时分割的感知，发现受访者一致偏爱高FPS的分割掩膜叠加，进一步强调了实时评估在AI辅助手术中的重要性。 

---
# Oscillation-Reduced MXFP4 Training for Vision Transformers 

**Title (ZH)**: Oscillation-Reduced MXFP4 Training for Vision Transformers 

**Authors**: Yuxiang Chen, Haocheng Xi, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20853)  

**Abstract**: Pre-training Transformers in FP4 precision is becoming a promising approach to gain substantial speedup, but it comes with a considerable loss of accuracy. Microscaling (MX) data format provides a fine-grained per-group quantization method to improve the representation ability of the FP4 format and is supported by the next-generation Blackwell GPU architecture. However, training with MXFP4 data format still results in significant degradation and there is a lack of systematic research on the reason.
In this work, we propose a novel training method TetraJet for a more accurate FP4 training. We comprehensively evaluate all of the quantizers involved in the training, and identify the weight oscillation problem in the forward pass as the main source of the degradation in MXFP4 training. Therefore, we introduce two novel methods, EMA Quantizer (Q-EMA) and Adaptive Ramping Optimizer (Q-Ramping), to resolve the oscillation problem. Extensive experiments on Vision Transformers demonstrate that TetraJet consistently outperforms the existing 4-bit training methods, and Q-EMA & Q-Ramping can provide additional enhancement by effectively reducing oscillation. We decreased the accuracy degradation by more than $50\%$ compared to the baseline, and can even achieve competitive performance compared to full precision training. The codes are available at this https URL 

**Abstract (ZH)**: 四阶训练精度提升的新训练方法：TetraJet及EMA量化器和自适应加温优化器在视觉变换器中的综合评估 

---
# Structured Preference Optimization for Vision-Language Long-Horizon Task Planning 

**Title (ZH)**: 视觉-语言长时序任务规划的结构化偏好优化 

**Authors**: Xiwen Liang, Min Lin, Weiqi Ruan, Rongtao Xu, Yuecheng Liu, Jiaqi Chen, Bingqian Lin, Yuzheng Zhuang, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20742)  

**Abstract**: Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines. 

**Abstract (ZH)**: 现有的视觉-语言任务规划方法在短期任务上表现优异，但在动态环境下的复杂长期任务规划方面往往表现不佳。这些问题主要源于有效训练模型以生成高质量长期任务推理过程的难度。为了解决这个问题，我们提出了结构化偏好优化（SPO），旨在通过结构化偏好评估和优化训练策略来增强长期任务规划中的推理和动作选择。具体来说，SPO 引入了：1) 基于偏好评分与优化，系统地根据任务相关性、视觉定位和历史一致性评估推理链；2) 逐步训练，其中模型从简单任务逐步过渡到复杂任务，提高其在长期情景中的泛化能力，并增强推理的稳健性。为了推进视觉-语言长期任务规划的研究，我们介绍了ExtendaBench，这是一个综合性基准，包含VirtualHome和Habitat 2.0中的1,509个任务，这些任务被分类为超短期、短期、中期和长期任务。实验结果表明，SPO 显著提高了推理质量和最终决策准确性，在长期任务上优于之前的方法，并证实了偏好驱动优化在视觉-语言任务规划中的有效性。具体而言，SPO 在VirtualHome上的GCR提高了5.98%，SR提高了4.68%，在Habitat上的GCR提高了3.30%，SR提高了2.11%，均优于最佳基线方法。 

---
# Advancing AI-Powered Medical Image Synthesis: Insights from MedVQA-GI Challenge Using CLIP, Fine-Tuned Stable Diffusion, and Dream-Booth + LoRA 

**Title (ZH)**: 基于MedVQA-GI挑战赛的AI驱动医学图像合成进展：CLIP、微调 Stable Diffusion、Dream-Booth + LoRA 的见解 

**Authors**: Ojonugwa Oluwafemi Ejiga Peter, Md Mahmudur Rahman, Fahmi Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2502.20667)  

**Abstract**: The MEDVQA-GI challenge addresses the integration of AI-driven text-to-image generative models in medical diagnostics, aiming to enhance diagnostic capabilities through synthetic image generation. Existing methods primarily focus on static image analysis and lack the dynamic generation of medical imagery from textual descriptions. This study intends to partially close this gap by introducing a novel approach based on fine-tuned generative models to generate dynamic, scalable, and precise images from textual descriptions. Particularly, our system integrates fine-tuned Stable Diffusion and DreamBooth models, as well as Low-Rank Adaptation (LORA), to generate high-fidelity medical images. The problem is around two sub-tasks namely: image synthesis (IS) and optimal prompt production (OPG). The former creates medical images via verbal prompts, whereas the latter provides prompts that produce high-quality images in specified categories. The study emphasizes the limitations of traditional medical image generation methods, such as hand sketching, constrained datasets, static procedures, and generic models. Our evaluation measures showed that Stable Diffusion surpasses CLIP and DreamBooth + LORA in terms of producing high-quality, diversified images. Specifically, Stable Diffusion had the lowest Fréchet Inception Distance (FID) scores (0.099 for single center, 0.064 for multi-center, and 0.067 for combined), indicating higher image quality. Furthermore, it had the highest average Inception Score (2.327 across all datasets), indicating exceptional diversity and quality. This advances the field of AI-powered medical diagnosis. Future research will concentrate on model refining, dataset augmentation, and ethical considerations for efficiently implementing these advances into clinical practice 

**Abstract (ZH)**: MEDVQA-GI挑战赛旨在将AI驱动的文本生成图像生成模型融入医学诊断，通过合成图像生成增强诊断能力。现有方法主要侧重于静态图像分析，缺乏从文本描述中动态生成医学图像的能力。本研究通过引入基于微调生成模型的新方法，旨在部分解决这一问题，以生成动态、可扩展和精确的图像。特别是，我们的系统集成了微调的Stable Diffusion和DreamBooth模型以及低秩适应（LORA），以生成高保真医学图像。研究围绕两个子任务：图像合成（IS）和最优提示生成（OPG）。前者通过口头提示生成医学图像，后者提供能够生成指定类别优质图像的提示。研究强调了传统医学图像生成方法的局限性，如手工草图、受限的数据集、静态流程和通用模型。评估结果表明，Stable Diffusion在生成高质量和多样化图像方面优于CLIP和DreamBooth + LORA。具体而言，Stable Diffusion在单中心（0.099）、多中心（0.064）和联合数据集（0.067）的弗雷歇 inception 距离（FID）得分最低，表明图像质量更高。此外，在所有数据集中的平均 inception 分数最高（2.327），表明具有出色多样性和质量。这推进了人工智能辅助医学诊断领域的发展。未来研究将集中于模型优化、数据集扩充和伦理考量，以有效地将这些进步应用于临床实践。 

---
# Unified Kernel-Segregated Transpose Convolution Operation 

**Title (ZH)**: 统一内核分离转置卷积操作 

**Authors**: Vijay Srinivas Tida, Md Imran Hossen, Liqun Shan, Sai Venkatesh Chilukoti, Sonya Hsu, Xiali Hei  

**Link**: [PDF](https://arxiv.org/pdf/2502.20493)  

**Abstract**: The optimization of the transpose convolution layer for deep learning applications is achieved with the kernel segregation mechanism. However, kernel segregation has disadvantages, such as computing extra elements to obtain the output feature map with odd dimensions while launching a thread. To mitigate this problem, we introduce a unified kernel segregation approach that limits the usage of memory and computational resources by employing one unified kernel to execute four sub-kernels. The findings reveal that the suggested approach achieves an average computational speedup of 2.03x (3.89x) when tested on specific datasets with an RTX 2070 GPU (Intel Xeon CPU). The ablation study shows an average computational speedup of 3.5x when evaluating the transpose convolution layers from well-known Generative Adversarial Networks (GANs). The implementation of the proposed method for the transpose convolution layers in the EB-GAN model demonstrates significant memory savings of up to 35 MB. 

**Abstract (ZH)**: 针对深度学习应用的转置卷积层优化通过内核分割机制实现。然而，内核分割存在计算额外元素以获得奇数维度输出特征图时需启动线程的问题。为解决这一问题，我们提出了一种统一的内核分割方法，通过使用一个统一的内核执行四个子内核来限制内存和计算资源的使用。实验结果显示，该方法在RTX 2070 GPU（Intel Xeon CPU）上特定数据集上实现了平均2.03倍（3.89倍）的计算加速。消融研究显示，在评估来自知名生成对抗网络（GANs）的转置卷积层时，该方法实现了平均3.5倍的计算加速。将所提方法应用于EB-GAN模型的转置卷积层，显示出高达35 MB的显著内存节省。 

---
