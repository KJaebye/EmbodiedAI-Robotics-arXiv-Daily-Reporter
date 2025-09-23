# DA-Mamba: Dialogue-aware selective state-space model for multimodal engagement estimation 

**Title (ZH)**: DA-Mamba: 基于对话的多模态参与估计的选择性状态空间模型 

**Authors**: Shenwei Kang, Xin Zhang, Wen Liu, Bin Li, Yujie Liu, Bo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17711)  

**Abstract**: Human engagement estimation in conversational scenarios is essential for applications such as adaptive tutoring, remote healthcare assessment, and socially aware human--computer interaction. Engagement is a dynamic, multimodal signal conveyed by facial expressions, speech, gestures, and behavioral cues over time. In this work we introduce DA-Mamba, a dialogue-aware multimodal architecture that replaces attention-heavy dialogue encoders with Mamba-based selective state-space processing to achieve linear time and memory complexity while retaining expressive cross-modal reasoning. We design a Mamba dialogue-aware selective state-space model composed of three core modules: a Dialogue-Aware Encoder, and two Mamba-based fusion mechanisms: Modality-Group Fusion and Partner-Group Fusion, these modules achieve expressive dialogue understanding. Extensive experiments on three standard benchmarks (NoXi, NoXi-Add, and MPIIGI) show that DA-Mamba surpasses prior state-of-the-art (SOTA) methods in concordance correlation coefficient (CCC), while reducing training time and peak memory; these gains enable processing much longer sequences and facilitate real-time deployment in resource-constrained, multi-party conversational settings. The source code will be available at: this https URL. 

**Abstract (ZH)**: 基于对话的多模态人类参与度估计在自适应辅导、远程健康评估及社会意识人机交互等应用中至关重要。参与度是一种随时间变化的多模态信号，通过面部表情、语音、手势和行为线索传递。在此工作中，我们引入了DA-Mamba对话意识多模态架构，通过使用基于Mamba的选择性状态空间处理机制替代注意力密集型对话编码器，以实现线性时间和空间复杂度的同时保持多模态推理的表达性。我们设计了一个包含三个核心模块的Mamba对话意识选择性状态空间模型：对话意识编码器，以及两种基于Mamba的融合机制：模态组融合和伴侣组融合，这些模块实现了对话理解的表达性。在三个标准基准数据集（NoXi、NoXi-Add和MPIIGI）上的广泛实验表明，DA-Mamba在一致性相关系数（CCC）上超越了先前的最先进的（SOTA）方法，同时减少了训练时间和峰值内存；这些改进使得处理更长的序列和在资源受限的多党对话环境中实现实时部署成为可能。源代码将在以下地址提供：this https URL。 

---
# A Multimodal Conversational Assistant for the Characterization of Agricultural Plots from Geospatial Open Data 

**Title (ZH)**: 基于地理空间开放数据的农业地块特征化多模态对话助理 

**Authors**: Juan Cañada, Raúl Alonso, Julio Molleda, Fidel Díez  

**Link**: [PDF](https://arxiv.org/pdf/2509.17544)  

**Abstract**: The increasing availability of open Earth Observation (EO) and agricultural datasets holds great potential for supporting sustainable land management. However, their high technical entry barrier limits accessibility for non-expert users. This study presents an open-source conversational assistant that integrates multimodal retrieval and large language models (LLMs) to enable natural language interaction with heterogeneous agricultural and geospatial data. The proposed architecture combines orthophotos, Sentinel-2 vegetation indices, and user-provided documents through retrieval-augmented generation (RAG), allowing the system to flexibly determine whether to rely on multimodal evidence, textual knowledge, or both in formulating an answer. To assess response quality, we adopt an LLM-as-a-judge methodology using Qwen3-32B in a zero-shot, unsupervised setting, applying direct scoring in a multi-dimensional quantitative evaluation framework. Preliminary results show that the system is capable of generating clear, relevant, and context-aware responses to agricultural queries, while remaining reproducible and scalable across geographic regions. The primary contributions of this work include an architecture for fusing multimodal EO and textual knowledge sources, a demonstration of lowering the barrier to access specialized agricultural information through natural language interaction, and an open and reproducible design. 

**Abstract (ZH)**: 开放地球观测（EO）和农业数据集的不断增加为可持续土地管理提供了巨大潜力。然而，它们的技术门槛限制了非专家用户的访问。本研究介绍了一个开源对话助手，该助手集成了多模态检索和大规模语言模型（LLMs），以实现与异构农业和地理空间数据的自然语言交互。提出的架构通过检索增强生成（RAG）结合正射影像、Sentinel-2植被指数和用户提供的文档，使系统能够灵活地决定在形成答案时是依赖多模态证据、文本知识，还是两者兼而有之。为评估响应质量，我们采用LLM作为裁判的方法，使用Qwen3-32B在零样本、无监督设置下进行评估，并在多维度定量评估框架中直接评分。初步结果表明，该系统能够生成清晰、相关且具有上下文意识的农业查询响应，同时具备可复现性和可扩展性。本文的主要贡献包括多模态EO和文本知识源融合的架构、通过自然语言交互降低访问专业农业信息的门槛的示范，以及开放和可复现的设计。 

---
# UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning 

**Title (ZH)**: 统一对象指引用与分割：面向像素级视觉推理的统一框架 

**Authors**: Ye Liu, Zongyang Ma, Junfu Pu, Zhongang Qi, Yang Wu, Ying Shan, Chang Wen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18094)  

**Abstract**: Recent advances in Large Multi-modal Models (LMMs) have demonstrated their remarkable success as general-purpose multi-modal assistants, with particular focuses on holistic image- and video-language understanding. Conversely, less attention has been given to scaling fine-grained pixel-level understanding capabilities, where the models are expected to realize pixel-level alignment between visual signals and language semantics. Some previous studies have applied LMMs to related tasks such as region-level captioning and referring expression segmentation. However, these models are limited to performing either referring or segmentation tasks independently and fail to integrate these fine-grained perception capabilities into visual reasoning. To bridge this gap, we propose UniPixel, a large multi-modal model capable of flexibly comprehending visual prompt inputs and generating mask-grounded responses. Our model distinguishes itself by seamlessly integrating pixel-level perception with general visual understanding capabilities. Specifically, UniPixel processes visual prompts and generates relevant masks on demand, and performs subsequent reasoning conditioning on these intermediate pointers during inference, thereby enabling fine-grained pixel-level reasoning. The effectiveness of our approach has been verified on 10 benchmarks across a diverse set of tasks, including pixel-level referring/segmentation and object-centric understanding in images/videos. A novel PixelQA task that jointly requires referring, segmentation, and question answering is also designed to verify the flexibility of our method. 

**Abstract (ZH)**: Recent advances in Large Multi-modal Models (LMMs) 已经展示了它们作为通用多模态助手的显著成功，特别是在整体图像和视频语言理解方面的突出表现。然而，较少关注细粒度像素级理解能力的扩展，模型在此方面期望实现视觉信号与语言语义的像素级对齐。虽然一些先前的研究将LMMs应用于区域级描述和引用表达分割等任务，但这些模型只能独立执行参考或分割任务，无法将这些细粒度感知能力融入到视觉推理中。为解决这一问题，我们提出了UniPixel，这是一种能够灵活理解视觉提示输入并生成掩膜导向响应的大规模多模态模型。我们的模型通过无缝整合像素级感知与通用视觉理解能力而区别于其他方法。具体而言，UniPixel根据需要处理视觉提示并生成相关掩膜，在推理过程中根据这些中间指针进行后续的推理，从而实现细粒度的像素级推理。我们在10个跨任务的数据集上验证了该方法的有效性，包括像素级引用/分割和图像/视频中的对象中心理解。我们还设计了一个新的PixelQA任务，该任务联合要求引用、分割和问答，以验证方法的灵活性。 

---
# Trainee Action Recognition through Interaction Analysis in CCATT Mixed-Reality Training 

**Title (ZH)**: 基于CCATT混合现实培训中的交互分析的学徒动作识别 

**Authors**: Divya Mereddy, Marcos Quinones-Grueiro, Ashwin T S, Eduardo Davalos, Gautam Biswas, Kent Etherton, Tyler Davis, Katelyn Kay, Jill Lear, Benjamin Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2509.17888)  

**Abstract**: This study examines how Critical Care Air Transport Team (CCATT) members are trained using mixed-reality simulations that replicate the high-pressure conditions of aeromedical evacuation. Each team - a physician, nurse, and respiratory therapist - must stabilize severely injured soldiers by managing ventilators, IV pumps, and suction devices during flight. Proficient performance requires clinical expertise and cognitive skills, such as situational awareness, rapid decision-making, effective communication, and coordinated task management, all of which must be maintained under stress. Recent advances in simulation and multimodal data analytics enable more objective and comprehensive performance evaluation. In contrast, traditional instructor-led assessments are subjective and may overlook critical events, thereby limiting generalizability and consistency. However, AI-based automated and more objective evaluation metrics still demand human input to train the AI algorithms to assess complex team dynamics in the presence of environmental noise and the need for accurate re-identification in multi-person tracking. To address these challenges, we introduce a systematic, data-driven assessment framework that combines Cognitive Task Analysis (CTA) with Multimodal Learning Analytics (MMLA). We have developed a domain-specific CTA model for CCATT training and a vision-based action recognition pipeline using a fine-tuned Human-Object Interaction model, the Cascade Disentangling Network (CDN), to detect and track trainee-equipment interactions over time. These interactions automatically yield performance indicators (e.g., reaction time, task duration), which are mapped onto a hierarchical CTA model tailored to CCATT operations, enabling interpretable, domain-relevant performance evaluations. 

**Abstract (ZH)**: 本研究探讨了使用混合现实模拟进行训练的重症监护航空转运团队（CCATT）成员如何在模拟航空医疗后送高压条件中接受训练。每个团队——由一名医生、一名护士和一名呼吸治疗师组成——必须在飞行过程中通过管理呼吸机、静脉输液泵和吸痰装置来稳定严重受伤的士兵。熟练的表现需要临床专业知识和认知技能，如情景意识、快速决策能力、有效沟通和协调任务管理，所有这些都必须在压力下维持。最新的发展使模拟和多模态数据分析能够提供更客观和全面的表现评估。相比之下，传统的以教师为主导的评估具有主观性，并且可能会忽略重要的事件，从而限制了一般性和一致性。然而，基于人工智能的自动化和更客观的评估指标仍然需要人工输入，以训练人工智能算法评估复杂团队动态，尤其是在面对环境噪声和多人员追踪精确重新识别需求的情况下。为了解决这些挑战，我们引入了一种系统性的数据驱动评估框架，结合了认知任务分析（CTA）与多模态学习分析（MMLA）。我们为CCATT培训开发了特定领域的CTA模型，并使用细调的人机交互模型——级联解耦网络（CDN）——构建了一种基于视觉的动作识别管道，以检测和跟踪训练员与设备的交互。这些交互自动生成了表现指标（例如，反应时间、任务持续时间），并将这些指标映射到针对CCATT操作定制的层级CTA模型上，从而实现可解释且领域相关的性能评估。 

---
# Qwen3-Omni Technical Report 

**Title (ZH)**: Qwen3-全域技术报告 

**Authors**: Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.17765)  

**Abstract**: We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license. 

**Abstract (ZH)**: Qwen3-Omni：一种在文本、图像、音频和视频上均保持领先性能的统一多模态模型 

---
# Multimodal Medical Image Classification via Synergistic Learning Pre-training 

**Title (ZH)**: 多模态医疗图像分类的协同学习预训练 

**Authors**: Qinghua Lin, Guang-Hai Liu, Zuoyong Li, Yang Li, Yuting Jiang, Xiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17492)  

**Abstract**: Multimodal pathological images are usually in clinical diagnosis, but computer vision-based multimodal image-assisted diagnosis faces challenges with modality fusion, especially in the absence of expert-annotated data. To achieve the modality fusion in multimodal images with label scarcity, we propose a novel ``pretraining + fine-tuning" framework for multimodal semi-supervised medical image classification. Specifically, we propose a synergistic learning pretraining framework of consistency, reconstructive, and aligned learning. By treating one modality as an augmented sample of another modality, we implement a self-supervised learning pre-train, enhancing the baseline model's feature representation capability. Then, we design a fine-tuning method for multimodal fusion. During the fine-tuning stage, we set different encoders to extract features from the original modalities and provide a multimodal fusion encoder for fusion modality. In addition, we propose a distribution shift method for multimodal fusion features, which alleviates the prediction uncertainty and overfitting risks caused by the lack of labeled samples. We conduct extensive experiments on the publicly available gastroscopy image datasets Kvasir and Kvasirv2. Quantitative and qualitative results demonstrate that the proposed method outperforms the current state-of-the-art classification methods. The code will be released at: this https URL. 

**Abstract (ZH)**: 基于多模态病理图像的预训练与微调框架在临床诊断中的应用：面对标注数据稀缺的多模态图像融合 

---
# MVCL-DAF++: Enhancing Multimodal Intent Recognition via Prototype-Aware Contrastive Alignment and Coarse-to-Fine Dynamic Attention Fusion 

**Title (ZH)**: MVCL-DAF++: 基于原型意识对比对齐和由粗到细动态注意力融合的多模态意图识别增强方法 

**Authors**: Haofeng Huang, Yifei Han, Long Zhang, Bin Li, Yangfan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.17446)  

**Abstract**: Multimodal intent recognition (MMIR) suffers from weak semantic grounding and poor robustness under noisy or rare-class conditions. We propose MVCL-DAF++, which extends MVCL-DAF with two key modules: (1) Prototype-aware contrastive alignment, aligning instances to class-level prototypes to enhance semantic consistency; and (2) Coarse-to-fine attention fusion, integrating global modality summaries with token-level features for hierarchical cross-modal interaction. On MIntRec and MIntRec2.0, MVCL-DAF++ achieves new state-of-the-art results, improving rare-class recognition by +1.05\% and +4.18\% WF1, respectively. These results demonstrate the effectiveness of prototype-guided learning and coarse-to-fine fusion for robust multimodal understanding. The source code is available at this https URL. 

**Abstract (ZH)**: 多模态意图识别（MMIR）受到语义关联弱和鲁棒性差的限制。我们提出MVCL-DAF++，它扩展了MVCL-DAF，引入了两个关键模块：（1）原型感知对比对齐，将实例对齐到类别级原型以增强语义一致性；（2）从粗到细注意融合，结合全局模态摘要与标记级特征进行分层跨模态交互。在MIntRec和MIntRec2.0上，MVCL-DAF++取得了新的最先进成果，分别将罕见类别的识别提高1.05%和4.18% WF1。这些结果证明了在鲁棒多模态理解中原型引导学习和从粗到细融合的有效性。源代码可从以下链接获取。 

---
# SAEC: Scene-Aware Enhanced Edge-Cloud Collaborative Industrial Vision Inspection with Multimodal LLM 

**Title (ZH)**: 基于多模态大语言模型的场景aware增强边缘-云协作工业视觉检测SAEC 

**Authors**: Yuhao Tian, Zheming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17136)  

**Abstract**: Industrial vision inspection requires high accuracy under stringent resource constraints, yet existing approaches face a fundamental trade-off. Multimodal LLMs (MLLMs) deliver strong reasoning capabilities but incur prohibitive computational costs, while lightweight edge models often fail on complex cases. In this paper, we present SAEC, a scene-aware enhanced edge-cloud collaborative industrial vision inspection framework with MLLM. The framework is composed of three synergistic components: (1) Efficient MLLM Fine-Tuning for Complex Defect Inspection, (2) Lightweight Multiscale Scene-Complexity Estimation, and (3) Adaptive Edge-Cloud Scheduler. Together, these modules enable robust defect detection by tailoring multimodal reasoning to scene complexity and dynamically balancing computation between edge and cloud resources. Experimental results on MVTec AD and KSDD2 datasets demonstrate that SAEC attains 85.11% and 82.72% accuracy, surpassing Qwen by 22.1% and 20.8%, and LLaVA by 33.3% and 31.6%. It also reduces runtime by up to 22.4% and cuts energy per correct decision by 40%-74%. The code is available at this https URL. 

**Abstract (ZH)**: 基于场景感知增强的边缘-云协作工业视觉检测框架SAEC以人为本工学模型的高效微调、轻量级多尺度场景复杂性估计及自适应边缘-云调度 

---
# Informative Text-Image Alignment for Visual Affordance Learning with Foundation Models 

**Title (ZH)**: 面向视觉功能学习的基础模型驱动的信息性文本-图像对齐 

**Authors**: Qian Zhang, Lin Zhang, Xing Fang, Mingxin Zhang, Zhiyuan Wei, Ran Song, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17074)  

**Abstract**: Visual affordance learning is crucial for robots to understand and interact effectively with the physical world. Recent advances in this field attempt to leverage pre-trained knowledge of vision-language foundation models to learn affordance properties with limited training data, providing a novel paradigm for visual affordance learning. However, these methods overlook the significance of maintaining feature alignment between visual images and language descriptions for identifying affordance areas with textual guidance, and thus may lead to suboptimal results. In this paper, we present an informative framework for text-guided affordance learning, which involves information-based constraints to achieve text-image alignment at feature level. Specifically, we design an affordance mutual information constraint that helps learn appropriate textual prompts and task-oriented visual features simultaneously by maximizing the mutual information between the features of the affordance areas in the input images and the corresponding textual prompts. In addition, we propose an object-level information constraint that maximizes the mutual information between the visual features of a given object and the text features of the category it belongs to. This enables the model to capture high-quality representations for the object, providing more reliable semantic priors for identifying affordance regions. Experimental results on the AGD20K dataset show that the proposed method outperforms existing approaches and achieves the new state-of-the-art in one-shot affordance learning. 

**Abstract (ZH)**: 基于文本指导的视觉拟态学习：特征级文本-图像对齐的信息约束方法 

---
# ME-Mamba: Multi-Expert Mamba with Efficient Knowledge Capture and Fusion for Multimodal Survival Analysis 

**Title (ZH)**: ME-Mamba: 多专家Mamba在多模态生存分析中的高效知识捕获与融合 

**Authors**: Chengsheng Zhang, Linhao Qu, Xiaoyu Liu, Zhijian Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.16900)  

**Abstract**: Survival analysis using whole-slide images (WSIs) is crucial in cancer research. Despite significant successes, pathology images typically only provide slide-level labels, which hinders the learning of discriminative representations from gigapixel WSIs. With the rapid advancement of high-throughput sequencing technologies, multimodal survival analysis integrating pathology images and genomics data has emerged as a promising approach. We propose a Multi-Expert Mamba (ME-Mamba) system that captures discriminative pathological and genomic features while enabling efficient integration of both modalities. This approach achieves complementary information fusion without losing critical information from individual modalities, thereby facilitating accurate cancer survival analysis. Specifically, we first introduce a Pathology Expert and a Genomics Expert to process unimodal data separately. Both experts are designed with Mamba architectures that incorporate conventional scanning and attention-based scanning mechanisms, allowing them to extract discriminative features from long instance sequences containing substantial redundant or irrelevant information. Second, we design a Synergistic Expert responsible for modality fusion. It explicitly learns token-level local correspondences between the two modalities via Optimal Transport, and implicitly enhances distribution consistency through a global cross-modal fusion loss based on Maximum Mean Discrepancy. The fused feature representations are then passed to a mamba backbone for further integration. Through the collaboration of the Pathology Expert, Genomics Expert, and Synergistic Expert, our method achieves stable and accurate survival analysis with relatively low computational complexity. Extensive experimental results on five datasets in The Cancer Genome Atlas (TCGA) demonstrate our state-of-the-art performance. 

**Abstract (ZH)**: 使用全视野图像进行生存分析在癌症研究中至关重要。尽管取得了显著成功，病理图像通常仅提供滑块级标签，这妨碍了从 gigapixel 全视野图像中学习判别表征。随着高通量测序技术的迅猛发展，整合病理图像和基因组数据的多模态生存分析已成为一个有前途的方法。我们提出了一种多专家蜜獾（ME-Mamba）系统，该系统能够捕获判别性的病理和基因组特征，同时促进两者的有效整合。该方法在不丢失单一模态关键信息的前提下实现互补信息融合，从而促进准确的癌症生存分析。具体来说，我们首先引入病理专家和基因组专家分别处理单模态数据。这两种专家都采用蜜獾架构，结合传统的扫描和基于注意力的扫描机制，使其能够从包含大量冗余或无关信息的长实例序列中提取判别性特征。其次，我们设计了一种协同专家负责模态融合。它通过最优传输显式学习两个模态之间的token级局部对应关系，并通过最大均值偏差的全局跨模态融合损失隐式增强分布一致性。融合后的特征表示随后传递给蜜獾骨干网络进行进一步整合。通过病理专家、基因组专家和协同专家的合作，我们的方法实现了相对较低的计算复杂度下的稳定和准确的生存分析。在癌症基因组图谱（TCGA）的五个数据集上的广泛实验结果证明了我们方法的最优性能。 

---
# Learning from Gene Names, Expression Values and Images: Contrastive Masked Text-Image Pretraining for Spatial Transcriptomics Representation Learning 

**Title (ZH)**: 从基因名称、表达值和图像中学习：空间转录组学表示学习的对比掩码文本-图像预训练 

**Authors**: Jiahe Qian, Yaoyu Fang, Ziqiao Weng, Xinkun Wang, Lee A. Cooper, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.16892)  

**Abstract**: Spatial transcriptomics aims to connect high-resolution histology images with spatially resolved gene expression. To achieve better performance on downstream tasks such as gene expression prediction, large-scale pre-training is required to obtain generalisable representations that can bridge histology and transcriptomics across tissues, protocols, and laboratories. Existing cross-modal pre-training approaches for spatial transcriptomics rely on either gene names or expression values in isolation, which strips the gene branch of essential semantics and breaks the association between each gene and its quantitative magnitude. In addition, by restricting supervision to image-text alignment, these methods ignore intrinsic visual cues that are critical for learning robust image features. We present CoMTIP, the first Contrastive Masked Text-Image Pretraining framework that jointly learns from images, gene names, and expression values while capturing fine-grained visual context for spatial transcriptomics. The vision branch uses Masked Feature Modeling to reconstruct occluded patches and learn context-aware image embeddings. The text branch applies a scalable Gene-Text Encoder that processes all gene sentences in parallel, enriches each gene and its numerical value with dedicated embeddings, and employs Pair-aware Adversarial Training (PAAT) to preserve correct gene-value associations. Image and text representations are aligned in a shared InfoNCE-optimised space. Experiments on public spatial transcriptomics datasets show that CoMTIP not only surpasses previous methods on diverse downstream tasks but also achieves zero-shot gene expression prediction, a capability that existing approaches do not provide. 

**Abstract (ZH)**: 基于文本-图像的对比掩码预训练框架：面向空间转录组学的细粒度视觉上下文学习 

---
# Seeing Culture: A Benchmark for Visual Reasoning and Grounding 

**Title (ZH)**: 睹文化：视觉推理与语义 grounding 的基准 

**Authors**: Burak Satar, Zhixin Ma, Patrick A. Irawan, Wilfried A. Mulyawan, Jing Jiang, Ee-Peng Lim, Chong-Wah Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.16517)  

**Abstract**: Multimodal vision-language models (VLMs) have made substantial progress in various tasks that require a combined understanding of visual and textual content, particularly in cultural understanding tasks, with the emergence of new cultural datasets. However, these datasets frequently fall short of providing cultural reasoning while underrepresenting many cultures. In this paper, we introduce the Seeing Culture Benchmark (SCB), focusing on cultural reasoning with a novel approach that requires VLMs to reason on culturally rich images in two stages: i) selecting the correct visual option with multiple-choice visual question answering (VQA), and ii) segmenting the relevant cultural artifact as evidence of reasoning. Visual options in the first stage are systematically organized into three types: those originating from the same country, those from different countries, or a mixed group. Notably, all options are derived from a singular category for each type. Progression to the second stage occurs only after a correct visual option is chosen. The SCB benchmark comprises 1,065 images that capture 138 cultural artifacts across five categories from seven Southeast Asia countries, whose diverse cultures are often overlooked, accompanied by 3,178 questions, of which 1,093 are unique and meticulously curated by human annotators. Our evaluation of various VLMs reveals the complexities involved in cross-modal cultural reasoning and highlights the disparity between visual reasoning and spatial grounding in culturally nuanced scenarios. The SCB serves as a crucial benchmark for identifying these shortcomings, thereby guiding future developments in the field of cultural reasoning. this https URL 

**Abstract (ZH)**: 多模态Vision-Language模型（VLMs）在文化理解任务中取得了显著进展，特别是在新兴文化数据集的支持下，这些任务要求对视觉和文本内容进行综合理解。然而，这些数据集在提供文化推理方面经常不足，并且未能代表许多文化。本文介绍了文化理解基准（SCB），这是一种新的方法，要求VLMs在两个阶段中对丰富文化内容的图像进行推理：i) 使用多项选择视觉问答（VQA）选择正确的视觉选项，ii) 对相关文化 artifact 进行分割作为推理证据。在第一个阶段中，视觉选项系统性地分为三类：同一国家的选项、不同国家的选项或混合组。值得注意的是，每种类型的所有选项都源自同一个类别。仅当正确选择视觉选项后，才能进入第二个阶段。SCB基准包括1,065张图像，涵盖来自七个东南亚国家（通常文化多样性和丰富性未得到充分关注）的五大类别中的138个文化 artifact，共有3,178个问题，其中1,093个为独特问题，由人工注释员精心策划。我们对各种VLM的评估揭示了跨模态文化推理的复杂性，并突显了在文化细致入微的情境下视觉推理与空间定位之间的差距。SCB作为识别这些不足的关键基准，将指导该领域未来的发展。 

---
