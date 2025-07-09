# Pedestrian Intention Prediction via Vision-Language Foundation Models 

**Title (ZH)**: 基于视觉-语言基础模型的行人意图预测 

**Authors**: Mohsen Azarmi, Mahdi Rezaei, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04141)  

**Abstract**: Prediction of pedestrian crossing intention is a critical function in autonomous vehicles. Conventional vision-based methods of crossing intention prediction often struggle with generalizability, context understanding, and causal reasoning. This study explores the potential of vision-language foundation models (VLFMs) for predicting pedestrian crossing intentions by integrating multimodal data through hierarchical prompt templates. The methodology incorporates contextual information, including visual frames, physical cues observations, and ego-vehicle dynamics, into systematically refined prompts to guide VLFMs effectively in intention prediction. Experiments were conducted on three common datasets-JAAD, PIE, and FU-PIP. Results demonstrate that incorporating vehicle speed, its variations over time, and time-conscious prompts significantly enhances the prediction accuracy up to 19.8%. Additionally, optimised prompts generated via an automatic prompt engineering framework yielded 12.5% further accuracy gains. These findings highlight the superior performance of VLFMs compared to conventional vision-based models, offering enhanced generalisation and contextual understanding for autonomous driving applications. 

**Abstract (ZH)**: 基于视觉-语言基础模型的行人过街意图预测研究 

---
# From Query to Explanation: Uni-RAG for Multi-Modal Retrieval-Augmented Learning in STEM 

**Title (ZH)**: 从查询到解释：Uni-RAG在STEM领域多模态检索增强学习中的应用 

**Authors**: Xinyi Wu, Yanhao Jia, Luwei Xiao, Shuai Zhao, Fengkuang Chiang, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2507.03868)  

**Abstract**: In AI-facilitated teaching, leveraging various query styles to interpret abstract educational content is crucial for delivering effective and accessible learning experiences. However, existing retrieval systems predominantly focus on natural text-image matching and lack the capacity to address the diversity and ambiguity inherent in real-world educational scenarios. To address this limitation, we develop a lightweight and efficient multi-modal retrieval module, named Uni-Retrieval, which extracts query-style prototypes and dynamically matches them with tokens from a continually updated Prompt Bank. This Prompt Bank encodes and stores domain-specific knowledge by leveraging a Mixture-of-Expert Low-Rank Adaptation (MoE-LoRA) module and can be adapted to enhance Uni-Retrieval's capability to accommodate unseen query types at test time. To enable natural language educational content generation, we integrate the original Uni-Retrieval with a compact instruction-tuned language model, forming a complete retrieval-augmented generation pipeline named Uni-RAG. Given a style-conditioned query, Uni-RAG first retrieves relevant educational materials and then generates human-readable explanations, feedback, or instructional content aligned with the learning objective. Experimental results on SER and other multi-modal benchmarks show that Uni-RAG outperforms baseline retrieval and RAG systems in both retrieval accuracy and generation quality, while maintaining low computational cost. Our framework provides a scalable, pedagogically grounded solution for intelligent educational systems, bridging retrieval and generation to support personalized, explainable, and efficient learning assistance across diverse STEM scenarios. 

**Abstract (ZH)**: 基于AI辅助教学中的多模态查询风格检索模块Uni-Retrieval及其应用 

---
# INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling 

**Title (ZH)**: INTER: 通过交互指导采样减轻大型视觉-语言模型的幻觉问题 

**Authors**: Xin Dong, Shichao Dong, Jin Wang, Jing Huang, Li Zhou, Zenghui Sun, Lihua Jing, Jingsong Lan, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.05056)  

**Abstract**: Hallucinations in large vision-language models (LVLMs) pose significant challenges for real-world applications, as LVLMs may generate responses that appear plausible yet remain inconsistent with the associated visual content. This issue rarely occurs in human cognition. We argue that this discrepancy arises from humans' ability to effectively leverage multimodal interaction information in data samples. Specifically, humans typically first gather multimodal information, analyze the interactions across modalities for understanding, and then express their understanding through language. Motivated by this observation, we conduct extensive experiments on popular LVLMs and obtained insights that surprisingly reveal human-like, though less pronounced, cognitive behavior of LVLMs on multimodal samples. Building on these findings, we further propose \textbf{INTER}: \textbf{Inter}action Guidance Sampling, a novel training-free algorithm that mitigate hallucinations without requiring additional data. Specifically, INTER explicitly guides LVLMs to effectively reapply their understanding of multimodal interaction information when generating responses, thereby reducing potential hallucinations. On six benchmarks including VQA and image captioning tasks, INTER achieves an average improvement of up to 3.4\% on five LVLMs compared to the state-of-the-art decoding strategy. The code will be released when the paper is accepted. 

**Abstract (ZH)**: 大型视觉-语言模型中的幻觉对实际应用构成了重大挑战，因为这些模型可能生成看似合理但实际上与关联的视觉内容不一致的响应。这一问题在人类认知中很少出现。我们认为这种差异源于人类能够有效利用数据样本中的多模态交互信息。具体来说，人类通常首先收集多模态信息，分析模态间的交互以进行理解，然后通过语言表达理解。受此观察的启发，我们在流行的大型视觉-语言模型上进行了广泛的实验，并获得了一些令人惊讶的见解，即大型视觉-语言模型在多模态样本上表现出类似人类，尽管不那么明显的认知行为。基于这些发现，我们进一步提出了INTER：交互引导采样，这是一种无需额外数据的创新训练算法，旨在减轻幻觉。具体而言，INTER 显式地指导大型视觉-语言模型在生成响应时有效重新应用对多模态交互信息的理解，从而减少潜在的幻觉。在包括VQA和图像字幕任务在内的六个基准上，INTER 相较于最先进的解码策略，平均提高了多达3.4%。论文被接受后将发布代码。 

---
# Multi-modal Representations for Fine-grained Multi-label Critical View of Safety Recognition 

**Title (ZH)**: 多模态表示在细粒度多标签安全关键视角识别中的应用 

**Authors**: Britty Baby, Vinkle Srivastav, Pooja P. Jain, Kun Yuan, Pietro Mascagni, Nicolas Padoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05007)  

**Abstract**: The Critical View of Safety (CVS) is crucial for safe laparoscopic cholecystectomy, yet assessing CVS criteria remains a complex and challenging task, even for experts. Traditional models for CVS recognition depend on vision-only models learning with costly, labor-intensive spatial annotations. This study investigates how text can be harnessed as a powerful tool for both training and inference in multi-modal surgical foundation models to automate CVS recognition. Unlike many existing multi-modal models, which are primarily adapted for multi-class classification, CVS recognition requires a multi-label framework. Zero-shot evaluation of existing multi-modal surgical models shows a significant performance gap for this task. To address this, we propose CVS-AdaptNet, a multi-label adaptation strategy that enhances fine-grained, binary classification across multiple labels by aligning image embeddings with textual descriptions of each CVS criterion using positive and negative prompts. By adapting PeskaVLP, a state-of-the-art surgical foundation model, on the Endoscapes-CVS201 dataset, CVS-AdaptNet achieves 57.6 mAP, improving over the ResNet50 image-only baseline (51.5 mAP) by 6 points. Our results show that CVS-AdaptNet's multi-label, multi-modal framework, enhanced by textual prompts, boosts CVS recognition over image-only methods. We also propose text-specific inference methods, that helps in analysing the image-text alignment. While further work is needed to match state-of-the-art spatial annotation-based methods, this approach highlights the potential of adapting generalist models to specialized surgical tasks. Code: this https URL 

**Abstract (ZH)**: 基于文本的多模态手术基础模型在安全评估中的应用：CVS-AdaptNet研究 

---
# Hear-Your-Click: Interactive Video-to-Audio Generation via Object-aware Contrastive Audio-Visual Fine-tuning 

**Title (ZH)**: 听你的点击：基于对象aware对比音频-视觉微调的交互式视频到音频生成 

**Authors**: Yingshan Liang, Keyu Fan, Zhicheng Du, Yiran Wang, Qingyang Shi, Xinyu Zhang, Jiasheng Lu, Peiwu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04959)  

**Abstract**: Video-to-audio (V2A) generation shows great potential in fields such as film production. Despite significant advances, current V2A methods, which rely on global video information, struggle with complex scenes and often fail to generate audio tailored to specific objects or regions in the videos. To address these limitations, we introduce Hear-Your-Click, an interactive V2A framework that enables users to generate sounds for specific objects in the videos by simply clicking on the frame. To achieve this, we propose Object-aware Contrastive Audio-Visual Fine-tuning (OCAV) with a Mask-guided Visual Encoder (MVE) to obtain object-level visual features aligned with corresponding audio segments. Furthermore, we tailor two data augmentation strategies: Random Video Stitching (RVS) and Mask-guided Loudness Modulation (MLM), aimed at enhancing the model's sensitivity to the segmented objects. To effectively measure the audio-visual correspondence, we design a new evaluation metric, the CAV score, for evaluation. Extensive experiments demonstrate that our framework offers more precise control and improved generation performance across various metrics. Project Page: this https URL 

**Abstract (ZH)**: 视频到音频（V2A）生成在电影制作等领域展现出巨大的潜力。尽管取得了显著进展，当前依赖全局视频信息的V2A方法在处理复杂场景时常常无法生成针对视频中特定对象或区域的定制化音频。为解决这些局限性，我们引入了Hear-Your-Click，这是一个交互式的V2A框架，允许用户通过单击帧来生成视频中特定对象的声音。为此，我们提出了对象感知对比音频-视觉微调（OCAV）和掩码引导视觉编码器（MVE），以获得与相应音频段对齐的对象级视觉特征。此外，我们定制了两种数据增强策略：随机视频拼接（RVS）和掩码引导响度调制（MLM），旨在提高模型对分割对象的敏感度。为了有效衡量音频-视觉对应关系，我们设计了一种新的评估指标——CAV分数。广泛的实验表明，我们的框架在多种指标上提供了更精确的控制和改进的生成性能。项目页面: this https URL。 

---
# EXPOTION: Facial Expression and Motion Control for Multimodal Music Generation 

**Title (ZH)**: EXPOTION：多模态音乐生成中的面部表情和动作控制 

**Authors**: Fathinah Izzati, Xinyue Li, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04955)  

**Abstract**: We propose Expotion (Facial Expression and Motion Control for Multimodal Music Generation), a generative model leveraging multimodal visual controls - specifically, human facial expressions and upper-body motion - as well as text prompts to produce expressive and temporally accurate music. We adopt parameter-efficient fine-tuning (PEFT) on the pretrained text-to-music generation model, enabling fine-grained adaptation to the multimodal controls using a small dataset. To ensure precise synchronization between video and music, we introduce a temporal smoothing strategy to align multiple modalities. Experiments demonstrate that integrating visual features alongside textual descriptions enhances the overall quality of generated music in terms of musicality, creativity, beat-tempo consistency, temporal alignment with the video, and text adherence, surpassing both proposed baselines and existing state-of-the-art video-to-music generation models. Additionally, we introduce a novel dataset consisting of 7 hours of synchronized video recordings capturing expressive facial and upper-body gestures aligned with corresponding music, providing significant potential for future research in multimodal and interactive music generation. 

**Abstract (ZH)**: Expotion：基于多模态视觉控制的音乐生成模型 

---
# SPATIA: Multimodal Model for Prediction and Generation of Spatial Cell Phenotypes 

**Title (ZH)**: SPATIA：多模态模型用于空间细胞表型的预测与生成 

**Authors**: Zhenglun Kong, Mufan Qiu, John Boesen, Xiang Lin, Sukwon Yun, Tianlong Chen, Manolis Kellis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2507.04704)  

**Abstract**: Understanding how cellular morphology, gene expression, and spatial organization jointly shape tissue function is a central challenge in biology. Image-based spatial transcriptomics technologies now provide high-resolution measurements of cell images and gene expression profiles, but machine learning methods typically analyze these modalities in isolation or at limited resolution. We address the problem of learning unified, spatially aware representations that integrate cell morphology, gene expression, and spatial context across biological scales. This requires models that can operate at single-cell resolution, reason across spatial neighborhoods, and generalize to whole-slide tissue organization. Here, we introduce SPATIA, a multi-scale generative and predictive model for spatial transcriptomics. SPATIA learns cell-level embeddings by fusing image-derived morphological tokens and transcriptomic vector tokens using cross-attention and then aggregates them at niche and tissue levels using transformer modules to capture spatial dependencies. SPATIA incorporates token merging in its generative diffusion decoder to synthesize high-resolution cell images conditioned on gene expression. We assembled a multi-scale dataset consisting of 17 million cell-gene pairs, 1 million niche-gene pairs, and 10,000 tissue-gene pairs across 49 donors, 17 tissue types, and 12 disease states. We benchmark SPATIA against 13 existing models across 12 individual tasks, which span several categories including cell annotation, cell clustering, gene imputation, cross-modal prediction, and image generation. SPATIA achieves improved performance over all baselines and generates realistic cell morphologies that reflect transcriptomic perturbations. 

**Abstract (ZH)**: 基于图像的空间转录组学多尺度生成和预测模型SPATIA 

---
# What's Making That Sound Right Now? Video-centric Audio-Visual Localization 

**Title (ZH)**: 当前是什么声音？基于视频的音视频定位 

**Authors**: Hahyeon Choi, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2507.04667)  

**Abstract**: Audio-Visual Localization (AVL) aims to identify sound-emitting sources within a visual scene. However, existing studies focus on image-level audio-visual associations, failing to capture temporal dynamics. Moreover, they assume simplified scenarios where sound sources are always visible and involve only a single object. To address these limitations, we propose AVATAR, a video-centric AVL benchmark that incorporates high-resolution temporal information. AVATAR introduces four distinct scenarios -- Single-sound, Mixed-sound, Multi-entity, and Off-screen -- enabling a more comprehensive evaluation of AVL models. Additionally, we present TAVLO, a novel video-centric AVL model that explicitly integrates temporal information. Experimental results show that conventional methods struggle to track temporal variations due to their reliance on global audio features and frame-level mappings. In contrast, TAVLO achieves robust and precise audio-visual alignment by leveraging high-resolution temporal modeling. Our work empirically demonstrates the importance of temporal dynamics in AVL and establishes a new standard for video-centric audio-visual localization. 

**Abstract (ZH)**: 音频-视觉定位（AVL）旨在识别视觉场景中的声源。然而，现有研究关注图像级别的音频-视觉关联，未能捕捉时间动态性。此外，它们假设声源始终可见且仅涉及单一物体的简化场景。为解决这些限制，我们提出AVATAR，一个以视频为中心的AVL基准，包含高分辨率时间信息。AVATAR引入了四种不同的场景——单声源、混合声源、多实体和离屏——以实现更全面的AVL模型评估。此外，我们还提出了TAVLO，一种以视频为中心的AVL模型，明确整合时间信息。实验结果表明，传统方法由于依赖全局音频特征和帧级映射，难以跟踪时间变化。相比之下，TAVLO通过利用高分辨率的时间建模实现稳健且精确的音频-视觉对齐。我们的工作从经验上证明了时间动态性在AVL中的重要性，并建立了以视频为中心的音频-视觉定位的新标准。 

---
# MVL-Loc: Leveraging Vision-Language Model for Generalizable Multi-Scene Camera Relocalization 

**Title (ZH)**: MVL-Loc：利用视觉-语言模型实现可泛化的多场景相机重新定位 

**Authors**: Zhendong Xiao, Wu Wei, Shujie Ji, Shan Yang, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04509)  

**Abstract**: Camera relocalization, a cornerstone capability of modern computer vision, accurately determines a camera's position and orientation (6-DoF) from images and is essential for applications in augmented reality (AR), mixed reality (MR), autonomous driving, delivery drones, and robotic navigation. Unlike traditional deep learning-based methods that regress camera pose from images in a single scene, which often lack generalization and robustness in diverse environments, we propose MVL-Loc, a novel end-to-end multi-scene 6-DoF camera relocalization framework. MVL-Loc leverages pretrained world knowledge from vision-language models (VLMs) and incorporates multimodal data to generalize across both indoor and outdoor settings. Furthermore, natural language is employed as a directive tool to guide the multi-scene learning process, facilitating semantic understanding of complex scenes and capturing spatial relationships among objects. Extensive experiments on the 7Scenes and Cambridge Landmarks datasets demonstrate MVL-Loc's robustness and state-of-the-art performance in real-world multi-scene camera relocalization, with improved accuracy in both positional and orientational estimates. 

**Abstract (ZH)**: 多场景6-自由度相机重定位框架MVL-Loc 

---
# Multimedia Verification Through Multi-Agent Deep Research Multimodal Large Language Models 

**Title (ZH)**: 多代理深度研究驱动的多媒体验证通过多模态大型语言模型 

**Authors**: Huy Hoan Le, Van Sy Thinh Nguyen, Thi Le Chi Dang, Vo Thanh Khang Nguyen, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04410)  

**Abstract**: This paper presents our submission to the ACMMM25 - Grand Challenge on Multimedia Verification. We developed a multi-agent verification system that combines Multimodal Large Language Models (MLLMs) with specialized verification tools to detect multimedia misinformation. Our system operates through six stages: raw data processing, planning, information extraction, deep research, evidence collection, and report generation. The core Deep Researcher Agent employs four tools: reverse image search, metadata analysis, fact-checking databases, and verified news processing that extracts spatial, temporal, attribution, and motivational context. We demonstrate our approach on a challenge dataset sample involving complex multimedia content. Our system successfully verified content authenticity, extracted precise geolocation and timing information, and traced source attribution across multiple platforms, effectively addressing real-world multimedia verification scenarios. 

**Abstract (ZH)**: 本文呈现了我们参加ACMMM25多媒体验证大赛的提交内容。我们开发了一个结合多模态大型语言模型和专门验证工具的多代理验证系统，以检测多媒体错误信息。该系统通过六 stages：原始数据处理、计划、信息提取、深度研究、证据收集和报告生成。核心深度研究员代理使用了四种工具：反向图像搜索、元数据分析、事实核查数据库以及提取空间、时间、归属和动机上下文的可信新闻处理。我们利用一个包含复杂多媒体内容的挑战数据集样本展示了我们的方法。我们的系统成功验证了内容的真实性，提取了精确的地理定位和时间信息，并跨多个平台追踪了来源归属，有效应对了实际的多媒体验证场景。 

---
# M$^3$-Med: A Benchmark for Multi-lingual, Multi-modal, and Multi-hop Reasoning in Medical Instructional Video Understanding 

**Title (ZH)**: M$^3$-Med：多语言、多模态和多跳推理在医学教学视频理解中的基准测试 

**Authors**: Shenxi Liu, Kan Li, Mingyang Zhao, Yuhang Tian, Bin Li, Shoujun Zhou, Hongliang Li, Fuxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04289)  

**Abstract**: With the rapid progress of artificial intelligence (AI) in multi-modal understanding, there is increasing potential for video comprehension technologies to support professional domains such as medical education. However, existing benchmarks suffer from two primary limitations: (1) Linguistic Singularity: they are largely confined to English, neglecting the need for multilingual resources; and (2) Shallow Reasoning: their questions are often designed for surface-level information retrieval, failing to properly assess deep multi-modal integration. To address these limitations, we present M3-Med, the first benchmark for Multi-lingual, Multi-modal, and Multi-hop reasoning in Medical instructional video understanding. M3-Med consists of medical questions paired with corresponding video segments, annotated by a team of medical experts. A key innovation of M3-Med is its multi-hop reasoning task, which requires a model to first locate a key entity in the text, then find corresponding visual evidence in the video, and finally synthesize information across both modalities to derive the answer. This design moves beyond simple text matching and poses a substantial challenge to a model's deep cross-modal understanding capabilities. We define two tasks: Temporal Answer Grounding in Single Video (TAGSV) and Temporal Answer Grounding in Video Corpus (TAGVC). We evaluated several state-of-the-art models and Large Language Models (LLMs) on M3-Med. The results reveal a significant performance gap between all models and human experts, especially on the complex multi-hop questions where model performance drops sharply. M3-Med effectively highlights the current limitations of AI models in deep cross-modal reasoning within specialized domains and provides a new direction for future research. 

**Abstract (ZH)**: 多语言、多模态和多跳推理的医疗指令视频理解基准（M3-Med） 

---
# ZERO: Multi-modal Prompt-based Visual Grounding 

**Title (ZH)**: ZERO: 多模态提示驱动的视觉定位 

**Authors**: Sangbum Choi, Kyeongryeol Go  

**Link**: [PDF](https://arxiv.org/pdf/2507.04270)  

**Abstract**: Recent advances in artificial intelligence have led to the emergence of foundation models, large-scale pre-trained neural networks that serve as versatile starting points for a wide range of downstream tasks. In this work, we present ZERO, a zero-shot multi-prompt object detection model specifically designed for robust, production-ready deployment across diverse industrial domains. ZERO integrates direct image input with multiple user-defined prompts, which can include both textual and visual cues, and processes them through dedicated encoders to generate accurate detection outputs. The model architecture is optimized for scalability, with a total of 1.033 TFLOPS and 622.346 million parameters, and is trained using a domain-specific image database exceeding one billion images. For the CVPR 2025 Foundational Few-Shot Object Detection (FSOD) Challenge, we introduce a domain-specific fine-tuning strategy that emphasizes prompt diversity and conservative pseudo-labeling, enabling effective adaptation to new domains with minimal supervision. Our approach demonstrates practical advantages in flexibility, efficiency, and real-world applicability, achieving strong performance on the RF20VL-fsod benchmark despite limited annotation budgets. The results highlight the potential of prompt-driven, data-centric AI for scalable and adaptive object detection in dynamic industrial environments. 

**Abstract (ZH)**: 近期人工智能的进展催生了基础模型，即大规模预训练神经网络，这些模型可以作为广泛下游任务的多功能起点。在本工作中，我们提出了ZERO，这是一种零样本多提示对象检测模型，专门设计用于在多样化的工业领域中进行稳健的生产部署。ZERO结合了直接图像输入与多个用户定义的提示，这些提示可以包括文本和视觉线索，并通过专用编码器生成准确的检测输出。模型架构针对可扩展性进行了优化，总共有1.033 TFLOPS和622.346百万参数，并使用包含超过十亿张图像的领域专用图像数据库进行训练。在CVPR 2025基础少量样本对象检测（FSOD）挑战中，我们提出了一种领域专用的微调策略，强调提示多样性并采用保守的伪标签方法，以在最少监督的情况下有效适应新领域。我们的方法在灵活性、效率和实际应用性方面表现出实际优势，尽管在注释预算有限的情况下仍取得了出色性能。结果突显了以提示驱动的数据为中心的人工智能在动态工业环境中的可扩展和适应性对象检测潜力。 

---
# Hierarchical Semantic-Visual Fusion of Visible and Near-infrared Images for Long-range Haze Removal 

**Title (ZH)**: 可见光和近红外图像分层语义-视觉融合在长距离消雾中的应用 

**Authors**: Yi Li, Xiaoxiong Wang, Jiawei Wang, Yi Chang, Kai Cao, Luxin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03893)  

**Abstract**: While image dehazing has advanced substantially in the past decade, most efforts have focused on short-range scenarios, leaving long-range haze removal under-explored. As distance increases, intensified scattering leads to severe haze and signal loss, making it impractical to recover distant details solely from visible images. Near-infrared, with superior fog penetration, offers critical complementary cues through multimodal fusion. However, existing methods focus on content integration while often neglecting haze embedded in visible images, leading to results with residual haze. In this work, we argue that the infrared and visible modalities not only provide complementary low-level visual features, but also share high-level semantic consistency. Motivated by this, we propose a Hierarchical Semantic-Visual Fusion (HSVF) framework, comprising a semantic stream to reconstruct haze-free scenes and a visual stream to incorporate structural details from the near-infrared modality. The semantic stream first acquires haze-robust semantic prediction by aligning modality-invariant intrinsic representations. Then the shared semantics act as strong priors to restore clear and high-contrast distant scenes under severe haze degradation. In parallel, the visual stream focuses on recovering lost structural details from near-infrared by fusing complementary cues from both visible and near-infrared images. Through the cooperation of dual streams, HSVF produces results that exhibit both high-contrast scenes and rich texture details. Moreover, we introduce a novel pixel-aligned visible-infrared haze dataset with semantic labels to facilitate benchmarking. Extensive experiments demonstrate the superiority of our method over state-of-the-art approaches in real-world long-range haze removal. 

**Abstract (ZH)**: 近红外与可见光多模态层次语义视融合的长距离去雾方法 

---
# Multimodal Alignment with Cross-Attentive GRUs for Fine-Grained Video Understanding 

**Title (ZH)**: 多模态注意力GRUs的细粒度视频理解 

**Authors**: Namho Kim, Junhwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.03531)  

**Abstract**: Fine-grained video classification requires understanding complex spatio-temporal and semantic cues that often exceed the capacity of a single modality. In this paper, we propose a multimodal framework that fuses video, image, and text representations using GRU-based sequence encoders and cross-modal attention mechanisms. The model is trained using a combination of classification or regression loss, depending on the task, and is further regularized through feature-level augmentation and autoencoding techniques. To evaluate the generality of our framework, we conduct experiments on two challenging benchmarks: the DVD dataset for real-world violence detection and the Aff-Wild2 dataset for valence-arousal estimation. Our results demonstrate that the proposed fusion strategy significantly outperforms unimodal baselines, with cross-attention and feature augmentation contributing notably to robustness and performance. 

**Abstract (ZH)**: 细粒度视频分类需要理解复杂的时空和语义线索，这些线索通常超出了单一模态的能力。本文提出了一种多模态框架，该框架使用基于GRU的序列编码器和跨模态注意机制融合视频、图像和文本表示。该模型根据不同任务使用分类或回归损失进行训练，并通过特征层面的增强和自编码技术进一步正则化。为了评估我们框架的一般性，我们在两个具有挑战性的基准上进行了实验：用于真实世界暴力检测的DVD数据集和用于情感估值的Aff-Wild2数据集。实验结果表明，提出的融合策略显著优于单模态基线，跨注意和特征增强对鲁棒性和性能的提升尤为显著。 

---
# Automated Grading of Students' Handwritten Graphs: A Comparison of Meta-Learning and Vision-Large Language Models 

**Title (ZH)**: 基于元学习和视觉大语言模型的学生手写图表的自动评分比较 

**Authors**: Behnam Parsaeifard, Martin Hlosta, Per Bergamin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03056)  

**Abstract**: With the rise of online learning, the demand for efficient and consistent assessment in mathematics has significantly increased over the past decade. Machine Learning (ML), particularly Natural Language Processing (NLP), has been widely used for autograding student responses, particularly those involving text and/or mathematical expressions. However, there has been limited research on autograding responses involving students' handwritten graphs, despite their prevalence in Science, Technology, Engineering, and Mathematics (STEM) curricula. In this study, we implement multimodal meta-learning models for autograding images containing students' handwritten graphs and text. We further compare the performance of Vision Large Language Models (VLLMs) with these specially trained metalearning models. Our results, evaluated on a real-world dataset collected from our institution, show that the best-performing meta-learning models outperform VLLMs in 2-way classification tasks. In contrast, in more complex 3-way classification tasks, the best-performing VLLMs slightly outperform the meta-learning models. While VLLMs show promising results, their reliability and practical applicability remain uncertain and require further investigation. 

**Abstract (ZH)**: 随着在线学习的兴起，过去十年中对数学高效一致评估的需求显著增加。机器学习（ML），特别是自然语言处理（NLP），已被广泛用于自动批改学生的答案，尤其是涉及文本和/或数学表达式的情况。然而，对于涉及学生手绘图表的答案批改，尽管这类图表在STEM课程中普遍存在，关于这方面的研究仍相对有限。在本研究中，我们实现了多模态元学习模型来自动批改包含学生手绘图表和文本的图像。我们进一步将视觉大型语言模型（VLLMs）的性能与其特别训练的元学习模型进行了比较。我们的结果，在我们机构收集的真实数据集上评估，显示最佳的元学习模型在二分类任务中优于VLLMs；而在更复杂的三分类任务中，最佳的VLLMs略微优于元学习模型。尽管VLLMs展现出良好的前景，但它们的可靠性和实际应用性仍有待进一步调查。 

---
