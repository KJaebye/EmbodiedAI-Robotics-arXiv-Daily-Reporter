# Enhancing multimodal analogical reasoning with Logic Augmented Generation 

**Title (ZH)**: 增强多模态类比推理的逻辑增强生成方法 

**Authors**: Anna Sofia Lippolis, Andrea Giovanni Nuzzolese, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11190)  

**Abstract**: Recent advances in Large Language Models have demonstrated their capabilities across a variety of tasks. However, automatically extracting implicit knowledge from natural language remains a significant challenge, as machines lack active experience with the physical world. Given this scenario, semantic knowledge graphs can serve as conceptual spaces that guide the automated text generation reasoning process to achieve more efficient and explainable results. In this paper, we apply a logic-augmented generation (LAG) framework that leverages the explicit representation of a text through a semantic knowledge graph and applies it in combination with prompt heuristics to elicit implicit analogical connections. This method generates extended knowledge graph triples representing implicit meaning, enabling systems to reason on unlabeled multimodal data regardless of the domain. We validate our work through three metaphor detection and understanding tasks across four datasets, as they require deep analogical reasoning capabilities. The results show that this integrated approach surpasses current baselines, performs better than humans in understanding visual metaphors, and enables more explainable reasoning processes, though still has inherent limitations in metaphor understanding, especially for domain-specific metaphors. Furthermore, we propose a thorough error analysis, discussing issues with metaphorical annotations and current evaluation methods. 

**Abstract (ZH)**: 近期大型语言模型的进展展示了其在各种任务中的能力。然而，自动从自然语言中提取隐性知识仍然是一个重要的挑战，因为机器缺乏对物理世界的主动经验。在这种情况下，语义知识图可以作为概念空间，指导自动文本生成的推理过程，以实现更高效和可解释的结果。在本文中，我们应用了一种逻辑增强生成（LAG）框架，该框架通过语义知识图的显式表示来增强文本，并结合提示启发式方法来引发隐性的类比连接。该方法生成表示隐含意义的扩展知识图三元组，使系统能够对未标记的多模态数据进行推理，而不论其领域如何。我们通过三个隐喻检测和理解任务在四个数据集中验证了我们的工作，这些任务需要深入的类比推理能力。结果表明，这种集成方法超越了当前基线，优于人类在理解视觉隐喻方面的表现，并允许更可解释的推理过程，尽管在隐喻理解方面仍然存在固有的局限性，特别是在领域特定隐喻方面。此外，我们提出了一种彻底的错误分析，讨论了隐喻注释和当前评估方法的问题。 

---
# TerraMind: Large-Scale Generative Multimodality for Earth Observation 

**Title (ZH)**: TerraMind: 大规模生成多模态地球观测 

**Authors**: Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, Rahul Ramachandran, Paolo Fraccaro, Thomas Brunschwiler, Gabriele Cavallaro, Juan Bernabe-Moreno, Nicolas Longépé  

**Link**: [PDF](https://arxiv.org/pdf/2504.11171)  

**Abstract**: We present TerraMind, the first any-to-any generative, multimodal foundation model for Earth observation (EO). Unlike other multimodal models, TerraMind is pretrained on dual-scale representations combining both token-level and pixel-level data across modalities. On a token level, TerraMind encodes high-level contextual information to learn cross-modal relationships, while on a pixel level, TerraMind leverages fine-grained representations to capture critical spatial nuances. We pretrained TerraMind on nine geospatial modalities of a global, large-scale dataset. In this paper, we demonstrate that (i) TerraMind's dual-scale early fusion approach unlocks a range of zero-shot and few-shot applications for Earth observation, (ii) TerraMind introduces "Thinking-in-Modalities" (TiM) -- the capability of generating additional artificial data during finetuning and inference to improve the model output -- and (iii) TerraMind achieves beyond state-of-the-art performance in community-standard benchmarks for EO like PANGAEA. The pretraining dataset, the model weights, and our code is open-sourced under a permissive license. 

**Abstract (ZH)**: We呈现TerraMind：一种用于地球观测的首个多尺度生成型多模态基础模型 

---
# MuSeD: A Multimodal Spanish Dataset for Sexism Detection in Social Media Videos 

**Title (ZH)**: MuSeD: 用于社交媒体视频中性别歧视检测的多模态西班牙语数据集 

**Authors**: Laura De Grazia, Pol Pastells, Mauro Vázquez Chas, Desmond Elliott, Danae Sánchez Villegas, Mireia Farrús, Mariona Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2504.11169)  

**Abstract**: Sexism is generally defined as prejudice and discrimination based on sex or gender, affecting every sector of society, from social institutions to relationships and individual behavior. Social media platforms amplify the impact of sexism by conveying discriminatory content not only through text but also across multiple modalities, highlighting the critical need for a multimodal approach to the analysis of sexism online. With the rise of social media platforms where users share short videos, sexism is increasingly spreading through video content. Automatically detecting sexism in videos is a challenging task, as it requires analyzing the combination of verbal, audio, and visual elements to identify sexist content. In this study, (1) we introduce MuSeD, a new Multimodal Spanish dataset for Sexism Detection consisting of $\approx$ 11 hours of videos extracted from TikTok and BitChute; (2) we propose an innovative annotation framework for analyzing the contribution of textual and multimodal labels in the classification of sexist and non-sexist content; and (3) we evaluate a range of large language models (LLMs) and multimodal LLMs on the task of sexism detection. We find that visual information plays a key role in labeling sexist content for both humans and models. Models effectively detect explicit sexism; however, they struggle with implicit cases, such as stereotypes, instances where annotators also show low agreement. This highlights the inherent difficulty of the task, as identifying implicit sexism depends on the social and cultural context. 

**Abstract (ZH)**: 性别歧视通常被定义为基于性别的偏见和歧视，影响社会的每一个领域，从社会机构到人际关系和个人行为。社交媒体平台通过不仅以文本形式还通过多种模态传播歧视性内容，强调了在线性别歧视分析的多模态方法的迫切需要。随着用户分享短视频的社交媒体平台兴起，性别歧视内容的传播越来越多地通过视频内容进行。自动检测视频中的性别歧视是一项具有挑战性的任务，因为它需要分析口头、音频和视觉元素的组合来识别性别歧视内容。在本研究中，(1) 我们介绍了MuSeD，一个包含源自TikTok和BitChute的约11小时视频的新多模态西班牙语性别歧视检测数据集；(2) 我们提出了一种创新的注释框架，用于分析文本标签和多模态标签在歧视性和非歧视性内容分类中的贡献；(3) 我们评估了一系列大规模语言模型（LLMs）和多模态LLMs在性别歧视检测任务中的性能。我们发现视觉信息在人类和模型标注性别歧视内容中起着关键作用。模型能够有效检测显性性别歧视；然而，它们在处理隐性案例，如刻板印象等方面存在困难，这些问题上注释者也表现出低一致性。这突显了该任务内在的难度，因为识别隐性性别歧视依赖于社会和文化背景。 

---
# DeepMLF: Multimodal language model with learnable tokens for deep fusion in sentiment analysis 

**Title (ZH)**: DeepMLF：具有可学习令牌的多模态语言模型在情感分析中的深度融合 

**Authors**: Efthymios Georgiou, Vassilis Katsouros, Yannis Avrithis, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2504.11082)  

**Abstract**: While multimodal fusion has been extensively studied in Multimodal Sentiment Analysis (MSA), the role of fusion depth and multimodal capacity allocation remains underexplored. In this work, we position fusion depth, scalability, and dedicated multimodal capacity as primary factors for effective fusion. We introduce DeepMLF, a novel multimodal language model (LM) with learnable tokens tailored toward deep fusion. DeepMLF leverages an audiovisual encoder and a pretrained decoder LM augmented with multimodal information across its layers. We append learnable tokens to the LM that: 1) capture modality interactions in a controlled fashion and 2) preserve independent information flow for each modality. These fusion tokens gather linguistic information via causal self-attention in LM Blocks and integrate with audiovisual information through cross-attention MM Blocks. Serving as dedicated multimodal capacity, this design enables progressive fusion across multiple layers, providing depth in the fusion process. Our training recipe combines modality-specific losses and language modelling loss, with the decoder LM tasked to predict ground truth polarity. Across three MSA benchmarks with varying dataset characteristics, DeepMLF achieves state-of-the-art performance. Our results confirm that deeper fusion leads to better performance, with optimal fusion depths (5-7) exceeding those of existing approaches. Additionally, our analysis on the number of fusion tokens reveals that small token sets ($\sim$20) achieve optimal performance. We examine the importance of representation learning order (fusion curriculum) through audiovisual encoder initialization experiments. Our ablation studies demonstrate the superiority of the proposed fusion design and gating while providing a holistic examination of DeepMLF's scalability to LLMs, and the impact of each training objective and embedding regularization. 

**Abstract (ZH)**: 多模态融合深度与容量分配在多模态情感分析中的研究：DeepMLF模型的设计与分析 

---
# CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors 

**Title (ZH)**: CDUPatch: 颜色驱动的通用 adversarial 贴片攻击用于双模可见-红外检测器 

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.10888)  

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios. 

**Abstract (ZH)**: 跨模态通用 adversarial 崩溃点攻击：面向可见光-红外目标检测系统 

---
# PuzzleBench: A Fully Dynamic Evaluation Framework for Large Multimodal Models on Puzzle Solving 

**Title (ZH)**: PuzzleBench: 一种用于益智谜题解决的全面动态评估框架（大型多模态模型） 

**Authors**: Zeyu Zhang, Zijian Chen, Zicheng Zhang, Yuze Sun, Yuan Tian, Ziheng Jia, Chunyi Li, Xiaohong Liu, Xiongkuo Min, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2504.10885)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated impressive capabilities across a wide range of multimodal tasks, achieving ever-increasing performance on various evaluation benchmarks. However, existing benchmarks are typically static and often overlap with pre-training datasets, leading to fixed complexity constraints and substantial data contamination issues. Meanwhile, manually annotated datasets are labor-intensive, time-consuming, and subject to human bias and inconsistency, leading to reliability and reproducibility issues. To address these problems, we propose a fully dynamic multimodal evaluation framework, named Open-ended Visual Puzzle Generation (OVPG), which aims to generate fresh, diverse, and verifiable evaluation data automatically in puzzle-solving tasks. Specifically, the OVPG pipeline consists of a raw material sampling module, a visual content generation module, and a puzzle rule design module, which ensures that each evaluation instance is primitive, highly randomized, and uniquely solvable, enabling continual adaptation to the evolving capabilities of LMMs. Built upon OVPG, we construct PuzzleBench, a dynamic and scalable benchmark comprising 11,840 VQA samples. It features six carefully designed puzzle tasks targeting three core LMM competencies, visual recognition, logical reasoning, and context understanding. PuzzleBench differs from static benchmarks that quickly become outdated. It enables ongoing dataset refreshing through OVPG and a rich set of open-ended puzzle designs, allowing seamless adaptation to the evolving capabilities of LMMs. 

**Abstract (ZH)**: 开放性视觉谜题生成的大规模多模态评估框架 

---
# ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness 

**Title (ZH)**: ColorBench：VLMs能否洞察彩色世界？一种全面的颜色感知、推理与鲁棒性基准测试 

**Authors**: Yijun Liang, Ming Li, Chenrui Fan, Ziyue Li, Dang Nguyen, Kwesi Cobbina, Shweta Bhardwaj, Jiuhai Chen, Fuxiao Liu, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.10514)  

**Abstract**: Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI. 

**Abstract (ZH)**: ColorBench: 一种用于评估视觉语言模型颜色理解能力的创新基准 

---
