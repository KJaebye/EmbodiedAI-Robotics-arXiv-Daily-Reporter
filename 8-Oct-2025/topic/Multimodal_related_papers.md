# Representation Potentials of Foundation Models for Multimodal Alignment: A Survey 

**Title (ZH)**: 基础模型在多模态对齐中的表示潜力：一个综述 

**Authors**: Jianglin Lu, Hailing Wang, Yi Xu, Yizhou Wang, Kuo Yang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05184)  

**Abstract**: Foundation models learn highly transferable representations through large-scale pretraining on diverse data. An increasing body of research indicates that these representations exhibit a remarkable degree of similarity across architectures and modalities. In this survey, we investigate the representation potentials of foundation models, defined as the latent capacity of their learned representations to capture task-specific information within a single modality while also providing a transferable basis for alignment and unification across modalities. We begin by reviewing representative foundation models and the key metrics that make alignment measurable. We then synthesize empirical evidence of representation potentials from studies in vision, language, speech, multimodality, and neuroscience. The evidence suggests that foundation models often exhibit structural regularities and semantic consistencies in their representation spaces, positioning them as strong candidates for cross-modal transfer and alignment. We further analyze the key factors that foster representation potentials, discuss open questions, and highlight potential challenges. 

**Abstract (ZH)**: 基础模型通过大规模多样数据的预训练学习到高度可迁移的表示。越来越多的研究表明，这些表示在不同架构和模态下表现出显著的相似性。在这篇综述中，我们探讨基础模型的表示潜力，即其学习表示在单一模态内捕获任务特定信息的能力，同时为跨模态的对齐和统一提供可迁移的基础。我们首先回顾代表性基础模型及其使对齐可度量的关键指标。然后，我们综合视觉、语言、语音、多模态和神经科学领域的实证证据，这些证据表明基础模型在表示空间中经常表现出结构规律性和语义一致性，使它们成为跨模态迁移和对齐的强大候选者。我们进一步分析促进表示潜力的关键因素，讨论开放问题，并指出潜在挑战。 

---
# Discrete Diffusion Models with MLLMs for Unified Medical Multimodal Generation 

**Title (ZH)**: 离散扩散模型结合MLLMs的统一医学多模态生成 

**Authors**: Jiawei Mao, Yuhan Wang, Lifeng Chen, Can Zhao, Yucheng Tang, Dong Yang, Liangqiong Qu, Daguang Xu, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.06131)  

**Abstract**: Recent advances in generative medical models are constrained by modality-specific scenarios that hinder the integration of complementary evidence from imaging, pathology, and clinical notes. This fragmentation limits their evolution into foundation models that can learn and reason across the full spectrum of biomedical data. We propose MeDiM, the first medical discrete diffusion model that learns shared distributions across modalities without modality-specific components. MeDiM unifies multiple generative tasks: translating between images and text, and jointly producing image-report pairs across domains in response to prompts. Built on a discrete diffusion framework, MeDiM bridges vision and language representations through a shared probabilistic space. To enable unified and flexible medical generation, we employ a multimodal large language model (MLLM) as the diffusion backbone, leveraging its prior knowledge and cross-modal reasoning. Two key designs are introduced: (1) removing the causal attention mask for bidirectional context, and (2) injecting continuous timestep embeddings for diffusion awareness. Experiments demonstrate high-fidelity medical generation (FID 16.60 on MIMIC-CXR and FID 24.19 on PathGen) and accurate report generation (METEOR 0.2650 and 0.2580). Jointly generated image-report pairs further enhance downstream performance (plus6.43 percent BLEU-1, plus18.57 percent BLEU-2, plus31.58 percent BLEU-3, plus4.80 percent METEOR), showing that MeDiM supports coherent and clinically grounded multimodal outputs. 

**Abstract (ZH)**: Recent Advances in Generative Medical Models Are Constrained by Modality-Specific Scenarios That Hinder the Integration of Complementary Evidence from Imaging, Pathology, and Clinical Notes 

---
# Controllable Audio-Visual Viewpoint Generation from 360° Spatial Information 

**Title (ZH)**: 从360°空间信息生成可控的音视频视角 

**Authors**: Christian Marinoni, Riccardo Fosco Gramaccioni, Eleonora Grassucci, Danilo Comminiello  

**Link**: [PDF](https://arxiv.org/pdf/2510.06060)  

**Abstract**: The generation of sounding videos has seen significant advancements with the advent of diffusion models. However, existing methods often lack the fine-grained control needed to generate viewpoint-specific content from larger, immersive 360-degree environments. This limitation restricts the creation of audio-visual experiences that are aware of off-camera events. To the best of our knowledge, this is the first work to introduce a framework for controllable audio-visual generation, addressing this unexplored gap. Specifically, we propose a diffusion model by introducing a set of powerful conditioning signals derived from the full 360-degree space: a panoramic saliency map to identify regions of interest, a bounding-box-aware signed distance map to define the target viewpoint, and a descriptive caption of the entire scene. By integrating these controls, our model generates spatially-aware viewpoint videos and audios that are coherently influenced by the broader, unseen environmental context, introducing a strong controllability that is essential for realistic and immersive audio-visual generation. We show audiovisual examples proving the effectiveness of our framework. 

**Abstract (ZH)**: 基于扩散模型的可控音视频生成：填补视角特定内容生成的空白 

---
# Detection and Measurement of Hailstones with Multimodal Large Language Models 

**Title (ZH)**: 使用多模态大型语言模型探测和测量 hailstones 

**Authors**: Moritz Alker, David C. Schedl, Andreas Stöckl  

**Link**: [PDF](https://arxiv.org/pdf/2510.06008)  

**Abstract**: This study examines the use of social media and news images to detect and measure hailstones, utilizing pre-trained multimodal large language models. The dataset for this study comprises 474 crowdsourced images of hailstones from documented hail events in Austria, which occurred between January 2022 and September 2024. These hailstones have maximum diameters ranging from 2 to 11cm. We estimate the hail diameters and compare four different models utilizing one-stage and two-stage prompting strategies. The latter utilizes additional size cues from reference objects, such as human hands, within the image. Our results show that pretrained models already have the potential to measure hailstone diameters from images with an average mean absolute error of 1.12cm for the best model. In comparison to a single-stage prompt, two-stage prompting improves the reliability of most models. Our study suggests that these off-the-shelf models, even without fine-tuning, can complement traditional hail sensors by extracting meaningful and spatially dense information from social media imagery, enabling faster and more detailed assessments of severe weather events. The automated real-time image harvesting from social media and other sources remains an open task, but it will make our approach directly applicable to future hail events. 

**Abstract (ZH)**: 本研究利用预训练多模态大语言模型检测和测量 hailstone，分析社交媒体和新闻图片中的 hailstone 图像。该研究的数据集包含来自 2022 年 1 月至 2024 年 9 月奥地利记录的 hailstone 事件的 474 张众包 hailstone 图片，直径范围从 2 cm 至 11 cm。我们估计 hailstone 直径并利用一阶段和两阶段提示策略分别进行了四种不同模型的评估。后者借助图像中参考物体（如人类手掌）的大小线索来提高模型性能。结果显示，预训练模型已具备从图像中测量 hailstone 直径的潜力，最佳模型的平均绝对误差为 1.12 cm。与一阶段提示相比，两阶段提示策略在大多数模型中提高了可靠性。本研究表明，这些即用型模型即使未经微调，也能通过社交媒体图像提取有意义且空间密集的信息，从而加快和细化对极端天气事件的评估。社交媒体等源的自动实时图像获取仍是待解决的问题，但将使我们的方法直接适用于未来的 hailstone 事件。 

---
# Seeing the Big Picture: Evaluating Multimodal LLMs' Ability to Interpret and Grade Handwritten Student Work 

**Title (ZH)**: 纵观全局：评估多模态大语言模型解释和评分手写学生作业的能力 

**Authors**: Owen Henkel, Bill Roberts, Doug Jaffe, Laurence Holt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05538)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) raise the question of their potential for grading, analyzing, and offering feedback on handwritten student classwork. This capability would be particularly beneficial in elementary and middle-school mathematics education, where most work remains handwritten, because seeing students' full working of a problem provides valuable insights into their learning processes, but is extremely time-consuming to grade. We present two experiments investigating MLLM performance on handwritten student mathematics classwork. Experiment A examines 288 handwritten responses from Ghanaian middle school students solving arithmetic problems with objective answers. In this context, models achieved near-human accuracy (95%, k = 0.90) but exhibited occasional errors that human educators would be unlikely to make. Experiment B evaluates 150 mathematical illustrations from American elementary students, where the drawings are the answer to the question. These tasks lack single objective answers and require sophisticated visual interpretation as well as pedagogical judgment in order to analyze and evaluate them. We attempted to separate MLLMs' visual capabilities from their pedagogical abilities by first asking them to grade the student illustrations directly, and then by augmenting the image with a detailed human description of the illustration. We found that when the models had to analyze the student illustrations directly, they struggled, achieving only k = 0.20 with ground truth scores, but when given human descriptions, their agreement levels improved dramatically to k = 0.47, which was in line with human-to-human agreement levels. This gap suggests MLLMs can "see" and interpret arithmetic work relatively well, but still struggle to "see" student mathematical illustrations. 

**Abstract (ZH)**: 最近在多模态大型语言模型方面的进展引发了对其在评估、分析和提供手写学生作业反馈方面的潜力的关注。这一能力特别有益于初高中数学教育，因为在这些阶段，大多数工作仍为手写，因为查看学生完整的问题解答过程可以提供有价值的学习过程洞察，但对其进行评分却极其耗时。我们进行了两项实验以研究多模态大型语言模型在处理手写学生数学作业方面的性能。实验A检查了288份加纳初中学生解决具有客观答案算术问题的手写回应。在这种情况下，模型达到了接近人类的准确率（95%，k = 0.90），但偶尔会出现人类教育者不会犯的错误。实验B评估了150份来自美国小学生的手绘数学图形，其中绘画是答案。这些任务缺乏单一客观答案，需要复杂的视觉解释以及教学判断，以分析和评价它们。我们尝试通过首先让模型直接评分学生的图形，然后通过增加详细的human描述来分离多模态大型语言模型的视觉能力和教学能力。我们发现，当模型需要直接分析学生的图形时，它们遇到了困难，准确率仅为k = 0.20，但当提供human描述时，其一致性水平显著提高到k = 0.47，与人类之间的共识水平相符。这一差距表明，多模态大型语言模型能够较好地“看到”并解释算术工作，但在“看到”学生数学图形方面仍然存在困难。 

---
# Exploring Student Choice and the Use of Multimodal Generative AI in Programming Learning 

**Title (ZH)**: 探索学生选择及其在编程学习中多模态生成AI的应用 

**Authors**: Xinying Hou, Ruiwei Xiao, Runlong Ye, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2510.05417)  

**Abstract**: The broad adoption of Generative AI (GenAI) is impacting Computer Science education, and recent studies found its benefits and potential concerns when students use it for programming learning. However, most existing explorations focus on GenAI tools that primarily support text-to-text interaction. With recent developments, GenAI applications have begun supporting multiple modes of communication, known as multimodality. In this work, we explored how undergraduate programming novices choose and work with multimodal GenAI tools, and their criteria for choices. We selected a commercially available multimodal GenAI platform for interaction, as it supports multiple input and output modalities, including text, audio, image upload, and real-time screen-sharing. Through 16 think-aloud sessions that combined participant observation with follow-up semi-structured interviews, we investigated student modality choices for GenAI tools when completing programming problems and the underlying criteria for modality selections. With multimodal communication emerging as the future of AI in education, this work aims to spark continued exploration on understanding student interaction with multimodal GenAI in the context of CS education. 

**Abstract (ZH)**: Generative AI工具有关的本科编程初学者的多模态选择与使用及其标准探索：对未来教育中AI交互的理解 

---
# AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement 

**Title (ZH)**: AUREXA-SE：结合跨注意力和Squeezeformer的音视频统一表示交换架构应用于语音增强 

**Authors**: M. Sajid, Deepanshu Gupta, Yash Modi, Sanskriti Jain, Harshith Jai Surya Ganji, A. Rahaman, Harshvardhan Choudhary, Nasir Saleem, Amir Hussain, M. Tanveer  

**Link**: [PDF](https://arxiv.org/pdf/2510.05295)  

**Abstract**: In this paper, we propose AUREXA-SE (Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement), a progressive bimodal framework tailored for audio-visual speech enhancement (AVSE). AUREXA-SE jointly leverages raw audio waveforms and visual cues by employing a U-Net-based 1D convolutional encoder for audio and a Swin Transformer V2 for efficient and expressive visual feature extraction. Central to the architecture is a novel bidirectional cross-attention mechanism, which facilitates deep contextual fusion between modalities, enabling rich and complementary representation learning. To capture temporal dependencies within the fused embeddings, a stack of lightweight Squeezeformer blocks combining convolutional and attention modules is introduced. The enhanced embeddings are then decoded via a U-Net-style decoder for direct waveform reconstruction, ensuring perceptually consistent and intelligible speech output. Experimental evaluations demonstrate the effectiveness of AUREXA-SE, achieving significant performance improvements over noisy baselines, with STOI of 0.516, PESQ of 1.323, and SI-SDR of -4.322 dB. The source code of AUREXA-SE is available at this https URL. 

**Abstract (ZH)**: 基于跨注意力和Squeezeformer的联合双向特征交换架构AUREXA-SE：面向视听speech增强的渐进式双模框架 

---
# Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices 

**Title (ZH)**: 小巧而强大：一种适用于电池供电小型设备高效多模态推理的软硬件协同设计方法 

**Authors**: Yilong Li, Shuai Zhang, Yijing Zeng, Hao Zhang, Xinmiao Xiong, Jingyu Liu, Pan Hu, Suman Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05109)  

**Abstract**: Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly half a day and LLaMA-3-8B for voice interactions up to almost 20.8 hours. 

**Abstract (ZH)**: 基于硬件-软件协同设计的大型多模态模型推理框架：NANOMIND 

---
