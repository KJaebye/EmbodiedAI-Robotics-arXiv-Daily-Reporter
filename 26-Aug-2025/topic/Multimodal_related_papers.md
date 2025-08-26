# Scene-Agnostic Traversability Labeling and Estimation via a Multimodal Self-supervised Framework 

**Title (ZH)**: 基于多模态自监督框架的场景无关通行性标签与估计 

**Authors**: Zipeng Fang, Yanbo Wang, Lei Zhao, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18249)  

**Abstract**: Traversability estimation is critical for enabling robots to navigate across diverse terrains and environments. While recent self-supervised learning methods achieve promising results, they often fail to capture the characteristics of non-traversable regions. Moreover, most prior works concentrate on a single modality, overlooking the complementary strengths offered by integrating heterogeneous sensory modalities for more robust traversability estimation. To address these limitations, we propose a multimodal self-supervised framework for traversability labeling and estimation. First, our annotation pipeline integrates footprint, LiDAR, and camera data as prompts for a vision foundation model, generating traversability labels that account for both semantic and geometric cues. Then, leveraging these labels, we train a dual-stream network that jointly learns from different modalities in a decoupled manner, enhancing its capacity to recognize diverse traversability patterns. In addition, we incorporate sparse LiDAR-based supervision to mitigate the noise introduced by pseudo labels. Finally, extensive experiments conducted across urban, off-road, and campus environments demonstrate the effectiveness of our approach. The proposed automatic labeling method consistently achieves around 88% IoU across diverse datasets. Compared to existing self-supervised state-of-the-art methods, our multimodal traversability estimation network yields consistently higher IoU, improving by 1.6-3.5% on all evaluated datasets. 

**Abstract (ZH)**: 多模态自监督框架在非通行区域标注与估计中的应用 

---
# M3DMap: Object-aware Multimodal 3D Mapping for Dynamic Environments 

**Title (ZH)**: M3DMap: 具有物体意识的多模态动态环境3D地图构建 

**Authors**: Dmitry Yudin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17044)  

**Abstract**: 3D mapping in dynamic environments poses a challenge for modern researchers in robotics and autonomous transportation. There are no universal representations for dynamic 3D scenes that incorporate multimodal data such as images, point clouds, and text. This article takes a step toward solving this problem. It proposes a taxonomy of methods for constructing multimodal 3D maps, classifying contemporary approaches based on scene types and representations, learning methods, and practical applications. Using this taxonomy, a brief structured analysis of recent methods is provided. The article also describes an original modular method called M3DMap, designed for object-aware construction of multimodal 3D maps for both static and dynamic scenes. It consists of several interconnected components: a neural multimodal object segmentation and tracking module; an odometry estimation module, including trainable algorithms; a module for 3D map construction and updating with various implementations depending on the desired scene representation; and a multimodal data retrieval module. The article highlights original implementations of these modules and their advantages in solving various practical tasks, from 3D object grounding to mobile manipulation. Additionally, it presents theoretical propositions demonstrating the positive effect of using multimodal data and modern foundational models in 3D mapping methods. Details of the taxonomy and method implementation are available at this https URL. 

**Abstract (ZH)**: 动态环境下的3D建图是现代机器人与自主运输领域研究人员面临的挑战。缺乏能够整合图像、点云和文本等多种模式数据的通用动态3D场景表示方法。本文朝着解决这一问题迈出了一步，提出了一种构造多模式3D地图的方法 taxonomy，基于场景类型、表示方法、学习方法和实际应用对当前方法进行分类。利用此 taxonomy，简要分析了近期的方法。文中还描述了一个原创的模块化方法 M3DMap，旨在构建适合静态和动态场景的物体感知多模式3D地图。该方法由几个相互关联的组件组成：一个神经多模式物体分割和跟踪模块；一个里程计估计模块，包括可训练算法；一个3D地图构建和更新模块，根据所需的场景表示有不同的实现方式；一个多模式数据检索模块。本文突出了这些模块的原创实现及其在从3D物体定位到移动操作等各种实际任务中的优势。此外，还提出了理论命题，展示了使用多模式数据和现代基础模型在3D建图方法中发挥的积极效果。更多详细内容和方法实现细节请参见此 <https://> 地址。 

---
# SEAM: Semantically Equivalent Across Modalities Benchmark for Vision-Language Models 

**Title (ZH)**: SEAM: 不同模态下的语义等价基准测试 for 视觉-语言模型 

**Authors**: Zhenwei Tang, Difan Jiao, Blair Yang, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2508.18179)  

**Abstract**: Evaluating whether vision-language models (VLMs) reason consistently across representations is challenging because modality comparisons are typically confounded by task differences and asymmetric information. We introduce SEAM, a benchmark that pairs semantically equivalent inputs across four domains that have existing standardized textual and visual notations. By employing distinct notation systems across modalities, in contrast to OCR-based image-text pairing, SEAM provides a rigorous comparative assessment of the textual-symbolic and visual-spatial reasoning capabilities of VLMs. Across 21 contemporary models, we observe systematic modality imbalance: vision frequently lags language in overall performance, despite the problems containing semantically equivalent information, and cross-modal agreement is relatively low. Our error analysis reveals two main drivers: textual perception failures from tokenization in domain notation and visual perception failures that induce hallucinations. We also show that our results are largely robust to visual transformations. SEAM establishes a controlled, semantically equivalent setting for measuring and improving modality-agnostic reasoning. 

**Abstract (ZH)**: 评估 vision-language 模型在不同表示层面上一致推理的挑战性在于，模态比较通常会被任务差异和信息不对称所混淆。我们介绍了 SEAM，这是一个基准测试，它在四个现有标准化文本和视觉符号表示的领域中配对语义等效的输入。通过在模态之间采用不同的符号系统，不同于基于 OCR 的图像-文本配对，SEAM 提供了对 VLM 文本-符号和视觉-空间推理能力的严格比较评估。在 21 个当代模型中，我们观察到系统性的模态不平衡：尽管问题包含语义等效信息，视觉经常在整体表现上落后于语言，且跨模态一致率相对较低。我们的错误分析揭示了两大驱动因素：来自领域符号表示中词元化失败的文本感知错误和引发幻觉的视觉感知错误。我们还展示了我们的结果对视觉变换具有较大的稳健性。SEAM 为测量和提升跨模态一致推理能力建立了可控的语义等效环境。 

---
# Mimicking the Physicist's Eye:A VLM-centric Approach for Physics Formula Discovery 

**Title (ZH)**: 模拟物理学家的视角：基于VLM的方法在物理公式发现中的应用 

**Authors**: Jiaqi Liu, Songning Lai, Pengze Li, Di Yu, Wenjie Zhou, Yiyang Zhou, Peng Xia, Zijun Wang, Xi Chen, Shixiang Tang, Lei Bai, Wanli Ouyang, Mingyu Ding, Huaxiu Yao, Aoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17380)  

**Abstract**: Automated discovery of physical laws from observational data in the real world is a grand challenge in AI. Current methods, relying on symbolic regression or LLMs, are limited to uni-modal data and overlook the rich, visual phenomenological representations of motion that are indispensable to physicists. This "sensory deprivation" severely weakens their ability to interpret the inherent spatio-temporal patterns within dynamic phenomena. To address this gap, we propose VIPER-R1, a multimodal model that performs Visual Induction for Physics-based Equation Reasoning to discover fundamental symbolic formulas. It integrates visual perception, trajectory data, and symbolic reasoning to emulate the scientific discovery process. The model is trained via a curriculum of Motion Structure Induction (MSI), using supervised fine-tuning to interpret kinematic phase portraits and to construct hypotheses guided by a Causal Chain of Thought (C-CoT), followed by Reward-Guided Symbolic Calibration (RGSC) to refine the formula structure with reinforcement learning. During inference, the trained VIPER-R1 acts as an agent: it first posits a high-confidence symbolic ansatz, then proactively invokes an external symbolic regression tool to perform Symbolic Residual Realignment (SR^2). This final step, analogous to a physicist's perturbation analysis, reconciles the theoretical model with empirical data. To support this research, we introduce PhysSymbol, a new 5,000-instance multimodal corpus. Experiments show that VIPER-R1 consistently outperforms state-of-the-art VLM baselines in accuracy and interpretability, enabling more precise discovery of physical laws. Project page: this https URL 

**Abstract (ZH)**: 从观测数据中自动发现物理定律是人工智能领域的重大挑战。当前的方法依赖于符号回归或大语言模型，局限于单模态数据，并忽视了运动的丰富、视觉表现形式，这种表现形式对于物理学家来说是不可或缺的。这种“感觉剥夺”严重削弱了它们解释动态现象内在时空模式的能力。为了解决这一差距，我们提出了一种多模态模型VIPER-R1，该模型通过视觉诱导进行基于物理方程的逻辑推理，以发现基本的符号公式。该模型将视觉感知、轨迹数据和符号推理结合起来，模拟了科学发现的过程。模型通过运动结构诱导（MSI）的课程进行训练，采用监督微调来解释运动相位图，并由因果链思维（C-CoT）引导建立假设，随后通过基于奖励的符号校准（RGSC）利用强化学习进一步细化公式结构。在推理过程中，训练后的VIPER-R1作为一个代理：首先提出一个高置信度的符号假设，然后主动调用外部符号回归工具进行符号残差校准（SR^2）。这一最终步骤类似于物理学家的摄动分析，将理论模型与实证数据统一起来。为了支持这项研究，我们引入了PhysSymbol，这是一个新的包含5,000个实例的多模态语料库。实验结果表明，VIPER-R1在准确性和可解释性方面均优于现有的最先进的视觉语言模型基准，使物理定律的精确发现成为可能。项目页面：this https URL。 

---
# ERF-BA-TFD+: A Multimodal Model for Audio-Visual Deepfake Detection 

**Title (ZH)**: ERF-BA-TFD+: 多模态音频-视觉深伪检测模型 

**Authors**: Xin Zhang, Jiaming Chu, Jian Zhao, Yuchu Jiang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17282)  

**Abstract**: Deepfake detection is a critical task in identifying manipulated multimedia content. In real-world scenarios, deepfake content can manifest across multiple modalities, including audio and video. To address this challenge, we present ERF-BA-TFD+, a novel multimodal deepfake detection model that combines enhanced receptive field (ERF) and audio-visual fusion. Our model processes both audio and video features simultaneously, leveraging their complementary information to improve detection accuracy and robustness. The key innovation of ERF-BA-TFD+ lies in its ability to model long-range dependencies within the audio-visual input, allowing it to better capture subtle discrepancies between real and fake content. In our experiments, we evaluate ERF-BA-TFD+ on the DDL-AV dataset, which consists of both segmented and full-length video clips. Unlike previous benchmarks, which focused primarily on isolated segments, the DDL-AV dataset allows us to assess the model's performance in a more comprehensive and realistic setting. Our method achieves state-of-the-art results on this dataset, outperforming existing techniques in terms of both accuracy and processing speed. The ERF-BA-TFD+ model demonstrated its effectiveness in the "Workshop on Deepfake Detection, Localization, and Interpretability," Track 2: Audio-Visual Detection and Localization (DDL-AV), and won first place in this competition. 

**Abstract (ZH)**: 基于增强感受野和音视频融合的多模态深仿生成分检测模型（ERF-BA-TFD+） 

---
# Dynamic Fusion Multimodal Network for SpeechWellness Detection 

**Title (ZH)**: 动态融合多模态网络用于语音健康检测 

**Authors**: Wenqiang Sun, Han Yin, Jisheng Bai, Jianfeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18057)  

**Abstract**: Suicide is one of the leading causes of death among adolescents. Previous suicide risk prediction studies have primarily focused on either textual or acoustic information in isolation, the integration of multimodal signals, such as speech and text, offers a more comprehensive understanding of an individual's mental state. Motivated by this, and in the context of the 1st SpeechWellness detection challenge, we explore a lightweight multi-branch multimodal system based on a dynamic fusion mechanism for speechwellness detection. To address the limitation of prior approaches that rely on time-domain waveforms for acoustic analysis, our system incorporates both time-domain and time-frequency (TF) domain acoustic features, as well as semantic representations. In addition, we introduce a dynamic fusion block to adaptively integrate information from different modalities. Specifically, it applies learnable weights to each modality during the fusion process, enabling the model to adjust the contribution of each modality. To enhance computational efficiency, we design a lightweight structure by simplifying the original baseline model. Experimental results demonstrate that the proposed system exhibits superior performance compared to the challenge baseline, achieving a 78% reduction in model parameters and a 5% improvement in accuracy. 

**Abstract (ZH)**: 青少年自杀是导致死亡的主要原因之一。以往的自杀风险预测研究主要集中在文本或声学信息单一模态的数据上，将语音和文本等多模态信号的融合提供了一种更全面理解个体心理状态的方法。受到这一启发，并在第1届SpeechWellness检测挑战赛的背景下，我们探索了一种基于动态融合机制的轻量级多分支多模态系统，用于SpeechWellness检测。为了克服之前依赖时域波形进行声学分析的局限性，我们的系统结合了时域和时频域声学特征以及语义表示。此外，我们引入了一个动态融合块，以适应性地整合不同模态的信息。具体而言，该块在融合过程中为每个模态应用可学习的权重，使模型能够调整每个模态的贡献。为了提升计算效率，我们通过简化原始基线模型设计了一个轻量级结构。实验结果表明，所提出系统的表现优于挑战基线，模型参数减少了78%，准确率提高了5%。 

---
# AVAM: Universal Training-free Adaptive Visual Anchoring Embedded into Multimodal Large Language Model for Multi-image Question Answering 

**Title (ZH)**: AVAM：嵌入多模态大规模语言模型的通用无训练自适应视觉锚定 

**Authors**: Kang Zeng, Guojin Zhong, Jintao Cheng, Jin Yuan, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17860)  

**Abstract**: The advancement of Multimodal Large Language Models (MLLMs) has driven significant progress in Visual Question Answering (VQA), evolving from Single to Multi Image VQA (MVQA). However, the increased number of images in MVQA inevitably introduces substantial visual redundancy that is irrelevant to question answering, negatively impacting both accuracy and efficiency. To address this issue, existing methods lack flexibility in controlling the number of compressed visual tokens and tend to produce discrete visual fragments, which hinder MLLMs' ability to comprehend images holistically. In this paper, we propose a straightforward yet universal Adaptive Visual Anchoring strategy, which can be seamlessly integrated into existing MLLMs, offering significant accuracy improvements through adaptive compression. Meanwhile, to balance the results derived from both global and compressed visual input, we further introduce a novel collaborative decoding mechanism, enabling optimal performance. Extensive experiments validate the effectiveness of our method, demonstrating consistent performance improvements across various MLLMs. The code will be publicly available. 

**Abstract (ZH)**: Multimodal Large Language Models的进展推动了视觉问答（VQA）的显著进步，从单图VQA（SVQA）发展到多图VQA（MVQA）。然而，MVQA中图片数量的增加不可避免地带来了与问题回答无关的大量视觉冗余，这不仅影响准确性，还降低了效率。为解决这一问题，现有方法在控制压缩视觉令牌数量方面缺乏灵活性，倾向于生成离散的视觉片段，从而阻碍了MLLMs对图像的整体理解能力。本文提出了一种简单而通用的自适应视觉锚定策略，可以无缝集成到现有的MLLMs中，通过自适应压缩提高准确性。同时，为了平衡来自全局和压缩视觉输入的结果，我们进一步引入了一种新的协作解码机制，以实现最优性能。大量实验证明了我们方法的有效性，展示了在各种MLLMs上的一致性能提升。代码将公开可用。 

---
# Instant Preference Alignment for Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型的即时偏好对齐 

**Authors**: Yang Li, Songlin Yang, Xiaoxuan Han, Wei Wang, Jing Dong, Yueming Lyu, Ziyu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.17718)  

**Abstract**: Text-to-image (T2I) generation has greatly enhanced creative expression, yet achieving preference-aligned generation in a real-time and training-free manner remains challenging. Previous methods often rely on static, pre-collected preferences or fine-tuning, limiting adaptability to evolving and nuanced user intents. In this paper, we highlight the need for instant preference-aligned T2I generation and propose a training-free framework grounded in multimodal large language model (MLLM) priors. Our framework decouples the task into two components: preference understanding and preference-guided generation. For preference understanding, we leverage MLLMs to automatically extract global preference signals from a reference image and enrich a given prompt using structured instruction design. Our approach supports broader and more fine-grained coverage of user preferences than existing methods. For preference-guided generation, we integrate global keyword-based control and local region-aware cross-attention modulation to steer the diffusion model without additional training, enabling precise alignment across both global attributes and local elements. The entire framework supports multi-round interactive refinement, facilitating real-time and context-aware image generation. Extensive experiments on the Viper dataset and our collected benchmark demonstrate that our method outperforms prior approaches in both quantitative metrics and human evaluations, and opens up new possibilities for dialog-based generation and MLLM-diffusion integration. 

**Abstract (ZH)**: 基于多模态大型语言模型的无需训练的文本到图像生成 

---
# Multimodal Representation Learning Conditioned on Semantic Relations 

**Title (ZH)**: 基于语义关系的多模态表示学习 

**Authors**: Yang Qiao, Yuntong Hu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.17497)  

**Abstract**: Multimodal representation learning has advanced rapidly with contrastive models such as CLIP, which align image-text pairs in a shared embedding space. However, these models face limitations: (1) they typically focus on image-text pairs, underutilizing the semantic relations across different pairs. (2) they directly match global embeddings without contextualization, overlooking the need for semantic alignment along specific subspaces or relational dimensions; and (3) they emphasize cross-modal contrast, with limited support for intra-modal consistency. To address these issues, we propose Relation-Conditioned Multimodal Learning RCML, a framework that learns multimodal representations under natural-language relation descriptions to guide both feature extraction and alignment. Our approach constructs many-to-many training pairs linked by semantic relations and introduces a relation-guided cross-attention mechanism that modulates multimodal representations under each relation context. The training objective combines inter-modal and intra-modal contrastive losses, encouraging consistency across both modalities and semantically related samples. Experiments on different datasets show that RCML consistently outperforms strong baselines on both retrieval and classification tasks, highlighting the effectiveness of leveraging semantic relations to guide multimodal representation learning. 

**Abstract (ZH)**: 基于语义关系条件的多模态学习RCML 

---
# Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework 

**Title (ZH)**: 基于模态特定语音增强和噪声自适应融合的声传导麦克风框架 

**Authors**: Yunsik Kim, Yoonyoung Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.17336)  

**Abstract**: Body\-conduction microphone signals (BMS) bypass airborne sound, providing strong noise resistance. However, a complementary modality is required to compensate for the inherent loss of high\-frequency information. In this study, we propose a novel multi\-modal framework that combines BMS and acoustic microphone signals (AMS) to achieve both noise suppression and high\-frequency reconstruction. Unlike conventional multi\-modal approaches that simply merge features, our method employs two specialized networks\: a mapping-based model to enhance BMS and a masking-based model to denoise AMS. These networks are integrated through a dynamic fusion mechanism that adapts to local noise conditions, ensuring the optimal use of each modality's strengths. We performed evaluations on the TAPS dataset, augmented with DNS\-2023 noise clips, using objective speech quality metrics. The results clearly demonstrate that our approach outperforms single\-modal solutions in a wide range of noisy environments. 

**Abstract (ZH)**: 基于传导的麦克风信号和声学麦克风信号的新型多模态框架：噪声抑制与高频重建 

---
# Explain Before You Answer: A Survey on Compositional Visual Reasoning 

**Title (ZH)**: 解释再作答：关于组合视觉推理的综述 

**Authors**: Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi, Gholamreza Haffari, Ranjay Krishna, Jiajun Wu, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17298)  

**Abstract**: Compositional visual reasoning has emerged as a key research frontier in multimodal AI, aiming to endow machines with the human-like ability to decompose visual scenes, ground intermediate concepts, and perform multi-step logical inference. While early surveys focus on monolithic vision-language models or general multimodal reasoning, a dedicated synthesis of the rapidly expanding compositional visual reasoning literature is still missing. We fill this gap with a comprehensive survey spanning 2023 to 2025 that systematically reviews 260+ papers from top venues (CVPR, ICCV, NeurIPS, ICML, ACL, etc.). We first formalize core definitions and describe why compositional approaches offer advantages in cognitive alignment, semantic fidelity, robustness, interpretability, and data efficiency. Next, we trace a five-stage paradigm shift: from prompt-enhanced language-centric pipelines, through tool-enhanced LLMs and tool-enhanced VLMs, to recently minted chain-of-thought reasoning and unified agentic VLMs, highlighting their architectural designs, strengths, and limitations. We then catalog 60+ benchmarks and corresponding metrics that probe compositional visual reasoning along dimensions such as grounding accuracy, chain-of-thought faithfulness, and high-resolution perception. Drawing on these analyses, we distill key insights, identify open challenges (e.g., limitations of LLM-based reasoning, hallucination, a bias toward deductive reasoning, scalable supervision, tool integration, and benchmark limitations), and outline future directions, including world-model integration, human-AI collaborative reasoning, and richer evaluation protocols. By offering a unified taxonomy, historical roadmap, and critical outlook, this survey aims to serve as a foundational reference and inspire the next generation of compositional visual reasoning research. 

**Abstract (ZH)**: 组成式视觉推理已成为多模态AI的关键研究前沿，旨在赋予机器类似人类的能力，分解视觉场景、 grounding 中间概念，并进行多步逻辑推理。虽然早期综述主要关注于整体型视觉语言模型或一般多模态推理，但专门梳理这一迅速扩展的组成式视觉推理文献仍然缺失。我们通过一个涵盖2023至2025年的全面综述填补这一空白，系统回顾了来自顶级会议（CVPR、ICCV、NeurIPS、ICML、ACL等）的260多篇论文。我们首先正式化核心定义，并描述组成式方法在认知对齐、语义保真度、鲁棒性、可解释性以及数据效率方面的优势。接着，我们跟踪了五个阶段的范式转变：从增强型提示语言中心流程，经过工具增强的大型语言模型和视觉语言模型，到最近提出的链式推理和统一仿生视觉语言模型，突出它们的架构设计、优势和局限性。我们随后整理了60多种基准和相应的度量标准，这些标准从不同的维度（如grounding准确性、链式推理忠实性和高分辨率感知）探索组成式视觉推理。基于这些分析，我们提炼关键见解，识别开放挑战（如基于大型语言模型的推理限制、幻觉、倾向于演绎推理、可扩展监督、工具集成以及基准限制），并概述未来方向，包括世界模型集成、人类-人工智能协作推理和更丰富的评估协议。通过提供统一的分类体系、历史路线图和批判性展望，本综述旨在成为基础性参考文献，并激发下一代组成式视觉推理研究。 

---
# How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System 

**Title (ZH)**: 如何使医疗AI系统更安全？模拟多模态医疗RAG系统的漏洞和威胁 

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2508.17215)  

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems. 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs) 增强检索增强生成 (RAG) 的 MedThreatRAG：系统性探究医疗 RAG 系统中的漏洞 

---
# Multi-Agent Visual-Language Reasoning for Comprehensive Highway Scene Understanding 

**Title (ZH)**: 多Agent视觉-语言推理以实现全面高速公路场景理解 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17205)  

**Abstract**: This paper introduces a multi-agent framework for comprehensive highway scene understanding, designed around a mixture-of-experts strategy. In this framework, a large generic vision-language model (VLM), such as GPT-4o, is contextualized with domain knowledge to generates task-specific chain-of-thought (CoT) prompts. These fine-grained prompts are then used to guide a smaller, efficient VLM (e.g., Qwen2.5-VL-7B) in reasoning over short videos, along with complementary modalities as applicable. The framework simultaneously addresses multiple critical perception tasks, including weather classification, pavement wetness assessment, and traffic congestion detection, achieving robust multi-task reasoning while balancing accuracy and computational efficiency. To support empirical validation, we curated three specialized datasets aligned with these tasks. Notably, the pavement wetness dataset is multimodal, combining video streams with road weather sensor data, highlighting the benefits of multimodal reasoning. Experimental results demonstrate consistently strong performance across diverse traffic and environmental conditions. From a deployment perspective, the framework can be readily integrated with existing traffic camera systems and strategically applied to high-risk rural locations, such as sharp curves, flood-prone lowlands, or icy bridges. By continuously monitoring the targeted sites, the system enhances situational awareness and delivers timely alerts, even in resource-constrained environments. 

**Abstract (ZH)**: 一种基于专家混合策略的全面高速公路场景理解多agent框架 

---
# PlantVillageVQA: A Visual Question Answering Dataset for Benchmarking Vision-Language Models in Plant Science 

**Title (ZH)**: PlantVillageVQA：植物科学领域视觉问答数据集，用于评估视觉语言模型的基准性能 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Mohammad Zabed Hossain, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2508.17117)  

**Abstract**: PlantVillageVQA is a large-scale visual question answering (VQA) dataset derived from the widely used PlantVillage image corpus. It was designed to advance the development and evaluation of vision-language models for agricultural decision-making and analysis. The PlantVillageVQA dataset comprises 193,609 high-quality question-answer (QA) pairs grounded over 55,448 images spanning 14 crop species and 38 disease conditions. Questions are organised into 3 levels of cognitive complexity and 9 distinct categories. Each question category was phrased manually following expert guidance and generated via an automated two-stage pipeline: (1) template-based QA synthesis from image metadata and (2) multi-stage linguistic re-engineering. The dataset was iteratively reviewed by domain experts for scientific accuracy and relevancy. The final dataset was evaluated using three state-of-the-art models for quality assessment. Our objective remains to provide a publicly available, standardised and expert-verified database to enhance diagnostic accuracy for plant disease identifications and advance scientific research in the agricultural domain. Our dataset will be open-sourced at this https URL. 

**Abstract (ZH)**: PlantVillageVQA是源自广泛使用的PlantVillage图像库的大规模视觉问答（VQA）数据集，旨在推动农业决策和分析中视觉-语言模型的发展与评估。PlantVillageVQA数据集包含193,609个高质量的问题-答案（QA）对，涵盖55,448张图像中的14种作物和38种病害条件。问题按照3个认知复杂度级别和9个不同的类别组织。每个问题类别都是根据专家指导手工表述并通过自动化两阶段管道生成：（1）基于图像元数据的模板QA合成；（2）多阶段语言重构。该数据集经过领域专家迭代审查以确保科学准确性和相关性。最终数据集使用三个最先进的模型进行质量评估。我们的目标是提供一个公开可用、标准化且经专家验证的数据库，以提高植物病害诊断的准确性并推动农业领域的科学研究。我们的数据集将在以下链接开源：此httpsURL。 

---
# Multimodal Appearance based Gaze-Controlled Virtual Keyboard with Synchronous Asynchronous Interaction for Low-Resource Settings 

**Title (ZH)**: 基于多模态外观的眼动控制虚拟键盘及其在低资源环境下的同步异步交互方法 

**Authors**: Yogesh Kumar Meena, Manish Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16606)  

**Abstract**: Over the past decade, the demand for communication devices has increased among individuals with mobility and speech impairments. Eye-gaze tracking has emerged as a promising solution for hands-free communication; however, traditional appearance-based interfaces often face challenges such as accuracy issues, involuntary eye movements, and difficulties with extensive command sets. This work presents a multimodal appearance-based gaze-controlled virtual keyboard that utilises deep learning in conjunction with standard camera hardware, incorporating both synchronous and asynchronous modes for command selection. The virtual keyboard application supports menu-based selection with nine commands, enabling users to spell and type up to 56 English characters, including uppercase and lowercase letters, punctuation, and a delete function for corrections. The proposed system was evaluated with twenty able-bodied participants who completed specially designed typing tasks using three input modalities: (i) a mouse, (ii) an eye-tracker, and (iii) an unmodified webcam. Typing performance was measured in terms of speed and information transfer rate (ITR) at both command and letter levels. Average typing speeds were 18.3+-5.31 letters/min (mouse), 12.60+-2.99letters/min (eye-tracker, synchronous), 10.94 +- 1.89 letters/min (webcam, synchronous), 11.15 +- 2.90 letters/min (eye-tracker, asynchronous), and 7.86 +- 1.69 letters/min (webcam, asynchronous). ITRs were approximately 80.29 +- 15.72 bits/min (command level) and 63.56 +- 11 bits/min (letter level) with webcam in synchronous mode. The system demonstrated good usability and low workload with webcam input, highlighting its user-centred design and promise as an accessible communication tool in low-resource settings. 

**Abstract (ZH)**: 过去十年间，移动和言语障碍个体对通信设备的需求不断增加。目光追踪技术已成为无需手部操作的通信 promising 解决方案；然而，传统的基于外观的界面往往面临准确性问题、不自主的眼球运动以及广泛的命令集难以处理的挑战。本文提出了一种结合深度学习和标准摄像头硬件的多模态基于外观的目光控制虚拟键盘，该系统包括同步和异步模式以进行命令选择。虚拟键盘应用程序通过菜单选择支持多达九个命令，使用户能够拼写和输入56个英文字符，包括大小写字母、标点符号以及删除功能以进行更正。所提出的系统在二十名健全参与者中进行了评估，他们使用三种输入模式完成了专门设计的打字任务：（i）鼠标，（ii）眼动追踪器，（iii）未修改的网络摄像头。从命令和字母层面测量了打字性能，包括速度和信息传输率（ITR）。平均打字速度分别为鼠标18.3±5.31个字母/分钟，眼动追踪器同步模式12.60±2.99个字母/分钟，网络摄像头同步模式10.94±1.89个字母/分钟，眼动追踪器异步模式11.15±2.90个字母/分钟，以及网络摄像头异步模式7.86±1.69个字母/分钟。同步模式下网络摄像头的ITR分别为命令级别约80.29±15.72比特/分钟、字母级别约63.56±11比特/分钟。该系统在使用网络摄像头输入时展示了良好的易用性和较低的工作负荷，突显了其以用户为中心的设计，并使其成为低资源环境下易于访问的通信工具的潜力。 

---
