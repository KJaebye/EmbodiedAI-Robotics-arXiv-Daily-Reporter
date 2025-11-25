# EEG-VLM: A Hierarchical Vision-Language Model with Multi-Level Feature Alignment and Visually Enhanced Language-Guided Reasoning for EEG Image-Based Sleep Stage Prediction 

**Title (ZH)**: EEG-VLM：基于多级特征对齐和视觉增强语言引导推理的EEG图像睡眠阶段预测的层次视觉语言模型 

**Authors**: Xihe Qiu, Gengchen Ma, Haoyu Wang, Chen Zhan, Xiaoyu Tan, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.19155)  

**Abstract**: Sleep stage classification based on electroencephalography (EEG) is fundamental for assessing sleep quality and diagnosing sleep-related disorders. However, most traditional machine learning methods rely heavily on prior knowledge and handcrafted features, while existing deep learning models still struggle to jointly capture fine-grained time-frequency patterns and achieve clinical interpretability. Recently, vision-language models (VLMs) have made significant progress in the medical domain, yet their performance remains constrained when applied to physiological waveform data, especially EEG signals, due to their limited visual understanding and insufficient reasoning capability. To address these challenges, we propose EEG-VLM, a hierarchical vision-language framework that integrates multi-level feature alignment with visually enhanced language-guided reasoning for interpretable EEG-based sleep stage classification. Specifically, a specialized visual enhancement module constructs high-level visual tokens from intermediate-layer features to extract rich semantic representations of EEG images. These tokens are further aligned with low-level CLIP features through a multi-level alignment mechanism, enhancing the VLM's image-processing capability. In addition, a Chain-of-Thought (CoT) reasoning strategy decomposes complex medical inference into interpretable logical steps, effectively simulating expert-like decision-making. Experimental results demonstrate that the proposed method significantly improves both the accuracy and interpretability of VLMs in EEG-based sleep stage classification, showing promising potential for automated and explainable EEG analysis in clinical settings. 

**Abstract (ZH)**: 基于电生理波形数据的视觉语言模型在睡眠阶段分类中的应用：一种具备解释性的多层次视觉语言框架 

---
# Synthesizing Visual Concepts as Vision-Language Programs 

**Title (ZH)**: 视觉概念合成作为视觉-语言程序 

**Authors**: Antonia Wüst, Wolfgang Stammer, Hikaru Shindo, Lukas Helff, Devendra Singh Dhami, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2511.18964)  

**Abstract**: Vision-Language models (VLMs) achieve strong performance on multimodal tasks but often fail at systematic visual reasoning tasks, leading to inconsistent or illogical outputs. Neuro-symbolic methods promise to address this by inducing interpretable logical rules, though they exploit rigid, domain-specific perception modules. We propose Vision-Language Programs (VLP), which combine the perceptual flexibility of VLMs with systematic reasoning of program synthesis. Rather than embedding reasoning inside the VLM, VLP leverages the model to produce structured visual descriptions that are compiled into neuro-symbolic programs. The resulting programs execute directly on images, remain consistent with task constraints, and provide human-interpretable explanations that enable easy shortcut mitigation. Experiments on synthetic and real-world datasets demonstrate that VLPs outperform direct and structured prompting, particularly on tasks requiring complex logical reasoning. 

**Abstract (ZH)**: Vision-Language模型（VLMs）在多模态任务中表现出色，但在系统视觉推理任务中往往表现不佳，导致输出不一致或不合逻辑。神经符号方法有望通过诱导可解释的逻辑规则来解决这一问题，尽管它们依赖于刚性且领域特定的感知模块。我们提出了Vision-Language程序（VLP），它结合了VLMs的感知灵活性与程序合成的系统推理能力。VLP 不是在模型内部嵌入推理，而是利用模型生成结构化的视觉描述，并将这些描述编译成神经-符号程序。生成的程序可以直接执行于图像上，保持与任务约束的一致性，并提供可由人类解释的说明，从而易于消除捷径。实验表明，VLPs在需要复杂逻辑推理的任务中优于直接和结构化提示。 

---
# GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction 

**Title (ZH)**: GContextFormer：一种基于全局上下文的混合多头注意力模型及其在 multimodal 轨迹预测中的应用，带有缩放加性聚合 

**Authors**: Yuzhi Chen, Yuanchang Xie, Lei Zhao, Pan Liu, Yajie Zou, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18874)  

**Abstract**: Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: this https URL. 

**Abstract (ZH)**: 多模态轨迹预测通过生成多个可能的未来轨迹来应对来自意图模糊和执行变化的车辆运动不确定性。然而，依赖高清地图的模型由于成本高昂的数据采集、延迟的更新以及对污染输入的敏感性，导致预测失败。无地图方法缺乏全局上下文，对等注意机制过度放大直线模式而抑制过渡模式，导致运动意图对齐不良。本文提出GContextFormer，一种插即用的编码器-解码器架构，结合全局上下文感知的混合注意机制和缩放相加聚合，实现意图对齐的多模态预测，无需依赖地图。运动感知编码器通过有界缩放相加聚合构建场景级意图先验并在共享全局上下文中细化每种模式表示，缓解模式间的抑制并促进意图对齐。层次交互解码器将社会推理分解为双重路径交叉注意：标准路径确保在代理模式对上均匀的几何覆盖，而邻居上下文增强路径强调显著交互，门控模块调节它们的贡献以维持覆盖重点平衡。实验结果显示，GContextFormer在TOD-VT数据集中的八个高速路匝道场景中优于现有基线方法。与现有的变压器模型相比，GContextFormer通过空间分布实现了更强的鲁棒性和在高曲率和过渡区域的集中改进。通过运动模式区分和邻居上下文调节实现可解释性，模块化架构支持向跨域多模态推理任务扩展。 

---
# MAGMA-Edu: Multi-Agent Generative Multimodal Framework for Text-Diagram Educational Question Generation 

**Title (ZH)**: MAGMA-Edu：多代理生成多模态文本-图表教育问题生成框架 

**Authors**: Zhenyu Wu, Jian Li, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18714)  

**Abstract**: Educational illustrations play a central role in communicating abstract concepts, yet current multimodal large language models (MLLMs) remain limited in producing pedagogically coherent and semantically consistent educational visuals. We introduce MAGMA-Edu, a self-reflective multi-agent framework that unifies textual reasoning and diagrammatic synthesis for structured educational problem generation. Unlike existing methods that treat text and image generation independently, MAGMA-Edu employs a two-stage co-evolutionary pipeline: (1) a generation-verification-reflection loop that iteratively refines question statements and solutions for mathematical accuracy, and (2) a code-based intermediate representation that enforces geometric fidelity and semantic alignment during image rendering. Both stages are guided by internal self-reflection modules that evaluate and revise outputs until domain-specific pedagogical constraints are met. Extensive experiments on multimodal educational benchmarks demonstrate the superiority of MAGMA-Edu over state-of-the-art MLLMs. Compared to GPT-4o, MAGMA-Edu improves the average textual metric from 57.01 to 92.31 (+35.3 pp) and boosts image-text consistency (ITC) from 13.20 to 85.24 (+72 pp). Across all model backbones, MAGMA-Edu achieves the highest scores (Avg-Text 96.20, ITC 99.12), establishing a new state of the art for multimodal educational content generation and demonstrating the effectiveness of self-reflective multi-agent collaboration in pedagogically aligned vision-language reasoning. 

**Abstract (ZH)**: 教育插图在传达抽象概念中扮演着关键角色，然而当前的多模态大型语言模型在生成教学连贯且语义一致的教育可视化方面仍然有限。我们引入了MAGMA-Edu，这是一种自反性的多代理框架，结合了文本推理和图示合成，用于结构化教育问题生成。MAGMA-Edu不同于现有方法将文本和图像生成独立处理，它采用两阶段共生演化管道：（1）生成-验证-反思循环，迭代细化数学准确性的问题陈述和解决方案；（2）基于代码的中间表示，在图像渲染过程中确保几何保真度和语义对齐。两个阶段均由内部自反模块引导，评估和修订输出，直到满足特定领域的教学约束。在多模态教育基准测试中的广泛实验表明，MAGMA-Edu在多模态教育内容生成方面优于当前最先进的多模态大型语言模型。与GPT-4o相比，MAGMA-Edu在平均文本指标上提高了35.3个百分点（从57.01提高到92.31），并提高了图像-文本一致性（ITC）72个百分点（从13.20提高到85.24）。在所有模型底座上，MAGMA-Edu取得了最高分数（平均文本96.20，ITC 99.12），为多模态教育内容生成设立了新的前沿，并展示了教学对齐的视觉-语言推理中自反性多代理协作的有效性。 

---
# ChemVTS-Bench: Evaluating Visual-Textual-Symbolic Reasoning of Multimodal Large Language Models in Chemistry 

**Title (ZH)**: ChemVTS-基准：评估多模态大型语言模型在化学领域的视觉-文本-符号推理能力 

**Authors**: Zhiyuan Huang, Baichuan Yang, Zikun He, Yanhong Wu, Fang Hongyu, Zhenhe Liu, Lin Dongsheng, Bing Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.17909)  

**Abstract**: Chemical reasoning inherently integrates visual, textual, and symbolic modalities, yet existing benchmarks rarely capture this complexity, often relying on simple image-text pairs with limited chemical semantics. As a result, the actual ability of Multimodal Large Language Models (MLLMs) to process and integrate chemically meaningful information across modalities remains unclear. We introduce \textbf{ChemVTS-Bench}, a domain-authentic benchmark designed to systematically evaluate the Visual-Textual-Symbolic (VTS) reasoning abilities of MLLMs. ChemVTS-Bench contains diverse and challenging chemical problems spanning organic molecules, inorganic materials, and 3D crystal structures, with each task presented in three complementary input modes: (1) visual-only, (2) visual-text hybrid, and (3) SMILES-based symbolic input. This design enables fine-grained analysis of modality-dependent reasoning behaviors and cross-modal integration. To ensure rigorous and reproducible evaluation, we further develop an automated agent-based workflow that standardizes inference, verifies answers, and diagnoses failure modes. Extensive experiments on state-of-the-art MLLMs reveal that visual-only inputs remain challenging, structural chemistry is the hardest domain, and multimodal fusion mitigates but does not eliminate visual, knowledge-based, or logical errors, highlighting ChemVTS-Bench as a rigorous, domain-faithful testbed for advancing multimodal chemical reasoning. All data and code will be released to support future research. 

**Abstract (ZH)**: ChemVTS-Bench：一种系统评估多模态大型语言模型化学推理能力的标准基准 

---
# M3-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark 

**Title (ZH)**: M3-Bench：多模态、多跳、多线程工具使用机器学习大模型代理基准 

**Authors**: Yang Zhou, Mingyu Zhao, Zhenting Wang, Difei Gu, Bangwei Guo, Ruosong Ye, Ligong Han, Can Jin, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2511.17729)  

**Abstract**: We present M^3-Bench, the first benchmark for evaluating multimodal tool use under the Model Context Protocol. The benchmark targets realistic, multi-hop and multi-threaded workflows that require visual grounding and textual reasoning, cross-tool dependencies, and persistence of intermediate resources across steps. We introduce a similarity-driven alignment that serializes each tool call, embeds signatures with a sentence encoder, and performs similarity-bucketed Hungarian matching to obtain auditable one-to-one correspondences. On top of this alignment, we report interpretable metrics that decouple semantic fidelity from workflow consistency. The benchmark spans 28 servers with 231 tools, and provides standardized trajectories curated through an Executor & Judge pipeline with human verification; an auxiliary four large language models (LLMs) judge ensemble reports end-task Task Completion and information grounding. Evaluations of representative state-of-the-art Multimodal LLMs (MLLMs) reveal persistent gaps in multimodal MCP tool use, particularly in argument fidelity and structure consistency, underscoring the need for methods that jointly reason over images, text, and tool graphs. Our Benchmark's anonymous repository is at this https URL 

**Abstract (ZH)**: M^3-Bench：Model Context Protocol下多模态工具使用评估基准 

---
# Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens 

**Title (ZH)**: 视觉链思考：通过连续视觉标记教会VLMs更好地看与思考 

**Authors**: Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, Xudong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19418)  

**Abstract**: Vision-Language Models (VLMs) excel at reasoning in linguistic space but struggle with perceptual understanding that requires dense visual perception, e.g., spatial reasoning and geometric awareness. This limitation stems from the fact that current VLMs have limited mechanisms to capture dense visual information across spatial dimensions. We introduce Chain-of-Visual-Thought (COVT), a framework that enables VLMs to reason not only in words but also through continuous visual tokens-compact latent representations that encode rich perceptual cues. Within a small budget of roughly 20 tokens, COVT distills knowledge from lightweight vision experts, capturing complementary properties such as 2D appearance, 3D geometry, spatial layout, and edge structure. During training, the VLM with COVT autoregressively predicts these visual tokens to reconstruct dense supervision signals (e.g., depth, segmentation, edges, and DINO features). At inference, the model reasons directly in the continuous visual token space, preserving efficiency while optionally decoding dense predictions for interpretability. Evaluated across more than ten diverse perception benchmarks, including CV-Bench, MMVP, RealWorldQA, MMStar, WorldMedQA, and HRBench, integrating COVT into strong VLMs such as Qwen2.5-VL and LLaVA consistently improves performance by 3% to 16% and demonstrates that compact continuous visual thinking enables more precise, grounded, and interpretable multimodal intelligence. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在语言空间推理方面表现出色，但在需要密集视觉感知的能力方面存在局限，如空间推理和几何意识。这种局限性源于当前VLMs在捕获跨空间维度的密集视觉信息方面的机制有限。我们提出了连续视觉思考链（COVT）框架，该框架使VLMs不仅在语言中推理，还在连续视觉标记中推理——这些紧凑的潜在表示编码丰富的感知线索。在大约20个令牌的小预算内，COVT从轻量级视觉专家中提取知识，捕捉包括2D外观、3D几何、空间布局和边缘结构在内的互补属性。在训练期间，带有COVT的VLM自回归预测这些视觉标记以重建密集监督信号（例如，深度、分割、边缘和DINO特征）。在推理期间，模型直接在连续视觉标记空间中推理，保持高效的同时可选地解码密集预测以提高可解释性。在CV-Bench、MMVP、RealWorldQA、MMStar、WorldMedQA和HRBench等多个多样的感知基准测试中评估，将COVT集成到强VLMs如Qwen2.5-VL和LLaVA中，性能普遍提高了3%到16%，证明了紧凑的连续视觉思考能够实现更精确、更落地和可解释的多模态智能。 

---
# UniGame: Turning a Unified Multimodal Model Into Its Own Adversary 

**Title (ZH)**: UniGame: 将统一多模态模型转变为自身的对手模型 

**Authors**: Zhaolong Su, Wang Lu, Hao Chen, Sharon Li, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19413)  

**Abstract**: Unified Multimodal Models (UMMs) have shown impressive performance in both understanding and generation with a single architecture. However, UMMs still exhibit a fundamental inconsistency: understanding favors compact embeddings, whereas generation favors reconstruction-rich representations. This structural trade-off produces misaligned decision boundaries, degraded cross-modal coherence, and heightened vulnerability under distributional and adversarial shifts. In this paper, we present UniGame, a self-adversarial post-training framework that directly targets the inconsistencies. By applying a lightweight perturber at the shared token interface, UniGame enables the generation branch to actively seek and challenge fragile understanding, turning the model itself into its own adversary. Experiments demonstrate that UniGame significantly improves the consistency (+4.6%). Moreover, it also achieves substantial improvements in understanding (+3.6%), generation (+0.02), out-of-distribution and adversarial robustness (+4.8% and +6.2% on NaturalBench and AdVQA). The framework is architecture-agnostic, introduces less than 1% additional parameters, and is complementary to existing post-training methods. These results position adversarial self-play as a general and effective principle for enhancing the coherence, stability, and unified competence of future multimodal foundation models. The official code is available at: this https URL 

**Abstract (ZH)**: 统一多模态模型中的自对抗后训练框架：UniGame 

---
# Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation 

**Title (ZH)**: Medusa：跨模态迁移式对抗攻击在多模态医疗检索增强生成中的应用 

**Authors**: Yingjia Shang, Yi Liu, Huimin Wang, Furong Li, Wenfang Sun, Wu Chengyu, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19257)  

**Abstract**: With the rapid advancement of retrieval-augmented vision-language models, multimodal medical retrieval-augmented generation (MMed-RAG) systems are increasingly adopted in clinical decision support. These systems enhance medical applications by performing cross-modal retrieval to integrate relevant visual and textual evidence for tasks, e.g., report generation and disease diagnosis. However, their complex architecture also introduces underexplored adversarial vulnerabilities, particularly via visual input perturbations. In this paper, we propose Medusa, a novel framework for crafting cross-modal transferable adversarial attacks on MMed-RAG systems under a black-box setting. Specifically, Medusa formulates the attack as a perturbation optimization problem, leveraging a multi-positive InfoNCE loss (MPIL) to align adversarial visual embeddings with medically plausible but malicious textual targets, thereby hijacking the retrieval process. To enhance transferability, we adopt a surrogate model ensemble and design a dual-loop optimization strategy augmented with invariant risk minimization (IRM). Extensive experiments on two real-world medical tasks, including medical report generation and disease diagnosis, demonstrate that Medusa achieves over 90% average attack success rate across various generation models and retrievers under appropriate parameter configuration, while remaining robust against four mainstream defenses, outperforming state-of-the-art baselines. Our results reveal critical vulnerabilities in the MMed-RAG systems and highlight the necessity of robustness benchmarking in safety-critical medical applications. The code and data are available at this https URL. 

**Abstract (ZH)**: 基于检索增强的多模态医学生成系统（MMed-RAG）的跨模态可转移 adversarial 攻击框架：Medusa 

---
# CLASH: A Benchmark for Cross-Modal Contradiction Detection 

**Title (ZH)**: CLASH: 多模态矛盾检测基准 

**Authors**: Teodora Popordanoska, Jiameng Li, Matthew B. Blaschko  

**Link**: [PDF](https://arxiv.org/pdf/2511.19199)  

**Abstract**: Contradictory multimodal inputs are common in real-world settings, yet existing benchmarks typically assume input consistency and fail to evaluate cross-modal contradiction detection - a fundamental capability for preventing hallucinations and ensuring reliability. We introduce CLASH, a novel benchmark for multimodal contradiction detection, featuring COCO images paired with contradictory captions containing controlled object-level or attribute-level contradictions. The samples include targeted questions evaluated in both multiple-choice and open-ended formats. The benchmark provides an extensive fine-tuning set filtered through automated quality checks, alongside a smaller human-verified diagnostic set. Our analysis of state-of-the-art models reveals substantial limitations in recognizing cross-modal conflicts, exposing systematic modality biases and category-specific weaknesses. Furthermore, we empirically demonstrate that targeted fine-tuning on CLASH substantially enhances conflict detection capabilities. 

**Abstract (ZH)**: 矛盾的多模态输入在现实世界中很常见，现有基准通常假设输入一致性并未能评估跨模态矛盾检测能力——这是预防幻觉和确保可靠性的一项基本能力。我们引入了CLASH，一个全新的多模态矛盾检测基准，该基准包括COCO图像配对矛盾的caption，控制在对象级别或属性级别。样本包括在选择题和开放式格式下评估的目标问题。基准数据集包括经过自动化质量检查过滤的广泛微调集，以及较小的人工验证诊断集。我们的高级分析揭示了最先进的模型在识别跨模态冲突方面存在重大局限性，暴露出系统的模态偏差和类别特异性弱点。此外，我们实证证明，针对CLASH进行目标微调极大地提升了冲突检测能力。 

---
# ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection 

**Title (ZH)**: ConceptGuard: 多模态风险检测下的 proactive 安全文本和图像生成视频 

**Authors**: Ruize Ma, Minghong Cai, Yilei Jiang, Jiaming Han, Yi Feng, Yingshui Tan, Xiaoyong Zhu, Bo Zhang, Bo Zheng, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2511.18780)  

**Abstract**: Recent progress in video generative models has enabled the creation of high-quality videos from multimodal prompts that combine text and images. While these systems offer enhanced controllability, they also introduce new safety risks, as harmful content can emerge from individual modalities or their interaction. Existing safety methods are often text-only, require prior knowledge of the risk category, or operate as post-generation auditors, struggling to proactively mitigate such compositional, multimodal risks. To address this challenge, we present ConceptGuard, a unified safeguard framework for proactively detecting and mitigating unsafe semantics in multimodal video generation. ConceptGuard operates in two stages: First, a contrastive detection module identifies latent safety risks by projecting fused image-text inputs into a structured concept space; Second, a semantic suppression mechanism steers the generative process away from unsafe concepts by intervening in the prompt's multimodal conditioning. To support the development and rigorous evaluation of this framework, we introduce two novel benchmarks: ConceptRisk, a large-scale dataset for training on multimodal risks, and T2VSafetyBench-TI2V, the first benchmark adapted from T2VSafetyBench for the Text-and-Image-to-Video (TI2V) safety setting. Comprehensive experiments on both benchmarks show that ConceptGuard consistently outperforms existing baselines, achieving state-of-the-art results in both risk detection and safe video generation. 

**Abstract (ZH)**: 近期视频生成模型的发展使得从结合文本和图像的多模态提示中创建高质量视频成为可能。尽管这些系统提高了可控性，但也引入了新的安全风险，因为有害内容可能源自单一模态或它们的交互。现有安全方法通常是文本-only 的，需要事先知道风险类别，或者作为后生成审查器运行，难以积极缓解这类组合多模态风险。为解决这一挑战，我们提出了ConceptGuard，这是一种统一的安全防护框架，旨在前瞻性地检测和缓解多模态视频生成中的不安全语义。ConceptGuard分为两个阶段：首先，对比检测模块通过将融合的图像-文本输入投影到结构化的概念空间中来识别潜在的安全风险；其次，语义抑制机制通过干预提示的多模态条件来引导生成过程远离不安全的概念。为支持该框架的开发和严格评估，我们引入了两个新的基准：ConceptRisk，一个大规模的数据集，用于多模态风险训练；以及T2VSafetyBench-TI2V，这是首个针对文本和图像生成视频安全设置改编的T2VSafetyBench基准。在两个基准上的全面实验表明，ConceptGuard在风险检测和安全视频生成方面均优于现有基线，达到最先进的技术水平。 

---
# AIRHILT: A Human-in-the-Loop Testbed for Multimodal Conflict Detection in Aviation 

**Title (ZH)**: AIRHILT：航空多模态冲突检测的人机交互实验平台 

**Authors**: Omar Garib, Jayaprakash D. Kambhampaty, Olivia J. Pinon Fischer, Dimitri N. Mavris  

**Link**: [PDF](https://arxiv.org/pdf/2511.18718)  

**Abstract**: We introduce AIRHILT (Aviation Integrated Reasoning, Human-in-the-Loop Testbed), a modular and lightweight simulation environment designed to evaluate multimodal pilot and air traffic control (ATC) assistance systems for aviation conflict detection. Built on the open-source Godot engine, AIRHILT synchronizes pilot and ATC radio communications, visual scene understanding from camera streams, and ADS-B surveillance data within a unified, scalable platform. The environment supports pilot- and controller-in-the-loop interactions, providing a comprehensive scenario suite covering both terminal area and en route operational conflicts, including communication errors and procedural mistakes. AIRHILT offers standardized JSON-based interfaces that enable researchers to easily integrate, swap, and evaluate automatic speech recognition (ASR), visual detection, decision-making, and text-to-speech (TTS) models. We demonstrate AIRHILT through a reference pipeline incorporating fine-tuned Whisper ASR, YOLO-based visual detection, ADS-B-based conflict logic, and GPT-OSS-20B structured reasoning, and present preliminary results from representative runway-overlap scenarios, where the assistant achieves an average time-to-first-warning of approximately 7.7 s, with average ASR and vision latencies of approximately 5.9 s and 0.4 s, respectively. The AIRHILT environment and scenario suite are openly available, supporting reproducible research on multimodal situational awareness and conflict detection in aviation; code and scenarios are available at this https URL. 

**Abstract (ZH)**: AIRHILT（航空集成推理、有人参与测试平台）：一种用于评估航空冲突检测多模态飞行员与空中交通管制辅助系统的模块化轻量级模拟环境 

---
# Modality-Collaborative Low-Rank Decomposers for Few-Shot Video Domain Adaptation 

**Title (ZH)**: 基于模态协作低秩分解的少样本视频领域自适应 

**Authors**: Yuyang Wanyan, Xiaoshan Yang, Weiming Dong, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18711)  

**Abstract**: In this paper, we study the challenging task of Few-Shot Video Domain Adaptation (FSVDA). The multimodal nature of videos introduces unique challenges, necessitating the simultaneous consideration of both domain alignment and modality collaboration in a few-shot scenario, which is ignored in previous literature. We observe that, under the influence of domain shift, the generalization performance on the target domain of each individual modality, as well as that of fused multimodal features, is constrained. Because each modality is comprised of coupled features with multiple components that exhibit different domain shifts. This variability increases the complexity of domain adaptation, thereby reducing the effectiveness of multimodal feature integration. To address these challenges, we introduce a novel framework of Modality-Collaborative LowRank Decomposers (MC-LRD) to decompose modality-unique and modality-shared features with different domain shift levels from each modality that are more friendly for domain alignment. The MC-LRD comprises multiple decomposers for each modality and Multimodal Decomposition Routers (MDR). Each decomposer has progressively shared parameters across different modalities. The MDR is leveraged to selectively activate the decomposers to produce modality-unique and modality-shared features. To ensure efficient decomposition, we apply orthogonal decorrelation constraints separately to decomposers and subrouters, enhancing their diversity. Furthermore, we propose a cross-domain activation consistency loss to guarantee that target and source samples of the same category exhibit consistent activation preferences of the decomposers, thereby facilitating domain alignment. Extensive experimental results on three public benchmarks demonstrate that our model achieves significant improvements over existing methods. 

**Abstract (ZH)**: Few-Shot 视频领域适应的模态协作低秩分解方法 

---
# Multimodal Real-Time Anomaly Detection and Industrial Applications 

**Title (ZH)**: 多模态实时异常检测及其工业应用 

**Authors**: Aman Verma, Keshav Samdani, Mohd. Samiuddin Shafi  

**Link**: [PDF](https://arxiv.org/pdf/2511.18698)  

**Abstract**: This paper presents the design, implementation, and evolution of a comprehensive multimodal room-monitoring system that integrates synchronized video and audio processing for real-time activity recognition and anomaly detection. We describe two iterations of the system: an initial lightweight implementation using YOLOv8, ByteTrack, and the Audio Spectrogram Transformer (AST), and an advanced version that incorporates multi-model audio ensembles, hybrid object detection, bidirectional cross-modal attention, and multi-method anomaly detection. The evolution demonstrates significant improvements in accuracy, robustness, and industrial applicability. The advanced system combines three audio models (AST, Wav2Vec2, and HuBERT) for comprehensive audio understanding, dual object detectors (YOLO and DETR) for improved accuracy, and sophisticated fusion mechanisms for enhanced cross-modal learning. Experimental evaluation shows the system's effectiveness in general monitoring scenarios as well as specialized industrial safety applications, achieving real-time performance on standard hardware while maintaining high accuracy. 

**Abstract (ZH)**: 本文介绍了综合多模态房间监控系统的架构、实现及其演变，该系统结合了同步视频和音频处理，以实现实时活动识别和异常检测。我们描述了该系统的两个版本：一个使用YOLOv8、ByteTrack和Audio Spectrogram Transformer (AST)的初步轻量级实现，以及一个包含多模型音频集成、混合对象检测、双向跨模态注意和多方法异常检测的高级版本。系统的演变展示了在准确性、稳健性和工业适用性方面的重要改进。高级系统结合了三种音频模型（AST、Wav2Vec2和HuBERT），实现了全面的音频理解，采用了双对象检测器（YOLO和DETR）以提高准确性，并通过复杂的融合机制增强了跨模态学习。实验评估表明，该系统在一般监控场景以及专门的工业安全应用中均表现出有效性，并能够实现实现在标准硬件上的实时性能同时保持高准确性。 

---
# Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives 

**Title (ZH)**: 多模态持续学习：从多场景视角出发的MLLMs的研究 

**Authors**: Kai Jiang, Siqi Huang, Xiangyu Chen, Jiawei Shao, Hongyuan Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.18507)  

**Abstract**: Continual learning in visual understanding aims to deal with catastrophic forgetting in Multimodal Large Language Models (MLLMs). MLLMs deployed on devices have to continuously adapt to dynamic scenarios in downstream tasks, such as variations in background and perspective, to effectively perform complex visual tasks. To this end, we construct a multimodal visual understanding dataset (MSVQA) encompassing four different scenarios and perspectives including high altitude, underwater, low altitude and indoor, to investigate the catastrophic forgetting in MLLMs under the dynamics of scenario shifts in real-world data streams. Furthermore, we propose mUltimodal coNtInual learning with MLLMs From multi-scenarIo pERspectives (UNIFIER) to address visual discrepancies while learning different scenarios. Specifically, it decouples the visual information from different scenarios into distinct branches within each vision block and projects them into the same feature space. A consistency constraint is imposed on the features of each branch to maintain the stability of visual representations across scenarios. Extensive experiments on the MSVQA dataset demonstrate that UNIFIER effectively alleviates forgetting of cross-scenario tasks and achieves knowledge accumulation within the same scenario. 

**Abstract (ZH)**: 持续学习在视觉理解中的应用旨在解决多模态大型语言模型（MLLMs）中的灾难性遗忘问题。部署在设备上的MLLMs需要不断适应下游任务中的动态场景，如背景和视角的变化，以有效地执行复杂的视觉任务。为此，我们构建了一个涵盖高海拔、水下、低海拔和室内四种不同场景和视角的多模态视觉理解数据集（MSVQA），以研究真实世界数据流中场景转换动态下的MLLMs的灾难性遗忘问题。此外，我们提出了基于多场景视角的多模态持续学习方法（UNIFIER），以在学习不同场景的同时解决视觉差异问题。具体来说，UNIFIER 在每个视觉块内将来自不同场景的视觉信息分离成不同的分支，并将它们投影到同一特征空间。在每个分支的特征上施加一致性约束，以维持跨场景的视觉表示的稳定性。在MSVQA数据集上的广泛实验表明，UNIFIER 有效缓解了跨场景任务的遗忘问题，并在相同场景内实现了知识积累。 

---
# Can a Second-View Image Be a Language? Geometric and Semantic Cross-Modal Reasoning for X-ray Prohibited Item Detection 

**Title (ZH)**: 第二视角图像能成为一种语言吗？X射线禁止物品检测的几何与语义跨模态推理 

**Authors**: Chuang Peng, Renshuai Tao, Zhongwei Ren, Xianglong Liu, Yunchao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2511.18385)  

**Abstract**: Automatic X-ray prohibited items detection is vital for security inspection and has been widely studied. Traditional methods rely on visual modality, often struggling with complex threats. While recent studies incorporate language to guide single-view images, human inspectors typically use dual-view images in practice. This raises the question: can the second view provide constraints similar to a language modality? In this work, we introduce DualXrayBench, the first comprehensive benchmark for X-ray inspection that includes multiple views and modalities. It supports eight tasks designed to test cross-view reasoning. In DualXrayBench, we introduce a caption corpus consisting of 45,613 dual-view image pairs across 12 categories with corresponding captions. Building upon these data, we propose the Geometric (cross-view)-Semantic (cross-modality) Reasoner (GSR), a multimodal model that jointly learns correspondences between cross-view geometry and cross-modal semantics, treating the second-view images as a "language-like modality". To enable this, we construct the GSXray dataset, with structured Chain-of-Thought sequences: <top>, <side>, <conclusion>. Comprehensive evaluations on DualXrayBench demonstrate that GSR achieves significant improvements across all X-ray tasks, offering a new perspective for real-world X-ray inspection. 

**Abstract (ZH)**: 自动X射线禁止物品检测对于安全检查至关重要且已被广泛研究。 

---
# AnyExperts: On-Demand Expert Allocation for Multimodal Language Models with Mixture of Expert 

**Title (ZH)**: AnyExperts: 按需分配专家机制的多模态语言模型专家混合架构 

**Authors**: Yuting Gao, Wang Lan, Hengyuan Zhao, Linjiang Huang, Si Liu, Qingpei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.18314)  

**Abstract**: Multimodal Mixture-of-Experts (MoE) models offer a promising path toward scalable and efficient large vision-language systems. However, existing approaches rely on rigid routing strategies (typically activating a fixed number of experts per token) ignoring the inherent heterogeneity in semantic importance across modalities. This leads to suboptimal compute allocation, where redundant tokens consume as many resources as critical ones. To address this, we propose AnyExperts, a novel on-demand, budget-aware dynamic routing framework that allocates a variable total number of expert slots per token based on its semantic importance. Crucially, to prevent uncontrolled compute growth, the total slots per token are constrained within a fixed range, and each slot is filled by either a real expert or a virtual expert, with the virtual share capped at a small maximum (e.g., 20%). The model then adaptively balances the real-to-virtual ratio per token, assigning more real experts to semantically rich regions and relying more on virtual experts for redundant content. Evaluated across diverse tasks in visual understanding, audio understanding, and NLP understanding, AnyExperts improves performance under the same compute budget. Notably, on general image/video tasks, it achieves comparable accuracy with 40% fewer real expert activations; on text-dense tasks (OCR and NLP), it maintains performance while reducing real expert usage by 10%. These results demonstrate that fine-grained, importance-driven expert allocation significantly enhances both the efficiency and effectiveness of multimodal MoE models. 

**Abstract (ZH)**: 多模态Mixture-of-Experts (MoE)模型提供了构建可扩展和高效的大规模视觉-语言系统的有希望途径。然而，现有方法依赖于刚性的路由策略（通常每个令牌激活固定数量的专家），忽视了不同模态内固有的语义重要性异质性。这导致了计算资源分配不优化，其中冗余令牌消耗与关键令牌相同数量的资源。为解决这一问题，我们提出了AnyExperts，这是一种新型的按需、预算感知的动态路由框架，根据其语义重要性为每个令牌分配可变数量的专家槽。最关键的是，为防止计算资源无控制的增长，每个令牌的总槽数被限制在一个固定范围内，并且每个槽要么由真正的专家填充，要么由虚拟专家填充，虚拟部分的最大值被限制在一个小的范围内（例如，20%）。模型随后根据每个令牌的实际情况动态平衡真实专家和虚拟专家的比例，对富含语义的区域分配更多真实专家，并对冗余内容更多依赖虚拟专家。在视觉理解、音频理解以及自然语言处理等多种任务上进行评估，AnyExperts在相同的计算预算下提高了性能。值得注意的是，在通用图像/视频任务上，它通过减少40%的真实专家激活次数实现了 comparable 准确度；在文本密集的任务（OCR和NLP）上，它保持了性能同时将真实专家的使用量减少了10%。这些结果表明，精细粒度、基于重要性的专家分配显着增强了多模态MoE模型的效率和效果。 

---
# Bias Is a Subspace, Not a Coordinate: A Geometric Rethinking of Post-hoc Debiasing in Vision-Language Models 

**Title (ZH)**: 偏差是一个子空间，而不是一个坐标：视觉-语言模型中后验去偏见的一种几何重思。 

**Authors**: Dachuan Zhao, Weiyue Li, Zhenda Shen, Yushu Qiu, Bowen Xu, Haoyu Chen, Yongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.18123)  

**Abstract**: Vision-Language Models (VLMs) have become indispensable for multimodal reasoning, yet their representations often encode and amplify demographic biases, resulting in biased associations and misaligned predictions in downstream tasks. Such behavior undermines fairness and distorts the intended alignment between vision and language. Recent post-hoc approaches attempt to mitigate bias by replacing the most attribute-correlated embedding coordinates with neutral values. However, our systematic analysis reveals three critical failures of this coordinate-wise approach: feature entanglement, poor cross-dataset generalization, and incomplete bias removal. We find that bias is not localized to a few coordinates but is instead distributed across a few linear subspaces. To address these limitations, we propose $\textbf{S}$ubspace $\textbf{P}$rojection $\textbf{D}$ebiasing ($\textbf{SPD}$), a geometrically principled framework that identifies and removes the entire subspace of linearly decodable bias while reinserting a neutral mean component to preserve semantic fidelity. Extensive experiments across zero-shot classification, text-to-image retrieval, and image generation validate the effectiveness of SPD: our method achieves more robust debiasing with an average improvement of $18.5\%$ across four fairness metrics, while maintaining minimal loss in task performance compared to the best debiasing baseline. 

**Abstract (ZH)**: 基于子空间投影的视觉语言模型去偏方法（Subspace Projection Debiasing for Vision-Language Models） 

---
# VCU-Bridge: Hierarchical Visual Connotation Understanding via Semantic Bridging 

**Title (ZH)**: VCU-桥梁：基于语义桥梁的层级视觉内涵理解 

**Authors**: Ming Zhong, Yuanlei Wang, Liuzhou Zhang, Arctanx An, Renrui Zhang, Hao Liang, Ming Lu, Ying Shen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18121)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel on benchmarks, their processing paradigm differs from the human ability to integrate visual information. Unlike humans who naturally bridge details and high-level concepts, models tend to treat these elements in isolation. Prevailing evaluation protocols often decouple low-level perception from high-level reasoning, overlooking their semantic and causal dependencies, which yields non-diagnostic results and obscures performance bottlenecks. We present VCU-Bridge, a framework that operationalizes a human-like hierarchy of visual connotation understanding: multi-level reasoning that advances from foundational perception through semantic bridging to abstract connotation, with an explicit evidence-to-inference trace from concrete cues to abstract conclusions. Building on this framework, we construct HVCU-Bench, a benchmark for hierarchical visual connotation understanding with explicit, level-wise diagnostics. Comprehensive experiments demonstrate a consistent decline in performance as reasoning progresses to higher levels. We further develop a data generation pipeline for instruction tuning guided by Monte Carlo Tree Search (MCTS) and show that strengthening low-level capabilities yields measurable gains at higher levels. Interestingly, it not only improves on HVCU-Bench but also brings benefits on general benchmarks (average +2.53%), especially with substantial gains on MMStar (+7.26%), demonstrating the significance of the hierarchical thinking pattern and its effectiveness in enhancing MLLM capabilities. The project page is at this https URL . 

**Abstract (ZH)**: 维基桥：一种框架及其在层次视觉内涵理解上的应用与评估 

---
# Plan-X: Instruct Video Generation via Semantic Planning 

**Title (ZH)**: Plan-X: 基于语义规划的视频生成 

**Authors**: Lun Huang, You Xie, Hongyi Xu, Tianpei Gu, Chenxu Zhang, Guoxian Song, Zenan Li, Xiaochen Zhao, Linjie Luo, Guillermo Sapiro  

**Link**: [PDF](https://arxiv.org/pdf/2511.17986)  

**Abstract**: Diffusion Transformers have demonstrated remarkable capabilities in visual synthesis, yet they often struggle with high-level semantic reasoning and long-horizon planning. This limitation frequently leads to visual hallucinations and mis-alignments with user instructions, especially in scenarios involving complex scene understanding, human-object interactions, multi-stage actions, and in-context motion reasoning. To address these challenges, we propose Plan-X, a framework that explicitly enforces high-level semantic planning to instruct video generation process. At its core lies a Semantic Planner, a learnable multimodal language model that reasons over the user's intent from both text prompts and visual context, and autoregressively generates a sequence of text-grounded spatio-temporal semantic tokens. These semantic tokens, complementary to high-level text prompt guidance, serve as structured "semantic sketches" over time for the video diffusion model, which has its strength at synthesizing high-fidelity visual details. Plan-X effectively integrates the strength of language models in multimodal in-context reasoning and planning, together with the strength of diffusion models in photorealistic video synthesis. Extensive experiments demonstrate that our framework substantially reduces visual hallucinations and enables fine-grained, instruction-aligned video generation consistent with multimodal context. 

**Abstract (ZH)**: Plan-X：通过显式高层次语义规划指导视频生成过程 

---
# PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning 

**Title (ZH)**: PA-FAS：基于路径增强强化学习的可解释和泛化的多模态人脸防欺骗方法 

**Authors**: Yingjie Ma, Xun Lin, Yong Xu, Weicheng Xie, Zitong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17927)  

**Abstract**: Face anti-spoofing (FAS) has recently advanced in multimodal fusion, cross-domain generalization, and interpretability. With large language models and reinforcement learning (RL), strategy-based training offers new opportunities to jointly model these aspects. However, multimodal reasoning is more complex than unimodal reasoning, requiring accurate feature representation and cross-modal verification while facing scarce, high-quality annotations, which makes direct application of RL sub-optimal. We identify two key limitations of supervised fine-tuning plus RL (SFT+RL) for multimodal FAS: (1) limited multimodal reasoning paths restrict the use of complementary modalities and shrink the exploration space after SFT, weakening the effect of RL; and (2) mismatched single-task supervision versus diverse reasoning paths causes reasoning confusion, where models may exploit shortcuts by mapping images directly to answers and ignoring the intended reasoning. To address this, we propose PA-FAS, which enhances reasoning paths by constructing high-quality extended reasoning sequences from limited annotations, enriching paths and relaxing exploration constraints. We further introduce an answer-shuffling mechanism during SFT to force comprehensive multimodal analysis instead of using superficial cues, thereby encouraging deeper reasoning and mitigating shortcut learning. PA-FAS significantly improves multimodal reasoning accuracy and cross-domain generalization, and better unifies multimodal fusion, generalization, and interpretability for trustworthy FAS. 

**Abstract (ZH)**: 基于策略的多模态面部防伪处理（PA-FAS）：增强推理路径以提高跨域泛化和可解释性 

---
# Decoupled Audio-Visual Dataset Distillation 

**Title (ZH)**: 解耦音频-视觉数据集精练 

**Authors**: Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2511.17890)  

**Abstract**: Audio-Visual Dataset Distillation aims to compress large-scale datasets into compact subsets while preserving the performance of the original data. However, conventional Distribution Matching (DM) methods struggle to capture intrinsic cross-modal alignment. Subsequent studies have attempted to introduce cross-modal matching, but two major challenges remain: (i) independently and randomly initialized encoders lead to inconsistent modality mapping spaces, increasing training difficulty; and (ii) direct interactions between modalities tend to damage modality-specific (private) information, thereby degrading the quality of the distilled data. To address these challenges, we propose DAVDD, a pretraining-based decoupled audio-visual distillation framework. DAVDD leverages a diverse pretrained bank to obtain stable modality features and uses a lightweight decoupler bank to disentangle them into common and private representations. To effectively preserve cross-modal structure, we further introduce Common Intermodal Matching together with a Sample-Distribution Joint Alignment strategy, ensuring that shared representations are aligned both at the sample level and the global distribution level. Meanwhile, private representations are entirely isolated from cross-modal interaction, safeguarding modality-specific cues throughout distillation. Extensive experiments across multiple benchmarks show that DAVDD achieves state-of-the-art results under all IPC settings, demonstrating the effectiveness of decoupled representation learning for high-quality audio-visual dataset distillation. Code will be released. 

**Abstract (ZH)**: audio-visual数据集精简旨在压缩大规模数据集为紧凑子集的同时保留原始数据的性能。然而，传统的分布匹配方法难以捕捉内在的跨模态对齐。随后的研究尝试引入跨模态匹配，但仍存在两大挑战：（i）独立且随机初始化的编码器导致不一致的模态映射空间，增加训练难度；（ii）模态之间直接交互往往损害模态特定（私有）信息，从而降低精简数据的质量。为应对这些挑战，我们提出DAVDD，一个基于预训练的解耦音频-视觉精简框架。DAVDD 利用多样化的预训练库获得稳定的模态特征，并使用一个轻量级的解耦库将它们分解为公共和私有表示。为了有效保留跨模态结构，我们进一步引入公共跨模态匹配和样本-分布联合对齐策略，确保共享表示在样本级别和全局分布级别均对齐。同时，私有表示完全隔离跨模态交互，在精简过程中保护模态特定线索。在多个基准上的广泛实验表明，DAVDD 在所有IPC设置下均取得了最先进的结果，证明了解耦表示学习在高质量音频-视觉数据集精简中的有效性。代码将开源。 

---
# Reconstruction-Driven Multimodal Representation Learning for Automated Media Understanding 

**Title (ZH)**: 基于重建驱动的多模态表示学习的自动化媒体理解 

**Authors**: Yassir Benhammou, Suman Kalyan, Sujay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.17596)  

**Abstract**: Broadcast and media organizations increasingly rely on artificial intelligence to automate the labor-intensive processes of content indexing, tagging, and metadata generation. However, existing AI systems typically operate on a single modality-such as video, audio, or text-limiting their understanding of complex, cross-modal relationships in broadcast material. In this work, we propose a Multimodal Autoencoder (MMAE) that learns unified representations across text, audio, and visual data, enabling end-to-end automation of metadata extraction and semantic clustering. The model is trained on the recently introduced LUMA dataset, a fully aligned benchmark of multimodal triplets representative of real-world media content. By minimizing joint reconstruction losses across modalities, the MMAE discovers modality-invariant semantic structures without relying on large paired or contrastive datasets. We demonstrate significant improvements in clustering and alignment metrics (Silhouette, ARI, NMI) compared to linear baselines, indicating that reconstruction-based multimodal embeddings can serve as a foundation for scalable metadata generation and cross-modal retrieval in broadcast archives. These results highlight the potential of reconstruction-driven multimodal learning to enhance automation, searchability, and content management efficiency in modern broadcast workflows. 

**Abstract (ZH)**: 广播和媒体组织 increasingly依靠人工智能来自动化内容索引、标签生成和元数据生成等劳动密集型过程。然而，现有的AI系统通常仅在一个模态上操作，如视频、音频或文本，限制了其对广播材料中复杂跨模态关系的理解。在本工作中，我们提出了一种多模态自编码器（MMAE），它在文本、音频和视觉数据之间学习统一表示，从而实现元数据提取和语义聚类的端到端自动化。该模型在近期推出的LUMA数据集上进行训练，这是一个代表真实世界媒体内容的完全对齐的多模态三元组基准数据集。通过对模态之间的联合重构损失进行最小化，MMAE在不依赖大量配对或对比数据集的情况下发现模态不变的语义结构。与线性基线相比，我们在聚类和对齐指标（Silhouette、ARI、NMI）上取得了显著改进，这表明基于重构的多模态嵌入可以作为广播档案中可扩展的元数据生成和跨模态检索的基础。这些结果突显了以重构驱动的多模态学习在增强现代广播工作流中的自动化、可搜索性和内容管理效率方面的潜力。 

---
# Emotion and Intention Guided Multi-Modal Learning for Sticker Response Selection 

**Title (ZH)**: 基于情感和意图引导的多模态学习表情回复选择 

**Authors**: Yuxuan Hu, Jian Chen, Yuhao Wang, Zixuan Li, Jing Xiong, Pengyue Jia, Wei Wang, Chengming Li, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.17587)  

**Abstract**: Stickers are widely used in online communication to convey emotions and implicit intentions. The Sticker Response Selection (SRS) task aims to select the most contextually appropriate sticker based on the dialogue. However, existing methods typically rely on semantic matching and model emotional and intentional cues separately, which can lead to mismatches when emotions and intentions are misaligned. To address this issue, we propose Emotion and Intention Guided Multi-Modal Learning (EIGML). This framework is the first to jointly model emotion and intention, effectively reducing the bias caused by isolated modeling and significantly improving selection accuracy. Specifically, we introduce Dual-Level Contrastive Framework to perform both intra-modality and inter-modality alignment, ensuring consistent representation of emotional and intentional features within and across modalities. In addition, we design an Intention-Emotion Guided Multi-Modal Fusion module that integrates emotional and intentional information progressively through three components: Emotion-Guided Intention Knowledge Selection, Intention-Emotion Guided Attention Fusion, and Similarity-Adjusted Matching Mechanism. This design injects rich, effective information into the model and enables a deeper understanding of the dialogue, ultimately enhancing sticker selection performance. Experimental results on two public SRS datasets show that EIGML consistently outperforms state-of-the-art baselines, achieving higher accuracy and a better understanding of emotional and intentional features. Code is provided in the supplementary materials. 

**Abstract (ZH)**: 基于情感和意图引导的多模态学习（EIGML）方法用于表情贴图响应选择 

---
