# AIRoA MoMa Dataset: A Large-Scale Hierarchical Dataset for Mobile Manipulation 

**Title (ZH)**: AIRoA MoMa 数据集：用于移动操作的大型层次化数据集 

**Authors**: Ryosuke Takanami, Petr Khrapchenkov, Shu Morikuni, Jumpei Arima, Yuta Takaba, Shunsuke Maeda, Takuya Okubo, Genki Sano, Satoshi Sekioka, Aoi Kadoya, Motonari Kambara, Naoya Nishiura, Haruto Suzuki, Takanori Yoshimoto, Koya Sakamoto, Shinnosuke Ono, Hu Yang, Daichi Yashima, Aoi Horo, Tomohiro Motoda, Kensuke Chiyoma, Hiroshi Ito, Koki Fukuda, Akihito Goto, Kazumi Morinaga, Yuya Ikeda, Riko Kawada, Masaki Yoshikawa, Norio Kosuge, Yuki Noguchi, Kei Ota, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2509.25032)  

**Abstract**: As robots transition from controlled settings to unstructured human environments, building generalist agents that can reliably follow natural language instructions remains a central challenge. Progress in robust mobile manipulation requires large-scale multimodal datasets that capture contact-rich and long-horizon tasks, yet existing resources lack synchronized force-torque sensing, hierarchical annotations, and explicit failure cases. We address this gap with the AIRoA MoMa Dataset, a large-scale real-world multimodal dataset for mobile manipulation. It includes synchronized RGB images, joint states, six-axis wrist force-torque signals, and internal robot states, together with a novel two-layer annotation schema of sub-goals and primitive actions for hierarchical learning and error analysis. The initial dataset comprises 25,469 episodes (approx. 94 hours) collected with the Human Support Robot (HSR) and is fully standardized in the LeRobot v2.1 format. By uniquely integrating mobile manipulation, contact-rich interaction, and long-horizon structure, AIRoA MoMa provides a critical benchmark for advancing the next generation of Vision-Language-Action models. The first version of our dataset is now available at this https URL . 

**Abstract (ZH)**: 随着机器人从受控环境过渡到未结构化的居住环境，构建能够可靠遵循自然语言指令的通才代理仍是主要挑战。为了提高稳健的移动操控进展，需要大规模多模态数据集来捕捉富含接触的长期任务，但现有资源缺乏同步的力-力矩感知、层次化注释和明确的失败案例。我们通过AIRoA MoMa数据集填补了这一空白，这是一个用于移动操控的大规模现实世界多模态数据集。该数据集包括同步的RGB图像、关节状态、六轴手腕力-力矩信号以及内部机器人状态，并提供了用于层次化学习和错误分析的新型两层注释方案。初始数据集包含25,469个时期（约94小时），使用Human Support Robot（HSR）收集，并完全符合LeRobot v2.1格式。通过唯一地整合移动操控、富含接触的交互以及长期结构，AIRoA MoMa为推动下一代视觉-语言-动作模型的发展提供了关键基准。我们的数据集第一版现已可用，网址为：this https URL。 

---
# IA-VLA: Input Augmentation for Vision-Language-Action models in settings with semantically complex tasks 

**Title (ZH)**: IA-VLA: 输入增强在语义复杂任务设置下视觉-语言-行动模型中的应用 

**Authors**: Eric Hannus, Miika Malin, Tran Nguyen Le, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2509.24768)  

**Abstract**: Vision-language-action models (VLAs) have become an increasingly popular approach for addressing robot manipulation problems in recent years. However, such models need to output actions at a rate suitable for robot control, which limits the size of the language model they can be based on, and consequently, their language understanding capabilities. Manipulation tasks may require complex language instructions, such as identifying target objects by their relative positions, to specify human intention. Therefore, we introduce IA-VLA, a framework that utilizes the extensive language understanding of a large vision language model as a pre-processing stage to generate improved context to augment the input of a VLA. We evaluate the framework on a set of semantically complex tasks which have been underexplored in VLA literature, namely tasks involving visual duplicates, i.e., visually indistinguishable objects. A dataset of three types of scenes with duplicate objects is used to compare a baseline VLA against two augmented variants. The experiments show that the VLA benefits from the augmentation scheme, especially when faced with language instructions that require the VLA to extrapolate from concepts it has seen in the demonstrations. For the code, dataset, and videos, see this https URL. 

**Abstract (ZH)**: 基于视觉-语言-行动模型（VLAs）的框架：利用大规模视觉语言模型增强语境以应对视觉重复对象的复杂任务 

---
# Multi-Modal Manipulation via Multi-Modal Policy Consensus 

**Title (ZH)**: 多模态操作 via 多模态策略共识 

**Authors**: Haonan Chen, Jiaming Xu, Hongyu Chen, Kaiwen Hong, Binghao Huang, Chaoqi Liu, Jiayuan Mao, Yunzhu Li, Yilun Du, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2509.23468)  

**Abstract**: Effectively integrating diverse sensory modalities is crucial for robotic manipulation. However, the typical approach of feature concatenation is often suboptimal: dominant modalities such as vision can overwhelm sparse but critical signals like touch in contact-rich tasks, and monolithic architectures cannot flexibly incorporate new or missing modalities without retraining. Our method factorizes the policy into a set of diffusion models, each specialized for a single representation (e.g., vision or touch), and employs a router network that learns consensus weights to adaptively combine their contributions, enabling incremental of new representations. We evaluate our approach on simulated manipulation tasks in {RLBench}, as well as real-world tasks such as occluded object picking, in-hand spoon reorientation, and puzzle insertion, where it significantly outperforms feature-concatenation baselines on scenarios requiring multimodal reasoning. Our policy further demonstrates robustness to physical perturbations and sensor corruption. We further conduct perturbation-based importance analysis, which reveals adaptive shifts between modalities. 

**Abstract (ZH)**: 有效整合多种传感模态对于机器人操作至关重要。然而，特征堆叠的典型方法往往不尽如人意：在接触丰富的任务中，视觉等主导模态可能会压垮触觉等稀疏但关键的信号，而单一架构难以灵活地整合新出现或缺失的模态而不重新训练。我们的方法将策略因子化为一系列专门针对单一表示（如视觉或触觉）的扩散模型，并采用一个路由器网络来学习共识权重，以适应性地结合它们的贡献，从而实现新的表示的增量整合。我们在《RLBench》上的模拟操作任务以及实际操作任务（如遮挡物抓取、手持调羹重定位和拼图插入）中评估了该方法，结果表明其在需要多模态推理的情景中显著优于特征堆叠基准。我们的策略还进一步展示了对物理干扰和传感器故障的鲁棒性。我们还进行了基于扰动的重要分析，揭示了不同模态之间的适应性转变。 

---
# RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark 

**Title (ZH)**: RealUnify: 统一模型真能从统一中受益吗？一个全面的基准测试 

**Authors**: Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, Zuyan Liu, Bohan Zeng, Ruizhe Chen, Qixun Wang, Zhuoran Zhang, Xinlong Chen, Chengzhuo Tong, Bozhou Li, Chaoyou Fu, Qiang Liu, Haotian Wang, Wenjing Yang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24897)  

**Abstract**: The integration of visual understanding and generation into unified multimodal models represents a significant stride toward general-purpose AI. However, a fundamental question remains unanswered by existing benchmarks: does this architectural unification actually enable synergetic interaction between the constituent capabilities? Existing evaluation paradigms, which primarily assess understanding and generation in isolation, are insufficient for determining whether a unified model can leverage its understanding to enhance its generation, or use generative simulation to facilitate deeper comprehension. To address this critical gap, we introduce RealUnify, a benchmark specifically designed to evaluate bidirectional capability synergy. RealUnify comprises 1,000 meticulously human-annotated instances spanning 10 categories and 32 subtasks. It is structured around two core axes: 1) Understanding Enhances Generation, which requires reasoning (e.g., commonsense, logic) to guide image generation, and 2) Generation Enhances Understanding, which necessitates mental simulation or reconstruction (e.g., of transformed or disordered visual inputs) to solve reasoning tasks. A key contribution is our dual-evaluation protocol, which combines direct end-to-end assessment with a diagnostic stepwise evaluation that decomposes tasks into distinct understanding and generation phases. This protocol allows us to precisely discern whether performance bottlenecks stem from deficiencies in core abilities or from a failure to integrate them. Through large-scale evaluations of 12 leading unified models and 6 specialized baselines, we find that current unified models still struggle to achieve effective synergy, indicating that architectural unification alone is insufficient. These results highlight the need for new training strategies and inductive biases to fully unlock the potential of unified modeling. 

**Abstract (ZH)**: 视觉理解与生成的整合融入统一多模态模型代表了通用人工智能的重要进展。然而，现有基准未能回答一个基本问题：这种架构整合是否真的能够促进各个组件能力之间的协同互动？现有的评估范式主要孤立地评估理解和生成，不足以判断统一模型是否能够利用其理解能力增强生成，或将生成模拟用于促进更深刻的理解。为填补这一关键缺口，我们引入了RealUnify基准，专门用于评估双向能力协同效应。RealUnify包含1,000个精心人工标注的实例，覆盖10个类别和32个子任务，并围绕两个核心轴构建：1）理解增强生成，要求通过推理（如常识、逻辑）指导图像生成；2）生成增强理解，需要通过心智模拟或重建（如处理变形或混乱的视觉输入）解决推理任务。我们的主要贡献是双评估协议，结合了直接端到端评估和诊断分解评估，后者将任务分解为独立的理解和生成阶段。该协议使我们能够精确判断性能瓶颈是源于核心能力的缺陷还是未能将它们整合。通过12个领先统一模型和6个专门基线的大规模评估，我们发现当前的统一模型仍然难以实现有效协同，表明架构整合本身是不够的。这些结果强调了需要新的训练策略和归纳偏置以完全释放统一建模的潜力。 

---
# AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors 

**Title (ZH)**: AttAnchor: 用注意力锚点引导跨模态 token 对齐的 VLMs 方法 

**Authors**: Junyang Zhang, Tianyi Zhu, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2509.23109)  

**Abstract**: A fundamental reason for the dominance of attention over RNNs and LSTMs in LLMs is its ability to capture long-range dependencies by modeling direct interactions between all tokens, overcoming the sequential limitations of recurrent architectures. Similarly, a key reason why today's vision language models (VLMs) hallucinate and underperform pure language models is that they rely on direct concatenation of image and text tokens with a modality-blinded positional encoding, which conveniently adopts the pretrained LLM backbone but forces unnecessary long-distance attention between semantically related tokens across modalities. This underscores the urgent need for mechanisms that efficiently enhance token locality and cross-modal alignment. In response, we propose Attention Anchor, a parameter-free framework that efficiently groups semantically similar tokens across modalities, improving cross-modal locality. By inserting text tokens near relevant visual patches, we create semantic signposts that reveal true content-based cross-modal attention scores, guiding the model to focus on the correct image regions for tasks such as VQA, MMBench and POPE. This improves answer accuracy and reduces hallucinations without disrupting the prompt's semantic flow. AttAnchor achieves improvements across 13 out of 15 different metrics and benchmarks, including up to 32% gains on reasoning tasks and up to 15% improvements on hallucination benchmarks. AttAnchor enables TinyLLaVA 1B to outperform much larger models like LLaVA 7B and QwenVL 3B on POPE with only 0.1% inference time overhead. To the best of our knowledge, this work is among the first to investigate mixed-modal token grouping, where text and image tokens are clustered jointly into shared groups rather than being grouped within a single modality or merely aligned post-hoc with additional alignment losses. 

**Abstract (ZH)**: 一种基本的原因是注意力机制在大规模语言模型中主宰循环神经网络和长短期记忆网络，是因为它能够通过建模所有令牌之间的直接交互来捕捉长距离依赖关系，从而克服了循环架构的顺序限制。类似地，当今的视觉语言模型（VLMs）产生幻觉并低于纯粹的语言模型的一个关键原因是它们依赖于将图像和文本令牌直接拼接在一起，并使用跨模态盲化的位置编码，这方便地采用了预训练的大规模语言模型主干，但迫使跨模态的语义相关令牌之间进行不必要的长距离注意力。这强调了急需能够高效增强令牌局部性和跨模态对齐的机制。为此，我们提出了注意力锚点（Attention Anchor），这是一种无需参数的框架，能够高效地跨模态聚类语义相似的令牌，从而改进跨模态局部性。通过将文本令牌放置在相关的视觉片段附近，我们创建了语义路标，揭示了基于内容的跨模态注意力评分，从而引导模型在如VQA、MMBench和POPE等任务中聚焦于正确的图像区域。这提高了答案准确性和减少了幻觉现象，而不破坏提示的语义流动。AttAnchor 在 15 个不同的指标和基准测试中实现了 13 个的改进，包括在推理任务中高达 32% 的收益，在幻觉基准测试中高达 15% 的改进。AttAnchor 使得 TinyLLaVA 1B 在仅增加了 0.1% 推理时间开销的情况下，在 POPE 任务上超越了更大的模型如 LLaVA 7B 和 QwenVL 3B。据我们所知，这项工作是首批研究混合模态令牌分组的尝试之一，其中文本和图像令牌被联合聚类到共享组中，而不是在单一模态内聚类或仅在额外的对齐损失下后验对齐。 

---
# VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding 

**Title (ZH)**: VTPerception-R1: 通过显式的视觉和文本感知接地增强多模态推理 

**Authors**: Yizhuo Ding, Mingkang Chen, Zhibang Feng, Tong Xiao, Wanying Qu, Wenqi Shao, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24776)  

**Abstract**: Multimodal large language models (MLLMs) often struggle to ground reasoning in perceptual evidence. We present a systematic study of perception strategies-explicit, implicit, visual, and textual-across four multimodal benchmarks and two MLLMs. Our findings show that explicit perception, especially when paired with textual cues, consistently yields the best improvements, particularly for smaller models. Based on this insight, we propose VTPerception-R1, a unified two-stage framework that decouples perception from reasoning. Stage 1 introduces perception-augmented fine-tuning, and Stage 2 applies perception-aware reinforcement learning with novel visual, textual, and consistency rewards. Experiments demonstrate that VTPerception-R1 significantly improves reasoning accuracy and robustness across diverse tasks, offering a scalable and auditable solution for perception-grounded multimodal reasoning. Our code is available at: this https URL. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）往往难以将推理扎根于感知证据。我们系统研究了感知策略（显式、隐式、视觉和文本）在四个多模态基准和两个MLLMs上的应用。我们的研究发现，尤其是当与文本提示结合时，显式感知可以持续带来最佳改进，尤其是在较小型模型上。基于此洞察，我们提出了一种统一的两阶段框架VTPerception-R1，该框架将感知与推理分离。第一阶段引入感知增强的微调，第二阶段应用感知感知增强的强化学习，并采用新的视觉、文本和一致性奖励。实验表明，VTPerception-R1 显著提高了不同任务的推理准确性和鲁棒性，提供了一种可扩展且可审计的感知导向多模态推理解决方案。我们的代码可在以下链接获取：this https URL。 

---
# A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity 

**Title (ZH)**: A TRIANGLE 矩形实现超越余弦相似性的多模态对齐 

**Authors**: Giordano Cicchetti, Eleonora Grassucci, Danilo Comminiello  

**Link**: [PDF](https://arxiv.org/pdf/2509.24734)  

**Abstract**: Multimodal learning plays a pivotal role in advancing artificial intelligence systems by incorporating information from multiple modalities to build a more comprehensive representation. Despite its importance, current state-of-the-art models still suffer from severe limitations that prevent the successful development of a fully multimodal model. Such methods may not provide indicators that all the involved modalities are effectively aligned. As a result, some modalities may not be aligned, undermining the effectiveness of the model in downstream tasks where multiple modalities should provide additional information that the model fails to exploit. In this paper, we present TRIANGLE: TRI-modAl Neural Geometric LEarning, the novel proposed similarity measure that is directly computed in the higher-dimensional space spanned by the modality embeddings. TRIANGLE improves the joint alignment of three modalities via a triangle-area similarity, avoiding additional fusion layers or pairwise similarities. When incorporated in contrastive losses replacing cosine similarity, TRIANGLE significantly boosts the performance of multimodal modeling, while yielding interpretable alignment rationales. Extensive evaluation in three-modal tasks such as video-text and audio-text retrieval or audio-video classification, demonstrates that TRIANGLE achieves state-of-the-art results across different datasets improving the performance of cosine-based methods up to 9 points of Recall@1. 

**Abstract (ZH)**: 多模态学习在通过集成多种模态信息构建更全面表示以推动人工智能系统发展中发挥关键作用。尽管其重要性，当前最先进的模型仍然遭受严重限制，阻碍了完全多模态模型的成功开发。现有方法可能无法提供所有涉及模态有效对齐的指标。因此，在下游任务中，当多个模态应该提供额外信息但模型未能充分利用时，某些模态可能未对齐，从而削弱了模型的有效性。在本文中，我们提出了TRIANGLE：TRI模态神经几何学习，这是一种直接在由模态嵌入构成的高维空间中计算的新颖相似度度量。TRIANGLE通过三角形面积相似度提高三模态的联合对齐，避免了额外的融合层或成对相似度。当在对比损失中用cosine相似度替换时，TRIANGLE显著提升了多模态建模的性能，同时提供了可解释的对齐理由。在视频-文本和音频-文本检索或音频-视频分类等三项任务的广泛评估中，TRIANGLE在不同数据集中达到了最先进的效果，将基于cosine的方法的Recall@1性能提高了最高9个百分点。 

---
# Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks 

**Title (ZH)**: 欧几里得的馈赠：通过几何代行任务提升视觉语言模型的空间知觉与推理能力 

**Authors**: Shijie Lian, Changti Wu, Laurence Tianruo Yang, Hang Yuan, Bin Yu, Lei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24473)  

**Abstract**: Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs).To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. To enable the model to acquire and apply Euclidean principles from these geometry problems, we employed Group Relative Policy Optimization (GRPO) to finetune the Qwen2.5VL family and RoboBrain2.0 family, inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy of all evaluated models rose from 34.5% to 40.5%, improving by 5.5 percentage points. Among them, RoboBrain2.0-Euclid-7B achieves 49.6\% accuracy, surpassing the previous state-of-the-art model, this http URL our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in this https URL. 

**Abstract (ZH)**: 空间智能涵盖了丰富的能力，包括可视化和变换形状、心理旋转物体、判断相对位置和包含关系，以及估算数量。然而，这仍然是多模态大型语言模型（MLLMs）的一个关键未解挑战。为了填补这一空白，我们提出将欧几里得几何问题解决视为一个替代任务。具体地，我们精心构建了一个多模态数据集，名为Euclid30K，包含约30,000个平面和立体几何问题。为了使模型能够从这些几何问题中学习和应用欧几里得原理，我们使用群相对策略优化（GRPO）微调了Qwen2.5VL家族和RoboBrain2.0家族，激励这些模型识别形状、计数和关联实体，并利用欧几里得原理进行多步演绎推理。我们的实验表明，这些模型在四个空间推理基准测试（Super-CLEVR、Omni3DBench、VSI-Bench和MindCube）上实现了显著的零样本提升，无需任何特定任务的调整。值得注意的是，经过Euclid30K训练后，所有评估模型的VSI-Bench平均准确率从34.5%提升至40.5%，提高了5.5个百分点。其中，RoboBrain2.0-Euclid-7B达到了49.6%的准确率，超过了当时的最先进的模型。据我们所知，这是首次系统性研究证明几何中心化微调可以赋予视觉语言模型广泛的空间技能。代码和Euclid30K数据集可在以下网址获取。 

---
# Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models 

**Title (ZH)**: Uni-X：通过两端分离架构缓解模态冲突的统一多模态模型 

**Authors**: Jitai Hao, Hao Liu, Xinyan Xiao, Qiang Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24365)  

**Abstract**: Unified Multimodal Models (UMMs) built on shared autoregressive (AR) transformers are attractive for their architectural simplicity. However, we identify a critical limitation: when trained on multimodal inputs, modality-shared transformers suffer from severe gradient conflicts between vision and text, particularly in shallow and deep layers. We trace this issue to the fundamentally different low-level statistical properties of images and text, while noting that conflicts diminish in middle layers where representations become more abstract and semantically aligned. To overcome this challenge, we propose Uni-X, a two-end-separated, middle-shared architecture. Uni-X dedicates its initial and final layers to modality-specific processing, while maintaining shared parameters in the middle layers for high-level semantic fusion. This X-shaped design not only eliminates gradient conflicts at both ends but also further alleviates residual conflicts in the shared layers. Extensive experiments validate the effectiveness of Uni-X. Under identical training conditions, Uni-X achieves superior training efficiency compared to strong baselines. When scaled to 3B parameters with larger training data, Uni-X matches or surpasses 7B AR-based UMMs, achieving a GenEval score of 82 for image generation alongside strong performance in text and vision understanding tasks. These results establish Uni-X as a parameter-efficient and scalable foundation for future unified multimodal modeling. Our code is available at this https URL 

**Abstract (ZH)**: 统一跨模态模型Uni-X：两端分离中间共享的架构设计 

---
# Uncovering Grounding IDs: How External Cues Shape Multi-Modal Binding 

**Title (ZH)**: 揭示接地ID：外部线索如何塑造多模态绑定 

**Authors**: Hosein Hasani, Amirmohammad Izadi, Fatemeh Askari, Mobin Bagherian, Sadegh Mohammadian, Mohammad Izadi, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2509.24072)  

**Abstract**: Large vision-language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as robust within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism explaining how external cues enhance multimodal binding, offering both interpretability and practical improvements in robustness. 

**Abstract (ZH)**: 大型多模态视觉-语言模型在多模态基准测试中表现出色，但在结构化推理和精确定位方面仍有限制。近期研究表明，增加简单的视觉结构，如分区和注解，可以提高准确性，但这些改进背后的内部机制尚不明确。我们探讨了这一现象，并提出了Grounding IDs的概念，即由外部线索诱导出的潜藏标识符，它将对象与其指定的分区在不同模态中绑定在一起。通过表示分析，我们发现这些标识符在嵌入空间中表现出内在的一致性，并减少了图像与文本之间的模态差异。因果干预进一步证实这些标识符在对象与符号线索之间起着中介作用。我们展示了Grounding IDs增强了相关组件之间的注意力，从而提高了跨模态定位并减少了幻觉。综上，我们的研究结果将Grounding IDs识别为一种关键的符号机制，它解释了外部线索如何增强多模态绑定，提供了可解释性和在鲁棒性方面的实际改进。 

---
# Vision-Grounded Machine Interpreting: Improving the Translation Process through Visual Cues 

**Title (ZH)**: 基于视觉的机器解释：通过视觉线索改进翻译过程 

**Authors**: Claudio Fantinuoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23957)  

**Abstract**: Machine Interpreting systems are currently implemented as unimodal, real-time speech-to-speech architectures, processing translation exclusively on the basis of the linguistic signal. Such reliance on a single modality, however, constrains performance in contexts where disambiguation and adequacy depend on additional cues, such as visual, situational, or pragmatic information. This paper introduces Vision-Grounded Interpreting (VGI), a novel approach designed to address the limitations of unimodal machine interpreting. We present a prototype system that integrates a vision-language model to process both speech and visual input from a webcam, with the aim of priming the translation process through contextual visual information. To evaluate the effectiveness of this approach, we constructed a hand-crafted diagnostic corpus targeting three types of ambiguity. In our evaluation, visual grounding substantially improves lexical disambiguation, yields modest and less stable gains for gender resolution, and shows no benefit for syntactic ambiguities. We argue that embracing multimodality represents a necessary step forward for advancing translation quality in machine interpreting. 

**Abstract (ZH)**: 基于视觉的地基机器连贯性解释（Vision-Grounded Interpreting: VGI） 

---
# Preserving Cross-Modal Stability for Visual Unlearning in Multimodal Scenarios 

**Title (ZH)**: 在多模态场景中保留跨模态稳定性以实现视觉遗忘 

**Authors**: Jinghan Xu Yuyang Zhang Qixuan Cai Jiancheng Chen Keqiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23895)  

**Abstract**: Visual modality is the most vulnerable to privacy leakage in real-world multimodal applications like autonomous driving with visual and radar data; Machine unlearning removes specific training data from pre-trained models to address privacy leakage, however, existing methods fail to preserve cross-modal knowledge and maintain intra-class structural stability of retain data, leading to reduced overall and other modalities' performance during visual unlearning; to address these challenges, we propose a Cross-modal Contrastive Unlearning (CCU) framework, which integrates three key components: (a) selective visual unlearning: employing inverse contrastive learning to dissociate visual representations from their original semantics, (b) cross-modal knowledge retention: preserving other modalities' discriminability through semantic consistency, and (c) dual-set contrastive separation: preserving the model performance via isolation of structural perturbations between the unlearn set and retain set; extensive experiments on three datasets demonstrate the superiority of CCU, and our method achieves a 7.12% accuracy improvement with only 7% of the unlearning time compared to the top-accuracy baseline. 

**Abstract (ZH)**: 跨模态对比遗忘（CCU）框架：解决视觉模态在自主驾驶等多模态应用中的隐私泄露问题 

---
# Compose and Fuse: Revisiting the Foundational Bottlenecks in Multimodal Reasoning 

**Title (ZH)**: 重组与融合：重新审视多模态推理的基础瓶颈 

**Authors**: Yucheng Wang, Yifan Hou, Aydin Javadov, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23744)  

**Abstract**: Multimodal large language models (MLLMs) promise enhanced reasoning by integrating diverse inputs such as text, vision, and audio. Yet cross-modal reasoning remains underexplored, with conflicting reports on whether added modalities help or harm performance. These inconsistencies stem from a lack of controlled evaluation frameworks and analysis of models' internals to isolate when and why modality interactions support or undermine reasoning. We address this gap through a logic-grounded evaluation framework that categorizes multimodal reasoning into six interaction patterns, varying how facts are distributed across modalities and logically combined. Empirically, additional modalities enhance reasoning only when they provide independent and sufficient reasoning paths, while redundant or chained entailment support often hurts performance. Moreover, reasoning degrades in three systematic ways: weaker modalities drag down overall performance, conflicts bias preference toward certain modalities, and joint signals from different modalities fail to be integrated effectively. Therefore, we identify two core failures: task-composition bottleneck, where recognition and reasoning cannot be jointly executed in one pass, and fusion bottleneck, where early integration introduces bias. For further investigation, we find that attention patterns fail to encode fact usefulness, but a simple two-step prompting (recognize then reason) restores performance, confirming the task-composition bottleneck. Moreover, modality identity remains recoverable in early layers, and softening attention in early fusion improves reasoning, highlighting biased fusion as another failure mode. Overall, our findings show that integration, not perception, is the main barrier to multimodal reasoning, suggesting composition-aware training and early fusion control as promising directions. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过集成文本、视觉和音频等多种输入以增强推理能力。然而，跨模态推理依然未被充分探索，不同研究对增加模态是否有助于性能存在矛盾。这些不一致源于缺乏受控的评估框架和对模型内部机制的分析，以确定何时及为何模态交互支持或妨碍推理。我们通过一个逻辑导向的评估框架来填补这一空白，将多模态推理分为六种交互模式，根据不同事实在各模态中的分布及逻辑组合方式。实证研究显示，只有当额外的模态提供独立且充分的推理路径时，多模态推理才会增强，而冗余或链式推理支持通常会损害性能。此外，推理会在三种系统性方式中退化：较弱的模态拉低整体表现，冲突偏向某些模态的偏好，不同模态的联合信号难以有效整合。因此，我们识别出两种核心失败：任务组合瓶颈，即识别和推理不能在一次通过中共同执行；融合瓶颈，早期集成引入偏差。进一步探索发现，注意力模式未能编码事实有用性，而简单的两步提示（识别后推理）恢复了性能，证实了任务组合瓶颈。此外，模态身份在早期层中仍可恢复，早期融合中的软化注意力改善了推理，强调了偏向性融合为另一种失败模式。总体而言，我们的发现表明，整合而非感知是多模态推理的主要障碍，建议任务感知训练和早期融合控制作为有前途的方向。 

---
# RIV: Recursive Introspection Mask Diffusion Vision Language Model 

**Title (ZH)**: 递归introspection掩码扩散视觉语言模型 

**Authors**: YuQian Li, Limeng Qiao, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23625)  

**Abstract**: Mask Diffusion-based Vision Language Models (MDVLMs) have achieved remarkable progress in multimodal understanding tasks. However, these models are unable to correct errors in generated tokens, meaning they lack self-correction capability. In this paper, we propose Recursive Introspection Mask Diffusion Vision Language Model (RIV), which equips the model with self-correction ability through two novel mechanisms. The first is Introspection Training, where an Introspection Model is introduced to identify errors within generated sequences. Introspection Training enables the model to detect not only grammatical and spelling mistakes, but more importantly, logical errors. The second is Recursive Inference. Beginning with the standard unmasking step, the learned Introspection Model helps to identify errors in the output sequence and remask them. This alternating ($\text{unmask}\rightarrow\text{introspection}\rightarrow\text{remask}$) process is repeated recursively until reliable results are obtained. Experimental results on multiple benchmarks demonstrate that the proposed RIV achieves state-of-the-art performance, outperforming most existing MDVLMs. 

**Abstract (ZH)**: 基于递归反省掩码扩散的视觉语言模型（RIV）在多模态理解任务中取得了显著进展，然而这些模型无法纠正生成的令牌中的错误，意味着它们缺乏自我修正能力。本文提出了一种递归反省掩码扩散视觉语言模型（RIV），通过两种新的机制赋予模型自我修正能力。首先是反省训练，引入一种反省模型来识别生成序列中的错误。反省训练使模型不仅能检测到语法和拼写错误，更重要的是，能检测逻辑错误。其次是递归推理。从标准的去掩码步骤开始，学习到的反省模型帮助识别输出序列中的错误并重新掩码。这种交替（去掩码→反省→重新掩码）过程会递归重复，直到获得可靠的结果。在多个基准上的实验结果表明，提出的RIV达到了最先进的性能，优于大多数现有的MDVLMs。 

---
# Disentanglement of Variations with Multimodal Generative Modeling 

**Title (ZH)**: 多模态生成建模中变异因素的解耦 

**Authors**: Yijie Zhang, Yiyang Shen, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23548)  

**Abstract**: Multimodal data are prevalent across various domains, and learning robust representations of such data is paramount to enhancing generation quality and downstream task performance. To handle heterogeneity and interconnections among different modalities, recent multimodal generative models extract shared and private (modality-specific) information with two separate variables. Despite attempts to enforce disentanglement between these two variables, these methods struggle with challenging datasets where the likelihood model is insufficient. In this paper, we propose Information-disentangled Multimodal VAE (IDMVAE) to explicitly address this issue, with rigorous mutual information-based regularizations, including cross-view mutual information maximization for extracting shared variables, and a cycle-consistency style loss for redundancy removal using generative augmentations. We further introduce diffusion models to improve the capacity of latent priors. These newly proposed components are complementary to each other. Compared to existing approaches, IDMVAE shows a clean separation between shared and private information, demonstrating superior generation quality and semantic coherence on challenging datasets. 

**Abstract (ZH)**: 多模态数据在多个领域普遍存在，学习such数据的稳健表示对于提高生成质量和下游任务性能至关重要。为了处理不同模态之间的异质性和相互关联性，最近提出的多模态生成模型通过两个独立的变量来提取共享和特定于模态的私有信息。尽管试图在这些变量之间引入解耦，但在复杂数据集上，当似然模型不足时，这些方法仍难以应对。在本文中，我们提出了一种信息解耦多模态VAE（IDMVAE）来明确解决这一问题，通过严格的互信息正则化，包括跨视图互信息最大化以提取共享变量，以及通过生成增强实现冗余去除的循环一致性损失。此外，我们引入了扩散模型以提高潜在先验的能力。这些新提出的组件彼此互补。与现有方法相比，IDMVAE在复杂数据集上显示出了共享和私有信息的清晰分离，展示了更优秀的生成质量和语义一致性。 

---
# S$^3$F-Net: A Multi-Modal Approach to Medical Image Classification via Spatial-Spectral Summarizer Fusion Network 

**Title (ZH)**: S$^3$F-Net：一种基于空间-光谱摘要融合网络的多模态医学图像分类方法 

**Authors**: Md. Saiful Bari Siddiqui, Mohammed Imamul Hassan Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23442)  

**Abstract**: Convolutional Neural Networks have become a cornerstone of medical image analysis due to their proficiency in learning hierarchical spatial features. However, this focus on a single domain is inefficient at capturing global, holistic patterns and fails to explicitly model an image's frequency-domain characteristics. To address these challenges, we propose the Spatial-Spectral Summarizer Fusion Network (S$^3$F-Net), a dual-branch framework that learns from both spatial and spectral representations simultaneously. The S$^3$F-Net performs a fusion of a deep spatial CNN with our proposed shallow spectral encoder, SpectraNet. SpectraNet features the proposed SpectralFilter layer, which leverages the Convolution Theorem by applying a bank of learnable filters directly to an image's full Fourier spectrum via a computation-efficient element-wise multiplication. This allows the SpectralFilter layer to attain a global receptive field instantaneously, with its output being distilled by a lightweight summarizer network. We evaluate S$^3$F-Net across four medical imaging datasets spanning different modalities to validate its efficacy and generalizability. Our framework consistently and significantly outperforms its strong spatial-only baseline in all cases, with accuracy improvements of up to 5.13%. With a powerful Bilinear Fusion, S$^3$F-Net achieves a SOTA competitive accuracy of 98.76% on the BRISC2025 dataset. Concatenation Fusion performs better on the texture-dominant Chest X-Ray Pneumonia dataset, achieving 93.11% accuracy, surpassing many top-performing, much deeper models. Our explainability analysis also reveals that the S$^3$F-Net learns to dynamically adjust its reliance on each branch based on the input pathology. These results verify that our dual-domain approach is a powerful and generalizable paradigm for medical image analysis. 

**Abstract (ZH)**: 卷积神经网络已成为医学图像分析的基石，得益于其在学习层次空间特征方面的 proficiency。然而，这种单一领域的专注于低效地捕捉全局整体模式，并且未能明确建模图像的频域特征。为了解决这些挑战，我们提出了一种双分支架构——空间-频谱总结融合网络（S$^3$F-Net），该架构能够同时从空间和频谱表示中学习。S$^3$F-Net将一个深度空间CNN与我们提出的浅层频谱编码器SpectraNet融合。SpectraNet包含我们提议的SpectralFilter层，该层通过高效元素级乘法直接应用一组可学习的滤波器到图像的全傅里叶谱上，利用卷积定理。SpectralFilter层可以即时获得全局感受野，并通过一个轻量级的总结网络对其输出进行提炼。我们在四个涵盖不同模态的医学成像数据集上评估S$^3$F-Net，以验证其有效性和泛化能力。我们的框架在所有情况下都显著优于其强大的仅空间基线，在某些情况下准确率提高了5.13%。借助强大的双线性融合，S$^3$F-Net在BRISC2025数据集上实现了98.76%的竞争力准确率。在纹理占主导地位的胸部X光肺炎数据集上，拼接融合的方法达到了93.11%的准确率，超过了许多更深层次的顶级模型。我们的可解释性分析还表明，S$^3$F-Net能够根据输入病理动态调整其对各个分支的依赖。这些结果证实了我们双域方法在医学图像分析中的强大和泛化能力。 

---
# DentVLM: A Multimodal Vision-Language Model for Comprehensive Dental Diagnosis and Enhanced Clinical Practice 

**Title (ZH)**: DentVLM：一种多模态视觉-语言模型，用于全面的口腔诊断和增强的临床实践 

**Authors**: Zijie Meng, Jin Hao, Xiwei Dai, Yang Feng, Jiaxiang Liu, Bin Feng, Huikai Wu, Xiaotang Gai, Hengchuan Zhu, Tianxiang Hu, Yangyang Wu, Hongxia Xu, Jin Li, Jun Xiao, Xiaoqiang Liu, Joey Tianyi Zhou, Fudong Zhu, Zhihe Zhao, Lunguo Xia, Bing Fang, Jimeng Sun, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23344)  

**Abstract**: Diagnosing and managing oral diseases necessitate advanced visual interpretation across diverse imaging modalities and integrated information synthesis. While current AI models excel at isolated tasks, they often fall short in addressing the complex, multimodal requirements of comprehensive clinical dental practice. Here we introduce DentVLM, a multimodal vision-language model engineered for expert-level oral disease diagnosis. DentVLM was developed using a comprehensive, large-scale, bilingual dataset of 110,447 images and 2.46 million visual question-answering (VQA) pairs. The model is capable of interpreting seven 2D oral imaging modalities across 36 diagnostic tasks, significantly outperforming leading proprietary and open-source models by 19.6% higher accuracy for oral diseases and 27.9% for malocclusions. In a clinical study involving 25 dentists, evaluating 1,946 patients and encompassing 3,105 QA pairs, DentVLM surpassed the diagnostic performance of 13 junior dentists on 21 of 36 tasks and exceeded that of 12 senior dentists on 12 of 36 tasks. When integrated into a collaborative workflow, DentVLM elevated junior dentists' performance to senior levels and reduced diagnostic time for all practitioners by 15-22%. Furthermore, DentVLM exhibited promising performance across three practical utility scenarios, including home-based dental health management, hospital-based intelligent diagnosis and multi-agent collaborative interaction. These findings establish DentVLM as a robust clinical decision support tool, poised to enhance primary dental care, mitigate provider-patient imbalances, and democratize access to specialized medical expertise within the field of dentistry. 

**Abstract (ZH)**: 口腔疾病诊断和管理需要在多种成像模态下进行高级视觉解释并综合集成信息。当前的AI模型在单个任务上表现出色，但往往难以应对全面临床牙科实践中的复杂、多模态需求。我们介绍了DentVLM，这是一种针对专家级口腔疾病诊断设计的多模态视觉-语言模型。DentVLM 使用一个包含 110,447 幅图像和 246 万个视觉问答 (VQA) 对的全面大规模双语数据集进行开发。该模型能够解析 2D 口腔影像的 7 种模态并完成 36 项诊断任务，其在口腔疾病和错颌畸形的诊断准确性方面分别比领先的专业和开源模型高出 19.6% 和 27.9%。在涉及 25 名牙科医生的临床研究中，评估了 1,946 位患者和 3,105 个问答对，DentVLM 在 36 项中的 21 项诊断任务上超过了 13 名初级牙医的表现，在 12 项诊断任务上则超过了 12 名高级牙医的表现。当整合到协作工作流程中时，DentVLM 提升了初级牙医的表现至高级水平，并降低了所有牙医的诊断时间 15% 至 22%。此外，DentVLM 在家庭口腔健康管理、医院智能诊断和多智能体协作交互三种实际应用场景中表现出色。这些发现确立了DentVLM作为稳健的临床决策支持工具的地位，有望提升初级牙科护理质量，缓解提供者与患者之间的失衡，并在牙科领域普及专业化医疗专业知识的获取。 

---
# Seeing Symbols, Missing Cultures: Probing Vision-Language Models' Reasoning on Fire Imagery and Cultural Meaning 

**Title (ZH)**: 看到符号，忽视文化：探究视觉-语言模型在火灾图像和文化意义推理中的局限性 

**Authors**: Haorui Yu, Qiufeng Yi, Yijia Chu, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23311)  

**Abstract**: Vision-Language Models (VLMs) often appear culturally competent but rely on superficial pattern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themed cultural imagery through both classification and explanation analysis. Testing multiple models on Western festivals, non-Western traditions, and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresented cultural events, frequently offering vague labels or dangerously misclassifying emergencies as celebrations. These failures expose the risks of symbolic shortcuts and highlight the need for cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodal systems. 

**Abstract (ZH)**: 视觉-语言模型往往在文化上表现出一定的能力，但实际依赖于表面的模式匹配而非真正的文化理解。我们提出了一种诊断框架，通过分类和解释分析，探究视觉-语言模型在文化主题火元素图像上的推理。对不同文化背景下的西方节日、非西方传统和紧急场景进行测试揭示了系统的偏差：模型能够正确识别重要的西方节日，但在处理欠代表的文化事件时表现困难，经常提供含糊的标签或将紧急情况错误地分类为庆祝活动。这些失败揭示了符号捷径的风险，并强调了在确保可解释性和公平性的同时，需要超越准确率指标的文化评估。 

---
# Learning Regional Monsoon Patterns with a Multimodal Attention U-Net 

**Title (ZH)**: 使用多模态注意力U-Net学习区域季风模式 

**Authors**: Swaib Ilias Mazumder, Manish Kumar, Aparajita Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23267)  

**Abstract**: Accurate monsoon rainfall prediction is vital for India's agriculture, water management, and climate risk planning, yet remains challenging due to sparse ground observations and complex regional variability. We present a multimodal deep learning framework for high-resolution precipitation classification that leverages satellite and Earth observation data. Unlike previous rainfall prediction models based on coarse 5-50 km grids, we curate a new 1 km resolution dataset for five Indian states, integrating seven key geospatial modalities: land surface temperature, vegetation (NDVI), soil moisture, relative humidity, wind speed, elevation, and land use, covering the June-September 2024 monsoon season. Our approach uses an attention-guided U-Net architecture to capture spatial patterns and temporal dependencies across modalities, combined with focal and dice loss functions to handle rainfall class imbalance defined by the India Meteorological Department (IMD). Experiments demonstrate that our multimodal framework consistently outperforms unimodal baselines and existing deep learning methods, especially in extreme rainfall categories. This work contributes a scalable framework, benchmark dataset, and state-of-the-art results for regional monsoon forecasting, climate resilience, and geospatial AI applications in India. 

**Abstract (ZH)**: 准确的季风雨量预测对于印度的农业、水资源管理和气候风险规划至关重要，但由于地面观测稀疏和区域复杂性，仍具有挑战性。我们提出了一种多模态深度学习框架，用于高分辨率降水分类，该框架利用了卫星和地球observation数据。与基于5-50 km粗网格的以往降雨预测模型不同，我们为五个印度邦编制了一个1 km分辨率的新数据集，整合了七个关键的地理空间模态：地表温度、植被（NDVI）、土壤湿度、相对湿度、风速、海拔和土地用途，涵盖了2024年6月至9月的季风雨季。我们的方法使用注意力引导的U-Net架构来捕捉模态间的空间模式和时间依赖性，并采用焦点损失和Dice损失函数来处理印度气象部门（IMD）定义的降雨类别不平衡。实验结果表明，我们的多模态框架在极端降雨类别中始终优于单模态基线和现有深度学习方法。这项工作为印度地区的季风雨量预报、气候韧性和地理空间AI应用提供了可扩展的框架、基准数据集和最先进的结果。 

---
# Self-Consistency as a Free Lunch: Reducing Hallucinations in Vision-Language Models via Self-Reflection 

**Title (ZH)**: 自我一致性作为免费午餐：通过自我反思减少视觉语言模型的幻觉 

**Authors**: Mingfei Han, Haihong Hao, Jinxing Zhou, Zhihui Li, Yuhui Zheng, Xueqing Deng, Linjie Yang, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23236)  

**Abstract**: Vision-language models often hallucinate details, generating non-existent objects or inaccurate attributes that compromise output reliability. Existing methods typically address these issues via extensive human annotations or external supervision from more powerful models. In this work, we present a novel framework that leverages the model's self-consistency between long responses and short answers to generate preference pairs for training. We observe that short binary questions tend to yield highly reliable responses, which can be used to query the target model to evaluate and rank its generated responses. Specifically, we design a self-reflection pipeline where detailed model responses are compared against concise binary answers, and inconsistency signals are utilized to automatically curate high-quality training data without human annotations or external model-based supervision. By relying solely on self-consistency rather than external supervision, our method offers a scalable and efficient solution that effectively reduces hallucinations using unlabeled data. Extensive experiments on multiple benchmarks, i.e., AMBER, MultiObject-Hal (ROPE), Object HalBench, and MMHal-Bench, demonstrate significant improvements in factual grounding and reliability. Moreover, our approach maintains robust instruction-following ability, as evidenced by enhanced performance on LLaVA-Bench and MMBench. 

**Abstract (ZH)**: Vision-language模型常常出现幻觉现象，生成不存在的物体或不准确的属性，影响输出的可靠性。现有方法通常通过大量的人工注释或更强模型的外部监督来解决这些问题。在此工作中，我们提出了一种新颖的框架，利用模型在长响应和短答案之间的一致性来生成训练偏好对。我们观察到，简短的二元问题往往会生成高度可靠的响应，这些响应可以用来查询目标模型以评估和排序其生成的响应。具体而言，我们设计了一条自我反思管道，其中详细的模型响应与简短的二元答案进行比较，不一致性信号被用来自动收集高质量的训练数据，而无需人工注释或基于模型的外部监督。通过仅仅依赖自我一致性而非外部监督，我们的方法提供了一种可扩展且高效的解决方案，能够有效利用未标注数据减少幻觉现象。在AMBER、MultiObject-Hal (ROPE)、Object HalBench和MMHal-Bench等多个基准上的 extensive 实验表明，在事实依据和可靠性方面取得了显著改进。此外，我们的方法在LLaVA-Bench和MMBench上的增强性能验证了其稳健的指令跟随能力。 

---
# Liaohe-CobotMagic-PnP: an Imitation Learning Dataset of Intelligent Robot for Industrial Applications 

**Title (ZH)**: 辽河-CobotMagic-PnP：面向工业应用的智能机器人模仿学习数据集 

**Authors**: Chen Yizhe, Wang Qi, Hu Dongxiao, Jingzhe Fang, Liu Sichao, Zixin An, Hongliang Niu, Haoran Liu, Li Dong, Chuanfen Feng, Lan Dapeng, Liu Yu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23111)  

**Abstract**: In Industry 4.0 applications, dynamic environmental interference induces highly nonlinear and strongly coupled interactions between the environmental state and robotic behavior. Effectively representing dynamic environmental states through multimodal sensor data fusion remains a critical challenge in current robotic datasets. To address this, an industrial-grade multimodal interference dataset is presented, designed for robotic perception and control under complex conditions. The dataset integrates multi-dimensional interference features including size, color, and lighting variations, and employs high-precision sensors to synchronously collect visual, torque, and joint-state measurements. Scenarios with geometric similarity exceeding 85\% and standardized lighting gradients are included to ensure real-world representativeness. Microsecond-level time-synchronization and vibration-resistant data acquisition protocols, implemented via the Robot Operating System (ROS), guarantee temporal and operational fidelity. Experimental results demonstrate that the dataset enhances model validation robustness and improves robotic operational stability in dynamic, interference-rich environments. The dataset is publicly available at:this https URL. 

**Abstract (ZH)**: 在工业4.0应用中，动态环境干扰导致环境状态与机器人行为之间呈现高度非线性的强耦合交互。通过多模态传感器数据融合有效地表示动态环境状态仍然是当前机器人数据集中的一项关键挑战。为此，提出了一种工业级别的多模态干扰数据集，旨在在复杂条件下增强机器人的感知与控制能力。该数据集整合了尺寸、颜色和光照等多维干扰特征，并采用高精度传感器同步采集视觉、扭矩和关节状态数据。包含几何相似度超过85%且标准化光照梯度的场景，以确保真实场景的代表性和适用性。通过机器人操作系统（ROS）实现微秒级时间同步和抗振动数据采集协议，保证了时间和操作上的准确性。实验结果表明，该数据集增强了模型验证的稳健性，并提高了机器人在动态、干扰丰富的环境中的操作稳定性。该数据集已公开发布于此：this https URL。 

---
# MMeViT: Multi-Modal ensemble ViT for Post-Stroke Rehabilitation Action Recognition 

**Title (ZH)**: MMeViT: 多模态集成ViT在中风后康复动作识别中的应用 

**Authors**: Ye-eun Kim, Suhyeon Lim, Andrew J. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23044)  

**Abstract**: Rehabilitation therapy for stroke patients faces a supply shortage despite the increasing demand. To address this issue, remote monitoring systems that reduce the burden on medical staff are emerging as a viable alternative. A key component of these remote monitoring systems is Human Action Recognition (HAR) technology, which classifies actions. However, existing HAR studies have primarily focused on non-disable individuals, making them unsuitable for recognizing the actions of stroke patients. HAR research for stroke has largely concentrated on classifying relatively simple actions using machine learning rather than deep learning. In this study, we designed a system to monitor the actions of stroke patients, focusing on domiciliary upper limb Activities of Daily Living (ADL). Our system utilizes IMU (Inertial Measurement Unit) sensors and an RGB-D camera, which are the most common modalities in HAR. We directly collected a dataset through this system, investigated an appropriate preprocess and proposed a deep learning model suitable for processing multimodal data. We analyzed the collected dataset and found that the action data of stroke patients is less clustering than that of non-disabled individuals. Simultaneously, we found that the proposed model learns similar tendencies for each label in data with features that are difficult to clustering. This study suggests the possibility of expanding the deep learning model, which has learned the action features of stroke patients, to not only simple action recognition but also feedback such as assessment contributing to domiciliary rehabilitation in future research. The code presented in this study is available at this https URL. 

**Abstract (ZH)**: 尽管对中风患者的康复治疗需求不断增加，但康复治疗供应却面临短缺。为解决这一问题，减少医护人员负担的远程监控系统正逐渐成为可行的替代方案。这些远程监控系统的关键组成部分是人体动作识别（HAR）技术，该技术用于分类动作。然而，现有的HAR研究主要集中在非残疾个体上，使得它们不适用于识别中风患者的动作。对于中风患者的HAR研究大多侧重于使用机器学习而非深度学习来分类相对简单的动作。本研究设计了一个系统来监测中风患者的动作，重点是居家上肢日常生活活动（ADL）。该系统利用了惯性测量单元（IMU）传感器和RGB-D相机，这是HAR中常用的模态。我们直接通过该系统收集了数据集，研究了合适的预处理方法，并提出了适用于处理多模态数据的深度学习模型。我们对收集的数据集进行了分析，发现中风患者的动作数据比非残疾个体更分散。同时，我们发现提出的模型在难以聚类的数据特征中，学习了每种标签的相似趋势。本研究表明，已学习中风患者动作特征的深度学习模型不仅可用于简单的动作识别，还可为未来的研究中提供居家康复评估等方面的反馈。本研究中提供的代码可在以下网址获取。 

---
# Sensor-Adaptive Flood Mapping with Pre-trained Multi-Modal Transformers across SAR and Multispectral Modalities 

**Title (ZH)**: 基于预训练多模态变换器的SAR和多光谱模态自适应洪水 mapping 

**Authors**: Tomohiro Tanaka, Narumasa Tsutsumida  

**Link**: [PDF](https://arxiv.org/pdf/2509.23035)  

**Abstract**: Floods are increasingly frequent natural disasters causing extensive human and economic damage, highlighting the critical need for rapid and accurate flood inundation mapping. While remote sensing technologies have advanced flood monitoring capabilities, operational challenges persist: single-sensor approaches face weather-dependent data availability and limited revisit periods, while multi-sensor fusion methods require substantial computational resources and large-scale labeled datasets. To address these limitations, this study introduces a novel sensor-flexible flood detection methodology by fine-tuning Presto, a lightweight ($\sim$0.4M parameters) multi-modal pre-trained transformer that processes both Synthetic Aperture Radar (SAR) and multispectral (MS) data at the pixel level. Our approach uniquely enables flood mapping using SAR-only, MS-only, or combined SAR+MS inputs through a single model architecture, addressing the critical operational need for rapid response with whatever sensor data becomes available first during disasters. We evaluated our method on the Sen1Floods11 dataset against the large-scale Prithvi-100M baseline ($\sim$100M parameters) across three realistic data availability scenarios. The proposed model achieved superior performance with an F1 score of 0.896 and mIoU of 0.886 in the optimal sensor-fusion scenario, outperforming the established baseline. Crucially, the model demonstrated robustness by maintaining effective performance in MS-only scenarios (F1: 0.893) and functional capabilities in challenging SAR-only conditions (F1: 0.718), confirming the advantage of multi-modal pre-training for operational flood mapping. Our parameter-efficient, sensor-flexible approach offers an accessible and robust solution for real-world disaster scenarios requiring immediate flood extent assessment regardless of sensor availability constraints. 

**Abstract (ZH)**: 洪水频率增加，导致广泛的人类和经济损害，突显了快速准确绘制洪水淹没图的迫切需求。虽然遥感技术提升了洪水监测能力，但仍存在运营挑战：单传感器方法面临依赖天气的数据可用性和有限的重访周期，而多传感器融合方法则需要大量的计算资源和大规模标注数据集。为应对这些局限性，本研究引入了一种新颖的传感器灵活洪水检测方法，通过微调 Presto（一个轻量级，约0.4M参数的多模态预训练变换器），该模型在像素级别处理合成孔径雷达（SAR）和多光谱（MS）数据。我们的方法能够通过单一模型架构使用SAR仅有的、MS仅有的或联合SAR+MS输入进行洪水制图，满足灾害期间 whichever传感器数据首先可用时迅速响应的迫切需求。我们在Sen1Floods11数据集上将本方法与大规模Prithvi-100M基线（约100M参数）进行了评估，在三种实际的数据可用性场景中。所提出模型在最佳传感器融合场景下的F1分数为0.896，mIoU为0.886，优于现有的基线。值得注意的是，该模型展示了稳健性，在MS仅有的场景中保持了有效的性能（F1: 0.893），并在具有挑战性的SAR仅有的条件下也能发挥作用（F1: 0.718），证明了多模态预训练在运营洪水绘图方面的优势。我们的参数高效、传感器灵活的方法为不受传感器可用性限制的实际灾害场景提供了便捷而稳健的解决方案，用于立即评估洪水范围。 

---
# MMPB: It's Time for Multi-Modal Personalization 

**Title (ZH)**: MMPB: 是时候实现多模态个性化了 

**Authors**: Jaeik Kim, Woojin Kim, Woohyeon Park, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.22820)  

**Abstract**: Visual personalization is essential in user-facing AI systems such as smart homes and healthcare, where aligning model behavior with user-centric concepts is critical. However, recent large Vision-Language Models (VLMs), despite their broad applicability, remain underexplored in their ability to adapt to individual users. In this paper, we introduce MMPB, the first extensive benchmark for evaluating VLMs on personalization. MMPB comprises 10k image-query pairs and includes 111 personalizable concepts across four categories: humans, animals, objects, and characters, with the human category enriched with preference-grounded queries. We structure personalization into three main task types, each highlighting a different key property of VLMs. Using 23 widely used VLMs including both open- and closed-source models, we evaluate personalization performance via a three-stage protocol: concept injection, multi-turn dialogue, and personalized querying. Our findings indicate that most VLMs (including some closed-source models) struggle with personalization, particularly in maintaining consistency over dialogue, handling user preferences, and adapting to visual cues. Our analysis reveals that the challenges in VLM personalization (such as refusal behaviors and long-context forgetting) highlight substantial room for improvement. By identifying these limitations and offering a scalable benchmark, MMPB offers valuable insights and a solid foundation for future research toward truly personalized multi-modal AI. Project Page: this http URL 

**Abstract (ZH)**: 视觉个性化对于智能家庭和医疗等用户-facing AI 系统至关重要，其中模型行为与用户中心的概念对齐是关键。然而，尽管近期大规模视觉-语言模型(VLMs)具有广泛的应用前景，但在适应个体用户方面仍被严重忽视。本文介绍了MMPB，这是首个用于评估VLMs个性化能力的全面基准。MMPB包含10,000幅图像查询对，并涵盖了跨四大类别（人类、动物、物体和角色）的111个可个性化概念，其中人类类别还包含了基于偏好的查询。我们将个性化任务结构化为三大主要任务类型，每种任务类型均强调VLMs的不同关键属性。我们使用23个广泛使用的VLMs，包括开源和闭源模型，通过三阶段协议评估个性化性能：概念注入、多轮对话和个性化查询。我们的研究结果表明，大多数VLMs（包括一些闭源模型）在个性化方面面临挑战，特别是在对话一致性、处理用户偏好和适应视觉线索方面。我们的分析表明，VLMs个性化面临的挑战（如拒绝行为和长上下文遗忘）凸显了显著的改进空间。通过识别这些限制并提供一个可扩展的基准，MMPB为未来真正个性化多模态AI的研究提供了宝贵的见解和坚实的基础。项目页面：this http URL 

---
# MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning 

**Title (ZH)**: MILR：通过测试时潜在推理提高多模态图像生成 

**Authors**: Yapeng Mi, Hengli Li, Yanpeng Zhao, Chenxi Li, Huimin Wu, Xiaojian Ma, Song-Chun Zhu, Ying Nian Wu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22761)  

**Abstract**: Reasoning-augmented machine learning systems have shown improved performance in various domains, including image generation. However, existing reasoning-based methods for image generation either restrict reasoning to a single modality (image or text) or rely on high-quality reasoning data for fine-tuning. To tackle these limitations, we propose MILR, a test-time method that jointly reasons over image and text in a unified latent vector space. Reasoning in MILR is performed by searching through vector representations of discrete image and text tokens. Practically, this is implemented via the policy gradient method, guided by an image quality critic. We instantiate MILR within the unified multimodal understanding and generation (MUG) framework that natively supports language reasoning before image synthesis and thus facilitates cross-modal reasoning. The intermediate model outputs, which are to be optimized, serve as the unified latent space, enabling MILR to operate entirely at test time. We evaluate MILR on GenEval, T2I-CompBench, and WISE, achieving state-of-the-art results on all benchmarks. Notably, on knowledge-intensive WISE, MILR attains an overall score of 0.63, improving over the baseline by 80%. Our further analysis indicates that joint reasoning in the unified latent space is the key to its strong performance. Moreover, our qualitative studies reveal MILR's non-trivial ability in temporal and cultural reasoning, highlighting the efficacy of our reasoning method. 

**Abstract (ZH)**: 增强推理的机器学习系统在各个领域显示出了改进的性能，包括图像生成。然而，现有的基于推理的图像生成方法要么将推理限制在单一模态（图像或文本）上，要么依赖高质量的推理数据进行微调。为了解决这些问题，我们提出了一种名为MILR的测试时方法，它在统一的潜在向量空间中同时对图像和文本进行推理。在MILR中，通过搜索离散图像和文本标记的向量表示来进行推理。实际中，这一过程通过策略梯度方法实现，并由图像质量批评家引导。我们将在原生支持语言推理后再进行图像合成的统一多模态理解和生成（MUG）框架中实现MILR，从而促进跨模态推理。作为要优化的中间模型输出充当统一的潜在空间，使MILR完全在测试时运行。我们在GenEval、T2I-CompBench和WISE上评估了MILR，实现了所有基准上的最佳结果。特别是在知识密集型的WISE上，MILR获得了0.63的整体得分，比基线高出80%。进一步的分析表明，统一潜在空间中的联合推理是其高性能的关键。此外，我们的定性研究还揭示了MILR在时间和文化推理方面的非平凡能力，突显了我们推理方法的有效性。 

---
# Index-MSR: A high-efficiency multimodal fusion framework for speech recognition 

**Title (ZH)**: 基于索引的多模态融合框架：高效率语音识别 

**Authors**: Jinming Chen, Lu Wang, Zheshu Song, Wei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22744)  

**Abstract**: Driven by large scale datasets and LLM based architectures, automatic speech recognition (ASR) systems have achieved remarkable improvements in accuracy. However, challenges persist for domain-specific terminology, and short utterances lacking semantic coherence, where recognition performance often degrades significantly. In this work, we present Index-MSR, an efficient multimodal speech recognition framework. At its core is a novel Multimodal Fusion Decoder (MFD), which effectively incorporates text-related information from videos (e.g., subtitles and presentation slides) into the speech recognition. This cross-modal integration not only enhances overall ASR accuracy but also yields substantial reductions in substitution errors. Extensive evaluations on both an in-house subtitle dataset and a public AVSR dataset demonstrate that Index-MSR achieves sota accuracy, with substitution errors reduced by 20,50%. These results demonstrate that our approach efficiently exploits text-related cues from video to improve speech recognition accuracy, showing strong potential in applications requiring strict audio text synchronization, such as audio translation. 

**Abstract (ZH)**: 基于大规模数据集和基于LLM的架构，自动语音识别（ASR）系统在准确率上取得了显著改进。然而，特定领域术语和缺乏语义连贯性的短语音片段仍存在挑战，这些情况下识别性能往往会显著下降。本文介绍了Index-MSR，一种高效的多模态语音识别框架。其核心是一款新颖的多模态融合解码器（MFD），能够有效将视频中的文本相关信息（如字幕和演示幻灯片）融入到语音识别中。这种跨模态集成不仅提升了整体ASR准确率，还大幅减少了替换错误。在内部字幕数据集和公共AVSR数据集上的 extensive 评估表明，Index-MSR 达到了最先进的准确率，替换错误减少了20.5%。这些结果表明，我们的方法能够高效利用视频中的文本线索来提高语音识别准确率，特别是在需要严格音频文本同步的应用场景中，如语音翻译方面展现出强大的潜力。 

---
# CompareBench: A Benchmark for Visual Comparison Reasoning in Vision-Language Models 

**Title (ZH)**: CompareBench: 视觉对比推理基准在视觉-语言模型中的应用 

**Authors**: Jie Cai, Kangning Yang, Lan Fu, Jiaming Ding, Jinlong Li, Huiming Sun, Daitao Xing, Jinglin Shen, Zibo Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22737)  

**Abstract**: We introduce CompareBench, a benchmark for evaluating visual comparison reasoning in vision-language models (VLMs), a fundamental yet understudied skill. CompareBench consists of 1000 QA pairs across four tasks: quantity (600), temporal (100), geometric (200), and spatial (100). It is derived from two auxiliary datasets that we constructed: TallyBench (2000 counting images with QA) and HistCaps (515 historical images with bilingual captions). We evaluate both closed-source APIs (OpenAI, Gemini, Claude) and open-source models (Qwen2.5-VL and Qwen3-VL series). Results show clear scaling trends but also reveal critical limitations: even the strongest models consistently fail at temporal ordering and spatial relations, and they often make mistakes in basic counting and geometric comparisons that are trivial for humans. These findings demonstrate that visual comparison remains a systematic blind spot for current VLMs. By providing controlled, diverse, and diagnostic evaluation, CompareBench establishes a foundation for advancing more reliable multimodal reasoning. 

**Abstract (ZH)**: CompareBench: 一个评估视觉语言模型中视觉比较推理能力的标准数据集 

---
# Multi-Modal Sentiment Analysis with Dynamic Attention Fusion 

**Title (ZH)**: 多模态情感分析中的动态注意融合 

**Authors**: Sadia Abdulhalim, Muaz Albaghdadi, Moshiur Farazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22729)  

**Abstract**: Traditional sentiment analysis has long been a unimodal task, relying solely on text. This approach overlooks non-verbal cues such as vocal tone and prosody that are essential for capturing true emotional intent. We introduce Dynamic Attention Fusion (DAF), a lightweight framework that combines frozen text embeddings from a pretrained language model with acoustic features from a speech encoder, using an adaptive attention mechanism to weight each modality per utterance. Without any finetuning of the underlying encoders, our proposed DAF model consistently outperforms both static fusion and unimodal baselines on a large multimodal benchmark. We report notable gains in F1-score and reductions in prediction error and perform a variety of ablation studies that support our hypothesis that the dynamic weighting strategy is crucial for modeling emotionally complex inputs. By effectively integrating verbal and non-verbal information, our approach offers a more robust foundation for sentiment prediction and carries broader impact for affective computing applications -- from emotion recognition and mental health assessment to more natural human computer interaction. 

**Abstract (ZH)**: 传统情感分析长期以来一直是一个单模态任务，仅依赖文本。这种做法忽视了如音调和语调等非言语线索，这些线索对于捕捉真实的情感意图至关重要。我们 introduce 动态注意力融合（Dynamic Attention Fusion，DAF），这是一个轻量级框架，结合了预训练语言模型的冻结文本嵌入和语音编码器的声学特征，使用自适应注意力机制为每个utterance加权不同模态。不微调底层编码器的情况下，我们提出的DAF模型在大型多模态基准测试中持续优于静态融合和单模态基线。我们报告了F1分数的显著提高和预测误差的减少，并进行了多种消融研究，支持我们假设动态加权策略对于建模情感复杂输入至关重要。通过有效整合言语和非言语信息，我们的方法为情感预测提供了更稳健的基础，并对情感计算应用具有更广泛的影响——从情绪识别和心理健康评估到更自然的人机交互。 

---
# LayoutAgent: A Vision-Language Agent Guided Compositional Diffusion for Spatial Layout Planning 

**Title (ZH)**: 布局智能体：面向空间布局规划的视觉-语言引导组成性扩散方法 

**Authors**: Zezhong Fan, Xiaohan Li, Luyi Ma, Kai Zhao, Liang Peng, Topojoy Biswas, Evren Korpeoglu, Kaushiki Nag, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22720)  

**Abstract**: Designing realistic multi-object scenes requires not only generating images, but also planning spatial layouts that respect semantic relations and physical plausibility. On one hand, while recent advances in diffusion models have enabled high-quality image generation, they lack explicit spatial reasoning, leading to unrealistic object layouts. On the other hand, traditional spatial planning methods in robotics emphasize geometric and relational consistency, but they struggle to capture semantic richness in visual scenes. To bridge this gap, in this paper, we propose LayoutAgent, an agentic framework that unifies vision-language reasoning with compositional diffusion for layout generation. Given multiple input images with target objects in them, our method first employs visual-language model to preprocess the inputs through segmentation, object size estimation, scene graph construction, and prompt rewriting. Then we leverage compositional diffusion-a method traditionally used in robotics-to synthesize bounding boxes that respect object relations encoded in the scene graph for spatial layouts. In the end, a foreground-conditioned image generator composes the complete scene by rendering the objects into the planned layout guided by designed prompts. Experiments demonstrate that LayoutAgent outperforms other state-of-the-art layout generation models in layout coherence, spatial realism and aesthetic alignment. 

**Abstract (ZH)**: 设计现实的多对象场景不仅需要生成图像，还需要规划遵守语义关系和物理合理性的空间布局。为弥合这一差距，本文提出了一种名为LayoutAgent的代理框架，该框架将视觉语言推理与组合扩散相结合，用于布局生成。 

---
