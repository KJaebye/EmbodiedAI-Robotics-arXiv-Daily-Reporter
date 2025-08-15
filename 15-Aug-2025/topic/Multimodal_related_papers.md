# A Multimodal Neural Network for Recognizing Subjective Self-Disclosure Towards Social Robots 

**Title (ZH)**: 面向社会机器人的主观自我披露识别的多模态神经网络 

**Authors**: Henry Powell, Guy Laban, Emily S. Cross  

**Link**: [PDF](https://arxiv.org/pdf/2508.10828)  

**Abstract**: Subjective self-disclosure is an important feature of human social interaction. While much has been done in the social and behavioural literature to characterise the features and consequences of subjective self-disclosure, little work has been done thus far to develop computational systems that are able to accurately model it. Even less work has been done that attempts to model specifically how human interactants self-disclose with robotic partners. It is becoming more pressing as we require social robots to work in conjunction with and establish relationships with humans in various social settings. In this paper, our aim is to develop a custom multimodal attention network based on models from the emotion recognition literature, training this model on a large self-collected self-disclosure video corpus, and constructing a new loss function, the scale preserving cross entropy loss, that improves upon both classification and regression versions of this problem. Our results show that the best performing model, trained with our novel loss function, achieves an F1 score of 0.83, an improvement of 0.48 from the best baseline model. This result makes significant headway in the aim of allowing social robots to pick up on an interaction partner's self-disclosures, an ability that will be essential in social robots with social cognition. 

**Abstract (ZH)**: 主观自我披露是人类社会互动的重要特征。尽管在社会行为学文献中已经做了大量关于主观自我披露的特征和后果的研究，但目前开发能够准确建模其行为的计算系统的工作尚显不足。在试图建模人类如何与机器人伙伴进行自我披露方面的工作更是少之又少。随着我们要求社会机器人能够在各种社会设置中与人类协同工作，并建立关系，这种情况变得愈发紧迫。本文旨在基于情绪识别模型开发一个定制的多模态注意力网络，通过一个大型自收集自我披露视频语料库进行训练，并构建一种新的损失函数——尺度保持交叉熵损失，以改进分类和回归版本的问题。我们的结果表明，使用我们新提出的损失函数训练的最佳模型，实现了0.83的F1分数，相较于最佳基线模型提高了0.48。这一结果在让社会机器人能够察觉互动伙伴的自我披露方面取得了显著进展，而这种能力对具有社会认知的社会机器人来说至关重要。 

---
# MM-Food-100K: A 100,000-Sample Multimodal Food Intelligence Dataset with Verifiable Provenance 

**Title (ZH)**: MM-Food-100K: 一个带有可验证来源的100,000样本多模态食品智能数据集 

**Authors**: Yi Dong, Yusuke Muraoka, Scott Shi, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10429)  

**Abstract**: We present MM-Food-100K, a public 100,000-sample multimodal food intelligence dataset with verifiable provenance. It is a curated approximately 10% open subset of an original 1.2 million, quality-accepted corpus of food images annotated for a wide range of information (such as dish name, region of creation). The corpus was collected over six weeks from over 87,000 contributors using the Codatta contribution model, which combines community sourcing with configurable AI-assisted quality checks; each submission is linked to a wallet address in a secure off-chain ledger for traceability, with a full on-chain protocol on the roadmap. We describe the schema, pipeline, and QA, and validate utility by fine-tuning large vision-language models (ChatGPT 5, ChatGPT OSS, Qwen-Max) on image-based nutrition prediction. Fine-tuning yields consistent gains over out-of-box baselines across standard metrics; we report results primarily on the MM-Food-100K subset. We release MM-Food-100K for publicly free access and retain approximately 90% for potential commercial access with revenue sharing to contributors. 

**Abstract (ZH)**: 我们呈现MM-Food-100K：一个具有可验证来源的公共10万样本多模态食品智能数据集 

---
# A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering 

**Title (ZH)**: 基于课程学习的强化学习方法：利用RAG进行多模态问答 

**Authors**: Chenliang Zhang, Lin Wang, Yuanyuan Lu, Yusheng Qi, Kexin Wang, Peixu Hou, Wenshi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10337)  

**Abstract**: This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38\%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline. 

**Abstract (ZH)**: 本文描述了Dianping-Trust-Safety团队在META CRAG-MM挑战中提出的解决方案。该挑战要求构建一个综合的检索增强生成系统，具备多模态多轮问答能力。比赛包括三个任务：（1）利用基于图像的模拟知识图谱检索结构化数据进行问题回答，（2）从知识图谱和网络搜索结果中合成信息，和（3）处理需要上下文理解并从多个来源聚合信息的多轮对话。在任务1中，我们的解决方案基于视觉大型语言模型，并通过从GPT-4.1知识中监督微调来增强，同时应用了课程学习策略来引导强化学习，从而提高了答案的准确性并减少了幻觉。在任务2和任务3中，我们还利用了网络搜索API来引入外部知识，使系统能够更好地处理复杂的查询和多轮对话。我们的方法在任务1中取得了第一名，领先优势为52.38%，并在任务3中获得第三名，证明了在我们的训练管道中将课程学习与强化学习结合的有效性。 

---
# AddressVLM: Cross-view Alignment Tuning for Image Address Localization using Large Vision-Language Models 

**Title (ZH)**: AddressVLM：使用大型视觉-语言模型进行图像地址本地化的跨视图对齐调整 

**Authors**: Shixiong Xu, Chenghao Zhang, Lubin Fan, Yuan Zhou, Bin Fan, Shiming Xiang, Gaofeng Meng, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10667)  

**Abstract**: Large visual language models (LVLMs) have demonstrated impressive performance in coarse-grained geo-localization at the country or city level, but they struggle with fine-grained street-level localization within urban areas. In this paper, we explore integrating city-wide address localization capabilities into LVLMs, facilitating flexible address-related question answering using street-view images. A key challenge is that the street-view visual question-and-answer (VQA) data provides only microscopic visual cues, leading to subpar performance in fine-tuned models. To tackle this issue, we incorporate perspective-invariant satellite images as macro cues and propose cross-view alignment tuning including a satellite-view and street-view image grafting mechanism, along with an automatic label generation mechanism. Then LVLM's global understanding of street distribution is enhanced through cross-view matching. Our proposed model, named AddressVLM, consists of two-stage training protocols: cross-view alignment tuning and address localization tuning. Furthermore, we have constructed two street-view VQA datasets based on image address localization datasets from Pittsburgh and San Francisco. Qualitative and quantitative evaluations demonstrate that AddressVLM outperforms counterpart LVLMs by over 9% and 12% in average address localization accuracy on these two datasets, respectively. 

**Abstract (ZH)**: 面向街道级别的地址本地化大型视觉语言模型 

---
# Serial Over Parallel: Learning Continual Unification for Multi-Modal Visual Object Tracking and Benchmarking 

**Title (ZH)**: 串行优于并行：学习多模态视觉目标跟踪中的持续统一方法及基准测试 

**Authors**: Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Chunyang Cheng, Tao Zhou, Xiaojun Wu, Josef Kittler  

**Link**: [PDF](https://arxiv.org/pdf/2508.10655)  

**Abstract**: Unifying multiple multi-modal visual object tracking (MMVOT) tasks draws increasing attention due to the complementary nature of different modalities in building robust tracking systems. Existing practices mix all data sensor types in a single training procedure, structuring a parallel paradigm from the data-centric perspective and aiming for a global optimum on the joint distribution of the involved tasks. However, the absence of a unified benchmark where all types of data coexist forces evaluations on separated benchmarks, causing \textit{inconsistency} between training and testing, thus leading to performance \textit{degradation}. To address these issues, this work advances in two aspects: \ding{182} A unified benchmark, coined as UniBench300, is introduced to bridge the inconsistency by incorporating multiple task data, reducing inference passes from three to one and cutting time consumption by 27\%. \ding{183} The unification process is reformulated in a serial format, progressively integrating new tasks. In this way, the performance degradation can be specified as knowledge forgetting of previous tasks, which naturally aligns with the philosophy of continual learning (CL), motivating further exploration of injecting CL into the unification process. Extensive experiments conducted on two baselines and four benchmarks demonstrate the significance of UniBench300 and the superiority of CL in supporting a stable unification process. Moreover, while conducting dedicated analyses, the performance degradation is found to be negatively correlated with network capacity. Additionally, modality discrepancies contribute to varying degradation levels across tasks (RGBT > RGBD > RGBE in MMVOT), offering valuable insights for future multi-modal vision research. Source codes and the proposed benchmark is available at \textit{this https URL}. 

**Abstract (ZH)**: 统一多种多模态视觉对象跟踪任务（MMVOT）随着不同模态互补性在构建稳健跟踪系统中的作用而越来越受到关注。现有的实践将所有数据传感器类型混入单一训练过程，从数据为中心的角度构建并行范式，旨在针对涉及的任务联合分布达到全局最优。然而，缺乏一个所有类型数据共存的统一基准导致在分离的基准上进行评估，从而在训练和测试之间造成不一致，进而导致性能退化。为解决这些问题，本工作在两个方面进行了改进：1）提出一个统一样本集UniBench300，通过整合多种任务数据，将推理次数从三次减少到一次，并将时间消耗缩减27%来缓解不一致性问题。2）统一过程重新表述为顺序格式，逐步整合新的任务。这样，性能退化可以被规定为对先前任务的知识遗忘，这自然与连续学习（CL）的哲学相一致，激励将CL注入统一过程中的进一步探索。在两个基线上和四个基准上的广泛实验展示了UniBench300的重要性以及CL在支持稳定统一过程方面的优势。此外，在进行专门分析时发现，性能退化与网络容量呈负相关，并且模态差异导致不同任务的退化水平不同（在MMVOT中RGBT > RGBD > RGBE），为未来多模态视觉研究提供了宝贵的见解。源代码和提出的基准可在以下链接获取。 

---
# A Unified Multi-Agent Framework for Universal Multimodal Understanding and Generation 

**Title (ZH)**: 统一多Agent框架：面向通用多模态理解与生成 

**Authors**: Jiulin Li, Ping Huang, Yexin Li, Shuo Chen, Juewen Hu, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.10494)  

**Abstract**: Real-world multimodal applications often require any-to-any capabilities, enabling both understanding and generation across modalities including text, image, audio, and video. However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging. Existing approaches rely on rigid pipelines or tightly coupled architectures, limiting flexibility and scalability. We propose MAGUS (Multi-Agent Guided Unified Multimodal System), a modular framework that unifies multimodal understanding and generation via two decoupled phases: Cognition and Deliberation. MAGUS enables symbolic multi-agent collaboration within a shared textual workspace. In the Cognition phase, three role-conditioned multimodal LLM agents - Perceiver, Planner, and Reflector - engage in collaborative dialogue to perform structured understanding and planning. The Deliberation phase incorporates a Growth-Aware Search mechanism that orchestrates LLM-based reasoning and diffusion-based generation in a mutually reinforcing manner. MAGUS supports plug-and-play extensibility, scalable any-to-any modality conversion, and semantic alignment - all without the need for joint training. Experiments across multiple benchmarks, including image, video, and audio generation, as well as cross-modal instruction following, demonstrate that MAGUS outperforms strong baselines and state-of-the-art systems. Notably, on the MME benchmark, MAGUS surpasses the powerful closed-source model GPT-4o. 

**Abstract (ZH)**: 多模态应用通常需要任意到任意的能力，以跨文本、图像、音频和视频等多种模态实现理解和生成。然而，如何结合自回归语言模型（LLMs）的推理能力和扩散模型的高保真生成能力依然具有挑战性。现有方法依赖于刚性的工作流程或紧密耦合的架构，限制了灵活性和可扩展性。我们提出了MAGUS（多代理引导统一多模态系统），这是一种模块化的框架，通过两个分离的阶段——认知和决断，来统一多模态的理解和生成。MAGUS允许在共享的文本工作空间中进行符号化的多代理合作。在认知阶段，三位基于角色的多模态LLM代理——Perceiver、Planner和Reflector——进行协作对话，以执行结构化理解和计划。决断阶段采用感知增长搜索机制，协调基于LLM的推理和基于扩散的生成，在相互强化的过程中进行调控。MAGUS支持插拔式扩展、大规模任意到任意的模态转换以及语义对齐，无需联合训练。在多种基准测试中，包括图像、视频和音频生成，以及跨模态指令跟随，MAGUS均表现出色，超越了强大的基线和最先进的系统。值得注意的是，MAGUS在MME基准测试中超过了强大的闭源模型GPT-4o。 

---
# Empowering Morphing Attack Detection using Interpretable Image-Text Foundation Model 

**Title (ZH)**: 基于可解释的图像-文本基础模型的形态变化攻击检测增强方法 

**Authors**: Sushrut Patwardhan, Raghavendra Ramachandra, Sushma Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10110)  

**Abstract**: Morphing attack detection has become an essential component of face recognition systems for ensuring a reliable verification scenario. In this paper, we present a multimodal learning approach that can provide a textual description of morphing attack detection. We first show that zero-shot evaluation of the proposed framework using Contrastive Language-Image Pretraining (CLIP) can yield not only generalizable morphing attack detection, but also predict the most relevant text snippet. We present an extensive analysis of ten different textual prompts that include both short and long textual prompts. These prompts are engineered by considering the human understandable textual snippet. Extensive experiments were performed on a face morphing dataset that was developed using a publicly available face biometric dataset. We present an evaluation of SOTA pre-trained neural networks together with the proposed framework in the zero-shot evaluation of five different morphing generation techniques that are captured in three different mediums. 

**Abstract (ZH)**: 形态攻击检测已成为确保面部识别系统可靠验证场景的必要组成部分。本文提出了一种多模态学习方法，可为形态攻击检测提供文本描述。我们首先展示使用对比语言-图像预训练（CLIP）进行零样本评估可以不仅实现泛化的形态攻击检测，还能预测最相关的文本片段。我们探讨了包括短文本提示和长文本提示在内的十个不同的文本提示。这些提示通过考虑人类可理解的文本片段进行工程设计。我们在使用公开可用的面部生物特征数据集开发的一个面部形态数据集上进行了大量实验。我们提出了对当前最佳预训练神经网络与所提出框架在五个不同形态生成技术的零样本评估中的评估，这些技术在三个不同的媒介中被捕获。 

---
