# Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems 

**Title (ZH)**: Automotive-ENV：车辆界面系统中多模态代理的基准测试 

**Authors**: Junfeng Yan, Biao Wu, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21143)  

**Abstract**: Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents. 

**Abstract (ZH)**: 多模态代理在通用GUI交互中展现了强大的性能，但在汽车系统中的应用尚未被广泛探索。车载GUI提出了独特的挑战：驾驶员注意力有限、严格的安全要求以及复杂的基于位置的交互模式。为应对这些挑战，我们引入了Automotive-ENV，这是首款针对车辆GUI定制的高度逼真基准和交互环境。该平台定义了185个参数化任务，涵盖显式控制、隐式意图理解以及安全感知任务，并提供了结构化的多模态观察数据和精确的程序检查，以实现可复现的评估。在此基准之上，我们提出了ASURADA，一种地理感知的多模态代理，该代理结合GPS信息的上下文，基于位置、环境条件和地区驾驶规范动态调整行动。实验结果显示，地理感知信息显著提高了安全感知任务的成功率，强调了车载环境中地理位置上下文的重要性。我们将发布Automotive-ENV，包含所有任务和基准测试工具，以促进安全且适应性强的车载代理的发展。 

---
# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning 

**Title (ZH)**: DeFacto: 基于图像的反事实思考以确保证据驱动和忠实的推理 

**Authors**: Tianrun Xu, Haoda Jing, Ye Li, Yuquan Wei, Jun Feng, Guanyu Chen, Haichuan Gao, Tianren Zhang, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20912)  

**Abstract**: Recent advances in multimodal language models (MLLMs) have achieved remarkable progress in vision-language reasoning, especially with the emergence of "thinking with images," which integrates explicit visual steps into the reasoning process. While this paradigm strengthens image-based reasoning, a significant challenge remains: models may arrive at correct answers by relying on irrelevant or spurious regions, driven by prior knowledge or dataset biases. Even when the answer is correct, flawed reasoning indicates that the model has not truly understood the image, highlighting the critical importance of reasoning fidelity in multimodal tasks. To address this issue, we propose DeFacto, a counterfactual reasoning framework that jointly enforces accurate answering and faithful reasoning. A key component of our approach is the design of three complementary training paradigms: (i) positive, (ii) counterfactual, and (iii) random-masking. To enable these paradigms, we develop a pipeline that automatically localizes question-relevant evidence and constructs positive, counterfactual, and random variants, resulting in a dataset of about 100k images. Building on this framework, we train multimodal language models with GRPO-based reinforcement learning, where we design three complementary rewards to guide the model toward accurate answering and evidence-grounded reasoning. Experiments on diverse benchmarks demonstrate that DeFacto substantially improves both answer accuracy and reasoning faithfulness, establishing a stronger foundation for interpretable multimodal reasoning. The code is available on GitHub and the dataset is released on HuggingFace. 

**Abstract (ZH)**: Recent Advances in Multimodal Language Models: Addressing Challenges in Vision-Language Reasoning with DeFacto 

---
# DisCoCLIP: A Distributional Compositional Tensor Network Encoder for Vision-Language Understanding 

**Title (ZH)**: 分布组合张量网络编码器：用于视觉-语言理解的DisCoCLIP 

**Authors**: Kin Ian Lo, Hala Hawashin, Mina Abbaszadeh, Tilen Limback-Stokin, Hadi Wazni, Mehrnoosh Sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.21287)  

**Abstract**: Recent vision-language models excel at large-scale image-text alignment but often neglect the compositional structure of language, leading to failures on tasks that hinge on word order and predicate-argument structure. We introduce DisCoCLIP, a multimodal encoder that combines a frozen CLIP vision transformer with a novel tensor network text encoder that explicitly encodes syntactic structure. Sentences are parsed with a Combinatory Categorial Grammar parser to yield distributional word tensors whose contractions mirror the sentence's grammatical derivation. To keep the model efficient, high-order tensors are factorized with tensor decompositions, reducing parameter count from tens of millions to under one million. Trained end-to-end with a self-supervised contrastive loss, DisCoCLIP markedly improves sensitivity to verb semantics and word order: it raises CLIP's SVO-Probes verb accuracy from 77.6% to 82.4%, boosts ARO attribution and relation scores by over 9% and 4%, and achieves 93.7% on a newly introduced SVO-Swap benchmark. These results demonstrate that embedding explicit linguistic structure via tensor networks yields interpretable, parameter-efficient representations that substantially improve compositional reasoning in vision-language tasks. 

**Abstract (ZH)**: Recent Vision-Language Models Often Neglect Linguistic Compositionality: Introducing DisCoCLIP 

---
# Unlocking Financial Insights: An advanced Multimodal Summarization with Multimodal Output Framework for Financial Advisory Videos 

**Title (ZH)**: 解锁财务洞察：一种用于金融顾问视频的先进多模态总结框架及多模态输出模型 

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.20961)  

**Abstract**: The dynamic propagation of social media has broadened the reach of financial advisory content through podcast videos, yet extracting insights from lengthy, multimodal segments (30-40 minutes) remains challenging. We introduce FASTER (Financial Advisory Summariser with Textual Embedded Relevant images), a modular framework that tackles three key challenges: (1) extracting modality-specific features, (2) producing optimized, concise summaries, and (3) aligning visual keyframes with associated textual points. FASTER employs BLIP for semantic visual descriptions, OCR for textual patterns, and Whisper-based transcription with Speaker diarization as BOS features. A modified Direct Preference Optimization (DPO)-based loss function, equipped with BOS-specific fact-checking, ensures precision, relevance, and factual consistency against the human-aligned summary. A ranker-based retrieval mechanism further aligns keyframes with summarized content, enhancing interpretability and cross-modal coherence. To acknowledge data resource scarcity, we introduce Fin-APT, a dataset comprising 470 publicly accessible financial advisory pep-talk videos for robust multimodal research. Comprehensive cross-domain experiments confirm FASTER's strong performance, robustness, and generalizability when compared to Large Language Models (LLMs) and Vision-Language Models (VLMs). By establishing a new standard for multimodal summarization, FASTER makes financial advisory content more accessible and actionable, thereby opening new avenues for research. The dataset and code are available at: this https URL 

**Abstract (ZH)**: 社交媒体动态传播通过播客视频拓宽了财经顾问内容的覆盖面，但从中提取见解仍面临挑战，尤其是在长达30-40分钟的多模态段落中。我们提出了一种模块化框架FASTER（Financial Advisory Summariser with Textual Embedded Relevant images），以应对三个关键挑战：（1）提取特定模态特征，（2）生成优化的摘要，（3）将视觉关键帧与相关文本要点对齐。FASTER使用BLIP进行语义视觉描述，OCR进行文本模式识别，并使用基于Whisper的转录与讲者定位作为BOS特征。通过结合针对BOS的具体事实核查的改进的Direct Preference Optimization (DPO)-基于损失函数，FASTER确保了精准性、相关性和事实一致性，与人工对齐的摘要比对。基于排名的检索机制进一步将关键帧与摘要内容对齐，增强了可解释性和跨模态一致性。为应对数据资源稀缺，我们引入了Fin-APT数据集，包含470个公开可访问的财经顾问激励视频，以支撑稳健的多模态研究。跨领域实验证明，与大型语言模型（LLMs）和视觉-语言模型（VLMs）相比，FASTER在性能、稳健性和通用性方面表现出色。通过确立新的多模态总结标准，FASTER使财经顾问内容更加易于获取和实用，并为研究开辟了新途径。数据集和代码可在以下链接获取：this https URL。 

---
# Revolutionizing Precise Low Back Pain Diagnosis via Contrastive Learning 

**Title (ZH)**: 基于对比学习的精确腰椎疼痛诊断革命 

**Authors**: Thanh Binh Le, Hoang Nhat Khang Vo, Tan-Ha Mai, Trong Nhan Phan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20813)  

**Abstract**: Low back pain affects millions worldwide, driving the need for robust diagnostic models that can jointly analyze complex medical images and accompanying text reports. We present LumbarCLIP, a novel multimodal framework that leverages contrastive language-image pretraining to align lumbar spine MRI scans with corresponding radiological descriptions. Built upon a curated dataset containing axial MRI views paired with expert-written reports, LumbarCLIP integrates vision encoders (ResNet-50, Vision Transformer, Swin Transformer) with a BERT-based text encoder to extract dense representations. These are projected into a shared embedding space via learnable projection heads, configurable as linear or non-linear, and normalized to facilitate stable contrastive training using a soft CLIP loss. Our model achieves state-of-the-art performance on downstream classification, reaching up to 95.00% accuracy and 94.75% F1-score on the test set, despite inherent class imbalance. Extensive ablation studies demonstrate that linear projection heads yield more effective cross-modal alignment than non-linear variants. LumbarCLIP offers a promising foundation for automated musculoskeletal diagnosis and clinical decision support. 

**Abstract (ZH)**: 低背部疼痛影响全球数百万人，推动了需要能够联合分析复杂医学图像和相应文字报告的 robust 诊断模型的发展。我们提出 LumbarCLIP，这是一种新颖的多模态框架，利用对比语言-图像预训练对齐腰椎 MRI 扫描与其相应的放射学描述。LumbarCLIP 依托于一个经过策展的数据集，该数据集包含轴向 MRI 视图及其由专家撰写的报告，综合了视觉编码器（ResNet-50、Vision Transformer、Swin Transformer）与基于 BERT 的文本编码器以提取密集表示。通过可学习的投影头将这些表示投影到共享的嵌入空间中，配置为线性或非线性，并通过软 CLIP 损失进行规范化，以实现稳定的对比训练。尽管存在固有的类别不平衡，我们的模型在下游分类任务中实现了最先进的性能，测试集上准确率达到 95.00%、F1 分数达到 94.75%。广泛的消融研究显示，线性投影头相比非线性变体能更有效地实现跨模态对齐。LumbarCLIP 为自动肌肉骨骼诊断和临床决策支持提供了一个有前景的基础。 

---
# Provenance Analysis of Archaeological Artifacts via Multimodal RAG Systems 

**Title (ZH)**: 考古文物多模态RAG系统中的溯源分析 

**Authors**: Tuo Zhang, Yuechun Sun, Ruiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20769)  

**Abstract**: In this work, we present a retrieval-augmented generation (RAG)-based system for provenance analysis of archaeological artifacts, designed to support expert reasoning by integrating multimodal retrieval and large vision-language models (VLMs). The system constructs a dual-modal knowledge base from reference texts and images, enabling raw visual, edge-enhanced, and semantic retrieval to identify stylistically similar objects. Retrieved candidates are synthesized by the VLM to generate structured inferences, including chronological, geographical, and cultural attributions, alongside interpretive justifications. We evaluate the system on a set of Eastern Eurasian Bronze Age artifacts from the British Museum. Expert evaluation demonstrates that the system produces meaningful and interpretable outputs, offering scholars concrete starting points for analysis and significantly alleviating the cognitive burden of navigating vast comparative corpora. 

**Abstract (ZH)**: 基于检索增强生成（RAG）的考古 artifacts 起源分析系统：结合多模态检索和大型视觉语言模型支持专家推理 

---
# Seeing Through Words, Speaking Through Pixels: Deep Representational Alignment Between Vision and Language Models 

**Title (ZH)**: 透过词语看世界，通过像素说话：视觉模型与语言模型的深层表示对齐 

**Authors**: Zoe Wanying He, Sean Trott, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.20751)  

**Abstract**: Recent studies show that deep vision-only and language-only models--trained on disjoint modalities--nonetheless project their inputs into a partially aligned representational space. Yet we still lack a clear picture of where in each network this convergence emerges, what visual or linguistic cues support it, whether it captures human preferences in many-to-many image-text scenarios, and how aggregating exemplars of the same concept affects alignment. Here, we systematically investigate these questions. We find that alignment peaks in mid-to-late layers of both model types, reflecting a shift from modality-specific to conceptually shared representations. This alignment is robust to appearance-only changes but collapses when semantics are altered (e.g., object removal or word-order scrambling), highlighting that the shared code is truly semantic. Moving beyond the one-to-one image-caption paradigm, a forced-choice "Pick-a-Pic" task shows that human preferences for image-caption matches are mirrored in the embedding spaces across all vision-language model pairs. This pattern holds bidirectionally when multiple captions correspond to a single image, demonstrating that models capture fine-grained semantic distinctions akin to human judgments. Surprisingly, averaging embeddings across exemplars amplifies alignment rather than blurring detail. Together, our results demonstrate that unimodal networks converge on a shared semantic code that aligns with human judgments and strengthens with exemplar aggregation. 

**Abstract (ZH)**: 近期研究表明，尽管深度纯视觉模型和纯语言模型在各自独立的模态下训练，它们依然将输入投影到一个部分对齐的表示空间中。然而，我们仍然缺乏清晰的认识：这种收敛在每个网络中的哪个层次出现，哪些视觉或语言线索支持这一过程，它是否捕捉到了人类在一对多的图像-文本场景中的偏好，以及同一概念的示例聚合如何影响对齐。在这里，我们系统地研究了这些问题。我们发现这种对齐在两种模型类型的中到后期层中达到峰值，反映了从模态特定表示到概念共享表示的转变。这种对齐在仅外观变化时是稳健的，但在语义变化（例如，移除对象或词序打乱）时会崩溃，突显了共享代码真正的语义属性。超越一对一的图像-标题范式，“选一张图片”任务表明，人类对图像-标题匹配的偏好在所有视觉-语言模型对的嵌入空间中得到了镜像。当多个标题对应单个图像时，这一模式双向成立，证明模型捕捉到了类似人类判断的细微语义区别。令人惊讶的是，跨示例平均嵌入反而增强了对齐而非模糊细节。综上所述，我们的结果表明，单模网络收敛于一个与人类判断相一致的共享语义代码，并且随着示例聚合而增强。 

---
