# Audio and Multiscale Visual Cues Driven Cross-modal Transformer for Idling Vehicle Detection 

**Title (ZH)**: 基于音频和多尺度视觉线索驱动的跨模态变压器在Idle车辆检测中的应用 

**Authors**: Xiwen Li, Ross Whitaker, Tolga Tasdizen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16102)  

**Abstract**: Idling vehicle detection (IVD) supports real-time systems that reduce pollution and emissions by dynamically messaging drivers to curb excess idling behavior. In computer vision, IVD has become an emerging task that leverages video from surveillance cameras and audio from remote microphones to localize and classify vehicles in each frame as moving, idling, or engine-off. As with other cross-modal tasks, the key challenge lies in modeling the correspondence between audio and visual modalities, which differ in representation but provide complementary cues -- video offers spatial and motion context, while audio conveys engine activity beyond the visual field. The previous end-to-end model, which uses a basic attention mechanism, struggles to align these modalities effectively, often missing vehicle detections. To address this issue, we propose AVIVDNetv2, a transformer-based end-to-end detection network. It incorporates a cross-modal transformer with global patch-level learning, a multiscale visual feature fusion module, and decoupled detection heads. Extensive experiments show that AVIVDNetv2 improves mAP by 7.66 over the disjoint baseline and 9.42 over the E2E baseline, with consistent AP gains across all vehicle categories. Furthermore, AVIVDNetv2 outperforms the state-of-the-art method for sounding object localization, establishing a new performance benchmark on the AVIVD dataset. 

**Abstract (ZH)**: 基于多模态的车辆怠速检测（AVIVDNetv2）：一种端到端的检测网络 

---
# Towards Explainable AI: Multi-Modal Transformer for Video-based Image Description Generation 

**Title (ZH)**: 面向可解释人工智能：基于视频的图像描述生成的多模态变换器 

**Authors**: Lakshita Agarwal, Bindu Verma  

**Link**: [PDF](https://arxiv.org/pdf/2504.16788)  

**Abstract**: Understanding and analyzing video actions are essential for producing insightful and contextualized descriptions, especially for video-based applications like intelligent monitoring and autonomous systems. The proposed work introduces a novel framework for generating natural language descriptions from video datasets by combining textual and visual modalities. The suggested architecture makes use of ResNet50 to extract visual features from video frames that are taken from the Microsoft Research Video Description Corpus (MSVD), and Berkeley DeepDrive eXplanation (BDD-X) datasets. The extracted visual characteristics are converted into patch embeddings and then run through an encoder-decoder model based on Generative Pre-trained Transformer-2 (GPT-2). In order to align textual and visual representations and guarantee high-quality description production, the system uses multi-head self-attention and cross-attention techniques. The model's efficacy is demonstrated by performance evaluation using BLEU (1-4), CIDEr, METEOR, and ROUGE-L. The suggested framework outperforms traditional methods with BLEU-4 scores of 0.755 (BDD-X) and 0.778 (MSVD), CIDEr scores of 1.235 (BDD-X) and 1.315 (MSVD), METEOR scores of 0.312 (BDD-X) and 0.329 (MSVD), and ROUGE-L scores of 0.782 (BDD-X) and 0.795 (MSVD). By producing human-like, contextually relevant descriptions, strengthening interpretability, and improving real-world applications, this research advances explainable AI. 

**Abstract (ZH)**: 理解并分析视频动作对于生成洞察性和上下文相关描述至关重要，尤其是在基于视频的应用如智能监控和自主系统中。本文提出了一种结合文本和视觉模态的新框架，用于从视频数据集中生成自然语言描述。该建议架构利用ResNet50从来自Microsoft Research Video Description Corpus (MSVD) 和Berkeley DeepDrive eXplanation (BDD-X) 数据集的视频帧中提取视觉特征。提取的视觉特征被转换为斑块嵌入，并通过基于生成预训练变压器-2 (GPT-2) 的编码器-解码器模型进行处理。为了对齐文本和视觉表示并确保高质量描述的生成，系统使用了多头自注意力和交叉注意力技术。通过使用BLEU (1-4), CIDEr, METEOR, 和 ROUGE-L 进行性能评估，展示了该模型的有效性。与传统的传统方法相比，所提出的框架在BLEU-4上的得分为0.755 (BDD-X) 和0.778 (MSVD)，CIDEr分数为1.235 (BDD-X) 和1.315 (MSVD)，METEOR分为0.312 (BDD-X) 和0.329 (MSVD)，ROUGE-L分为0.782 (BDD-X) 和0.795 (MSVD)。通过生成人类like、上下文相关描述，增强可解释性和改进实际应用，本文推进了可解释人工智能的发展。 

---
# Detecting and Understanding Hateful Contents in Memes Through Captioning and Visual Question-Answering 

**Title (ZH)**: 通过描述和视觉问答检测和理解 meme 中的 hateful 内容 

**Authors**: Ali Anaissi, Junaid Akram, Kunal Chaturvedi, Ali Braytee  

**Link**: [PDF](https://arxiv.org/pdf/2504.16723)  

**Abstract**: Memes are widely used for humor and cultural commentary, but they are increasingly exploited to spread hateful content. Due to their multimodal nature, hateful memes often evade traditional text-only or image-only detection systems, particularly when they employ subtle or coded references. To address these challenges, we propose a multimodal hate detection framework that integrates key components: OCR to extract embedded text, captioning to describe visual content neutrally, sub-label classification for granular categorization of hateful content, RAG for contextually relevant retrieval, and VQA for iterative analysis of symbolic and contextual cues. This enables the framework to uncover latent signals that simpler pipelines fail to detect. Experimental results on the Facebook Hateful Memes dataset reveal that the proposed framework exceeds the performance of unimodal and conventional multimodal models in both accuracy and AUC-ROC. 

**Abstract (ZH)**: 表情包广泛用于幽默和文化评论，但它们越来越多地被用于传播仇恨内容。由于其多模态性质，仇恨表情包常常规避传统只针对文本或图像的检测系统，尤其是在它们使用含蓄或编码的引用时。为应对这些挑战，我们提出了一种多模态仇恨内容检测框架，该框架整合了关键组件：OCR用于提取嵌入文本、字幕描述视觉内容、子标签分类进行仇恨内容的细粒度分类、基于检索的生成（RAG）以实现上下文相关的内容检索，以及视觉问答（VQA）进行迭代分析象征性和上下文线索。这使得该框架能够发现简单流水线未能检测到的潜在信号。实验结果表明，在Facebook仇恨表情包数据集上，所提框架在准确率和AUC-ROC方面均超过了单模态和传统多模态模型。 

---
# Can Large Language Models Help Multimodal Language Analysis? MMLA: A Comprehensive Benchmark 

**Title (ZH)**: 大语言模型能辅助多模态语言分析吗？MMLA：一项全面的基准 

**Authors**: Hanlei Zhang, Zhuohang Li, Yeshuang Zhu, Hua Xu, Peiwu Wang, Jinchao Zhang, Jie Zhou, Haige Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16427)  

**Abstract**: Multimodal language analysis is a rapidly evolving field that leverages multiple modalities to enhance the understanding of high-level semantics underlying human conversational utterances. Despite its significance, little research has investigated the capability of multimodal large language models (MLLMs) to comprehend cognitive-level semantics. In this paper, we introduce MMLA, a comprehensive benchmark specifically designed to address this gap. MMLA comprises over 61K multimodal utterances drawn from both staged and real-world scenarios, covering six core dimensions of multimodal semantics: intent, emotion, dialogue act, sentiment, speaking style, and communication behavior. We evaluate eight mainstream branches of LLMs and MLLMs using three methods: zero-shot inference, supervised fine-tuning, and instruction tuning. Extensive experiments reveal that even fine-tuned models achieve only about 60%~70% accuracy, underscoring the limitations of current MLLMs in understanding complex human language. We believe that MMLA will serve as a solid foundation for exploring the potential of large language models in multimodal language analysis and provide valuable resources to advance this field. The datasets and code are open-sourced at this https URL. 

**Abstract (ZH)**: 多模态语言分析是一个迅速发展的领域，它利用多种模态来增强对人类对话表达高级语义的理解。尽管其重要性不言而喻，但鲜有研究探讨多模态大规模语言模型（MLLMs）在理解认知层次语义方面的能力。本文我们引入了MMLA，这是一个专门为此空白设计的综合基准。MMLA 包含超过61,000条多模态对话数据，涵盖了六种核心的多模态语义维度：意图、情绪、对话行为、情感、说话风格和沟通行为。我们使用三种方法评估了八种主流的LLM和MLLM分支：零样本推理、监督微调和指令微调。广泛的实验表明，即使是微调后的模型也只能达到约60%~70%的准确率，突显了当前MLLMs在理解复杂人类语言方面的局限性。我们认为MMLA 将为探索大规模语言模型在多模态语言分析中的潜力提供坚实的基础，并提供有价值的数据资源来推进这一领域的发展。数据集和代码已开源于此链接：this https URL。 

---
