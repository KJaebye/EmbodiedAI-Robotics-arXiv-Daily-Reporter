# Exploring Machine Learning and Language Models for Multimodal Depression Detection 

**Title (ZH)**: 探索机器学习和语言模型在 multimodal 抑郁检测中的应用 

**Authors**: Javier Si Zhao Hong, Timothy Zoe Delaya, Sherwyn Chan Yin Kit, Pai Chet Ng, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20805)  

**Abstract**: This paper presents our approach to the first Multimodal Personality-Aware Depression Detection Challenge, focusing on multimodal depression detection using machine learning and deep learning models. We explore and compare the performance of XGBoost, transformer-based architectures, and large language models (LLMs) on audio, video, and text features. Our results highlight the strengths and limitations of each type of model in capturing depression-related signals across modalities, offering insights into effective multimodal representation strategies for mental health prediction. 

**Abstract (ZH)**: 本文提出了我们参加首次多模态个性意识抑郁症检测挑战赛的方法，重点在于使用机器学习和深度学习模型进行多模态抑郁症检测。我们探索并比较了XGBoost、基于变换器的架构以及大语言模型（LLMs）在音频、视频和文本特征上的性能，结果显示了每种类型模型在捕捉跨模态的抑郁症相关信号方面的优势与局限性，为我们提供了有关精神健康预测的有效多模态表示策略的见解。 

---
# Speech Emotion Recognition via Entropy-Aware Score Selection 

**Title (ZH)**: 基于熵感知评分选择的语音情绪识别 

**Authors**: ChenYi Chua, JunKai Wong, Chengxin Chen, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20796)  

**Abstract**: In this paper, we propose a multimodal framework for speech emotion recognition that leverages entropy-aware score selection to combine speech and textual predictions. The proposed method integrates a primary pipeline that consists of an acoustic model based on wav2vec2.0 and a secondary pipeline that consists of a sentiment analysis model using RoBERTa-XLM, with transcriptions generated via Whisper-large-v3. We propose a late score fusion approach based on entropy and varentropy thresholds to overcome the confidence constraints of primary pipeline predictions. A sentiment mapping strategy translates three sentiment categories into four target emotion classes, enabling coherent integration of multimodal predictions. The results on the IEMOCAP and MSP-IMPROV datasets show that the proposed method offers a practical and reliable enhancement over traditional single-modality systems. 

**Abstract (ZH)**: 本文提出一种熵意识评分选择的多模态框架用于语音情绪识别，并结合语音和文本预测。所提出的方法包含一个基于wav2vec2.0的声音模型为主的管道和一个使用RoBERTa-XLM进行情感分析的次级管道，通过Whisper-large-v3生成转录。我们提出了一种基于熵和变熵阈值的后期评分融合方法，以克服主要管道预测的信心约束。情感映射策略将三种情感类别转换为四种目标情绪类别，使多模态预测得以一致融合。在IEMOCAP和MSP-IMPROV数据集上的结果表明，所提出的方法比传统的单模态系统提供了实用且可靠的改进。 

---
# MM-HSD: Multi-Modal Hate Speech Detection in Videos 

**Title (ZH)**: 多模态视频仇恨言论检测：MM-HSD 

**Authors**: Berta Céspedes-Sarrias, Carlos Collado-Capell, Pablo Rodenas-Ruiz, Olena Hrynenko, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.20546)  

**Abstract**: While hate speech detection (HSD) has been extensively studied in text, existing multi-modal approaches remain limited, particularly in videos. As modalities are not always individually informative, simple fusion methods fail to fully capture inter-modal dependencies. Moreover, previous work often omits relevant modalities such as on-screen text and audio, which may contain subtle hateful content and thus provide essential cues, both individually and in combination with others. In this paper, we present MM-HSD, a multi-modal model for HSD in videos that integrates video frames, audio, and text derived from speech transcripts and from frames (i.e.~on-screen text) together with features extracted by Cross-Modal Attention (CMA). We are the first to use CMA as an early feature extractor for HSD in videos, to systematically compare query/key configurations, and to evaluate the interactions between different modalities in the CMA block. Our approach leads to improved performance when on-screen text is used as a query and the rest of the modalities serve as a key. Experiments on the HateMM dataset show that MM-HSD outperforms state-of-the-art methods on M-F1 score (0.874), using concatenation of transcript, audio, video, on-screen text, and CMA for feature extraction on raw embeddings of the modalities. The code is available at this https URL 

**Abstract (ZH)**: 多模态视频仇恨言论检测模型：MM-HSD 

---
# How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding 

**Title (ZH)**: 多模态LLMs解决图像任务：视觉 grounding、任务推理与答案解码之窗 

**Authors**: Zhuoran Yu, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20279)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在广泛的视觉-语言任务中表现出了强烈的能力，但其内部处理动态仍被探讨不足。在本文中，我们引入了一种探针框架，系统地分析MLLMs如何在不同层中处理视觉和文本输入。我们训练线性分类器，从每个层提取的词元嵌入中预测细粒度的视觉类别（例如，狗的品种），使用标准化的锚定问题。为了揭示不同层的功能作用，我们在这三种类型的控制提示变化下评估这些探针：（1）词汇变体测试对表层变化的敏感性；（2）语义否定变体通过修改提示中的视觉概念翻转预期答案；（3）输出格式变体保持推理但改变答案格式。将我们的框架应用于LLaVA-1.5、LLaVA-Next-LLaMA-3和Qwen2-VL，我们发现了一个一致的阶段式结构，在早期层中进行视觉定位，在中间层中支持词汇整合和语义推理，在最终层中准备特定任务的输出。进一步结果显示，尽管在视觉词元化、指令调优数据和预训练语料库的变化中，整体阶段式结构保持稳定，但每个阶段的特定层分配随基础大语言模型架构的变化显著变化。我们的发现提供了一种关于MLLMs逐层组织的统一视角，并提出了一种轻量级、模型无关的方法来分析多模态表示动态。 

---
# A Novel Framework for Automated Explain Vision Model Using Vision-Language Models 

**Title (ZH)**: 一种基于视觉-语言模型的自动化解释视觉模型的新框架 

**Authors**: Phu-Vinh Nguyen, Tan-Hanh Pham, Chris Ngo, Truong Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2508.20227)  

**Abstract**: The development of many vision models mainly focuses on improving their performance using metrics such as accuracy, IoU, and mAP, with less attention to explainability due to the complexity of applying xAI methods to provide a meaningful explanation of trained models. Although many existing xAI methods aim to explain vision models sample-by-sample, methods explaining the general behavior of vision models, which can only be captured after running on a large dataset, are still underexplored. Furthermore, understanding the behavior of vision models on general images can be very important to prevent biased judgments and help identify the model's trends and patterns. With the application of Vision-Language Models, this paper proposes a pipeline to explain vision models at both the sample and dataset levels. The proposed pipeline can be used to discover failure cases and gain insights into vision models with minimal effort, thereby integrating vision model development with xAI analysis to advance image analysis. 

**Abstract (ZH)**: 多种视觉模型的发展主要集中在使用准确率、IoU和mAP等指标提高其性能，但较少关注可解释性，这主要是因为将xAI方法应用于提供有意义的解释较为复杂。尽管存在许多旨在样本级解释视觉模型的xAI方法，但在大型数据集上运行以捕获视觉模型一般行为的方法仍然尚未得到充分利用。此外，理解视觉模型在一般图像上的行为对于防止有偏的判断并帮助识别模型的趋势和模式非常重要。在视觉语言模型的应用下，本文提出了一种管道，用于在样本级和数据集级解释视觉模型。该提出的管道可以用于发现失败案例并以最少的努力获得关于视觉模型的见解，从而将视觉模型开发与xAI分析整合起来，以促进图像分析的发展。 

---
