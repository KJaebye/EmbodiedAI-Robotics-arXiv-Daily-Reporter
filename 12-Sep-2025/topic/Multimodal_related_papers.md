# Visual Grounding from Event Cameras 

**Title (ZH)**: 事件相机的视觉定位 

**Authors**: Lingdong Kong, Dongyue Lu, Ao Liang, Rong Li, Yuhao Dong, Tianshuai Hu, Lai Xing Ng, Wei Tsang Ooi, Benoit R. Cottereau  

**Link**: [PDF](https://arxiv.org/pdf/2509.09584)  

**Abstract**: Event cameras capture changes in brightness with microsecond precision and remain reliable under motion blur and challenging illumination, offering clear advantages for modeling highly dynamic scenes. Yet, their integration with natural language understanding has received little attention, leaving a gap in multimodal perception. To address this, we introduce Talk2Event, the first large-scale benchmark for language-driven object grounding using event data. Built on real-world driving scenarios, Talk2Event comprises 5,567 scenes, 13,458 annotated objects, and more than 30,000 carefully validated referring expressions. Each expression is enriched with four structured attributes -- appearance, status, relation to the viewer, and relation to surrounding objects -- that explicitly capture spatial, temporal, and relational cues. This attribute-centric design supports interpretable and compositional grounding, enabling analysis that moves beyond simple object recognition to contextual reasoning in dynamic environments. We envision Talk2Event as a foundation for advancing multimodal and temporally-aware perception, with applications spanning robotics, human-AI interaction, and so on. 

**Abstract (ZH)**: 事件相机以微秒级精度捕捉亮度变化，即使在运动模糊和复杂光照条件下仍保持可靠性能，为建模高度动态场景提供了明显优势。然而，它们与自然语言理解的集成方面尚未得到充分关注，这在多模态感知方面留下了一定差距。为解决这一问题，我们引入了Talk2Event，这是一个基于事件数据的语言驱动对象定位的首个大规模基准。Talk2Event涵盖了5,567个场景、13,458个标注对象以及超过30,000个仔细验证的指示表达式。每个表达式都包含四个结构化属性——外观、状态、与观者的相关性以及与周围对象的相关性——这些属性明确捕捉了空间、时间及关系线索。该属性中心设计支持可解释和组合式定位，能够超越简单的对象识别，实现动态环境中的上下文推理。我们设想Talk2Event将成为推动多模态及时间敏感感知的基础，应用于机器人技术、人机交互等领域。 

---
# Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization 

**Title (ZH)**: 多模态LLM能清晰地“看见”材料吗？一种用于材料表征的多模态基准测试 

**Authors**: Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jinpu Yang, Hongqing Liang, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09307)  

**Abstract**: Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at this https URL. 

**Abstract (ZH)**: 材料表征是获取材料信息的基础，揭示了加工-微观结构-性能关系，指导材料设计与优化。虽然多模态大语言模型（MLLMs）在材料科学领域的生成性和预测性任务中已显示出潜力，但其理解和处理现实世界的表征影像数据的能力仍然未被充分探索。为弥补这一缺口，我们提出了MatCha——首个材料表征影像理解基准，包含1500个要求专家级领域知识的问题。MatCha涵盖了材料研究的四个关键阶段，包含21项不同的任务，旨在反映材料科学家所面临的实际挑战。我们在MatCha上对最新一代MLLMs的评估结果显示，这些模型在性能上与人类专家之间存在显著差距。这些模型在处理需要较高水平专业知识和复杂视觉感知的问题时表现出能力下降。即使是简单的少量示例提示和链式思考提示也难以缓解这些限制。这些发现表明，现有的MLLMs在适应现实世界的材料表征场景方面仍然表现有限。我们希望MatCha能够促进未来在新材料发现和自主科学代理等领域的研究。MatCha可在以下网址获取：this https URL。 

---
# Target-oriented Multimodal Sentiment Classification with Counterfactual-enhanced Debiasing 

**Title (ZH)**: 面向目标的多模态情感分类与反事实增强去偏见 

**Authors**: Zhiyue Liu, Fanrong Ma, Xin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09160)  

**Abstract**: Target-oriented multimodal sentiment classification seeks to predict sentiment polarity for specific targets from image-text pairs. While existing works achieve competitive performance, they often over-rely on textual content and fail to consider dataset biases, in particular word-level contextual biases. This leads to spurious correlations between text features and output labels, impairing classification accuracy. In this paper, we introduce a novel counterfactual-enhanced debiasing framework to reduce such spurious correlations. Our framework incorporates a counterfactual data augmentation strategy that minimally alters sentiment-related causal features, generating detail-matched image-text samples to guide the model's attention toward content tied to sentiment. Furthermore, for learning robust features from counterfactual data and prompting model decisions, we introduce an adaptive debiasing contrastive learning mechanism, which effectively mitigates the influence of biased words. Experimental results on several benchmark datasets show that our proposed method outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 面向目标的多模态情感分类旨在从图像-文本对中预测特定目标的情感极性。尽管现有工作取得了竞争力的表现，但它们往往过度依赖文本内容，并未考虑数据集偏见，特别是字面级上下文偏见。这导致了文本特征与输出标签之间的虚假相关性，影响了分类准确性。在本文中，我们提出了一种新颖的反事实增强去偏见框架以降低此类虚假相关性。我们的框架结合了一种最小改变情感相关因果特征的反事实数据增强策略，生成细节匹配的图像-文本样本以引导模型关注与情感相关的内容。此外，为了从反事实数据中学习稳健的特征并促使模型决策，我们引入了一种适应性去偏见对比学习机制，有效缓解了偏色词汇的影响。在几个基准数据集上的实验结果表明，我们提出的方法优于现有最先进的基线方法。 

---
# SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework for High-Performance Vision-Language-Action Models 

**Title (ZH)**: SQAP-VLA：一种面向高性能视觉-语言-动作模型的协同量化意识剪枝框架 

**Authors**: Hengyu Fang, Yijiang Liu, Yuan Du, Li Du, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09090)  

**Abstract**: Vision-Language-Action (VLA) models exhibit unprecedented capabilities for embodied intelligence. However, their extensive computational and memory costs hinder their practical deployment. Existing VLA compression and acceleration approaches conduct quantization or token pruning in an ad-hoc manner but fail to enable both for a holistic efficiency improvement due to an observed incompatibility. This work introduces SQAP-VLA, the first structured, training-free VLA inference acceleration framework that simultaneously enables state-of-the-art quantization and token pruning. We overcome the incompatibility by co-designing the quantization and token pruning pipeline, where we propose new quantization-aware token pruning criteria that work on an aggressively quantized model while improving the quantizer design to enhance pruning effectiveness. When applied to standard VLA models, SQAP-VLA yields significant gains in computational efficiency and inference speed while successfully preserving core model performance, achieving a $\times$1.93 speedup and up to a 4.5\% average success rate enhancement compared to the original model. 

**Abstract (ZH)**: SQAP-VLA：同时实现先进量化和 token 裁剪的结构化无训练加速框架 

---
# Recurrence Meets Transformers for Universal Multimodal Retrieval 

**Title (ZH)**: 循环神经网络结合变换器用于通用多模态检索 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2509.08897)  

**Abstract**: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: this https URL 

**Abstract (ZH)**: 随着多模态检索及其在LLMs和多模态LLMs中的应用的迅速发展，越来越复杂的检索任务相继出现。现有的方法主要依赖于针对特定任务对视觉语言模型进行细调，并且局限于单模态查询或文档。在本文中，我们提出了一种名为ReT-2的统一检索模型，该模型支持包含图像和文本的多模态查询，并能够在文本和图像共存的多模态文档集合中进行跨模态搜索。ReT-2利用多层表示和基于LSTM启发式门控机制的递归Transformer架构，动态地在层间和模态间整合信息，捕捉精细的视觉和文本细节。我们在挑战性的M2KR和M-BEIR基准上，对ReT-2在不同检索配置下的性能进行了评估。结果表明，ReT-2在各种场景中均能够实现最优性能，相比先前的方法具有更快的推理速度和更低的内存使用。当集成到检索增强生成管道中时，ReT-2还能够提高Encyclopedic-VQA和InfoSeek数据集的下游性能。我们的源代码和训练模型已在以下网址公开：this https URL。 

---
