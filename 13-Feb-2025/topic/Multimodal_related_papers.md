# A Novel Approach to for Multimodal Emotion Recognition : Multimodal semantic information fusion 

**Title (ZH)**: 一种新颖的多模态情感识别方法：多模态语义信息融合 

**Authors**: Wei Dai, Dequan Zheng, Feng Yu, Yanrong Zhang, Yaohui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2502.08573)  

**Abstract**: With the advancement of artificial intelligence and computer vision technologies, multimodal emotion recognition has become a prominent research topic. However, existing methods face challenges such as heterogeneous data fusion and the effective utilization of modality correlations. This paper proposes a novel multimodal emotion recognition approach, DeepMSI-MER, based on the integration of contrastive learning and visual sequence compression. The proposed method enhances cross-modal feature fusion through contrastive learning and reduces redundancy in the visual modality by leveraging visual sequence compression. Experimental results on two public datasets, IEMOCAP and MELD, demonstrate that DeepMSI-MER significantly improves the accuracy and robustness of emotion recognition, validating the effectiveness of multimodal feature fusion and the proposed approach. 

**Abstract (ZH)**: 随着人工智能和计算机视觉技术的发展，多模态情感识别已成为一个突出的研究课题。然而，现有方法面临着异质数据融合和有效利用模态相关性的挑战。本文提出了一种基于对比学习和视觉序列压缩集成的新型多模态情感识别方法DeepMSI-MER。该方法通过对比学习增强跨模态特征融合，并通过利用视觉序列压缩减少视觉模态的冗余。在两个公开数据集IEMOCAP和MELD上的实验结果表明，DeepMSI-MER显著提高了情感识别的准确性和鲁棒性，验证了多模态特征融合的有效性和所提出方法的有效性。 

---
# mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data 

**Title (ZH)**: mmE5: 通过高质量合成数据提高多模态多语言嵌入效果 

**Authors**: Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2502.08468)  

**Abstract**: Multimodal embedding models have gained significant attention for their ability to map data from different modalities, such as text and images, into a unified representation space. However, the limited labeled multimodal data often hinders embedding performance. Recent approaches have leveraged data synthesis to address this problem, yet the quality of synthetic data remains a critical bottleneck. In this work, we identify three criteria for high-quality synthetic multimodal data. First, broad scope ensures that the generated data covers diverse tasks and modalities, making it applicable to various downstream scenarios. Second, robust cross-modal alignment makes different modalities semantically consistent. Third, high fidelity ensures that the synthetic data maintains realistic details to enhance its reliability. Guided by these principles, we synthesize datasets that: (1) cover a wide range of tasks, modality combinations, and languages, (2) are generated via a deep thinking process within a single pass of a multimodal large language model, and (3) incorporate real-world images with accurate and relevant texts, ensuring fidelity through self-evaluation and refinement. Leveraging these high-quality synthetic and labeled datasets, we train a multimodal multilingual E5 model mmE5. Extensive experiments demonstrate that mmE5 achieves state-of-the-art performance on the MMEB Benchmark and superior multilingual performance on the XTD benchmark. Our codes, datasets and models are released in this https URL. 

**Abstract (ZH)**: 高质量合成多模态数据的三大标准及其在mmE5模型中的应用 

---
# Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions 

**Title (ZH)**: 具有模糊名称和复杂交互对象的综合草图+文本查询检索 

**Authors**: Prajwal Gatti, Kshitij Parikh, Dhriti Prasanna Paul, Manish Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2502.08438)  

**Abstract**: Non-native speakers with limited vocabulary often struggle to name specific objects despite being able to visualize them, e.g., people outside Australia searching for numbats. Further, users may want to search for such elusive objects with difficult-to-sketch interactions, e.g., numbat digging in the ground. In such common but complex situations, users desire a search interface that accepts composite multimodal queries comprising hand-drawn sketches of difficult-to-name but easy-to-draw objects and text describing difficult-to-sketch but easy-to-verbalize object attributes or interaction with the scene. This novel problem statement distinctly differs from the previously well-researched TBIR (text-based image retrieval) and SBIR (sketch-based image retrieval) problems. To study this under-explored task, we curate a dataset, CSTBIR (Composite Sketch+Text Based Image Retrieval), consisting of approx. 2M queries and 108K natural scene images. Further, as a solution to this problem, we propose a pretrained multimodal transformer-based baseline, STNET (Sketch+Text Network), that uses a hand-drawn sketch to localize relevant objects in the natural scene image, and encodes the text and image to perform image retrieval. In addition to contrastive learning, we propose multiple training objectives that improve the performance of our model. Extensive experiments show that our proposed method outperforms several state-of-the-art retrieval methods for text-only, sketch-only, and composite query modalities. We make the dataset and code available at our project website. 

**Abstract (ZH)**: 非母语者因词汇量有限，在命名特定物体时往往存在困难，尽管他们能够想象这些物体，例如澳大利亚以外的人搜索数batim。此外，用户可能希望使用难以绘制的交互方式来查找这些难以描述的物体，例如数batim在地面挖洞。在这种常见但复杂的场景下，用户希望有一个能够接受组合多模态查询的搜索界面，这些查询包含难以命名但容易绘制的物体的手绘草图，以及描述难以绘制但容易描述的物体属性或与场景的交互的文本。这一新颖的问题陈述与此前广泛研究的TBIR（基于文本的图像检索）和SBIR（基于素描的图像检索）问题大不相同。为了研究这一尚未充分探索的任务，我们收集了一个数据集CSTBIR（组合素描+文本基于图像检索），包含约200万查询和10.8万自然场景图像。此外，为了解决这一问题，我们提出了一种预训练的多模态变压器基线模型STNET（素描+文本网络），该模型利用手绘草图在自然场景图像中定位相关物体，并编码文本和图像以进行图像检索。除了对比学习外，我们还提出多个训练目标以提高模型性能。广泛实验表明，我们提出的方法在仅文本、仅素描和组合查询模式下优于多种最新检索方法。我们在项目网站上提供了数据集和代码。 

---
# Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting 

**Title (ZH)**: 通过约束意识型提示减轻多模态空间关系中的幻觉现象 

**Authors**: Jiarui Wu, Zhuo Liu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2502.08317)  

**Abstract**: Spatial relation hallucinations pose a persistent challenge in large vision-language models (LVLMs), leading to generate incorrect predictions about object positions and spatial configurations within an image. To address this issue, we propose a constraint-aware prompting framework designed to reduce spatial relation hallucinations. Specifically, we introduce two types of constraints: (1) bidirectional constraint, which ensures consistency in pairwise object relations, and (2) transitivity constraint, which enforces relational dependence across multiple objects. By incorporating these constraints, LVLMs can produce more spatially coherent and consistent outputs. We evaluate our method on three widely-used spatial relation datasets, demonstrating performance improvements over existing approaches. Additionally, a systematic analysis of various bidirectional relation analysis choices and transitivity reference selections highlights greater possibilities of our methods in incorporating constraints to mitigate spatial relation hallucinations. 

**Abstract (ZH)**: 空间关系幻觉是大型视觉语言模型（LVLMs）面临的一个持续性挑战，导致模型在生成图像中对象位置和空间配置的错误预测。为解决这一问题，我们提出了一种约束感知的提示框架，旨在减少空间关系幻觉。具体而言，我们引入了两种类型的约束：（1）双向约束，确保对象对之间的关系一致；（2）传递性约束， enforced 多个对象间的关系依赖。通过纳入这些约束，LVLMs 可以生成更具空间一致性和连贯性的输出。我们在三个广泛使用的空间关系数据集上评估了我们的方法，显示出性能改进。此外，对各种双向关系分析选择和传递性引用选择的系统分析表明，我们的方法具有更大的能力来纳入约束以减轻空间关系幻觉。 

---
# What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations 

**Title (ZH)**: 科学演讲的视频到文本摘要数据集：关于这场演讲说什么？ 

**Authors**: Dongqi Liu, Chenxi Whitehouse, Xi Yu, Louis Mahon, Rohit Saxena, Zheng Zhao, Yifu Qiu, Mirella Lapata, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2502.08279)  

**Abstract**: Transforming recorded videos into concise and accurate textual summaries is a growing challenge in multimodal learning. This paper introduces VISTA, a dataset specifically designed for video-to-text summarization in scientific domains. VISTA contains 18,599 recorded AI conference presentations paired with their corresponding paper abstracts. We benchmark the performance of state-of-the-art large models and apply a plan-based framework to better capture the structured nature of abstracts. Both human and automated evaluations confirm that explicit planning enhances summary quality and factual consistency. However, a considerable gap remains between models and human performance, highlighting the challenges of scientific video summarization. 

**Abstract (ZH)**: 将记录视频转换为精练准确的文字摘要是多模态学习中的一个 growing challenge。本文介绍了 VISTA，一个专门为科学领域视频到文本摘要设计的数据集。VISTA 包含 18,599 场记录的人工智能会议演讲及其相应的论文摘要。我们 benchmarks 状态-of-the-art 大型模型的性能，并应用基于计划的框架以更好地捕捉摘要的结构化特性。人类和自动评估均证实明确的计划能够提高摘要质量并增强事实一致性。然而，模型与人类性能之间仍存在较大差距，突显了科学视频摘要化的挑战。 

---
# ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources 

**Title (ZH)**: ADMN：一种适应层动态输入噪声和计算资源的多模态网络 

**Authors**: Jason Wu, Kang Yang, Lance Kaplan, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2502.07862)  

**Abstract**: Multimodal deep learning systems are deployed in dynamic scenarios due to the robustness afforded by multiple sensing modalities. Nevertheless, they struggle with varying compute resource availability (due to multi-tenancy, device heterogeneity, etc.) and fluctuating quality of inputs (from sensor feed corruption, environmental noise, etc.). Current multimodal systems employ static resource provisioning and cannot easily adapt when compute resources change over time. Additionally, their reliance on processing sensor data with fixed feature extractors is ill-equipped to handle variations in modality quality. Consequently, uninformative modalities, such as those with high noise, needlessly consume resources better allocated towards other modalities. We propose ADMN, a layer-wise Adaptive Depth Multimodal Network capable of tackling both challenges - it adjusts the total number of active layers across all modalities to meet compute resource constraints, and continually reallocates layers across input modalities according to their modality quality. Our evaluations showcase ADMN can match the accuracy of state-of-the-art networks while reducing up to 75% of their floating-point operations. 

**Abstract (ZH)**: 层适应深度多模态网络：应对计算资源波动和输入质量变化的挑战 

---
