# ADMC: Attention-based Diffusion Model for Missing Modalities Feature Completion 

**Title (ZH)**: 基于注意力的扩散模型在缺失模态特征完成中的应用 

**Authors**: Wei Zhang, Juan Chen, Yanbo J. Wang, En Zhu, Xuan Yang, Yiduo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05624)  

**Abstract**: Multimodal emotion and intent recognition is essential for automated human-computer interaction, It aims to analyze users' speech, text, and visual information to predict their emotions or intent. One of the significant challenges is that missing modalities due to sensor malfunctions or incomplete data. Traditional methods that attempt to reconstruct missing information often suffer from over-coupling and imprecise generation processes, leading to suboptimal outcomes. To address these issues, we introduce an Attention-based Diffusion model for Missing Modalities feature Completion (ADMC). Our framework independently trains feature extraction networks for each modality, preserving their unique characteristics and avoiding over-coupling. The Attention-based Diffusion Network (ADN) generates missing modality features that closely align with authentic multimodal distribution, enhancing performance across all missing-modality scenarios. Moreover, ADN's cross-modal generation offers improved recognition even in full-modality contexts. Our approach achieves state-of-the-art results on the IEMOCAP and MIntRec benchmarks, demonstrating its effectiveness in both missing and complete modality scenarios. 

**Abstract (ZH)**: 基于注意力的扩散模型在缺失模态特征完成中的应用：多模态情感和意图识别对于自动人机交互至关重要，旨在分析用户的语音、文本和视觉信息以预测其情感或意图。一个显著的挑战是由于传感器故障或数据不完整导致的缺失模态。传统的尝试重建缺失信息的方法通常会遭受过度耦合和不精确生成过程的困扰，导致次优结果。为了解决这些问题，我们提出了一种基于注意力的扩散模型（ADMC）用于缺失模态特征完成。我们的框架独立训练每个模态的特征提取网络，保留其独特特征，避免过度耦合。基于注意力的扩散网络（ADN）生成与真实多模态分布高度一致的缺失模态特征，提高所有缺失模态场景下的性能。此外，ADN的跨模态生成在全模态场景下也能提供更好的识别效果。我们的方法在IEMOCAP和MIntRec基准测试中取得了最先进的结果，证明了它在缺失和完整模态场景下的有效性。 

---
# Cultivating Multimodal Intelligence: Interpretive Reasoning and Agentic RAG Approaches to Dermatological Diagnosis 

**Title (ZH)**: 培养多模态智能：解释性推理与代理性RAG方法在皮肤病诊断中的应用 

**Authors**: Karishma Thakrar, Shreyas Basavatia, Akshay Daftardar  

**Link**: [PDF](https://arxiv.org/pdf/2507.05520)  

**Abstract**: The second edition of the 2025 ImageCLEF MEDIQA-MAGIC challenge, co-organized by researchers from Microsoft, Stanford University, and the Hospital Clinic of Barcelona, focuses on multimodal dermatology question answering and segmentation, using real-world patient queries and images. This work addresses the Closed Visual Question Answering (CVQA) task, where the goal is to select the correct answer to multiple-choice clinical questions based on both user-submitted images and accompanying symptom descriptions. The proposed approach combines three core components: (1) fine-tuning open-source multimodal models from the Qwen, Gemma, and LLaMA families on the competition dataset, (2) introducing a structured reasoning layer that reconciles and adjudicates between candidate model outputs, and (3) incorporating agentic retrieval-augmented generation (agentic RAG), which adds relevant information from the American Academy of Dermatology's symptom and condition database to fill in gaps in patient context. The team achieved second place with a submission that scored sixth, demonstrating competitive performance and high accuracy. Beyond competitive benchmarks, this research addresses a practical challenge in telemedicine: diagnostic decisions must often be made asynchronously, with limited input and with high accuracy and interpretability. By emulating the systematic reasoning patterns employed by dermatologists when evaluating skin conditions, this architecture provided a pathway toward more reliable automated diagnostic support systems. 

**Abstract (ZH)**: 2025 ImageCLEF MEDIQA-MAGIC挑战的第二版，由微软、斯坦福大学和巴塞罗那医院诊所的研究人员共同组织，专注于多模态皮肤病问答和分割，使用实际患者的查询和图像。该研究针对闭合视觉问答（CVQA）任务，目标是基于用户提交的图像和伴随的症状描述，选择正确的临床问题答案。所提出的方法结合了三个核心组件：（1）在竞赛数据集上微调来自Qwen、Gemma和LLaMA家族的开源多模态模型，（2）引入结构化推理层，以协调和裁决候选模型输出，以及（3）结合有能检索增强生成（有能RAG），从美国皮肤病学会的症状和状况数据库中添加相关的信息，以填补患者背景中的空白。团队凭借提交的第六名成绩获得了亚军，展示了竞争力和高准确性。除了竞争基准，这项研究还解决了远程医疗中的一个实际挑战：诊断决策经常需要在有限的输入和高准确性和可解释性的条件下异步做出。通过模拟皮肤状况评估中皮肤科医生采用的系统推理模式，该架构为更可靠的自动化诊断支持系统提供了一条路径。 

---
# Fine-Grained Vision-Language Modeling for Multimodal Training Assistants in Augmented Reality 

**Title (ZH)**: 增强现实环境中细粒度多模态训练助手的视觉-语言建模 

**Authors**: Haochen Huang, Jiahuan Pei, Mohammad Aliannejadi, Xin Sun, Moonisa Ahsan, Pablo Cesar, Chuang Yu, Zhaochun Ren, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05515)  

**Abstract**: Vision-language models (VLMs) are essential for enabling AI-powered smart assistants to interpret and reason in multimodal environments. However, their application in augmented reality (AR) training remains largely unexplored. In this work, we introduce a comprehensive dataset tailored for AR training, featuring systematized vision-language tasks, and evaluate nine state-of-the-art VLMs on it. Our results reveal that even advanced models, including GPT-4o, struggle with fine-grained assembly tasks, achieving a maximum F1 score of just 40.54% on state detection. These findings highlight the demand for enhanced datasets, benchmarks, and further research to improve fine-grained vision-language alignment. Beyond technical contributions, our work has broader social implications, particularly in empowering blind and visually impaired users with equitable access to AI-driven learning opportunities. We provide all related resources, including the dataset, source code, and evaluation results, to support the research community. 

**Abstract (ZH)**: 基于视觉-语言模型的增强现实训练数据集及评估：超越细粒度视觉-语言对齐的技术与社会影响 

---
# NeoBabel: A Multilingual Open Tower for Visual Generation 

**Title (ZH)**: NeoBabel: 多语言开放塔结构的视觉生成模型 

**Authors**: Mohammad Mahdi Derakhshani, Dheeraj Varghese, Marzieh Fadaee, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2507.06137)  

**Abstract**: Text-to-image generation advancements have been predominantly English-centric, creating barriers for non-English speakers and perpetuating digital inequities. While existing systems rely on translation pipelines, these introduce semantic drift, computational overhead, and cultural misalignment. We introduce NeoBabel, a novel multilingual image generation framework that sets a new Pareto frontier in performance, efficiency and inclusivity, supporting six languages: English, Chinese, Dutch, French, Hindi, and Persian. The model is trained using a combination of large-scale multilingual pretraining and high-resolution instruction tuning. To evaluate its capabilities, we expand two English-only benchmarks to multilingual equivalents: m-GenEval and m-DPG. NeoBabel achieves state-of-the-art multilingual performance while retaining strong English capability, scoring 0.75 on m-GenEval and 0.68 on m-DPG. Notably, it performs on par with leading models on English tasks while outperforming them by +0.11 and +0.09 on multilingual benchmarks, even though these models are built on multilingual base LLMs. This demonstrates the effectiveness of our targeted alignment training for preserving and extending crosslingual generalization. We further introduce two new metrics to rigorously assess multilingual alignment and robustness to code-mixed prompts. Notably, NeoBabel matches or exceeds English-only models while being 2-4x smaller. We release an open toolkit, including all code, model checkpoints, a curated dataset of 124M multilingual text-image pairs, and standardized multilingual evaluation protocols, to advance inclusive AI research. Our work demonstrates that multilingual capability is not a trade-off but a catalyst for improved robustness, efficiency, and cultural fidelity in generative AI. 

**Abstract (ZH)**: 多语言图像生成框架NeoBabel：性能、效率和包容性的新前沿 

---
# Enhancing Synthetic CT from CBCT via Multimodal Fusion and End-To-End Registration 

**Title (ZH)**: 通过多模态融合和端到端配准增强CBCT合成CT 

**Authors**: Maximilian Tschuchnig, Lukas Lamminger, Philipp Steininger, Michael Gadermayr  

**Link**: [PDF](https://arxiv.org/pdf/2507.06067)  

**Abstract**: Cone-Beam Computed Tomography (CBCT) is widely used for intraoperative imaging due to its rapid acquisition and low radiation dose. However, CBCT images typically suffer from artifacts and lower visual quality compared to conventional Computed Tomography (CT). A promising solution is synthetic CT (sCT) generation, where CBCT volumes are translated into the CT domain. In this work, we enhance sCT generation through multimodal learning by jointly leveraging intraoperative CBCT and preoperative CT data. To overcome the inherent misalignment between modalities, we introduce an end-to-end learnable registration module within the sCT pipeline. This model is evaluated on a controlled synthetic dataset, allowing precise manipulation of data quality and alignment parameters. Further, we validate its robustness and generalizability on two real-world clinical datasets. Experimental results demonstrate that integrating registration in multimodal sCT generation improves sCT quality, outperforming baseline multimodal methods in 79 out of 90 evaluation settings. Notably, the improvement is most significant in cases where CBCT quality is low and the preoperative CT is moderately misaligned. 

**Abstract (ZH)**: 基于锥束计算机断层成像的多模态合成CT生成研究 

---
# Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge 

**Title (ZH)**: 探索基于语义共现知识的部分多标签学习 

**Authors**: Xin Wu, Fei Teng, Yue Feng, Kaibo Shi, Zhuosheng Lin, Ji Zhang, James Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05992)  

**Abstract**: Partial multi-label learning aims to extract knowledge from incompletely annotated data, which includes known correct labels, known incorrect labels, and unknown labels. The core challenge lies in accurately identifying the ambiguous relationships between labels and instances. In this paper, we emphasize that matching co-occurrence patterns between labels and instances is key to addressing this challenge. To this end, we propose Semantic Co-occurrence Insight Network (SCINet), a novel and effective framework for partial multi-label learning. Specifically, SCINet introduces a bi-dominant prompter module, which leverages an off-the-shelf multimodal model to capture text-image correlations and enhance semantic alignment. To reinforce instance-label interdependencies, we develop a cross-modality fusion module that jointly models inter-label correlations, inter-instance relationships, and co-occurrence patterns across instance-label assignments. Moreover, we propose an intrinsic semantic augmentation strategy that enhances the model's understanding of intrinsic data semantics by applying diverse image transformations, thereby fostering a synergistic relationship between label confidence and sample difficulty. Extensive experiments on four widely-used benchmark datasets demonstrate that SCINet surpasses state-of-the-art methods. 

**Abstract (ZH)**: 部分多标签学习旨在从不完全注释的数据中提取知识，其中包括已知正确标签、已知错误标签和未知标签。核心挑战在于准确识别标签和实例之间的模糊关系。在本文中，我们强调匹配标签和实例之间的共现模式是解决这一挑战的关键。为此，我们提出了一种新的有效框架——语义共现洞察网络（SCINet），用于部分多标签学习。具体而言，SCINet 引入了一个双主导提示模块，该模块利用现成的多模态模型来捕捉文本-图像相关性并增强语义对齐。为了增强实例标签之间的相互依赖性，我们开发了一种跨模态融合模块，该模块联合建模标签间相关性、实例间关系以及实例标签分配中的共现模式。此外，我们提出了一个内在语义增强策略，通过应用多种图像变换来增强模型对内在数据语义的理解，从而促进标签置信度与样本难度之间的协同关系。在四个广泛使用的基准数据集上的广泛实验表明，SCINet 性能超越了现有最先进的方法。 

---
# Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model 

**Title (ZH)**: Llama Nemoretriever Colembed：性能最优的文本-图像检索模型 

**Authors**: Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Osmulski, Yauhen Babakhin, Zhiding Yu, Benedikt Schifferer, Even Oldridge  

**Link**: [PDF](https://arxiv.org/pdf/2507.05513)  

**Abstract**: Motivated by the growing demand for retrieval systems that operate across modalities, we introduce llama-nemoretriever-colembed, a unified text-image retrieval model that delivers state-of-the-art performance across multiple benchmarks. We release two model variants, 1B and 3B. The 3B model achieves state of the art performance, scoring NDCG@5 91.0 on ViDoRe V1 and 63.5 on ViDoRe V2, placing first on both leaderboards as of June 27, 2025.
Our approach leverages the NVIDIA Eagle2 Vision-Language model (VLM), modifies its architecture by replacing causal attention with bidirectional attention, and integrates a ColBERT-style late interaction mechanism to enable fine-grained multimodal retrieval in a shared embedding space. While this mechanism delivers superior retrieval accuracy, it introduces trade-offs in storage and efficiency. We provide a comprehensive analysis of these trade-offs. Additionally, we adopt a two-stage training strategy to enhance the model's retrieval capabilities. 

**Abstract (ZH)**: 受跨模态检索系统需求增长的驱动，我们介绍了llama-nemoretriever-colembed，这是一种统一的文本-图像检索模型，Across Modalities的检索模型，在多个基准测试中实现了最先进的性能。我们发布了两种模型变体，1B和3B。3B模型在ViDoRe V1和ViDoRe V2上分别取得了NDCG@5 91.0和63.5的最佳性能，截至2025年6月27日，在两个排行榜上都获得第一。 

---
