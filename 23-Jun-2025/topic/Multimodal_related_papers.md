# Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning 

**Title (ZH)**: 多模态融合学习在机器人任务规划中解决广义旅行商问题 

**Authors**: Jiaqi Chen, Mingfeng Fan, Xuefeng Zhang, Jingsong Liang, Yuhong Cao, Guohua Wu, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.16931)  

**Abstract**: Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios. 

**Abstract (ZH)**: 有效的多模态融合学习框架对于移动机器人任务规划至关重要，特别是在仓库检索和环境监控等应用中。我们提出了一种多模态融合学习（MMFL）框架，结合图和图像表示，以捕获问题的互补方面，并学习一种能够在实时生成高质量任务规划方案的策略。具体来说，我们首先引入了一种基于坐标的应用图像构建器，将GTSP实例转换为具有空间信息的表示。然后设计了一种自适应分辨率缩放策略，以增强不同问题规模的适应性，并开发了一种多模态融合模块，具有专门为几何和空间特征设计的瓶颈，以实现有效的融合。广泛实验表明，我们的MMFL方法在各种GTSP实例中均显著优于现有方法，同时保持了适用于实时机器人应用所需的计算效率。进一步的物理机器人测试验证了其在真实世界场景中的实际有效性。 

---
# IsoNet: Causal Analysis of Multimodal Transformers for Neuromuscular Gesture Classification 

**Title (ZH)**: IsoNet: 多模态变换器的神经肌肉手势分类因果分析 

**Authors**: Eion Tyacke, Kunal Gupta, Jay Patel, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16744)  

**Abstract**: Hand gestures are a primary output of the human motor system, yet the decoding of their neuromuscular signatures remains a bottleneck for basic neuroscience and assistive technologies such as prosthetics. Traditional human-machine interface pipelines rely on a single biosignal modality, but multimodal fusion can exploit complementary information from sensors. We systematically compare linear and attention-based fusion strategies across three architectures: a Multimodal MLP, a Multimodal Transformer, and a Hierarchical Transformer, evaluating performance on scenarios with unimodal and multimodal inputs. Experiments use two publicly available datasets: NinaPro DB2 (sEMG and accelerometer) and HD-sEMG 65-Gesture (high-density sEMG and force). Across both datasets, the Hierarchical Transformer with attention-based fusion consistently achieved the highest accuracy, surpassing the multimodal and best single-modality linear-fusion MLP baseline by over 10% on NinaPro DB2 and 3.7% on HD-sEMG. To investigate how modalities interact, we introduce an Isolation Network that selectively silences unimodal or cross-modal attention pathways, quantifying each group of token interactions' contribution to downstream decisions. Ablations reveal that cross-modal interactions contribute approximately 30% of the decision signal across transformer layers, highlighting the importance of attention-driven fusion in harnessing complementary modality information. Together, these findings reveal when and how multimodal fusion would enhance biosignal classification and also provides mechanistic insights of human muscle activities. The study would be beneficial in the design of sensor arrays for neurorobotic systems. 

**Abstract (ZH)**: 多模态融合在解码手部手势神经肌电信号中的应用：从单模态到基于注意力的多模态变压器架构的系统比较 

---
# Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens 

**Title (ZH)**: 机器心智想象：通过latent视觉标记赋能多模态推理 

**Authors**: Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17218)  

**Abstract**: Vision-language models (VLMs) excel at multimodal understanding, yet their text-only decoding forces them to verbalize visual reasoning, limiting performance on tasks that demand visual imagination. Recent attempts train VLMs to render explicit images, but the heavy image-generation pre-training often hinders the reasoning ability. Inspired by the way humans reason with mental imagery-the internal construction and manipulation of visual cues-we investigate whether VLMs can reason through interleaved multimodal trajectories without producing explicit images. To this end, we present a Machine Mental Imagery framework, dubbed as Mirage, which augments VLM decoding with latent visual tokens alongside ordinary text. Concretely, whenever the model chooses to ``think visually'', it recasts its hidden states as next tokens, thereby continuing a multimodal trajectory without generating pixel-level images. Begin by supervising the latent tokens through distillation from ground-truth image embeddings, we then switch to text-only supervision to make the latent trajectory align tightly with the task objective. A subsequent reinforcement learning stage further enhances the multimodal reasoning capability. Experiments on diverse benchmarks demonstrate that Mirage unlocks stronger multimodal reasoning without explicit image generation. 

**Abstract (ZH)**: Vision-语言模型在多模态理解方面表现出色，但在仅依赖文本解码时，被迫通过语言描述视觉推理，限制了其在要求视觉想象的任务中的性能。最近的研究尝试训练VLMs生成明确的图像，但重大的图像生成预训练往往阻碍了其推理能力。受人类通过内心视觉化（即内部构建和操作视觉线索）进行推理的方式启发，我们研究了VLMs是否可以在不生成明确图像的情况下通过交错的多模态轨迹进行推理。为此，我们提出了一种称为Mirage的机器内心视觉框架，该框架在常规文本中加入潜在的视觉标记以增强VLM解码。具体而言，每当模型选择“视觉思考”时，它会重新解释其隐藏状态为下一个标记，从而在不解码像素级图像的情况下继续多模态轨迹。通过从真实图像嵌入中蒸馏监督潜在标记，然后切换为仅文本监督，使潜在轨迹紧密对齐任务目标。随后的强化学习阶段进一步增强了多模态推理能力。在多种基准上的实验表明，Mirage在无需生成图像的情况下解锁了更强的多模态推理能力。 

---
# MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation 

**Title (ZH)**: MEXA: 向泛化多模态推理的动态多专家聚合研究 

**Authors**: Shoubin Yu, Yue Zhang, Ziyang Wang, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17113)  

**Abstract**: Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks. 

**Abstract (ZH)**: 结合预训练专家模型在可扩展的多模态推理中具有巨大潜力，但由于输入模态和任务复杂性的不断增加，建立统一框架仍然具有挑战性。为应对这一挑战，我们引入了MEXA，这是一种无需训练的框架，能够在多种专家模型之间进行模态和任务意识聚合，以在不同且独特的领域中实现有效的多模态推理。MEXA根据输入模态和任务特定的推理需求（即技能）动态选择专家模型。每个专家模型专门处理特定的模态任务对，并生成可解释的文本推理输出。MEXA然后使用大型推理模型（LRM）聚合和推理这些输出以生成最终答案。这种模块化设计允许在不同领域中灵活透明地进行多模态推理，而无需额外的训练开销。我们在视频推理、音频推理、3D理解以及医疗问答等多个多模态基准上广泛评估了该方法。MEXA在多种多模态基准上持续提供性能改进，突出了专家驱动的选择和聚合在多种多模态推理任务中的有效性与普适性。 

---
# With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You 

**Title (ZH)**: 基于有限数据的多模态对齐，让STRUCTURE引领你 

**Authors**: Fabian Gröger, Shuo Wen, Huyen Le, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2506.16895)  

**Abstract**: Multimodal models have demonstrated powerful capabilities in complex tasks requiring multimodal alignment including zero-shot classification and cross-modal retrieval. However, existing models typically rely on millions of paired multimodal samples, which are prohibitively expensive or infeasible to obtain in many domains. In this work, we explore the feasibility of building multimodal models with limited amount of paired data by aligning pretrained unimodal foundation models. We show that high-quality alignment is possible with as few as tens of thousands of paired samples$\unicode{x2013}$less than $1\%$ of the data typically used in the field. To achieve this, we introduce STRUCTURE, an effective regularization technique that preserves the neighborhood geometry of the latent space of unimodal encoders. Additionally, we show that aligning last layers is often suboptimal and demonstrate the benefits of aligning the layers with the highest representational similarity across modalities. These two components can be readily incorporated into existing alignment methods, yielding substantial gains across 24 zero-shot image classification and retrieval benchmarks, with average relative improvement of $51.6\%$ in classification and $91.8\%$ in retrieval tasks. Our results highlight the effectiveness and broad applicability of our framework for limited-sample multimodal learning and offer a promising path forward for resource-constrained domains. 

**Abstract (ZH)**: 基于有限配对数据构建多模态模型：结构化正则化方法及其应用 

---
# GeoGuess: Multimodal Reasoning based on Hierarchy of Visual Information in Street View 

**Title (ZH)**: GeoGuess：基于街道视图中视觉信息层次性的多模态推理 

**Authors**: Fenghua Cheng, Jinxiang Wang, Sen Wang, Zi Huang, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16633)  

**Abstract**: Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention as a benchmark for Artificial Intelligence (AI). Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Lack of reasoning on hierarchical visual clues at different levels of granularity, e.g., local details and global context, is of little discussion, despite its frequent involvement in real scenarios. To bridge the gap, we introduce a novel and challenging task for multimodal reasoning, namely GeoGuess. Given a street view image, the task is to identify its location and provide a detailed explanation. A system that succeeds in GeoGuess should be able to detect tiny visual clues, perceive the broader landscape, and associate with vast geographic knowledge. Therefore, GeoGuess would require the ability to reason between hierarchical visual information and geographic knowledge. In this work, we establish a benchmark for GeoGuess by introducing a specially curated dataset GeoExplain which consists of panoramas-geocoordinates-explanation tuples. Additionally, we present a multimodal and multilevel reasoning method, namely SightSense which can make prediction and generate comprehensive explanation based on hierarchy of visual information and external knowledge. Our analysis and experiments demonstrate their outstanding performance in GeoGuess. 

**Abstract (ZH)**: 多模态推理是跨不同数据模态理解、整合和推断信息的过程。它 recently 吸引了人工智能领域的广泛关注。尽管有多样化的任务来评估多模态推理能力，它们仍然存在局限性。缺乏在不同粒度级别上对层次视觉线索进行推理的讨论，尽管这些线索在现实场景中经常出现。为弥补这一差距，我们介绍了一个新的具有挑战性的多模态推理任务，名为GeoGuess。给定一张街景图像，任务是识别其位置并提供详细的解释。一个在GeoGuess中成功系统的应该能够检测细微的视觉线索、感知更广阔的景观，并关联大量的地理知识。因此，GeoGuess 将需要在层次视觉信息和地理知识之间进行推理的能力。在本文中，我们通过引入一个特别策划的数据集GeoExplain来建立GeoGuess的基准，该数据集包含全景图-地理坐标-解释三元组。此外，我们提出了一种多模态和多层次推理方法，名为SightSense，它可以基于视觉信息的层次结构和外部知识进行预测和生成全面的解释。我们的分析和实验证明了SightSense在GeoGuess中的出色表现。 

---
# Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding 

**Title (ZH)**: 基于视觉引导的片段化理解：提升RAG的多模态文档理解 

**Authors**: Vishesh Tripathi, Tanmay Odapally, Indraneel Das, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16035)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval and question answering, but traditional text-based chunking methods struggle with complex document structures, multi-page tables, embedded figures, and contextual dependencies across page boundaries. We present a novel multimodal document chunking approach that leverages Large Multimodal Models (LMMs) to process PDF documents in batches while maintaining semantic coherence and structural integrity. Our method processes documents in configurable page batches with cross-batch context preservation, enabling accurate handling of tables spanning multiple pages, embedded visual elements, and procedural content. We evaluate our approach on a curated dataset of PDF documents with manually crafted queries, demonstrating improvements in chunk quality and downstream RAG performance. Our vision-guided approach achieves better accuracy compared to traditional vanilla RAG systems, with qualitative analysis showing superior preservation of document structure and semantic coherence. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 系统通过利用大规模多模态模型对 PDF 文档进行批量处理，实现了信息检索和问答的革命性变化，但传统基于文本的片段化方法难以处理复杂的文档结构、跨页面的多页表格、嵌入的图形以及页面边界处的上下文依赖关系。我们提出了一种新颖的多模态文档片段化方法，利用大规模多模态模型在保持语义连贯性和结构完整性的同时对 PDF 文档进行批量处理。该方法以可配置的页面批处理方式进行处理，同时保留跨批处理的上下文，从而能够准确处理跨页面的表格、嵌入的视觉元素以及程序性内容。我们使用精心编写的 PDF 文档数据集和手动构建的问题对我们的方法进行了评估，展示了片段质量和下游 RAG 性能的提升。与传统的基线 RAG 系统相比，我们的基于视觉指导的方法在准确性上更胜一筹，定性分析表明其在保持文档结构和语义连贯性方面表现更优。 

---
# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion 

**Title (ZH)**: 双关语：基于多视图融合的鲁棒音频生成歌词检测 

**Authors**: Markus Frohmann, Gabriel Meseguer-Brocal, Markus Schedl, Elena V. Epure  

**Link**: [PDF](https://arxiv.org/pdf/2506.15981)  

**Abstract**: The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at this https URL. 

**Abstract (ZH)**: 基于AI的音乐生成工具的迅速发展正在颠覆音乐行业，同时也给艺术家、版权持有者和提供者带来了挑战。这 necessitates可靠的方法来检测此类AI生成的内容。然而，现有的检测器要么依赖于音频要么依赖于歌词，都面临着关键的实际限制：基于音频的检测器无法泛化到新的或未见过的生成器，并且容易受到音频干扰；基于歌词的方法需要格式干净且准确的歌词，在实践中难以获得。为克服这些限制，我们提出了一种新的、实际可行的方法：一个结合自动转录的歌词和捕捉与歌词相关音频特征的多模态模块化晚期融合管道。通过直接依赖于音频中的歌词方面，我们的方法增强了鲁棒性，减轻了对低级伪影的敏感性，并使得实际应用成为可能。实验表明，我们的方法DE-detect在性能上优于现有的基于歌词的检测器，并且对音频干扰具有更高的鲁棒性。因此，它提供了一种有效的、稳健的解决方案，用于实际场景中检测AI生成的音乐。我们的代码可以在此处访问： this https URL。 

---
# Heterogeneous-Modal Unsupervised Domain Adaptation via Latent Space Bridging 

**Title (ZH)**: 跨模态无监督领域适应通过潜在空间桥梁 

**Authors**: Jiawen Yang, Shuhao Chen, Yucong Duan, Ke Tang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15971)  

**Abstract**: Unsupervised domain adaptation (UDA) methods effectively bridge domain gaps but become struggled when the source and target domains belong to entirely distinct modalities. To address this limitation, we propose a novel setting called Heterogeneous-Modal Unsupervised Domain Adaptation (HMUDA), which enables knowledge transfer between completely different modalities by leveraging a bridge domain containing unlabeled samples from both modalities. To learn under the HMUDA setting, we propose Latent Space Bridging (LSB), a specialized framework designed for the semantic segmentation task. Specifically, LSB utilizes a dual-branch architecture, incorporating a feature consistency loss to align representations across modalities and a domain alignment loss to reduce discrepancies between class centroids across domains. Extensive experiments conducted on six benchmark datasets demonstrate that LSB achieves state-of-the-art performance. 

**Abstract (ZH)**: 异质模态无监督领域适应（HMUDA） 

---
