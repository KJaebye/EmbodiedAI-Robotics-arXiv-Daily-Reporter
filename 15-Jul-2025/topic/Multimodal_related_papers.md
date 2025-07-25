# Active Probing with Multimodal Predictions for Motion Planning 

**Title (ZH)**: 基于多模态预测的主动探测运动规划 

**Authors**: Darshan Gadginmath, Farhad Nawaz, Minjun Sung, Faizan M Tariq, Sangjae Bae, David Isele, Fabio Pasqualetti, Jovin Dsa  

**Link**: [PDF](https://arxiv.org/pdf/2507.09822)  

**Abstract**: Navigation in dynamic environments requires autonomous systems to reason about uncertainties in the behavior of other agents. In this paper, we introduce a unified framework that combines trajectory planning with multimodal predictions and active probing to enhance decision-making under uncertainty. We develop a novel risk metric that seamlessly integrates multimodal prediction uncertainties through mixture models. When these uncertainties follow a Gaussian mixture distribution, we prove that our risk metric admits a closed-form solution, and is always finite, thus ensuring analytical tractability. To reduce prediction ambiguity, we incorporate an active probing mechanism that strategically selects actions to improve its estimates of behavioral parameters of other agents, while simultaneously handling multimodal uncertainties. We extensively evaluate our framework in autonomous navigation scenarios using the MetaDrive simulation environment. Results demonstrate that our active probing approach successfully navigates complex traffic scenarios with uncertain predictions. Additionally, our framework shows robust performance across diverse traffic agent behavior models, indicating its broad applicability to real-world autonomous navigation challenges. Code and videos are available at this https URL. 

**Abstract (ZH)**: 动态环境中的导航需要自主系统对其他代理行为的不确定性进行推理。本文介绍了一种将轨迹规划与多模态预测及主动探测相结合的统一框架，以在不确定性下增强决策能力。我们开发了一种新的风险度量方法，通过混合模型无缝整合多模态预测的不确定性。当这些不确定性遵循高斯混合分布时，我们证明了我们的风险度量方法具有闭式解，并且始终保持有限，从而确保了分析上的可处理性。为了减少预测的不确定性，我们引入了一种主动探测机制，战略性地选择动作以改进对其他代理行为参数的估计，同时处理多模态不确定性。我们在MetaDrive模拟环境中广泛评估了我们的框架，结果显示我们的主动探测方法能够成功导航复杂的交通场景中的不确定预测。此外，我们的框架在多种交通代理行为模型中表现出稳健的性能，表明其适用于实际的自主导航挑战。代码和视频可通过此链接获取。 

---
# Multimodal HD Mapping for Intersections by Intelligent Roadside Units 

**Title (ZH)**: 由智能路边单元实现的交差点多模态高清映射 

**Authors**: Zhongzhang Chen, Miao Fan, Shengtong Xu, Mengmeng Yang, Kun Jiang, Xiangzeng Liu, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.08903)  

**Abstract**: High-definition (HD) semantic mapping of complex intersections poses significant challenges for traditional vehicle-based approaches due to occlusions and limited perspectives. This paper introduces a novel camera-LiDAR fusion framework that leverages elevated intelligent roadside units (IRUs). Additionally, we present RS-seq, a comprehensive dataset developed through the systematic enhancement and annotation of the V2X-Seq dataset. RS-seq includes precisely labelled camera imagery and LiDAR point clouds collected from roadside installations, along with vectorized maps for seven intersections annotated with detailed features such as lane dividers, pedestrian crossings, and stop lines. This dataset facilitates the systematic investigation of cross-modal complementarity for HD map generation using IRU data. The proposed fusion framework employs a two-stage process that integrates modality-specific feature extraction and cross-modal semantic integration, capitalizing on camera high-resolution texture and precise geometric data from LiDAR. Quantitative evaluations using the RS-seq dataset demonstrate that our multimodal approach consistently surpasses unimodal methods. Specifically, compared to unimodal baselines evaluated on the RS-seq dataset, the multimodal approach improves the mean Intersection-over-Union (mIoU) for semantic segmentation by 4\% over the image-only results and 18\% over the point cloud-only results. This study establishes a baseline methodology for IRU-based HD semantic mapping and provides a valuable dataset for future research in infrastructure-assisted autonomous driving systems. 

**Abstract (ZH)**: 基于高架智能路边单元的高分辨率语义地图构建：复杂交叉口的摄像机-LiDAR融合框架及RS-seq数据集 

---
# (Almost) Free Modality Stitching of Foundation Models 

**Title (ZH)**: 近乎免费的基础模型模态拼接 

**Authors**: Jaisidh Singh, Diganta Misra, Boris Knyazev, Antonio Orvieto  

**Link**: [PDF](https://arxiv.org/pdf/2507.10015)  

**Abstract**: Foundation multi-modal models are often designed by stitching of multiple existing pretrained uni-modal models: for example, an image classifier with an autoregressive text model. This stitching process is performed by training a connector module that aims to align the representation-representation or representation-input spaces of these uni-modal models. However, given the complexity of training such connectors on large scale web-based datasets coupled with the ever-increasing number of available pretrained uni-modal models, the task of uni-modal models selection and subsequent connector module training becomes computationally demanding. To address this under-studied critical problem, we propose Hypernetwork Model Alignment (Hyma), a novel all-in-one solution for optimal uni-modal model selection and connector training by leveraging hypernetworks. Specifically, our framework utilizes the parameter prediction capability of a hypernetwork to obtain jointly trained connector modules for $N \times M$ combinations of uni-modal models. In our experiments, Hyma reduces the optimal uni-modal model pair search cost by $10\times$ (averaged across all experiments), while matching the ranking and trained connector performance obtained via grid search across a suite of diverse multi-modal benchmarks. 

**Abstract (ZH)**: 基于超网络的模态模型联盟（Hyma）：一种新的统一解决方案，用于高效选择最优单模模型并训练连接器模块 

---
# A Brain Tumor Segmentation Method Based on CLIP and 3D U-Net with Cross-Modal Semantic Guidance and Multi-Level Feature Fusion 

**Title (ZH)**: 基于CLIP和3D U-Net的跨模态语义指导及多级特征融合的脑肿瘤分割方法 

**Authors**: Mingda Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09966)  

**Abstract**: Precise segmentation of brain tumors from magnetic resonance imaging (MRI) is essential for neuro-oncology diagnosis and treatment planning. Despite advances in deep learning methods, automatic segmentation remains challenging due to tumor morphological heterogeneity and complex three-dimensional spatial relationships. Current techniques primarily rely on visual features extracted from MRI sequences while underutilizing semantic knowledge embedded in medical reports. This research presents a multi-level fusion architecture that integrates pixel-level, feature-level, and semantic-level information, facilitating comprehensive processing from low-level data to high-level concepts. The semantic-level fusion pathway combines the semantic understanding capabilities of Contrastive Language-Image Pre-training (CLIP) models with the spatial feature extraction advantages of 3D U-Net through three mechanisms: 3D-2D semantic bridging, cross-modal semantic guidance, and semantic-based attention mechanisms. Experimental validation on the BraTS 2020 dataset demonstrates that the proposed model achieves an overall Dice coefficient of 0.8567, representing a 4.8% improvement compared to traditional 3D U-Net, with a 7.3% Dice coefficient increase in the clinically important enhancing tumor (ET) region. 

**Abstract (ZH)**: 从磁共振成像中精确分割脑肿瘤是神经肿瘤诊断和治疗规划不可或缺的部分。尽管深度学习方法取得了进展，但由于肿瘤形态异质性和复杂的三维空间关系，自动分割仍然具有挑战性。当前技术主要依赖于从MRI序列中提取的视觉特征，而未能充分利用医学报告中蕴含的语义知识。本研究提出了一种多级融合架构，该架构整合了像素级、特征级和语义级信息，促进了从低级数据到高级概念的全面处理。语义级融合路径将对比语言-图像预训练（CLIP）模型的语义理解能力和3D U-Net的空间特征提取优势结合起来，通过三种机制实现了这一融合：三维-二维语义桥梁、跨模态语义指导和基于语义的注意力机制。在BraTS 2020数据集上的实验验证表明，所提出模型的整体Dice系数为0.8567，相较于传统的3D U-Net提高了4.8%，在临床上重要的增强肿瘤（ET）区域的Dice系数提高了7.3%。 

---
# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends 

**Title (ZH)**: 基于MLLM的富视觉文档理解综述：方法、挑战及新兴趋势 

**Authors**: Yihao Ding, Siwen Luo, Yue Dai, Yanbei Jiang, Zechuan Li, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.09861)  

**Abstract**: Visually-Rich Document Understanding (VRDU) has emerged as a critical field, driven by the need to automatically process documents containing complex visual, textual, and layout information. Recently, Multimodal Large Language Models (MLLMs) have shown remarkable potential in this domain, leveraging both Optical Character Recognition (OCR)-dependent and OCR-free frameworks to extract and interpret information in document images. This survey reviews recent advancements in MLLM-based VRDU, highlighting three core components: (1) methods for encoding and fusing textual, visual, and layout features; (2) training paradigms, including pretraining strategies, instruction-response tuning, and the trainability of different model modules; and (3) datasets utilized for pretraining, instruction-tuning, and supervised fine-tuning. Finally, we discuss the challenges and opportunities in this evolving field and propose future directions to advance the efficiency, generalizability, and robustness of VRDU systems. 

**Abstract (ZH)**: 富视觉文档理解（VRDU）已成为一项关键领域，由自动处理包含复杂视觉、文本和布局信息的文档的需求推动。近年来，多模态大型语言模型（MLLMs）在该领域展现出了显著潜力，利用依赖光学字符识别（OCR）和非OCR框架来提取和解析文档图像中的信息。本文综述了基于MLLM的VRDU的最新进展，强调了三个核心组成部分：（1）文本、视觉和布局特征的编码与融合方法；（2）训练范式，包括预训练策略、指令-响应调优以及不同模型模块的可训练性；以及（3）用于预训练、指令调优和监督微调的数据集。最后，我们讨论了该领域面临的挑战和机遇，并提出了推进VRDU系统的效率、泛化能力和鲁棒性的未来方向。 

---
# KEN: Knowledge Augmentation and Emotion Guidance Network for Multimodal Fake News Detection 

**Title (ZH)**: KEN: 基于知识增强和情绪指导的多模态假新闻检测网络 

**Authors**: Peican Zhu, Yubo Jing, Le Cheng, Keke Tang, Yangming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09647)  

**Abstract**: In recent years, the rampant spread of misinformation on social media has made accurate detection of multimodal fake news a critical research focus. However, previous research has not adequately understood the semantics of images, and models struggle to discern news authenticity with limited textual information. Meanwhile, treating all emotional types of news uniformly without tailored approaches further leads to performance degradation. Therefore, we propose a novel Knowledge Augmentation and Emotion Guidance Network (KEN). On the one hand, we effectively leverage LVLM's powerful semantic understanding and extensive world knowledge. For images, the generated captions provide a comprehensive understanding of image content and scenes, while for text, the retrieved evidence helps break the information silos caused by the closed and limited text and context. On the other hand, we consider inter-class differences between different emotional types of news through balanced learning, achieving fine-grained modeling of the relationship between emotional types and authenticity. Extensive experiments on two real-world datasets demonstrate the superiority of our KEN. 

**Abstract (ZH)**: 近年来，社交媒体上虚假信息的盛行使得多模态假新闻的准确检测成为关键研究方向。然而，先前的研究未能充分理解图像的语义，模型在仅靠有限的文字信息时难以辨别新闻的真实性。同时，不加区分地处理不同情感类型的新闻导致性能下降。因此，我们提出了一种新型的知识增强和情感引导网络（KEN）。一方面，我们有效地利用了LVLM强大的语义理解和广泛的世界知识。对于图像，生成的字幕提供了对图像内容和场景的全面理解，而对于文本，则检索到的证据有助于打破由封闭和有限的文字和上下文造成的信息孤岛。另一方面，我们通过平衡学习考虑了不同类型新闻之间的情感差异，实现了对情感类型与真实性之间关系的细致建模。在两个真实世界数据集上的广泛实验表明了KEN的优点。 

---
# MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models 

**Title (ZH)**: MENTOR: 效率优先的多模态条件调谐用于自回归视觉生成模型 

**Authors**: Haozhe Zhao, Zefan Cai, Shuzheng Si, Liang Chen, Jiuxiang Gu, Wen Xiao, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09574)  

**Abstract**: Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods. Dataset, code, and models are available at: this https URL 

**Abstract (ZH)**: recent文本到图像模型生成高质量结果但仍难以实现精确的视觉控制、平衡多模态输入，并且需要大量的训练来生成复杂的多模态图像。为了解决这些局限性，我们提出了一种名为MENTOR的新颖自回归(AR)框架，用于高效多模态条件调节以实现自回归多模态图像生成。MENTOR结合了自回归图像生成器和两阶段训练范式，能够在不依赖辅助适配器或交叉注意力模块的情况下，实现多模态输入与图像输出的细粒度、token级对齐。两阶段训练包括：(1) 多模态对齐阶段，建立稳健的像素级和语义级对齐，随后是 (2) 多模态指令调节阶段，平衡多模态输入的集成并增强生成可控性。尽管模型规模 modest、基组件 suboptimal且训练资源有限，MENTOR 在 DreamBench++基准测试中仍表现出色，超越了竞争对手的基础模型在概念保留和提示遵循方面的性能。此外，我们的方法还实现了优于扩散模型的优点，包括更高的图像重建保真度、更广泛的任务适应性和改进的训练效率。数据集、代码和模型可在以下网址获取：this https URL。 

---
# VDInstruct: Zero-Shot Key Information Extraction via Content-Aware Vision Tokenization 

**Title (ZH)**: VDInstruct: 基于内容aware视觉词元化的内容零样本关键信息提取 

**Authors**: Son Nguyen, Giang Nguyen, Hung Dao, Thao Do, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.09531)  

**Abstract**: Key Information Extraction (KIE) underpins the understanding of visual documents (e.g., receipts and contracts) by extracting precise semantic content and accurately capturing spatial structure. Yet existing multimodal large language models (MLLMs) often perform poorly on dense documents and rely on vision tokenization approaches that scale with image size, leading to redundant computation and memory inefficiency. To address these challenges, we introduce VDInstruct, an MLLM that separates spatial region detection from semantic feature extraction. Central to our model is a content-aware tokenization strategy: rather than fragmenting the entire image uniformly, it generates tokens in proportion to document complexity, preserving critical structure while eliminating wasted tokens. Leveraging a three-stage training paradigm, our model achieves state-of-the-art (SOTA) results on KIE benchmarks, matching or exceeding the accuracy of leading approaches while reducing the number of image tokens by roughly 3.6x. In zero-shot evaluations, VDInstruct surpasses strong baselines-such as DocOwl 1.5-by +5.5 F1 points, highlighting its robustness to unseen documents. These findings show that content-aware tokenization combined with explicit layout modeling offers a promising direction forward for document understanding. Data, source code, and model weights will be made publicly available. 

**Abstract (ZH)**: 基于内容感知的区域检测和语义特征提取分离的视觉文档关键信息提取 

---
# ViSP: A PPO-Driven Framework for Sarcasm Generation with Contrastive Learning 

**Title (ZH)**: ViSP：一种基于PPO的对比学习驱动的 sarcasm生成框架 

**Authors**: Changli Wang, Rui Wu, Fang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09482)  

**Abstract**: Human emotions are complex, with sarcasm being a subtle and distinctive form. Despite progress in sarcasm research, sarcasm generation remains underexplored, primarily due to the overreliance on textual modalities and the neglect of visual cues, as well as the mismatch between image content and sarcastic intent in existing datasets. In this paper, we introduce M2SaG, a multimodal sarcasm generation dataset with 4,970 samples, each containing an image, a sarcastic text, and a sarcasm target. To benchmark M2SaG, we propose ViSP, a generation framework that integrates Proximal Policy Optimization (PPO) and contrastive learning. PPO utilizes reward scores from DIP to steer the generation of sarcastic texts, while contrastive learning encourages the model to favor outputs with higher reward scores. These strategies improve overall generation quality and produce texts with more pronounced sarcastic intent. We evaluate ViSP across five metric sets and find it surpasses all baselines, including large language models, underscoring their limitations in sarcasm generation. Furthermore, we analyze the distributions of Sarcasm Scores and Factual Incongruity for both M2SaG and the texts generated by ViSP. The generated texts exhibit higher mean Sarcasm Scores (0.898 vs. 0.770) and Factual Incongruity (0.768 vs. 0.739), demonstrating that ViSP produces higher-quality sarcastic content than the original dataset. % The dataset and code will be publicly available. Our dataset and code will be released at \textit{this https URL}. 

**Abstract (ZH)**: 多模态讽刺生成数据集M2SaG及基准模型ViSP 

---
# From Classical Machine Learning to Emerging Foundation Models: Review on Multimodal Data Integration for Cancer Research 

**Title (ZH)**: 从经典机器学习到新兴基础模型：多模态数据集成在癌症研究中的综述 

**Authors**: Amgad Muneer, Muhammad Waqas, Maliazurina B Saad, Eman Showkatian, Rukhmini Bandyopadhyay, Hui Xu, Wentao Li, Joe Y Chang, Zhongxing Liao, Cara Haymaker, Luisa Solis Soto, Carol C Wu, Natalie I Vokes, Xiuning Le, Lauren A Byers, Don L Gibbons, John V Heymach, Jianjun Zhang, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09028)  

**Abstract**: Cancer research is increasingly driven by the integration of diverse data modalities, spanning from genomics and proteomics to imaging and clinical factors. However, extracting actionable insights from these vast and heterogeneous datasets remains a key challenge. The rise of foundation models (FMs) -- large deep-learning models pretrained on extensive amounts of data serving as a backbone for a wide range of downstream tasks -- offers new avenues for discovering biomarkers, improving diagnosis, and personalizing treatment. This paper presents a comprehensive review of widely adopted integration strategies of multimodal data to assist advance the computational approaches for data-driven discoveries in oncology. We examine emerging trends in machine learning (ML) and deep learning (DL), including methodological frameworks, validation protocols, and open-source resources targeting cancer subtype classification, biomarker discovery, treatment guidance, and outcome prediction. This study also comprehensively covers the shift from traditional ML to FMs for multimodal integration. We present a holistic view of recent FMs advancements and challenges faced during the integration of multi-omics with advanced imaging data. We identify the state-of-the-art FMs, publicly available multi-modal repositories, and advanced tools and methods for data integration. We argue that current state-of-the-art integrative methods provide the essential groundwork for developing the next generation of large-scale, pre-trained models poised to further revolutionize oncology. To the best of our knowledge, this is the first review to systematically map the transition from conventional ML to advanced FM for multimodal data integration in oncology, while also framing these developments as foundational for the forthcoming era of large-scale AI models in cancer research. 

**Abstract (ZH)**: 癌症研究 increasingly driven by the integration of diverse data modalities, spanning from genomics and proteomics to imaging and clinical factors. However, extracting actionable insights from these vast and heterogeneous datasets remains a key challenge. The rise of foundation models (FMs) — large deep-learning models pretrained on extensive amounts of data serving as a backbone for a wide range of downstream tasks — offers new avenues for discovering biomarkers, improving diagnosis, and personalizing treatment. This paper presents a comprehensive review of widely adopted integration strategies of multimodal data to assist in advancing computational approaches for data-driven discoveries in oncology. We examine emerging trends in machine learning (ML) and deep learning (DL), including methodological frameworks, validation protocols, and open-source resources targeting cancer subtype classification, biomarker discovery, treatment guidance, and outcome prediction. This study also comprehensively covers the shift from traditional ML to FMs for multimodal integration. We present a holistic view of recent FM advancements and challenges faced during the integration of multi-omics with advanced imaging data. We identify the state-of-the-art FMs, publicly available multi-modal repositories, and advanced tools and methods for data integration. We argue that current state-of-the-art integrative methods provide the essential groundwork for developing the next generation of large-scale, pre-trained models poised to further revolutionize oncology. To the best of our knowledge, this is the first review to systematically map the transition from conventional ML to advanced FMs for multimodal data integration in oncology, while also framing these developments as foundational for the forthcoming era of large-scale AI models in cancer research. 

---
