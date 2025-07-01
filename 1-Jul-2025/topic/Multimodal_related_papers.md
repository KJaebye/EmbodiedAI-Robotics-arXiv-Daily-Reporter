# Single-Frame Point-Pixel Registration via Supervised Cross-Modal Feature Matching 

**Title (ZH)**: 单帧点像素注册via监督跨模态特征匹配 

**Authors**: Yu Han, Zhiwei Huang, Yanting Zhang, Fangjun Ding, Shen Cai, Rui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22784)  

**Abstract**: Point-pixel registration between LiDAR point clouds and camera images is a fundamental yet challenging task in autonomous driving and robotic perception. A key difficulty lies in the modality gap between unstructured point clouds and structured images, especially under sparse single-frame LiDAR settings. Existing methods typically extract features separately from point clouds and images, then rely on hand-crafted or learned matching strategies. This separate encoding fails to bridge the modality gap effectively, and more critically, these methods struggle with the sparsity and noise of single-frame LiDAR, often requiring point cloud accumulation or additional priors to improve reliability. Inspired by recent progress in detector-free matching paradigms (e.g. MatchAnything), we revisit the projection-based approach and introduce the detector-free framework for direct point-pixel matching between LiDAR and camera views. Specifically, we project the LiDAR intensity map into a 2D view from the LiDAR perspective and feed it into an attention-based detector-free matching network, enabling cross-modal correspondence estimation without relying on multi-frame accumulation. To further enhance matching reliability, we introduce a repeatability scoring mechanism that acts as a soft visibility prior. This guides the network to suppress unreliable matches in regions with low intensity variation, improving robustness under sparse input. Extensive experiments on KITTI, nuScenes, and MIAS-LCEC-TF70 benchmarks demonstrate that our method achieves state-of-the-art performance, outperforming prior approaches on nuScenes (even those relying on accumulated point clouds), despite using only single-frame LiDAR. 

**Abstract (ZH)**: 激光雷达点云与相机图像之间的点像素注册是自主驾驶和机器人感知领域中的一个基础但具有挑战性的任务。现有方法通常分别从点云和图像中提取特征，然后依赖手工设计或学习的匹配策略。这种单独编码方式无法有效弥合模态差异，并且更为关键的是，这些方法在处理单帧稀疏激光雷达的稀疏性和噪声时表现不佳，常需点云累积或额外先验以提高可靠性。受最近在无检测器匹配范式（如MatchAnything）进展的启发，我们重新审视基于投影的方法，并引入一种无检测器框架，用于激光雷达和相机视图之间的直接点像素匹配。具体而言，我们将激光雷达强度图从激光雷达视角投影到2D视图中，并将其输入基于注意力机制的无检测器匹配网络，从而在不依赖多帧累积的情况下实现模态间的对应关系估计。为进一步提高匹配的可靠性，我们引入了一种重复性评分机制，作为软的可見性先验。该机制引导网络在低强度变化区域抑制不可靠的匹配，在稀疏输入下提高鲁棒性。在KITTI、nuScenes和MIAS-LCEC-TF70基准测试上的广泛实验表明，我们的方法达到了最先进的性能，在nuScenes上优于先前方法（即使那些依赖于累积点云的方法），尽管仅使用单帧激光雷达。 

---
# MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning 

**Title (ZH)**: MARBLE：多模态空间推理与规划的困难基准 

**Authors**: Yulun Jiang, Yekun Chai, Maria Brbić, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2506.22992)  

**Abstract**: The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE -- all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps. 

**Abstract (ZH)**: 多模态推理基准MARBLE：检验多模态语言模型在复杂多模态问题和环境中的逐步推理能力 

---
# Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking 

**Title (ZH)**: Mamba-FETrack V2: 重新审视基于帧事件的视觉对象跟踪的状态空间模型 

**Authors**: Shiao Wang, Ju Huang, Qingchuan Ma, Jinfeng Gao, Chunyi Xu, Xiao Wang, Lan Chen, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23783)  

**Abstract**: Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework. The source code and pre-trained models will be released on this https URL 

**Abstract (ZH)**: 结合传统RGB摄像头与生物启发式事件摄像头进行鲁棒目标跟踪的研究近年来引起了越来越多的关注。然而，大多数现有的多模态跟踪算法高度依赖于复杂性的Vision Transformer架构进行特征提取和跨模态融合，这不仅带来了巨大的计算开销，也限制了跨模态交互的有效性。在本文中，我们提出了一种基于线性复杂度Vision Mamba网络的高效RGB-事件目标跟踪框架，称为Mamba-FETrack V2。具体来说，我们首先设计了一个轻量级的提示生成器，该生成器利用每种模态的嵌入特征以及共享的提示池，动态生成模态特定的学习提示向量。这些提示与模态特定的嵌入特征一起传递给基于Vision Mamba的FEMamba骨干网络，该骨干网络以统一的方式实现提示引导的特征提取、跨模态交互和融合。最后，融合的表示传递到跟踪头以实现准确的目标定位。在包括短期COESOT数据集和长期数据集FE108和FELT V2的多个RGB-事件跟踪基准上的广泛实验评估证明了所提出跟踪框架的优越性能和效率。源代码和预训练模型将发布在该网址：<https://>。 

---
# Unified Multimodal Understanding via Byte-Pair Visual Encoding 

**Title (ZH)**: 统一多模态理解 via 字节对视觉编码 

**Authors**: Wanpeng Zhang, Yicheng Feng, Hao Luo, Yijiang Li, Zihao Yue, Sipeng Zheng, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23639)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in vision-language understanding, yet effectively aligning different modalities remains a fundamental challenge. We present a framework that unifies multimodal understanding by applying byte-pair encoding to visual tokens. Unlike conventional approaches that rely on modality-specific encoders, our method directly incorporates structural information into visual tokens, mirroring successful tokenization strategies in text-only language models. We introduce a priority-guided encoding scheme that considers both frequency and spatial consistency, coupled with a multi-stage training procedure based on curriculum-driven data composition. These enhancements enable the transformer model to better capture cross-modal relationships and reason with visual information. Comprehensive experiments demonstrate improved performance across diverse vision-language tasks. By bridging the gap between visual and textual representations, our approach contributes to the advancement of more capable and efficient multimodal foundation models. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在视觉语言理解方面取得了显著进展，但不同模态的有效对齐依然是一个基本挑战。我们提出了一种框架，通过字元对编码统一多模态理解，而不像传统方法依赖于特定模态的编码器，我们的方法直接将结构信息融入视觉令牌，模仿仅文本语言模型中成功的分词策略。我们引入了一种优先级导向的编码方案，考虑频率和空间一致性，并结合基于递增式数据合成的多阶段训练过程。这些增强使得变压器模型更好地捕捉跨模态关系并处理视觉信息。全面的实验表明，我们的方法在多种视觉语言任务中表现出更好的性能。通过弥合视觉和文本表示之间的差距，我们的方法促进了更强大和高效的多模态基础模型的发展。 

---
# Why Settle for One? Text-to-ImageSet Generation and Evaluation 

**Title (ZH)**: 为什么只满足于一种？从文本到图像集的生成与评估 

**Authors**: Chengyou Jia, Xin Shen, Zhuohang Dang, Zhuohang Dang, Changliang Xia, Weijia Wu, Xinyu Zhang, Hangwei Qian, Ivor W.Tsang, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23275)  

**Abstract**: Despite remarkable progress in Text-to-Image models, many real-world applications require generating coherent image sets with diverse consistency requirements. Existing consistent methods often focus on a specific domain with specific aspects of consistency, which significantly constrains their generalizability to broader applications. In this paper, we propose a more challenging problem, Text-to-ImageSet (T2IS) generation, which aims to generate sets of images that meet various consistency requirements based on user instructions. To systematically study this problem, we first introduce $\textbf{T2IS-Bench}$ with 596 diverse instructions across 26 subcategories, providing comprehensive coverage for T2IS generation. Building on this, we propose $\textbf{T2IS-Eval}$, an evaluation framework that transforms user instructions into multifaceted assessment criteria and employs effective evaluators to adaptively assess consistency fulfillment between criteria and generated sets. Subsequently, we propose $\textbf{AutoT2IS}$, a training-free framework that maximally leverages pretrained Diffusion Transformers' in-context capabilities to harmonize visual elements to satisfy both image-level prompt alignment and set-level visual consistency. Extensive experiments on T2IS-Bench reveal that diverse consistency challenges all existing methods, while our AutoT2IS significantly outperforms current generalized and even specialized approaches. Our method also demonstrates the ability to enable numerous underexplored real-world applications, confirming its substantial practical value. Visit our project in this https URL. 

**Abstract (ZH)**: 尽管文本到图像模型取得了显著进展，许多实际应用仍需生成具有多种一致要求的协调图像集。现有的一致方法往往专注于特定领域和特定的一致性方面，这极大地限制了它们在更广泛应用中的通用性。本文提出了更具挑战性的问题——文本到图像集（T2IS）生成，旨在根据用户指令生成满足各种一致性要求的图像集。为了系统地研究这一问题，我们首先介绍了包含26个子类别中的596种多样化指令的T2IS-Bench，为T2IS生成提供了全面覆盖。在此基础上，我们提出了T2IS-Eval评估框架，将用户指令转化为多方面的评估标准，并采用有效的评估工具适应性地评估生成集与标准之间的一致性履行情况。随后，我们提出了无需训练的AutoT2IS框架，最大限度地利用预训练扩散变压器的上下文能力，使视觉元素协调以满足图像级提示对齐和图像集视觉一致性要求。在T2IS-Bench上的广泛实验表明，各种一致性挑战使现有所有方法失效，而我们的AutoT2IS在当前泛化和专门化方法中表现显著优越。我们的方法还展示了能够推动许多未充分探索的实际应用的能力，证实了其重要的实际价值。访问我们的项目请访问 this https URL。 

---
# CRISP-SAM2: SAM2 with Cross-Modal Interaction and Semantic Prompting for Multi-Organ Segmentation 

**Title (ZH)**: CRISP-SAM2: 具有跨模态交互和语义提示的多器官分割SAM2 

**Authors**: Xinlei Yu, Chanmiao Wang, Hui Jin, Ahmed Elazab, Gangyong Jia, Xiang Wan, Changqing Zou, Ruiquan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.23121)  

**Abstract**: Multi-organ medical segmentation is a crucial component of medical image processing, essential for doctors to make accurate diagnoses and develop effective treatment plans. Despite significant progress in this field, current multi-organ segmentation models often suffer from inaccurate details, dependence on geometric prompts and loss of spatial information. Addressing these challenges, we introduce a novel model named CRISP-SAM2 with CRoss-modal Interaction and Semantic Prompting based on SAM2. This model represents a promising approach to multi-organ medical segmentation guided by textual descriptions of organs. Our method begins by converting visual and textual inputs into cross-modal contextualized semantics using a progressive cross-attention interaction mechanism. These semantics are then injected into the image encoder to enhance the detailed understanding of visual information. To eliminate reliance on geometric prompts, we use a semantic prompting strategy, replacing the original prompt encoder to sharpen the perception of challenging targets. In addition, a similarity-sorting self-updating strategy for memory and a mask-refining process is applied to further adapt to medical imaging and enhance localized details. Comparative experiments conducted on seven public datasets indicate that CRISP-SAM2 outperforms existing models. Extensive analysis also demonstrates the effectiveness of our method, thereby confirming its superior performance, especially in addressing the limitations mentioned earlier. Our code is available at: this https URL\this http URL. 

**Abstract (ZH)**: 多器官医学分割是医学图像处理中的 crucial 组件，对于医生进行准确诊断和制定有效的治疗计划至关重要。尽管在该领域取得了显著进展，当前的多器官分割模型仍然存在细节不准确、依赖几何提示以及空间信息丢失等问题。为应对这些挑战，我们提出了一种基于 SAM2 的新型模型 CRISP-SAM2，该模型通过跨模态交互和语义提示，为基于器官文本描述的多器官医学分割提供了有 promise 的方法。该方法首先通过渐进的跨注意力交互机制将视觉和文本输入转化为跨模态上下文语义，然后将这些语义注入图像编码器以增强对视觉信息的理解。为了消除对几何提示的依赖，我们采用了语义提示策略，替代原始提示编码器以增强对困难目标的感知。此外，我们还应用了一种用于记忆的相似性排序自我更新策略和掩码细化过程，以便更好地适应医学成像并增强局部细节。在七个公开数据集上进行的对比实验表明，CRISP-SAM2 在多器官医学分割任务中优于现有模型。广泛的分析进一步证明了我们方法的有效性，从而确认其优越性能，特别是在解决上述提到的局限性方面。我们的代码可在：this https URL this http URL 获取。 

---
# MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings 

**Title (ZH)**: MoCa: 模态aware 连续预训练生成更好的双向多模态嵌入 

**Authors**: Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan Yang, Furu Wei, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23115)  

**Abstract**: Multimodal embedding models, built upon causal Vision Language Models (VLMs), have shown promise in various tasks. However, current approaches face three key limitations: the use of causal attention in VLM backbones is suboptimal for embedding tasks; scalability issues due to reliance on high-quality labeled paired data for contrastive learning; and limited diversity in training objectives and data. To address these issues, we propose MoCa, a two-stage framework for transforming pre-trained VLMs into effective bidirectional multimodal embedding models. The first stage, Modality-aware Continual Pre-training, introduces a joint reconstruction objective that simultaneously denoises interleaved text and image inputs, enhancing bidirectional context-aware reasoning. The second stage, Heterogeneous Contrastive Fine-tuning, leverages diverse, semantically rich multimodal data beyond simple image-caption pairs to enhance generalization and alignment. Our method addresses the stated limitations by introducing bidirectional attention through continual pre-training, scaling effectively with massive unlabeled datasets via joint reconstruction objectives, and utilizing diverse multimodal data for enhanced representation robustness. Experiments demonstrate that MoCa consistently improves performance across MMEB and ViDoRe-v2 benchmarks, achieving new state-of-the-art results, and exhibits strong scalability with both model size and training data on MMEB. 

**Abstract (ZH)**: 基于因果视觉语言模型的多模态嵌入模型在各种任务中显示出了潜力。然而，当前的方法面临三个关键技术限制：视觉语言模型背部中使用因果注意机制对于嵌入任务不够优化；由于依赖高质量的配对标注数据进行对比学习而导致的可扩展性问题；以及在训练目标和数据方面存在的局限性。为了解决这些问题，我们提出MoCa，一种两阶段框架，用于将预训练的视觉语言模型转化为有效的双向多模态嵌入模型。第一阶段，模态感知持续预训练，引入了一个联合重建目标，同时清理交错的文字和图像输入，增强双向上下文感知推理。第二阶段，异质对比微调，利用超越简单图像-描述对的多样化、语义丰富的多模态数据，增强泛化能力和对齐。通过引入双向注意机制进行持续预训练，MoCa能够通过联合重建目标有效地扩展到大规模未标注数据集上，并利用多样化多模态数据增强表示鲁棒性。实验表明，MoCa在MMEB和ViDoRe-v2基准测试中一致地提高了性能，并在MMEB上展示了强大的可扩展性，随着模型规模和训练数据量增加，表现更加出色。 

---
# Enhancing Live Broadcast Engagement: A Multi-modal Approach to Short Video Recommendations Using MMGCN and User Preferences 

**Title (ZH)**: 提升直播互动性：基于MMGCN和用户偏好多模态短视频推荐的方法 

**Authors**: Saeid Aghasoleymani Najafabadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23085)  

**Abstract**: The purpose of this paper is to explore a multi-modal approach to enhancing live broadcast engagement by developing a short video recommendation system that incorporates Multi-modal Graph Convolutional Networks (MMGCN) with user preferences. In order to provide personalized recommendations tailored to individual interests, the proposed system takes into account user interaction data, video content features, and contextual information. With the aid of a hybrid approach combining collaborative filtering and content-based filtering techniques, the system is able to capture nuanced relationships between users, video attributes, and engagement patterns. Three datasets are used to evaluate the effectiveness of the system: Kwai, TikTok, and MovieLens. Compared to baseline models, such as DeepFM, Wide & Deep, LightGBM, and XGBoost, the proposed MMGCN-based model shows superior performance. A notable feature of the proposed model is that it outperforms all baseline methods in capturing diverse user preferences and making accurate, personalized recommendations, resulting in a Kwai F1 score of 0.574, a Tiktok F1 score of 0.506, and a MovieLens F1 score of 0.197. We emphasize the importance of multi-modal integration and user-centric approaches in advancing recommender systems, emphasizing the role they play in enhancing content discovery and audience interaction on live broadcast platforms. 

**Abstract (ZH)**: 本文旨在通过结合多模态图卷积网络（MMGCN）和用户偏好，开发一种短视频推荐系统，探索提高直播互动的多模态方法。为了提供个性化推荐以匹配个体兴趣，所提出的系统考虑了用户交互数据、视频内容特征和上下文信息。利用混合方法结合协作过滤和基于内容的过滤技术，系统能够捕捉用户、视频属性和互动模式之间的细微关系。使用Kwai、TikTok和MovieLens等三个数据集评估系统的有效性。与基准模型DeepFM、Wide & Deep、LightGBM和XGBoost相比，基于MMGCN的模型表现出更优的性能。所提出模型的一个显著特点是，它在捕捉多样化的用户偏好和提供精准个性化推荐方面优于所有基准方法，分别在Kwai和TikTok上的F1分为0.574和0.506，在MovieLens上的F1分为0.197。我们强调多模态集成和用户中心方法在推动推荐系统发展中的重要性，突出它们在提高直播平台内容发现和观众互动方面的作用。 

---
# Ovis-U1 Technical Report 

**Title (ZH)**: Ovis-U1技术报告 

**Authors**: Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, Yang Li, Qing-Guo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23044)  

**Abstract**: In this report, we introduce Ovis-U1, a 3-billion-parameter unified model that integrates multimodal understanding, text-to-image generation, and image editing capabilities. Building on the foundation of the Ovis series, Ovis-U1 incorporates a diffusion-based visual decoder paired with a bidirectional token refiner, enabling image generation tasks comparable to leading models like GPT-4o. Unlike some previous models that use a frozen MLLM for generation tasks, Ovis-U1 utilizes a new unified training approach starting from a language model. Compared to training solely on understanding or generation tasks, unified training yields better performance, demonstrating the enhancement achieved by integrating these two tasks. Ovis-U1 achieves a score of 69.6 on the OpenCompass Multi-modal Academic Benchmark, surpassing recent state-of-the-art models such as Ristretto-3B and SAIL-VL-1.5-2B. In text-to-image generation, it excels with scores of 83.72 and 0.89 on the DPG-Bench and GenEval benchmarks, respectively. For image editing, it achieves 4.00 and 6.42 on the ImgEdit-Bench and GEdit-Bench-EN, respectively. As the initial version of the Ovis unified model series, Ovis-U1 pushes the boundaries of multimodal understanding, generation, and editing. 

**Abstract (ZH)**: Ovis-U1：一种统一的30亿参数多模态模型，集成 multimodal 理解、文本生成图像及图像编辑能力 

---
# Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval 

**Title (ZH)**: 掩码感知的文字到图像检索：参照表达分割与跨模态检索相结合 

**Authors**: Li-Cheng Shen, Jih-Kang Hsieh, Wei-Hua Li, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22864)  

**Abstract**: Text-to-image retrieval (TIR) aims to find relevant images based on a textual query, but existing approaches are primarily based on whole-image captions and lack interpretability. Meanwhile, referring expression segmentation (RES) enables precise object localization based on natural language descriptions but is computationally expensive when applied across large image collections. To bridge this gap, we introduce Mask-aware TIR (MaTIR), a new task that unifies TIR and RES, requiring both efficient image search and accurate object segmentation. To address this task, we propose a two-stage framework, comprising a first stage for segmentation-aware image retrieval and a second stage for reranking and object grounding with a multimodal large language model (MLLM). We leverage SAM 2 to generate object masks and Alpha-CLIP to extract region-level embeddings offline at first, enabling effective and scalable online retrieval. Secondly, MLLM is used to refine retrieval rankings and generate bounding boxes, which are matched to segmentation masks. We evaluate our approach on COCO and D$^3$ datasets, demonstrating significant improvements in both retrieval accuracy and segmentation quality over previous methods. 

**Abstract (ZH)**: 基于文本的图像检索与描述联立掩码（Mask-aware Text-to-image Retrieval and Segmentation, MaTIR） 

---
# EAGLE: Efficient Alignment of Generalized Latent Embeddings for Multimodal Survival Prediction with Interpretable Attribution Analysis 

**Title (ZH)**: EAGLE: 效率较高的通用潜在嵌入的高效对齐方法，用于具可解释 Attribution 分析的多模态生存预测 

**Authors**: Aakash Tripathi, Asim Waqas, Matthew B. Schabath, Yasin Yilmaz, Ghulam Rasool  

**Link**: [PDF](https://arxiv.org/pdf/2506.22446)  

**Abstract**: Accurate cancer survival prediction requires integration of diverse data modalities that reflect the complex interplay between imaging, clinical parameters, and textual reports. However, existing multimodal approaches suffer from simplistic fusion strategies, massive computational requirements, and lack of interpretability-critical barriers to clinical adoption. We present EAGLE (Efficient Alignment of Generalized Latent Embeddings), a novel deep learning framework that addresses these limitations through attention-based multimodal fusion with comprehensive attribution analysis. EAGLE introduces four key innovations: (1) dynamic cross-modal attention mechanisms that learn hierarchical relationships between modalities, (2) massive dimensionality reduction (99.96%) while maintaining predictive performance, (3) three complementary attribution methods providing patient-level interpretability, and (4) a unified pipeline enabling seamless adaptation across cancer types. We evaluated EAGLE on 911 patients across three distinct malignancies: glioblastoma (GBM, n=160), intraductal papillary mucinous neoplasms (IPMN, n=171), and non-small cell lung cancer (NSCLC, n=580). Patient-level analysis showed high-risk individuals relied more heavily on adverse imaging features, while low-risk patients demonstrated balanced modality contributions. Risk stratification identified clinically meaningful groups with 4-fold (GBM) to 5-fold (NSCLC) differences in median survival, directly informing treatment intensity decisions. By combining state-of-the-art performance with clinical interpretability, EAGLE bridges the gap between advanced AI capabilities and practical healthcare deployment, offering a scalable solution for multimodal survival prediction that enhances both prognostic accuracy and physician trust in automated predictions. 

**Abstract (ZH)**: 高效整合一般化隐空间嵌入的EAGLE框架：基于注意力的多模态融合与全面的归因分析 

---
