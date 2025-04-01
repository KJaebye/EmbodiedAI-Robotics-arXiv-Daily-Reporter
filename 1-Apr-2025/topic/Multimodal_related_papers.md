# VET: A Visual-Electronic Tactile System for Immersive Human-Machine Interaction 

**Title (ZH)**: 视觉-电子触觉系统：沉浸式人机交互系统 

**Authors**: Cong Zhang, Yisheng Yangm, Shilong Mu, Chuqiao Lyu, Shoujie Li, Xinyue Chai, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.23440)  

**Abstract**: In the pursuit of deeper immersion in human-machine interaction, achieving higher-dimensional tactile input and output on a single interface has become a key research focus. This study introduces the Visual-Electronic Tactile (VET) System, which builds upon vision-based tactile sensors (VBTS) and integrates electrical stimulation feedback to enable bidirectional tactile communication. We propose and implement a system framework that seamlessly integrates an electrical stimulation film with VBTS using a screen-printing preparation process, eliminating interference from traditional methods. While VBTS captures multi-dimensional input through visuotactile signals, electrical stimulation feedback directly stimulates neural pathways, preventing interference with visuotactile information. The potential of the VET system is demonstrated through experiments on finger electrical stimulation sensitivity zones, as well as applications in interactive gaming and robotic arm teleoperation. This system paves the way for new advancements in bidirectional tactile interaction and its broader applications. 

**Abstract (ZH)**: 基于视觉的电子触觉（VET）系统：实现单界面的高维触觉输入与输出 

---
# LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation 

**Title (ZH)**: LaViC: 调整大型视觉-语言模型以实现视觉意识对话推荐 

**Authors**: Hyunsik Jeon, Satoshi Koide, Yu Wang, Zhankui He, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2503.23312)  

**Abstract**: Conversational recommender systems engage users in dialogues to refine their needs and provide more personalized suggestions. Although textual information suffices for many domains, visually driven categories such as fashion or home decor potentially require detailed visual information related to color, style, or design. To address this challenge, we propose LaViC (Large Vision-Language Conversational Recommendation Framework), a novel approach that integrates compact image representations into dialogue-based recommendation systems. LaViC leverages a large vision-language model in a two-stage process: (1) visual knowledge self-distillation, which condenses product images from hundreds of tokens into a small set of visual tokens in a self-distillation manner, significantly reducing computational overhead, and (2) recommendation prompt tuning, which enables the model to incorporate both dialogue context and distilled visual tokens, providing a unified mechanism for capturing textual and visual features. To support rigorous evaluation of visually-aware conversational recommendation, we construct a new dataset by aligning Reddit conversations with Amazon product listings across multiple visually oriented categories (e.g., fashion, beauty, and home). This dataset covers realistic user queries and product appearances in domains where visual details are crucial. Extensive experiments demonstrate that LaViC significantly outperforms text-only conversational recommendation methods and open-source vision-language baselines. Moreover, LaViC achieves competitive or superior accuracy compared to prominent proprietary baselines (e.g., GPT-3.5-turbo, GPT-4o-mini, and GPT-4o), demonstrating the necessity of explicitly using visual data for capturing product attributes and showing the effectiveness of our vision-language integration. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 基于视觉语言的大规模对话推荐框架（LaViC） 

---
# Visual Acoustic Fields 

**Title (ZH)**: 视觉声场 

**Authors**: Yuelei Li, Hyunjin Kim, Fangneng Zhan, Ri-Zhao Qiu, Mazeyu Ji, Xiaojun Shan, Xueyan Zou, Paul Liang, Hanspeter Pfister, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24270)  

**Abstract**: Objects produce different sounds when hit, and humans can intuitively infer how an object might sound based on its appearance and material properties. Inspired by this intuition, we propose Visual Acoustic Fields, a framework that bridges hitting sounds and visual signals within a 3D space using 3D Gaussian Splatting (3DGS). Our approach features two key modules: sound generation and sound localization. The sound generation module leverages a conditional diffusion model, which takes multiscale features rendered from a feature-augmented 3DGS to generate realistic hitting sounds. Meanwhile, the sound localization module enables querying the 3D scene, represented by the feature-augmented 3DGS, to localize hitting positions based on the sound sources. To support this framework, we introduce a novel pipeline for collecting scene-level visual-sound sample pairs, achieving alignment between captured images, impact locations, and corresponding sounds. To the best of our knowledge, this is the first dataset to connect visual and acoustic signals in a 3D context. Extensive experiments on our dataset demonstrate the effectiveness of Visual Acoustic Fields in generating plausible impact sounds and accurately localizing impact sources. Our project page is at this https URL. 

**Abstract (ZH)**: 视觉声场：一种基于3D高斯散斑的框架将打击声音和视觉信号连接到3D空间中 

---
# Predicting Targeted Therapy Resistance in Non-Small Cell Lung Cancer Using Multimodal Machine Learning 

**Title (ZH)**: 使用多模态机器学习预测非小细胞肺癌的靶向治疗耐药性 

**Authors**: Peiying Hua, Andrea Olofson, Faraz Farhadi, Liesbeth Hondelink, Gregory Tsongalis, Konstantin Dragnev, Dagmar Hoegemann Savellano, Arief Suriawinata, Laura Tafe, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2503.24165)  

**Abstract**: Lung cancer is the primary cause of cancer death globally, with non-small cell lung cancer (NSCLC) emerging as its most prevalent subtype. Among NSCLC patients, approximately 32.3% have mutations in the epidermal growth factor receptor (EGFR) gene. Osimertinib, a third-generation EGFR-tyrosine kinase inhibitor (TKI), has demonstrated remarkable efficacy in the treatment of NSCLC patients with activating and T790M resistance EGFR mutations. Despite its established efficacy, drug resistance poses a significant challenge for patients to fully benefit from osimertinib. The absence of a standard tool to accurately predict TKI resistance, including that of osimertinib, remains a critical obstacle. To bridge this gap, in this study, we developed an interpretable multimodal machine learning model designed to predict patient resistance to osimertinib among late-stage NSCLC patients with activating EGFR mutations, achieving a c-index of 0.82 on a multi-institutional dataset. This machine learning model harnesses readily available data routinely collected during patient visits and medical assessments to facilitate precision lung cancer management and informed treatment decisions. By integrating various data types such as histology images, next generation sequencing (NGS) data, demographics data, and clinical records, our multimodal model can generate well-informed recommendations. Our experiment results also demonstrated the superior performance of the multimodal model over single modality models (c-index 0.82 compared with 0.75 and 0.77), thus underscoring the benefit of combining multiple modalities in patient outcome prediction. 

**Abstract (ZH)**: 非小细胞肺癌是最主要的癌症死亡原因，其中具有表皮生长因子受体（EGFR）基因突变的非小细胞肺癌（NSCLC）是最常见的亚型。在NSCLC患者中，约32.3%的患者具有EGFR基因突变。第三代EGFR酪氨酸激酶抑制剂奥希替尼在具有激活突变和T790M突变的NSCLC患者中表现出显著的治疗效果。尽管奥希替尼已经在临床上证明了其有效性，但药物耐药性仍然阻碍了患者充分利用该药物所带来的益处。缺乏标准工具准确预测酪氨酸激酶抑制剂（TKI）耐药性，包括奥希替尼，仍然是一个关键障碍。为了填补这一空白，本研究开发了一种可解释的多模态机器学习模型，旨在预测具有激活EGFR突变的晚期NSCLC患者对奥希替尼的耐药性，在多机构数据集上取得了0.82的c-index。该机器学习模型利用患者就诊和医疗评估中常规收集的可用数据，有助于实现精准的肺癌管理和知情的治疗决策。通过整合如组织学图像、下一代 sequencing（NGS）数据、人口统计数据和临床记录等多种数据类型，我们的多模态模型能够生成有针对性的建议。实验结果还表明，多模态模型的表现优于单模态模型（c-index为0.82，而单模态模型分别为0.75和0.77），这突显了在患者预后预测中结合多种模态的好处。 

---
# H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding 

**Title (ZH)**: H2VU基准：一个全面的层次化整体视频理解基准 

**Authors**: Qi Wu, Quanlong Zheng, Yanhao Zhang, Junlin Xie, Jinguo Luo, Kuo Wang, Peng Liu, Qingsong Xie, Ru Zhen, Haonan Lu, Zhenyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24008)  

**Abstract**: With the rapid development of multimodal models, the demand for assessing video understanding capabilities has been steadily increasing. However, existing benchmarks for evaluating video understanding exhibit significant limitations in coverage, task diversity, and scene adaptability. These shortcomings hinder the accurate assessment of models' comprehensive video understanding capabilities. To tackle this challenge, we propose a hierarchical and holistic video understanding (H2VU) benchmark designed to evaluate both general video and online streaming video comprehension. This benchmark contributes three key features:
Extended video duration: Spanning videos from brief 3-second clips to comprehensive 1.5-hour recordings, thereby bridging the temporal gaps found in current benchmarks. Comprehensive assessment tasks: Beyond traditional perceptual and reasoning tasks, we have introduced modules for countercommonsense comprehension and trajectory state tracking. These additions test the models' deep understanding capabilities beyond mere prior knowledge. Enriched video data: To keep pace with the rapid evolution of current AI agents, we have expanded first-person streaming video datasets. This expansion allows for the exploration of multimodal models' performance in understanding streaming videos from a first-person perspective. Extensive results from H2VU reveal that existing multimodal large language models (MLLMs) possess substantial potential for improvement in our newly proposed evaluation tasks. We expect that H2VU will facilitate advancements in video understanding research by offering a comprehensive and in-depth analysis of MLLMs. 

**Abstract (ZH)**: 面向视频理解的层级化综合性基准（H2VU）：评估通用视频和流式视频理解能力 

---
# AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference 

**Title (ZH)**: AirCache: 激活跨模态相关性键值缓存压缩以实现高效的大规模视觉-语言模型推理 

**Authors**: Kai Huang, Hao Zou, Bochen Wang, Ye Xi, Zhen Xie, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23956)  

**Abstract**: Recent advancements in Large Visual Language Models (LVLMs) have gained significant attention due to their remarkable reasoning capabilities and proficiency in generalization. However, processing a large number of visual tokens and generating long-context outputs impose substantial computational overhead, leading to excessive demands for key-value (KV) cache. To address this critical bottleneck, we propose AirCache, a novel KV cache compression method aimed at accelerating LVLMs inference. This work systematically investigates the correlations between visual and textual tokens within the attention mechanisms of LVLMs. Our empirical analysis reveals considerable redundancy in cached visual tokens, wherein strategically eliminating these tokens preserves model performance while significantly accelerating context generation. Inspired by these findings, we introduce an elite observation window for assessing the importance of visual components in the KV cache, focusing on stable inter-modal relevancy modeling with enhanced multi-perspective consistency. Additionally, we develop an adaptive layer-wise budget allocation strategy that capitalizes on the strength and skewness of token importance distribution, showcasing superior efficiency compared to uniform allocation. Comprehensive evaluations across multiple LVLMs and benchmarks demonstrate that our method achieves comparable performance to the full cache while retaining only 10% of visual KV cache, thereby reducing decoding latency by 29% to 66% across various batch size and prompt length of inputs. Notably, as cache retention rates decrease, our method exhibits increasing performance advantages over existing approaches. 

**Abstract (ZH)**: 最近在大型视觉语言模型（LVLMs）方面的进展因其卓越的推理能力及泛化能力而引起了广泛关注。然而，处理大量视觉令牌和生成长上下文输出带来了显著的计算成本，导致了对键值（KV）缓存的极大需求。为解决这一关键瓶颈，我们提出了AirCache，一种旨在加速LVLMs推理的新型KV缓存压缩方法。这项工作系统地探讨了LVLMs中的注意力机制中视觉令牌和文本令牌之间的关联。实证分析表明，缓存中的视觉令牌存在大量冗余，通过有选择地消除这些令牌，可以在保持模型性能的同时显著加速上下文生成。受这一发现的启发，我们介绍了一种精英观察窗口，用于评估KV缓存中视觉组件的重要性，重点在于稳定跨模态相关性建模和增强多视角一致性。此外，我们还开发了一种自适应逐层预算分配策略，该策略充分利用了令牌重要性分布的优势和偏斜性，相比于均匀分配显示出了更优越的效率。在多个LVLMs和基准测试中的综合评估表明，我们的方法在保留仅10%视觉KV缓存的情况下，实现了与全缓存相当的性能，同时减少了29%至66%的解码延迟，无论输入的批大小还是提示长度如何。特别地，随着缓存保留率的下降，我们的方法相对于现有方法呈现出越来越大的性能优势。 

---
# Unimodal-driven Distillation in Multimodal Emotion Recognition with Dynamic Fusion 

**Title (ZH)**: 单模态驱动的多模态情感识别动态融合蒸馏 

**Authors**: Jiagen Li, Rui Yu, Huihao Huang, Huaicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23721)  

**Abstract**: Multimodal Emotion Recognition in Conversations (MERC) identifies emotional states across text, audio and video, which is essential for intelligent dialogue systems and opinion analysis. Existing methods emphasize heterogeneous modal fusion directly for cross-modal integration, but often suffer from disorientation in multimodal learning due to modal heterogeneity and lack of instructive guidance. In this work, we propose SUMMER, a novel heterogeneous multimodal integration framework leveraging Mixture of Experts with Hierarchical Cross-modal Fusion and Interactive Knowledge Distillation. Key components include a Sparse Dynamic Mixture of Experts (SDMoE) for capturing dynamic token-wise interactions, a Hierarchical Cross-Modal Fusion (HCMF) for effective fusion of heterogeneous modalities, and Interactive Knowledge Distillation (IKD), which uses a pre-trained unimodal teacher to guide multimodal fusion in latent and logit spaces. Experiments on IEMOCAP and MELD show SUMMER outperforms state-of-the-art methods, particularly in recognizing minority and semantically similar emotions. 

**Abstract (ZH)**: 多模态对话情感识别（MERC）识别跨文本、音频和视频的情感状态，对于智能对话系统和观点分析至关重要。现有的方法强调直接进行异模态融合以实现跨模态集成，但由于模态异质性导致的模态学习方向混乱问题，常常缺乏有效的指导。本文提出SUMMER，一种新颖的异模态融合框架，利用混合专家与层次跨模态融合及交互式知识蒸馏。关键组件包括稀疏动态混合专家（SDMoE）以捕捉动态的令牌级交互，层次跨模态融合（HCMF）以有效融合异质模态，以及交互式知识蒸馏（IKD），通过预训练的单模态教师在潜在空间和逻辑空间指导异模态融合。实验结果表明，SUMMER在IEMOCAP和MELD数据集上优于现有方法，特别是在识别少数和语义相似情感方面。 

---
# BiPVL-Seg: Bidirectional Progressive Vision-Language Fusion with Global-Local Alignment for Medical Image Segmentation 

**Title (ZH)**: BiPVL-Seg：双向渐进视觉-语言融合与全局-局部对齐在医学图像分割中的应用 

**Authors**: Rafi Ibn Sultan, Hui Zhu, Chengyin Li, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23534)  

**Abstract**: Medical image segmentation typically relies solely on visual data, overlooking the rich textual information clinicians use for diagnosis. Vision-language models attempt to bridge this gap, but existing approaches often process visual and textual features independently, resulting in weak cross-modal alignment. Simple fusion techniques fail due to the inherent differences between spatial visual features and sequential text embeddings. Additionally, medical terminology deviates from general language, limiting the effectiveness of off-the-shelf text encoders and further hindering vision-language alignment. We propose BiPVL-Seg, an end-to-end framework that integrates vision-language fusion and embedding alignment through architectural and training innovations, where both components reinforce each other to enhance medical image segmentation. BiPVL-Seg introduces bidirectional progressive fusion in the architecture, which facilitates stage-wise information exchange between vision and text encoders. Additionally, it incorporates global-local contrastive alignment, a training objective that enhances the text encoder's comprehension by aligning text and vision embeddings at both class and concept levels. Extensive experiments on diverse medical imaging benchmarks across CT and MR modalities demonstrate BiPVL-Seg's superior performance when compared with state-of-the-art methods in complex multi-class segmentation. Source code is available in this GitHub repository. 

**Abstract (ZH)**: 医学图像分割通常仅依赖视觉数据，忽略了临床医生用于诊断的丰富文本信息。视觉语言模型尝试弥合这一差距，但现有方法往往独立处理视觉和文本特征，导致模态间对齐较弱。简单的融合技术因空间视觉特征和序列文本嵌入之间的固有差异而失效。此外，医学术语与通用语言不同，限制了现成文本编码器的有效性，并进一步阻碍了视觉语言对齐。我们提出BiPVL-Seg，这是一种端到端框架，通过架构和训练创新将视觉语言融合和嵌入对齐结合在一起，两部分相互增强以提高医学图像分割性能。BiPVL-Seg引入了架构中的双向逐步融合，这促进了视觉和文本编码器逐阶段的信息交换。此外，它还结合了全局-局部对比对齐，这是一种训练目标，通过在类别和概念层面对齐文本和视觉嵌入来增强文本编码器的理解能力。在CT和MR模态的多种医学影像基准测试上的广泛实验表明，与最先进的方法相比，BiPVL-Seg在复杂多类分割中的性能更优。代码可在该GitHub仓库中获得。 

---
# JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization 

**Title (ZH)**: JavisDiT：联合音视频扩散变换器及其层次时空先验同步 

**Authors**: Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.23377)  

**Abstract**: This paper introduces JavisDiT, a novel Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG). Built upon the powerful Diffusion Transformer (DiT) architecture, JavisDiT is able to generate high-quality audio and video content simultaneously from open-ended user prompts. To ensure optimal synchronization, we introduce a fine-grained spatio-temporal alignment mechanism through a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator. This module extracts both global and fine-grained spatio-temporal priors, guiding the synchronization between the visual and auditory components. Furthermore, we propose a new benchmark, JavisBench, consisting of 10,140 high-quality text-captioned sounding videos spanning diverse scenes and complex real-world scenarios. Further, we specifically devise a robust metric for evaluating the synchronization between generated audio-video pairs in real-world complex content. Experimental results demonstrate that JavisDiT significantly outperforms existing methods by ensuring both high-quality generation and precise synchronization, setting a new standard for JAVG tasks. Our code, model, and dataset will be made publicly available at this https URL. 

**Abstract (ZH)**: JavisDiT：一种用于同步音视频生成的新型联合音视频扩散变换器 

---
# Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics 

**Title (ZH)**: 超越单一模态边界：多模态语义生成推荐 

**Authors**: Jing Zhu, Mingxuan Ju, Yozen Liu, Danai Koutra, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23333)  

**Abstract**: Generative recommendation (GR) has become a powerful paradigm in recommendation systems that implicitly links modality and semantics to item representation, in contrast to previous methods that relied on non-semantic item identifiers in autoregressive models. However, previous research has predominantly treated modalities in isolation, typically assuming item content is unimodal (usually text). We argue that this is a significant limitation given the rich, multimodal nature of real-world data and the potential sensitivity of GR models to modality choices and usage. Our work aims to explore the critical problem of Multimodal Generative Recommendation (MGR), highlighting the importance of modality choices in GR nframeworks. We reveal that GR models are particularly sensitive to different modalities and examine the challenges in achieving effective GR when multiple modalities are available. By evaluating design strategies for effectively leveraging multiple modalities, we identify key challenges and introduce MGR-LF++, an enhanced late fusion framework that employs contrastive modality alignment and special tokens to denote different modalities, achieving a performance improvement of over 20% compared to single-modality alternatives. 

**Abstract (ZH)**: 多模态生成推荐（MGR）：模态选择在生成推荐框架中的重要性 

---
# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers 

**Title (ZH)**: DiTFastAttnV2: 头向注意力压缩的多模态扩散变换器 

**Authors**: Hanling Zhang, Rundong Su, Zhihang Yuan, Pengtao Chen, Mingzhu Shen Yibo Fan, Shengen Yan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22796)  

**Abstract**: Text-to-image generation models, especially Multimodal Diffusion Transformers (MMDiT), have shown remarkable progress in generating high-quality images. However, these models often face significant computational bottlenecks, particularly in attention mechanisms, which hinder their scalability and efficiency. In this paper, we introduce DiTFastAttnV2, a post-training compression method designed to accelerate attention in MMDiT. Through an in-depth analysis of MMDiT's attention patterns, we identify key differences from prior DiT-based methods and propose head-wise arrow attention and caching mechanisms to dynamically adjust attention heads, effectively bridging this gap. We also design an Efficient Fused Kernel for further acceleration. By leveraging local metric methods and optimization techniques, our approach significantly reduces the search time for optimal compression schemes to just minutes while maintaining generation quality. Furthermore, with the customized kernel, DiTFastAttnV2 achieves a 68% reduction in attention FLOPs and 1.5x end-to-end speedup on 2K image generation without compromising visual fidelity. 

**Abstract (ZH)**: Text-to-image生成模型，尤其是多模态扩散变换器（MMDiT），在生成高质量图像方面取得了显著进展。然而，这些模型在注意力机制方面常常面临严重的计算瓶颈，这妨碍了它们的可扩展性和效率。在本文中，我们介绍了DiTFastAttnV2，这是一种旨在加速MMDiT中注意力机制的后训练压缩方法。通过对MMDiT注意力模式的深入分析，我们识别出与此前基于DiT的方法的关键差异，并提出了头导向箭头注意力和缓存机制，以动态调整注意力头，有效地弥合了这一差距。此外，我们还设计了一种高效融合内核以进一步加速处理。通过利用局部度量方法和优化技术，我们的方法显著减少了寻找最优压缩方案的时间，只需几分钟即可完成，并且在保持生成质量的同时。借助定制的内核，DiTFastAttnV2在不牺牲视觉保真度的情况下，实现了注意力FLOPs的68%减少和端到端1.5倍的速度提升，在2K图像生成方面的表现尤为显著。 

---
