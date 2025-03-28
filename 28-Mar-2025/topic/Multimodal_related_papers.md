# Unified Multimodal Discrete Diffusion 

**Title (ZH)**: 统一多模态离散扩散 

**Authors**: Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.20853)  

**Abstract**: Multimodal generative models that can understand and generate across multiple modalities are dominated by autoregressive (AR) approaches, which process tokens sequentially from left to right, or top to bottom. These models jointly handle images, text, video, and audio for various tasks such as image captioning, question answering, and image generation. In this work, we explore discrete diffusion models as a unified generative formulation in the joint text and image domain, building upon their recent success in text generation. Discrete diffusion models offer several advantages over AR models, including improved control over quality versus diversity of generated samples, the ability to perform joint multimodal inpainting (across both text and image domains), and greater controllability in generation through guidance. Leveraging these benefits, we present the first Unified Multimodal Discrete Diffusion (UniDisc) model which is capable of jointly understanding and generating text and images for a variety of downstream tasks. We compare UniDisc to multimodal AR models, performing a scaling analysis and demonstrating that UniDisc outperforms them in terms of both performance and inference-time compute, enhanced controllability, editability, inpainting, and flexible trade-off between inference time and generation quality. Code and additional visualizations are available at this https URL. 

**Abstract (ZH)**: 多模态生成模型能够在图像、文本、视频和音频等多个模态间进行理解和生成，目前主要依赖自回归（AR）方法，这些方法按从左到右或从上到下的顺序处理令牌。本工作中，我们探索离散扩散模型作为联合文本和图像域中的统一生成框架，并利用其在文本生成领域的近期成功。离散扩散模型相比自回归模型具有多个优势，包括提高生成样本的质量与多样性的控制、在文本和图像域内联合多模态填补的 ability，以及通过指导增强生成的可控性。利用这些优势，我们提出了首个统一多模态离散扩散（UniDisc）模型，该模型能够联合理解和生成文本和图像，适用于多种下游任务。我们将 UniDisc 与多模态自回归模型进行比较，并进行扩展性分析，结果表明 UniDisc 在性能、推理时延计算、可控性、可编辑性、填补能力以及推理时间与生成质量的灵活权衡方面均优于后者。相关代码和额外可视化信息可访问此链接。 

---
# Graph-to-Vision: Multi-graph Understanding and Reasoning using Vision-Language Models 

**Title (ZH)**: 图到视觉：利用视觉-语言模型进行多图理解与推理 

**Authors**: Ruizhou Li, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21435)  

**Abstract**: Graph Neural Networks (GNNs), as the dominant paradigm for graph-structured learning, have long faced dual challenges of exponentially escalating computational complexity and inadequate cross-scenario generalization capability. With the rapid advancement of multimodal learning, Vision-Language Models (VLMs) have demonstrated exceptional cross-modal relational reasoning capabilities and generalization capacities, thereby opening up novel pathways for overcoming the inherent limitations of conventional graph learning paradigms. However, current research predominantly concentrates on investigating the single-graph reasoning capabilities of VLMs, which fundamentally fails to address the critical requirement for coordinated reasoning across multiple heterogeneous graph data in real-world application scenarios. To address these limitations, we propose the first multi-graph joint reasoning benchmark for VLMs. Our benchmark encompasses four graph categories: knowledge graphs, flowcharts, mind maps, and route maps,with each graph group accompanied by three progressively challenging instruction-response pairs. Leveraging this benchmark, we conducted comprehensive capability assessments of state-of-the-art VLMs and performed fine-tuning on open-source models. This study not only addresses the underexplored evaluation gap in multi-graph reasoning for VLMs but also empirically validates their generalization superiority in graph-structured learning. 

**Abstract (ZH)**: 多图联合推理基准：Vision-Language模型在图结构学习中的跨图联合推理能力评估 

---
# StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion 

**Title (ZH)**: StyleMotif: 多模态运动风格化用以实现风格-内容交叉融合 

**Authors**: Ziyu Guo, Young Yoon Lee, Joseph Liu, Yizhak Ben-Shabat, Victor Zordan, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2503.21775)  

**Abstract**: We present StyleMotif, a novel Stylized Motion Latent Diffusion model, generating motion conditioned on both content and style from multiple modalities. Unlike existing approaches that either focus on generating diverse motion content or transferring style from sequences, StyleMotif seamlessly synthesizes motion across a wide range of content while incorporating stylistic cues from multi-modal inputs, including motion, text, image, video, and audio. To achieve this, we introduce a style-content cross fusion mechanism and align a style encoder with a pre-trained multi-modal model, ensuring that the generated motion accurately captures the reference style while preserving realism. Extensive experiments demonstrate that our framework surpasses existing methods in stylized motion generation and exhibits emergent capabilities for multi-modal motion stylization, enabling more nuanced motion synthesis. Source code and pre-trained models will be released upon acceptance. Project Page: this https URL 

**Abstract (ZH)**: StyleMotif：一种新颖的风格化运动潜扩散模型，该模型根据多种模态的内容和风格生成运动。 

---
# MAVERIX: Multimodal Audio-Visual Evaluation Reasoning IndeX 

**Title (ZH)**: MAVERIX: 多模态音视频评价推理索引 

**Authors**: Liuyue Xie, George Z. Wei, Avik Kuthiala, Ce Zheng, Ananya Bal, Mosam Dabhi, Liting Wen, Taru Rustagi, Ethan Lai, Sushil Khyalia, Rohan Choudhury, Morteza Ziyadi, Xu Zhang, Hao Yang, László A. Jeni  

**Link**: [PDF](https://arxiv.org/pdf/2503.21699)  

**Abstract**: Frontier models have either been language-only or have primarily focused on vision and language modalities. Although recent advancements in models with vision and audio understanding capabilities have shown substantial progress, the field lacks a standardized evaluation framework for thoroughly assessing their cross-modality perception performance. We introduce MAVERIX~(Multimodal Audio-Visual Evaluation Reasoning IndeX), a novel benchmark with 700 videos and 2,556 questions explicitly designed to evaluate multimodal models through tasks that necessitate close integration of video and audio information. MAVERIX uniquely provides models with audiovisual tasks, closely mimicking the multimodal perceptual experiences available to humans during inference and decision-making processes. To our knowledge, MAVERIX is the first benchmark aimed explicitly at assessing comprehensive audiovisual integration. Experiments with state-of-the-art models, including Gemini 1.5 Pro and o1, show performance approaching human levels (around 70% accuracy), while human experts reach near-ceiling performance (95.1%). With standardized evaluation protocols, a rigorously annotated pipeline, and a public toolkit, MAVERIX establishes a challenging testbed for advancing audiovisual multimodal intelligence. 

**Abstract (ZH)**: 多模态音频-视觉评价索引（MAVERIX）：一种新颖的基准测试，用于评估多模态模型的跨模态感知性能 

---
# Keyword-Oriented Multimodal Modeling for Euphemism Identification 

**Title (ZH)**: 面向关键词的多模态模型研究： euphemism 识别 

**Authors**: Yuxue Hu, Junsong Li, Meixuan Chen, Dongyu Su, Tongguan Wang, Ying Sha  

**Link**: [PDF](https://arxiv.org/pdf/2503.21504)  

**Abstract**: Euphemism identification deciphers the true meaning of euphemisms, such as linking "weed" (euphemism) to "marijuana" (target keyword) in illicit texts, aiding content moderation and combating underground markets. While existing methods are primarily text-based, the rise of social media highlights the need for multimodal analysis, incorporating text, images, and audio. However, the lack of multimodal datasets for euphemisms limits further research. To address this, we regard euphemisms and their corresponding target keywords as keywords and first introduce a keyword-oriented multimodal corpus of euphemisms (KOM-Euph), involving three datasets (Drug, Weapon, and Sexuality), including text, images, and speech. We further propose a keyword-oriented multimodal euphemism identification method (KOM-EI), which uses cross-modal feature alignment and dynamic fusion modules to explicitly utilize the visual and audio features of the keywords for efficient euphemism identification. Extensive experiments demonstrate that KOM-EI outperforms state-of-the-art models and large language models, and show the importance of our multimodal datasets. 

**Abstract (ZH)**: euphemism识别揭示隐喻的真实含义，例如将“weed”（隐语）关联到“marijuana”（目标关键词）等词汇在非法文本中的使用，助力内容审核并打击地下市场。尽管现有方法主要基于文本，社交媒体的兴起强调了多模态分析的需求，将文本、图像和音频结合起来。然而，缺乏多模态隐语数据集限制了进一步研究。为解决这一问题，我们视隐语及其对应的目标关键词为关键词，并首次引入一种以关键词为中心的多模态隐语数据集(KOM-Euph)，包括毒品、武器和性三个领域，包含文本、图像和语音数据。我们进一步提出了一种以关键词为中心的多模态隐语识别方法(KOM-EI)，该方法使用跨模态特征对齐和动态融合模块，明确利用关键词的视觉和音频特征，实现高效的隐语识别。大量实验表明，KOM-EI 在performance上优于现有模型和大语言模型，并凸显了我们多模态数据集的重要性。 

---
# FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval 

**Title (ZH)**: FineCIR: 明确解析组成图像检索中的细粒度修改语义 

**Authors**: Zixu Li, Zhiheng Fu, Yupeng Hu, Zhiwei Chen, Haokun Wen, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.21309)  

**Abstract**: Composed Image Retrieval (CIR) facilitates image retrieval through a multimodal query consisting of a reference image and modification text. The reference image defines the retrieval context, while the modification text specifies desired alterations. However, existing CIR datasets predominantly employ coarse-grained modification text (CoarseMT), which inadequately captures fine-grained retrieval intents. This limitation introduces two key challenges: (1) ignoring detailed differences leads to imprecise positive samples, and (2) greater ambiguity arises when retrieving visually similar images. These issues degrade retrieval accuracy, necessitating manual result filtering or repeated queries. To address these limitations, we develop a robust fine-grained CIR data annotation pipeline that minimizes imprecise positive samples and enhances CIR systems' ability to discern modification intents accurately. Using this pipeline, we refine the FashionIQ and CIRR datasets to create two fine-grained CIR datasets: Fine-FashionIQ and Fine-CIRR. Furthermore, we introduce FineCIR, the first CIR framework explicitly designed to parse the modification text. FineCIR effectively captures fine-grained modification semantics and aligns them with ambiguous visual entities, enhancing retrieval precision. Extensive experiments demonstrate that FineCIR consistently outperforms state-of-the-art CIR baselines on both fine-grained and traditional CIR benchmark datasets. Our FineCIR code and fine-grained CIR datasets are available at this https URL. 

**Abstract (ZH)**: 细粒度图像检索（细粒度CIR）通过结合参考图像和修改文本的多模态查询来促进图像检索。参考图像定义检索上下文，而修改文本指定所需的修改。然而，现有的CIR数据集主要采用粗粒度修改文本（CoarseMT），这未能充分捕捉到细粒度的检索意图。这一限制引入了两个关键挑战：（1）忽视细节差异会导致不精确的正样本，（2）在检索视觉上相似的图像时增加了更大的模糊性。这些问题降低了检索准确性，需要手动过滤结果或重复查询。为了解决这些问题，我们开发了一种稳健的细粒度CIR数据标注管道，以减少不精确的正样本并增强CIR系统准确区分修改意图的能力。使用该管道，我们对FashionIQ和CIRR数据集进行了细化，创建了两个细粒度CIR数据集：Fine-FashionIQ和Fine-CIRR。此外，我们引入了FineCIR，这是第一个明确设计用于解析修改文本的CIR框架。FineCIR有效地捕捉到细粒度的修改语义，并与模糊的视觉实体对齐，从而提高检索精度。广泛的实验表明，FineCIR在细粒度和传统CIR基准数据集上始终优于最先进的CIR基线系统。我们的FineCIR代码和细粒度CIR数据集可从此处访问。 

---
# InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression 

**Title (ZH)**: InternVL-X: 提升并加速 InternVL 系列模型的高效视觉 token 压缩 

**Authors**: Dongchen Lu, Yuyao Sun, Zilu Zhang, Leping Huang, Jianliang Zeng, Mao Shu, Huo Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.21307)  

**Abstract**: Most multimodal large language models (MLLMs) treat visual tokens as "a sequence of text", integrating them with text tokens into a large language model (LLM). However, a great quantity of visual tokens significantly increases the demand for computational resources and time. In this paper, we propose InternVL-X, which outperforms the InternVL model in both performance and efficiency by incorporating three visual token compression methods. First, we propose a novel vision-language projector, PVTC. This component integrates adjacent visual embeddings to form a local query and utilizes the transformed CLS token as a global query, then performs point-to-region cross-attention through these local and global queries to more effectively convert visual features. Second, we present a layer-wise visual token compression module, LVTC, which compresses tokens in the LLM shallow layers and then expands them through upsampling and residual connections in the deeper layers. This significantly enhances the model computational efficiency. Futhermore, we propose an efficient high resolution slicing method, RVTC, which dynamically adjusts the number of visual tokens based on image area or length filtering. RVTC greatly enhances training efficiency with only a slight reduction in performance. By utilizing 20% or fewer visual tokens, InternVL-X achieves state-of-the-art performance on 7 public MLLM benchmarks, and improves the average metric by 2.34% across 12 tasks. 

**Abstract (ZH)**: InternVL-X：通过三种视觉_token压缩方法在性能和效率上超越InternVL模型 

---
# Vision-to-Music Generation: A Survey 

**Title (ZH)**: 视觉到音乐生成：一个综述 

**Authors**: Zhaokai Wang, Chenxi Bao, Le Zhuo, Jingrui Han, Yang Yue, Yihong Tang, Victor Shea-Jay Huang, Yue Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.21254)  

**Abstract**: Vision-to-music Generation, including video-to-music and image-to-music tasks, is a significant branch of multimodal artificial intelligence demonstrating vast application prospects in fields such as film scoring, short video creation, and dance music synthesis. However, compared to the rapid development of modalities like text and images, research in vision-to-music is still in its preliminary stage due to its complex internal structure and the difficulty of modeling dynamic relationships with video. Existing surveys focus on general music generation without comprehensive discussion on vision-to-music. In this paper, we systematically review the research progress in the field of vision-to-music generation. We first analyze the technical characteristics and core challenges for three input types: general videos, human movement videos, and images, as well as two output types of symbolic music and audio music. We then summarize the existing methodologies on vision-to-music generation from the architecture perspective. A detailed review of common datasets and evaluation metrics is provided. Finally, we discuss current challenges and promising directions for future research. We hope our survey can inspire further innovation in vision-to-music generation and the broader field of multimodal generation in academic research and industrial applications. To follow latest works and foster further innovation in this field, we are continuously maintaining a GitHub repository at this https URL. 

**Abstract (ZH)**: 视觉到音乐生成：包括视频到音乐和图像到音乐任务，是多模态人工智能的一个重要分支，展示了在电影配乐、短视频创作和舞蹈音乐合成等领域广泛的应用前景。然而，由于其复杂内部结构和视频动态关系建模的难度，与文本和图像等模态相比，视觉到音乐的研究仍处于初级阶段。现有综述主要关注一般音乐生成，缺乏对视觉到音乐的全面讨论。本文系统回顾了视觉到音乐生成的研究进展。我们首先分析了三种输入类型（通用视频、人体运动视频和图像）以及两种输出类型（符号音乐和音频音乐）的技术特征和核心挑战。随后从架构视角总结了视觉到音乐生成的现有方法。提供了常见数据集和评估指标的详细综述。最后讨论了当前的研究挑战和未来研究的潜在方向。我们希望本文综述能够激发视觉到音乐生成及其更广泛领域的多模态生成在学术研究和工业应用中的创新。为了跟踪最新研究并促进该领域的进一步创新，我们持续维护一个GitHub仓库，链接为这个 https URL。 

---
