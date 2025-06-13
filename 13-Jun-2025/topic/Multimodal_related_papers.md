# Demonstrating Multi-Suction Item Picking at Scale via Multi-Modal Learning of Pick Success 

**Title (ZH)**: 基于多模态学习的多吸盘抓取物体大规模拣选演示 

**Authors**: Che Wang, Jeroen van Baar, Chaitanya Mitash, Shuai Li, Dylan Randle, Weiyao Wang, Sumedh Sontakke, Kostas E. Bekris, Kapil Katyal  

**Link**: [PDF](https://arxiv.org/pdf/2506.10359)  

**Abstract**: This work demonstrates how autonomously learning aspects of robotic operation from sparsely-labeled, real-world data of deployed, engineered solutions at industrial scale can provide with solutions that achieve improved performance. Specifically, it focuses on multi-suction robot picking and performs a comprehensive study on the application of multi-modal visual encoders for predicting the success of candidate robotic picks. Picking diverse items from unstructured piles is an important and challenging task for robot manipulation in real-world settings, such as warehouses. Methods for picking from clutter must work for an open set of items while simultaneously meeting latency constraints to achieve high throughput. The demonstrated approach utilizes multiple input modalities, such as RGB, depth and semantic segmentation, to estimate the quality of candidate multi-suction picks. The strategy is trained from real-world item picking data, with a combination of multimodal pretrain and finetune. The manuscript provides comprehensive experimental evaluation performed over a large item-picking dataset, an item-picking dataset targeted to include partial occlusions, and a package-picking dataset, which focuses on containers, such as boxes and envelopes, instead of unpackaged items. The evaluation measures performance for different item configurations, pick scenes, and object types. Ablations help to understand the effects of in-domain pretraining, the impact of different modalities and the importance of finetuning. These ablations reveal both the importance of training over multiple modalities but also the ability of models to learn during pretraining the relationship between modalities so that during finetuning and inference, only a subset of them can be used as input. 

**Abstract (ZH)**: 基于稀疏标注实际数据的工业规模自主学习在机器人操作中的应用：以多吸盘机器人拣选为例 

---
# VINCIE: Unlocking In-context Image Editing from Video 

**Title (ZH)**: VINCIE: 在上下文中解锁视频中的图像编辑 

**Authors**: Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10941)  

**Abstract**: In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (e.g., segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications. 

**Abstract (ZH)**: 基于上下文的图像编辑旨在根据包含文本和先前生成图像的上下文字序列修改图像。现有方法通常依赖于特定任务的管道和专家模型（例如，分割和 inpainting）来整理训练数据。在本文中，我们探索是否可以直接从视频中学习基于上下文的图像编辑模型。我们介绍了一种可扩展的方法来标注视频为交错的多模态序列。为了有效地从这些数据中学习，我们设计了一个在三个代理任务（下一帧预测、当前分割预测和下一分割预测）上训练的块因果扩散变换器。此外，我们提出了一种新的多轮图像编辑基准来推进该领域的研究。广泛的实验表明，我们的模型表现出强大的基于上下文的图像编辑能力，并在两个多轮图像编辑基准上实现了最先进的结果。尽管仅在视频上进行训练，我们的模型在多概念组合、故事情节生成和连续编辑应用方面也表现出令人 Promise 的能力。 

---
# M4V: Multi-Modal Mamba for Text-to-Video Generation 

**Title (ZH)**: 多模态Mamba：面向文本到视频生成的多模态模型 

**Authors**: Jiancheng Huang, Gengwei Zhang, Zequn Jie, Siyu Jiao, Yinlong Qian, Ling Chen, Yunchao Wei, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.10915)  

**Abstract**: Text-to-video generation has significantly enriched content creation and holds the potential to evolve into powerful world simulators. However, modeling the vast spatiotemporal space remains computationally demanding, particularly when employing Transformers, which incur quadratic complexity in sequence processing and thus limit practical applications. Recent advancements in linear-time sequence modeling, particularly the Mamba architecture, offer a more efficient alternative. Nevertheless, its plain design limits its direct applicability to multi-modal and spatiotemporal video generation tasks. To address these challenges, we introduce M4V, a Multi-Modal Mamba framework for text-to-video generation. Specifically, we propose a multi-modal diffusion Mamba (MM-DiM) block that enables seamless integration of multi-modal information and spatiotemporal modeling through a multi-modal token re-composition design. As a result, the Mamba blocks in M4V reduce FLOPs by 45% compared to the attention-based alternative when generating videos at 768$\times$1280 resolution. Additionally, to mitigate the visual quality degradation in long-context autoregressive generation processes, we introduce a reward learning strategy that further enhances per-frame visual realism. Extensive experiments on text-to-video benchmarks demonstrate M4V's ability to produce high-quality videos while significantly lowering computational costs. Code and models will be publicly available at this https URL. 

**Abstract (ZH)**: 多模态Mamba框架：面向文本到视频生成的时空建模与高效计算 

---
# Towards Scalable SOAP Note Generation: A Weakly Supervised Multimodal Framework 

**Title (ZH)**: 面向可扩展的SOAP笔记生成：一种弱监督多模态框架 

**Authors**: Sadia Kamal, Tim Oates, Joy Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.10328)  

**Abstract**: Skin carcinoma is the most prevalent form of cancer globally, accounting for over $8 billion in annual healthcare expenditures. In clinical settings, physicians document patient visits using detailed SOAP (Subjective, Objective, Assessment, and Plan) notes. However, manually generating these notes is labor-intensive and contributes to clinician burnout. In this work, we propose a weakly supervised multimodal framework to generate clinically structured SOAP notes from limited inputs, including lesion images and sparse clinical text. Our approach reduces reliance on manual annotations, enabling scalable, clinically grounded documentation while alleviating clinician burden and reducing the need for large annotated data. Our method achieves performance comparable to GPT-4o, Claude, and DeepSeek Janus Pro across key clinical relevance metrics. To evaluate clinical quality, we introduce two novel metrics MedConceptEval and Clinical Coherence Score (CCS) which assess semantic alignment with expert medical concepts and input features, respectively. 

**Abstract (ZH)**: 皮肤癌是全球最常见的癌症类型，每年在医疗保健支出中占比超过80亿美元。在临床环境中，医生利用详细的SOAP（主观、客观、评估、计划）笔记记录患者就诊情况。然而，手工生成这些笔记非常费时且增加了医务人员的职业倦怠。在本研究中，我们提出一种弱监督多模态框架，从有限输入（包括病损图像和稀疏临床文本）自动生成结构化的SOAP笔记。该方法减少了对人工标注的依赖，实现了可扩展的、基于临床的记录，减轻了医务人员的负担并减少了大量标注数据的需求。我们的方法在关键临床相关性指标上达到了与GPT-4o、Claude和DeepSeek Janus Pro相当的性能。为评估临床质量，我们引入了两个新指标：MedConceptEval和临床一致性分数（CCS），分别评估语义与专家医学概念的一致性和输入特征的一致性。 

---
# Cross-Learning Between ECG and PCG: Exploring Common and Exclusive Characteristics of Bimodal Electromechanical Cardiac Waveforms 

**Title (ZH)**: ECG与PCG之间的交叉学习：探讨双模电磁心脏波形的共性和特异性特征 

**Authors**: Sajjad Karimi, Amit J. Shah, Gari D. Clifford, Reza Sameni  

**Link**: [PDF](https://arxiv.org/pdf/2506.10212)  

**Abstract**: Simultaneous electrocardiography (ECG) and phonocardiogram (PCG) provide a comprehensive, multimodal perspective on cardiac function by capturing the heart's electrical and mechanical activities, respectively. However, the distinct and overlapping information content of these signals, as well as their potential for mutual reconstruction and biomarker extraction, remains incompletely understood, especially under varying physiological conditions and across individuals.
In this study, we systematically investigate the common and exclusive characteristics of ECG and PCG using the EPHNOGRAM dataset of simultaneous ECG-PCG recordings during rest and exercise. We employ a suite of linear and nonlinear machine learning models, including non-causal LSTM networks, to reconstruct each modality from the other and analyze the influence of causality, physiological state, and cross-subject variability. Our results demonstrate that nonlinear models, particularly non-causal LSTM, provide superior reconstruction performance, with reconstructing ECG from PCG proving more tractable than the reverse. Exercise and cross-subject scenarios present significant challenges, but envelope-based modeling that utilizes instantaneous amplitude features substantially improves cross-subject generalizability for cross-modal learning. Furthermore, we demonstrate that clinically relevant ECG biomarkers, such as fiducial points and QT intervals, can be estimated from PCG in cross-subject settings.
These findings advance our understanding of the relationship between electromechanical cardiac modalities, in terms of both waveform characteristics and the timing of cardiac events, with potential applications in novel multimodal cardiac monitoring technologies. 

**Abstract (ZH)**: 同时记录心电图和心音图提供了一种综合的多模态视角，用于捕捉心脏的电活动和机械活动。然而，这些信号的独特和重叠信息内容，以及它们的相互重构和生物标志物提取的潜力，尤其是在不同生理条件下和不同个体之间的理解仍然不够充分。 

---
# Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment 

**Title (ZH)**: multimodal知识版权在RAG-as-a-Service环境中的保障 

**Authors**: Tianyu Chen, Jian Lou, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10030)  

**Abstract**: As Retrieval-Augmented Generation (RAG) evolves into service-oriented platforms (Rag-as-a-Service) with shared knowledge bases, protecting the copyright of contributed data becomes essential. Existing watermarking methods in RAG focus solely on textual knowledge, leaving image knowledge unprotected. In this work, we propose AQUA, the first watermark framework for image knowledge protection in Multimodal RAG systems. AQUA embeds semantic signals into synthetic images using two complementary methods: acronym-based triggers and spatial relationship cues. These techniques ensure watermark signals survive indirect watermark propagation from image retriever to textual generator, being efficient, effective and imperceptible. Experiments across diverse models and datasets show that AQUA enables robust, stealthy, and reliable copyright tracing, filling a key gap in multimodal RAG protection. 

**Abstract (ZH)**: 基于多模态RAG系统的图像知识水印框架AQUA 

---
# WDMIR: Wavelet-Driven Multimodal Intent Recognition 

**Title (ZH)**: 小波驱动的多模态意图识别 

**Authors**: Weiyin Gong, Kai Zhang, Yanghai Zhang, Qi Liu, Xinjie Sun, Junyu Lu, Linbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10011)  

**Abstract**: Multimodal intent recognition (MIR) seeks to accurately interpret user intentions by integrating verbal and non-verbal information across video, audio and text modalities. While existing approaches prioritize text analysis, they often overlook the rich semantic content embedded in non-verbal cues. This paper presents a novel Wavelet-Driven Multimodal Intent Recognition(WDMIR) framework that enhances intent understanding through frequency-domain analysis of non-verbal information. To be more specific, we propose: (1) a wavelet-driven fusion module that performs synchronized decomposition and integration of video-audio features in the frequency domain, enabling fine-grained analysis of temporal dynamics; (2) a cross-modal interaction mechanism that facilitates progressive feature enhancement from bimodal to trimodal integration, effectively bridging the semantic gap between verbal and non-verbal information. Extensive experiments on MIntRec demonstrate that our approach achieves state-of-the-art performance, surpassing previous methods by 1.13% on accuracy. Ablation studies further verify that the wavelet-driven fusion module significantly improves the extraction of semantic information from non-verbal sources, with a 0.41% increase in recognition accuracy when analyzing subtle emotional cues. 

**Abstract (ZH)**: 多模态意图识别中基于小波驱动的多模态意图识别框架（Wavelet-Driven Multimodal Intent Recognition Framework） 

---
# Structured Graph Representations for Visual Narrative Reasoning: A Hierarchical Framework for Comics 

**Title (ZH)**: 面向视觉叙事推理的结构化图表示：漫画的层次框架 

**Authors**: Yi-Chun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10008)  

**Abstract**: This paper presents a hierarchical knowledge graph framework for the structured understanding of visual narratives, focusing on multimodal media such as comics. The proposed method decomposes narrative content into multiple levels, from macro-level story arcs to fine-grained event segments. It represents them through integrated knowledge graphs that capture semantic, spatial, and temporal relationships. At the panel level, we construct multimodal graphs that link visual elements such as characters, objects, and actions with corresponding textual components, including dialogue and captions. These graphs are integrated across narrative levels to support reasoning over story structure, character continuity, and event progression.
We apply our approach to a manually annotated subset of the Manga109 dataset and demonstrate its ability to support symbolic reasoning across diverse narrative tasks, including action retrieval, dialogue tracing, character appearance mapping, and panel timeline reconstruction. Evaluation results show high precision and recall across tasks, validating the coherence and interpretability of the framework. This work contributes a scalable foundation for narrative-based content analysis, interactive storytelling, and multimodal reasoning in visual media. 

**Abstract (ZH)**: 本文提出了一种分层次的知识图谱框架，用于结构化理解视觉叙事，重点关注如漫画等多模态媒体。该提出的方珐将叙事内容分解为多个层次，从宏观的故事弧线到细微的事件片段。通过集成的知识图谱表示这些内容，捕捉语义、空间和时间关系。在分镜层面上，我们构建多模态图，将视觉元素如角色、物体和动作与相应的文本组件（包括对话和图注）相链接。这些图在叙事层面进行集成，以支持对故事结构、角色连续性和事件进程的推理。我们将该方法应用于Manga109数据集的手动注释子集，并展示了其在多种叙事任务中支持符号推理的能力，包括动作检索、对话追踪、角色出场映射和分镜时间线重构。评估结果表明，在各项任务中具有高的精确率和召回率，证明了该框架的一致性和可解释性。该工作为基于叙事的内容分析、交互式叙事以及视觉媒体中的多模态推理提供了可扩展的基础。 

---
# Controllable Expressive 3D Facial Animation via Diffusion in a Unified Multimodal Space 

**Title (ZH)**: 统一多模态空间中的可控表情3D面部动画扩散生成 

**Authors**: Kangwei Liu, Junwu Liu, Xiaowei Yi, Jinlin Guo, Yun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10007)  

**Abstract**: Audio-driven emotional 3D facial animation encounters two significant challenges: (1) reliance on single-modal control signals (videos, text, or emotion labels) without leveraging their complementary strengths for comprehensive emotion manipulation, and (2) deterministic regression-based mapping that constrains the stochastic nature of emotional expressions and non-verbal behaviors, limiting the expressiveness of synthesized animations. To address these challenges, we present a diffusion-based framework for controllable expressive 3D facial animation. Our approach introduces two key innovations: (1) a FLAME-centered multimodal emotion binding strategy that aligns diverse modalities (text, audio, and emotion labels) through contrastive learning, enabling flexible emotion control from multiple signal sources, and (2) an attention-based latent diffusion model with content-aware attention and emotion-guided layers, which enriches motion diversity while maintaining temporal coherence and natural facial dynamics. Extensive experiments demonstrate that our method outperforms existing approaches across most metrics, achieving a 21.6\% improvement in emotion similarity while preserving physiologically plausible facial dynamics. Project Page: this https URL. 

**Abstract (ZH)**: 音频驱动的情感3D面部动画面临两个重要挑战：（1）依赖单一模式的控制信号（视频、文本或情感标签），而未能充分利用这些信号的互补优势以实现全方位的情感操控；（2）确定性的回归映射方式限制了情感表达和非言语行为的随机性，从而限制了合成动画的表达性。为应对这些挑战，我们提出了一种基于扩散的可控情感表达3D面部动画框架。我们的方法提出了两个关键创新：（1）以FLAME为中心的多模态情感绑定策略，通过对比学习对齐不同的模态（文本、音频和情感标签），从而从多种信号源中实现灵活的情感控制；（2）基于注意力的潜在扩散模型，具备内容感知的注意力机制和情感导向层，在保持时空连贯性和自然面部动态的同时丰富了动作多样性。广泛的经验研究显示，我们的方法在大多数评估指标上优于现有方法，情感相似度提高了21.6%，同时保持了生理上合理的面部动态。项目页面：this https URL。 

---
# HER2 Expression Prediction with Flexible Multi-Modal Inputs via Dynamic Bidirectional Reconstruction 

**Title (ZH)**: 基于动态双相重构的灵活多模态输入HER2表达预测 

**Authors**: Jie Qin, Wei Yang, Yan Su, Yiran Zhu, Weizhen Li, Yunyue Pan, Chengchang Pan, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10006)  

**Abstract**: Current HER2 assessment models for breast cancer predominantly analyze H&E or IHC images in isolation,despite clinical reliance on their synergistic interpretation. However, concurrent acquisition of both modalities is often hindered by workflow complexity and cost constraints. We propose an adaptive bimodal framework enabling flexible single-/dual-modality HER2 prediction through three innovations: 1) A dynamic branch selector that activates either single-modality reconstruction or dual-modality joint inference based on input completeness; 2) A bidirectional cross-modal GAN performing context-aware feature-space reconstruction of missing modalities; 3) A hybrid training protocol integrating adversarial learning and multi-task optimization. This architecture elevates single-modality H&E prediction accuracy from 71.44% to 94.25% while achieving 95.09% dual-modality accuracy, maintaining 90.28% reliability with sole IHC inputs. The framework's "dual-preferred, single-compatible" design delivers near-bimodal performance without requiring synchronized acquisition, particularly benefiting resource-limited settings through IHC infrastructure cost reduction. Experimental validation confirms 22.81%/12.90% accuracy improvements over H&E/IHC baselines respectively, with cross-modal reconstruction enhancing F1-scores to 0.9609 (HE to IHC) and 0.9251 (IHC to HE). By dynamically routing inputs through reconstruction-enhanced or native fusion pathways, the system mitigates performance degradation from missing data while preserving computational efficiency (78.55% parameter reduction in lightweight variant). This elastic architecture demonstrates significant potential for democratizing precise HER2 assessment across diverse healthcare settings. 

**Abstract (ZH)**: 当前的HER2评估模型主要单独分析H&E或IHC图像，尽管临床依赖于两者协同解析。然而，同时获取两种模态的数据常受限于工作流程复杂性和成本约束。我们提出一种自适应双模态框架，通过三项创新实现灵活的单/双模态HER2预测：1) 动态分支选择器，根据输入完整性激活单模态重建或双模态联合推断；2) 双向跨模态GAN，进行上下文感知的特征空间重建缺失模态；3) 结合对抗学习和多任务优化的混合训练协议。该架构将单模态H&E预测准确性从71.44%提升到94.25%，同时实现95.09%的双模态准确性，并在仅使用IHC输入时保持90.28%的可靠性。该框架的“双模态优先、单模态兼容”设计在无需同步获取的情况下实现接近双模态性能，尤其通过减少IHC基础设施的成本为资源受限环境带来益处。实验验证表明，与H&E/IHC基线相比，分别获得22.81%/12.90%的准确性提升，跨模态重建提高F1分数至HE到IHC为0.9609，IHC到HE为0.9251。通过动态路由输入并通过重建增强或原生融合路径，系统减轻了因数据缺失导致的性能下降，同时保持计算效率（轻量级变体参数减少了78.55%）。该弹性架构在不同医疗保健环境中实现精确HER2评估的普及化展现出显著潜力。 

---
# Multimodal Cinematic Video Synthesis Using Text-to-Image and Audio Generation Models 

**Title (ZH)**: 基于文本到图像和音频生成模型的多模态cinematic视频合成 

**Authors**: Sridhar S, Nithin A, Shakeel Rifath, Vasantha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2506.10005)  

**Abstract**: Advances in generative artificial intelligence have altered multimedia creation, allowing for automatic cinematic video synthesis from text inputs. This work describes a method for creating 60-second cinematic movies incorporating Stable Diffusion for high-fidelity image synthesis, GPT-2 for narrative structuring, and a hybrid audio pipeline using gTTS and YouTube-sourced music. It uses a five-scene framework, which is augmented by linear frame interpolation, cinematic post-processing (e.g., sharpening), and audio-video synchronization to provide professional-quality results. It was created in a GPU-accelerated Google Colab environment using Python 3.11. It has a dual-mode Gradio interface (Simple and Advanced), which supports resolutions of up to 1024x768 and frame rates of 15-30 FPS. Optimizations such as CUDA memory management and error handling ensure reliability. The experiments demonstrate outstanding visual quality, narrative coherence, and efficiency, furthering text-to-video synthesis for creative, educational, and industrial applications. 

**Abstract (ZH)**: 生成式人工智能的进步 telah改变了多媒体创作，使其能够从文本输入自动合成电影级视频。本文描述了一种方法，该方法结合了 Stable Diffusion 用于高保真图像合成、GPT-2 用于叙事结构化以及使用 gTTS 和 YouTube 来源音乐的混合音频流水线，以创建 60 秒的电影。该方法采用五场景框架，并通过线性帧插值、电影后处理（如锐化）以及音频-视频同步来提供专业级别的结果。该方法在使用 Python 3.11 的 GPU 加速 Google Colab 环境中创建。它具有支持高达 1024x768 的分辨率和 15-30 FPS 帧率的双重模式 Gradio 接口（简易模式和高级模式）。CUDA 内存管理和错误处理等优化确保了可靠性。实验结果表明，该方法在视觉质量、叙事连贯性和效率方面表现出色，进一步推动了文本到视频合成在创意、教育和工业领域的应用。 

---
