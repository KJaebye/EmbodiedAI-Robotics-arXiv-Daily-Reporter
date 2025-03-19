# Tracking Meets Large Multimodal Models for Driving Scenario Understanding 

**Title (ZH)**: 利用大型多模态模型进行 Driving 场景理解中的追踪方法 

**Authors**: Ayesha Ishaq, Jean Lahoud, Fahad Shahbaz Khan, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2503.14498)  

**Abstract**: Large Multimodal Models (LMMs) have recently gained prominence in autonomous driving research, showcasing promising capabilities across various emerging benchmarks. LMMs specifically designed for this domain have demonstrated effective perception, planning, and prediction skills. However, many of these methods underutilize 3D spatial and temporal elements, relying mainly on image data. As a result, their effectiveness in dynamic driving environments is limited. We propose to integrate tracking information as an additional input to recover 3D spatial and temporal details that are not effectively captured in the images. We introduce a novel approach for embedding this tracking information into LMMs to enhance their spatiotemporal understanding of driving scenarios. By incorporating 3D tracking data through a track encoder, we enrich visual queries with crucial spatial and temporal cues while avoiding the computational overhead associated with processing lengthy video sequences or extensive 3D inputs. Moreover, we employ a self-supervised approach to pretrain the tracking encoder to provide LMMs with additional contextual information, significantly improving their performance in perception, planning, and prediction tasks for autonomous driving. Experimental results demonstrate the effectiveness of our approach, with a gain of 9.5% in accuracy, an increase of 7.04 points in the ChatGPT score, and 9.4% increase in the overall score over baseline models on DriveLM-nuScenes benchmark, along with a 3.7% final score improvement on DriveLM-CARLA. Our code is available at this https URL 

**Abstract (ZH)**: 大型多模态模型在自动驾驶研究中取得了显著进展，展示了在各种新兴基准测试中的出色能力。针对该领域的专门设计的大型多模态模型在感知、规划和预测方面展现了有效的技术。然而，许多方法未能充分利用3D空间和时间元素，主要依赖图像数据。这限制了它们在动态驾驶环境中的有效性。我们提出将跟踪信息作为额外输入，以恢复图像中未能有效捕捉到的3D空间和时间细节。我们介绍了一种新的方法，将这种跟踪信息嵌入到大型多模态模型中，以增强其对驾驶场景的空间时间理解。通过在跟踪编码器中引入3D跟踪数据，我们丰富了视觉查询的关键空间和时间线索，同时避免了处理长视频序列或大量3D输入带来的计算开销。此外，我们采用半监督方法预训练跟踪编码器，为大型多模态模型提供额外的上下文信息，显著提高了其在感知、规划和预测任务中的性能。实验结果表明，与基准模型相比，在DriveLM-nuScenes基准测试中，我们的方法在准确率上提高了9.5%，在ChatGPT评分上提高了7.04分，在整体评分上提高了9.4%，在DriveLM-CARLA上的最终评分提高了3.7%。我们的代码可在此处获得。 

---
# CoCMT: Communication-Efficient Cross-Modal Transformer for Collaborative Perception 

**Title (ZH)**: CoCMT：通信高效的跨模态变压器协作感知 

**Authors**: Rujia Wang, Xiangbo Gao, Hao Xiang, Runsheng Xu, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13504)  

**Abstract**: Multi-agent collaborative perception enhances each agent perceptual capabilities by sharing sensing information to cooperatively perform robot perception tasks. This approach has proven effective in addressing challenges such as sensor deficiencies, occlusions, and long-range perception. However, existing representative collaborative perception systems transmit intermediate feature maps, such as bird-eye view (BEV) representations, which contain a significant amount of non-critical information, leading to high communication bandwidth requirements. To enhance communication efficiency while preserving perception capability, we introduce CoCMT, an object-query-based collaboration framework that optimizes communication bandwidth by selectively extracting and transmitting essential features. Within CoCMT, we introduce the Efficient Query Transformer (EQFormer) to effectively fuse multi-agent object queries and implement a synergistic deep supervision to enhance the positive reinforcement between stages, leading to improved overall performance. Experiments on OPV2V and V2V4Real datasets show CoCMT outperforms state-of-the-art methods while drastically reducing communication needs. On V2V4Real, our model (Top-50 object queries) requires only 0.416 Mb bandwidth, 83 times less than SOTA methods, while improving AP70 by 1.1 percent. This efficiency breakthrough enables practical collaborative perception deployment in bandwidth-constrained environments without sacrificing detection accuracy. 

**Abstract (ZH)**: 基于对象查询的合作Transformer：一种优化通信带宽的合作感知框架 

---
# MusicInfuser: Making Video Diffusion Listen and Dance 

**Title (ZH)**: MusicInfuser: 让视频扩散模型学会倾听和舞蹈 

**Authors**: Susung Hong, Ira Kemelmacher-Shlizerman, Brian Curless, Steven M. Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.14505)  

**Abstract**: We introduce MusicInfuser, an approach for generating high-quality dance videos that are synchronized to a specified music track. Rather than attempting to design and train a new multimodal audio-video model, we show how existing video diffusion models can be adapted to align with musical inputs by introducing lightweight music-video cross-attention and a low-rank adapter. Unlike prior work requiring motion capture data, our approach fine-tunes only on dance videos. MusicInfuser achieves high-quality music-driven video generation while preserving the flexibility and generative capabilities of the underlying models. We introduce an evaluation framework using Video-LLMs to assess multiple dimensions of dance generation quality. The project page and code are available at this https URL. 

**Abstract (ZH)**: 我们引入了MusicInfuser，一种生成高质量与指定音乐轨道同步的舞蹈视频的方法。我们展示了如何通过引入轻量级音乐-视频交叉注意力和低秩适配器，使现有的视频扩散模型适应音乐输入，而不是设计和训练一个新的跨模态音频-视频模型。与需要动捕数据的先前工作不同，我们的方法仅在舞蹈视频上进行微调。MusicInfuser在保持底层模型的灵活性和生成能力的同时实现了高质量的音乐驱动视频生成。我们引入了一个使用Video-LLMs评估舞蹈生成质量多个维度的评估框架。项目页面和代码可在以下链接获取。 

---
# The Power of Context: How Multimodality Improves Image Super-Resolution 

**Title (ZH)**: 上下文的力量：多模态如何提升图像超分辨率 

**Authors**: Kangfu Mei, Hossein Talebi, Mojtaba Ardakani, Vishal M. Patel, Peyman Milanfar, Mauricio Delbracio  

**Link**: [PDF](https://arxiv.org/pdf/2503.14503)  

**Abstract**: Single-image super-resolution (SISR) remains challenging due to the inherent difficulty of recovering fine-grained details and preserving perceptual quality from low-resolution inputs. Existing methods often rely on limited image priors, leading to suboptimal results. We propose a novel approach that leverages the rich contextual information available in multiple modalities -- including depth, segmentation, edges, and text prompts -- to learn a powerful generative prior for SISR within a diffusion model framework. We introduce a flexible network architecture that effectively fuses multimodal information, accommodating an arbitrary number of input modalities without requiring significant modifications to the diffusion process. Crucially, we mitigate hallucinations, often introduced by text prompts, by using spatial information from other modalities to guide regional text-based conditioning. Each modality's guidance strength can also be controlled independently, allowing steering outputs toward different directions, such as increasing bokeh through depth or adjusting object prominence via segmentation. Extensive experiments demonstrate that our model surpasses state-of-the-art generative SISR methods, achieving superior visual quality and fidelity. See project page at this https URL. 

**Abstract (ZH)**: 基于多模态丰富上下文信息的单张图像超分辨率方法 

---
# MP-GUI: Modality Perception with MLLMs for GUI Understanding 

**Title (ZH)**: MP-GUI: 基于MLLMs的模态感知与GUI理解 

**Authors**: Ziwei Wang, Weizhi Chen, Leyang Yang, Sheng Zhou, Shengchu Zhao, Hanbei Zhan, Jiongchao Jin, Liangcheng Li, Zirui Shao, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14021)  

**Abstract**: Graphical user interface (GUI) has become integral to modern society, making it crucial to be understood for human-centric systems. However, unlike natural images or documents, GUIs comprise artificially designed graphical elements arranged to convey specific semantic meanings. Current multi-modal large language models (MLLMs) already proficient in processing graphical and textual components suffer from hurdles in GUI understanding due to the lack of explicit spatial structure modeling. Moreover, obtaining high-quality spatial structure data is challenging due to privacy issues and noisy environments. To address these challenges, we present MP-GUI, a specially designed MLLM for GUI understanding. MP-GUI features three precisely specialized perceivers to extract graphical, textual, and spatial modalities from the screen as GUI-tailored visual clues, with spatial structure refinement strategy and adaptively combined via a fusion gate to meet the specific preferences of different GUI understanding tasks. To cope with the scarcity of training data, we also introduce a pipeline for automatically data collecting. Extensive experiments demonstrate that MP-GUI achieves impressive results on various GUI understanding tasks with limited data. 

**Abstract (ZH)**: 面向GUI的理解的多模态大型语言模型MP-GUI 

---
# Context-aware Multimodal AI Reveals Hidden Pathways in Five Centuries of Art Evolution 

**Title (ZH)**: 基于情境感知的多模态AI揭示五个世纪以来艺术 evolution 的隐藏路径 

**Authors**: Jin Kim, Byunghwee Lee, Taekho You, Jinhyuk Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.13531)  

**Abstract**: The rise of multimodal generative AI is transforming the intersection of technology and art, offering deeper insights into large-scale artwork. Although its creative capabilities have been widely explored, its potential to represent artwork in latent spaces remains underexamined. We use cutting-edge generative AI, specifically Stable Diffusion, to analyze 500 years of Western paintings by extracting two types of latent information with the model: formal aspects (e.g., colors) and contextual aspects (e.g., subject). Our findings reveal that contextual information differentiates between artistic periods, styles, and individual artists more successfully than formal elements. Additionally, using contextual keywords extracted from paintings, we show how artistic expression evolves alongside societal changes. Our generative experiment, infusing prospective contexts into historical artworks, successfully reproduces the evolutionary trajectory of artworks, highlighting the significance of mutual interaction between society and art. This study demonstrates how multimodal AI expands traditional formal analysis by integrating temporal, cultural, and historical contexts. 

**Abstract (ZH)**: 多模态生成AI的兴起正在transform技术与艺术的交汇点，提供对大规模艺术作品更深刻的理解。尽管其创造能力已被广泛探索，但其在潜在空间中代表艺术作品的潜力仍鲜有研究。我们使用最新的生成AI（具体为Stable Diffusion）来分析西方绘画五百年的演变，通过模型提取两种类型的潜在信息：形式要素（如颜色）和情境要素（如主题）。我们的研究发现，情境信息比形式要素更成功地区分了不同时期、风格和个别艺术家。此外，我们利用从绘画中提取的情境关键词，展示了艺术表达如何随着社会变化而演变。通过生成实验，将前瞻性的情境注入历史艺术品，成功再现了艺术作品的进化轨迹，突显了社会与艺术之间相互作用的重要性。本研究展示了多模态AI如何通过整合时间、文化和历史背景来扩展传统的形式分析。 

---
