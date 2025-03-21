# M3: 3D-Spatial MultiModal Memory 

**Title (ZH)**: M3: 3D-空间多模态记忆 

**Authors**: Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng, Jianglong Ye, Sifei Liu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16413)  

**Abstract**: We present 3D Spatial MultiModal Memory (M3), a multimodal memory system designed to retain information about medium-sized static scenes through video sources for visual perception. By integrating 3D Gaussian Splatting techniques with foundation models, M3 builds a multimodal memory capable of rendering feature representations across granularities, encompassing a wide range of knowledge. In our exploration, we identify two key challenges in previous works on feature splatting: (1) computational constraints in storing high-dimensional features for each Gaussian primitive, and (2) misalignment or information loss between distilled features and foundation model features. To address these challenges, we propose M3 with key components of principal scene components and Gaussian memory attention, enabling efficient training and inference. To validate M3, we conduct comprehensive quantitative evaluations of feature similarity and downstream tasks, as well as qualitative visualizations to highlight the pixel trace of Gaussian memory attention. Our approach encompasses a diverse range of foundation models, including vision-language models (VLMs), perception models, and large multimodal and language models (LMMs/LLMs). Furthermore, to demonstrate real-world applicability, we deploy M3's feature field in indoor scenes on a quadruped robot. Notably, we claim that M3 is the first work to address the core compression challenges in 3D feature distillation. 

**Abstract (ZH)**: 我们呈现了3D空间多模态记忆（M3），这是一个设计用于通过视频源保留中等规模静态场景信息的多模态记忆系统，适用于视觉感知。通过将3D高斯点绘制技术与基础模型相结合，M3构建了一个多模态记忆，能够渲染不同粒度下的特征表示，涵盖广泛的知识。在我们的研究中，我们识别了特征点绘制以前工作中存在的两个主要挑战：（1）存储每个高维特征的高计算成本，以及（2）提炼特征与基础模型特征之间的对齐或信息丢失。为了解决这些挑战，我们提出了M3，并包含主场景成分和高斯记忆注意力的关键组件，从而实现高效的训练和推理。为了验证M3的有效性，我们进行了全面的特征相似度定量评估和下游任务评估，并通过定性的可视化突出显示高斯记忆注意力的像素轨迹。我们的方法涵盖了多种基础模型，包括视觉-语言模型、感知模型以及大型多模态和语言模型。此外，为了展示其实用性，我们在四足机器人内部场景中部署了M3的特征场。值得注意的是，我们声称M3是首个解决3D特征提炼核心压缩挑战的工作。 

---
# MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance 

**Title (ZH)**: MagicMotion：基于密集到稀疏轨迹指导的可控视频生成 

**Authors**: Quanhao Li, Zhen Xing, Rui Wang, Hui Zhang, Qi Dai, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16421)  

**Abstract**: Recent advances in video generation have led to remarkable improvements in visual quality and temporal coherence. Upon this, trajectory-controllable video generation has emerged to enable precise object motion control through explicitly defined spatial paths. However, existing methods struggle with complex object movements and multi-object motion control, resulting in imprecise trajectory adherence, poor object consistency, and compromised visual quality. Furthermore, these methods only support trajectory control in a single format, limiting their applicability in diverse scenarios. Additionally, there is no publicly available dataset or benchmark specifically tailored for trajectory-controllable video generation, hindering robust training and systematic evaluation. To address these challenges, we introduce MagicMotion, a novel image-to-video generation framework that enables trajectory control through three levels of conditions from dense to sparse: masks, bounding boxes, and sparse boxes. Given an input image and trajectories, MagicMotion seamlessly animates objects along defined trajectories while maintaining object consistency and visual quality. Furthermore, we present MagicData, a large-scale trajectory-controlled video dataset, along with an automated pipeline for annotation and filtering. We also introduce MagicBench, a comprehensive benchmark that assesses both video quality and trajectory control accuracy across different numbers of objects. Extensive experiments demonstrate that MagicMotion outperforms previous methods across various metrics. Our project page are publicly available at this https URL. 

**Abstract (ZH)**: 近期在视频生成方面的进展显著提升了视觉质量和时间连贯性。在此基础上，轨迹可控的视频生成技术应运而生，能够通过明确定义的空间路径实现精确的物体运动控制。然而，现有方法在处理复杂的物体运动和多物体运动控制时存在困难，导致轨迹遵从不精确、物体一致性差以及视觉质量下降。此外，这些方法仅支持单一格式的轨迹控制，限制了它们在不同场景中的应用。同时，缺乏专门针对轨迹可控视频生成的公开数据集和基准，阻碍了稳健训练和系统评估。为了解决这些问题，我们提出MagicMotion，这是一种新颖的图像到视频生成框架，能够通过从密集到稀疏的三个条件层次实现轨迹控制：掩码、边界框和稀疏盒。给定输入图像和轨迹，MagicMotion能够无缝地沿定义的轨迹动画化物体，同时保持物体一致性和视觉质量。此外，我们还提供了MagicData，这是一个大规模的轨迹可控视频数据集，并提出了一种自动注释和过滤管道。我们还引入了MagicBench，这是一种综合基准，评估不同物体数量下的视频质量和轨迹控制准确性。大量实验表明，MagicMotion在多种度量标准上优于先前的方法。我们的项目页面在此网址公开：这个 https URL。 

---
# Structured-Noise Masked Modeling for Video, Audio and Beyond 

**Title (ZH)**: 结构化噪声掩蔽建模及其在视频、音频等领域的应用 

**Authors**: Aritra Bhowmik, Fida Mohammad Thoker, Carlos Hinojosa, Bernard Ghanem, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2503.16311)  

**Abstract**: Masked modeling has emerged as a powerful self-supervised learning framework, but existing methods largely rely on random masking, disregarding the structural properties of different modalities. In this work, we introduce structured noise-based masking, a simple yet effective approach that naturally aligns with the spatial, temporal, and spectral characteristics of video and audio data. By filtering white noise into distinct color noise distributions, we generate structured masks that preserve modality-specific patterns without requiring handcrafted heuristics or access to the data. Our approach improves the performance of masked video and audio modeling frameworks without any computational overhead. Extensive experiments demonstrate that structured noise masking achieves consistent improvement over random masking for standard and advanced masked modeling methods, highlighting the importance of modality-aware masking strategies for representation learning. 

**Abstract (ZH)**: 基于结构化噪声的掩码：一种符合视频和音频数据空间、时间和频谱特性的简单有效方法 

---
# Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models 

**Title (ZH)**: 多模态大语言模型中视频-token压缩的混合层级指令注入 

**Authors**: Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.16036)  

**Abstract**: Recent Multi-modal Large Language Models (MLLMs) have been challenged by the computational overhead resulting from massive video frames, often alleviated through compression strategies. However, the visual content is not equally contributed to user instructions, existing strategies (\eg, average pool) inevitably lead to the loss of potentially useful information. To tackle this, we propose the Hybrid-level Instruction Injection Strategy for Conditional Token Compression in MLLMs (HICom), utilizing the instruction as a condition to guide the compression from both local and global levels. This encourages the compression to retain the maximum amount of user-focused information while reducing visual tokens to minimize computational burden. Specifically, the instruction condition is injected into the grouped visual tokens at the local level and the learnable tokens at the global level, and we conduct the attention mechanism to complete the conditional compression. From the hybrid-level compression, the instruction-relevant visual parts are highlighted while the temporal-spatial structure is also preserved for easier understanding of LLMs. To further unleash the potential of HICom, we introduce a new conditional pre-training stage with our proposed dataset HICom-248K. Experiments show that our HICom can obtain distinguished video understanding ability with fewer tokens, increasing the performance by 2.43\% average on three multiple-choice QA benchmarks and saving 78.8\% tokens compared with the SOTA method. The code is available at this https URL. 

**Abstract (ZH)**: Recent Multi-modal Large Language Models中的混合级指令注入策略用于条件_token压缩 (HICom) 

---
# Beyond the Visible: Multispectral Vision-Language Learning for Earth Observation 

**Title (ZH)**: 超越可见光：地球观测的多光谱视觉-语言学习 

**Authors**: Clive Tinashe Marimo, Benedikt Blumenstiel, Maximilian Nitsche, Johannes Jakubik, Thomas Brunschwiler  

**Link**: [PDF](https://arxiv.org/pdf/2503.15969)  

**Abstract**: Vision-language models for Earth observation (EO) typically rely on the visual spectrum of data as the only model input, thus failing to leverage the rich spectral information available in the multispectral channels recorded by satellites. Therefore, in this paper, we introduce Llama3-MS-CLIP, the first vision-language model pre-trained with contrastive learning on a large-scale multispectral dataset and report on the performance gains due to the extended spectral range. Furthermore, we present the largest-to-date image-caption dataset for multispectral data, consisting of one million Sentinel-2 samples and corresponding textual descriptions generated with Llama3-LLaVA-Next and Overture Maps data. We develop a scalable captioning pipeline, which is validated by domain experts. We evaluate Llama3-MS-CLIP on multispectral zero-shot image classification and retrieval using three datasets of varying complexity. Our results demonstrate that Llama3-MS-CLIP significantly outperforms other RGB-based approaches, improving classification accuracy by 6.77% on average and retrieval performance by 4.63% mAP compared to the second-best model. Our results emphasize the relevance of multispectral vision-language learning. We release the image-caption dataset, code, and model weights under an open-source license. 

**Abstract (ZH)**: Vision-language模型应用于地球观测（EO）通常仅依赖视觉谱数据作为模型输入，未能充分利用多光谱卫星记录的丰富光谱信息。因此，在本文中，我们介绍了Llama3-MS-CLIP，这是首个采用对比学习在大规模多光谱数据集上进行预训练的vision-language模型，并报告了由于光谱范围的扩展所带来的性能提升。此外，我们呈现了迄今为止最大的多光谱数据图像配对集，包含一百万份Sentinel-2样本及其由Llama3-LLaVA-Next和Overture Maps数据生成的文本描述。我们开发了一种可扩展的配对管道，并通过领域专家验证。我们利用三个不同复杂度的数据集对Llama3-MS-CLIP进行多光谱零样本图像分类和检索评估。我们的结果显示，Llama3-MS-CLIP显著优于其他基于RGB的方法，在分类精度方面平均提高了6.77%，检索性能的mAP提高了4.63%，优于其他模型。我们的结果强调了多光谱vision-language学习的相关性。我们以开源许可发布图像配对集、代码和模型权重。 

---
