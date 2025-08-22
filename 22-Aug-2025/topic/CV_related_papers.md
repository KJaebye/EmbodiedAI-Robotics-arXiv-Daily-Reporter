# You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation 

**Title (ZH)**: 一次姿态检测： minimalist 的单目 RGB 多物体 9D 类别级姿态估计检测变压器 

**Authors**: Hakjin Lee, Junghoon Seo, Jaehoon Sim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14965)  

**Abstract**: Accurately recovering the full 9-DoF pose of unseen instances within specific categories from a single RGB image remains a core challenge for robotics and automation. Most existing solutions still rely on pseudo-depth, CAD models, or multi-stage cascades that separate 2D detection from pose estimation. Motivated by the need for a simpler, RGB-only alternative that learns directly at the category level, we revisit a longstanding question: Can object detection and 9-DoF pose estimation be unified with high performance, without any additional data? We show that they can with our method, YOPO, a single-stage, query-based framework that treats category-level 9-DoF estimation as a natural extension of 2D detection. YOPO augments a transformer detector with a lightweight pose head, a bounding-box-conditioned translation module, and a 6D-aware Hungarian matching cost. The model is trained end-to-end only with RGB images and category-level pose labels. Despite its minimalist design, YOPO sets a new state of the art on three benchmarks. On the REAL275 dataset, it achieves 79.6% $\rm{IoU}_{50}$ and 54.1% under the $10^\circ$$10{\rm{cm}}$ metric, surpassing prior RGB-only methods and closing much of the gap to RGB-D systems. The code, models, and additional qualitative results can be found on our project. 

**Abstract (ZH)**: 从单张RGB图像中准确恢复未见过的特定类别实例的全9-自由度姿态仍然是机器人技术和自动化领域的核心挑战。现有的大多数解决方案仍然依赖于伪深度、CAD模型或多阶段级联方法，将2D检测与姿态估计分离。受需要一种更简单、仅基于RGB且可在类别级别进行学习的替代方案的启发，我们重新审视了一个长期存在的问题：是否可以在没有任何额外数据的情况下，同时实现高性能的对象检测和9-自由度姿态估计？我们证明了可以通过我们的方法YOPO实现这一目标，这是一种单阶段、基于查询的框架，将类别级别的9-自由度估计视为2D检测的自然扩展。YOPO通过一个轻量级姿态头、一个基于边界框的平移模块和一个6D感知匈牙利匹配成本来增强变压器检测器。该模型仅使用RGB图像和类别级别的姿态标签进行端到端训练。尽管设计简洁，YOPO在三个基准测试上取得了新的最佳性能。在REAL275数据集上，它实现了79.6%的$\rm{IoU}_{50}$和54.1%的$10^\circ10\rm{cm}$指标，超越了之前仅基于RGB的方法，并显著缩小了与RGB-D系统之间的差距。更多代码、模型和额外的定性结果可在我们的项目中找到。 

---
# See it. Say it. Sorted: Agentic System for Compositional Diagram Generation 

**Title (ZH)**: 理解它。表达它。搞定它：一种自主系统用于组合性图表生成 

**Authors**: Hantao Zhang, Jingyang Liu, Ed Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15222)  

**Abstract**: We study sketch-to-diagram generation: converting rough hand sketches into precise, compositional diagrams. Diffusion models excel at photorealism but struggle with the spatial precision, alignment, and symbolic structure required for flowcharts. We introduce See it. Say it. Sorted., a training-free agentic system that couples a Vision-Language Model (VLM) with Large Language Models (LLMs) to produce editable Scalable Vector Graphics (SVG) programs. The system runs an iterative loop in which a Critic VLM proposes a small set of qualitative, relational edits; multiple candidate LLMs synthesize SVG updates with diverse strategies (conservative->aggressive, alternative, focused); and a Judge VLM selects the best candidate, ensuring stable improvement. This design prioritizes qualitative reasoning over brittle numerical estimates, preserves global constraints (e.g., alignment, connectivity), and naturally supports human-in-the-loop corrections. On 10 sketches derived from flowcharts in published papers, our method more faithfully reconstructs layout and structure than two frontier closed-source image generation LLMs (GPT-5 and Gemini-2.5-Pro), accurately composing primitives (e.g., multi-headed arrows) without inserting unwanted text. Because outputs are programmatic SVGs, the approach is readily extensible to presentation tools (e.g., PowerPoint) via APIs and can be specialized with improved prompts and task-specific tools. The codebase is open-sourced at this https URL. 

**Abstract (ZH)**: 我们研究草图到图表生成：将粗糙的手绘草图转换为精确的组合图表。扩散模型在逼真度方面表现优异，但在流chart所需的空间精度、对齐和符号结构方面存在困难。我们引入了“见它。说它。整理它。”这一无需训练的代理系统，该系统将视觉语言模型（VLM）与大型语言模型（LLM）结合，以生成可编辑的可缩放矢量图形（SVG）程序。该系统在一个迭代循环中运行，其中评论家VLM提出一小组定性的关系编辑；多个候选LLM使用不同的策略（保守型-激进型、替代型、聚焦型）合成SVG更新；而裁判VLM从中选择最佳候选，确保稳定改进。此设计强调定性推理而非脆弱的数值估计，保留了全局约束（如对齐、可连接性），并自然支持人类在环修正。在源自已发表论文的10个流chart草图上，我们的方法比两个前沿的闭源图像生成LLM（GPT-5和Gemini-2.5-Pro）更准确地重建布局和结构，无需插入不必要的文本即可正确组合基础元素（如多头箭头）。由于输出是程序化的SVG，该方法可以通过API扩展到演示工具（如PowerPoint），并通过改进提示和特定任务工具进一步专家定制。代码库在此公开 available at this https URL。 

---
# SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass 

**Title (ZH)**: SceneGen: 单张图像的一次前向传播生成三维场景 

**Authors**: Yanxu Meng, Haoning Wu, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15769)  

**Abstract**: 3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: this https URL. 

**Abstract (ZH)**: 3D内容生成由于其在VR/AR和嵌入式AI中的应用 recently attracted significant research interest。在本文中，我们针对单张场景图像内合成多个3D资产这一具有挑战性的任务进行了研究。具体而言，我们的贡献包括四个方面：（i）我们提出了SceneGen，一种新型框架，该框架以场景图像和对应的物体掩码为输入，同时生成具有几何和纹理的多个3D资产。值得注意的是，SceneGen 不需要优化或资产检索；（ii）我们引入了一种新颖的特征聚合模块，该模块在特征提取模块中结合了视觉和几何编码器的局部和全局场景信息。结合位置头，这使得在单次前向传递中生成3D资产及其相对空间位置成为可能；（iii）我们展示了SceneGen 直接扩展到多张图像输入场景的能力。尽管仅在单张图像输入上进行训练，但我们的架构设计使得使用多张图像输入时能够提高生成性能；（iv）广泛的定量和定性评估证实了我们方法的高效性和鲁棒性生成能力。我们相信，这种范式为高质量3D内容生成提供了一个新颖的解决方案，并有可能推动其在下游任务中的实际应用。代码和模型将在以下链接公开：this https URL。 

---
# StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding 

**Title (ZH)**: StreamMem: 查询无关的键值缓存内存用于流式视频理解 

**Authors**: Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.15717)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in visual-language reasoning, but their ability to efficiently handle long videos remains limited. Despite recent advances in long-context MLLMs, storing and attending to the key-value (KV) cache for long visual contexts incurs substantial memory and computational overhead. Existing visual compression methods require either encoding the entire visual context before compression or having access to the questions in advance, which is impractical for long video understanding and multi-turn conversational settings. In this work, we propose StreamMem, a query-agnostic KV cache memory mechanism for streaming video understanding. Specifically, StreamMem encodes new video frames in a streaming manner, compressing the KV cache using attention scores between visual tokens and generic query tokens, while maintaining a fixed-size KV memory to enable efficient question answering (QA) in memory-constrained, long-video scenarios. Evaluation on three long video understanding and two streaming video question answering benchmarks shows that StreamMem achieves state-of-the-art performance in query-agnostic KV cache compression and is competitive with query-aware compression approaches. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在视觉语言推理方面取得了显著进展，但其处理长视频的能力仍然有限。尽管在长上下文MLLM方面取得了一些进展，但存储和关注长视觉上下文的键值（KV）缓存会带来显著的内存和计算开销。现有的视觉压缩方法要么在压缩前对整个视觉上下文进行编码，要么需要提前获取问题，这对于长视频理解和多轮对话场景来说是不切实际的。在此工作中，我们提出了一种查询无关的KV缓存记忆机制StreamMem，用于流式视频理解。具体而言，StreamMem以流式方式编码新的视频帧，并使用视觉标记与通用查询标记之间的注意得分来压缩KV缓存，同时保持固定大小的KV内存以在内存受限的长视频场景中实现高效的问题回答（QA）。在三个长视频理解和两个流式视频问答基准上的评估显示，StreamMem在查询无关的KV缓存压缩方面达到了最先进的性能，并且与查询感知的压缩方法具有竞争力。 

---
# Are Virtual DES Images a Valid Alternative to the Real Ones? 

**Title (ZH)**: 虚拟DES图像是否是真实图像的有效替代方案？ 

**Authors**: Ana C. Perre, Luís A. Alexandre, Luís C. Freire  

**Link**: [PDF](https://arxiv.org/pdf/2508.15594)  

**Abstract**: Contrast-enhanced spectral mammography (CESM) is an imaging modality that provides two types of images, commonly known as low-energy (LE) and dual-energy subtracted (DES) images. In many domains, particularly in medicine, the emergence of image-to-image translation techniques has enabled the artificial generation of images using other images as input. Within CESM, applying such techniques to generate DES images from LE images could be highly beneficial, potentially reducing patient exposure to radiation associated with high-energy image acquisition. In this study, we investigated three models for the artificial generation of DES images (virtual DES): a pre-trained U-Net model, a U-Net trained end-to-end model, and a CycleGAN model. We also performed a series of experiments to assess the impact of using virtual DES images on the classification of CESM examinations into malignant and non-malignant categories. To our knowledge, this is the first study to evaluate the impact of virtual DES images on CESM lesion classification. The results demonstrate that the best performance was achieved with the pre-trained U-Net model, yielding an F1 score of 85.59% when using the virtual DES images, compared to 90.35% with the real DES images. This discrepancy likely results from the additional diagnostic information in real DES images, which contributes to a higher classification accuracy. Nevertheless, the potential for virtual DES image generation is considerable and future advancements may narrow this performance gap to a level where exclusive reliance on virtual DES images becomes clinically viable. 

**Abstract (ZH)**: 增强对比度光谱乳腺成像中虚拟双能减影图像的人工生成及其对病变分类的影响研究 

---
# LGMSNet: Thinning a medical image segmentation model via dual-level multiscale fusion 

**Title (ZH)**: LGMSNet: 通过双级别多尺度融合方法稀疏化医疗图像分割模型 

**Authors**: Chengqi Dong, Fenghe Tang, Rongge Mao, Xinpei Gao, S.Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15476)  

**Abstract**: Medical image segmentation plays a pivotal role in disease diagnosis and treatment planning, particularly in resource-constrained clinical settings where lightweight and generalizable models are urgently needed. However, existing lightweight models often compromise performance for efficiency and rarely adopt computationally expensive attention mechanisms, severely restricting their global contextual perception capabilities. Additionally, current architectures neglect the channel redundancy issue under the same convolutional kernels in medical imaging, which hinders effective feature extraction. To address these challenges, we propose LGMSNet, a novel lightweight framework based on local and global dual multiscale that achieves state-of-the-art performance with minimal computational overhead. LGMSNet employs heterogeneous intra-layer kernels to extract local high-frequency information while mitigating channel redundancy. In addition, the model integrates sparse transformer-convolutional hybrid branches to capture low-frequency global information. Extensive experiments across six public datasets demonstrate LGMSNet's superiority over existing state-of-the-art methods. In particular, LGMSNet maintains exceptional performance in zero-shot generalization tests on four unseen datasets, underscoring its potential for real-world deployment in resource-limited medical scenarios. The whole project code is in this https URL. 

**Abstract (ZH)**: 医疗图像分割在疾病诊断和治疗规划中扮演着关键角色，特别是在资源受限的临床环境中，需要轻量级且通用的模型。然而，现有的轻量级模型往往在性能与效率之间做出妥协，并且很少采用计算成本高的注意力机制，严重限制了它们的全局上下文感知能力。此外，当前架构忽视了医学成像中相同卷积核下的信道冗余问题，阻碍了有效的特征提取。为了解决这些挑战，我们提出了LGMSNet，这是一种基于局部和全局双多尺度的新型轻量级框架，能够在最小的计算开销下达到最先进的性能。LGMSNet 使用异构内层卷积核来提取局部高频信息并降低信道冗余。此外，该模型集成了稀疏变压器-卷积混合分支以捕获低频全局信息。在六个公开数据集上的广泛实验表明，LGMSNet 在现有最先进的方法中具有优势。特别是，LGMSNet 在四个未见过的数据集上的零样本泛化测试中保持出色的性能，突显了其在资源受限的医疗场景中实际部署的潜力。整个项目代码见此链接：https://xxxxxxxxxxxxxxx。 

---
# Bladder Cancer Diagnosis with Deep Learning: A Multi-Task Framework and Online Platform 

**Title (ZH)**: 基于深度学习的膀胱癌诊断：多任务框架与在线平台 

**Authors**: Jinliang Yu, Mingduo Xie, Yue Wang, Tianfan Fu, Xianglai Xu, Jiajun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15379)  

**Abstract**: Clinical cystoscopy, the current standard for bladder cancer diagnosis, suffers from significant reliance on physician expertise, leading to variability and subjectivity in diagnostic outcomes. There is an urgent need for objective, accurate, and efficient computational approaches to improve bladder cancer diagnostics.
Leveraging recent advancements in deep learning, this study proposes an integrated multi-task deep learning framework specifically designed for bladder cancer diagnosis from cystoscopic images. Our framework includes a robust classification model using EfficientNet-B0 enhanced with Convolutional Block Attention Module (CBAM), an advanced segmentation model based on ResNet34-UNet++ architecture with self-attention mechanisms and attention gating, and molecular subtyping using ConvNeXt-Tiny to classify molecular markers such as HER-2 and Ki-67. Additionally, we introduce a Gradio-based online diagnostic platform integrating all developed models, providing intuitive features including multi-format image uploads, bilingual interfaces, and dynamic threshold adjustments.
Extensive experimentation demonstrates the effectiveness of our methods, achieving outstanding accuracy (93.28%), F1-score (82.05%), and AUC (96.41%) for classification tasks, and exceptional segmentation performance indicated by a Dice coefficient of 0.9091. The online platform significantly improved the accuracy, efficiency, and accessibility of clinical bladder cancer diagnostics, enabling practical and user-friendly deployment. The code is publicly available.
Our multi-task framework and integrated online tool collectively advance the field of intelligent bladder cancer diagnosis by improving clinical reliability, supporting early tumor detection, and enabling real-time diagnostic feedback. These contributions mark a significant step toward AI-assisted decision-making in urology. 

**Abstract (ZH)**: 临床膀胱镜检查是目前膀胱癌诊断的标准方法，但严重依赖医师 expertise，导致诊断结果的变异性与主观性。迫切需要客观、准确且高效的计算方法以提高膀胱癌诊断效果。
本研究利用近期深度学习的进展，提出了一种专门针对膀胱癌从膀胱镜图像进行诊断的集成多任务深度学习框架。该框架包括使用增强的EfficientNet-B0和Convolutional Block Attention Module (CBAM) 的 robust分类模型、基于ResNet34-UNet++架构配备自注意力机制和注意力门控的高级分割模型，以及使用ConvNeXt-Tiny进行分子亚型分类，以区分如HER-2和Ki-67等分子标记。此外，我们引入了一个基于Gradio的在线诊断平台，集成了所有开发的模型，并提供了包括多格式图像上传、双语界面和动态阈值调整在内的直观功能。
广泛的实验证明了本方法的有效性，在分类任务中取得了卓越的准确率（93.28%）、F1分数（82.05%）和AUC（96.41%），分割性能由Dice系数0.9091表示。在线平台显著提高了临床膀胱癌诊断的准确率、效率和可访问性，实现了实用且用户友好的部署。代码已公开。
本多任务框架和集成在线工具共同推动了智能膀胱癌诊断领域的发展，通过提高临床可靠性、支持早期肿瘤检测以及提供实时诊断反馈。这些贡献标志着AI辅助决策在泌尿学中的重要一步。 

---
# Image-Conditioned 3D Gaussian Splat Quantization 

**Title (ZH)**: 基于图像条件的3D高斯点量化 

**Authors**: Xinshuang Liu, Runfa Blark Li, Keito Suzuki, Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15372)  

**Abstract**: 3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codes, which are quantized representations of the scene, are effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub. 

**Abstract (ZH)**: 基于图像条件的Gaussian斑点量化器（ICGS-Quantizer）：实现高效且适应场景变化的3D场景压缩与更新 

---
# Predicting Road Crossing Behaviour using Pose Detection and Sequence Modelling 

**Title (ZH)**: 基于姿态检测和序列建模的过马路行为预测 

**Authors**: Subhasis Dasgupta, Preetam Saha, Agniva Roy, Jaydip Sen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15336)  

**Abstract**: The world is constantly moving towards AI based systems and autonomous vehicles are now reality in different parts of the world. These vehicles require sensors and cameras to detect objects and maneuver according to that. It becomes important to for such vehicles to also predict from a distant if a person is about to cross a road or not. The current study focused on predicting the intent of crossing the road by pedestrians in an experimental setup. The study involved working with deep learning models to predict poses and sequence modelling for temporal predictions. The study analysed three different sequence modelling to understand the prediction behaviour and it was found out that GRU was better in predicting the intent compared to LSTM model but 1D CNN was the best model in terms of speed. The study involved video analysis, and the output of pose detection model was integrated later on to sequence modelling techniques for an end-to-end deep learning framework for predicting road crossing intents. 

**Abstract (ZH)**: 基于AI的系统的世界正不断进步，自动驾驶车辆现在在世界各地已成为现实。这些车辆需要传感器和摄像头来检测物体并据此进行操作。此类车辆还应在远处预测行人是否即将过马路变得十分重要。本研究专注于在实验设置中预测行人过马路的意图。研究涉及使用深度学习模型预测姿势及使用序列建模进行时间预测。研究分析了三种不同的序列建模以理解预测行为，发现GRU在预测意图方面优于LSTM模型，而1D CNN在速度方面效果最好。研究涉及视频分析，后续将姿态检测模型的输出集成到序列建模技术中，以构建一个端到端的深度学习框架来预测过马路的意图。 

---
# VideoEraser: Concept Erasure in Text-to-Video Diffusion Models 

**Title (ZH)**: VideoEraser: 文本到视频扩散模型中的概念擦除 

**Authors**: Naen Xu, Jinghuai Zhang, Changjiang Li, Zhi Chen, Chunyi Zhou, Qingming Li, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15314)  

**Abstract**: The rapid growth of text-to-video (T2V) diffusion models has raised concerns about privacy, copyright, and safety due to their potential misuse in generating harmful or misleading content. These models are often trained on numerous datasets, including unauthorized personal identities, artistic creations, and harmful materials, which can lead to uncontrolled production and distribution of such content. To address this, we propose VideoEraser, a training-free framework that prevents T2V diffusion models from generating videos with undesirable concepts, even when explicitly prompted with those concepts. Designed as a plug-and-play module, VideoEraser can seamlessly integrate with representative T2V diffusion models via a two-stage process: Selective Prompt Embedding Adjustment (SPEA) and Adversarial-Resilient Noise Guidance (ARNG). We conduct extensive evaluations across four tasks, including object erasure, artistic style erasure, celebrity erasure, and explicit content erasure. Experimental results show that VideoEraser consistently outperforms prior methods regarding efficacy, integrity, fidelity, robustness, and generalizability. Notably, VideoEraser achieves state-of-the-art performance in suppressing undesirable content during T2V generation, reducing it by 46% on average across four tasks compared to baselines. 

**Abstract (ZH)**: 文本到视频扩散模型的快速增长引发了对隐私、版权和安全的担忧，因为这些模型有可能被滥用以生成有害或误导性的内容。为应对这一问题，我们提出了一种无需训练的框架VideoEraser，该框架可防止T2V扩散模型在即使明确提示这些概念的情况下生成包含不良概念的视频。VideoEraser设计为即插即用模块，可以通过两阶段过程——选择性提示嵌入调整（SPEA）和抗对抗扰动噪声引导（ARNG）——无缝集成到代表性的T2V扩散模型中。我们在四个任务（包括物体擦除、艺术风格擦除、名人生涯擦除和明确内容擦除）上进行了广泛的评估。实验结果表明，VideoEraser在有效性、完整性和一致性、鲁棒性和泛化能力方面均优于先前的方法。值得注意的是，VideoEraser在T2V生成过程中抑制不良内容方面达到了最先进的性能，相比基线方法，平均减少46%。 

---
# First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection 

**Title (ZH)**: First RAG, Second SEG: 一种无需训练的迷彩目标检测 paradigm 

**Authors**: Wutao Liu, YiDan Wang, Pan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15313)  

**Abstract**: Camouflaged object detection (COD) poses a significant challenge in computer vision due to the high similarity between objects and their backgrounds. Existing approaches often rely on heavy training and large computational resources. While foundation models such as the Segment Anything Model (SAM) offer strong generalization, they still struggle to handle COD tasks without fine-tuning and require high-quality prompts to yield good performance. However, generating such prompts manually is costly and inefficient. To address these challenges, we propose \textbf{First RAG, Second SEG (RAG-SEG)}, a training-free paradigm that decouples COD into two stages: Retrieval-Augmented Generation (RAG) for generating coarse masks as prompts, followed by SAM-based segmentation (SEG) for refinement. RAG-SEG constructs a compact retrieval database via unsupervised clustering, enabling fast and effective feature retrieval. During inference, the retrieved features produce pseudo-labels that guide precise mask generation using SAM2. Our method eliminates the need for conventional training while maintaining competitive performance. Extensive experiments on benchmark COD datasets demonstrate that RAG-SEG performs on par with or surpasses state-of-the-art methods. Notably, all experiments are conducted on a \textbf{personal laptop}, highlighting the computational efficiency and practicality of our approach. We present further analysis in the Appendix, covering limitations, salient object detection extension, and possible improvements. 

**Abstract (ZH)**: 伪装物体检测 (COD) 由于物体与其背景高度相似，在计算机视觉中构成了显著的挑战。现有的方法往往依赖于大量训练和高性能计算资源。虽然基础模型如 Segment Anything Model (SAM) 能提供强大的泛化能力，但在处理 COD 任务时仍需微调，并且需要高质量的提示以获得良好的性能。然而，手动生成这样的提示成本高且效率低。为了解决这些挑战，我们提出了一种名为 \textbf{First RAG, Second SEG (RAG-SEG)} 的无需训练的范式，将 COD 分解为两个阶段：检索增强生成 (RAG) 用于生成粗略掩码作为提示，随后是基于 SAM 的分割 (SEG) 用于细化。RAG-SEG 通过无监督聚类构建紧凑的检索数据库，实现快速且有效的特征检索。在推理过程中，检索的特征生成伪标签，指导使用 SAM2 进行精确的掩码生成。该方法消除了传统训练的需求，同时保持了竞争性性能。大量基准 COD 数据集的实验表明，RAG-SEG 在性能上与当前最先进的方法相当或超过。值得注意的是，所有实验均在一台 \textbf{个人笔记本电脑} 上进行，突显了该方法的计算效率和实用性。我们在附录中进行了进一步分析，涵盖限制、显著物体检测扩展和可能的改进。 

---
# Explainable Knowledge Distillation for Efficient Medical Image Classification 

**Title (ZH)**: 可解释的知识蒸馏在高效医学图像分类中的应用 

**Authors**: Aqib Nazir Mir, Danish Raza Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15251)  

**Abstract**: This study comprehensively explores knowledge distillation frameworks for COVID-19 and lung cancer classification using chest X-ray (CXR) images. We employ high-capacity teacher models, including VGG19 and lightweight Vision Transformers (Visformer-S and AutoFormer-V2-T), to guide the training of a compact, hardware-aware student model derived from the OFA-595 supernet. Our approach leverages hybrid supervision, combining ground-truth labels with teacher models' soft targets to balance accuracy and computational efficiency. We validate our models on two benchmark datasets: COVID-QU-Ex and LCS25000, covering multiple classes, including COVID-19, healthy, non-COVID pneumonia, lung, and colon cancer. To interpret the spatial focus of the models, we employ Score-CAM-based visualizations, which provide insight into the reasoning process of both teacher and student networks. The results demonstrate that the distilled student model maintains high classification performance with significantly reduced parameters and inference time, making it an optimal choice in resource-constrained clinical environments. Our work underscores the importance of combining model efficiency with explainability for practical, trustworthy medical AI solutions. 

**Abstract (ZH)**: 本研究全面探讨了基于胸部X光图像（CXR）的COVID-19和肺癌分类的知识蒸馏框架。我们采用高容量的教师模型，包括VGG19和轻量级的Vision Transformers（Visformer-S和AutoFormer-V2-T），以指导来自OFA-595超网络的紧凑型硬件感知学生模型的训练。我们的方法利用了混合监督，结合真实标签和教师模型的软目标，以平衡准确性和计算效率。我们使用两个基准数据集COVID-QU-Ex和LCS25000进行模型验证，涵盖COVID-19、健康、非COVID肺炎、肺和结肠癌等多个类别。为了解释模型的空间关注点，我们使用Score-CAM基可视化方法，提供对教师和学生网络推理过程的见解。研究结果表明，蒸馏后学生模型在显著减少参数和推理时间的同时，保持了高分类性能，使其成为资源受限临床环境中的最佳选择。我们的工作强调了将模型效率与可解释性相结合对于实际可靠的医疗AI解决方案的重要性。 

---
# SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis 

**Title (ZH)**: SurgWound-Bench: 一项外科伤口诊断基准 

**Authors**: Jiahao Xu, Changchang Yin, Odysseas Chatzipanagiotou, Diamantis Tsilimigras, Kevin Clear, Bingsheng Yao, Dakuo Wang, Timothy Pawlik, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15189)  

**Abstract**: Surgical site infection (SSI) is one of the most common and costly healthcare-associated infections and and surgical wound care remains a significant clinical challenge in preventing SSIs and improving patient outcomes. While recent studies have explored the use of deep learning for preliminary surgical wound screening, progress has been hindered by concerns over data privacy and the high costs associated with expert annotation. Currently, no publicly available dataset or benchmark encompasses various types of surgical wounds, resulting in the absence of an open-source Surgical-Wound screening tool. To address this gap: (1) we present SurgWound, the first open-source dataset featuring a diverse array of surgical wound types. It contains 697 surgical wound images annotated by 3 professional surgeons with eight fine-grained clinical attributes. (2) Based on SurgWound, we introduce the first benchmark for surgical wound diagnosis, which includes visual question answering (VQA) and report generation tasks to comprehensively evaluate model performance. (3) Furthermore, we propose a three-stage learning framework, WoundQwen, for surgical wound diagnosis. In the first stage, we employ five independent MLLMs to accurately predict specific surgical wound characteristics. In the second stage, these predictions serve as additional knowledge inputs to two MLLMs responsible for diagnosing outcomes, which assess infection risk and guide subsequent interventions. In the third stage, we train a MLLM that integrates the diagnostic results from the previous two stages to produce a comprehensive report. This three-stage framework can analyze detailed surgical wound characteristics and provide subsequent instructions to patients based on surgical images, paving the way for personalized wound care, timely intervention, and improved patient outcomes. 

**Abstract (ZH)**: 手术切口感染（SSI）是最常见的医院关联感染之一，手术伤口护理依然是预防SSI和改善患者预后的重大临床挑战。尽管近期研究探索了深度学习在初步手术伤口筛查中的应用，但数据隐私问题和专家标注的高成本限制了进展。目前，尚无包含多种手术伤口类型的公开数据集或基准，导致缺乏开源的手术伤口筛查工具。为解决这一缺口：（1）我们呈现了SurgWound，这是首个包含多种手术伤口类型的开源数据集，包含697张由三位专业外科医生标注的手术伤口图像，附带八项精细临床属性。（2）基于SurgWound，我们引入了首个手术伤口诊断基准，包含视觉问答（VQA）和报告生成任务，以全面评估模型性能。（3）此外，我们提出了一种三阶段学习框架WoundQwen，用于手术伤口诊断。在第一阶段，我们使用五个独立的MLLM预测特定的手术伤口特征；在第二阶段，这些预测作为额外知识输入，用于两个MLLM进行诊断结果评估，以评估感染风险并指导后续干预；在第三阶段，我们训练一个MLLM综合前两阶段的诊断结果生成全面报告。这一三阶段框架可以分析详细的手术伤口特征，并基于手术图像为患者提供后续指导，为个性化伤口护理、及时干预和改善患者预后铺平道路。 

---
# Reversible Unfolding Network for Concealed Visual Perception with Generative Refinement 

**Title (ZH)**: 可逆展开网络：具有生成性精修的隐藏视觉感知 

**Authors**: Chunming He, Fengyang Xiao, Rihan Zhang, Chengyu Fang, Deng-Ping Fan, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15027)  

**Abstract**: Existing methods for concealed visual perception (CVP) often leverage reversible strategies to decrease uncertainty, yet these are typically confined to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose a reversible unfolding network with generative refinement, termed RUN++. Specifically, RUN++ first formulates the CVP task as a mathematical optimization problem and unfolds the iterative solution into a multi-stage deep network. This approach provides a principled way to apply reversible modeling across both mask and RGB domains while leveraging a diffusion model to resolve the resulting uncertainty. Each stage of the network integrates three purpose-driven modules: a Concealed Object Region Extraction (CORE) module applies reversible modeling to the mask domain to identify core object regions; a Context-Aware Region Enhancement (CARE) module extends this principle to the RGB domain to foster better foreground-background separation; and a Finetuning Iteration via Noise-based Enhancement (FINE) module provides a final refinement. The FINE module introduces a targeted Bernoulli diffusion model that refines only the uncertain regions of the segmentation mask, harnessing the generative power of diffusion for fine-detail restoration without the prohibitive computational cost of a full-image process. This unique synergy, where the unfolding network provides a strong uncertainty prior for the diffusion model, allows RUN++ to efficiently direct its focus toward ambiguous areas, significantly mitigating false positives and negatives. Furthermore, we introduce a new paradigm for building robust CVP systems that remain effective under real-world degradations and extend this concept into a broader bi-level optimization framework. 

**Abstract (ZH)**: 现有的隐藏视觉感知方法（CVP）通常利用可逆策略来减少不确定性，但这些方法通常局限于掩码域，导致RGB域的潜力被忽视。为了解决这一问题，我们提出了一种带有生成性精炼的可逆展开网络，称为RUN++。具体而言，RUN++首先将CVP任务形式化为数学优化问题，并将迭代解展开为多阶段深度网络。这种方法为在掩码域和RGB域上应用可逆建模提供了严格的途径，同时利用扩散模型解决由此产生的不确定性。网络的每一阶段整合了三个目标驱动的模块：隐藏对象区域提取（CORE）模块将可逆建模应用于掩码域以识别核心对象区域；上下文感知区域增强（CARE）模块将此原则扩展到RGB域以促进更好的前景-背景分离；并通过噪声增强的精细调整迭代（FINE）模块提供最终精炼。FINE模块引入了针对不确定区域的泊松扩散模型，仅对分割掩码中的不确定区域进行细化，利用扩散的生成能力进行细节恢复，而不需要全图像处理的高昂计算成本。这种独特的协同作用，即展开网络为扩散模型提供了强烈的不确定性先验，使RUN++能够有效地将注意力集中在模糊区域，显著降低了假阳性与假阴性。此外，我们还提出了一种新的建模范式，用于构建在真实世界退化下仍然有效的CVP系统，并将这一概念扩展到更广泛的多层次优化框架。 

---
# TAIGen: Training-Free Adversarial Image Generation via Diffusion Models 

**Title (ZH)**: TAIGen：无需训练的对抗图像生成方法基于扩散模型 

**Authors**: Susim Roy, Anubhooti Jain, Mayank Vatsa, Richa Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.15020)  

**Abstract**: Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen. 

**Abstract (ZH)**: 基于生成模型的对抗攻击通常会产生低质量图像并需要大量计算资源。尽管扩散模型能够生成高质量的图像，但在对抗生成时通常需要数百个采样步骤。本文介绍了一种无需训练的高效黑盒对抗图像生成方法TAIGen。TAIGen仅需从无条件扩散模型中生成3-20个采样步骤即可生成对抗样本。我们的关键发现是在混合步骤间隔中注入的扰动可以在不处理所有时间步的情况下实现可比的攻击效果。我们开发了一种选择性的RGB通道策略，该策略在红色通道上应用注意力图，并在绿色和蓝色通道上使用GradCAM引导的扰动。该设计保留了图像结构，同时最大化目标模型的误分类。TAIGen在所有测试数据集上均保持了PSNR超过30 dB的视觉质量。在使用VGGNet作为来源的ImageNet上，TAIGen在ResNet上的成功率达到了70.6%，在MNASNet上的成功率达到了80.8%，在ShuffleNet上的成功率达到了97.8%。与现有的基于扩散的攻击方法相比，该方法生成对抗样本的速度快10倍。该方法实现了最低的鲁棒准确率，表明它是最具有影响力的攻击方法，因为防御机制在净化由TAIGen生成的图像方面最不成功。 

---
# Fast Graph Neural Network for Image Classification 

**Title (ZH)**: 快速图神经网络图像分类 

**Authors**: Mustafa Mohammadi Gharasuie, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2508.14958)  

**Abstract**: The rapid progress in image classification has been largely driven by the adoption of Graph Convolutional Networks (GCNs), which offer a robust framework for handling complex data structures. This study introduces a novel approach that integrates GCNs with Voronoi diagrams to enhance image classification by leveraging their ability to effectively model relational data. Unlike conventional convolutional neural networks (CNNs), our method represents images as graphs, where pixels or regions function as vertices. These graphs are then refined using corresponding Delaunay triangulations, optimizing their representation. The proposed model achieves significant improvements in both preprocessing efficiency and classification accuracy across various benchmark datasets, surpassing state-of-the-art approaches, particularly in challenging scenarios involving intricate scenes and fine-grained categories. Experimental results, validated through cross-validation, underscore the effectiveness of combining GCNs with Voronoi diagrams for advancing image classification. This research not only presents a novel perspective on image classification but also expands the potential applications of graph-based learning paradigms in computer vision and unstructured data analysis. 

**Abstract (ZH)**: 图卷积网络与Voronoi图集成的图像分类新方法 

---
# Inference Time Debiasing Concepts in Diffusion Models 

**Title (ZH)**: 差异化概念在扩散模型中的推断时间去偏见 

**Authors**: Lucas S. Kupssinskü, Marco N. Bochernitsan, Jordan Kopper, Otávio Parraga, Rodrigo C. Barros  

**Link**: [PDF](https://arxiv.org/pdf/2508.14933)  

**Abstract**: We propose DeCoDi, a debiasing procedure for text-to-image diffusion-based models that changes the inference procedure, does not significantly change image quality, has negligible compute overhead, and can be applied in any diffusion-based image generation model. DeCoDi changes the diffusion process to avoid latent dimension regions of biased concepts. While most deep learning debiasing methods require complex or compute-intensive interventions, our method is designed to change only the inference procedure. Therefore, it is more accessible to a wide range of practitioners. We show the effectiveness of the method by debiasing for gender, ethnicity, and age for the concepts of nurse, firefighter, and CEO. Two distinct human evaluators manually inspect 1,200 generated images. Their evaluation results provide evidence that our method is effective in mitigating biases based on gender, ethnicity, and age. We also show that an automatic bias evaluation performed by the GPT4o is not significantly statistically distinct from a human evaluation. Our evaluation shows promising results, with reliable levels of agreement between evaluators and more coverage of protected attributes. Our method has the potential to significantly improve the diversity of images it generates by diffusion-based text-to-image generative models. 

**Abstract (ZH)**: 我们提出了DeCoDi，这是一种针对基于文本到图像扩散模型的去偏见程序，该程序改变推理过程，对图像质量影响不大，计算开销可忽略不计，并可应用于任何基于扩散的过程生成图像的模型。DeCoDi通过避免潜在维度中含有偏见概念的区域来改变扩散过程。尽管大多数深度学习去偏见方法需要复杂的或计算密集型的干预，我们的方法仅改变推理过程。因此，它更适用于广泛的实践者。我们通过去偏见护士、消防员和CEO等概念中的性别、种族和年龄偏见，展示了该方法的有效性。两名独立的人类评估员手动检查了1,200张生成的图像。他们的评估结果提供了证据，证明我们的方法在基于性别、种族和年龄方面有效减少了偏见。我们还展示了GPT4o执行的自动偏见评估与人类评估之间在统计上没有显著差异。我们的评估显示了很有希望的结果，评估者之间的一致性水平可靠，且涵盖了更多的受保护属性。该方法有可能显著提高基于文本到图像生成模型的图像多样性。 

---
# Heatmap Regression without Soft-Argmax for Facial Landmark Detection 

**Title (ZH)**: 无需软argmax的热图回归 Facial特征点检测 

**Authors**: Chiao-An Yang, Raymond A. Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14929)  

**Abstract**: Facial landmark detection is an important task in computer vision with numerous applications, such as head pose estimation, expression analysis, face swapping, etc. Heatmap regression-based methods have been widely used to achieve state-of-the-art results in this task. These methods involve computing the argmax over the heatmaps to predict a landmark. Since argmax is not differentiable, these methods use a differentiable approximation, Soft-argmax, to enable end-to-end training on deep-nets. In this work, we revisit this long-standing choice of using Soft-argmax and demonstrate that it is not the only way to achieve strong performance. Instead, we propose an alternative training objective based on the classic structured prediction framework. Empirically, our method achieves state-of-the-art performance on three facial landmark benchmarks (WFLW, COFW, and 300W), converging 2.2x faster during training while maintaining better/competitive accuracy. Our code is available here: this https URL. 

**Abstract (ZH)**: 面部关键点检测是计算机视觉中的一个重要任务，广泛应用于头部姿态估计、表情分析、面部替换等。基于热图回归的方法在这一任务中被广泛使用以实现最先进的成果。这些方法涉及在热图上计算argmax来预测关键点。由于argmax不具备可微性，这些方法使用可微近似Soft-argmax以在深度网络中实现端到端的训练。在本文中，我们重新审视了使用Soft-argmax这一长期选择，并证明它并非实现优异性能的唯一途径。相反，我们提出了一种基于经典结构化预测框架的替代训练目标。实验结果表明，我们的方法在三个面部关键点检测基准（WFLW、COFW和300W）上实现了最先进的性能，在训练过程中快2.2倍地收敛，并且具有更好的/竞争性的精度。我们的代码可在以下链接获取：this https URL。 

---
# The Impact of Image Resolution on Face Detection: A Comparative Analysis of MTCNN, YOLOv XI and YOLOv XII models 

**Title (ZH)**: 图像分辨率对面部检测的影响：MTCNN、YOLOv XI和YOLOv XII模型的比较分析 

**Authors**: Ahmet Can Ömercikoğlu, Mustafa Mansur Yönügül, Pakize Erdoğmuş  

**Link**: [PDF](https://arxiv.org/pdf/2507.23341)  

**Abstract**: Face detection is a crucial component in many AI-driven applications such as surveillance, biometric authentication, and human-computer interaction. However, real-world conditions like low-resolution imagery present significant challenges that degrade detection performance. In this study, we systematically investigate the impact of input resolution on the accuracy and robustness of three prominent deep learning-based face detectors: YOLOv11, YOLOv12, and MTCNN. Using the WIDER FACE dataset, we conduct extensive evaluations across multiple image resolutions (160x160, 320x320, and 640x640) and assess each model's performance using metrics such as precision, recall, mAP50, mAP50-95, and inference time. Results indicate that YOLOv11 outperforms YOLOv12 and MTCNN in terms of detection accuracy, especially at higher resolutions, while YOLOv12 exhibits slightly better recall. MTCNN, although competitive in landmark localization, lags in real-time inference speed. Our findings provide actionable insights for selecting resolution-aware face detection models suitable for varying operational constraints. 

**Abstract (ZH)**: 基于输入分辨率的深度学习面部检测模型性能研究：以YOLOv11、YOLOv12和MTCNN为例 

---
