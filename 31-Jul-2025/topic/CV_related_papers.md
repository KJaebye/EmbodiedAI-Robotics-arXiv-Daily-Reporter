# Viser: Imperative, Web-based 3D Visualization in Python 

**Title (ZH)**: Viser: 基于Web的Python imperative 3D可视化 

**Authors**: Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22885)  

**Abstract**: We present Viser, a 3D visualization library for computer vision and robotics. Viser aims to bring easy and extensible 3D visualization to Python: we provide a comprehensive set of 3D scene and 2D GUI primitives, which can be used independently with minimal setup or composed to build specialized interfaces. This technical report describes Viser's features, interface, and implementation. Key design choices include an imperative-style API and a web-based viewer, which improve compatibility with modern programming patterns and workflows. 

**Abstract (ZH)**: Viser：一种面向计算机视觉和机器人领域的3D可视化库 

---
# Advancing Fetal Ultrasound Image Quality Assessment in Low-Resource Settings 

**Title (ZH)**: 在资源匮乏地区提升胎兒超声图像质量评估方法 

**Authors**: Dongli He, Hu Wang, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2507.22802)  

**Abstract**: Accurate fetal biometric measurements, such as abdominal circumference, play a vital role in prenatal care. However, obtaining high-quality ultrasound images for these measurements heavily depends on the expertise of sonographers, posing a significant challenge in low-income countries due to the scarcity of trained personnel. To address this issue, we leverage FetalCLIP, a vision-language model pretrained on a curated dataset of over 210,000 fetal ultrasound image-caption pairs, to perform automated fetal ultrasound image quality assessment (IQA) on blind-sweep ultrasound data. We introduce FetalCLIP$_{CLS}$, an IQA model adapted from FetalCLIP using Low-Rank Adaptation (LoRA), and evaluate it on the ACOUSLIC-AI dataset against six CNN and Transformer baselines. FetalCLIP$_{CLS}$ achieves the highest F1 score of 0.757. Moreover, we show that an adapted segmentation model, when repurposed for classification, further improves performance, achieving an F1 score of 0.771. Our work demonstrates how parameter-efficient fine-tuning of fetal ultrasound foundation models can enable task-specific adaptations, advancing prenatal care in resource-limited settings. The experimental code is available at: this https URL. 

**Abstract (ZH)**: 准确的胎儿生物测量值，如腹围，对于产前护理至关重要。然而，获取用于这些测量的高质量超声图像很大程度上依赖于超声技师的专业技能，这在低收入国家因训练人员稀缺而成为一个重大挑战。为解决这一问题，我们利用FetalCLIP，一种在超过21万张胎儿超声图像配对描述数据集上预训练的视觉-语言模型，对盲扫超声数据进行自动胎儿超声图像质量评估（IQA）。我们引入了FetalCLIP$_{CLS}$，这是一种基于FetalCLIP并通过低秩适应（LoRA）调整的IQA模型，并在ACOUSLIC-AI数据集上与六种CNN和Transformer基线进行了对比评估。FetalCLIP$_{CLS}$实现了最高的F1分数0.757。此外，我们展示了当一个调整后的分割模型被重新用于分类时，性能进一步提高，实现了F1分数0.771。我们的工作证明了对胎儿超声基础模型进行参数高效的微调可以实现任务特定的适应，促进资源有限地区的产前护理。实验代码可在以下链接获取：this https URL。 

---
# LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text Pairing 

**Title (ZH)**: LOT斯时尚！基于草图-文本配对的多条件图像生成 

**Authors**: Federico Girella, Davide Talon, Ziyue Liu, Zanxi Ruan, Yiming Wang, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22627)  

**Abstract**: Fashion design is a complex creative process that blends visual and textual expressions. Designers convey ideas through sketches, which define spatial structure and design elements, and textual descriptions, capturing material, texture, and stylistic details. In this paper, we present LOcalized Text and Sketch for fashion image generation (LOTS), an approach for compositional sketch-text based generation of complete fashion outlooks. LOTS leverages a global description with paired localized sketch + text information for conditioning and introduces a novel step-based merging strategy for diffusion adaptation. First, a Modularized Pair-Centric representation encodes sketches and text into a shared latent space while preserving independent localized features; then, a Diffusion Pair Guidance phase integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we build on Fashionpedia to release Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Quantitative results show LOTS achieves state-of-the-art image generation performance on both global and localized metrics, while qualitative examples and a human evaluation study highlight its unprecedented level of design customization. 

**Abstract (ZH)**: 局部化文本与草图在时尚图像生成中的应用：一种基于组合草图-文本生成完整时尚外观的方法 

---
# HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors 

**Title (ZH)**: HRVVS: 一种通过分层自回归残差先验的高分辨率视频血管分割网络 

**Authors**: Xincheng Yao, Yijun Yang, Kangwei Guo, Ruiqiang Xiao, Haipeng Zhou, Haisu Tao, Jian Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22530)  

**Abstract**: The segmentation of the hepatic vasculature in surgical videos holds substantial clinical significance in the context of hepatectomy procedures. However, owing to the dearth of an appropriate dataset and the inherently complex task characteristics, few researches have been reported in this domain. To address this issue, we first introduce a high quality frame-by-frame annotated hepatic vasculature dataset containing 35 long hepatectomy videos and 11442 high-resolution frames. On this basis, we propose a novel high-resolution video vasculature segmentation network, dubbed as HRVVS. We innovatively embed a pretrained visual autoregressive modeling (VAR) model into different layers of the hierarchical encoder as prior information to reduce the information degradation generated during the downsampling process. In addition, we designed a dynamic memory decoder on a multi-view segmentation network to minimize the transmission of redundant information while preserving more details between frames. Extensive experiments on surgical video datasets demonstrate that our proposed HRVVS significantly outperforms the state-of-the-art methods. The source code and dataset will be publicly available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 肝血管在手术视频中的分割对于肝切除手术具有重要的临床意义。然而，由于缺乏合适的数据集以及任务本身的复杂性，该领域的相关研究很少。为了解决这一问题，我们首先介绍了一个高质量的帧级标注肝血管数据集，包含35个长肝切除手术视频和11442个高分辨率帧。在此基础上，我们提出了一种新颖的高分辨率视频血管分割网络，名为HRVVS。我们创新性地将在不同层级的层次编码器中嵌入预训练的视觉自回归模型（VAR）作为先验信息，以减少下采样过程中产生的信息降解。此外，我们设计了一种动态记忆解码器在多视图分割网络中，以减少冗余信息的传输并保留更多的帧间细节。在手术视频数据集上的广泛实验结果表明，我们提出的HRVVS显著优于现有方法。源代码和数据集将在\href{这个链接}{这个链接}公开。 

---
# Robust Adverse Weather Removal via Spectral-based Spatial Grouping 

**Title (ZH)**: 基于谱的空间分组的鲁棒恶劣天气去除 

**Authors**: Yuhwan Jeong, Yunseo Yang, Youngjo Yoon, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.22498)  

**Abstract**: Adverse weather conditions cause diverse and complex degradation patterns, driving the development of All-in-One (AiO) models. However, recent AiO solutions still struggle to capture diverse degradations, since global filtering methods like direct operations on the frequency domain fail to handle highly variable and localized distortions. To address these issue, we propose Spectral-based Spatial Grouping Transformer (SSGformer), a novel approach that leverages spectral decomposition and group-wise attention for multi-weather image restoration. SSGformer decomposes images into high-frequency edge features using conventional edge detection and low-frequency information via Singular Value Decomposition. We utilize multi-head linear attention to effectively model the relationship between these features. The fused features are integrated with the input to generate a grouping-mask that clusters regions based on the spatial similarity and image texture. To fully leverage this mask, we introduce a group-wise attention mechanism, enabling robust adverse weather removal and ensuring consistent performance across diverse weather conditions. We also propose a Spatial Grouping Transformer Block that uses both channel attention and spatial attention, effectively balancing feature-wise relationships and spatial dependencies. Extensive experiments show the superiority of our approach, validating its effectiveness in handling the varied and intricate adverse weather degradations. 

**Abstract (ZH)**: 基于谱的空间分组变换器：多天气图像恢复中的频谱分解和分组注意力方法 

---
# Towards Blind Bitstream-corrupted Video Recovery via a Visual Foundation Model-driven Framework 

**Title (ZH)**: 基于视觉基础模型驱动框架的盲比特流损伤视频恢复 

**Authors**: Tianyi Liu, Kejun Wu, Chen Cai, Yi Wang, Kim-Hui Yap, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2507.22481)  

**Abstract**: Video signals are vulnerable in multimedia communication and storage systems, as even slight bitstream-domain corruption can lead to significant pixel-domain degradation. To recover faithful spatio-temporal content from corrupted inputs, bitstream-corrupted video recovery has recently emerged as a challenging and understudied task. However, existing methods require time-consuming and labor-intensive annotation of corrupted regions for each corrupted video frame, resulting in a large workload in practice. In addition, high-quality recovery remains difficult as part of the local residual information in corrupted frames may mislead feature completion and successive content recovery. In this paper, we propose the first blind bitstream-corrupted video recovery framework that integrates visual foundation models with a recovery model, which is adapted to different types of corruption and bitstream-level prompts. Within the framework, the proposed Detect Any Corruption (DAC) model leverages the rich priors of the visual foundation model while incorporating bitstream and corruption knowledge to enhance corruption localization and blind recovery. Additionally, we introduce a novel Corruption-aware Feature Completion (CFC) module, which adaptively processes residual contributions based on high-level corruption understanding. With VFM-guided hierarchical feature augmentation and high-level coordination in a mixture-of-residual-experts (MoRE) structure, our method suppresses artifacts and enhances informative residuals. Comprehensive evaluations show that the proposed method achieves outstanding performance in bitstream-corrupted video recovery without requiring a manually labeled mask sequence. The demonstrated effectiveness will help to realize improved user experience, wider application scenarios, and more reliable multimedia communication and storage systems. 

**Abstract (ZH)**: 盲解码受损视频恢复框架：融合视觉基础模型与恢复模型 

---
# LIDAR: Lightweight Adaptive Cue-Aware Fusion Vision Mamba for Multimodal Segmentation of Structural Cracks 

**Title (ZH)**: LIDAR: 轻量级自适应特征aware融合视觉Mamba用于结构裂缝多模态分割 

**Authors**: Hui Liu, Chen Jia, Fan Shi, Xu Cheng, Mengfei Shi, Xia Xie, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22477)  

**Abstract**: Achieving pixel-level segmentation with low computational cost using multimodal data remains a key challenge in crack segmentation tasks. Existing methods lack the capability for adaptive perception and efficient interactive fusion of cross-modal features. To address these challenges, we propose a Lightweight Adaptive Cue-Aware Vision Mamba network (LIDAR), which efficiently perceives and integrates morphological and textural cues from different modalities under multimodal crack scenarios, generating clear pixel-level crack segmentation maps. Specifically, LIDAR is composed of a Lightweight Adaptive Cue-Aware Visual State Space module (LacaVSS) and a Lightweight Dual Domain Dynamic Collaborative Fusion module (LD3CF). LacaVSS adaptively models crack cues through the proposed mask-guided Efficient Dynamic Guided Scanning Strategy (EDG-SS), while LD3CF leverages an Adaptive Frequency Domain Perceptron (AFDP) and a dual-pooling fusion strategy to effectively capture spatial and frequency-domain cues across modalities. Moreover, we design a Lightweight Dynamically Modulated Multi-Kernel convolution (LDMK) to perceive complex morphological structures with minimal computational overhead, replacing most convolutional operations in LIDAR. Experiments on three datasets demonstrate that our method outperforms other state-of-the-art (SOTA) methods. On the light-field depth dataset, our method achieves 0.8204 in F1 and 0.8465 in mIoU with only 5.35M parameters. Code and datasets are available at this https URL. 

**Abstract (ZH)**: 使用多模态数据在较低计算成本下实现像素级裂缝分割仍是一项关键挑战。现有方法缺乏适应性感知能力和高效的跨模态特征交互融合能力。为应对这些挑战，我们提出了一种轻量级自适应线索感知视觉Mamba网络（LIDAR），该网络在多模态裂缝场景中高效地感知并融合了来自不同模态的形态学和纹理线索，生成清晰的像素级裂缝分割图。具体来说，LIDAR 包含一个轻量级自适应线索感知视觉状态空间模块（LacaVSS）和一个轻量级双域动态协作融合模块（LD3CF）。LacaVSS 通过提出的掩码引导高效动态引导扫描策略（EDG-SS）自适应建模裂缝线索，而 LD3CF 利用自适应频域感知器（AFDP）和双池化融合策略来有效地捕捉跨模态的空间和频域线索。此外，我们设计了一种轻量级动态调制多核卷积（LDMK），以在最小的计算开销下感知复杂的形态结构取代 LIDAR 中的大部分卷积操作。在三个数据集上的实验表明，我们的方法优于其他最先进的（SOTA）方法。在轻场深度数据集上，我们的方法仅使用 5.35 百万参数实现了 0.8204 的 F1 得分和 0.8465 的 mIoU。代码和数据集可在以下链接获取。 

---
# Visual Language Models as Zero-Shot Deepfake Detectors 

**Title (ZH)**: 视觉语言模型作为零样本仿冒检测器 

**Authors**: Viacheslav Pirogov  

**Link**: [PDF](https://arxiv.org/pdf/2507.22469)  

**Abstract**: The contemporary phenomenon of deepfakes, utilizing GAN or diffusion models for face swapping, presents a substantial and evolving threat in digital media, identity verification, and a multitude of other systems. The majority of existing methods for detecting deepfakes rely on training specialized classifiers to distinguish between genuine and manipulated images, focusing only on the image domain without incorporating any auxiliary tasks that could enhance robustness. In this paper, inspired by the zero-shot capabilities of Vision Language Models, we propose a novel VLM-based approach to image classification and then evaluate it for deepfake detection. Specifically, we utilize a new high-quality deepfake dataset comprising 60,000 images, on which our zero-shot models demonstrate superior performance to almost all existing methods. Subsequently, we compare the performance of the best-performing architecture, InstructBLIP, on the popular deepfake dataset DFDC-P against traditional methods in two scenarios: zero-shot and in-domain fine-tuning. Our results demonstrate the superiority of VLMs over traditional classifiers. 

**Abstract (ZH)**: 基于视觉语言模型的无监督深伪检测方法 

---
# Shallow Features Matter: Hierarchical Memory with Heterogeneous Interaction for Unsupervised Video Object Segmentation 

**Title (ZH)**: 浅层特征很重要：异质交互层次记忆在无监督视频物体分割中的应用 

**Authors**: Zheng Xiangyu, He Songcheng, Li Wanyun, Li Xiaoqiang, Zhang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.22465)  

**Abstract**: Unsupervised Video Object Segmentation (UVOS) aims to predict pixel-level masks for the most salient objects in videos without any prior annotations. While memory mechanisms have been proven critical in various video segmentation paradigms, their application in UVOS yield only marginal performance gains despite sophisticated design. Our analysis reveals a simple but fundamental flaw in existing methods: over-reliance on memorizing high-level semantic features. UVOS inherently suffers from the deficiency of lacking fine-grained information due to the absence of pixel-level prior knowledge. Consequently, memory design relying solely on high-level features, which predominantly capture abstract semantic cues, is insufficient to generate precise predictions. To resolve this fundamental issue, we propose a novel hierarchical memory architecture to incorporate both shallow- and high-level features for memory, which leverages the complementary benefits of pixel and semantic information. Furthermore, to balance the simultaneous utilization of the pixel and semantic memory features, we propose a heterogeneous interaction mechanism to perform pixel-semantic mutual interactions, which explicitly considers their inherent feature discrepancies. Through the design of Pixel-guided Local Alignment Module (PLAM) and Semantic-guided Global Integration Module (SGIM), we achieve delicate integration of the fine-grained details in shallow-level memory and the semantic representations in high-level memory. Our Hierarchical Memory with Heterogeneous Interaction Network (HMHI-Net) consistently achieves state-of-the-art performance across all UVOS and video saliency detection benchmarks. Moreover, HMHI-Net consistently exhibits high performance across different backbones, further demonstrating its superiority and robustness. Project page: this https URL . 

**Abstract (ZH)**: 无监督视频对象分割（UVOS）的目标是在无需任何先验标注的情况下，预测视频中最具显著性的对象的像素级掩码。尽管记忆机制在各种视频分割范式中已被证明是至关重要的，但在UVOS中的应用仅能获得微小的性能提升，尽管其设计相当复杂。我们的分析揭示了现有方法中的一个简单但基础的问题：过度依赖于记忆高层次语义特征。由于缺乏像素级的先验知识，UVOS本质上缺失了细粒度的信息。因此，仅依赖于高层次特征的记忆设计，这些特征主要捕捉抽象的语义线索，不足以生成精确的预测。为了解决这一根本问题，我们提出了一种新颖的分层记忆架构，以结合浅层和高层次特征进行记忆，从而利用像素和语义信息的互补优势。此外，为了平衡同时利用像素和语义记忆特征，我们提出了异构交互机制以在像素和语义之间进行相互作用，明确考虑它们固有的特征差异。通过设计像素引导局部对齐模块（PLAM）和语义引导全局集成模块（SGIM），我们实现了浅层记忆中细粒度细节与高层次记忆中语义表示的精细集成。我们的分层记忆与异构交互网络（HMHI-Net）在所有UVOS和视频显著性检测基准测试中均实现了最先进的性能。此外，HMHI-Net在不同主干网络上的表现始终非常出色，进一步证明了其优越性和鲁棒性。项目页面：this https URL。 

---
# Efficient Spatial-Temporal Modeling for Real-Time Video Analysis: A Unified Framework for Action Recognition and Object Tracking 

**Title (ZH)**: 高效的时空建模方法：统一框架下的动作识别与物体跟踪的实时视频分析 

**Authors**: Shahla John  

**Link**: [PDF](https://arxiv.org/pdf/2507.22421)  

**Abstract**: Real-time video analysis remains a challenging problem in computer vision, requiring efficient processing of both spatial and temporal information while maintaining computational efficiency. Existing approaches often struggle to balance accuracy and speed, particularly in resource-constrained environments. In this work, we present a unified framework that leverages advanced spatial-temporal modeling techniques for simultaneous action recognition and object tracking. Our approach builds upon recent advances in parallel sequence modeling and introduces a novel hierarchical attention mechanism that adaptively focuses on relevant spatial regions across temporal sequences. We demonstrate that our method achieves state-of-the-art performance on standard benchmarks while maintaining real-time inference speeds. Extensive experiments on UCF-101, HMDB-51, and MOT17 datasets show improvements of 3.2% in action recognition accuracy and 2.8% in tracking precision compared to existing methods, with 40% faster inference time. 

**Abstract (ZH)**: 实时视频分析依然是计算机视觉中的一个挑战性问题，需要高效处理时空信息的同时保持计算效率。现有方法往往难以在准确性和速度之间取得平衡，特别是在资源受限的环境中。在本工作中，我们提出了一种统一框架，利用先进的时空建模技术同时进行动作识别和对象跟踪。我们的方法建立在并行序列建模的 recent 进展之上，并引入了一种新颖的分层注意力机制，能够自适应地聚焦于时空序列中的相关时空区域。我们证明，我们的方法在标准基准测试上实现了最先进的性能，同时保持实时推理速度。在 UCF-101、HMDB-51 和 MOT17 数据集上的广泛实验表明，与现有方法相比，动作识别精度提高了 3.2%，跟踪精度提高了 2.8%，推理时间快了 40%。 

---
# Object Recognition Datasets and Challenges: A Review 

**Title (ZH)**: 物体识别数据集与挑战：一篇综述 

**Authors**: Aria Salari, Abtin Djavadifar, Xiangrui Liu, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.22361)  

**Abstract**: Object recognition is among the fundamental tasks in the computer vision applications, paving the path for all other image understanding operations. In every stage of progress in object recognition research, efforts have been made to collect and annotate new datasets to match the capacity of the state-of-the-art algorithms. In recent years, the importance of the size and quality of datasets has been intensified as the utility of the emerging deep network techniques heavily relies on training data. Furthermore, datasets lay a fair benchmarking means for competitions and have proved instrumental to the advancements of object recognition research by providing quantifiable benchmarks for the developed models. Taking a closer look at the characteristics of commonly-used public datasets seems to be an important first step for data-driven and machine learning researchers. In this survey, we provide a detailed analysis of datasets in the highly investigated object recognition areas. More than 160 datasets have been scrutinized through statistics and descriptions. Additionally, we present an overview of the prominent object recognition benchmarks and competitions, along with a description of the metrics widely adopted for evaluation purposes in the computer vision community. All introduced datasets and challenges can be found online at this http URL. 

**Abstract (ZH)**: 物体识别是计算机视觉应用中的基本任务，为所有其他图像理解操作铺平了道路。在物体识别研究的每一阶段进展中，人们都致力于收集和标注新的数据集以匹配最先进的算法的能力。近年来，数据集的规模和质量的重要性得到了加强，因为新兴的深度网络技术的效用在很大程度上依赖于训练数据。此外，数据集为竞赛提供了公平的基准，并通过提供可量化基准来促进物体识别研究的发展。对于数据驱动和机器学习研究人员来说，仔细审视常用公开数据集的特征似乎是一个重要的第一步。在本综述中，我们对高度研究的物体识别领域中的数据集进行了详细的分析。共有超过160个数据集通过统计和描述进行了审查。此外，我们还介绍了著名的物体识别基准和竞赛，并描述了计算机视觉社区中广泛采用的评价指标。所有介绍的数据集和挑战均可通过此链接在线查阅。 

---
# GVD: Guiding Video Diffusion Model for Scalable Video Distillation 

**Title (ZH)**: GVD: 引导视频扩散模型以实现可扩展的视频精简 

**Authors**: Kunyang Li, Jeffrey A Chan Santiago, Sarinda Dhanesh Samarasinghe, Gaowen Liu, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.22360)  

**Abstract**: To address the larger computation and storage requirements associated with large video datasets, video dataset distillation aims to capture spatial and temporal information in a significantly smaller dataset, such that training on the distilled data has comparable performance to training on all of the data. We propose GVD: Guiding Video Diffusion, the first diffusion-based video distillation method. GVD jointly distills spatial and temporal features, ensuring high-fidelity video generation across diverse actions while capturing essential motion information. Our method's diverse yet representative distillations significantly outperform previous state-of-the-art approaches on the MiniUCF and HMDB51 datasets across 5, 10, and 20 Instances Per Class (IPC). Specifically, our method achieves 78.29 percent of the original dataset's performance using only 1.98 percent of the total number of frames in MiniUCF. Additionally, it reaches 73.83 percent of the performance with just 3.30 percent of the frames in HMDB51. Experimental results across benchmark video datasets demonstrate that GVD not only achieves state-of-the-art performance but can also generate higher resolution videos and higher IPC without significantly increasing computational cost. 

**Abstract (ZH)**: 面向大型视频数据集的更大计算和存储需求，视频数据集蒸馏旨在通过显著较小的数据集捕获空间和时间信息，从而使在蒸馏数据上训练的性能与在全部数据上训练相当。我们提出GVD：引导视频扩散，这是一种基于扩散的视频蒸馏方法。GVD联合蒸馏空间和时间特征，确保在各种动作上生成高质量的视频的同时捕捉关键的运动信息。我们的方法在MiniUCF和HMDB51数据集上实现了多样而代表性的蒸馏，显著优于之前的最佳方法，分别在5, 10, 和20类例（IPC）上。具体而言，仅使用MiniUCF总帧数的1.98%，我们的方法达到了原数据集性能的78.29%。在HMDB51上，仅使用3.30%的帧，该方法达到73.83%的性能。基准视频数据集上的实验结果表明，GVD不仅能够实现最先进的性能，而且能够在不显著增加计算成本的情况下生成更高分辨率的视频和更高的IPC。 

---
# Shape Invariant 3D-Variational Autoencoder: Super Resolution in Turbulence flow 

**Title (ZH)**: 形不变3D变分自编码器：湍流流场超分辨率 

**Authors**: Anuraj Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2507.22082)  

**Abstract**: Deep learning provides a versatile suite of methods for extracting structured information from complex datasets, enabling deeper understanding of underlying fluid dynamic phenomena. The field of turbulence modeling, in particular, benefits from the growing availability of high-dimensional data obtained through experiments, field observations, and large-scale simulations spanning multiple spatio-temporal scales. This report presents a concise overview of both classical and deep learningbased approaches to turbulence modeling. It further investigates two specific challenges at the intersection of fluid dynamics and machine learning: the integration of multiscale turbulence models with deep learning architectures, and the application of deep generative models for super-resolution reconstruction 

**Abstract (ZH)**: 深度学习为从复杂数据集提取结构化信息提供了多功能的方法套件，有助于深度理解内在流动力学现象。特别是在湍流建模领域，通过实验、现场观测和多时空尺度的大规模模拟获得的高维数据越来越多，使得该领域受益匪浅。本报告简要介绍了经典方法和基于深度学习的湍流建模方法，并进一步探讨了流体力学和机器学习交汇处的两个具体挑战：多尺度湍流模型与深度学习架构的集成，以及使用深度生成模型进行超分辨率重建。 

---
# Dimensions of Vulnerability in Visual Working Memory: An AI-Driven Approach to Perceptual Comparison 

**Title (ZH)**: 视觉工作记忆中脆弱性的维度：一种基于人工智能的知觉比较方法 

**Authors**: Yuang Cao, Jiachen Zou, Chen Wei, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22067)  

**Abstract**: Human memory exhibits significant vulnerability in cognitive tasks and daily life. Comparisons between visual working memory and new perceptual input (e.g., during cognitive tasks) can lead to unintended memory distortions. Previous studies have reported systematic memory distortions after perceptual comparison, but understanding how perceptual comparison affects memory distortions in real-world objects remains a challenge. Furthermore, identifying what visual features contribute to memory vulnerability presents a novel research question. Here, we propose a novel AI-driven framework that generates naturalistic visual stimuli grounded in behaviorally relevant object dimensions to elicit similarity-induced memory biases. We use two types of stimuli -- image wheels created through dimension editing and dimension wheels generated by dimension activation values -- in three visual working memory (VWM) experiments. These experiments assess memory distortions under three conditions: no perceptual comparison, perceptual comparison with image wheels, and perceptual comparison with dimension wheels. The results show that similar dimensions, like similar images, can also induce memory distortions. Specifically, visual dimensions are more prone to distortion than semantic dimensions, indicating that the object dimensions of naturalistic visual stimuli play a significant role in the vulnerability of memory. 

**Abstract (ZH)**: 人类记忆在认知任务和日常生活中表现出显著的脆弱性。感知比较（如在认知任务中与新感知输入对比）可能导致无意中的记忆失真。尽管先前的研究报告了感知比较后的系统性记忆失真，但了解感知比较如何影响真实世界对象的记忆失真仍是挑战。此外，确定哪些视觉特征导致记忆脆弱性是一个新的研究问题。在这里，我们提出了一种新的基于AI的框架，该框架生成与行为相关对象维度一致的自然视觉刺激，以引发相似性引起的记忆偏差。我们使用两种类型的刺激——通过维度编辑创建的图像轮盘和通过维度激活值生成的维度轮盘——在三个视觉工作记忆（VWM）实验中进行测试。这些实验在三种情况下评估记忆失真：无感知比较、与图像轮盘进行感知比较以及与维度轮盘进行感知比较。结果表明，类似的维度，就像类似的图像一样，也能引起记忆失真。具体来说，视觉维度比语义维度更容易失真，表明自然视觉刺激的对象维度在记忆的脆弱性中起着重要作用。 

---
