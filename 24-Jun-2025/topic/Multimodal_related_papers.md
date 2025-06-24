# DefFusionNet: Learning Multimodal Goal Shapes for Deformable Object Manipulation via a Diffusion-based Probabilistic Model 

**Title (ZH)**: DefFusionNet：基于扩散概率模型的可变形物体 manipulation 多模态目标形状学习 

**Authors**: Bao Thach, Siyeon Kim, Britton Jordan, Mohanraj Shanthi, Tanner Watts, Shing-Hei Ho, James M. Ferguson, Tucker Hermans, Alan Kuntz  

**Link**: [PDF](https://arxiv.org/pdf/2506.18779)  

**Abstract**: Deformable object manipulation is critical to many real-world robotic applications, ranging from surgical robotics and soft material handling in manufacturing to household tasks like laundry folding. At the core of this important robotic field is shape servoing, a task focused on controlling deformable objects into desired shapes. The shape servoing formulation requires the specification of a goal shape. However, most prior works in shape servoing rely on impractical goal shape acquisition methods, such as laborious domain-knowledge engineering or manual manipulation. DefGoalNet previously posed the current state-of-the-art solution to this problem, which learns deformable object goal shapes directly from a small number of human demonstrations. However, it significantly struggles in multi-modal settings, where multiple distinct goal shapes can all lead to successful task completion. As a deterministic model, DefGoalNet collapses these possibilities into a single averaged solution, often resulting in an unusable goal. In this paper, we address this problem by developing DefFusionNet, a novel neural network that leverages the diffusion probabilistic model to learn a distribution over all valid goal shapes rather than predicting a single deterministic outcome. This enables the generation of diverse goal shapes and avoids the averaging artifacts. We demonstrate our method's effectiveness on robotic tasks inspired by both manufacturing and surgical applications, both in simulation and on a physical robot. Our work is the first generative model capable of producing a diverse, multi-modal set of deformable object goals for real-world robotic applications. 

**Abstract (ZH)**: 可变形对象操控对于许多现实世界的机器人应用至关重要，从外科机器人和制造业中的软材料处理到家务任务如衣物整理。这一重要机器人领域的核心是形状伺服技术，其旨在控制可变形对象达到预期形状。形状伺服的建模需要设定目标形状。然而，大多数现有的形状伺服研究依赖于不实际的目标形状获取方法，如繁重的领域知识工程或手动操作。DefGoalNet之前提出了当前最先进的解决方案，可以从少量的人类演示中直接学习可变形对象的目标形状。然而，它在多模态环境中表现不佳，在这种环境中，多种不同的目标形状都可以导致任务成功完成。作为确定性模型，DefGoalNet将这些可能性简化为单一的平均解，经常导致无法使用的目标。在本文中，我们通过开发DefFusionNet，一种新颖的神经网络，利用扩散概率模型来学习所有有效目标形状的分布，而不是预测单一确定性结果，以解决这一问题。这使我们能够生成多种多样的目标形状，并避免了平均化的缺陷。我们在模拟和实际机器人上展示了该方法在制造和外科应用启发的机器人任务中的有效性。我们的工作是首个能够为现实世界机器人应用生成多样化、多模态可变形对象目标的生成模型。 

---
# VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models 

**Title (ZH)**: VLA-OS：构建与剖析视觉-语言-行动模型中的规划表示与范式 

**Authors**: Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17561)  

**Abstract**: Recent studies on Vision-Language-Action (VLA) models have shifted from the end-to-end action-generation paradigm toward a pipeline involving task planning followed by action generation, demonstrating improved performance on various complex, long-horizon manipulation tasks. However, existing approaches vary significantly in terms of network architectures, planning paradigms, representations, and training data sources, making it challenging for researchers to identify the precise sources of performance gains and components to be further improved. To systematically investigate the impacts of different planning paradigms and representations isolating from network architectures and training data, in this paper, we introduce VLA-OS, a unified VLA architecture series capable of various task planning paradigms, and design a comprehensive suite of controlled experiments across diverse object categories (rigid and deformable), visual modalities (2D and 3D), environments (simulation and real-world), and end-effectors (grippers and dexterous hands). Our results demonstrate that: 1) visually grounded planning representations are generally better than language planning representations; 2) the Hierarchical-VLA paradigm generally achieves superior or comparable performance than other paradigms on task performance, pretraining, generalization ability, scalability, and continual learning ability, albeit at the cost of slower training and inference speeds. 

**Abstract (ZH)**: Recent Studies on Vision-Language-Action (VLA) Models Have Shifted toward a Planning-Driven Pipeline with Improved Performance on Complex Manipulation Tasks: Introducing VLA-OS 

---
# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval 

**Title (ZH)**: jina-embeddings-v4：通用多模态多语言检索嵌入 

**Authors**: Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh Eslami, Scott Martens, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18902)  

**Abstract**: We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-based information retrieval, cross-modal semantic similarity, and programming code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single- modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval. 

**Abstract (ZH)**: 我们介绍了jina-embeddings-v4，这是一个包含38亿参数的多模态嵌入模型，通过一种新型架构统一文本和图像表示，并支持在晚期交互风格中的单向量和多向量嵌入。该模型融合了特定任务的低秩适应（LoRA）适配器，以优化在各种检索情景下的性能，包括基于查询的信息检索、跨模态语义相似性和编程代码搜索。综合评估表明，jina-embeddings-v4 在单模态和跨模态检索任务上都达到了最先进的性能，特别擅长处理图表、表格、图表和混合媒体格式等视觉丰富内容。为了方便评估这种能力，我们还引入了Jina-VDR，这是一种专门为视觉丰富图像检索设计的新基准。 

---
# Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations 

**Title (ZH)**: 视觉作为一种方言：通过文本对齐表示统一视觉理解与生成 

**Authors**: Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18898)  

**Abstract**: This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at this https URL 

**Abstract (ZH)**: 本文提出了一种多模态框架，尝试在共享的离散语义表示中统一视觉理解和生成。其核心是文本对齐分词器（TA-Tok），它使用大语言模型（LLM）词汇表投影得到的文本对齐码本将图像转换为离散词元。通过将视觉和文本整合到一个扩大的词汇表统一空间中，我们的多模态LLM Tar能够通过共享接口进行跨模态输入和输出，无需特定模态的设计。此外，我们提出了自适应编码和解码以平衡效率和视觉细节，并提出了一种生成性反分词器以生成高保真视觉输出。为了满足多样化的解码需求，我们利用了两种互补的反分词器：快速自回归模型和基于扩散的模型。为了增强模态融合，我们研究了先进的预训练任务，证明了在视觉理解和生成方面的改进。跨基准实验表明，Tar与现有的多模态LLM方法相当或超越，实现了更快的收敛速度和更高的训练效率。代码、模型和数据可在以下链接获取。 

---
# OmniGen2: Exploration to Advanced Multimodal Generation 

**Title (ZH)**: OmniGen2：探索高级多模态生成 

**Authors**: Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18871)  

**Abstract**: In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: this https URL GitHub Link: this https URL 

**Abstract (ZH)**: 本研究介绍了OmniGen2，这是一个多功能且开源的生成模型，旨在为包括文本到图像、图像编辑和上下文生成在内的多种生成任务提供统一解决方案。与OmniGen v1不同，OmniGen2配备了用于文本和图像模态的两个独立解码路径，使用不同的参数和解耦的图像分词器。这一设计使得OmniGen2能够在不重新适应VAE输入的情况下建立在现有的多模态理解模型之上，从而保留了原始的文本生成能力。为了方便训练OmniGen2，我们开发了全面的数据构建管道，涵盖图像编辑和上下文生成数据。此外，我们还为图像生成任务引入了一种定制的反射机制，并基于OmniGen2构建了一个专门的反射数据集。尽管参数规模相对较小，OmniGen2在包括文本到图像和图像编辑在内的多个任务基准测试中达到了竞争性的成果。为了进一步评估上下文生成，也称为主题驱动任务，我们引入了一个名为OmniContext的新基准。在开源模型中，OmniGen2在一致性方面达到了最先进的性能。我们将在未来的研究中发布我们的模型、训练代码、数据集和数据构建管道。项目页面：https://this.url/project OmniGen2 GitHub链接：https://this.url/code 

---
# OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation 

**Title (ZH)**: OmniAvatar：基于自适应身体动画的高效音频驱动avatar视频生成 

**Authors**: Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18866)  

**Abstract**: Significant progress has been made in audio-driven human animation, while most existing methods focus mainly on facial movements, limiting their ability to create full-body animations with natural synchronization and fluidity. They also struggle with precise prompt control for fine-grained generation. To tackle these challenges, we introduce OmniAvatar, an innovative audio-driven full-body video generation model that enhances human animation with improved lip-sync accuracy and natural movements. OmniAvatar introduces a pixel-wise multi-hierarchical audio embedding strategy to better capture audio features in the latent space, enhancing lip-syncing across diverse scenes. To preserve the capability for prompt-driven control of foundation models while effectively incorporating audio features, we employ a LoRA-based training approach. Extensive experiments show that OmniAvatar surpasses existing models in both facial and semi-body video generation, offering precise text-based control for creating videos in various domains, such as podcasts, human interactions, dynamic scenes, and singing. Our project page is this https URL. 

**Abstract (ZH)**: 基于音频的全身视频生成取得了显著进展，尽管现有方法主要集中在面部运动上，限制了其创建自然同步和流畅全身动画的能力。它们在精细生成时也难以实现精确的提示控制。为应对这些挑战，我们提出了OmniAvatar，这是一种创新的基于音频的全身视频生成模型，通过改进唇部同步准确性和自然运动来增强人类动画。OmniAvatar 引入了一种像素级多层级音频嵌入策略，以更好地在潜在空间中捕捉音频特征，从而在多样化场景中提高唇部同步效果。为了保留基础模型驱动提示控制的能力同时有效融入音频特征，我们采用了基于LoRA的训练方法。广泛实验表明，OmniAvatar 在面部和半身视频生成方面均超越了现有模型，提供了精细的文本控制以在播客、人类互动、动态场景和唱歌等多种领域创建视频。我们的项目页面请点击：[该项目链接]。 

---
# TAMMs: Temporal-Aware Multimodal Model for Satellite Image Change Understanding and Forecasting 

**Title (ZH)**: TAMMs：时间感知多模态模型在卫星图像变化理解与预测中的应用 

**Authors**: Zhongbin Guo, Yuhao Wang, Ping Jian, Xinyue Chen, Wei Peng, Ertai E  

**Link**: [PDF](https://arxiv.org/pdf/2506.18862)  

**Abstract**: Satellite image time-series analysis demands fine-grained spatial-temporal reasoning, which remains a challenge for existing multimodal large language models (MLLMs). In this work, we study the capabilities of MLLMs on a novel task that jointly targets temporal change understanding and future scene generation, aiming to assess their potential for modeling complex multimodal dynamics over time. We propose TAMMs, a Temporal-Aware Multimodal Model for satellite image change understanding and forecasting, which enhances frozen MLLMs with lightweight temporal modules for structured sequence encoding and contextual prompting. To guide future image generation, TAMMs introduces a Semantic-Fused Control Injection (SFCI) mechanism that adaptively combines high-level semantic reasoning and structural priors within an enhanced ControlNet. This dual-path conditioning enables temporally consistent and semantically grounded image synthesis. Experiments demonstrate that TAMMs outperforms strong MLLM baselines in both temporal change understanding and future image forecasting tasks, highlighting how carefully designed temporal reasoning and semantic fusion can unlock the full potential of MLLMs for spatio-temporal understanding. 

**Abstract (ZH)**: 卫星图像时间序列分析要求精细的空间-时间推理，这是现有多模态大型语言模型（MLLMs）面临的一项挑战。在这项工作中，我们研究了MLLMs在一项新颖任务上的能力，该任务旨在同时理解时间变化和生成未来场景，以评估其在建模复杂多模态动态方面的潜力。我们提出了一种名为TAMMs的时间感知多模态模型，通过引入轻量级的时间模块增强冻结的MLLMs，以实现结构化序列编码和上下文提示。为了指导未来图像生成，TAMMs引入了一种语义融合控制注入（SFCI）机制，该机制可适应性结合高级语义推理和结构先验，从而实现时空一致性和语义指导的图像合成。实验结果显示，TAMMs在时间变化理解任务和未来图像预测任务上均优于强大的MLLM基线模型，突显了精心设计的时间推理和语义融合如何为MLLMs的时空理解潜力打开大门。 

---
# Generalizing Vision-Language Models to Novel Domains: A Comprehensive Survey 

**Title (ZH)**: 将视觉-语言模型扩展到新型域：一个全面的综述 

**Authors**: Xinyao Li, Jingjing Li, Fengling Li, Lei Zhu, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18504)  

**Abstract**: Recently, vision-language pretraining has emerged as a transformative technique that integrates the strengths of both visual and textual modalities, resulting in powerful vision-language models (VLMs). Leveraging web-scale pretraining data, these models exhibit strong zero-shot capabilities. However, their performance often deteriorates when confronted with domain-specific or specialized generalization tasks. To address this, a growing body of research focuses on transferring or generalizing the rich knowledge embedded in VLMs to various downstream applications. This survey aims to comprehensively summarize the generalization settings, methodologies, benchmarking and results in VLM literatures. Delving into the typical VLM structures, current literatures are categorized into prompt-based, parameter-based and feature-based methods according to the transferred modules. The differences and characteristics in each category are furthered summarized and discussed by revisiting the typical transfer learning (TL) settings, providing novel interpretations for TL in the era of VLMs. Popular benchmarks for VLM generalization are further introduced with thorough performance comparisons among the reviewed methods. Following the advances in large-scale generalizable pretraining, this survey also discusses the relations and differences between VLMs and up-to-date multimodal large language models (MLLM), e.g., DeepSeek-VL. By systematically reviewing the surging literatures in vision-language research from a novel and practical generalization prospective, this survey contributes to a clear landscape of current and future multimodal researches. 

**Abstract (ZH)**: 近期，视觉-语言预训练作为一项革新性技术，将视觉和文本模态的优势相结合，产生了强大的视觉-语言模型（VLMs）。这些模型凭借网络规模级别的预训练数据，展现出强大的零样本能力。然而，它们在面对特定领域或专门化泛化任务时，性能往往会下降。为解决这一问题，越来越多的研究致力于将视觉-语言模型中丰富的知识转移到各种下游应用中。本文综述旨在全面总结视觉-语言模型泛化设置、方法、基准测试和结果。通过探究典型视觉-语言模型结构，当前文献根据转移模块被分类为提示基、参数基和特征基方法，并通过回顾典型的迁移学习设置，进一步总结和讨论每个类别中的差异和特点，为视觉-语言模型时代提供了新颖的迁移学习解释。此外，本文还介绍了视觉-语言模型泛化的常用基准，并对所审查方法进行了彻底的性能比较。随着大规模可泛化预训练的进展，本文还讨论了视觉-语言模型与最新多模态大语言模型（MLLM），如DeepSeek-VL之间的关系和差异。通过从新颖和实用的泛化视角系统地回顾视觉-语言研究文献，本文为当前和未来的多模态研究奠定了清晰的框架。 

---
# Multimodal Fusion SLAM with Fourier Attention 

**Title (ZH)**: Fourier注意力融合多模态SLAM 

**Authors**: Youjie Zhou, Guofeng Mei, Yiming Wang, Yi Wan, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18204)  

**Abstract**: Visual SLAM is particularly challenging in environments affected by noise, varying lighting conditions, and darkness. Learning-based optical flow algorithms can leverage multiple modalities to address these challenges, but traditional optical flow-based visual SLAM approaches often require significant computational this http URL overcome this limitation, we propose FMF-SLAM, an efficient multimodal fusion SLAM method that utilizes fast Fourier transform (FFT) to enhance the algorithm efficiency. Specifically, we introduce a novel Fourier-based self-attention and cross-attention mechanism to extract features from RGB and depth signals. We further enhance the interaction of multimodal features by incorporating multi-scale knowledge distillation across modalities. We also demonstrate the practical feasibility of FMF-SLAM in real-world scenarios with real time performance by integrating it with a security robot by fusing with a global positioning module GNSS-RTK and global Bundle Adjustment. Our approach is validated using video sequences from TUM, TartanAir, and our real-world datasets, showcasing state-of-the-art performance under noisy, varying lighting, and dark this http URL code and datasets are available at this https URL. 

**Abstract (ZH)**: 视觉SLAM在受噪声、变化光照和黑暗影响的环境中特别具有挑战性。基于学习的光流算法可以通过利用多种模态来应对这些挑战，但传统的基于光流的视觉SLAM方法往往需要大量计算资源以克服这一限制。为了解决这个问题，我们提出了一种名为FMF-SLAM的高效多模态融合SLAM方法，该方法利用快速傅里叶变换（FFT）以提高算法效率。具体而言，我们引入了一种基于傅里叶的自注意力和跨注意力机制来从RGB和深度信号中提取特征，并通过跨模态的多尺度知识蒸馏增强了多模态特征的交互性。我们还通过将其与GNSS-RTK全球定位模块和全局 bundle 调整相结合的方式，实现在实时场景中的实际可行性。我们的方法使用来自TUM、TartanAir以及我们自己的现实世界数据集的视频序列进行验证，在噪声、变化光照和黑暗条件下展示了最先进的性能。相关代码和数据集可在以下网址获取。 

---
# ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image Generation 

**Title (ZH)**: ShareGPT-4o-Image：基于GPT-4o级图像生成的多模态模型对齐 

**Authors**: Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18095)  

**Abstract**: Recent advances in multimodal generative models have unlocked photorealistic, instruction-aligned image generation, yet leading systems like GPT-4o-Image remain proprietary and inaccessible. To democratize these capabilities, we present ShareGPT-4o-Image, the first dataset comprising 45K text-to-image and 46K text-and-image-to-image data, all synthesized using GPT-4o's image generation capabilities for distilling its advanced image generation abilities. Leveraging this dataset, we develop Janus-4o, a multimodal large language model capable of both text-to-image and text-and-image-to-image generation. Janus-4o not only significantly improves text-to-image generation over its predecessor, Janus-Pro, but also newly supports text-and-image-to-image generation. Notably, it achieves impressive performance in text-and-image-to-image generation from scratch, using only 91K synthetic samples and 6 hours of training on an 8 A800-GPU machine. We hope the release of ShareGPT-4o-Image and Janus-4o will foster open research in photorealistic, instruction-aligned image generation. 

**Abstract (ZH)**: Recent advances in多模态生成模型的Recent进展在真实感、指令对齐图像生成方面的突破，然而像GPT-4o-Image这样的领先系统仍然保持专有和不可访问状态。为了普及这些能力，我们提出ShareGPT-4o-Image，这是一个包含45,000个文本到图像和46,000个文本和图像到图像数据集，所有数据均使用GPT-4o的图像生成能力以提炼其先进的图像生成能力。利用该数据集，我们开发了Janus-4o，一个双模态大语言模型，能够进行文本到图像和文本和图像到图像的生成。Janus-4o不仅在文本到图像生成方面显著优于其 predecessor Janus-Pro，而且还新增了文本和图像到图像生成能力。值得注意的是，它能够从零开始在真实感、指令对齐图像生成方面取得出色表现，仅需91,000个合成样本和6小时的8 A800-GPU机器训练时间。我们希望ShareGPT-4o-Image和Janus-4o的发布能够促进真实感、指令对齐图像生成的开放研究。 

---
# Multimodal Medical Image Binding via Shared Text Embeddings 

**Title (ZH)**: 基于共享文本嵌入的多模态医疗图像融合 

**Authors**: Yunhao Liu, Suyang Xi, Shiqi Liu, Hong Ding, Chicheng Jin, Chenxi Yang, Junjun He, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18072)  

**Abstract**: Medical image analysis increasingly relies on the integration of multiple imaging modalities to capture complementary anatomical and functional information, enabling more accurate diagnosis and treatment planning. Achieving aligned feature representations across these diverse modalities is therefore important for effective multimodal analysis. While contrastive language-image pre-training (CLIP) and its variant have enabled image-text alignments, they require explicitly paired data between arbitrary two modalities, which is difficult to acquire in medical contexts. To address the gap, we present Multimodal Medical Image Binding with Text (M\textsuperscript{3}Bind), a novel pre-training framework that enables seamless alignment of multiple medical imaging modalities through a shared text representation space without requiring explicit paired data between any two medical image modalities. Specifically, based on the insight that different images can naturally bind with text, M\textsuperscript{3}Bind first fine-tunes pre-trained CLIP-like image-text models to align their modality-specific text embedding space while preserving their original image-text alignments. Subsequently, we distill these modality-specific text encoders into a unified model, creating a shared text embedding space. Experiments on X-ray, CT, retina, ECG, and pathological images on multiple downstream tasks demonstrate that M\textsuperscript{3}Bind achieves state-of-the-art performance in zero-shot, few-shot classification and cross-modal retrieval tasks compared to its CLIP-like counterparts. These results validate M\textsuperscript{3}Bind's effectiveness in achieving cross-image-modal alignment for medical analysis. 

**Abstract (ZH)**: 多模态医学图像与文本的绑定（M³Bind） 

---
# MUPA: Towards Multi-Path Agentic Reasoning for Grounded Video Question Answering 

**Title (ZH)**: MUPA: 向多路径主体推理的地基视频问答迈进 

**Authors**: Jisheng Dang, Huilin Song, Junbin Xiao, Bimei Wang, Han Peng, Haoxuan Li, Xun Yang, Meng Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18071)  

**Abstract**: Grounded Video Question Answering (Grounded VideoQA) requires aligning textual answers with explicit visual evidence. However, modern multimodal models often rely on linguistic priors and spurious correlations, resulting in poorly grounded predictions. In this work, we propose MUPA, a cooperative MUlti-Path Agentic approach that unifies video grounding, question answering, answer reflection and aggregation to tackle Grounded VideoQA. MUPA features three distinct reasoning paths on the interplay of grounding and QA agents in different chronological orders, along with a dedicated reflection agent to judge and aggregate the multi-path results to accomplish consistent QA and grounding. This design markedly improves grounding fidelity without sacrificing answer accuracy. Despite using only 2B parameters, our method outperforms all 7B-scale competitors. When scaled to 7B parameters, MUPA establishes new state-of-the-art results, with Acc@GQA of 30.3% and 47.4% on NExT-GQA and DeVE-QA respectively, demonstrating MUPA' effectiveness towards trustworthy video-language understanding. Our code is available in this https URL. 

**Abstract (ZH)**: 基于视觉的视频问答（Grounded VideoQA）要求将文本回答与明确的视觉证据对齐。然而，现代多模态模型往往依赖于语言先验和虚假的相关性，导致预测与可视化证据契合度不高。在本工作中，我们提出了一种新的协作多路径代理方法MUPA，统一了视频定位、问答、回答反思和结果聚合，以解决基于视觉的视频问答问题。MUPA包含三个不同的推理路径，这些路径在不同时间顺序上处理定位和问答代理之间的相互作用，并配备了一个专门的反思代理，用于判断和聚合多路径结果以实现一致的问答和定位。这种设计显著提高了定位准确性，同时保持答案的准确性。尽管仅使用20亿参数，我们的方法在所有70亿参数规模的竞争者中表现更优。当扩展到70亿参数时，MUPA建立了新的 state-of-the-art 结果，分别在NExT-GQA和DeVE-QA数据集上实现了30.3%和47.4%的Acc@GQA，证明了MUPA对于可信赖的视频-语言理解的有效性。我们的代码可通过此链接获取：https://github.com/alibaba/Qwen 

---
# PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding 

**Title (ZH)**: PP-DocBee2: 提升基线模型的多模态文档理解方法与高效数据集 

**Authors**: Kui Huang, Xinrong Chen, Wenyu Lv, Jincheng Liao, Guanzhong Wang, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18023)  

**Abstract**: This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: PP-DocBee2: 一种高级版的PP-DocBee，用于增强多模态文档理解 

---
# SurgVidLM: Towards Multi-grained Surgical Video Understanding with Large Language Model 

**Title (ZH)**: SurgVidLM：利用大型语言模型实现多粒度手术视频理解 

**Authors**: Guankun Wang, Wenjin Mo, Junyi Wang, Long Bai, Kun Yuan, Ming Hu, Jinlin Wu, Junjun He, Yiming Huang, Nicolas Padoy, Zhen Lei, Hongbin Liu, Nassir Navab, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.17873)  

**Abstract**: Recent advances in Multimodal Large Language Models have demonstrated great potential in the medical domain, facilitating users to understand surgical scenes and procedures. Beyond image-based methods, the exploration of Video Large Language Models (Vid-LLMs) has emerged as a promising avenue for capturing the complex sequences of information involved in surgery. However, there is still a lack of Vid-LLMs specialized for fine-grained surgical video understanding tasks, which is crucial for analyzing specific processes or details within a surgical procedure. To bridge this gap, we propose SurgVidLM, the first video language model designed to address both full and fine-grained surgical video comprehension. To train our SurgVidLM, we construct the SVU-31K dataset which consists of over 31K video-instruction pairs, enabling both holistic understanding and detailed analysis of surgical procedures. Furthermore, we introduce the StageFocus mechanism which is a two-stage framework performing the multi-grained, progressive understanding of surgical videos. We also develop the Multi-frequency Fusion Attention to effectively integrate low and high-frequency visual tokens, ensuring the retention of critical information. Experimental results demonstrate that SurgVidLM significantly outperforms state-of-the-art Vid-LLMs in both full and fine-grained video understanding tasks, showcasing its superior capability in capturing complex procedural contexts. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models for Fine-Grained Surgical Video Understanding 

---
# Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs 

**Title (ZH)**: 利用多模态大语言模型扩展医学案例检索任务的相关性评估 

**Authors**: Catarina Pires, Sérgio Nunes, Luís Filipe Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17782)  

**Abstract**: Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks. 

**Abstract (ZH)**: 利用多模态大型语言模型扩展相关性判断以支持医学和多模态信息检索评估 

---
# Multimodal Political Bias Identification and Neutralization 

**Title (ZH)**: 多模态政治偏见识别与中和 

**Authors**: Cedric Bernard, Xavier Pleimling, Amun Kharel, Chase Vickery  

**Link**: [PDF](https://arxiv.org/pdf/2506.17372)  

**Abstract**: Due to the presence of political echo chambers, it becomes imperative to detect and remove subjective bias and emotionally charged language from both the text and images of political articles. However, prior work has focused on solely the text portion of the bias rather than both the text and image portions. This is a problem because the images are just as powerful of a medium to communicate information as text is. To that end, we present a model that leverages both text and image bias which consists of four different steps. Image Text Alignment focuses on semantically aligning images based on their bias through CLIP models. Image Bias Scoring determines the appropriate bias score of images via a ViT classifier. Text De-Biasing focuses on detecting biased words and phrases and neutralizing them through BERT models. These three steps all culminate to the final step of debiasing, which replaces the text and the image with neutralized or reduced counterparts, which for images is done by comparing the bias scores. The results so far indicate that this approach is promising, with the text debiasing strategy being able to identify many potential biased words and phrases, and the ViT model showcasing effective training. The semantic alignment model also is efficient. However, more time, particularly in training, and resources are needed to obtain better results. A human evaluation portion was also proposed to ensure semantic consistency of the newly generated text and images. 

**Abstract (ZH)**: 由于存在政治回音室，检测并去除政治文章中文字和图像中的主观偏见和情绪化语言变得至关重要。然而，以往研究主要关注文字部分的偏见而非文字和图像两部分。鉴于图像同样是强有力的传播信息的媒介，我们提出了一种结合文字和图像偏见的模型，该模型包含四个步骤。图像文字对齐旨在通过CLIP模型在语义层面对图像进行对齐。图像偏见评分通过ViT分类器确定图像的适当偏见分数。文字脱偏重点在于检测具有偏见的词汇和短语并通过BERT模型进行中立化。这些三个步骤最终汇聚到脱偏的最后一步，即用中立化或减弱后的文字和图像替换原文字和图像，对图像而言是通过比较偏见分数实现的。初步结果表明，该方法具有潜力，文字脱偏策略能够识别出许多潜在的偏见词汇和短语，ViT模型也展示了有效的训练效果。语义对齐模型也具有高效性。不过，还需要更多时间和资源来获得更好的结果，并提议进行人工评估以确保新生成的文字和图像的一致性。 

---
# AI-based Multimodal Biometrics for Detecting Smartphone Distractions: Application to Online Learning 

**Title (ZH)**: 基于AI的多模态生物特征识别智能手机分心检测：在线学习应用 

**Authors**: Alvaro Becerra, Roberto Daza, Ruth Cobos, Aythami Morales, Mutlu Cukurova, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2506.17364)  

**Abstract**: This work investigates the use of multimodal biometrics to detect distractions caused by smartphone use during tasks that require sustained attention, with a focus on computer-based online learning. Although the methods are applicable to various domains, such as autonomous driving, we concentrate on the challenges learners face in maintaining engagement amid internal (e.g., motivation), system-related (e.g., course design) and contextual (e.g., smartphone use) factors. Traditional learning platforms often lack detailed behavioral data, but Multimodal Learning Analytics (MMLA) and biosensors provide new insights into learner attention. We propose an AI-based approach that leverages physiological signals and head pose data to detect phone use. Our results show that single biometric signals, such as brain waves or heart rate, offer limited accuracy, while head pose alone achieves 87%. A multimodal model combining all signals reaches 91% accuracy, highlighting the benefits of integration. We conclude by discussing the implications and limitations of deploying these models for real-time support in online learning environments. 

**Abstract (ZH)**: 本研究探讨了多模态生物特征识别在检测执行需要持续注意力的任务时因智能手机使用引起的分心现象中的应用，重点关注基于计算机的在线学习领域。尽管所采用的方法适用于多个领域，例如自动驾驶，但我们着重于学习者在面对内在（例如，动机）、系统相关（例如，课程设计）和情境因素（例如，智能手机使用）带来的挑战时保持参与的困难。传统的学习平台通常缺乏详细的行为数据，但多模态学习分析（MMLA）和生物传感器提供了关于学习者注意力的新见解。我们提出了一种基于人工智能的方法，利用生理信号和头部姿态数据来检测手机使用情况。研究结果表明，单一的生物特征信号（如脑电波或心率）的准确性有限，而单独使用头部姿态的准确性为87%。结合所有信号的多模态模型的准确性达到91%，这突显了集成的优势。最后，我们讨论了在在线学习环境中部署这些模型的implications和限制。 

---
# P2MFDS: A Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments 

**Title (ZH)**: P2MFDS：一种保护隐私的多模态浴室环境中老年人跌倒检测系统 

**Authors**: Haitian Wang, Yiren Wang, Xinyu Wang, Yumeng Miao, Yuliang Zhang, Yu Zhang, Atif Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2506.17332)  

**Abstract**: By 2050, people aged 65 and over are projected to make up 16 percent of the global population. As aging is closely associated with increased fall risk, particularly in wet and confined environments such as bathrooms where over 80 percent of falls occur. Although recent research has increasingly focused on non-intrusive, privacy-preserving approaches that do not rely on wearable devices or video-based monitoring, these efforts have not fully overcome the limitations of existing unimodal systems (e.g., WiFi-, infrared-, or mmWave-based), which are prone to reduced accuracy in complex environments. These limitations stem from fundamental constraints in unimodal sensing, including system bias and environmental interference, such as multipath fading in WiFi-based systems and drastic temperature changes in infrared-based methods. To address these challenges, we propose a Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments. First, we develop a sensor evaluation framework to select and fuse millimeter-wave radar with 3D vibration sensing, and use it to construct and preprocess a large-scale, privacy-preserving multimodal dataset in real bathroom settings, which will be released upon publication. Second, we introduce P2MFDS, a dual-stream network combining a CNN-BiLSTM-Attention branch for radar motion dynamics with a multi-scale CNN-SEBlock-Self-Attention branch for vibration impact detection. By uniting macro- and micro-scale features, P2MFDS delivers significant gains in accuracy and recall over state-of-the-art approaches. Code and pretrained models will be made available at: this https URL. 

**Abstract (ZH)**: 到2050年，65岁及以上人口预计将占全球人口的16%。由于老化与增加的跌倒风险密切相关，特别是在如浴室这样的潮湿和受限环境中，超过80%的跌倒事件在此类环境中发生。尽管近期研究 increasingly关注非侵入性、保护隐私的方法，这些方法不依赖于可穿戴设备或基于视频的监控，但这些努力仍未完全克服现有单一模态系统的局限性（例如，基于WiFi、红外或毫米波的方法在复杂环境中的准确率较低）。这些局限性源于单一模态传感的基本限制，包括系统偏差和环境干扰，如WiFi系统中的多径衰落和红外方法中的剧烈温度变化。为解决这些挑战，我们提出了一种隐私保护的多模态跌倒检测系统，专门适用于浴室环境中的老年人。首先，我们开发了一个传感器评估框架，选择并融合毫米波雷达与3D振动传感，并在实际的浴室环境中构建和预处理了一个大规模的隐私保护多模态数据集，该数据集将在发表时公开。其次，我们引入了P2MFDS，这是一种双流网络，结合了基于CNN-BiLSTM-Attention支路的雷达运动动力学检测与基于多尺度CNN-SEBlock-Self-Attention支路的振动冲击检测。通过结合宏观和微观特征，P2MFDS在准确性和召回率方面显著优于当前最先进的方法。代码和预训练模型将在此处发布：this https URL。 

---
# Efficient Quantification of Multimodal Interaction at Sample Level 

**Title (ZH)**: 多模态交互在样本级的高效量化 

**Authors**: Zequn Yang, Hongfa Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17248)  

**Abstract**: Interactions between modalities -- redundancy, uniqueness, and synergy -- collectively determine the composition of multimodal information. Understanding these interactions is crucial for analyzing information dynamics in multimodal systems, yet their accurate sample-level quantification presents significant theoretical and computational challenges. To address this, we introduce the Lightweight Sample-wise Multimodal Interaction (LSMI) estimator, rigorously grounded in pointwise information theory. We first develop a redundancy estimation framework, employing an appropriate pointwise information measure to quantify this most decomposable and measurable interaction. Building upon this, we propose a general interaction estimation method that employs efficient entropy estimation, specifically tailored for sample-wise estimation in continuous distributions. Extensive experiments on synthetic and real-world datasets validate LSMI's precision and efficiency. Crucially, our sample-wise approach reveals fine-grained sample- and category-level dynamics within multimodal data, enabling practical applications such as redundancy-informed sample partitioning, targeted knowledge distillation, and interaction-aware model ensembling. The code is available at this https URL. 

**Abstract (ZH)**: 不同模态之间的交互作用——冗余性、独特性和协同作用——共同决定了多模态信息的组成。理解这些交互作用对于分析多模态系统的信息动力学至关重要，但对其准确的样本级别量化面临重大的理论和计算挑战。为了解决这一问题，我们引入了基于点信息理论的轻量级样本级多模态交互（LSMI）估计器。我们首先开发了一种冗余估计框架，使用适当的信息测度来量化这种最具可分解性和可测量性的交互作用。在此基础上，我们提出了一种通用的交互估计方法，使用高效的熵估计方法，专门针对连续分布的样本级别估计进行了优化。在合成和真实世界数据集上的广泛实验验证了LSMI的精度和效率。最关键的是，我们的样本级别方法揭示了多模态数据中的细粒度样本级和类别级动态，使其在冗余指导的样本分割、靶向知识蒸馏和交互感知模型集成等实际应用中具有重要意义。代码可在以下链接获取：这个 https URL。 

---
