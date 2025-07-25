# Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving 

**Title (ZH)**: 使用语言和道路手册告知自动驾驶中的地图重建 

**Authors**: Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi  

**Link**: [PDF](https://arxiv.org/pdf/2506.10317)  

**Abstract**: Lane-topology prediction is a critical component of safe and reliable autonomous navigation. An accurate understanding of the road environment aids this task. We observe that this information often follows conventions encoded in natural language, through design codes that reflect the road structure and road names that capture the road functionality. We augment this information in a lightweight manner to SMERF, a map-prior-based online lane-topology prediction model, by combining structured road metadata from OSM maps and lane-width priors from Road design manuals with the road centerline encodings. We evaluate our method on two geo-diverse complex intersection scenarios. Our method shows improvement in both lane and traffic element detection and their association. We report results using four topology-aware metrics to comprehensively assess the model performance. These results demonstrate the ability of our approach to generalize and scale to diverse topologies and conditions. 

**Abstract (ZH)**: 车道拓扑预测是实现安全可靠自主导航的关键组件。准确理解道路环境有助于这一任务。我们观察到这些信息通常遵循在自然语言中编码的惯例，通过反映道路结构的设计代码和捕捉道路功能的道路名称体现。我们以轻量级的方式将这些信息补充到基于地图先验的在线车道拓扑预测模型SMERF中，结合开放street地图（OSM）中的结构化道路元数据和道路设计手册中的车道宽度先验，以及道路中心线编码。我们在两个地理位置和复杂度各异的交叉口场景上评估了该方法。我们的方法在车道和交通元素检测及其关联性方面都表现出改进。我们使用四种拓扑感知指标全面评估了模型性能。这些结果展示了我们方法泛化和适应各种拓扑结构和条件的能力。 

---
# SpectralAR: Spectral Autoregressive Visual Generation 

**Title (ZH)**: SpectralAR：谱自回归视觉生成 

**Authors**: Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Yueqi Duan, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10962)  

**Abstract**: Autoregressive visual generation has garnered increasing attention due to its scalability and compatibility with other modalities compared with diffusion models. Most existing methods construct visual sequences as spatial patches for autoregressive generation. However, image patches are inherently parallel, contradicting the causal nature of autoregressive modeling. To address this, we propose a Spectral AutoRegressive (SpectralAR) visual generation framework, which realizes causality for visual sequences from the spectral perspective. Specifically, we first transform an image into ordered spectral tokens with Nested Spectral Tokenization, representing lower to higher frequency components. We then perform autoregressive generation in a coarse-to-fine manner with the sequences of spectral tokens. By considering different levels of detail in images, our SpectralAR achieves both sequence causality and token efficiency without bells and whistles. We conduct extensive experiments on ImageNet-1K for image reconstruction and autoregressive generation, and SpectralAR achieves 3.02 gFID with only 64 tokens and 310M parameters. Project page: this https URL. 

**Abstract (ZH)**: 自回归视觉生成由于其可扩展性和与其他模态的兼容性，相比扩散模型越来越受到关注。大多数现有方法将视觉序列构建为空间 patches 进行自回归生成。然而，图像 patches 内在地是并行的，这与自回归建模的因果性质相矛盾。为解决这一问题，我们提出了一种从频谱视角实现视觉序列因果性的 Spectral AutoRegressive (SpectralAR) 视觉生成框架。具体而言，我们首先使用嵌套频谱分词将图像转换为有序的频谱 token，从低频到高频表示图像的不同频率分量。然后，我们以粗到细的方式对频谱 token 序列进行自回归生成。通过考虑图像的不同细节层级，我们的 SpectralAR 在实现序列因果性和 token 效率的同时，无需额外复杂性。我们在 ImageNet-1K 上进行了广泛的图像重建和自回归生成实验，SpectralAR 仅使用 64 个 token 和 310M 参数实现了 3.02 gFID。项目页面：this https URL。 

---
# VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos 

**Title (ZH)**: VRBench: 长叙事视频多步推理基准 

**Authors**: Jiashuo Yu, Yue Wu, Meng Chu, Zhifei Ren, Zizheng Huang, Pei Chu, Ruijie Zhang, Yinan He, Qirui Li, Songze Li, Zhenxiang Li, Zhongying Tu, Conghui He, Yu Qiao, Yali Wang, Yi Wang, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10857)  

**Abstract**: We present VRBench, the first long narrative video benchmark crafted for evaluating large models' multi-step reasoning capabilities, addressing limitations in existing evaluations that overlook temporal reasoning and procedural validity. It comprises 1,010 long videos (with an average duration of 1.6 hours), along with 9,468 human-labeled multi-step question-answering pairs and 30,292 reasoning steps with timestamps. These videos are curated via a multi-stage filtering process including expert inter-rater reviewing to prioritize plot coherence. We develop a human-AI collaborative framework that generates coherent reasoning chains, each requiring multiple temporally grounded steps, spanning seven types (e.g., event attribution, implicit inference). VRBench designs a multi-phase evaluation pipeline that assesses models at both the outcome and process levels. Apart from the MCQs for the final results, we propose a progress-level LLM-guided scoring metric to evaluate the quality of the reasoning chain from multiple dimensions comprehensively. Through extensive evaluations of 12 LLMs and 16 VLMs on VRBench, we undertake a thorough analysis and provide valuable insights that advance the field of multi-step reasoning. 

**Abstract (ZH)**: VRBench: 针对大规模语言模型多步推理能力评估的第一个长叙事视频基准 

---
# Post-Training Quantization for Video Matting 

**Title (ZH)**: 视频抠像的后训练量化 

**Authors**: Tianrui Zhu, Houyuan Chen, Ruihao Gong, Michele Magno, Haotong Qin, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10840)  

**Abstract**: Video matting is crucial for applications such as film production and virtual reality, yet deploying its computationally intensive models on resource-constrained devices presents challenges. Quantization is a key technique for model compression and acceleration. As an efficient approach, Post-Training Quantization (PTQ) is still in its nascent stages for video matting, facing significant hurdles in maintaining accuracy and temporal coherence. To address these challenges, this paper proposes a novel and general PTQ framework specifically designed for video matting models, marking, to the best of our knowledge, the first systematic attempt in this domain. Our contributions include: (1) A two-stage PTQ strategy that combines block-reconstruction-based optimization for fast, stable initial quantization and local dependency capture, followed by a global calibration of quantization parameters to minimize accuracy loss. (2) A Statistically-Driven Global Affine Calibration (GAC) method that enables the network to compensate for cumulative statistical distortions arising from factors such as neglected BN layer effects, even reducing the error of existing PTQ methods on video matting tasks up to 20%. (3) An Optical Flow Assistance (OFA) component that leverages temporal and semantic priors from frames to guide the PTQ process, enhancing the model's ability to distinguish moving foregrounds in complex scenes and ultimately achieving near full-precision performance even under ultra-low-bit quantization. Comprehensive quantitative and visual results show that our PTQ4VM achieves the state-of-the-art accuracy performance across different bit-widths compared to the existing quantization methods. We highlight that the 4-bit PTQ4VM even achieves performance close to the full-precision counterpart while enjoying 8x FLOP savings. 

**Abstract (ZH)**: 视频抠图在电影制作和虚拟现实等应用中至关重要，但将其计算密集型模型部署在资源受限设备上面临着挑战。量化是模型压缩和加速的关键技术。作为一种有效的手段，后训练量化（PTQ）仍处于初步阶段，特别是在视频抠图领域，保持准确性和时间一致性方面面临着重大挑战。为了解决这些挑战，本文提出了一种专为视频抠图模型设计的新型通用PTQ框架，据我们所知，这是在此领域中的首次系统性尝试。我们的贡献包括：（1）一种两阶段PTQ策略，结合块重建优化进行快速、稳定的初始量化和局部依赖捕捉，随后进行全局量化参数校准以最小化准确率损失。（2）一种统计驱动的全局仿射校准（GAC）方法，使网络能够补偿由未忽略BN层效应等因素引起的累积统计失真，甚至在视频抠图任务中将现有PTQ方法的误差降低多达20%。（3）一种光流辅助（OFA）组件，利用帧中的时间和语义先验引导PTQ过程，增强模型在复杂场景中区分移动前景的能力，最终即便在极低位宽量化下也能实现接近全精度的性能。综合定量和视觉结果表明，我们的PTQ4VM在不同位宽下相较于现有量化方法实现了最先进的准确率性能。我们强调，4-bit的PTQ4VM即使在享受8倍FLOP节省的情况下，性能也接近全精度的版本。 

---
# Generalist Models in Medical Image Segmentation: A Survey and Performance Comparison with Task-Specific Approaches 

**Title (ZH)**: 医学图像分割中的通用模型：一项综述及与任务专用方法的性能比较 

**Authors**: Andrea Moglia, Matteo Leccardi, Matteo Cavicchioli, Alice Maccarini, Marco Marcon, Luca Mainardi, Pietro Cerveri  

**Link**: [PDF](https://arxiv.org/pdf/2506.10825)  

**Abstract**: Following the successful paradigm shift of large language models, leveraging pre-training on a massive corpus of data and fine-tuning on different downstream tasks, generalist models have made their foray into computer vision. The introduction of Segment Anything Model (SAM) set a milestone on segmentation of natural images, inspiring the design of a multitude of architectures for medical image segmentation. In this survey we offer a comprehensive and in-depth investigation on generalist models for medical image segmentation. We start with an introduction on the fundamentals concepts underpinning their development. Then, we provide a taxonomy on the different declinations of SAM in terms of zero-shot, few-shot, fine-tuning, adapters, on the recent SAM 2, on other innovative models trained on images alone, and others trained on both text and images. We thoroughly analyze their performances at the level of both primary research and best-in-literature, followed by a rigorous comparison with the state-of-the-art task-specific models. We emphasize the need to address challenges in terms of compliance with regulatory frameworks, privacy and security laws, budget, and trustworthy artificial intelligence (AI). Finally, we share our perspective on future directions concerning synthetic data, early fusion, lessons learnt from generalist models in natural language processing, agentic AI and physical AI, and clinical translation. 

**Abstract (ZH)**: 跟随大型语言模型的成功范式转变，借助大规模数据的预训练和不同下游任务的微调，通用模型已经进入计算机视觉领域。Segment Anything Model (SAM) 的引入在自然图像分割上树立了一个里程碑，激发了多种针对医学图像分割的架构设计。在本文综述中，我们提供了一种全面而深入的调查，探讨通用模型在医学图像分割中的应用。我们首先介绍支撑其发展的基本概念。然后，我们从零样本、少样本、微调、适配器、最新SAM 2、仅图像训练的其他创新模型以及同时在文本和图像上训练的其他模型等方面提供了分类。我们从原始研究和文献最佳实践的角度详细分析了它们的性能，并进行了与最先进的特定任务模型的严格比较。我们强调了合规性、隐私和安全法规、预算以及可信赖人工智能（AI）方面所面临挑战的重要性。最后，我们分享了关于合成数据、早期融合、自然语言处理中通用模型的经验教训、自主AI和物理AI以及临床转化的未来方向观点。 

---
# BNMusic: Blending Environmental Noises into Personalized Music 

**Title (ZH)**: BNMusic: 将环境声音融合到个性化音乐中 

**Authors**: Chi Zuo, Martin B. Møller, Pablo Martínez-Nuevo, Huayang Huang, Yu Wu, Ye Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10754)  

**Abstract**: While being disturbed by environmental noises, the acoustic masking technique is a conventional way to reduce the annoyance in audio engineering that seeks to cover up the noises with other dominant yet less intrusive sounds. However, misalignment between the dominant sound and the noise-such as mismatched downbeats-often requires an excessive volume increase to achieve effective masking. Motivated by recent advances in cross-modal generation, in this work, we introduce an alternative method to acoustic masking, aiming to reduce the noticeability of environmental noises by blending them into personalized music generated based on user-provided text prompts. Following the paradigm of music generation using mel-spectrogram representations, we propose a Blending Noises into Personalized Music (BNMusic) framework with two key stages. The first stage synthesizes a complete piece of music in a mel-spectrogram representation that encapsulates the musical essence of the noise. In the second stage, we adaptively amplify the generated music segment to further reduce noise perception and enhance the blending effectiveness, while preserving auditory quality. Our experiments with comprehensive evaluations on MusicBench, EPIC-SOUNDS, and ESC-50 demonstrate the effectiveness of our framework, highlighting the ability to blend environmental noise with rhythmically aligned, adaptively amplified, and enjoyable music segments, minimizing the noticeability of the noise, thereby improving overall acoustic experiences. 

**Abstract (ZH)**: 利用个性化音乐融合环境噪音的掩蔽方法 

---
# PiPViT: Patch-based Visual Interpretable Prototypes for Retinal Image Analysis 

**Title (ZH)**: PiPViT：基于 patches 的视觉可解释原型.Retinal 图像分析 

**Authors**: Marzieh Oghbaie, Teresa Araújoa, Hrvoje Bogunović  

**Link**: [PDF](https://arxiv.org/pdf/2506.10669)  

**Abstract**: Background and Objective: Prototype-based methods improve interpretability by learning fine-grained part-prototypes; however, their visualization in the input pixel space is not always consistent with human-understandable biomarkers. In addition, well-known prototype-based approaches typically learn extremely granular prototypes that are less interpretable in medical imaging, where both the presence and extent of biomarkers and lesions are critical.
Methods: To address these challenges, we propose PiPViT (Patch-based Visual Interpretable Prototypes), an inherently interpretable prototypical model for image recognition. Leveraging a vision transformer (ViT), PiPViT captures long-range dependencies among patches to learn robust, human-interpretable prototypes that approximate lesion extent only using image-level labels. Additionally, PiPViT benefits from contrastive learning and multi-resolution input processing, which enables effective localization of biomarkers across scales.
Results: We evaluated PiPViT on retinal OCT image classification across four datasets, where it achieved competitive quantitative performance compared to state-of-the-art methods while delivering more meaningful explanations. Moreover, quantitative evaluation on a hold-out test set confirms that the learned prototypes are semantically and clinically relevant. We believe PiPViT can transparently explain its decisions and assist clinicians in understanding diagnostic outcomes. Github page: this https URL 

**Abstract (ZH)**: 背景与目标：基于原型的方法通过学习细粒度的部分原型提高了可解释性；然而，它们在输入像素空间中的可视化有时与人类可理解的生物标志物不一致。此外，著名的基于原型的方法通常学习极为细粒度的原型，在医学成像中可解释性较差，而医学成像中生物标志物和病变的存在及其范围至关重要。
方法：为了解决这些挑战，我们提出了一种名为PiPViT（patches-based visual interpretable prototypes）的模型，这是一种固有的可解释原型模型，用于图像识别。通过利用视觉变换器（ViT），PiPViT捕捉.patch之间的长程依赖关系，利用图像级标签学习鲁棒且人类可解释的原型，仅通过图像量级标签来近似病变范围。此外，PiPViT还受益于对比学习和多分辨率输入处理，这使其能够在不同尺度上有效地定位生物标志物。
结果：我们在四个数据集上对PiPViT进行了视网膜OCT图像分类的评估，其相较于最先进的方法在定量性能上具有竞争力，同时提供了更具有意义的解释。此外，对保留测试集的定量评估表明，学习到的原型具有语义和临床相关性。我们相信PiPViT能够透明地解释其决策，并辅助临床医生理解诊断结果。GitHub页面：this https URL 

---
# Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models 

**Title (ZH)**: 对称流匹配：基于评分生成模型的统一图像生成、分割和分类 

**Authors**: Francisco Caetano, Christiaan Viviers, Peter H.N. De With, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10634)  

**Abstract**: Flow Matching has emerged as a powerful framework for learning continuous transformations between distributions, enabling high-fidelity generative modeling. This work introduces Symmetrical Flow Matching (SymmFlow), a new formulation that unifies semantic segmentation, classification, and image generation within a single model. Using a symmetric learning objective, SymmFlow models forward and reverse transformations jointly, ensuring bi-directional consistency, while preserving sufficient entropy for generative diversity. A new training objective is introduced to explicitly retain semantic information across flows, featuring efficient sampling while preserving semantic structure, allowing for one-step segmentation and classification without iterative refinement. Unlike previous approaches that impose strict one-to-one mapping between masks and images, SymmFlow generalizes to flexible conditioning, supporting both pixel-level and image-level class labels. Experimental results on various benchmarks demonstrate that SymmFlow achieves state-of-the-art performance on semantic image synthesis, obtaining FID scores of 11.9 on CelebAMask-HQ and 7.0 on COCO-Stuff with only 25 inference steps. Additionally, it delivers competitive results on semantic segmentation and shows promising capabilities in classification tasks. The code will be publicly available. 

**Abstract (ZH)**: Symmetrical Flow Matching: Unifying Semantic Segmentation, Classification, and Image Generation with High-Fidelity Generative Modeling 

---
# DreamActor-H1: High-Fidelity Human-Product Demonstration Video Generation via Motion-designed Diffusion Transformers 

**Title (ZH)**: DreamActor-H1: 基于运动设计的扩散变换器高保真人-产品演示视频生成 

**Authors**: Lizhen Wang, Zhurong Xia, Tianshu Hu, Pengrui Wang, Pengfei Wang, Zerong Zheng, Ming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.10568)  

**Abstract**: In e-commerce and digital marketing, generating high-fidelity human-product demonstration videos is important for effective product presentation. However, most existing frameworks either fail to preserve the identities of both humans and products or lack an understanding of human-product spatial relationships, leading to unrealistic representations and unnatural interactions. To address these challenges, we propose a Diffusion Transformer (DiT)-based framework. Our method simultaneously preserves human identities and product-specific details, such as logos and textures, by injecting paired human-product reference information and utilizing an additional masked cross-attention mechanism. We employ a 3D body mesh template and product bounding boxes to provide precise motion guidance, enabling intuitive alignment of hand gestures with product placements. Additionally, structured text encoding is used to incorporate category-level semantics, enhancing 3D consistency during small rotational changes across frames. Trained on a hybrid dataset with extensive data augmentation strategies, our approach outperforms state-of-the-art techniques in maintaining the identity integrity of both humans and products and generating realistic demonstration motions. Project page: this https URL. 

**Abstract (ZH)**: 电子商务和数字营销中，生成高质量的人－产品示范视频对于有效的商品展示至关重要。然而，现有大多数框架要么无法同时保留人类和产品的身份，要么缺乏对人类－产品空间关系的理解，导致不真实的表示和不自然的交互。为了解决这些挑战，我们提出了一种基于扩散转换器（DiT）的框架。该方法通过注入成对的人－产品参考信息并利用附加的掩码交叉注意机制，同时保留人类身份和产品特定细节，如标志和纹理。我们采用3D身体网格模板和产品边界框提供精确的运动指导，使手部手势与产品放置直观对齐。此外，使用结构化文本编码增强类别级别的语义，从而在帧间小旋转变化中提高3D一致性。通过广泛数据增强策略训练的混合数据集，我们的方法在保持人类和产品身份完整性以及生成逼真示范动作方面优于现有技术。项目页面：this https URL。 

---
# Semantic Localization Guiding Segment Anything Model For Reference Remote Sensing Image Segmentation 

**Title (ZH)**: 基于语义定位的 Segment Anything 模型在参考遥感图像分割中的应用 

**Authors**: Shuyang Li, Shuang Wang, Zhuangzhuang Sun, Jing Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.10503)  

**Abstract**: The Reference Remote Sensing Image Segmentation (RRSIS) task generates segmentation masks for specified objects in images based on textual descriptions, which has attracted widespread attention and research interest. Current RRSIS methods rely on multi-modal fusion backbones and semantic segmentation heads but face challenges like dense annotation requirements and complex scene interpretation. To address these issues, we propose a framework named \textit{prompt-generated semantic localization guiding Segment Anything Model}(PSLG-SAM), which decomposes the RRSIS task into two stages: coarse localization and fine segmentation. In coarse localization stage, a visual grounding network roughly locates the text-described object. In fine segmentation stage, the coordinates from the first stage guide the Segment Anything Model (SAM), enhanced by a clustering-based foreground point generator and a mask boundary iterative optimization strategy for precise segmentation. Notably, the second stage can be train-free, significantly reducing the annotation data burden for the RRSIS task. Additionally, decomposing the RRSIS task into two stages allows for focusing on specific region segmentation, avoiding interference from complex this http URL further contribute a high-quality, multi-category manually annotated dataset. Experimental validation on two datasets (RRSIS-D and RRSIS-M) demonstrates that PSLG-SAM achieves significant performance improvements and surpasses existing state-of-the-art this http URL code will be made publicly available. 

**Abstract (ZH)**: 基于提示生成的语义定位引导分割一切模型（PSLG-SAM）的任务框架 

---
# Semi-Tensor-Product Based Convolutional Neural Networks 

**Title (ZH)**: 基于半张量积的卷积神经网络 

**Authors**: Daizhan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10407)  

**Abstract**: The semi-tensor product (STP) of vectors is a generalization of conventional inner product of vectors, which allows the factor vectors to of different dimensions. This paper proposes a domain-based convolutional product (CP). Combining domain-based CP with STP of vectors, a new CP is proposed. Since there is no zero or any other padding, it can avoid the junk information caused by padding. Using it, the STP-based convolutional neural network (CNN) is developed. Its application to image and third order signal identifications is considered. 

**Abstract (ZH)**: 基于半张量积的域域卷积产品及其在网络中的应用 

---
# Pisces: An Auto-regressive Foundation Model for Image Understanding and Generation 

**Title (ZH)**: Pisces：一种用于图像理解与生成的自回归基础模型 

**Authors**: Zhiyang Xu, Jiuhai Chen, Zhaojiang Lin, Xichen Pan, Lifu Huang, Tianyi Zhou, Madian Khabsa, Qifan Wang, Di Jin, Michihiro Yasunaga, Lili Yu, Xi Victoria Lin, Shaoliang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10395)  

**Abstract**: Recent advances in large language models (LLMs) have enabled multimodal foundation models to tackle both image understanding and generation within a unified framework. Despite these gains, unified models often underperform compared to specialized models in either task. A key challenge in developing unified models lies in the inherent differences between the visual features needed for image understanding versus generation, as well as the distinct training processes required for each modality. In this work, we introduce Pisces, an auto-regressive multimodal foundation model that addresses this challenge through a novel decoupled visual encoding architecture and tailored training techniques optimized for multimodal generation. Combined with meticulous data curation, pretraining, and finetuning, Pisces achieves competitive performance in both image understanding and image generation. We evaluate Pisces on over 20 public benchmarks for image understanding, where it demonstrates strong performance across a wide range of tasks. Additionally, on GenEval, a widely adopted benchmark for image generation, Pisces exhibits robust generative capabilities. Our extensive analysis reveals the synergistic relationship between image understanding and generation, and the benefits of using separate visual encoders, advancing the field of unified multimodal models. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）的进展使多模态基础模型能够在统一框架内解决图像理解与生成问题。尽管取得了这些进展，统一模型在各项任务中的表现往往不如专门针对某一任务训练的模型。开发统一模型的关键挑战在于图像理解和生成所需视觉特征的内在差异，以及每种模态所需的独特训练过程。在此项工作中，我们引入了Pisces，一种通过新颖的解耦视觉编码架构和针对多模态生成优化的定制训练技术的自回归多模态基础模型。结合细致的数据整理、预训练和微调，Pisces在图像理解与图像生成任务中均表现出竞争性的性能。我们在超过20个公开图像理解基准上评估了Pisces，结果表明其在多种任务上表现强劲。此外，在广泛采用的图像生成基准GenEval上，Pisces展示了稳健的生成能力。我们深入的分析揭示了图像理解和生成之间的协同关系，并证实了使用独立视觉编码器的优势，推动了统一多模态模型的发展。 

---
# UrbanSense:AFramework for Quantitative Analysis of Urban Streetscapes leveraging Vision Large Language Models 

**Title (ZH)**: UrbanSense：一种基于视觉大规模语言模型的都市街道景观定量分析框架 

**Authors**: Jun Yin, Jing Zhong, Peilin Li, Pengyu Zeng, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.10342)  

**Abstract**: Urban cultures and architectural styles vary significantly across cities due to geographical, chronological, historical, and socio-political factors. Understanding these differences is essential for anticipating how cities may evolve in the future. As representative cases of historical continuity and modern innovation in China, Beijing and Shenzhen offer valuable perspectives for exploring the transformation of urban streetscapes. However, conventional approaches to urban cultural studies often rely on expert interpretation and historical documentation, which are difficult to standardize across different contexts. To address this, we propose a multimodal research framework based on vision-language models, enabling automated and scalable analysis of urban streetscape style differences. This approach enhances the objectivity and data-driven nature of urban form research. The contributions of this study are as follows: First, we construct UrbanDiffBench, a curated dataset of urban streetscapes containing architectural images from different periods and regions. Second, we develop UrbanSense, the first vision-language-model-based framework for urban streetscape analysis, enabling the quantitative generation and comparison of urban style representations. Third, experimental results show that Over 80% of generated descriptions pass the t-test (p less than 0.05). High Phi scores (0.912 for cities, 0.833 for periods) from subjective evaluations confirm the method's ability to capture subtle stylistic differences. These results highlight the method's potential to quantify and interpret urban style evolution, offering a scientifically grounded lens for future design. 

**Abstract (ZH)**: 城市的文化与建筑风格因地理、历史、社会政治等因素在不同城市间存在显著差异。理解这些差异对于预测城市未来的演化至关重要。作为中国历史连续性和现代创新的代表案例，北京和深圳为探索城市街道景观的演变提供了宝贵视角。然而，传统城市文化研究方法往往依赖于专家解释和历史文献，难以在不同背景下标准化。为此，我们提出基于视觉-语言模型的多模态研究框架，实现对城市街道景观风格差异的自动化和可扩展分析。该方法增强了城市形态研究的客观性和数据驱动性质。本研究的贡献包括：首先，构建了包含不同历史时期和地区建筑图片的UrbanDiffBench数据集；其次，开发了基于视觉-语言模型的第一种城市街道景观分析框架UrbanSense，能够定量生成和比较城市风格表示；最后，实验结果表明，超过80%生成的描述通过了t检验（p<0.05），主观评价的高Phi分数（城市为0.912，时期为0.833）证实了该方法捕获微妙风格差异的能力。这些结果突显了该方法量化和解读城市风格演化的潜力，为未来设计提供了一种科学视角。 

---
# Using Vision Language Models to Detect Students' Academic Emotion through Facial Expressions 

**Title (ZH)**: 使用视觉语言模型检测学生通过面部表情表达的学术情绪 

**Authors**: Deliang Wang, Chao Yang, Gaowei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10334)  

**Abstract**: Students' academic emotions significantly influence their social behavior and learning performance. Traditional approaches to automatically and accurately analyze these emotions have predominantly relied on supervised machine learning algorithms. However, these models often struggle to generalize across different contexts, necessitating repeated cycles of data collection, annotation, and training. The emergence of Vision-Language Models (VLMs) offers a promising alternative, enabling generalization across visual recognition tasks through zero-shot prompting without requiring fine-tuning. This study investigates the potential of VLMs to analyze students' academic emotions via facial expressions in an online learning environment. We employed two VLMs, Llama-3.2-11B-Vision-Instruct and Qwen2.5-VL-7B-Instruct, to analyze 5,000 images depicting confused, distracted, happy, neutral, and tired expressions using zero-shot prompting. Preliminary results indicate that both models demonstrate moderate performance in academic facial expression recognition, with Qwen2.5-VL-7B-Instruct outperforming Llama-3.2-11B-Vision-Instruct. Notably, both models excel in identifying students' happy emotions but fail to detect distracted behavior. Additionally, Qwen2.5-VL-7B-Instruct exhibits relatively high performance in recognizing students' confused expressions, highlighting its potential for practical applications in identifying content that causes student confusion. 

**Abstract (ZH)**: 学生的情绪对其社交行为和学习表现有显著影响。传统的自动准确分析这些情绪的方法主要依赖于监督机器学习算法。然而，这些模型往往难以在不同情境下泛化，需要反复的数据收集、注释和训练。视觉-语言模型（VLMs）的出现为其提供了有前景的替代方案，通过零样本提示实现跨视觉识别任务的泛化而无需微调。本研究探讨了VLMs在在线学习环境中通过面部表情分析学生学术情绪的潜力。我们使用了两个VLMs，Llama-3.2-11B-Vision-Instruct和Qwen2.5-VL-7B-Instruct，对5,000张表情图像（包括困惑、分心、快乐、中性和平静）进行零样本提示分析。初步结果显示，两种模型在学术面部表情识别方面表现出中等性能，Qwen2.5-VL-7B-Instruct优于Llama-3.2-11B-Vision-Instruct。值得注意的是，两种模型都擅长识别学生的情绪喜悦，但在检测分心行为方面表现不佳。此外，Qwen2.5-VL-7B-Instruct在识别学生困惑的表情方面表现出相对较高的性能，突显了其在识别引起学生困惑的内容方面的潜在实际应用价值。 

---
# ScoreMix: Improving Face Recognition via Score Composition in Diffusion Generators 

**Title (ZH)**: ScoreMix: 通过扩散生成器中的评分合成提高面部识别性能 

**Authors**: Parsa Rahimi, Sebastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2506.10226)  

**Abstract**: In this paper, we propose ScoreMix, a novel yet simple data augmentation strategy leveraging the score compositional properties of diffusion models to enhance discriminator performance, particularly under scenarios with limited labeled data. By convexly mixing the scores from different class-conditioned trajectories during diffusion sampling, we generate challenging synthetic samples that significantly improve discriminative capabilities in all studied benchmarks. We systematically investigate class-selection strategies for mixing and discover that greater performance gains arise when combining classes distant in the discriminator's embedding space, rather than close in the generator's condition space. Moreover, we empirically show that, under standard metrics, the correlation between the generator's learned condition space and the discriminator's embedding space is minimal. Our approach achieves notable performance improvements without extensive parameter searches, demonstrating practical advantages for training discriminative models while effectively mitigating problems regarding collections of large datasets. Paper website: this https URL 

**Abstract (ZH)**: 在本文中，我们提出了一种新颖且简单的数据增强策略ScoreMix，该策略利用扩散模型的分数组成特性来增强判别器性能，尤其是在标记数据有限的情况下。通过在扩散采样过程中凸性混合不同类条件轨迹的分数，我们生成了具有挑战性的合成样本，这些样本在所有研究的基准测试中显著提高了辨别能力。我们系统地研究了混合的类选择策略，并发现当组合在判别器嵌入空间中距离较远的类时，可以获得更大的性能提升，而非在生成器条件空间中接近的类。此外，我们实证表明，在标准指标下，生成器学习的条件空间与判别器的嵌入空间之间的相关性最小。我们的方法在无需进行广泛的参数搜索的情况下取得了显著性能提升，显示了在训练辨别模型方面的实用优势，并有效缓解了大规模数据集收集的问题。论文网站：this https URL 

---
# Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment 

**Title (ZH)**: 可扩展的非同构3D分子生成通过旋转对齐 

**Authors**: Yuhui Ding, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.10186)  

**Abstract**: Equivariant diffusion models have achieved impressive performance in 3D molecule generation. These models incorporate Euclidean symmetries of 3D molecules by utilizing an SE(3)-equivariant denoising network. However, specialized equivariant architectures limit the scalability and efficiency of diffusion models. In this paper, we propose an approach that relaxes such equivariance constraints. Specifically, our approach learns a sample-dependent SO(3) transformation for each molecule to construct an aligned latent space. A non-equivariant diffusion model is then trained over the aligned representations. Experimental results demonstrate that our approach performs significantly better than previously reported non-equivariant models. It yields sample quality comparable to state-of-the-art equivariant diffusion models and offers improved training and sampling efficiency. Our code is available at this https URL 

**Abstract (ZH)**: 三维分子生成中不变性扩散模型已在各个方面取得了显著成果。这些模型通过利用一个SE(3)-不变性去噪网络，将3D分子的欧几里得对称性纳入其中。然而，专门设计的不变性架构限制了扩散模型的可扩展性和效率。在本文中，我们提出了一种放宽此类不变性约束的方法。具体来说，我们的方法为每个分子学习一个样本依赖的SO(3)变换，以构建对齐的潜在空间。然后在一个对齐的表征上训练一个非不变性扩散模型。实验结果显示，我们的方法在样本质量上明显优于之前报告的非不变性模型，并提供了更好的训练和采样效率。我们的代码可在以下链接获得：this https URL 

---
# Detecção da Psoríase Utilizando Visão Computacional: Uma Abordagem Comparativa Entre CNNs e Vision Transformers 

**Title (ZH)**: 利用计算机视觉检测银屑病：CNNs与Vision Transformers的比较研究 

**Authors**: Natanael Lucena, Fábio S. da Silva, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2506.10119)  

**Abstract**: This paper presents a comparison of the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) in the task of multi-classifying images containing lesions of psoriasis and diseases similar to it. Models pre-trained on ImageNet were adapted to a specific data set. Both achieved high predictive metrics, but the ViTs stood out for their superior performance with smaller models. Dual Attention Vision Transformer-Base (DaViT-B) obtained the best results, with an f1-score of 96.4%, and is recommended as the most efficient architecture for automated psoriasis detection. This article reinforces the potential of ViTs for medical image classification tasks. 

**Abstract (ZH)**: 这篇论文比较了卷积神经网络（CNNs）和视觉变换器（ViTs）在鉴别银屑病及其类似疾病皮肤病图像多分类任务中的性能。预训练的ImageNet模型适应了特定数据集。两者都获得了较高的预测指标，但ViTs凭借更小模型的优越性能脱颖而出。Dual Attention Vision Transformer-Base (DaViT-B) 达到了96.4%的F1分数，并被推荐为自动银屑病检测的最高效架构。本文强化了ViTs在医学图像分类任务中的潜力。 

---
# Ambient Diffusion Omni: Training Good Models with Bad Data 

**Title (ZH)**: Ambient Diffusion Omni: 使用不良数据训练优质模型 

**Authors**: Giannis Daras, Adrian Rodriguez-Munoz, Adam Klivans, Antonio Torralba, Constantinos Daskalakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.10038)  

**Abstract**: We show how to use low-quality, synthetic, and out-of-distribution images to improve the quality of a diffusion model. Typically, diffusion models are trained on curated datasets that emerge from highly filtered data pools from the Web and other sources. We show that there is immense value in the lower-quality images that are often discarded. We present Ambient Diffusion Omni, a simple, principled framework to train diffusion models that can extract signal from all available images during training. Our framework exploits two properties of natural images -- spectral power law decay and locality. We first validate our framework by successfully training diffusion models with images synthetically corrupted by Gaussian blur, JPEG compression, and motion blur. We then use our framework to achieve state-of-the-art ImageNet FID, and we show significant improvements in both image quality and diversity for text-to-image generative modeling. The core insight is that noise dampens the initial skew between the desired high-quality distribution and the mixed distribution we actually observe. We provide rigorous theoretical justification for our approach by analyzing the trade-off between learning from biased data versus limited unbiased data across diffusion times. 

**Abstract (ZH)**: 我们展示了如何利用低质量、合成和离分布域的图像来提高扩散模型的质量。我们证明了被常丢弃的低质量图像中蕴含的巨大价值。我们提出了Ambient Diffusion Omni这一简单且原理明确的框架，该框架能够利用所有可用图像在训练过程中提取信号。我们的框架利用了自然图像的两种特性——谱功率律衰减和局部性。我们首先通过成功训练被高斯模糊、JPEG压缩和运动模糊等合成噪声破坏的图像来验证该框架的有效性。然后，我们使用该框架实现了Imagenet FID的最新成果，并展示了在文本生成图像时图像质量和多样性的显著改进。核心洞见在于噪声减弱了所需的高质量分布与我们实际观察到的混合分布之间的初始偏差。我们通过对扩散时间段内从有偏数据学习与有限无偏数据学习之间的权衡进行严谨的理论分析，为我们的方法提供了理论依据。 

---
# EQ-TAA: Equivariant Traffic Accident Anticipation via Diffusion-Based Accident Video Synthesis 

**Title (ZH)**: EQ-TAA: 基于扩散模型的事故视频合成的同态交通事故预判 

**Authors**: Jianwu Fang, Lei-Lei Li, Zhedong Zheng, Hongkai Yu, Jianru Xue, Zhengguo Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.10002)  

**Abstract**: Traffic Accident Anticipation (TAA) in traffic scenes is a challenging problem for achieving zero fatalities in the future. Current approaches typically treat TAA as a supervised learning task needing the laborious annotation of accident occurrence duration. However, the inherent long-tailed, uncertain, and fast-evolving nature of traffic scenes has the problem that real causal parts of accidents are difficult to identify and are easily dominated by data bias, resulting in a background confounding issue. Thus, we propose an Attentive Video Diffusion (AVD) model that synthesizes additional accident video clips by generating the causal part in dashcam videos, i.e., from normal clips to accident clips. AVD aims to generate causal video frames based on accident or accident-free text prompts while preserving the style and content of frames for TAA after video generation. This approach can be trained using datasets collected from various driving scenes without any extra annotations. Additionally, AVD facilitates an Equivariant TAA (EQ-TAA) with an equivariant triple loss for an anchor accident-free video clip, along with the generated pair of contrastive pseudo-normal and pseudo-accident clips. Extensive experiments have been conducted to evaluate the performance of AVD and EQ-TAA, and competitive performance compared to state-of-the-art methods has been obtained. 

**Abstract (ZH)**: 交通事故预见（TAA）在交通场景中的研究是一个挑战性问题，旨在实现未来的零伤亡目标。当前方法通常将TAA视为需要 laborious 事故发生时间标注的监督学习任务。然而，交通场景固有的长尾分布、不确定性及快速演变特性导致事故的真实因果部分难以识别，且容易受到数据偏差的影响，产生背景混杂问题。因此，我们提出了一种注意视频扩散（AVD）模型，通过在行车记录仪视频中生成因果部分，即从正常片段到事故片段，来合成额外的事故视频片段。AVD 的目标是在事故或无事故文本提示的基础上生成因果视频帧，同时保留视频生成后的帧的风格和内容，以实现TAA。此方法可以通过收集各种驾驶场景下的数据集进行训练，无需额外标注。此外，AVD 还通过锚定无事故视频片段及生成对比伪正常和伪事故片段的不变三重损失，促进了一种不变性TAA（EQ-TAA）。进行了广泛的实验以评估AVD和EQ-TAA的性能，结果表明其在与现有最佳方法的性能上具有竞争力。 

---
