# Hallucinating 360°: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting 

**Title (ZH)**: 全景街景生成：基于局部场景扩散和概率性提示的虚拟视角生成 

**Authors**: Fei Teng, Kai Luo, Sheng Wu, Siyu Li, Pujun Guo, Jiale Wei, Kunyu Peng, Jiaming Zhang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06971)  

**Abstract**: Panoramic perception holds significant potential for autonomous driving, enabling vehicles to acquire a comprehensive 360° surround view in a single shot. However, autonomous driving is a data-driven task. Complete panoramic data acquisition requires complex sampling systems and annotation pipelines, which are time-consuming and labor-intensive. Although existing street view generation models have demonstrated strong data regeneration capabilities, they can only learn from the fixed data distribution of existing datasets and cannot achieve high-quality, controllable panoramic generation. In this paper, we propose the first panoramic generation method Percep360 for autonomous driving. Percep360 enables coherent generation of panoramic data with control signals based on the stitched panoramic data. Percep360 focuses on two key aspects: coherence and controllability. Specifically, to overcome the inherent information loss caused by the pinhole sampling process, we propose the Local Scenes Diffusion Method (LSDM). LSDM reformulates the panorama generation as a spatially continuous diffusion process, bridging the gaps between different data distributions. Additionally, to achieve the controllable generation of panoramic images, we propose a Probabilistic Prompting Method (PPM). PPM dynamically selects the most relevant control cues, enabling controllable panoramic image generation. We evaluate the effectiveness of the generated images from three perspectives: image quality assessment (i.e., no-reference and with reference), controllability, and their utility in real-world Bird's Eye View (BEV) segmentation. Notably, the generated data consistently outperforms the original stitched images in no-reference quality metrics and enhances downstream perception models. The source code will be publicly available at this https URL. 

**Abstract (ZH)**: 全景感知在自动驾驶中具有重要潜力，能够使车辆在单次拍摄中获得全面的360° Surround View。然而，自动驾驶是一项数据驱动的任务。完整的全景数据获取需要复杂的采样系统和注释管道，这既耗时又劳动密集。尽管现有的街景生成模型展示了强大的数据再生能力，但它们只能从现有数据集的固定数据分布中学习，无法实现高质量、可控的全景生成。在本文中，我们提出了第一个适用于自动驾驶的全景生成方法Percep360。Percep360能够基于拼接的全景数据进行具有控制信号的连贯生成。Percep360重点关注两个关键方面：连贯性和可控性。具体地，为了克服针孔采样过程中的固有信息损失，我们提出了局部场景扩散方法（LSDM）。LSDM将全景生成重新定义为一个空间连续的扩散过程，弥合了不同数据分布之间的差距。此外，为了实现全景图的可控生成，我们提出了概率提示方法（PPM）。PPM动态选择最相关的控制线索，实现可控的全景图像生成。我们从三个角度评估生成图像的有效性：图像质量评估（无参考和有参考）、可控性及其在真实世界Bird's Eye View (BEV)分割中的实用性。值得注意的是，生成的数据在无参考质量指标中始终优于原始拼接图像，并能增强下游感知模型。源代码将在此URL公开。 

---
# StixelNExT++: Lightweight Monocular Scene Segmentation and Representation for Collective Perception 

**Title (ZH)**: StixelNExT++：轻量级单目场景分割与表示及其在集体感知中的应用 

**Authors**: Marcel Vosshans, Omar Ait-Aider, Youcef Mezouar, Markus Enzweiler  

**Link**: [PDF](https://arxiv.org/pdf/2507.06687)  

**Abstract**: This paper presents StixelNExT++, a novel approach to scene representation for monocular perception systems. Building on the established Stixel representation, our method infers 3D Stixels and enhances object segmentation by clustering smaller 3D Stixel units. The approach achieves high compression of scene information while remaining adaptable to point cloud and bird's-eye-view representations. Our lightweight neural network, trained on automatically generated LiDAR-based ground truth, achieves real-time performance with computation times as low as 10 ms per frame. Experimental results on the Waymo dataset demonstrate competitive performance within a 30-meter range, highlighting the potential of StixelNExT++ for collective perception in autonomous systems. 

**Abstract (ZH)**: StixelNExT++：一种用于单目感知系统的新型场景表示方法 

---
# Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach 

**Title (ZH)**: 隐含声学映射到达角估计：一种自监督方法 

**Authors**: Adrian S. Roman, Iran R. Roman, Juan P. Bello  

**Link**: [PDF](https://arxiv.org/pdf/2507.07066)  

**Abstract**: Acoustic mapping techniques have long been used in spatial audio processing for direction of arrival estimation (DoAE). Traditional beamforming methods for acoustic mapping, while interpretable, often rely on iterative solvers that can be computationally intensive and sensitive to acoustic variability. On the other hand, recent supervised deep learning approaches offer feedforward speed and robustness but require large labeled datasets and lack interpretability. Despite their strengths, both methods struggle to consistently generalize across diverse acoustic setups and array configurations, limiting their broader applicability. We introduce the Latent Acoustic Mapping (LAM) model, a self-supervised framework that bridges the interpretability of traditional methods with the adaptability and efficiency of deep learning methods. LAM generates high-resolution acoustic maps, adapts to varying acoustic conditions, and operates efficiently across different microphone arrays. We assess its robustness on DoAE using the LOCATA and STARSS benchmarks. LAM achieves comparable or superior localization performance to existing supervised methods. Additionally, we show that LAM's acoustic maps can serve as effective features for supervised models, further enhancing DoAE accuracy and underscoring its potential to advance adaptive, high-performance sound localization systems. 

**Abstract (ZH)**: 声学映射技术在空间音频处理中长期用于到达方向估计（DoAE）。传统声学映射的波束形成方法虽然具有可解释性，但往往依赖于计算成本高且对声学变异性敏感的迭代求解器。另一方面，最近的监督深度学习方法虽然提供前向处理速度和鲁棒性，但需要大量标注数据集且缺乏可解释性。尽管各自具有优势，这两种方法仍难以在多种声学设置和阵列配置中一致地泛化，限制了它们的广泛应用。我们引入了潜声学映射（LAM）模型，这是一种自我监督框架，将传统方法的可解释性与深度学习方法的适应性和效率相结合。LAM生成高分辨率声学地图，能够适应变化的声学条件，并在不同的麦克风阵列上高效运行。我们使用LOCATA和STARSS基准评估其在到达方向估计中的鲁棒性。LAM在定位性能上达到了与现有监督方法相当或更优的效果。此外，我们展示了LAM的声学地图可以作为监督模型的有效特征，进一步提高到达方向估计的准确性，并强调了其在推进适应性强的高性能声定位系统方面潜力。 

---
# Comparative Analysis of CNN and Transformer Architectures with Heart Cycle Normalization for Automated Phonocardiogram Classification 

**Title (ZH)**: 基于心脏周期归一化的CNN和Transformer架构的自动心音图分类比较分析 

**Authors**: Martin Sondermann, Pinar Bisgin, Niklas Tschorn, Anja Burmann, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2507.07058)  

**Abstract**: The automated classification of phonocardiogram (PCG) recordings represents a substantial advancement in cardiovascular diagnostics. This paper presents a systematic comparison of four distinct models for heart murmur detection: two specialized convolutional neural networks (CNNs) and two zero-shot universal audio transformers (BEATs), evaluated using fixed-length and heart cycle normalization approaches. Utilizing the PhysioNet2022 dataset, a custom heart cycle normalization method tailored to individual cardiac rhythms is introduced. The findings indicate the following AUROC values: the CNN model with fixed-length windowing achieves 79.5%, the CNN model with heart cycle normalization scores 75.4%, the BEATs transformer with fixed-length windowing achieves 65.7%, and the BEATs transformer with heart cycle normalization results in 70.1%.
The findings indicate that physiological signal constraints, especially those introduced by different normalization strategies, have a substantial impact on model performance. The research provides evidence-based guidelines for architecture selection in clinical settings, emphasizing the need for a balance between accuracy and computational efficiency. Although specialized CNNs demonstrate superior performance overall, the zero-shot transformer models may offer promising efficiency advantages during development, such as faster training and evaluation cycles, despite their lower classification accuracy. These findings highlight the potential of automated classification systems to enhance cardiac diagnostics and improve patient care. 

**Abstract (ZH)**: 自动分类 Phonocardiogram (PCG) 录音代表了心血管诊断的一个重大进步。本文系统比较了四种不同的心音异常检测模型：两种专门的卷积神经网络 (CNNs) 和两种零样本通用音频变压器 (BEATs)，评估方法包括固定长度和心脏周期归一化。使用 PhysioNet2022 数据集，介绍了针对个体心律定制的心脏周期归一化方法。研究结果表明，AUROC 值分别为：采用固定长度窗口的 CNN 模型为 79.5%，采用心脏周期归一化的 CNN 模型为 75.4%，采用固定长度窗口的 BEATs 转换器为 65.7%，采用心脏周期归一化的 BEATs 转换器为 70.1%。研究表明，生理信号约束，特别是由不同的归一化策略引入的约束，对模型性能有重大影响。研究提供了临床环境中的架构选择指南，强调了准确性与计算效率之间的平衡。尽管专门的 CNN 在整体上表现出色，但零样本变换器模型在开发过程中可能提供显著的效率优势，例如更快的训练和评估周期，尽管它们的分类准确性较低。这些发现突显了自动化分类系统在提高心脏诊断和患者护理方面的潜力。 

---
# CheXPO: Preference Optimization for Chest X-ray VLMs with Counterfactual Rationale 

**Title (ZH)**: CheXPO：基于反事实解释的胸部X光VLMs偏好优化 

**Authors**: Xiao Liang, Jiawei Hu, Di Wang, Zhi Ma, Lin Zhao, Ronghan Li, Bo Wan, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06959)  

**Abstract**: Vision-language models (VLMs) are prone to hallucinations that critically compromise reliability in medical applications. While preference optimization can mitigate these hallucinations through clinical feedback, its implementation faces challenges such as clinically irrelevant training samples, imbalanced data distributions, and prohibitive expert annotation costs. To address these challenges, we introduce CheXPO, a Chest X-ray Preference Optimization strategy that combines confidence-similarity joint mining with counterfactual rationale. Our approach begins by synthesizing a unified, fine-grained multi-task chest X-ray visual instruction dataset across different question types for supervised fine-tuning (SFT). We then identify hard examples through token-level confidence analysis of SFT failures and use similarity-based retrieval to expand hard examples for balancing preference sample distributions, while synthetic counterfactual rationales provide fine-grained clinical preferences, eliminating the need for additional expert input. Experiments show that CheXPO achieves 8.93% relative performance gain using only 5% of SFT samples, reaching state-of-the-art performance across diverse clinical tasks and providing a scalable, interpretable solution for real-world radiology applications. 

**Abstract (ZH)**: Vision-language模型在医学应用中容易产生幻觉，这对可靠性构成严重威胁。尽管通过临床反馈可以优化偏好以减轻这些幻觉，但其实施面临临床无关的训练样本、数据分布不平衡和专家注释成本高昂的挑战。为应对这些挑战，我们引入了CheXPO，一种结合置信相似性联合开采与反事实解释的胸部X光偏好优化策略。该方法首先通过多层次胸部X光视觉指令数据集的合成，为监督微调（SFT）提供统一的多任务训练。随后，通过分析SFT失败的标记级置信度识别困难样本，并利用基于相似性的检索扩展这些困难样本，以平衡偏好样本分布。合成的反事实解释提供了细粒度的临床偏好，从而消除了额外专家输入的需要。实验表明，仅使用SFT样本的5%，CheXPO就能实现8.93%的相对性能提升，达到了各种临床任务的州最先进技术，并提供了一个可扩展且可解释的现实世界放射学应用解决方案。 

---
# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization 

**Title (ZH)**: IAP：感知aware定位与扰动优化下的隐形 adversarial 块攻击 

**Authors**: Subrat Kishore Dutta, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06856)  

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective. 

**Abstract (ZH)**: 基于感知导向定位与扰动优化的高 invisibility 对抗补丁攻击框架 

---
# Physics-Grounded Motion Forecasting via Equation Discovery for Trajectory-Guided Image-to-Video Generation 

**Title (ZH)**: 基于物理原理的动力学预测：方程发现方法在轨迹引导的图像生成视频中的应用 

**Authors**: Tao Feng, Xianbing Zhao, Zhenhua Chen, Tien Tsin Wong, Hamid Rezatofighi, Gholamreza Haffari, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06830)  

**Abstract**: Recent advances in diffusion-based and autoregressive video generation models have achieved remarkable visual realism. However, these models typically lack accurate physical alignment, failing to replicate real-world dynamics in object motion. This limitation arises primarily from their reliance on learned statistical correlations rather than capturing mechanisms adhering to physical laws. To address this issue, we introduce a novel framework that integrates symbolic regression (SR) and trajectory-guided image-to-video (I2V) models for physics-grounded video forecasting. Our approach extracts motion trajectories from input videos, uses a retrieval-based pre-training mechanism to enhance symbolic regression, and discovers equations of motion to forecast physically accurate future trajectories. These trajectories then guide video generation without requiring fine-tuning of existing models. Evaluated on scenarios in Classical Mechanics, including spring-mass, pendulums, and projectile motions, our method successfully recovers ground-truth analytical equations and improves the physical alignment of generated videos over baseline methods. 

**Abstract (ZH)**: Recent Advances in Physics-Grounded Video Forecasting via Symbolic Regression and Trajectory-Guided Image-to-Video Models 

---
# Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data 

**Title (ZH)**: Speckle2Self: 不依赖干净数据的自监督超声 speckle 去噪 

**Authors**: Xuesong Li, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06828)  

**Abstract**: Image denoising is a fundamental task in computer vision, particularly in medical ultrasound (US) imaging, where speckle noise significantly degrades image quality. Although recent advancements in deep neural networks have led to substantial improvements in denoising for natural images, these methods cannot be directly applied to US speckle noise, as it is not purely random. Instead, US speckle arises from complex wave interference within the body microstructure, making it tissue-dependent. This dependency means that obtaining two independent noisy observations of the same scene, as required by pioneering Noise2Noise, is not feasible. Additionally, blind-spot networks also cannot handle US speckle noise due to its high spatial dependency. To address this challenge, we introduce Speckle2Self, a novel self-supervised algorithm for speckle reduction using only single noisy observations. The key insight is that applying a multi-scale perturbation (MSP) operation introduces tissue-dependent variations in the speckle pattern across different scales, while preserving the shared anatomical structure. This enables effective speckle suppression by modeling the clean image as a low-rank signal and isolating the sparse noise component. To demonstrate its effectiveness, Speckle2Self is comprehensively compared with conventional filter-based denoising algorithms and SOTA learning-based methods, using both realistic simulated US images and human carotid US images. Additionally, data from multiple US machines are employed to evaluate model generalization and adaptability to images from unseen domains. \textit{Code and datasets will be released upon acceptance. 

**Abstract (ZH)**: 基于单噪声观察的斑点自缩减方法：Speckle2Self 

---
# Democratizing High-Fidelity Co-Speech Gesture Video Generation 

**Title (ZH)**: 高保真同声手势视频生成的民主化 

**Authors**: Xu Yang, Shaoli Huang, Shenbo Xie, Xuelin Chen, Yifei Liu, Changxing Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.06812)  

**Abstract**: Co-speech gesture video generation aims to synthesize realistic, audio-aligned videos of speakers, complete with synchronized facial expressions and body gestures. This task presents challenges due to the significant one-to-many mapping between audio and visual content, further complicated by the scarcity of large-scale public datasets and high computational demands. We propose a lightweight framework that utilizes 2D full-body skeletons as an efficient auxiliary condition to bridge audio signals with visual outputs. Our approach introduces a diffusion model conditioned on fine-grained audio segments and a skeleton extracted from the speaker's reference image, predicting skeletal motions through skeleton-audio feature fusion to ensure strict audio coordination and body shape consistency. The generated skeletons are then fed into an off-the-shelf human video generation model with the speaker's reference image to synthesize high-fidelity videos. To democratize research, we present CSG-405-the first public dataset with 405 hours of high-resolution videos across 71 speech types, annotated with 2D skeletons and diverse speaker demographics. Experiments show that our method exceeds state-of-the-art approaches in visual quality and synchronization while generalizing across speakers and contexts. 

**Abstract (ZH)**: 同步语言手势视频生成旨在合成与音频同步、具有同步面部表情和身体手势的模拟视频。由于音频与视觉内容之间存在显著的一对多映射关系，加之大规模公共数据集稀缺和高计算需求的复杂性，使得此任务充满挑战。我们提出了一种轻量级框架，利用2D全身骨架作为高效辅助条件以桥接音频信号和视觉输出。我们的方法通过细粒度音频片段和从演讲者参考图像中提取的骨架条件化一个扩散模型，预测通过骨架-音频特征融合得到的骨架动作，确保严格的音频同步和身体形状一致性。生成的骨架随后被输入到带有演讲者参考图像的标准现成人类视频生成模型中以合成高保真视频。为了普惠研究，我们呈现了CSG-405数据集——首个包含405小时高分辨率视频、覆盖71种语音类型且附带2D骨架和多元演讲者人口统计信息的公共数据集。实验结果表明，我们的方法在视觉质量和同步性方面超过了现有最先进的方法，能够在不同演讲者和场景下泛化。 

---
# DIFFUMA: High-Fidelity Spatio-Temporal Video Prediction via Dual-Path Mamba and Diffusion Enhancement 

**Title (ZH)**: DIFFUMA: 高保真时空视频预测通过双路径Mamba和扩散增强 

**Authors**: Xinyu Xie, Weifeng Cao, Jun Shi, Yangyang Hu, Hui Liang, Wanyong Liang, Xiaoliang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.06738)  

**Abstract**: Spatio-temporal video prediction plays a pivotal role in critical domains, ranging from weather forecasting to industrial automation. However, in high-precision industrial scenarios such as semiconductor manufacturing, the absence of specialized benchmark datasets severely hampers research on modeling and predicting complex processes. To address this challenge, we make a twofold this http URL, we construct and release the Chip Dicing Lane Dataset (CHDL), the first public temporal image dataset dedicated to the semiconductor wafer dicing process. Captured via an industrial-grade vision system, CHDL provides a much-needed and challenging benchmark for high-fidelity process modeling, defect detection, and digital twin this http URL, we propose DIFFUMA, an innovative dual-path prediction architecture specifically designed for such fine-grained dynamics. The model captures global long-range temporal context through a parallel Mamba module, while simultaneously leveraging a diffusion module, guided by temporal features, to restore and enhance fine-grained spatial details, effectively combating feature degradation. Experiments demonstrate that on our CHDL benchmark, DIFFUMA significantly outperforms existing methods, reducing the Mean Squared Error (MSE) by 39% and improving the Structural Similarity (SSIM) from 0.926 to a near-perfect 0.988. This superior performance also generalizes to natural phenomena datasets. Our work not only delivers a new state-of-the-art (SOTA) model but, more importantly, provides the community with an invaluable data resource to drive future research in industrial AI. 

**Abstract (ZH)**: 空间时间视频预测在天气预报、工业自动化等领域发挥着关键作用。然而，在如半导体制造这样的高精度工业场景中，缺乏专门的基准数据集严重阻碍了复杂过程建模和预测的研究。为应对这一挑战，我们从两个方面着手：首先，我们构建并发布了Chip Dicing Lane Dataset (CHDL)，这是首个专注于半导体晶圆切割过程的公开时间图像数据集。通过工业级视觉系统捕获，CHDL为高保真工艺建模、缺陷检测和数字孪生提供了急需且具有挑战性的基准。其次，我们提出了DIFFUMA，一种专门针对此类精细动态的创新双路径预测架构。该模型通过并行Mamba模块捕捉全局长时间上下文，同时利用受时间特征指导的扩散模块恢复和增强细粒度的空间细节，有效对抗特征退化。实验结果表明，在我们的CHDL基准上，DIFFUMA显著优于现有方法，使均方误差（MSE）降低了39%，结构相似性（SSIM）从0.926提升到近乎完美的0.988。这种性能优势同样适用于自然现象数据集。我们的工作不仅提供了新的前沿模型，更重要的是为社区提供了宝贵的数据资源，推动未来工业AI的研究。 

---
# Photometric Stereo using Gaussian Splatting and inverse rendering 

**Title (ZH)**: 基于高斯点扩散和逆渲染的光度立体视觉 

**Authors**: Matéo Ducastel, David Tschumperlé, Yvain Quéau  

**Link**: [PDF](https://arxiv.org/pdf/2507.06684)  

**Abstract**: Recent state-of-the-art algorithms in photometric stereo rely on neural networks and operate either through prior learning or inverse rendering optimization. Here, we revisit the problem of calibrated photometric stereo by leveraging recent advances in 3D inverse rendering using the Gaussian Splatting formalism. This allows us to parameterize the 3D scene to be reconstructed and optimize it in a more interpretable manner. Our approach incorporates a simplified model for light representation and demonstrates the potential of the Gaussian Splatting rendering engine for the photometric stereo problem. 

**Abstract (ZH)**: Recent State-of-the-Art Algorithms in Photometric Stereo Rely on Neural Networks and Operate through Prior Learning or Inverse Rendering Optimization: Revisiting Calibrated Photometric Stereo with Gaussian Splatting Formalism 

---
# MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval 

**Title (ZH)**: MS-DPPs：多源行列式点过程在文本到图像检索中复合属性上下文多样性精炼中的应用 

**Authors**: Naoya Sogi, Takashi Shibata, Makoto Terao, Masanori Suganuma, Takayuki Okatani  

**Link**: [PDF](https://arxiv.org/pdf/2507.06654)  

**Abstract**: Result diversification (RD) is a crucial technique in Text-to-Image Retrieval for enhancing the efficiency of a practical application. Conventional methods focus solely on increasing the diversity metric of image appearances. However, the diversity metric and its desired value vary depending on the application, which limits the applications of RD. This paper proposes a novel task called CDR-CA (Contextual Diversity Refinement of Composite Attributes). CDR-CA aims to refine the diversities of multiple attributes, according to the application's context. To address this task, we propose Multi-Source DPPs, a simple yet strong baseline that extends the Determinantal Point Process (DPP) to multi-sources. We model MS-DPP as a single DPP model with a unified similarity matrix based on a manifold representation. We also introduce Tangent Normalization to reflect contexts. Extensive experiments demonstrate the effectiveness of the proposed method. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于上下文的复合属性多样化 refinement (CDR-CA)：文本到图像检索中的结果多样化 

---
# EXAONE Path 2.0: Pathology Foundation Model with End-to-End Supervision 

**Title (ZH)**: EXAONE 路径 2.0：端到端监督的病理学基础模型 

**Authors**: Myungjang Pyeon, Janghyeon Lee, Minsoo Lee, Juseung Yun, Hwanil Choi, Jonghyun Kim, Jiwon Kim, Yi Hu, Jongseong Jang, Soonyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06639)  

**Abstract**: In digital pathology, whole-slide images (WSIs) are often difficult to handle due to their gigapixel scale, so most approaches train patch encoders via self-supervised learning (SSL) and then aggregate the patch-level embeddings via multiple instance learning (MIL) or slide encoders for downstream tasks. However, patch-level SSL may overlook complex domain-specific features that are essential for biomarker prediction, such as mutation status and molecular characteristics, as SSL methods rely only on basic augmentations selected for natural image domains on small patch-level area. Moreover, SSL methods remain less data efficient than fully supervised approaches, requiring extensive computational resources and datasets to achieve competitive performance. To address these limitations, we present EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Using only 37k WSIs for training, EXAONE Path 2.0 achieves state-of-the-art average performance across 10 biomarker prediction tasks, demonstrating remarkable data efficiency. 

**Abstract (ZH)**: 数字病理学中，全滑片图像（WSIs）因其 gigapixel 规模而难以处理，因此大多数方法通过自主监督学习（SSL）训练补丁编码器，然后通过多重实例学习（MIL）或滑片编码器聚合补丁级别嵌入用于下游任务。然而，补丁级别的 SSL 可能会忽略对于生物标志物预测至关重要的复杂领域特定特征，如突变状态和分子特性，因为 SSL 方法仅依赖于针对自然图像领域选择的基本增强，在小补丁级别区域上应用。此外，SSL 方法在数据效率上仍不及完全监督方法，需要大量计算资源和数据集才能实现竞争力的性能。为解决这些限制，我们提出了 EXAONE Path 2.0，这是一种在直接的滑片级别监督下学习补丁级别表示的病理学基础模型。仅使用 37,000 张 WSIs 进行训练，EXAONE Path 2.0 在 10 项生物标志物预测任务中实现了最先进的平均性能，展示了显著的数据效率。 

---
# EA: An Event Autoencoder for High-Speed Vision Sensing 

**Title (ZH)**: EA: 一种用于高speed视觉感知的事件自动编码器 

**Authors**: Riadul Islam, Joey Mulé, Dhandeep Challagundla, Shahmir Rizvi, Sean Carson  

**Link**: [PDF](https://arxiv.org/pdf/2507.06459)  

**Abstract**: High-speed vision sensing is essential for real-time perception in applications such as robotics, autonomous vehicles, and industrial automation. Traditional frame-based vision systems suffer from motion blur, high latency, and redundant data processing, limiting their performance in dynamic environments. Event cameras, which capture asynchronous brightness changes at the pixel level, offer a promising alternative but pose challenges in object detection due to sparse and noisy event streams. To address this, we propose an event autoencoder architecture that efficiently compresses and reconstructs event data while preserving critical spatial and temporal features. The proposed model employs convolutional encoding and incorporates adaptive threshold selection and a lightweight classifier to enhance recognition accuracy while reducing computational complexity. Experimental results on the existing Smart Event Face Dataset (SEFD) demonstrate that our approach achieves comparable accuracy to the YOLO-v4 model while utilizing up to $35.5\times$ fewer parameters. Implementations on embedded platforms, including Raspberry Pi 4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8 FPS. The proposed classifier exhibits up to 87.84x better FPS than the state-of-the-art and significantly improves event-based vision performance, making it ideal for low-power, high-speed applications in real-time edge computing. 

**Abstract (ZH)**: 高速视觉感知对于机器人、自动驾驶车辆和工业自动化等应用的实时感知至关重要。传统的基于帧的视觉系统由于存在运动模糊、高延迟和冗余数据处理等问题，限制了它们在动态环境中的性能。事件摄像头通过在像素级捕获异步亮度变化，提供了有前途的替代方案，但由于事件流稀疏且噪声大，给物体检测带来了挑战。为了解决这个问题，我们提出了一种事件自编码器架构，该架构能够高效地压缩和重构事件数据，同时保留关键的空间和时间特征。所提出模型采用了卷积编码，并结合自适应阈值选择和轻量级分类器，以提高识别准确性并降低计算复杂度。在现有的Smart Event Face Dataset (SEFD)上的实验结果表明，我们的方法在参数量减少到YOLO-v4模型的35.5倍的同时达到了相当的准确性。在Raspberry Pi 4B和NVIDIA Jetson Nano等嵌入式平台上实现显示了高帧率，从8 FPS到44.8 FPS。所提出的分类器在帧率上比最先进的方法高出87.84倍，显著提高了基于事件的视觉性能，使其成为低功耗、高帧率的实时边缘计算应用的理想选择。 

---
# A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry 

**Title (ZH)**: 利用三维几何进行不确定性量化的一种概率方法 

**Authors**: Rushil Desai, Frederik Warburg, Trevor Darrell, Marissa Ramirez de Chanlatte  

**Link**: [PDF](https://arxiv.org/pdf/2507.06269)  

**Abstract**: Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making. 

**Abstract (ZH)**: 量化基于-signed距离函数-的神经隐式3D表示中的不确定性仍然是一个重大挑战，原因在于计算效率低下、可扩展性问题以及几何不一致。现有的方法通常忽视直接的几何集成，导致不确定性地图校准不良。我们提出了BayesSDF，这是一种基于概率框架的新型不确定性量化方法，特别适用于具有3D环境（如森林）的科学模拟应用（如森林中的流体流动建模），其中精确的表面几何和对表面几何不确定性有了准确的认识是至关重要的。与基于辐射的模型（如NeRF或3D正态分布散点图）相比，SDF定义连续可微的几何形状，使其更适合物理建模和分析。BayesSDF利用拉普拉斯近似通过海森矩阵度量来量化局部表面不稳定性，从而实现高效的、具备表面意识的不确定性估计。我们的方法表明，不确定性预测与重建不当的几何结构高度相关，为下游使用提供了可操作的信心度量。在合成和真实世界数据集上进行的广泛评估表明，BayesSDF在校准和几何一致性方面均优于现有方法，为不确定性感知的3D场景重建、模拟和机器人决策提供了坚实的基础。 

---
