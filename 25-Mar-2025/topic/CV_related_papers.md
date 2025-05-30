# Online 3D Scene Reconstruction Using Neural Object Priors 

**Title (ZH)**: 基于神经对象先验的在线三维场景重建 

**Authors**: Thomas Chabal, Shizhe Chen, Jean Ponce, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2503.18897)  

**Abstract**: This paper addresses the problem of reconstructing a scene online at the level of objects given an RGB-D video sequence. While current object-aware neural implicit representations hold promise, they are limited in online reconstruction efficiency and shape completion. Our main contributions to alleviate the above limitations are twofold. First, we propose a feature grid interpolation mechanism to continuously update grid-based object-centric neural implicit representations as new object parts are revealed. Second, we construct an object library with previously mapped objects in advance and leverage the corresponding shape priors to initialize geometric object models in new videos, subsequently completing them with novel views as well as synthesized past views to avoid losing original object details. Extensive experiments on synthetic environments from the Replica dataset, real-world ScanNet sequences and videos captured in our laboratory demonstrate that our approach outperforms state-of-the-art neural implicit models for this task in terms of reconstruction accuracy and completeness. 

**Abstract (ZH)**: 本文针对给定RGB-D视频序列在线重建场景中逐对象级别的问题进行了研究。虽然当前的对象感知神经隐式表示充满潜力，但在在线重建效率和形状完成方面存在局限性。本文为缓解上述局限性做出了两项主要贡献。首先，我们提出了一种特征格点插值机制，以连续更新基于格点的对象中心神经隐式表示，随着新的对象部分被揭示。其次，我们预先构建了一个对象库，包含之前映射的对象，并利用相应的形状先验来初始化新的视频中的几何对象模型，随后通过新的视角以及合成的过去视角来完成它们，以避免丢失原始对象的细节。在来自Replica数据集的合成环境、真实世界ScanNet序列以及在实验室拍摄的视频上的广泛实验表明，本文方法在重建准确性和完整性方面优于该任务的现有神经隐式模型。 

---
# Any6D: Model-free 6D Pose Estimation of Novel Objects 

**Title (ZH)**: Any6D: 无需模型的新型物体6D姿态估计 

**Authors**: Taeyeop Lee, Bowen Wen, Minjun Kang, Gyuree Kang, In So Kweon, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2503.18673)  

**Abstract**: We introduce Any6D, a model-free framework for 6D object pose estimation that requires only a single RGB-D anchor image to estimate both the 6D pose and size of unknown objects in novel scenes. Unlike existing methods that rely on textured 3D models or multiple viewpoints, Any6D leverages a joint object alignment process to enhance 2D-3D alignment and metric scale estimation for improved pose accuracy. Our approach integrates a render-and-compare strategy to generate and refine pose hypotheses, enabling robust performance in scenarios with occlusions, non-overlapping views, diverse lighting conditions, and large cross-environment variations. We evaluate our method on five challenging datasets: REAL275, Toyota-Light, HO3D, YCBINEOAT, and LM-O, demonstrating its effectiveness in significantly outperforming state-of-the-art methods for novel object pose estimation. Project page: this https URL 

**Abstract (ZH)**: Any6D：一种仅需单张RGB-D 锚图即可进行未知对象六自由度姿态估计的模型无依赖框架 

---
# PanopticSplatting: End-to-End Panoptic Gaussian Splatting 

**Title (ZH)**: 全景splatting: 全局端到端全景高斯splatting 

**Authors**: Yuxuan Xie, Xuan Yu, Changjian Jiang, Sitong Mao, Shunbo Zhou, Rui Fan, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18073)  

**Abstract**: Open-vocabulary panoptic reconstruction is a challenging task for simultaneous scene reconstruction and understanding. Recently, methods have been proposed for 3D scene understanding based on Gaussian splatting. However, these methods are multi-staged, suffering from the accumulated errors and the dependence of hand-designed components. To streamline the pipeline and achieve global optimization, we propose PanopticSplatting, an end-to-end system for open-vocabulary panoptic reconstruction. Our method introduces query-guided Gaussian segmentation with local cross attention, lifting 2D instance masks without cross-frame association in an end-to-end way. The local cross attention within view frustum effectively reduces the training memory, making our model more accessible to large scenes with more Gaussians and objects. In addition, to address the challenge of noisy labels in 2D pseudo masks, we propose label blending to promote consistent 3D segmentation with less noisy floaters, as well as label warping on 2D predictions which enhances multi-view coherence and segmentation accuracy. Our method demonstrates strong performances in 3D scene panoptic reconstruction on the ScanNet-V2 and ScanNet++ datasets, compared with both NeRF-based and Gaussian-based panoptic reconstruction methods. Moreover, PanopticSplatting can be easily generalized to numerous variants of Gaussian splatting, and we demonstrate its robustness on different Gaussian base models. 

**Abstract (ZH)**: 面向开放式词汇泛视图重建的端到端高斯点云化方法 

---
# Video-T1: Test-Time Scaling for Video Generation 

**Title (ZH)**: 视频-T1：视频生成的测试时缩放 

**Authors**: Fangfu Liu, Hanyang Wang, Yimo Cai, Kaiyan Zhang, Xiaohang Zhan, Yueqi Duan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18942)  

**Abstract**: With the scale capability of increasing training data, model size, and computational cost, video generation has achieved impressive results in digital creation, enabling users to express creativity across various domains. Recently, researchers in Large Language Models (LLMs) have expanded the scaling to test-time, which can significantly improve LLM performance by using more inference-time computation. Instead of scaling up video foundation models through expensive training costs, we explore the power of Test-Time Scaling (TTS) in video generation, aiming to answer the question: if a video generation model is allowed to use non-trivial amount of inference-time compute, how much can it improve generation quality given a challenging text prompt. In this work, we reinterpret the test-time scaling of video generation as a searching problem to sample better trajectories from Gaussian noise space to the target video distribution. Specifically, we build the search space with test-time verifiers to provide feedback and heuristic algorithms to guide searching process. Given a text prompt, we first explore an intuitive linear search strategy by increasing noise candidates at inference time. As full-step denoising all frames simultaneously requires heavy test-time computation costs, we further design a more efficient TTS method for video generation called Tree-of-Frames (ToF) that adaptively expands and prunes video branches in an autoregressive manner. Extensive experiments on text-conditioned video generation benchmarks demonstrate that increasing test-time compute consistently leads to significant improvements in the quality of videos. Project page: this https URL 

**Abstract (ZH)**: 随着训练数据、模型规模和计算成本的增加，视频生成在数字创作领域取得了令人印象深刻的成果，使用户能够在各个领域表达创意。最近，大型语言模型（LLMs）的研究人员将扩展范围扩大到测试时间，通过更多的推理时间计算可以显著提高LLM的性能。我们没有通过昂贵的训练成本来扩大视频基础模型的规模，而是探索了测试时间缩放（TTS）在视频生成中的作用，旨在回答这样一个问题：如果允许视频生成模型在推理时间使用非平凡量的计算资源，对于具有挑战性的文本提示，它的生成质量可以提高多少。在这项工作中，我们将视频生成的测试时间缩放重新解释为一个搜索问题，即从高斯噪声空间到目标视频分布中采样更好的轨迹。具体而言，我们构建了测试时间验证器和启发式算法来提供反馈并引导搜索过程。给定一个文本提示，我们首先通过在推理时间增加噪声候选者来探索一个直观的线性搜索策略。由于同时对所有帧进行完整的去噪计算需要大量的测试时间计算成本，我们进一步设计了一种更高效的视频生成的TTS方法，称为帧树（ToF），该方法以自回归方式适当地扩展和修剪视频分支。在针对条件文本视频生成基准的广泛实验中，我们证明了增加测试时间计算资源可以一致地显著提高视频质量。项目页面：这个 https URL。 

---
# Dual-domain Multi-path Self-supervised Diffusion Model for Accelerated MRI Reconstruction 

**Title (ZH)**: 加速MRI重建的双域多路径自我监督扩散模型 

**Authors**: Yuxuan Zhang, Jinkui Hao, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.18836)  

**Abstract**: Magnetic resonance imaging (MRI) is a vital diagnostic tool, but its inherently long acquisition times reduce clinical efficiency and patient comfort. Recent advancements in deep learning, particularly diffusion models, have improved accelerated MRI reconstruction. However, existing diffusion models' training often relies on fully sampled data, models incur high computational costs, and often lack uncertainty estimation, limiting their clinical applicability. To overcome these challenges, we propose a novel framework, called Dual-domain Multi-path Self-supervised Diffusion Model (DMSM), that integrates a self-supervised dual-domain diffusion model training scheme, a lightweight hybrid attention network for the reconstruction diffusion model, and a multi-path inference strategy, to enhance reconstruction accuracy, efficiency, and explainability. Unlike traditional diffusion-based models, DMSM eliminates the dependency on training from fully sampled data, making it more practical for real-world clinical settings. We evaluated DMSM on two human MRI datasets, demonstrating that it achieves favorable performance over several supervised and self-supervised baselines, particularly in preserving fine anatomical structures and suppressing artifacts under high acceleration factors. Additionally, our model generates uncertainty maps that correlate reasonably well with reconstruction errors, offering valuable clinically interpretable guidance and potentially enhancing diagnostic confidence. 

**Abstract (ZH)**: 磁共振成像(MRI)是一种重要的诊断工具，但其固有的长时间采集时间降低了临床效率和患者的舒适度。最近深度学习，尤其是扩散模型的发展，提高了加速MRI重建的效果。然而，现有的扩散模型训练通常依赖于完全采样数据，模型计算成本高，并且往往缺乏不确定性估计，限制了其临床应用。为克服这些挑战，我们提出了一种新型框架，称为双域多路径自主监督扩散模型(Dual-domain Multi-path Self-supervised Diffusion Model, DMSM)，该框架结合了双域自主监督扩散模型训练方案、轻量级混合注意力网络以及多路径推理策略，以提高重建精度、效率和可解释性。与传统的基于扩散的方法不同，DMSM 消除了对完全采样数据进行训练的依赖，使其更适合现实临床环境。我们在两个人类MRI数据集上评估了DMSM，结果显示，即使在高加速因子下，它仍能优于多种监督和自主监督的基线模型，特别是在保留精细解剖结构和抑制伪影方面。此外，我们的模型生成的不确定性图与重建误差的相关性较好，提供了有价值的临床可解释指导，可能增强诊断信心。 

---
# Frequency Dynamic Convolution for Dense Image Prediction 

**Title (ZH)**: 频率动态卷积用于密集图像预测 

**Authors**: Linwei Chen, Lin Gu, Liang Li, Chenggang Yan, Ying Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18783)  

**Abstract**: While Dynamic Convolution (DY-Conv) has shown promising performance by enabling adaptive weight selection through multiple parallel weights combined with an attention mechanism, the frequency response of these weights tends to exhibit high similarity, resulting in high parameter costs but limited adaptability. In this work, we introduce Frequency Dynamic Convolution (FDConv), a novel approach that mitigates these limitations by learning a fixed parameter budget in the Fourier domain. FDConv divides this budget into frequency-based groups with disjoint Fourier indices, enabling the construction of frequency-diverse weights without increasing the parameter cost. To further enhance adaptability, we propose Kernel Spatial Modulation (KSM) and Frequency Band Modulation (FBM). KSM dynamically adjusts the frequency response of each filter at the spatial level, while FBM decomposes weights into distinct frequency bands in the frequency domain and modulates them dynamically based on local content. Extensive experiments on object detection, segmentation, and classification validate the effectiveness of FDConv. We demonstrate that when applied to ResNet-50, FDConv achieves superior performance with a modest increase of +3.6M parameters, outperforming previous methods that require substantial increases in parameter budgets (e.g., CondConv +90M, KW +76.5M). Moreover, FDConv seamlessly integrates into a variety of architectures, including ConvNeXt, Swin-Transformer, offering a flexible and efficient solution for modern vision tasks. The code is made publicly available at this https URL. 

**Abstract (ZH)**: 频率动态卷积：一种新型的参数预算学习方法及其应用 

---
# Mechanistic Interpretability of Fine-Tuned Vision Transformers on Distorted Images: Decoding Attention Head Behavior for Transparent and Trustworthy AI 

**Title (ZH)**: 微调的视觉变换器在失真图像上的机制可解释性：解码注意力头行为以实现透明和可信赖的AI 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2503.18762)  

**Abstract**: Mechanistic interpretability improves the safety, reliability, and robustness of large AI models. This study examined individual attention heads in vision transformers (ViTs) fine tuned on distorted 2D spectrogram images containing non relevant content (axis labels, titles, color bars). By introducing extraneous features, the study analyzed how transformer components processed unrelated information, using mechanistic interpretability to debug issues and reveal insights into transformer architectures. Attention maps assessed head contributions across layers. Heads in early layers (1 to 3) showed minimal task impact with ablation increased MSE loss slightly ({\mu}=0.11%, {\sigma}=0.09%), indicating focus on less critical low level features. In contrast, deeper heads (e.g., layer 6) caused a threefold higher loss increase ({\mu}=0.34%, {\sigma}=0.02%), demonstrating greater task importance. Intermediate layers (6 to 11) exhibited monosemantic behavior, attending exclusively to chirp regions. Some early heads (1 to 4) were monosemantic but non task relevant (e.g. text detectors, edge or corner detectors). Attention maps distinguished monosemantic heads (precise chirp localization) from polysemantic heads (multiple irrelevant regions). These findings revealed functional specialization in ViTs, showing how heads processed relevant vs. extraneous information. By decomposing transformers into interpretable components, this work enhanced model understanding, identified vulnerabilities, and advanced safer, more transparent AI. 

**Abstract (ZH)**: 机制可解释性提高大型AI模型的安全性、可靠性和鲁棒性。本研究探讨了在包含无关内容（轴标签、标题、颜色条）的失真2D谱图图像上微调的视觉变换器（ViTs）中的个体注意力头部。通过引入额外特征，研究分析了变压器组件处理无关信息的方式，利用机制可解释性来调试问题并揭示变压器架构的见解。注意力图评估了各层头部的贡献。早期层（1到3）的头部对任务影响较小，移除后稍增均方误差损失（μ=0.11%，σ=0.09%），表明关注于较为次要的低级特征。相比之下，较深层（如第6层）导致损失增加三倍多（μ=0.34%，σ=0.02%），表明任务重要性更大。中层（6到11）表现出单一语义行为，仅关注 chirp 区域。一些早期头部（1到4）表现出单一语义但与任务无关（例如文本检测器、边缘或角落检测器）。注意力图区分了单一语义头部（精确 chirp 定位）和多重语义头部（多个无关区域）。这些发现揭示了ViTs的功能专业化，展示了头部如何处理相关 vs. 无关信息。通过将变压器分解为可解释组件，本研究增强了模型理解，识别出了漏洞，并推进了更安全、更透明的AI。 

---
# EgoSurgery-HTS: A Dataset for Egocentric Hand-Tool Segmentation in Open Surgery Videos 

**Title (ZH)**: EgoSurgery-HTS：开放手术视频中的自我中心手 TOOL分割数据集 

**Authors**: Nathan Darjana, Ryo Fujii, Hideo Saito, Hiroki Kajita  

**Link**: [PDF](https://arxiv.org/pdf/2503.18755)  

**Abstract**: Egocentric open-surgery videos capture rich, fine-grained details essential for accurately modeling surgical procedures and human behavior in the operating room. A detailed, pixel-level understanding of hands and surgical tools is crucial for interpreting a surgeon's actions and intentions. We introduce EgoSurgery-HTS, a new dataset with pixel-wise annotations and a benchmark suite for segmenting surgical tools, hands, and interacting tools in egocentric open-surgery videos. Specifically, we provide a labeled dataset for (1) tool instance segmentation of 14 distinct surgical tools, (2) hand instance segmentation, and (3) hand-tool segmentation to label hands and the tools they manipulate. Using EgoSurgery-HTS, we conduct extensive evaluations of state-of-the-art segmentation methods and demonstrate significant improvements in the accuracy of hand and hand-tool segmentation in egocentric open-surgery videos compared to existing datasets. The dataset will be released at this https URL. 

**Abstract (ZH)**: 自视点开放手术视频捕捉到富含细粒度细节，对于准确建模手术过程和手术室中的人类行为至关重要。对手和手术器械的详细像素级理解对于解析外科医生的动作和意图至关重要。我们介绍了EgoSurgery-HTS新数据集及其分割外科器械、手部和手工具交互的基准套件，提供了用于（1）14种不同手术器械实例分割，（2）手实例分割，以及（3）手工具分割以标注手部及其操控的器械的标注数据。使用EgoSurgery-HTS，我们对最先进的分割方法进行了广泛评估，并展示了在自视点开放手术视频中手和手工具分割准确性上的显著改进，超过了现有数据集。数据集将在以下链接发布：这个 https URL。 

---
# Dig2DIG: Dig into Diffusion Information Gains for Image Fusion 

**Title (ZH)**: Dig2DIG: 探究扩散信息增益的图像融合方法 

**Authors**: Bing Cao, Baoshuo Cai, Changqing Zhang, Qinghua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18627)  

**Abstract**: Image fusion integrates complementary information from multi-source images to generate more informative results. Recently, the diffusion model, which demonstrates unprecedented generative potential, has been explored in image fusion. However, these approaches typically incorporate predefined multimodal guidance into diffusion, failing to capture the dynamically changing significance of each modality, while lacking theoretical guarantees. To address this issue, we reveal a significant spatio-temporal imbalance in image denoising; specifically, the diffusion model produces dynamic information gains in different image regions with denoising steps. Based on this observation, we Dig into the Diffusion Information Gains (Dig2DIG) and theoretically derive a diffusion-based dynamic image fusion framework that provably reduces the upper bound of the generalization error. Accordingly, we introduce diffusion information gains (DIG) to quantify the information contribution of each modality at different denoising steps, thereby providing dynamic guidance during the fusion process. Extensive experiments on multiple fusion scenarios confirm that our method outperforms existing diffusion-based approaches in terms of both fusion quality and inference efficiency. 

**Abstract (ZH)**: 图像融合通过整合多源图像中的互补信息以生成更具信息量的结果。近期，展现出前所未有的生成潜力的扩散模型被探索应用于图像融合。然而，这些方法通常将预定义的多模态指导信息融入扩散模型中，未能捕捉到每种模态动态变化的重要性，且缺乏理论保证。为了解决这一问题，我们揭示了图像去噪过程中时空不平衡的现象；具体而言，扩散模型在去噪步骤中于不同的图像区域产生动态的信息增益。基于这一观察，我们探索了扩散信息增益（Dig2DIG），并理论上推导出一个基于扩散的动态图像融合框架，可证明地降低泛化误差的上界。据此，我们引入了扩散信息增益（DIG）来量化每种模态在不同去噪步骤中的信息贡献，从而在融合过程中提供动态指导。多项融合场景下的实验结果证实，我们的方法在融合质量和推理效率方面均优于现有的基于扩散模型的方法。 

---
# EvAnimate: Event-conditioned Image-to-Video Generation for Human Animation 

**Title (ZH)**: 基于事件的人像动画的图像到视频生成：EvAnimate 

**Authors**: Qiang Qu, Ming Li, Xiaoming Chen, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18552)  

**Abstract**: Conditional human animation transforms a static reference image into a dynamic sequence by applying motion cues such as poses. These motion cues are typically derived from video data but are susceptible to limitations including low temporal resolution, motion blur, overexposure, and inaccuracies under low-light conditions. In contrast, event cameras provide data streams with exceptionally high temporal resolution, a wide dynamic range, and inherent resistance to motion blur and exposure issues. In this work, we propose EvAnimate, a framework that leverages event streams as motion cues to animate static human images. Our approach employs a specialized event representation that transforms asynchronous event streams into 3-channel slices with controllable slicing rates and appropriate slice density, ensuring compatibility with diffusion models. Subsequently, a dual-branch architecture generates high-quality videos by harnessing the inherent motion dynamics of the event streams, thereby enhancing both video quality and temporal consistency. Specialized data augmentation strategies further enhance cross-person generalization. Finally, we establish a new benchmarking, including simulated event data for training and validation, and a real-world event dataset capturing human actions under normal and extreme scenarios. The experiment results demonstrate that EvAnimate achieves high temporal fidelity and robust performance in scenarios where traditional video-derived cues fall short. 

**Abstract (ZH)**: 基于事件的条件人体动画框架 

---
# HiRes-FusedMIM: A High-Resolution RGB-DSM Pre-trained Model for Building-Level Remote Sensing Applications 

**Title (ZH)**: HiRes-FusedMIM：一种用于建筑级遥感应用的高分辨率RGB-DSM预训练模型 

**Authors**: Guneet Mutreja, Philipp Schuegraf, Ksenia Bittner  

**Link**: [PDF](https://arxiv.org/pdf/2503.18540)  

**Abstract**: Recent advances in self-supervised learning have led to the development of foundation models that have significantly advanced performance in various computer vision tasks. However, despite their potential, these models often overlook the crucial role of high-resolution digital surface models (DSMs) in understanding urban environments, particularly for building-level analysis, which is essential for applications like digital twins. To address this gap, we introduce HiRes-FusedMIM, a novel pre-trained model specifically designed to leverage the rich information contained within high-resolution RGB and DSM data. HiRes-FusedMIM utilizes a dual-encoder simple masked image modeling (SimMIM) architecture with a multi-objective loss function that combines reconstruction and contrastive objectives, enabling it to learn powerful, joint representations from both modalities. We conducted a comprehensive evaluation of HiRes-FusedMIM on a diverse set of downstream tasks, including classification, semantic segmentation, and instance segmentation. Our results demonstrate that: 1) HiRes-FusedMIM outperforms previous state-of-the-art geospatial methods on several building-related datasets, including WHU Aerial and LoveDA, demonstrating its effectiveness in capturing and leveraging fine-grained building information; 2) Incorporating DSMs during pre-training consistently improves performance compared to using RGB data alone, highlighting the value of elevation information for building-level analysis; 3) The dual-encoder architecture of HiRes-FusedMIM, with separate encoders for RGB and DSM data, significantly outperforms a single-encoder model on the Vaihingen segmentation task, indicating the benefits of learning specialized representations for each modality. To facilitate further research and applications in this direction, we will publicly release the trained model weights. 

**Abstract (ZH)**: Recent Advances in Self-Supervised Learning Have Led to the Development of Foundation Models that Have Significantly Advanced Performance in Various Computer Vision Tasks: Addressing the Overlooked Role of High-Resolution Digital Surface Models in Urban Analysis with HiRes-FusedMIM 

---
# MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse 

**Title (ZH)**: MetaSpatial：增强元宇宙中VLMs的三维空间推理能力 

**Authors**: Zhenyu Pan, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18470)  

**Abstract**: We present MetaSpatial, the first reinforcement learning (RL)-based framework designed to enhance 3D spatial reasoning in vision-language models (VLMs), enabling real-time 3D scene generation without the need for hard-coded optimizations. MetaSpatial addresses two core challenges: (i) the lack of internalized 3D spatial reasoning in VLMs, which limits their ability to generate realistic layouts, and (ii) the inefficiency of traditional supervised fine-tuning (SFT) for layout generation tasks, as perfect ground truth annotations are unavailable. Our key innovation is a multi-turn RL-based optimization mechanism that integrates physics-aware constraints and rendered image evaluations, ensuring generated 3D layouts are coherent, physically plausible, and aesthetically consistent. Methodologically, MetaSpatial introduces an adaptive, iterative reasoning process, where the VLM refines spatial arrangements over multiple turns by analyzing rendered outputs, improving scene coherence progressively. Empirical evaluations demonstrate that MetaSpatial significantly enhances the spatial consistency and formatting stability of various scale models. Post-training, object placements are more realistic, aligned, and functionally coherent, validating the effectiveness of RL for 3D spatial reasoning in metaverse, AR/VR, digital twins, and game development applications. Our code, data, and training pipeline are publicly available at this https URL. 

**Abstract (ZH)**: MetaSpatial：一种基于强化学习的增强视觉语言模型三维空间推理的框架 

---
# Resource-Efficient Motion Control for Video Generation via Dynamic Mask Guidance 

**Title (ZH)**: 基于动态掩码引导的资源高效运动控制视频生成 

**Authors**: Sicong Feng, Jielong Yang, Li Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.18386)  

**Abstract**: Recent advances in diffusion models bring new vitality to visual content creation. However, current text-to-video generation models still face significant challenges such as high training costs, substantial data requirements, and difficulties in maintaining consistency between given text and motion of the foreground object. To address these challenges, we propose mask-guided video generation, which can control video generation through mask motion sequences, while requiring limited training data. Our model enhances existing architectures by incorporating foreground masks for precise text-position matching and motion trajectory control. Through mask motion sequences, we guide the video generation process to maintain consistent foreground objects throughout the sequence. Additionally, through a first-frame sharing strategy and autoregressive extension approach, we achieve more stable and longer video generation. Extensive qualitative and quantitative experiments demonstrate that this approach excels in various video generation tasks, such as video editing and generating artistic videos, outperforming previous methods in terms of consistency and quality. Our generated results can be viewed in the supplementary materials. 

**Abstract (ZH)**: 近期扩散模型的发展为视觉内容创作带来了新的活力。然而，当前的文本生成视频模型仍面临着高昂的训练成本、大量的数据需求以及文本与前景物体运动一致性维护的难题。为解决这些挑战，我们提出了掩码引导的视频生成方法，该方法通过掩码运动序列控制视频生成，同时仅需要有限的训练数据。我们的模型通过引入前景掩码增强现有架构，实现精确的文字位置匹配和运动轨迹控制。通过掩码运动序列，我们引导视频生成过程，确保序列中前景物体的一致性。此外，通过首帧共享策略和自回归扩展方法，我们实现了更为稳定和长久的视频生成。广泛的定性和定量实验表明，该方法在视频编辑和生成艺术视频等多种视频生成任务中表现出色，一致性和质量均优于先前方法。生成的结果详见补充材料。 

---
# Voxel-based Point Cloud Geometry Compression with Space-to-Channel Context 

**Title (ZH)**: 基于体素的点云几何压缩：空间到通道上下文方法 

**Authors**: Bojun Liu, Yangzhi Ma, Ao Luo, Li Li, Dong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18283)  

**Abstract**: Voxel-based methods are among the most efficient for point cloud geometry compression, particularly with dense point clouds. However, they face limitations due to a restricted receptive field, especially when handling high-bit depth point clouds. To overcome this issue, we introduce a stage-wise Space-to-Channel (S2C) context model for both dense point clouds and low-level sparse point clouds. This model utilizes a channel-wise autoregressive strategy to effectively integrate neighborhood information at a coarse resolution. For high-level sparse point clouds, we further propose a level-wise S2C context model that addresses resolution limitations by incorporating Geometry Residual Coding (GRC) for consistent-resolution cross-level prediction. Additionally, we use the spherical coordinate system for its compact representation and enhance our GRC approach with a Residual Probability Approximation (RPA) module, which features a large kernel size. Experimental results show that our S2C context model not only achieves bit savings while maintaining or improving reconstruction quality but also reduces computational complexity compared to state-of-the-art voxel-based compression methods. 

**Abstract (ZH)**: 基于体素的方法是点云几何压缩中最高效的手段，特别是对于稠密点云。然而，这些方法由于 receptive field 受限，在处理高 bit 深度点云时面临限制。为克服这一问题，我们引入了一种分阶段的 Space-to-Channel (S2C) 上下文模型，适用于稠密点云和低层级稀疏点云。该模型利用通道间自回归策略，在粗分辨率下有效整合邻域信息。对于高层稀疏点云，我们进一步提出了一种分层级的 S2C 上下文模型，通过引入 Geometry Residual Coding (GRC) 进行一致分辨率跨层级预测，来解决分辨率限制问题。此外，我们使用球坐标系统以实现紧凑表示，并通过 Residual Probability Approximation (RPA) 模块增强 GRC 方法，该模块具备大核大小。实验结果表明，我们的 S2C 上下文模型不仅能节省位宽同时保持或提升重建质量，还能降低与现有最佳体素基压缩方法相比的计算复杂度。 

---
# PG-SAM: Prior-Guided SAM with Medical for Multi-organ Segmentation 

**Title (ZH)**: PG-SAM: 以先验知识引导的SAM在多器官分割中的应用 

**Authors**: Yiheng Zhong, Zihong Luo, Chengzhi Liu, Feilong Tang, Zelin Peng, Ming Hu, Yingzhen Hu, Jionglong Su, Zongyuan Geand, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.18227)  

**Abstract**: Segment Anything Model (SAM) demonstrates powerful zero-shot capabilities; however, its accuracy and robustness significantly decrease when applied to medical image segmentation. Existing methods address this issue through modality fusion, integrating textual and image information to provide more detailed priors. In this study, we argue that the granularity of text and the domain gap affect the accuracy of the priors. Furthermore, the discrepancy between high-level abstract semantics and pixel-level boundary details in images can introduce noise into the fusion process. To address this, we propose Prior-Guided SAM (PG-SAM), which employs a fine-grained modality prior aligner to leverage specialized medical knowledge for better modality alignment. The core of our method lies in efficiently addressing the domain gap with fine-grained text from a medical LLM. Meanwhile, it also enhances the priors' quality after modality alignment, ensuring more accurate segmentation. In addition, our decoder enhances the model's expressive capabilities through multi-level feature fusion and iterative mask optimizer operations, supporting unprompted learning. We also propose a unified pipeline that effectively supplies high-quality semantic information to SAM. Extensive experiments on the Synapse dataset demonstrate that the proposed PG-SAM achieves state-of-the-art performance. Our anonymous code is released at this https URL. 

**Abstract (ZH)**: Segment Anything Model (SAM)在医疗图像分割中的细粒度先验引导方法：Prior-Guided SAM (PG-SAM)的研究 

---
# Self-Attention Diffusion Models for Zero-Shot Biomedical Image Segmentation: Unlocking New Frontiers in Medical Imaging 

**Title (ZH)**: 自注意力扩散模型在零样本生物医学图像分割中的应用：开启医学成像的新前沿 

**Authors**: Abderrachid Hamrani, Anuradha Godavarty  

**Link**: [PDF](https://arxiv.org/pdf/2503.18170)  

**Abstract**: Producing high-quality segmentation masks for medical images is a fundamental challenge in biomedical image analysis. Recent research has explored large-scale supervised training to enable segmentation across various medical imaging modalities and unsupervised training to facilitate segmentation without dense annotations. However, constructing a model capable of segmenting diverse medical images in a zero-shot manner without any annotations remains a significant hurdle. This paper introduces the Attention Diffusion Zero-shot Unsupervised System (ADZUS), a novel approach that leverages self-attention diffusion models for zero-shot biomedical image segmentation. ADZUS harnesses the intrinsic capabilities of pre-trained diffusion models, utilizing their generative and discriminative potentials to segment medical images without requiring annotated training data or prior domain-specific knowledge. The ADZUS architecture is detailed, with its integration of self-attention mechanisms that facilitate context-aware and detail-sensitive segmentations being highlighted. Experimental results across various medical imaging datasets, including skin lesion segmentation, chest X-ray infection segmentation, and white blood cell segmentation, reveal that ADZUS achieves state-of-the-art performance. Notably, ADZUS reached Dice scores ranging from 88.7\% to 92.9\% and IoU scores from 66.3\% to 93.3\% across different segmentation tasks, demonstrating significant improvements in handling novel, unseen medical imagery. It is noteworthy that while ADZUS demonstrates high effectiveness, it demands substantial computational resources and extended processing times. The model's efficacy in zero-shot settings underscores its potential to reduce reliance on costly annotations and seamlessly adapt to new medical imaging tasks, thereby expanding the diagnostic capabilities of AI-driven medical imaging technologies. 

**Abstract (ZH)**: 基于注意力扩散模型的零样本无监督医疗图像分割系统（ADZUS） 

---
# DiffusionTalker: Efficient and Compact Speech-Driven 3D Talking Head via Personalizer-Guided Distillation 

**Title (ZH)**: DiffusionTalker: 高效且紧凑的个性化蒸馏驱动3D说话头模型 

**Authors**: Peng Chen, Xiaobao Wei, Ming Lu, Hui Chen, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.18159)  

**Abstract**: Real-time speech-driven 3D facial animation has been attractive in academia and industry. Traditional methods mainly focus on learning a deterministic mapping from speech to animation. Recent approaches start to consider the nondeterministic fact of speech-driven 3D face animation and employ the diffusion model for the task. Existing diffusion-based methods can improve the diversity of facial animation. However, personalized speaking styles conveying accurate lip language is still lacking, besides, efficiency and compactness still need to be improved. In this work, we propose DiffusionTalker to address the above limitations via personalizer-guided distillation. In terms of personalization, we introduce a contrastive personalizer that learns identity and emotion embeddings to capture speaking styles from audio. We further propose a personalizer enhancer during distillation to enhance the influence of embeddings on facial animation. For efficiency, we use iterative distillation to reduce the steps required for animation generation and achieve more than 8x speedup in inference. To achieve compactness, we distill the large teacher model into a smaller student model, reducing our model's storage by 86.4\% while minimizing performance loss. After distillation, users can derive their identity and emotion embeddings from audio to quickly create personalized animations that reflect specific speaking styles. Extensive experiments are conducted to demonstrate that our method outperforms state-of-the-art methods. The code will be released at: this https URL. 

**Abstract (ZH)**: 基于个性化引导蒸馏的实时语音驱动3D面部动画 

---
# Efficient Deep Learning Approaches for Processing Ultra-Widefield Retinal Imaging 

**Title (ZH)**: 高效的深度学习方法用于处理超广场视网膜成像 

**Authors**: Siwon Kim, Wooyung Yun, Jeongbin Oh, Soomok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.18151)  

**Abstract**: Deep learning has emerged as the predominant solution for classifying medical images. We intend to apply these developments to the ultra-widefield (UWF) retinal imaging dataset. Since UWF images can accurately diagnose various retina diseases, it is very important to clas sify them accurately and prevent them with early treatment. However, processing images manually is time-consuming and labor-intensive, and there are two challenges to automating this process. First, high perfor mance usually requires high computational resources. Artificial intelli gence medical technology is better suited for places with limited medical resources, but using high-performance processing units in such environ ments is challenging. Second, the problem of the accuracy of colour fun dus photography (CFP) methods. In general, the UWF method provides more information for retinal diagnosis than the CFP method, but most of the research has been conducted based on the CFP method. Thus, we demonstrate that these problems can be efficiently addressed in low performance units using methods such as strategic data augmentation and model ensembles, which balance performance and computational re sources while utilizing UWF images. 

**Abstract (ZH)**: 深度学习已成为医学图像分类的主要解决方案。我们打算将这些进展应用到超广field (UWF) 视网膜成像数据集中。由于UWF图像可以准确诊断各种视网膜疾病，因此准确分类这些图像并进行早期治疗至关重要。然而，手动处理图像耗时且劳动密集，自动化此过程面临两个挑战。首先，高性能通常需要高计算资源。人工智能医疗技术更适合资源有限的医疗机构，但在这些环境中使用高性能处理单元具有挑战性。其次，色fundus摄影（CFP）方法准确性的问题。总体而言，UWF方法为视网膜诊断提供了更多的信息，但大多数研究都是基于CFP方法进行的。因此，我们展示了一种使用战略数据增强和模型集成等方法，在低性能单元中高效解决这些问题，同时平衡性能和计算资源并利用UWF图像。 

---
# Co-SemDepth: Fast Joint Semantic Segmentation and Depth Estimation on Aerial Images 

**Title (ZH)**: 共语义深度：快速联合空域图像语义分割与深度估计 

**Authors**: Yara AlaaEldin, Francesca Odone  

**Link**: [PDF](https://arxiv.org/pdf/2503.17982)  

**Abstract**: Understanding the geometric and semantic properties of the scene is crucial in autonomous navigation and particularly challenging in the case of Unmanned Aerial Vehicle (UAV) navigation. Such information may be by obtained by estimating depth and semantic segmentation maps of the surrounding environment and for their practical use in autonomous navigation, the procedure must be performed as close to real-time as possible. In this paper, we leverage monocular cameras on aerial robots to predict depth and semantic maps in low-altitude unstructured environments. We propose a joint deep-learning architecture that can perform the two tasks accurately and rapidly, and validate its effectiveness on MidAir and Aeroscapes benchmark datasets. Our joint-architecture proves to be competitive or superior to the other single and joint architecture methods while performing its task fast predicting 20.2 FPS on a single NVIDIA quadro p5000 GPU and it has a low memory footprint. All codes for training and prediction can be found on this link: this https URL 

**Abstract (ZH)**: 理解场景的几何和语义特性对于自主导航至关重要，特别是在无人飞行器(UAV)导航中更具挑战性。通过估算周围环境的深度和语义分割图，并将其用于自主导航，该过程必须尽可能接近实时进行。本文利用空中机器人上的单目相机在低空未结构化环境中预测深度和语义地图。我们提出了一种联合深度学习架构，可以在准确和快速地执行两项任务的同时得到验证，其在MidAir和Aeroscapes基准数据集上的有效性得到了验证。我们的联合架构在速度方面表现出色，每秒预测帧数达到20.2 FPS，且内存占用低。所有用于训练和预测的代码可以在以下链接找到：this https URL。 

---
# Shot Sequence Ordering for Video Editing: Benchmarks, Metrics, and Cinematology-Inspired Computing Methods 

**Title (ZH)**: 视频编辑中的镜头序列排序：基准、评估指标及	cinematology-启发式计算方法 

**Authors**: Yuzhi Li, Haojun Xu, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.17975)  

**Abstract**: With the rising popularity of short video platforms, the demand for video production has increased substantially. However, high-quality video creation continues to rely heavily on professional editing skills and a nuanced understanding of visual language. To address this challenge, the Shot Sequence Ordering (SSO) task in AI-assisted video editing has emerged as a pivotal approach for enhancing video storytelling and the overall viewing experience. Nevertheless, the progress in this field has been impeded by a lack of publicly available benchmark datasets. In response, this paper introduces two novel benchmark datasets, AVE-Order and ActivityNet-Order. Additionally, we employ the Kendall Tau distance as an evaluation metric for the SSO task and propose the Kendall Tau Distance-Cross Entropy Loss. We further introduce the concept of Cinematology Embedding, which incorporates movie metadata and shot labels as prior knowledge into the SSO model, and constructs the AVE-Meta dataset to validate the method's effectiveness. Experimental results indicate that the proposed loss function and method substantially enhance SSO task accuracy. All datasets are publicly accessible at this https URL. 

**Abstract (ZH)**: 随着短视频平台的兴起，对视频制作的需求大幅增加。然而，高质量视频创作仍然高度依赖专业的编辑技巧和对视觉语言的细微理解。为此，人工智能辅助视频编辑中的镜头序列排序（Shot Sequence Ordering, SSO）任务成为了提升视频叙事能力和整体观感的关键方法。然而，领域进展受限于缺乏公开的基准数据集。针对这一问题，本文提出了两个新的基准数据集，AVE-Order和ActivityNet-Order。此外，我们使用Kendall Tau距离作为SSO任务的评估指标，并提出了Kendall Tau距离-交叉熵损失。为进一步优化，我们引入了电影学嵌入的概念，将电影元数据和镜头标签的先验知识融入到SSO模型中，并构建AVE-Meta数据集以验证方法的有效性。实验结果表明，所提出的损失函数和方法显著提高了SSO任务的准确性。所有数据集均可通过此链接访问。 

---
# Cat-AIR: Content and Task-Aware All-in-One Image Restoration 

**Title (ZH)**: Cat-AIR：内容与任务 Awareness 全方位图像恢复 

**Authors**: Jiachen Jiang, Tianyu Ding, Ke Zhang, Jinxin Zhou, Tianyi Chen, Ilya Zharkov, Zhihui Zhu, Luming Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17915)  

**Abstract**: All-in-one image restoration seeks to recover high-quality images from various types of degradation using a single model, without prior knowledge of the corruption source. However, existing methods often struggle to effectively and efficiently handle multiple degradation types. We present Cat-AIR, a novel \textbf{C}ontent \textbf{A}nd \textbf{T}ask-aware framework for \textbf{A}ll-in-one \textbf{I}mage \textbf{R}estoration. Cat-AIR incorporates an alternating spatial-channel attention mechanism that adaptively balances the local and global information for different tasks. Specifically, we introduce cross-layer channel attentions and cross-feature spatial attentions that allocate computations based on content and task complexity. Furthermore, we propose a smooth learning strategy that allows for seamless adaptation to new restoration tasks while maintaining performance on existing ones. Extensive experiments demonstrate that Cat-AIR achieves state-of-the-art results across a wide range of restoration tasks, requiring fewer FLOPs than previous methods, establishing new benchmarks for efficient all-in-one image restoration. 

**Abstract (ZH)**: 一种内容和任务感知的全目标任务图像恢复框架Cat-AIR 

---
# good4cir: Generating Detailed Synthetic Captions for Composed Image Retrieval 

**Title (ZH)**: good4cir: 生成适合组合图像检索的详细合成caption 

**Authors**: Pranavi Kolouju, Eric Xing, Robert Pless, Nathan Jacobs, Abby Stylianou  

**Link**: [PDF](https://arxiv.org/pdf/2503.17871)  

**Abstract**: Composed image retrieval (CIR) enables users to search images using a reference image combined with textual modifications. Recent advances in vision-language models have improved CIR, but dataset limitations remain a barrier. Existing datasets often rely on simplistic, ambiguous, or insufficient manual annotations, hindering fine-grained retrieval. We introduce good4cir, a structured pipeline leveraging vision-language models to generate high-quality synthetic annotations. Our method involves: (1) extracting fine-grained object descriptions from query images, (2) generating comparable descriptions for target images, and (3) synthesizing textual instructions capturing meaningful transformations between images. This reduces hallucination, enhances modification diversity, and ensures object-level consistency. Applying our method improves existing datasets and enables creating new datasets across diverse domains. Results demonstrate improved retrieval accuracy for CIR models trained on our pipeline-generated datasets. We release our dataset construction framework to support further research in CIR and multi-modal retrieval. 

**Abstract (ZH)**: 基于图像合成的图像检索（Composed Image Retrieval, CIR）使用户能够使用参考图像结合文本修改来搜索图像。近期视觉语言模型的进步提高了CIR的性能，但数据集限制仍然是一个障碍。现有数据集往往依赖于简单、含糊或不足的人工注释，妨碍了细粒度的检索。我们引入了good4cir，这是一种结构化的管道，利用视觉语言模型生成高质量的合成注释。我们的方法包括：(1) 从查询图像中提取细粒度的物体描述，(2) 为目标图像生成可比的描述，(3) 综合文本指令捕获图像之间的有意义变换。这减少了幻觉，增强了修改的多样性，并确保了物体级别的一致性。应用我们的方法可以改善现有数据集，并跨越多个领域创建新的数据集。实验结果表明，基于我们管道生成的数据集训练的CIR模型检索精度有所提高。我们发布了数据集构建框架以支持CIR和多模态检索的进一步研究。 

---
# FundusGAN: A Hierarchical Feature-Aware Generative Framework for High-Fidelity Fundus Image Generation 

**Title (ZH)**: FundusGAN：一种层次特征意识的生成框架，用于高保真眼底图像生成 

**Authors**: Qingshan Hou, Meng Wang, Peng Cao, Zou Ke, Xiaoli Liu, Huazhu Fu, Osmar R. Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2503.17831)  

**Abstract**: Recent advancements in ophthalmology foundation models such as RetFound have demonstrated remarkable diagnostic capabilities but require massive datasets for effective pre-training, creating significant barriers for development and deployment. To address this critical challenge, we propose FundusGAN, a novel hierarchical feature-aware generative framework specifically designed for high-fidelity fundus image synthesis. Our approach leverages a Feature Pyramid Network within its encoder to comprehensively extract multi-scale information, capturing both large anatomical structures and subtle pathological features. The framework incorporates a modified StyleGAN-based generator with dilated convolutions and strategic upsampling adjustments to preserve critical retinal structures while enhancing pathological detail representation. Comprehensive evaluations on the DDR, DRIVE, and IDRiD datasets demonstrate that FundusGAN consistently outperforms state-of-the-art methods across multiple metrics (SSIM: 0.8863, FID: 54.2, KID: 0.0436 on DDR). Furthermore, disease classification experiments reveal that augmenting training data with FundusGAN-generated images significantly improves diagnostic accuracy across multiple CNN architectures (up to 6.49\% improvement with ResNet50). These results establish FundusGAN as a valuable foundation model component that effectively addresses data scarcity challenges in ophthalmological AI research, enabling more robust and generalizable diagnostic systems while reducing dependency on large-scale clinical data collection. 

**Abstract (ZH)**: Recent advancements in眼科领域的基础模型如RetFound已经展示了显著的诊断能力，但需要大量的数据集进行有效的预训练，这为开发和部署带来了巨大障碍。为解决这一关键挑战，我们提出了FundusGAN，一种专门用于高保真视网膜图像合成的新型分层特征感知生成框架。我们的方法在其编码器中采用特征金字塔网络，全面提取多尺度信息，捕捉较大的解剖结构和细微的病理特征。框架结合了改进的基于StyleGAN的生成器，使用了扩张卷积和策略性上采样调整，以保留关键视网膜结构的同时增强病理细节的表示。在DDR、DRIVE和IDRiD数据集上的综合评估表明，FundusGAN在多个指标上（DRR数据集上的SSIM：0.8863，FID：54.2，KID：0.0436）始终优于最先进的方法。此外，疾病分类实验表明，使用FundusGAN生成的图像增强训练数据可以显著提高多种CNN架构的诊断准确性（ResNet50模型的最大提升为6.49%）。这些结果确立了FundusGAN作为眼科AI研究中有效解决数据稀缺问题的基础模型组件的地位，有助于构建更加稳健和通用的诊断系统，同时减少了对大规模临床数据收集的依赖。 

---
# GaussianFocus: Constrained Attention Focus for 3D Gaussian Splatting 

**Title (ZH)**: GaussianFocus: 受约束的关注点优化用于3D 高斯渲染 

**Authors**: Zexu Huang, Min Xu, Stuart Perry  

**Link**: [PDF](https://arxiv.org/pdf/2503.17798)  

**Abstract**: Recent developments in 3D reconstruction and neural rendering have significantly propelled the capabilities of photo-realistic 3D scene rendering across various academic and industrial fields. The 3D Gaussian Splatting technique, alongside its derivatives, integrates the advantages of primitive-based and volumetric representations to deliver top-tier rendering quality and efficiency. Despite these advancements, the method tends to generate excessive redundant noisy Gaussians overfitted to every training view, which degrades the rendering quality. Additionally, while 3D Gaussian Splatting excels in small-scale and object-centric scenes, its application to larger scenes is hindered by constraints such as limited video memory, excessive optimization duration, and variable appearance across views. To address these challenges, we introduce GaussianFocus, an innovative approach that incorporates a patch attention algorithm to refine rendering quality and implements a Gaussian constraints strategy to minimize redundancy. Moreover, we propose a subdivision reconstruction strategy for large-scale scenes, dividing them into smaller, manageable blocks for individual training. Our results indicate that GaussianFocus significantly reduces unnecessary Gaussians and enhances rendering quality, surpassing existing State-of-The-Art (SoTA) methods. Furthermore, we demonstrate the capability of our approach to effectively manage and render large scenes, such as urban environments, whilst maintaining high fidelity in the visual output. 

**Abstract (ZH)**: Recent developments in 3D reconstruction and neural rendering have significantly propelled the capabilities of photo-realistic 3D scene rendering across various academic and industrial fields. 

---
# Progressive Prompt Detailing for Improved Alignment in Text-to-Image Generative Models 

**Title (ZH)**: 渐进式提示细化以改善文本到图像生成模型中的对齐 

**Authors**: Ketan Suhaas Saichandran, Xavier Thomas, Prakhar Kaushik, Deepti Ghadiyaram  

**Link**: [PDF](https://arxiv.org/pdf/2503.17794)  

**Abstract**: Text-to-image generative models often struggle with long prompts detailing complex scenes, diverse objects with distinct visual characteristics and spatial relationships. In this work, we propose SCoPE (Scheduled interpolation of Coarse-to-fine Prompt Embeddings), a training-free method to improve text-to-image alignment by progressively refining the input prompt in a coarse-to-fine-grained manner. Given a detailed input prompt, we first decompose it into multiple sub-prompts which evolve from describing broad scene layout to highly intricate details. During inference, we interpolate between these sub-prompts and thus progressively introduce finer-grained details into the generated image. Our training-free plug-and-play approach significantly enhances prompt alignment, achieves an average improvement of up to +4% in Visual Question Answering (VQA) scores over the Stable Diffusion baselines on 85% of the prompts from the GenAI-Bench dataset. 

**Abstract (ZH)**: 基于文本到图像生成模型在处理长提示描述复杂场景、多种具有独特视觉特征和空间关系的对象时常常表现不佳。本文提出了一种名为SCoPE（逐步插值从粗到细提示嵌入）的无训练方法，通过逐步细化输入提示的方式改进文本到图像的对齐。给定一个详细的输入提示，我们首先将其分解为多个子提示，从描述广泛的场景布局逐渐转向复杂的细节描述。在推理过程中，我们在这些建子提示之间进行插值，从而逐步将更细微的细节引入生成的图像。我们提出的无训练即插即用方法显著提高了提示对齐的效果，在GenAI-Bench数据集中85%的提示上，与 Stable Diffusion 基线相比，平均提高了4%的Visual Question Answering (VQA) 分数。 

---
# DynASyn: Multi-Subject Personalization Enabling Dynamic Action Synthesis 

**Title (ZH)**: DynASyn: 多主题个性化动态动作合成 

**Authors**: Yongjin Choi, Chanhun Park, Seung Jun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2503.17728)  

**Abstract**: Recent advances in text-to-image diffusion models spurred research on personalization, i.e., a customized image synthesis, of subjects within reference images. Although existing personalization methods are able to alter the subjects' positions or to personalize multiple subjects simultaneously, they often struggle to modify the behaviors of subjects or their dynamic interactions. The difficulty is attributable to overfitting to reference images, which worsens if only a single reference image is available. We propose DynASyn, an effective multi-subject personalization from a single reference image addressing these challenges. DynASyn preserves the subject identity in the personalization process by aligning concept-based priors with subject appearances and actions. This is achieved by regularizing the attention maps between the subject token and images through concept-based priors. In addition, we propose concept-based prompt-and-image augmentation for an enhanced trade-off between identity preservation and action diversity. We adopt an SDE-based editing guided by augmented prompts to generate diverse appearances and actions while maintaining identity consistency in the augmented images. Experiments show that DynASyn is capable of synthesizing highly realistic images of subjects with novel contexts and dynamic interactions with the surroundings, and outperforms baseline methods in both quantitative and qualitative aspects. 

**Abstract (ZH)**: Recent Advances in Text-to-Image Diffusion Models Spur Research on Personalization of Subjects within Reference Images: DynASyn, an Effective Multi-Subject Personalization from a Single Reference Image 

---
# ProtoGS: Efficient and High-Quality Rendering with 3D Gaussian Prototypes 

**Title (ZH)**: ProtoGS: 高效的高质量化三维高斯原型渲染 

**Authors**: Zhengqing Gao, Dongting Hu, Jia-Wang Bian, Huan Fu, Yan Li, Tongliang Liu, Mingming Gong, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17486)  

**Abstract**: 3D Gaussian Splatting (3DGS) has made significant strides in novel view synthesis but is limited by the substantial number of Gaussian primitives required, posing challenges for deployment on lightweight devices. Recent methods address this issue by compressing the storage size of densified Gaussians, yet fail to preserve rendering quality and efficiency. To overcome these limitations, we propose ProtoGS to learn Gaussian prototypes to represent Gaussian primitives, significantly reducing the total Gaussian amount without sacrificing visual quality. Our method directly uses Gaussian prototypes to enable efficient rendering and leverage the resulting reconstruction loss to guide prototype learning. To further optimize memory efficiency during training, we incorporate structure-from-motion (SfM) points as anchor points to group Gaussian primitives. Gaussian prototypes are derived within each group by clustering of K-means, and both the anchor points and the prototypes are optimized jointly. Our experiments on real-world and synthetic datasets prove that we outperform existing methods, achieving a substantial reduction in the number of Gaussians, and enabling high rendering speed while maintaining or even enhancing rendering fidelity. 

**Abstract (ZH)**: 基于原型的3D高斯点云合成（ProtoGS：学习高斯原型以减少高斯 primitives 数量而不牺牲视觉质量） 

---
# Spatiotemporal Learning with Context-aware Video Tubelets for Ultrasound Video Analysis 

**Title (ZH)**: 基于上下文意识视频管段的时空学习超声视频分析 

**Authors**: Gary Y. Li, Li Chen, Bryson Hicks, Nikolai Schnittke, David O. Kessler, Jeffrey Shupp, Maria Parker, Cristiana Baloescu, Christopher Moore, Cynthia Gregory, Kenton Gregory, Balasundar Raju, Jochen Kruecker, Alvin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17475)  

**Abstract**: Computer-aided pathology detection algorithms for video-based imaging modalities must accurately interpret complex spatiotemporal information by integrating findings across multiple frames. Current state-of-the-art methods operate by classifying on video sub-volumes (tubelets), but they often lose global spatial context by focusing only on local regions within detection ROIs. Here we propose a lightweight framework for tubelet-based object detection and video classification that preserves both global spatial context and fine spatiotemporal features. To address the loss of global context, we embed tubelet location, size, and confidence as inputs to the classifier. Additionally, we use ROI-aligned feature maps from a pre-trained detection model, leveraging learned feature representations to increase the receptive field and reduce computational complexity. Our method is efficient, with the spatiotemporal tubelet classifier comprising only 0.4M parameters. We apply our approach to detect and classify lung consolidation and pleural effusion in ultrasound videos. Five-fold cross-validation on 14,804 videos from 828 patients shows our method outperforms previous tubelet-based approaches and is suited for real-time workflows. 

**Abstract (ZH)**: 基于视频成像模态的病理检测算法必须通过整合多帧信息准确解读复杂的空时信息。当前最先进的方法通过分类视频子体积（管节）来操作，但往往会因为仅关注检测ROI内的局部区域而丢失全局空间上下文。我们提出了一种轻量级框架，用于基于管节的对象检测和视频分类，能够同时保留全局空间上下文和精细的空时特征。为了解决全局上下文缺失的问题，我们将管节的位置、大小和置信度作为分类器的输入。此外，我们利用预训练检测模型的ROI对齐特征图，通过利用学习到的特征表示来增加感受野并降低计算复杂度。我们的方法高效，仅包含0.4M参数的空时管节分类器。我们应用该方法来检测和分类超声视频中的肺实变和胸腔积液。在828名患者的14,804段视频上进行五折交叉验证表明，我们的方法优于之前的管节基方法，并适用于实时工作流。 

---
# Enhancing Subsequent Video Retrieval via Vision-Language Models (VLMs) 

**Title (ZH)**: 通过视觉-语言模型增强后续视频检索 

**Authors**: Yicheng Duan, Xi Huang, Duo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17415)  

**Abstract**: The rapid growth of video content demands efficient and precise retrieval systems. While vision-language models (VLMs) excel in representation learning, they often struggle with adaptive, time-sensitive video retrieval. This paper introduces a novel framework that combines vector similarity search with graph-based data structures. By leveraging VLM embeddings for initial retrieval and modeling contextual relationships among video segments, our approach enables adaptive query refinement and improves retrieval accuracy. Experiments demonstrate its precision, scalability, and robustness, offering an effective solution for interactive video retrieval in dynamic environments. 

**Abstract (ZH)**: 视频内容的快速增长对高效精确的检索系统提出了需求。虽然视觉语言模型（VLMs）在表示学习方面表现出色，但在适应性和时间敏感的视频检索方面常常遇到困难。本文提出了一种将向量相似度搜索与图基数据结构相结合的新框架。通过利用VLM嵌入进行初始检索，并建模视频片段之间的上下文关系，我们的方法能够实现适应性查询 refinement 并提高检索精度。实验结果显示其精度、可扩展性和鲁棒性，提供了一种有效的交互式视频检索解决方案，适用于动态环境。 

---
