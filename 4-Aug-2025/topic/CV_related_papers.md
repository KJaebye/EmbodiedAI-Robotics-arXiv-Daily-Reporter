# Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging 

**Title (ZH)**: 全手操掃描：使用交接和高斯点合并的人手双臂机器人创建视觉准确的数字双生对象模型 

**Authors**: Tianshuang Qiu, Zehan Ma, Karim El-Refai, Hiya Shah, Chung Min Kim, Justin Kerr, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.00354)  

**Abstract**: 3D Gaussian Splats (3DGSs) are 3D object models derived from multi-view images. Such "digital twins" are useful for simulations, virtual reality, marketing, robot policy fine-tuning, and part inspection. 3D object scanning usually requires multi-camera arrays, precise laser scanners, or robot wrist-mounted cameras, which have restricted workspaces. We propose Omni-Scan, a pipeline for producing high-quality 3D Gaussian Splat models using a bi-manual robot that grasps an object with one gripper and rotates the object with respect to a stationary camera. The object is then re-grasped by a second gripper to expose surfaces that were occluded by the first gripper. We present the Omni-Scan robot pipeline using DepthAny-thing, Segment Anything, as well as RAFT optical flow models to identify and isolate objects held by a robot gripper while removing the gripper and the background. We then modify the 3DGS training pipeline to support concatenated datasets with gripper occlusion, producing an omni-directional (360 degree view) model of the object. We apply Omni-Scan to part defect inspection, finding that it can identify visual or geometric defects in 12 different industrial and household objects with an average accuracy of 83%. Interactive videos of Omni-Scan 3DGS models can be found at this https URL 

**Abstract (ZH)**: 基于双臂机器人的360度三维高斯点云模型生成方法 

---
# IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation 

**Title (ZH)**: IGL-Nav: 增量三维高斯定位用于图像目标导航 

**Authors**: Wenxuan Guo, Xiuwei Xu, Hang Yin, Ziwei Wang, Jianjiang Feng, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00823)  

**Abstract**: Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view image-goal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose. Project page: this https URL. 

**Abstract (ZH)**: 基于图像的目标三维视觉导航 

---
# Reducing the gap between general purpose data and aerial images in concentrated solar power plants 

**Title (ZH)**: 减少集中式太阳能电站通用数据与航空图像之间的差距 

**Authors**: M.A. Pérez-Cutiño, J. Valverde, J. Capitán, J.M. Díaz-Báñez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00440)  

**Abstract**: In the context of Concentrated Solar Power (CSP) plants, aerial images captured by drones present a unique set of challenges. Unlike urban or natural landscapes commonly found in existing datasets, solar fields contain highly reflective surfaces, and domain-specific elements that are uncommon in traditional computer vision benchmarks. As a result, machine learning models trained on generic datasets struggle to generalize to this setting without extensive retraining and large volumes of annotated data. However, collecting and labeling such data is costly and time-consuming, making it impractical for rapid deployment in industrial applications.
To address this issue, we propose a novel approach: the creation of AerialCSP, a virtual dataset that simulates aerial imagery of CSP plants. By generating synthetic data that closely mimic real-world conditions, our objective is to facilitate pretraining of models before deployment, significantly reducing the need for extensive manual labeling. Our main contributions are threefold: (1) we introduce AerialCSP, a high-quality synthetic dataset for aerial inspection of CSP plants, providing annotated data for object detection and image segmentation; (2) we benchmark multiple models on AerialCSP, establishing a baseline for CSP-related vision tasks; and (3) we demonstrate that pretraining on AerialCSP significantly improves real-world fault detection, particularly for rare and small defects, reducing the need for extensive manual labeling. AerialCSP is made publicly available at this https URL. 

**Abstract (ZH)**: 基于 concentrated solar power (CSP) 系统的无人机航拍图像面临的挑战及解决方案：AerialCSP 数据集的创建与应用 

---
# Controllable Pedestrian Video Editing for Multi-View Driving Scenarios via Motion Sequence 

**Title (ZH)**: 基于运动序列的多视角驾驶场景可控行人视频编辑 

**Authors**: Danzhen Fu, Jiagao Hu, Daiguo Zhou, Fei Wang, Zepeng Wang, Wenhua Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00299)  

**Abstract**: Pedestrian detection models in autonomous driving systems often lack robustness due to insufficient representation of dangerous pedestrian scenarios in training datasets. To address this limitation, we present a novel framework for controllable pedestrian video editing in multi-view driving scenarios by integrating video inpainting and human motion control techniques. Our approach begins by identifying pedestrian regions of interest across multiple camera views, expanding detection bounding boxes with a fixed ratio, and resizing and stitching these regions into a unified canvas while preserving cross-view spatial relationships. A binary mask is then applied to designate the editable area, within which pedestrian editing is guided by pose sequence control conditions. This enables flexible editing functionalities, including pedestrian insertion, replacement, and removal. Extensive experiments demonstrate that our framework achieves high-quality pedestrian editing with strong visual realism, spatiotemporal coherence, and cross-view consistency. These results establish the proposed method as a robust and versatile solution for multi-view pedestrian video generation, with broad potential for applications in data augmentation and scenario simulation in autonomous driving. 

**Abstract (ZH)**: 自动驾驶系统中的行人检测模型往往由于训练数据集中危险行人场景表示不足而缺乏鲁棒性。为解决这一限制，我们提出了一种集成视频插补和人类运动控制技术的多视角驱动场景可控行人视频编辑框架。该方法首先在多个摄像头视图中标识行人区域，按固定比例扩展检测边界框，将这些区域调整尺寸和拼接成统一画布，同时保持多视角空间关系。随后应用二进制掩码指定可编辑区域，在该区域内通过姿态序列控制条件引导行人编辑，以实现灵活的编辑功能，包括行人插入、替换和移除。实验结果表明，该框架能够实现高质量、视觉真实感强、时空一致性好以及多视角一致性的行人编辑。这些结果确立了该方法作为多视角行人视频生成的 robust 和 versatile 解决方案的地位，并在自主驾驶中的数据增强和场景模拟方面具有广泛的应用潜力。 

---
# Multi-Agent Game Generation and Evaluation via Audio-Visual Recordings 

**Title (ZH)**: 基于音频-视觉记录的多agent游戏生成与评估 

**Authors**: Alexia Jolicoeur-Martineau  

**Link**: [PDF](https://arxiv.org/pdf/2508.00632)  

**Abstract**: While AI excels at generating text, audio, images, and videos, creating interactive audio-visual content such as video games remains challenging. Current LLMs can generate JavaScript games and animations, but lack automated evaluation metrics and struggle with complex content that normally requires teams of humans working for many months (multi-shot, multi-agents) using assets made by artists. To tackle these issues, we built a new metric and a multi-agent system.
We propose AVR-Eval, a relative metric for multimedia content quality using Audio-Visual Recordings (AVRs). An omni-modal model (processing text, video, and audio) compares the AVRs of two contents, with a text model reviewing evaluations to determine superiority. We show that AVR-Eval properly identifies good from broken or mismatched content.
We built AVR-Agent, a multi-agent system generating JavaScript code from a bank of multimedia assets (audio, images, 3D models). The coding agent selects relevant assets, generates multiple initial codes, uses AVR-Eval to identify the best version, and iteratively improves it through omni-modal agent feedback from the AVR.
We run experiments on games and animations with AVR-Eval (win rate of content A against B). We find that content generated by AVR-Agent has a significantly higher win rate against content made through one-shot generation. However, models struggle to leverage custom assets and AVR feedback effectively, showing no higher win rate. This reveals a critical gap: while humans benefit from high-quality assets and audio-visual feedback, current coding models do not seem to utilize these resources as effectively, highlighting fundamental differences between human and machine content creation approaches. 

**Abstract (ZH)**: 尽管AI在生成文本、音频、图像和视频方面表现出色，但创作交互式音频视觉内容（如视频游戏）仍具有挑战性。当前的大语言模型可以生成JavaScript游戏和动画，但缺乏自动评估指标，并且难以处理通常需要多人团队花费数月时间创作的复杂内容（多回合、多智能体），并且使用的是艺术家制作的资源。为解决这些问题，我们构建了一个新的评估指标和多智能体系统。

我们提出了AVR-Eval，这是一种基于音频-视觉记录（AVRs）的多媒体内容质量相对指标。一种跨模态模型（处理文本、视频和音频）比较两个内容的AVRs，并通过文本模型评审以确定其优势。实验表明AVR-Eval能够正确识别优质内容与受损或匹配不当的内容。

我们构建了AVR-Agent，这是一种多智能体系统，能够从多媒体资产库（音频、图像、3D模型）生成JavaScript代码。编码智能体选择相关资产，生成多个初始代码，使用AVR-Eval识别最佳版本，并通过跨模态智能体反馈从AVR进行迭代优化。

我们在游戏中进行了AVR-Eval实验（内容A对B的胜率）。结果显示，由AVR-Agent生成的内容胜率显著高于通过一次性生成的内容。然而，模型在利用自定义资产和AVR反馈方面表现出色受限，未能实现更高的胜率。这揭示了一个关键差距：尽管人类可以从高质量资产和音频视觉反馈中受益，当前的编码模型似乎未能有效利用这些资源，突显了人类和机器内容创作方法之间的重要差异。 

---
# SpA2V: Harnessing Spatial Auditory Cues for Audio-driven Spatially-aware Video Generation 

**Title (ZH)**: SpA2V：利用空间听觉线索的音频驱动空间aware视频生成 

**Authors**: Kien T. Pham, Yingqing He, Yazhou Xing, Qifeng Chen, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00782)  

**Abstract**: Audio-driven video generation aims to synthesize realistic videos that align with input audio recordings, akin to the human ability to visualize scenes from auditory input. However, existing approaches predominantly focus on exploring semantic information, such as the classes of sounding sources present in the audio, limiting their ability to generate videos with accurate content and spatial composition. In contrast, we humans can not only naturally identify the semantic categories of sounding sources but also determine their deeply encoded spatial attributes, including locations and movement directions. This useful information can be elucidated by considering specific spatial indicators derived from the inherent physical properties of sound, such as loudness or frequency. As prior methods largely ignore this factor, we present SpA2V, the first framework explicitly exploits these spatial auditory cues from audios to generate videos with high semantic and spatial correspondence. SpA2V decomposes the generation process into two stages: 1) Audio-guided Video Planning: We meticulously adapt a state-of-the-art MLLM for a novel task of harnessing spatial and semantic cues from input audio to construct Video Scene Layouts (VSLs). This serves as an intermediate representation to bridge the gap between the audio and video modalities. 2) Layout-grounded Video Generation: We develop an efficient and effective approach to seamlessly integrate VSLs as conditional guidance into pre-trained diffusion models, enabling VSL-grounded video generation in a training-free manner. Extensive experiments demonstrate that SpA2V excels in generating realistic videos with semantic and spatial alignment to the input audios. 

**Abstract (ZH)**: 基于音频驱动的视频生成旨在合成与输入音频记录相匹配的现实主义视频，类似于人类仅通过听觉输入就能想象场景的能力。然而，现有方法主要集中在探索语义信息，如音频中存在的声源类别，限制了其生成准确内容和空间构图的视频的能力。相比之下，人类不仅能自然地识别声源的语义类别，还能确定其深层次编码的空间属性，包括位置和运动方向。这些有用信息可以通过考虑从声音固有的物理属性中派生的特定空间指标来阐明，例如响度或频率。由于先验方法大多忽略了这一因素，我们提出SpA2V，这是第一个明确利用音频中的空间听觉线索来生成具有高语义和空间对应关系的视频的框架。SpA2V将生成过程分解为两个阶段：1) 音频引导的视频规划：我们精细地调整了一种最先进的MLLM，用于一项新的任务，即利用输入音频中的空间和语义线索构建视频场景布局（VSLs）。这作为中介表示，为音频和视频模态之间的鸿沟架起桥梁。2) 布局指导的视频生成：我们开发了一种高效且有效的方法，将VSLs无缝集成到预训练的扩散模型中作为条件指导，从而以无训练的方式实现VSL指导的视频生成。广泛的实验表明，SpA2V在生成与输入音频在语义和空间上对齐的现实主义视频方面表现出色。 

---
# Sample-Aware Test-Time Adaptation for Medical Image-to-Image Translation 

**Title (ZH)**: 基于样本的测试时自适应医学图像到图像翻译 

**Authors**: Irene Iele, Francesco Di Feola, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2508.00766)  

**Abstract**: Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods. Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both out-of-distribution and in-distribution samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios. The code is available at: this https URL. 

**Abstract (ZH)**: 图像到图像的翻译技术在医学影像中 emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion. However, it suffers from limitations in handling out-of-distribution samples without causing performance degradation. To address this limitation, we propose a novel Test-Time Adaptation (TTA)框架 that dynamically adjusts the translation process based on the characteristics of each test sample. Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on in-distribution samples that do not require adaptation. We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and T1 to T2 MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods. Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both out-of-distribution and in-distribution samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios. The code is available at: this https URL. 

---
# Is It Really You? Exploring Biometric Verification Scenarios in Photorealistic Talking-Head Avatar Videos 

**Title (ZH)**: 真的是你？探索 photorealistic 沟通头部 avatar 视频中的生物特征验证场景 

**Authors**: Laura Pedrouzo-Rodriguez, Pedro Delgado-DeRobles, Luis F. Gomez, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00748)  

**Abstract**: Photorealistic talking-head avatars are becoming increasingly common in virtual meetings, gaming, and social platforms. These avatars allow for more immersive communication, but they also introduce serious security risks. One emerging threat is impersonation: an attacker can steal a user's avatar-preserving their appearance and voice-making it nearly impossible to detect its fraudulent usage by sight or sound alone. In this paper, we explore the challenge of biometric verification in such avatar-mediated scenarios. Our main question is whether an individual's facial motion patterns can serve as reliable behavioral biometrics to verify their identity when the avatar's visual appearance is a facsimile of its owner. To answer this question, we introduce a new dataset of realistic avatar videos created using a state-of-the-art one-shot avatar generation model, GAGAvatar, with genuine and impostor avatar videos. We also propose a lightweight, explainable spatio-temporal Graph Convolutional Network architecture with temporal attention pooling, that uses only facial landmarks to model dynamic facial gestures. Experimental results demonstrate that facial motion cues enable meaningful identity verification with AUC values approaching 80%. The proposed benchmark and biometric system are available for the research community in order to bring attention to the urgent need for more advanced behavioral biometric defenses in avatar-based communication systems. 

**Abstract (ZH)**: 逼真 talking-head  avatar 在虚拟会议、游戏和社会平台中的应用日益增多，这些 avatar 增强了沉浸式通信，但也引入了严重安全风险。一种新兴威胁是冒充：攻击者可以盗用用户的 avatar —— 保留其外观和声音——使得通过视觉或听觉单独检测其欺诈使用几乎不可能。本文探讨了这种 avatar 介导场景中的生物特征验证挑战。我们的主要问题是，当 avatar 的视觉外观与其所有者相似时，个体的面部运动模式是否能够作为可靠的生物行为特征，用于验证其身份。为了解答这个问题，我们引入了一个使用最先进的单次生成模型 GAGAvatar 创建的现实 avatar 视频新数据集，其中包括真实和冒充 avatar 视频。我们还提出了一种轻量级、可解释的时间空间图卷积网络架构，该架构使用时间注意力池化，仅基于面部特征点来建模动态面部表情。实验结果表明，面部运动线索能够实现有意义的身份验证，AUC 值接近 80%。提出的基准和生物特征系统可供研究界使用，旨在引起对基于 avatar 的通信系统中更先进生物行为特征防御的迫切需求的关注。 

---
# D3: Training-Free AI-Generated Video Detection Using Second-Order Features 

**Title (ZH)**: D3: 基于二阶特征的无需训练的AI生成视频检测 

**Authors**: Chende Zheng, Ruiqi suo, Chenhao Lin, Zhengyu Zhao, Le Yang, Shuai Liu, Minghui Yang, Cong Wang, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00701)  

**Abstract**: The evolution of video generation techniques, such as Sora, has made it increasingly easy to produce high-fidelity AI-generated videos, raising public concern over the dissemination of synthetic content. However, existing detection methodologies remain limited by their insufficient exploration of temporal artifacts in synthetic videos. To bridge this gap, we establish a theoretical framework through second-order dynamical analysis under Newtonian mechanics, subsequently extending the Second-order Central Difference features tailored for temporal artifact detection. Building on this theoretical foundation, we reveal a fundamental divergence in second-order feature distributions between real and AI-generated videos. Concretely, we propose Detection by Difference of Differences (D3), a novel training-free detection method that leverages the above second-order temporal discrepancies. We validate the superiority of our D3 on 4 open-source datasets (Gen-Video, VideoPhy, EvalCrafter, VidProM), 40 subsets in total. For example, on GenVideo, D3 outperforms the previous best method by 10.39% (absolute) mean Average Precision. Additional experiments on time cost and post-processing operations demonstrate D3's exceptional computational efficiency and strong robust performance. Our code is available at this https URL. 

**Abstract (ZH)**: 视频生成技术（如Sora）的进化使得高保真AI生成视频的生产变得日益容易，引发了公众对合成内容传播的关注。然而，现有的检测方法仍然受限于它们对合成视频中时间伪迹的不足探索。为解决这一问题，我们通过牛顿力学下的二次动力学分析建立了一个理论框架，随后扩展了适用于时间伪迹检测的二次中心差分特征。基于这一理论基础，我们揭示了实际视频和AI生成视频在二次特征分布上的基本差异。具体而言，我们提出了差的差检测（D3），这是一种无需训练的新型检测方法，利用上述二次时间差异。我们在4个开源数据集（Gen-Video、VideoPhy、EvalCrafter、VidProM）的总计40个子集上验证了D3的优势，在GenVideo数据集上，D3绝对均值平均精度优于之前的最佳方法10.39%。额外的实验展示了D3的出色计算效率和强大的稳健性能。我们的代码可在以下链接获取。 

---
# TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation 

**Title (ZH)**: TopoTTA：拓扑增强测试时自适应技术用于管状结构分割 

**Authors**: Jiale Zhou, Wenhan Wang, Shikun Li, Xiaolei Qu, Xin Guo, Yizhong Liu, Wenzhong Tang, Xun Lin, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00442)  

**Abstract**: Tubular structure segmentation (TSS) is important for various applications, such as hemodynamic analysis and route navigation. Despite significant progress in TSS, domain shifts remain a major challenge, leading to performance degradation in unseen target domains. Unlike other segmentation tasks, TSS is more sensitive to domain shifts, as changes in topological structures can compromise segmentation integrity, and variations in local features distinguishing foreground from background (e.g., texture and contrast) may further disrupt topological continuity. To address these challenges, we propose Topology-enhanced Test-Time Adaptation (TopoTTA), the first test-time adaptation framework designed specifically for TSS. TopoTTA consists of two stages: Stage 1 adapts models to cross-domain topological discrepancies using the proposed Topological Meta Difference Convolutions (TopoMDCs), which enhance topological representation without altering pre-trained parameters; Stage 2 improves topological continuity by a novel Topology Hard sample Generation (TopoHG) strategy and prediction alignment on hard samples with pseudo-labels in the generated pseudo-break regions. Extensive experiments across four scenarios and ten datasets demonstrate TopoTTA's effectiveness in handling topological distribution shifts, achieving an average improvement of 31.81% in clDice. TopoTTA also serves as a plug-and-play TTA solution for CNN-based TSS models. 

**Abstract (ZH)**: Tubular Structure Segmentation (TSS) 的拓扑增强测试时自适应（TopoTTA） 

---
# Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting 

**Title (ZH)**: 接触意识的非可视区域完成：基于多区域修复的人与物体交互 

**Authors**: Seunggeun Chi, Enna Sachdeva, Pin-Hao Huang, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.00427)  

**Abstract**: Amodal completion, which is the process of inferring the full appearance of objects despite partial occlusions, is crucial for understanding complex human-object interactions (HOI) in computer vision and robotics. Existing methods, such as those that use pre-trained diffusion models, often struggle to generate plausible completions in dynamic scenarios because they have a limited understanding of HOI. To solve this problem, we've developed a new approach that uses physical prior knowledge along with a specialized multi-regional inpainting technique designed for HOI. By incorporating physical constraints from human topology and contact information, we define two distinct regions: the primary region, where occluded object parts are most likely to be, and the secondary region, where occlusions are less probable. Our multi-regional inpainting method uses customized denoising strategies across these regions within a diffusion model. This improves the accuracy and realism of the generated completions in both their shape and visual detail. Our experimental results show that our approach significantly outperforms existing methods in HOI scenarios, moving machine perception closer to a more human-like understanding of dynamic environments. We also show that our pipeline is robust even without ground-truth contact annotations, which broadens its applicability to tasks like 3D reconstruction and novel view/pose synthesis. 

**Abstract (ZH)**: 无感知完成：基于物理先验的动态场景中复杂人-物交互理解 

---
# $MV_{Hybrid}$: Improving Spatial Transcriptomics Prediction with Hybrid State Space-Vision Transformer Backbone in Pathology Vision Foundation Models 

**Title (ZH)**: $MV_{Hybrid}$: 使用混合状态空间-视觉变换器骨干改进病理科视觉基础模型中的空间转录组学预测 

**Authors**: Won June Cho, Hongjun Yoon, Daeky Jeong, Hyeongyeol Lim, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00383)  

**Abstract**: Spatial transcriptomics reveals gene expression patterns within tissue context, enabling precision oncology applications such as treatment response prediction, but its high cost and technical complexity limit clinical adoption. Predicting spatial gene expression (biomarkers) from routine histopathology images offers a practical alternative, yet current vision foundation models (VFMs) in pathology based on Vision Transformer (ViT) backbones perform below clinical standards. Given that VFMs are already trained on millions of diverse whole slide images, we hypothesize that architectural innovations beyond ViTs may better capture the low-frequency, subtle morphological patterns correlating with molecular phenotypes. By demonstrating that state space models initialized with negative real eigenvalues exhibit strong low-frequency bias, we introduce $MV_{Hybrid}$, a hybrid backbone architecture combining state space models (SSMs) with ViT. We compare five other different backbone architectures for pathology VFMs, all pretrained on identical colorectal cancer datasets using the DINOv2 self-supervised learning method. We evaluate all pretrained models using both random split and leave-one-study-out (LOSO) settings of the same biomarker dataset. In LOSO evaluation, $MV_{Hybrid}$ achieves 57% higher correlation than the best-performing ViT and shows 43% smaller performance degradation compared to random split in gene expression prediction, demonstrating superior performance and robustness, respectively. Furthermore, $MV_{Hybrid}$ shows equal or better downstream performance in classification, patch retrieval, and survival prediction tasks compared to that of ViT, showing its promise as a next-generation pathology VFM backbone. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 基于状态空间模型的空间转录组学在病理视觉基础模型中的应用：混合架构MV_{Hybrid}的性能评估 

---
# GV-VAD : Exploring Video Generation for Weakly-Supervised Video Anomaly Detection 

**Title (ZH)**: GV-VAD : 探索视频生成在弱监督视频异常检测中的应用 

**Authors**: Suhang Cai, Xiaohao Peng, Chong Wang, Xiaojie Cai, Jiangbo Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.00312)  

**Abstract**: Video anomaly detection (VAD) plays a critical role in public safety applications such as intelligent surveillance. However, the rarity, unpredictability, and high annotation cost of real-world anomalies make it difficult to scale VAD datasets, which limits the performance and generalization ability of existing models. To address this challenge, we propose a generative video-enhanced weakly-supervised video anomaly detection (GV-VAD) framework that leverages text-conditioned video generation models to produce semantically controllable and physically plausible synthetic videos. These virtual videos are used to augment training data at low cost. In addition, a synthetic sample loss scaling strategy is utilized to control the influence of generated synthetic samples for efficient training. The experiments show that the proposed framework outperforms state-of-the-art methods on UCF-Crime datasets. The code is available at this https URL. 

**Abstract (ZH)**: 基于生成视频增强弱监督视频异常检测的框架（GV-VAD） 

---
# Beamformed 360° Sound Maps: U-Net-Driven Acoustic Source Segmentation and Localization 

**Title (ZH)**: 360°声图谱的波束形成：基于U-Net的声音源分割与定位 

**Authors**: Belman Jahir Rodriguez, Sergio F. Chevtchenko, Marcelo Herrera Martinez, Yeshwant Bethy, Saeed Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00307)  

**Abstract**: We introduce a U-net model for 360° acoustic source localization formulated as a spherical semantic segmentation task. Rather than regressing discrete direction-of-arrival (DoA) angles, our model segments beamformed audio maps (azimuth and elevation) into regions of active sound presence. Using delay-and-sum (DAS) beamforming on a custom 24-microphone array, we generate signals aligned with drone GPS telemetry to create binary supervision masks. A modified U-Net, trained on frequency-domain representations of these maps, learns to identify spatially distributed source regions while addressing class imbalance via the Tversky loss. Because the network operates on beamformed energy maps, the approach is inherently array-independent and can adapt to different microphone configurations without retraining from scratch. The segmentation outputs are post-processed by computing centroids over activated regions, enabling robust DoA estimates. Our dataset includes real-world open-field recordings of a DJI Air 3 drone, synchronized with 360° video and flight logs across multiple dates and locations. Experimental results show that U-net generalizes across environments, providing improved angular precision, offering a new paradigm for dense spatial audio understanding beyond traditional Sound Source Localization (SSL). 

**Abstract (ZH)**: 一种用于360°声源定位的U-net模型：作为一种球面语义分割任务的 formulations 

---
# Jet Image Generation in High Energy Physics Using Diffusion Models 

**Title (ZH)**: 高能物理中基于扩散模型的喷流图像生成 

**Authors**: Victor D. Martinez, Vidya Manian, Sudhir Malik  

**Link**: [PDF](https://arxiv.org/pdf/2508.00250)  

**Abstract**: This article presents, for the first time, the application of diffusion models for generating jet images corresponding to proton-proton collision events at the Large Hadron Collider (LHC). The kinematic variables of quark, gluon, W-boson, Z-boson, and top quark jets from the JetNet simulation dataset are mapped to two-dimensional image representations. Diffusion models are trained on these images to learn the spatial distribution of jet constituents. We compare the performance of score-based diffusion models and consistency models in accurately generating class-conditional jet images. Unlike approaches based on latent distributions, our method operates directly in image space. The fidelity of the generated images is evaluated using several metrics, including the Fréchet Inception Distance (FID), which demonstrates that consistency models achieve higher fidelity and generation stability compared to score-based diffusion models. These advancements offer significant improvements in computational efficiency and generation accuracy, providing valuable tools for High Energy Physics (HEP) research. 

**Abstract (ZH)**: 本文首次介绍了扩散模型在生成大型强子对撞机（LHC）质子-质子碰撞事件对应的喷流图像方面的应用。从JetNet模拟数据集中，夸克、胶子、W介子、Z介子和顶夸克喷流的动力学变量被映射到二维图像表示。通过这些图像，训练扩散模型学习喷流组成部分的空间分布。我们比较了基于评分的扩散模型和一致性模型在准确生成类条件喷流图像方面的性能。不同于基于潜在分布的方法，我们的方法直接在图像空间中操作。使用多种度量标准，包括弗雷谢特-incception距离（FID），表明一致性模型在图像保真度和生成稳定性方面优于基于评分的扩散模型。这些进步为高能物理（HEP）研究提供了在计算效率和生成准确度方面的显著改善，提供了宝贵的工具。 

---
# GEPAR3D: Geometry Prior-Assisted Learning for 3D Tooth Segmentation 

**Title (ZH)**: GEPAR3D: 几何先验辅助的3D牙齿分割学习 

**Authors**: Tomasz Szczepański, Szymon Płotka, Michal K. Grzeszczyk, Arleta Adamowicz, Piotr Fudalej, Przemysław Korzeniowski, Tomasz Trzciński, Arkadiusz Sitek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00155)  

**Abstract**: Tooth segmentation in Cone-Beam Computed Tomography (CBCT) remains challenging, especially for fine structures like root apices, which is critical for assessing root resorption in orthodontics. We introduce GEPAR3D, a novel approach that unifies instance detection and multi-class segmentation into a single step tailored to improve root segmentation. Our method integrates a Statistical Shape Model of dentition as a geometric prior, capturing anatomical context and morphological consistency without enforcing restrictive adjacency constraints. We leverage a deep watershed method, modeling each tooth as a continuous 3D energy basin encoding voxel distances to boundaries. This instance-aware representation ensures accurate segmentation of narrow, complex root apices. Trained on publicly available CBCT scans from a single center, our method is evaluated on external test sets from two in-house and two public medical centers. GEPAR3D achieves the highest overall segmentation performance, averaging a Dice Similarity Coefficient (DSC) of 95.0% (+2.8% over the second-best method) and increasing recall to 95.2% (+9.5%) across all test sets. Qualitative analyses demonstrated substantial improvements in root segmentation quality, indicating significant potential for more accurate root resorption assessment and enhanced clinical decision-making in orthodontics. We provide the implementation and dataset at this https URL. 

**Abstract (ZH)**: 三维锥形束计算机断层扫描中牙齿分割仍然具有挑战性，尤其是在对于诸如根尖这样的精细结构的分割中，这对于评估正畸中的根吸收至关重要。我们提出了一种名为GEPAR3D的新方法，该方法将实例检测和多类分割统一到单一步骤中，以改进根的分割。该方法结合了牙齿排列的统计形状模型作为几何先验，捕捉解剖学上下文和形态学一致性，而不施加限制性 adjacency 约束。我们利用一种深度分水岭方法，将每颗牙齿建模为包含体素到边界距离的连续3D能量盆地。这种实例感知的表示确保了对狭窄且复杂的根尖分割的准确度。该方法在单一中心公开提供的CBCT扫描数据上进行训练，并在外购的两家和两家公共医学中心的测试集上进行评估。GEPAR3D取得了最高的整体分割性能，平均Dice相似系数（DSC）为95.0%（超过第二佳方法2.8%），并在所有测试集上的召回率提高了9.5%至95.2%。定性分析表明根分割质量显著提高，显示了在正畸中更准确评估根吸收和增强临床决策方面的重要潜力。我们在以下链接提供了该方法的实现和数据集：this https URL。 

---
# Punching Bag vs. Punching Person: Motion Transferability in Videos 

**Title (ZH)**: 拳击袋 vs. 拳击人：视频中的动作迁移能力 

**Authors**: Raiyaan Abdullah, Jared Claypoole, Michael Cogswell, Ajay Divakaran, Yogesh Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2508.00085)  

**Abstract**: Action recognition models demonstrate strong generalization, but can they effectively transfer high-level motion concepts across diverse contexts, even within similar distributions? For example, can a model recognize the broad action "punching" when presented with an unseen variation such as "punching person"? To explore this, we introduce a motion transferability framework with three datasets: (1) Syn-TA, a synthetic dataset with 3D object motions; (2) Kinetics400-TA; and (3) Something-Something-v2-TA, both adapted from natural video datasets. We evaluate 13 state-of-the-art models on these benchmarks and observe a significant drop in performance when recognizing high-level actions in novel contexts. Our analysis reveals: 1) Multimodal models struggle more with fine-grained unknown actions than with coarse ones; 2) The bias-free Syn-TA proves as challenging as real-world datasets, with models showing greater performance drops in controlled settings; 3) Larger models improve transferability when spatial cues dominate but struggle with intensive temporal reasoning, while reliance on object and background cues hinders generalization. We further explore how disentangling coarse and fine motions can improve recognition in temporally challenging datasets. We believe this study establishes a crucial benchmark for assessing motion transferability in action recognition. Datasets and relevant code: this https URL. 

**Abstract (ZH)**: 行动识别模型展示了强大的泛化能力，但它们能否有效地在多样化的情境中转移高层次的运动概念，即使在相似的分布范围内也是如此？例如，当呈现一个未见过的变化，如“打人”时，模型能否识别出“打击”这一广泛的行动？为了探究这一问题，我们引入了一个运动转移性框架，并提供了三个数据集：(1) Syn-TA，一个包含3D物体运动的合成数据集；(2) Kinetics400-TA；以及(3) Something-Something-v2-TA，这三个数据集都改编自自然视频数据集。我们在这些基准上评估了13个最先进的模型，并观察到在新颖情境中识别高层次行动时性能显著下降。我们的分析显示：1) 多模态模型在细粒度的未知动作上比在粗粒度的动作上挣扎更多；2) 偏见自由的Syn-TA与现实世界的数据集一样具有挑战性，模型在受控环境中表现出更大的性能下降；3) 较大的模型在空间线索占主导时能够提高转移性，但在密集的时间推理方面挣扎，而依赖物体和背景线索则阻碍了泛化。我们进一步探讨如何分离粗粒度和细粒度的运动以改善时序挑战数据集中的识别效果。我们认为这项研究为评估动作识别中的运动转移性建立了关键基准。数据集和相关代码：https://github.com/your-repo。 

---
