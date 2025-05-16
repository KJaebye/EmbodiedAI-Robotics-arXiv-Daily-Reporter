# Large-Scale Gaussian Splatting SLAM 

**Title (ZH)**: 大规模高斯插值 SLAM 

**Authors**: Zhe Xin, Chenyang Wu, Penghui Huang, Yanyong Zhang, Yinian Mao, Guoquan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09915)  

**Abstract**: The recently developed Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown encouraging and impressive results for visual SLAM. However, most representative methods require RGBD sensors and are only available for indoor environments. The robustness of reconstruction in large-scale outdoor scenarios remains unexplored. This paper introduces a large-scale 3DGS-based visual SLAM with stereo cameras, termed LSG-SLAM. The proposed LSG-SLAM employs a multi-modality strategy to estimate prior poses under large view changes. In tracking, we introduce feature-alignment warping constraints to alleviate the adverse effects of appearance similarity in rendering losses. For the scalability of large-scale scenarios, we introduce continuous Gaussian Splatting submaps to tackle unbounded scenes with limited memory. Loops are detected between GS submaps by place recognition and the relative pose between looped keyframes is optimized utilizing rendering and feature warping losses. After the global optimization of camera poses and Gaussian points, a structure refinement module enhances the reconstruction quality. With extensive evaluations on the EuRoc and KITTI datasets, LSG-SLAM achieves superior performance over existing Neural, 3DGS-based, and even traditional approaches. Project page: this https URL. 

**Abstract (ZH)**: 大规模3D高斯布判lor方法的立体视觉SLAM（LSG-SLAM） 

---
# Visual Fidelity Index for Generative Semantic Communications with Critical Information Embedding 

**Title (ZH)**: 生成semantic通信中关键信息嵌入的视觉保真度指标 

**Authors**: Jianhao Huang, Qunsong Zeng, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10405)  

**Abstract**: Generative semantic communication (Gen-SemCom) with large artificial intelligence (AI) model promises a transformative paradigm for 6G networks, which reduces communication costs by transmitting low-dimensional prompts rather than raw data. However, purely prompt-driven generation loses fine-grained visual details. Additionally, there is a lack of systematic metrics to evaluate the performance of Gen-SemCom systems. To address these issues, we develop a hybrid Gen-SemCom system with a critical information embedding (CIE) framework, where both text prompts and semantically critical features are extracted for transmissions. First, a novel approach of semantic filtering is proposed to select and transmit the semantically critical features of images relevant to semantic label. By integrating the text prompt and critical features, the receiver reconstructs high-fidelity images using a diffusion-based generative model. Next, we propose the generative visual information fidelity (GVIF) metric to evaluate the visual quality of the generated image. By characterizing the statistical models of image features, the GVIF metric quantifies the mutual information between the distorted features and their original counterparts. By maximizing the GVIF metric, we design a channel-adaptive Gen-SemCom system that adaptively control the volume of features and compression rate according to the channel state. Experimental results validate the GVIF metric's sensitivity to visual fidelity, correlating with both the PSNR and critical information volume. In addition, the optimized system achieves superior performance over benchmarking schemes in terms of higher PSNR and lower FID scores. 

**Abstract (ZH)**: 大型人工智能模型驱动的生成语义通信（Gen-SemCom）：结合关键信息嵌入框架的综合解决方案 

---
# SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity 

**Title (ZH)**: SpikeVideoFormer: 一种高效的基于汉明注意力的尖峰驱动视频变换器，时间复杂度为$\mathcal{O}(T)$ 

**Authors**: Shihao Zou, Qingfeng Li, Wei Ji, Jingjing Li, Yongkui Yang, Guoqi Li, Chao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.10352)  

**Abstract**: Spiking Neural Networks (SNNs) have shown competitive performance to Artificial Neural Networks (ANNs) in various vision tasks, while offering superior energy efficiency. However, existing SNN-based Transformers primarily focus on single-image tasks, emphasizing spatial features while not effectively leveraging SNNs' efficiency in video-based vision tasks. In this paper, we introduce SpikeVideoFormer, an efficient spike-driven video Transformer, featuring linear temporal complexity $\mathcal{O}(T)$. Specifically, we design a spike-driven Hamming attention (SDHA) which provides a theoretically guided adaptation from traditional real-valued attention to spike-driven attention. Building on SDHA, we further analyze various spike-driven space-time attention designs and identify an optimal scheme that delivers appealing performance for video tasks, while maintaining only linear temporal complexity. The generalization ability and efficiency of our model are demonstrated across diverse downstream video tasks, including classification, human pose tracking, and semantic segmentation. Empirical results show our method achieves state-of-the-art (SOTA) performance compared to existing SNN approaches, with over 15\% improvement on the latter two tasks. Additionally, it matches the performance of recent ANN-based methods while offering significant efficiency gains, achieving $\times 16$, $\times 10$ and $\times 5$ improvements on the three tasks. this https URL 

**Abstract (ZH)**: 基于尖峰神经网络的高效视频 Transformer：SpikeVideoFormer 

---
# Modeling Saliency Dataset Bias 

**Title (ZH)**: 建模显著性数据集偏差 

**Authors**: Matthias Kümmerer, Harneet Khanuja, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2505.10169)  

**Abstract**: Recent advances in image-based saliency prediction are approaching gold standard performance levels on existing benchmarks. Despite this success, we show that predicting fixations across multiple saliency datasets remains challenging due to dataset bias. We find a significant performance drop (around 40%) when models trained on one dataset are applied to another. Surprisingly, increasing dataset diversity does not resolve this inter-dataset gap, with close to 60% attributed to dataset-specific biases. To address this remaining generalization gap, we propose a novel architecture extending a mostly dataset-agnostic encoder-decoder structure with fewer than 20 dataset-specific parameters that govern interpretable mechanisms such as multi-scale structure, center bias, and fixation spread. Adapting only these parameters to new data accounts for more than 75% of the generalization gap, with a large fraction of the improvement achieved with as few as 50 samples. Our model sets a new state-of-the-art on all three datasets of the MIT/Tuebingen Saliency Benchmark (MIT300, CAT2000, and COCO-Freeview), even when purely generalizing from unrelated datasets, but with a substantial boost when adapting to the respective training datasets. The model also provides valuable insights into spatial saliency properties, revealing complex multi-scale effects that combine both absolute and relative sizes. 

**Abstract (ZH)**: 基于图像的显著性预测 recent 进展接近现有基准上的黄金标准性能水平。尽管取得了这一成功，我们展示了在多个显著性数据集上预测注视点依然具有挑战性，这是由于数据集偏差所致。我们发现，当在一个数据集上训练的模型应用于另一个数据集时，性能下降幅度可达约 40%。令人大感意外的是，增加数据集多样性并不能解决这一跨数据集的性能差距，有近 60% 的性能差距归因于数据集特定的偏差。为了解决剩余的泛化差距，我们提出了一种新颖的架构，扩展了一种大部分数据集无关的编码-解码结构，并添加少于 20 个数据集特定参数来控制可解释机制，如多尺度结构、中心偏向和注视分布。仅对这些参数进行适应即可解释约 75% 的泛化差距，大部分改进仅使用 50 个样本即可实现。在 MIT/Tuebingen 可视性显著性基准中的三个数据集（MIT300、CAT2000 和 COCO-Freeview）上，我们的模型即使纯粹泛化到无关数据集时也达到了新的最佳性能，但在适应相应的训练数据集时性能获得显著提升。该模型还提供了关于空间显著性特性的宝贵见解，揭示了复杂的多尺度效应，这些效应结合了绝对和相对尺寸。 

---
# ORL-LDM: Offline Reinforcement Learning Guided Latent Diffusion Model Super-Resolution Reconstruction 

**Title (ZH)**: ORL-LDM：离线强化学习指导的 latent 差分模型超分辨率重建 

**Authors**: Shijie Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10027)  

**Abstract**: With the rapid advancement of remote sensing technology, super-resolution image reconstruction is of great research and practical significance. Existing deep learning methods have made progress but still face limitations in handling complex scenes and preserving image details. This paper proposes a reinforcement learning-based latent diffusion model (LDM) fine-tuning method for remote sensing image super-resolution. The method constructs a reinforcement learning environment with states, actions, and rewards, optimizing decision objectives through proximal policy optimization (PPO) during the reverse denoising process of the LDM model. Experiments on the RESISC45 dataset show significant improvements over the baseline model in PSNR, SSIM, and LPIPS, with PSNR increasing by 3-4dB, SSIM improving by 0.08-0.11, and LPIPS reducing by 0.06-0.10, particularly in structured and complex natural scenes. The results demonstrate the method's effectiveness in enhancing super-resolution quality and adaptability across scenes. 

**Abstract (ZH)**: 基于强化学习的潜扩散模型遥感图像超分辨率细调方法 

---
# Application of YOLOv8 in monocular downward multiple Car Target detection 

**Title (ZH)**: YOLOv8在单目向下多目标车辆检测中的应用 

**Authors**: Shijie Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10016)  

**Abstract**: Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited this http URL address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional this http URL improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection. 

**Abstract (ZH)**: 自主驾驶技术正逐步变革传统驾驶方式，标志着现代交通的重要里程碑。物体检测是自主系统的关键基石，对于提升驾驶安全、实现自主功能、提高交通效率以及促进有效应急响应至关重要。然而，当前的技术如用于环境感知的雷达、用于道路感知的摄像头以及车辆传感器网络仍面临高成本、易受天气和光照条件影响以及局限性等问题。为解决这些问题，本文基于YOLOv8提出了一种改进的自主目标检测网络。通过将结构重参数化技术、双向金字塔结构网络模型和新型检测管道集成到YOLOv8框架中，该方法实现了对多尺度、小型和远程物体的高效和精准检测。实验结果表明，改进后的模型在物体检测精度达到65%时，能够有效检测大小物体，展示了相较于传统方法的显著进步。改进后的模型在实际应用中具有巨大潜力，特别适合应用于如中国大学生方程式汽车自主驾驶竞赛（FSAC）等自主驾驶比赛中，尤其在单一目标和小型物体检测场景中表现出色。 

---
# AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection 

**Title (ZH)**: AdaptCLIP: CLIP的通用视觉异常检测适应方法 

**Authors**: Bin-Bin Gao, Yue Zhu, Jiangtao Yan, Yuezhi Cai, Weixi Zhang, Meng Wang, Jun Liu, Yong Liu, Lei Wang, Chengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09926)  

**Abstract**: Universal visual anomaly detection aims to identify anomalies from novel or unseen vision domains without additional fine-tuning, which is critical in open scenarios. Recent studies have demonstrated that pre-trained vision-language models like CLIP exhibit strong generalization with just zero or a few normal images. However, existing methods struggle with designing prompt templates, complex token interactions, or requiring additional fine-tuning, resulting in limited flexibility. In this work, we present a simple yet effective method called AdaptCLIP based on two key insights. First, adaptive visual and textual representations should be learned alternately rather than jointly. Second, comparative learning between query and normal image prompt should incorporate both contextual and aligned residual features, rather than relying solely on residual features. AdaptCLIP treats CLIP models as a foundational service, adding only three simple adapters, visual adapter, textual adapter, and prompt-query adapter, at its input or output ends. AdaptCLIP supports zero-/few-shot generalization across domains and possesses a training-free manner on target domains once trained on a base dataset. AdaptCLIP achieves state-of-the-art performance on 12 anomaly detection benchmarks from industrial and medical domains, significantly outperforming existing competitive methods. We will make the code and model of AdaptCLIP available at this https URL. 

**Abstract (ZH)**: 通用视觉异常检测旨在无需额外微调的情况下识别新型或未见视觉领域的异常，这在开放场景中至关重要。现有研究表明，仅通过预训练视觉-语言模型如CLIP即可实现强大的泛化能力。然而，现有方法在设计提示模板、处理复杂 token 交互或需要额外微调方面存在困难，这限制了其灵活性。本文基于两个关键洞察提出了一种简单而有效的方法，称为AdaptCLIP。首先，视觉和文本表示应交替学习而非联合学习。其次，查询和正常图像提示的比较学习应结合上下文特征和对齐残差特征，而不仅仅是依赖残差特征。AdaptCLIP将CLIP模型视为基础服务，在其输入或输出端仅添加三个简单的适配器：视觉适配器、文本适配器和提示-查询适配器。AdaptCLIP在训练后支持跨领域的零/少样本泛化，并在目标领域中无需训练即可保持无训练状态。AdaptCLIP在12个工业和医疗领域的异常检测基准测试中取得了最先进的性能，显著优于现有竞争方法。我们将在此网址<https://>提供AdaptCLIP的代码和模型。 

---
# UOD: Universal One-shot Detection of Anatomical Landmarks 

**Title (ZH)**: UOD: 全局一次性解剖标志检测 

**Authors**: Heqin Zhu, Quan Quan, Qingsong Yao, Zaiyi Liu, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2306.07615)  

**Abstract**: One-shot medical landmark detection gains much attention and achieves great success for its label-efficient training process. However, existing one-shot learning methods are highly specialized in a single domain and suffer domain preference heavily in the situation of multi-domain unlabeled data. Moreover, one-shot learning is not robust that it faces performance drop when annotating a sub-optimal image. To tackle these issues, we resort to developing a domain-adaptive one-shot landmark detection framework for handling multi-domain medical images, named Universal One-shot Detection (UOD). UOD consists of two stages and two corresponding universal models which are designed as combinations of domain-specific modules and domain-shared modules. In the first stage, a domain-adaptive convolution model is self-supervised learned to generate pseudo landmark labels. In the second stage, we design a domain-adaptive transformer to eliminate domain preference and build the global context for multi-domain data. Even though only one annotated sample from each domain is available for training, the domain-shared modules help UOD aggregate all one-shot samples to detect more robust and accurate landmarks. We investigated both qualitatively and quantitatively the proposed UOD on three widely-used public X-ray datasets in different anatomical domains (i.e., head, hand, chest) and obtained state-of-the-art performances in each domain. The code is available at this https URL. 

**Abstract (ZH)**: 面向多域的一次性医学 landmarks 检测框架：通用一次性检测（UOD） 

---
