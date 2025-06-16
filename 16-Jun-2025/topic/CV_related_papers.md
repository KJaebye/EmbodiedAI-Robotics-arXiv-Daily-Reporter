# Linearly Solving Robust Rotation Estimation 

**Title (ZH)**: 线性求解稳健的旋转估计 

**Authors**: Yinlong Liu, Tianyu Huang, Zhi-Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11547)  

**Abstract**: Rotation estimation plays a fundamental role in computer vision and robot tasks, and extremely robust rotation estimation is significantly useful for safety-critical applications. Typically, estimating a rotation is considered a non-linear and non-convex optimization problem that requires careful design. However, in this paper, we provide some new perspectives that solving a rotation estimation problem can be reformulated as solving a linear model fitting problem without dropping any constraints and without introducing any singularities. In addition, we explore the dual structure of a rotation motion, revealing that it can be represented as a great circle on a quaternion sphere surface. Accordingly, we propose an easily understandable voting-based method to solve rotation estimation. The proposed method exhibits exceptional robustness to noise and outliers and can be computed in parallel with graphics processing units (GPUs) effortlessly. Particularly, leveraging the power of GPUs, the proposed method can obtain a satisfactory rotation solution for large-scale($10^6$) and severely corrupted (99$\%$ outlier ratio) rotation estimation problems under 0.5 seconds. Furthermore, to validate our theoretical framework and demonstrate the superiority of our proposed method, we conduct controlled experiments and real-world dataset experiments. These experiments provide compelling evidence supporting the effectiveness and robustness of our approach in solving rotation estimation problems. 

**Abstract (ZH)**: 旋转估计在计算机视觉和机器人任务中发挥着基础性作用，且极其鲁棒的旋转估计对于关键安全应用具有重要意义。通常，估计一个旋转被认为是非线性且非凸的优化问题，需要仔细设计。然而，在本文中，我们提供了一些新的视角，即旋转估计问题可以重新表述为无需去除任何约束且无需引入奇异性的线性模型拟合问题。此外，我们探索了旋转运动的伴随结构，揭示其可以表示为四元数球面上的一条大圆。据此，我们提出了一种易于理解的投票基方法来解决旋转估计问题。所提出的方法对噪声和离群点具有卓越的鲁棒性，并且可以轻松地并行计算于图形处理单元（GPUs）上。特别是，利用GPU的强大功能，所提出的方法可以在0.5秒内为大规模（百万级别）和严重污染（99%离群点比例）的旋转估计问题提供满意的旋转解决方案。此外，为了验证我们的理论框架并展示所提出方法的优越性，我们进行了受控实验和真实世界数据集实验。这些实验为支持我们方法在解决旋转估计问题的有效性和鲁棒性提供了令人信服的证据。 

---
# VLM@school -- Evaluation of AI image understanding on German middle school knowledge 

**Title (ZH)**: VLM@学校——德国中学生知识中AI图像理解的评估 

**Authors**: René Peinl, Vincent Tischler  

**Link**: [PDF](https://arxiv.org/pdf/2506.11604)  

**Abstract**: This paper introduces a novel benchmark dataset designed to evaluate the capabilities of Vision Language Models (VLMs) on tasks that combine visual reasoning with subject-specific background knowledge in the German language. In contrast to widely used English-language benchmarks that often rely on artificially difficult or decontextualized problems, this dataset draws from real middle school curricula across nine domains including mathematics, history, biology, and religion. The benchmark includes over 2,000 open-ended questions grounded in 486 images, ensuring that models must integrate visual interpretation with factual reasoning rather than rely on superficial textual cues. We evaluate thirteen state-of-the-art open-weight VLMs across multiple dimensions, including domain-specific accuracy and performance on adversarial crafted questions. Our findings reveal that even the strongest models achieve less than 45% overall accuracy, with particularly poor performance in music, mathematics, and adversarial settings. Furthermore, the results indicate significant discrepancies between success on popular benchmarks and real-world multimodal understanding. We conclude that middle school-level tasks offer a meaningful and underutilized avenue for stress-testing VLMs, especially in non-English contexts. The dataset and evaluation protocol serve as a rigorous testbed to better understand and improve the visual and linguistic reasoning capabilities of future AI systems. 

**Abstract (ZH)**: 这篇论文介绍了一个新型基准数据集，旨在评估视觉语言模型(VLMs)在结合视觉推理与特定学科背景知识的德语文本任务中的能力。与通常依赖于人工制造的困难或去语境化问题的广泛使用的英语基准不同，该数据集来源于涵盖数学、历史、生物和宗教等九个领域的实际初中课程。该基准包括基于486张图像的超过2000个开放性问题，确保模型必须将视觉解释与事实推理结合起来，而不能仅仅依靠表面的文字线索。我们对十三个最新的开放权重VLMs在多个维度上进行了评估，包括学科特定的准确性以及在对抗性问题上的表现。我们的研究发现，即使是最强大的模型的整体准确率也低于45%，特别是在音乐、数学和对抗性环境中表现尤为不佳。此外，结果表明，在流行基准上的成功与实际多模态理解之间存在显著差异。我们认为，初中水平的任务为测试VLMs提供了一个有意义且尚未充分利用的途径，特别是在非英语背景下。该数据集和评估协议提供了一个严格的测试平台，有助于更深入地理解并提高未来AI系统的视觉和语言推理能力。 

---
# SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies 

**Title (ZH)**: SAIL：比演示更快执行 imitation 学习策略 

**Authors**: Nadun Ranawaka Arachchige, Zhenyang Chen, Wonsuhk Jung, Woo Chul Shin, Rohan Bansal, Pierre Barroso, Yu Hang He, Yingyang Celine Lin, Benjamin Joffe, Shreyas Kousik, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11948)  

**Abstract**: Offline Imitation Learning (IL) methods such as Behavior Cloning are effective at acquiring complex robotic manipulation skills. However, existing IL-trained policies are confined to executing the task at the same speed as shown in demonstration data. This limits the task throughput of a robotic system, a critical requirement for applications such as industrial automation. In this paper, we introduce and formalize the novel problem of enabling faster-than-demonstration execution of visuomotor policies and identify fundamental challenges in robot dynamics and state-action distribution shifts. We instantiate the key insights as SAIL (Speed Adaptation for Imitation Learning), a full-stack system integrating four tightly-connected components: (1) a consistency-preserving action inference algorithm for smooth motion at high speed, (2) high-fidelity tracking of controller-invariant motion targets, (3) adaptive speed modulation that dynamically adjusts execution speed based on motion complexity, and (4) action scheduling to handle real-world system latencies. Experiments on 12 tasks across simulation and two real, distinct robot platforms show that SAIL achieves up to a 4x speedup over demonstration speed in simulation and up to 3.2x speedup in the real world. Additional detail is available at this https URL 

**Abstract (ZH)**: Offlineimitation学习（IL）方法如行为克隆在获取复杂的机器人操作技能方面是有效的。然而，现有的IL训练策略仅限于以演示数据中所示的速度执行任务。这限制了机器人系统的任务处理速率，这是诸如工业自动化等应用中的一个关键要求。在本文中，我们引入并形式化了使感知运动策略以演示数据速度以上的速率执行的新型问题，并识别出机器人动力学和状态-动作分布转移中的基本挑战。我们以SAIL（速度适应的模仿学习）为关键洞察点实现这一目标，它是一个集成了四个紧密连接组件的全栈系统：（1）一种保持一致性的动作推断算法，用于高速下的平滑运动；（2）高度保真的控制器不变运动目标跟踪；（3）适应性的速度调节，基于运动复杂性动态调整执行速度；（4）动作调度以处理实际系统延迟。在模拟和两个真实、不同的机器人平台上进行的12项任务实验表明，SAIL在模拟中实现了演示速度4倍以上的加速，在现实世界中实现了3.2倍以上的加速。更多详细信息请参见此链接：this https URL。 

---
# MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser 

**Title (ZH)**: MindGrab for BrainChop: 适用于命令行和浏览器的快速准确头骨去除方法 

**Authors**: Armina Fani, Mike Doan, Isabelle Le, Alex Fedorov, Malte Hoffmann, Chris Rorden, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2506.11860)  

**Abstract**: We developed MindGrab, a parameter- and memory-efficient deep fully-convolutional model for volumetric skull-stripping in head images of any modality. Its architecture, informed by a spectral interpretation of dilated convolutions, was trained exclusively on modality-agnostic synthetic data. MindGrab was evaluated on a retrospective dataset of 606 multimodal adult-brain scans (T1, T2, DWI, MRA, PDw MRI, EPI, CT, PET) sourced from the SynthStrip dataset. Performance was benchmarked against SynthStrip, ROBEX, and BET using Dice scores, with Wilcoxon signed-rank significance tests. MindGrab achieved a mean Dice score of 95.9 with standard deviation (SD) 1.6 across modalities, significantly outperforming classical methods (ROBEX: 89.1 SD 7.7, P < 0.05; BET: 85.2 SD 14.4, P < 0.05). Compared to SynthStrip (96.5 SD 1.1, P=0.0352), MindGrab delivered equivalent or superior performance in nearly half of the tested scenarios, with minor differences (<3% Dice) in the others. MindGrab utilized 95% fewer parameters (146,237 vs. 2,566,561) than SynthStrip. This efficiency yielded at least 2x faster inference, 50% lower memory usage on GPUs, and enabled exceptional performance (e.g., 10-30x speedup, and up to 30x memory reduction) and accessibility on a wider range of hardware, including systems without high-end GPUs. MindGrab delivers state-of-the-art accuracy with dramatically lower resource demands, supported in brainchop-cli (this https URL) and at this http URL. 

**Abstract (ZH)**: 我们开发了MindGrab，一种高效参数和内存消耗的深全卷积模型，用于任何模态头部图像的体素颅骨去除。该模型的架构受到拉伸卷积频谱解释的启发，并仅在模态无关的合成数据上进行训练。MindGrab在SynthStrip数据集中606个多模态成人脑扫描（T1、T2、DWI、MRA、PDw MRI、EPI、CT、PET）上进行了评估，并使用Dice分数与SynthStrip、ROBEX和BET进行了基准测试，结果通过Wilcoxon符号秩检验。MindGrab在各模态中获得了平均Dice分数95.9，标准差1.6，显著优于经典方法（ROBEX：89.1，标准差7.7，P < 0.05；BET：85.2，标准差14.4，P < 0.05）。与SynthStrip（96.5，标准差1.1，P=0.0352）相比，MindGrab在近一半的测试场景中提供了相当或更优的性能，在其他场景中仅存在轻微差异（Dice相差<3%）。与SynthStrip相比，MindGrab参数量减少了95%（146,237 vs. 2,566,561）。这一效率带来了至少2倍的推理速度提升，50%的GPU内存使用降低，并在广泛硬件上（包括没有高性能GPU的系统）实现了卓越的性能（例如，10-30倍的速度提升和最高30倍的内存缩减）。MindGrab以大幅减少的资源需求提供了最先进的准确性，并可通过brainchop-cli（此链接：[this https URL]）和此链接获得支持。 

---
# Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation 

**Title (ZH)**: 基于在线聚类蒸馏的心脏超声视频表示的自监督学习 

**Authors**: Divyanshu Mishra, Mohammadreza Salehi, Pramit Saha, Olga Patey, Aris T. Papageorghiou, Yuki M. Asano, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2506.11777)  

**Abstract**: Self-supervised learning (SSL) has achieved major advances in natural images and video understanding, but challenges remain in domains like echocardiography (heart ultrasound) due to subtle anatomical structures, complex temporal dynamics, and the current lack of domain-specific pre-trained models. Existing SSL approaches such as contrastive, masked modeling, and clustering-based methods struggle with high intersample similarity, sensitivity to low PSNR inputs common in ultrasound, or aggressive augmentations that distort clinically relevant features. We present DISCOVR (Distilled Image Supervision for Cross Modal Video Representation), a self-supervised dual branch framework for cardiac ultrasound video representation learning. DISCOVR combines a clustering-based video encoder that models temporal dynamics with an online image encoder that extracts fine-grained spatial semantics. These branches are connected through a semantic cluster distillation loss that transfers anatomical knowledge from the evolving image encoder to the video encoder, enabling temporally coherent representations enriched with fine-grained semantic understanding. Evaluated on six echocardiography datasets spanning fetal, pediatric, and adult populations, DISCOVR outperforms both specialized video anomaly detection methods and state-of-the-art video-SSL baselines in zero-shot and linear probing setups, and achieves superior segmentation transfer. 

**Abstract (ZH)**: Distilled Image Supervision for Cross Modal Video Representation in Cardiac Ultrasound 

---
# Pose Matters: Evaluating Vision Transformers and CNNs for Human Action Recognition on Small COCO Subsets 

**Title (ZH)**: 姿态至关重要：小规模COCO子集上视觉变压器与CNN对人体动作识别的评估 

**Authors**: MingZe Tang, Madiha Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11678)  

**Abstract**: This study explores human action recognition using a three-class subset of the COCO image corpus, benchmarking models from simple fully connected networks to transformer architectures. The binary Vision Transformer (ViT) achieved 90% mean test accuracy, significantly exceeding multiclass classifiers such as convolutional networks (approximately 35%) and CLIP-based models (approximately 62-64%). A one-way ANOVA (F = 61.37, p < 0.001) confirmed these differences are statistically significant. Qualitative analysis with SHAP explainer and LeGrad heatmaps indicated that the ViT localizes pose-specific regions (e.g., lower limbs for walking or running), while simpler feed-forward models often focus on background textures, explaining their errors. These findings emphasize the data efficiency of transformer representations and the importance of explainability techniques in diagnosing class-specific failures. 

**Abstract (ZH)**: 本研究使用COCO图像 corpus 的三分类子集探究人体动作识别，benchmark 各种从简单全连接网络到 transformer 架构的模型。二分类 Vision Transformer (ViT) 达到了 90% 的平均测试准确率，显著超过了基于卷积网络的多分类器（大约 35%）和 CLIP 基础模型（大约 62-64%）。单因子方差分析 (F = 61.37, p < 0.001) 确认这些差异具有统计显著性。通过 SHAP 解释器和 LeGrad 热力图的定性分析表明，ViT 专注于姿态特定区域（例如行走或跑步时的下肢），而较简单的前馈模型则通常关注背景纹理，解释了它们的错误。这些发现强调了 transformer 表征的数据效率及其解释性技术在诊断类别特定失败中的重要性。 

---
# A$^2$LC: Active and Automated Label Correction for Semantic Segmentation 

**Title (ZH)**: A$^2$LC: 主动和自动标签纠正方法及其在语义分割中的应用 

**Authors**: Youjin Jeon, Kyusik Cho, Suhan Woo, Euntai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.11599)  

**Abstract**: Active Label Correction (ALC) has emerged as a promising solution to the high cost and error-prone nature of manual pixel-wise annotation in semantic segmentation, by selectively identifying and correcting mislabeled data. Although recent work has improved correction efficiency by generating pseudo-labels using foundation models, substantial inefficiencies still remain. In this paper, we propose Active and Automated Label Correction for semantic segmentation (A$^2$LC), a novel and efficient ALC framework that integrates an automated correction stage into the conventional pipeline. Specifically, the automated correction stage leverages annotator feedback to perform label correction beyond the queried samples, thereby maximizing cost efficiency. In addition, we further introduce an adaptively balanced acquisition function that emphasizes underrepresented tail classes and complements the automated correction mechanism. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate that A$^2$LC significantly outperforms previous state-of-the-art methods. Notably, A$^2$LC achieves high efficiency by outperforming previous methods using only 20% of their budget, and demonstrates strong effectiveness by yielding a 27.23% performance improvement under an equivalent budget constraint on the Cityscapes dataset. The code will be released upon acceptance. 

**Abstract (ZH)**: 主动和自动化标签修正 (A$^2$LC)：面向语义分割的高效标签修正框架 

---
# OV-MAP : Open-Vocabulary Zero-Shot 3D Instance Segmentation Map for Robots 

**Title (ZH)**: OV-MAP : 开 vocabulary 无样本识别 3D 实例分割图 for 机器人 

**Authors**: Juno Kim, Yesol Park, Hye-Jung Yoon, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11585)  

**Abstract**: We introduce OV-MAP, a novel approach to open-world 3D mapping for mobile robots by integrating open-features into 3D maps to enhance object recognition capabilities. A significant challenge arises when overlapping features from adjacent voxels reduce instance-level precision, as features spill over voxel boundaries, blending neighboring regions together. Our method overcomes this by employing a class-agnostic segmentation model to project 2D masks into 3D space, combined with a supplemented depth image created by merging raw and synthetic depth from point clouds. This approach, along with a 3D mask voting mechanism, enables accurate zero-shot 3D instance segmentation without relying on 3D supervised segmentation models. We assess the effectiveness of our method through comprehensive experiments on public datasets such as ScanNet200 and Replica, demonstrating superior zero-shot performance, robustness, and adaptability across diverse environments. Additionally, we conducted real-world experiments to demonstrate our method's adaptability and robustness when applied to diverse real-world environments. 

**Abstract (ZH)**: 开放世界环境下用于移动机器人的新型3D测绘方法：通过将开放特征集成到3D地图中以增强物体识别能力 

---
# FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation 

**Title (ZH)**: FIMA-Q：通过Fishers信息矩阵逼近的视觉变换器后训练量化 

**Authors**: Zhuguanyu Wu, Shihe Wang, Jiayi Zhang, Jiaxin Chen, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11543)  

**Abstract**: Post-training quantization (PTQ) has stood out as a cost-effective and promising model compression paradigm in recent years, as it avoids computationally intensive model retraining. Nevertheless, current PTQ methods for Vision Transformers (ViTs) still suffer from significant accuracy degradation, especially under low-bit quantization. To address these shortcomings, we analyze the prevailing Hessian-guided quantization loss, and uncover certain limitations of conventional Hessian approximations. By following the block-wise reconstruction framework, we propose a novel PTQ method for ViTs, dubbed FIMA-Q. Specifically, we firstly establish the connection between KL divergence and FIM, which enables fast computation of the quantization loss during reconstruction. We further propose an efficient FIM approximation method, namely DPLR-FIM, by employing the diagonal plus low-rank principle, and formulate the ultimate quantization loss. Our extensive experiments, conducted across various vision tasks with representative ViT-based architectures on public datasets, demonstrate that our method substantially promotes the accuracy compared to the state-of-the-art approaches, especially in the case of low-bit quantization. The source code is available at this https URL. 

**Abstract (ZH)**: 后训练量化（PTQ）近年来因其避免了计算密集型的模型重新训练而凸显出成本效益和前景，成为了模型压缩的一种有效范式。然而，当前针对视觉transformer（ViTs）的PTQ方法依然面临着显著的准确率下降问题，尤其是在低位量化的情况下。为解决这些问题，我们分析了现有的海森矩阵引导的量化损失，并揭示了传统海森矩阵近似的一些局限性。通过遵循块重建框架，我们提出了一种新颖的针对ViTs的PTQ方法，称为FIMA-Q。具体地，我们首先建立了KL散度与FIM之间的关联，这使得在重建过程中的量化损失计算变得快速。我们进一步提出了一种高效的FIM近似方法，即DPLR-FIM，通过使用对角占优加低秩的原则，并因此制定了最终的量化损失函数。通过在公共数据集上的各种视觉任务中使用代表性的ViT架构进行广泛实验，我们的方法在低位量化的情况下显著提升了准确率，并且与最先进的方法相比表现更优。源代码可在此网址获得。 

---
# Voxel-Level Brain States Prediction Using Swin Transformer 

**Title (ZH)**: 基于Swin Transformer的体素级别脑状态预测 

**Authors**: Yifei Sun, Daniel Chahine, Qinghao Wen, Tianming Liu, Xiang Li, Yixuan Yuan, Fernando Calamante, Jinglei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.11455)  

**Abstract**: Understanding brain dynamics is important for neuroscience and mental health. Functional magnetic resonance imaging (fMRI) enables the measurement of neural activities through blood-oxygen-level-dependent (BOLD) signals, which represent brain states. In this study, we aim to predict future human resting brain states with fMRI. Due to the 3D voxel-wise spatial organization and temporal dependencies of the fMRI data, we propose a novel architecture which employs a 4D Shifted Window (Swin) Transformer as encoder to efficiently learn spatio-temporal information and a convolutional decoder to enable brain state prediction at the same spatial and temporal resolution as the input fMRI data. We used 100 unrelated subjects from the Human Connectome Project (HCP) for model training and testing. Our novel model has shown high accuracy when predicting 7.2s resting-state brain activities based on the prior 23.04s fMRI time series. The predicted brain states highly resemble BOLD contrast and dynamics. This work shows promising evidence that the spatiotemporal organization of the human brain can be learned by a Swin Transformer model, at high resolution, which provides a potential for reducing the fMRI scan time and the development of brain-computer interfaces in the future. 

**Abstract (ZH)**: 理解大脑动力学对于神经科学和心理健康至关重要。功能性磁共振成像(fMRI)通过血氧水平依赖(BOLD)信号测量神经活动，反映大脑状态。本研究旨在利用fMRI预测未来的人类静息状态大脑活动。由于fMRI数据存在3D体素级别的空间组织和时间依赖性，我们提出了一种新型架构，该架构采用4D移位窗口(Swin)变换器作为编码器以高效地学习空间-时间信息，并采用卷积解码器以在与输入fMRI数据相同的空间和时间分辨率下实现大脑状态预测。我们使用来自Human Connectome Project (HCP)的100名无关受试者的数据进行模型训练和测试。我们的新型模型在基于先前23.04秒fMRI时间序列预测7.2秒静息状态大脑活动方面显示出了高准确性。预测的大脑状态高度类似于BOLD对比度和动态。本工作提供了令人鼓舞的证据，表明SwinTransformer模型可以在高分辨率下学习人类大脑的空间-时间组织，这为未来减少fMRI扫描时间和开发脑机接口提供了潜在可能。 

---
# DPUV4E: High-Throughput DPU Architecture Design for CNN on Versal ACAP 

**Title (ZH)**: DPUV4E：基于Versal ACAP的CNN高吞吐量DPU架构设计 

**Authors**: Guoyu Li, Pengbo Zheng, Jian Weng, Enshan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11441)  

**Abstract**: Convolutional Neural Networks (CNNs) remain prevalent in computer vision applications, and FPGAs, known for their flexibility and energy efficiency, have become essential components in heterogeneous acceleration systems. However, traditional FPGAs face challenges in balancing performance and versatility due to limited on-chip resources. AMD's Versal ACAP architecture, tailored for AI applications, incorporates AI Engines (AIEs) to deliver high computational power. Nevertheless, the platform suffers from insufficient memory bandwidth, hindering the full utilization of the AIEs' theoretical performance. In this paper, we present DPUV4E for the Versal architecture, providing configurations ranging from 2PE ($32.6$ TOPS) to 8PE ($131.0$ TOPS). We design two computation units, Conv PE and DWC PE, to support different computational patterns. Each computation unit's data flow efficiently utilizes the data reuse opportunities to mitigate bandwidth bottlenecks. Additionally, we extend the functionality of each PE to utilize AIEs for non-convolutional operations, reducing resource overhead. Experiments on over 50 models show that compared to previous designs, our design provides $8.6\times$ the TOPS/W of traditional FPGA-based DPU designs, while reducing DSP usage by $95.8\%$, LUT usage by $44.7\%$, and latency to $68.5\%$ under single-batch conditions. For end-to-end inference, our design improving throughput by up to $2.2\times$ for depth-wise convolution models and up to $1.3\times$ for standard models. 

**Abstract (ZH)**: 基于Versal架构的DPUV4E：提供从2PE（32.6 TOPS）到8PE（131.0 TOPS）的配置 

---
# Synthetic Geology -- Structural Geology Meets Deep Learning 

**Title (ZH)**: 合成地质学——构造地质学与深度学习的融合 

**Authors**: Simon Ghyselincks, Valeriia Okhmak, Stefano Zampini, George Turkiyyah, David Keyes, Eldad Haber  

**Link**: [PDF](https://arxiv.org/pdf/2506.11164)  

**Abstract**: Visualizing the first few kilometers of the Earth's subsurface, a long-standing challenge gating a virtually inexhaustible list of important applications, is coming within reach through deep learning. Building on techniques of generative artificial intelligence applied to voxelated images, we demonstrate a method that extends surface geological data supplemented by boreholes to a three-dimensional subsurface region by training a neural network. The Earth's land area having been extensively mapped for geological features, the bottleneck of this or any related technique is the availability of data below the surface. We close this data gap in the development of subsurface deep learning by designing a synthetic data-generator process that mimics eons of geological activity such as sediment compaction, volcanic intrusion, and tectonic dynamics to produce a virtually limitless number of samples of the near lithosphere. A foundation model trained on such synthetic data is able to generate a 3D image of the subsurface from a previously unseen map of surface topography and geology, showing increasing fidelity with increasing access to borehole data, depicting such structures as layers, faults, folds, dikes, and sills. We illustrate the early promise of the combination of a synthetic lithospheric generator with a trained neural network model using generative flow matching. Ultimately, such models will be fine-tuned on data from applicable campaigns, such as mineral prospecting in a given region. Though useful in itself, a regionally fine-tuned models may be employed not as an end but as a means: as an AI-based regularizer in a more traditional inverse problem application, in which the objective function represents the mismatch of additional data with physical models with applications in resource exploration, hazard assessment, and geotechnical engineering. 

**Abstract (ZH)**: 利用深度学习接近可视化地球表层下几百米的地层结构：一个长期的技术挑战，通过生成人工智能技术应用于体素化图像，我们展示了一种方法，该方法利用地表地质数据和井孔数据扩展至三维地下区域，并通过训练神经网络实现。通过设计一种模拟地质活动（如沉积物压实、火山侵入和构造动力学）的合成数据生成过程，我们解决了地下数据的短缺问题，为地下深度学习的发展生成了几乎无限数量的近地幔样本。基于此类合成数据的基座模型能够生成前所未见的地表地形和地质图的三维地下图像，并随着获取更多井孔数据而逐渐提高图像准确性，展现诸如地层、断层、褶皱、岩脉和岩基等结构。我们使用生成流匹配展示了合成地壳生成器与训练神经网络模型结合的早期潜力。最终，这些模型将在适用的勘探活动中进行微调，如特定地区的矿产勘探。尽管这种模型自身已非常有用，但它可以通过作为基于人工智能的正则化器应用于传统逆问题中来发挥更大的作用，其中目标函数代表额外数据与物理模型之间的不匹配，适用于资源勘探、灾害评估和地质工程。 

---
# Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting 

**Title (ZH)**: 评估语音神经表示中各向异性的影响：关键词识别案例研究 

**Authors**: Guillaume Wisniewski, Séverine Guillaume, Clara Rosina Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2506.11096)  

**Abstract**: Pretrained speech representations like wav2vec2 and HuBERT exhibit strong anisotropy, leading to high similarity between random embeddings. While widely observed, the impact of this property on downstream tasks remains unclear. This work evaluates anisotropy in keyword spotting for computational documentary linguistics. Using Dynamic Time Warping, we show that despite anisotropy, wav2vec2 similarity measures effectively identify words without transcription. Our results highlight the robustness of these representations, which capture phonetic structures and generalize across speakers. Our results underscore the importance of pretraining in learning rich and invariant speech representations. 

**Abstract (ZH)**: 预训练语音表示（如wav2vec2和HuBERT）表现出强烈的各向异性，导致随机嵌入之间高度相似。尽管普遍观察到这一特性，但其对下游任务的影响尚不明确。本研究评估了各向异性对计算文档语言学中关键词定位任务的影响。通过动态时间规整，我们显示尽管存在各向异性，wav2vec2的相似性度量仍能有效识别无字幕的单词。我们的结果突显了这些表示的稳健性，它们能够捕捉音素结构并在不同说话人之间泛化。我们的结果强调了预训练在学习丰富且不变的语音表示中的重要性。 

---
# I Can't Believe It's Not Real: CV-MuSeNet: Complex-Valued Multi-Signal Segmentation 

**Title (ZH)**: 我不敢相信这不是真实的：CV-MuSeNet：复值多信号分割 

**Authors**: Sangwon Shin, Mehmet C. Vuran  

**Link**: [PDF](https://arxiv.org/pdf/2506.11048)  

**Abstract**: The increasing congestion of the radio frequency spectrum presents challenges for efficient spectrum utilization. Cognitive radio systems enable dynamic spectrum access with the aid of recent innovations in neural networks. However, traditional real-valued neural networks (RVNNs) face difficulties in low signal-to-noise ratio (SNR) environments, as they were not specifically developed to capture essential wireless signal properties such as phase and amplitude. This work presents CMuSeNet, a complex-valued multi-signal segmentation network for wideband spectrum sensing, to address these limitations. Extensive hyperparameter analysis shows that a naive conversion of existing RVNNs into their complex-valued counterparts is ineffective. Built on complex-valued neural networks (CVNNs) with a residual architecture, CMuSeNet introduces a complexvalued Fourier spectrum focal loss (CFL) and a complex plane intersection over union (CIoU) similarity metric to enhance training performance. Extensive evaluations on synthetic, indoor overthe-air, and real-world datasets show that CMuSeNet achieves an average accuracy of 98.98%-99.90%, improving by up to 9.2 percentage points over its real-valued counterpart and consistently outperforms state of the art. Strikingly, CMuSeNet achieves the accuracy level of its RVNN counterpart in just two epochs, compared to the 27 epochs required for RVNN, while reducing training time by up to a 92.2% over the state of the art. The results highlight the effectiveness of complex-valued architectures in improving weak signal detection and training efficiency for spectrum sensing in challenging low-SNR environments. The dataset is available at: this https URL 

**Abstract (ZH)**: 射频频谱拥堵的挑战及其认知无线电系统的动态频谱访问：一种用于宽带频谱感知的复值多信号分割网络（CMuSeNet） 

---
# Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation 

**Title (ZH)**: 角度域导向：潜在扩散需要旋转而非外推。 

**Authors**: Cheng Jin, Zhenyu Xiao, Chutao Liu, Yuantao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11039)  

**Abstract**: Classifier-free guidance (CFG) has emerged as a pivotal advancement in text-to-image latent diffusion models, establishing itself as a cornerstone technique for achieving high-quality image synthesis. However, under high guidance weights, where text-image alignment is significantly enhanced, CFG also leads to pronounced color distortions in the generated images. We identify that these distortions stem from the amplification of sample norms in the latent space. We present a theoretical framework that elucidates the mechanisms of norm amplification and anomalous diffusion phenomena induced by classifier-free guidance. Leveraging our theoretical insights and the latent space structure, we propose an Angle Domain Guidance (ADG) algorithm. ADG constrains magnitude variations while optimizing angular alignment, thereby mitigating color distortions while preserving the enhanced text-image alignment achieved at higher guidance weights. Experimental results demonstrate that ADG significantly outperforms existing methods, generating images that not only maintain superior text alignment but also exhibit improved color fidelity and better alignment with human perceptual preferences. 

**Abstract (ZH)**: 无分类指导（CFG）已成为文本到图像潜在扩散模型中的关键性进展，确立了其作为实现高质量图像合成的核心技术地位。然而，在高指导权重下，尽管文本与图像的对齐显著增强，CFG 也会导致生成图像中出现明显的颜色失真。我们发现这些失真源于潜在空间中样本范数的放大。我们提供了一个理论框架，阐述了无分类指导引起的范数放大和异常扩散现象的机制。利用我们的理论洞察和潜在空间结构，我们提出了角度域指导（ADG）算法。ADG 控制幅度变化同时优化角度对齐，从而减轻颜色失真并保留高指导权重下增强的文本与图像对齐。实验结果表明，ADG 显著优于现有方法，生成的图像不仅保持了更佳的文本对齐，而且具有更高的颜色保真度和更好的符合人类感知偏好的对齐。 

---
# Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity 

**Title (ZH)**: Tversky神经网络：基于可微Tversky相似性的心理可解释深度学习 

**Authors**: Moussa Koulako Bala Doumbouya, Dan Jurafsky, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2506.11035)  

**Abstract**: Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception. In contrast, Tversky (1977) proposed an axiomatic theory of similarity based on a representation of objects as sets of features, and their similarity as a function of common and distinctive features. However, this model has not been used in deep learning before, partly due to the challenge of incorporating discrete set operations. We develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR. Through experiments with image recognition and language modeling, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer, which employs geometric similarity. On the NABirds image classification task, a frozen ResNet-50 adapted with a Tversky projection layer achieves a 24.7% relative accuracy improvement over the linear layer adapter baseline. With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.5%, and its parameter count by 34.8%. Finally, we propose a unified interpretation of both projection layers as computing similarities of input stimuli to learned prototypes, for which we also propose a novel visualization technique highlighting the interpretability of Tversky projection layers. Our work offers a new paradigm for thinking about the similarity model implicit in deep learning, and designing networks that are interpretable under an established theory of psychological similarity. 

**Abstract (ZH)**: 心理学研究表明，深度学习中的几何相似性模型与人类感知不相符，因为其度量属性如对称性无法符合人类感知。相比之下，Tversky（1977）提出了一种基于对象特征集合及其相似性由共性和独特特征决定的公理化相似性理论。然而，该模型在此之前未被应用于深度学习，部分原因是难以整合离散集操作。我们开发了一个可微分的Tversky相似性参数化模型，可以通过梯度下降学习，并推导出如Tversky投影层等神经网络构建块，与线性投影层不同，它可以建模如异或这样的非线性函数。通过图像识别和语言建模实验，我们展示了Tversky投影层是线性相似性投影层的良好替代方案。在NABirds图像分类任务中，冻结的ResNet-50使用Tversky投影层适配相比线性层适配基线提高了24.7%的相对准确率。GPT-2使用Tversky投影层在PTB上的困惑度降低7.5%，参数量减少34.8%。最后，我们提出了两种投影层作为一个统一解释，即计算输入刺激与学习原型相似性的模型，并提出了新的可视化技术，以突出Tversky投影层的可解释性。我们的工作提供了一个新的范式来思考深度学习中隐含的相似性模型，并设计在已建立的心理相似性理论下具有可解释性的网络。 

---
# Task-aligned prompting improves zero-shot detection of AI-generated images by Vision-Language Models 

**Title (ZH)**: 面向任务的提示改善了视觉-语言模型在零样本检测AI生成图像方面的性能 

**Authors**: Zoher Kachwala, Danishjeet Singh, Danielle Yang, Filippo Menczer  

**Link**: [PDF](https://arxiv.org/pdf/2506.11031)  

**Abstract**: As image generators produce increasingly realistic images, concerns about potential misuse continue to grow. Supervised detection relies on large, curated datasets and struggles to generalize across diverse generators. In this work, we investigate the use of pre-trained Vision-Language Models (VLMs) for zero-shot detection of AI-generated images. While off-the-shelf VLMs exhibit some task-specific reasoning and chain-of-thought prompting offers gains, we show that task-aligned prompting elicits more focused reasoning and significantly improves performance without fine-tuning. Specifically, prefixing the model's response with the phrase ``Let's examine the style and the synthesis artifacts'' -- a method we call zero-shot-s$^2$ -- boosts Macro F1 scores by 8%-29% for two widely used open-source models. These gains are consistent across three recent, diverse datasets spanning human faces, objects, and animals with images generated by 16 different models -- demonstrating strong generalization. We further evaluate the approach across three additional model sizes and observe improvements in most dataset-model combinations -- suggesting robustness to model scale. Surprisingly, self-consistency, a behavior previously observed in language reasoning, where aggregating answers from diverse reasoning paths improves performance, also holds in this setting. Even here, zero-shot-s$^2$ scales better than chain-of-thought in most cases -- indicating that it elicits more useful diversity. Our findings show that task-aligned prompts elicit more focused reasoning and enhance latent capabilities in VLMs, like the detection of AI-generated images -- offering a simple, generalizable, and explainable alternative to supervised methods. Our code is publicly available on github: this https URL. 

**Abstract (ZH)**: 随着图像生成器生成的图像越来越逼真，潜在滥用的问题日益引起关注。监督检测依赖于大规模的策划数据集，并难以在多种生成器上泛化。在本文中，我们研究了预训练的视觉-语言模型（VLMs）在零样本检测AI生成图像中的应用。尽管即用型VLMs表现出一定的任务特定推理能力，且链式提示技术能提升表现，但我们证明了任务对齐的提示能引发更集中的推理，并在无需微调的情况下显著提高性能。具体而言，通过在模型响应前加上短语“Let's examine the style and the synthesis artifacts”——我们称之为零样本s$^2$方法——能够将两种广泛使用的开源模型的宏F1得分提高8%-29%。这些增益在跨越人类面部、物体和动物的三个近期多样数据集上是一致的，展示了较强的泛化能力。我们还在三个额外的模型大小上评估了该方法，并在多数数据集-模型组合中观察到性能提升——表明该方法对模型规模具有鲁棒性。令人惊讶的是，在这里，自我一致性——即从多种推理路径汇总答案以提升性能的现象——也有效。在此设置中，零样本s$^2$在多数情况下比链式提示更有优势——表明它能引发更有用的多样性。我们的研究发现任务对齐的提示能引发更集中的推理，并增强VLMs的潜在能力，如检测AI生成图像的能力——提供了一种简单、可泛化且可解释的替代监督方法。我们的代码已在github上公开：this https URL。 

---
