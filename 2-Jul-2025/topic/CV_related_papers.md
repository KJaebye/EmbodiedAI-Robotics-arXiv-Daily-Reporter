# Box Pose and Shape Estimation and Domain Adaptation for Large-Scale Warehouse Automation 

**Title (ZH)**: 大规模仓库自动化中的盒子姿态、形状估计与领域适应 

**Authors**: Xihang Yu, Rajat Talak, Jingnan Shi, Ulrich Viereck, Igor Gilitschenski, Luca Carlone  

**Link**: [PDF](https://arxiv.org/pdf/2507.00984)  

**Abstract**: Modern warehouse automation systems rely on fleets of intelligent robots that generate vast amounts of data -- most of which remains unannotated. This paper develops a self-supervised domain adaptation pipeline that leverages real-world, unlabeled data to improve perception models without requiring manual annotations. Our work focuses specifically on estimating the pose and shape of boxes and presents a correct-and-certify pipeline for self-supervised box pose and shape estimation. We extensively evaluate our approach across a range of simulated and real industrial settings, including adaptation to a large-scale real-world dataset of 50,000 images. The self-supervised model significantly outperforms models trained solely in simulation and shows substantial improvements over a zero-shot 3D bounding box estimation baseline. 

**Abstract (ZH)**: 现代仓库自动化系统依赖于大量的智能机器人，这些机器人生成了大量数据，其中大部分未标注。本文开发了一种自我监督的领域适应管道，利用现实世界的未标注数据来改进感知模型，无需人工标注。我们的工作专注于估计箱子的姿态和形状，并提出了一种自我监督的箱子姿态和形状估计的正确并认证管道。我们广泛评估了该方法在多个模拟和现实工业环境中的性能，包括对一个包含50,000张图像的大规模现实世界数据集进行适应。自我监督模型显著优于仅在模拟中训练的模型，并在零样本3D边界框估计基线方面显示出大幅改进。 

---
# Sim2Real Diffusion: Learning Cross-Domain Adaptive Representations for Transferable Autonomous Driving 

**Title (ZH)**: Sim2Real 蒸发：跨域自适应表示学习以实现可转移的自动驾驶 

**Authors**: Chinmay Vilas Samak, Tanmay Vilas Samak, Bing Li, Venkat Krovi  

**Link**: [PDF](https://arxiv.org/pdf/2507.00236)  

**Abstract**: Simulation-based design, optimization, and validation of autonomous driving algorithms have proven to be crucial for their iterative improvement over the years. Nevertheless, the ultimate measure of effectiveness is their successful transition from simulation to reality (sim2real). However, existing sim2real transfer methods struggle to comprehensively address the autonomy-oriented requirements of balancing: (i) conditioned domain adaptation, (ii) robust performance with limited examples, (iii) modularity in handling multiple domain representations, and (iv) real-time performance. To alleviate these pain points, we present a unified framework for learning cross-domain adaptive representations for sim2real transferable autonomous driving algorithms using conditional latent diffusion models. Our framework offers options to leverage: (i) alternate foundation models, (ii) a few-shot fine-tuning pipeline, and (iii) textual as well as image prompts for mapping across given source and target domains. It is also capable of generating diverse high-quality samples when diffusing across parameter spaces such as times of day, weather conditions, seasons, and operational design domains. We systematically analyze the presented framework and report our findings in the form of critical quantitative metrics and ablation studies, as well as insightful qualitative examples and remarks. Additionally, we demonstrate the serviceability of the proposed approach in bridging the sim2real gap for end-to-end autonomous driving using a behavioral cloning case study. Our experiments indicate that the proposed framework is capable of bridging the perceptual sim2real gap by over 40%. We hope that our approach underscores the potential of generative diffusion models in sim2real transfer, offering a pathway toward more robust and adaptive autonomous driving. 

**Abstract (ZH)**: 基于仿真训练、优化与验证的自主驾驶算法在多年发展中被证明至关重要。然而，其实现最终目标是从仿真到现实的有效过渡（sim2real）才是衡量其有效性的标准。现有sim2real转移方法难以全面解决自主性导向的需求，包括：（i）条件领域的适应性，（ii）有限样本下的鲁棒性能，（iii）对多种领域表示的模块化处理，以及（iv）实时性能。为缓解这些痛点，我们提出了一种统一框架，利用条件潜在扩散模型学习跨域自适应表示，以实现适用于sim2real转移的自主驾驶算法。该框架提供了利用（i）替代基础模型、（ii）少量样本的微调流水线以及（iii）文本和图像提示以跨越给定源域和目标域进行映射的选项。当沿时间、天气条件、季节和操作设计领域等参数空间扩散时，该框架还能够生成多样化的高质量样本。我们系统分析了所提出的框架，并以关键的定量指标、消融研究以及启发性的定性示例和注释的形式报告了我们的发现。此外，我们展示了该方法跨越端到端自主驾驶的sim2real差距的能力，通过行为克隆案例研究进行了验证。实验结果表明，所提出框架可将感知sim2real差距缩小超过40%。我们希望我们的方法强调生成式扩散模型在sim2real转移中的潜力，提供了一条通往更加鲁棒和适应型自主驾驶的途径。 

---
# Rethink 3D Object Detection from Physical World 

**Title (ZH)**: 重新思考物理世界中的3D目标检测 

**Authors**: Satoshi Tanaka, Koji Minoda, Fumiya Watanabe, Takamasa Horibe  

**Link**: [PDF](https://arxiv.org/pdf/2507.00190)  

**Abstract**: High-accuracy and low-latency 3D object detection is essential for autonomous driving systems. While previous studies on 3D object detection often evaluate performance based on mean average precision (mAP) and latency, they typically fail to address the trade-off between speed and accuracy, such as 60.0 mAP at 100 ms vs 61.0 mAP at 500 ms. A quantitative assessment of the trade-offs between different hardware devices and accelerators remains unexplored, despite being critical for real-time applications. Furthermore, they overlook the impact on collision avoidance in motion planning, for example, 60.0 mAP leading to safer motion planning or 61.0 mAP leading to high-risk motion planning. In this paper, we introduce latency-aware AP (L-AP) and planning-aware AP (P-AP) as new metrics, which consider the physical world such as the concept of time and physical constraints, offering a more comprehensive evaluation for real-time 3D object detection. We demonstrate the effectiveness of our metrics for the entire autonomous driving system using nuPlan dataset, and evaluate 3D object detection models accounting for hardware differences and accelerators. We also develop a state-of-the-art performance model for real-time 3D object detection through latency-aware hyperparameter optimization (L-HPO) using our metrics. Additionally, we quantitatively demonstrate that the assumption "the more point clouds, the better the recognition performance" is incorrect for real-time applications and optimize both hardware and model selection using our metrics. 

**Abstract (ZH)**: 高精度低延迟的3D物体检测对于自主驾驶系统至关重要。虽然以往关于3D物体检测的研究通常基于均值平均精度（mAP）和延迟来评估性能，但它们通常未能解决速度与精度之间的权衡问题，例如60.0 mAP在100毫秒 vs 61.0 mAP在500毫秒。不同硬件设备和加速器之间的权衡量化评估至今未被探索，这一点对于实时应用至关重要。此外，它们忽略了运动规划中的碰撞避免影响，例如60.0 mAP可能导致更安全的运动规划或61.0 mAP可能导致高风险的运动规划。在本文中，我们介绍了延迟感知平均精度（L-AP）和规划感知平均精度（P-AP）作为新的评价指标，这些指标考虑了现实世界的物理概念和物理约束，为实时3D物体检测提供了更全面的评估方法。我们使用nuPlan数据集展示了我们指标在整套自主驾驶系统中的有效性，并评估了考虑硬件差异和加速器的3D物体检测模型。我们还通过我们指标的延迟感知超参数优化（L-HPO）开发了实时3D物体检测的最先进的性能模型。此外，我们定量展示了“点云越多，识别性能越好”的假设不适用于实时应用，并使用我们指标优化了硬件和模型选择。 

---
# Towards Open-World Human Action Segmentation Using Graph Convolutional Networks 

**Title (ZH)**: 基于图卷积网络的开放世界人体动作分割 

**Authors**: Hao Xing, Kai Zhe Boey, Gordon Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.00756)  

**Abstract**: Human-object interaction segmentation is a fundamental task of daily activity understanding, which plays a crucial role in applications such as assistive robotics, healthcare, and autonomous systems. Most existing learning-based methods excel in closed-world action segmentation, they struggle to generalize to open-world scenarios where novel actions emerge. Collecting exhaustive action categories for training is impractical due to the dynamic diversity of human activities, necessitating models that detect and segment out-of-distribution actions without manual annotation. To address this issue, we formally define the open-world action segmentation problem and propose a structured framework for detecting and segmenting unseen actions. Our framework introduces three key innovations: 1) an Enhanced Pyramid Graph Convolutional Network (EPGCN) with a novel decoder module for robust spatiotemporal feature upsampling. 2) Mixup-based training to synthesize out-of-distribution data, eliminating reliance on manual annotations. 3) A novel Temporal Clustering loss that groups in-distribution actions while distancing out-of-distribution samples.
We evaluate our framework on two challenging human-object interaction recognition datasets: Bimanual Actions and 2 Hands and Object (H2O) datasets. Experimental results demonstrate significant improvements over state-of-the-art action segmentation models across multiple open-set evaluation metrics, achieving 16.9% and 34.6% relative gains in open-set segmentation (F1@50) and out-of-distribution detection performances (AUROC), respectively. Additionally, we conduct an in-depth ablation study to assess the impact of each proposed component, identifying the optimal framework configuration for open-world action segmentation. 

**Abstract (ZH)**: 开放世界人-物交互分割：一种日常活动理解的基础任务，对于辅助机器人、医疗健康和自主系统等应用至关重要。现有的基于学习的方法在封闭世界动作分割方面表现出色，但在开放世界场景中难以泛化，其中新动作不断出现。由于人类活动的动态多样性，收集完整的动作类别进行训练是不切实际的，因此需要能够检测和分割未标注动作的模型。为此，我们正式定义开放世界动作分割问题，并提出了一种结构化框架以检测和分割未见过的动作。该框架引入了三项关键创新：1）增强的分层图卷积网络（EPGCN），配有新颖的解码器模块，用于稳健的空间-时间特征上采样。2）基于Mixup的训练方法，合成未标注数据，减少对人工标注的依赖。3）一种新颖的时间聚类损失，用于将已知动作分组，并将未知样本区分开来。我们在两个具有挑战性的手-物交互识别数据集Bimanual Actions and 2 Hands and Object (H2O)上评估了我们的框架。实验结果表明，与最先进的动作分割模型相比，在多个开放集评估指标上取得了显著改进，在开放集分割（F1@50）和未标注检测性能（AUROC）方面分别实现了16.9%和34.6%的相对提升。此外，我们还进行了深入的消融研究，评估了每个提议组件的影响，并确定了开放世界动作分割的最佳框架配置。 

---
# SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures 

**Title (ZH)**: SurgiSR4K: 一种用于机器人辅助微创手术的高分辨率内窥镜视频数据集 

**Authors**: Fengyi Jiang, Xiaorui Zhang, Lingbo Jin, Ruixing Liang, Yuxin Chen, Adi Chola Venkatesh, Jason Culman, Tiantian Wu, Lirong Shao, Wenqing Sun, Cong Gao, Hallie McNamara, Jingpei Lu, Omid Mohareri  

**Link**: [PDF](https://arxiv.org/pdf/2507.00209)  

**Abstract**: High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries. 

**Abstract (ZH)**: 高分辨率成像对于提高视觉清晰度并在微创手术（MIS）中实现精确的计算机辅助引导至关重要。尽管4K内窥镜系统的应用日益广泛，但仍存在显著的数据缺口，缺乏专门针对机器人辅助MIS的原生4K公开数据集。我们介绍了SurgiSR4K，这是首个可公开访问的以原生4K分辨率捕获的手术成像和视频数据集，代表了机器人辅助手术的现实条件。SurgiSR4K包含了包括镜面反射、工具遮挡、出血和软组织变形在内的多种视觉场景，精心设计以反映腹腔镜和机器人手术中面临的常见挑战。该数据集为受益于高分辨率数据的一系列计算机视觉任务打开了可能性，如超分辨率（SR）、烟雾去除、手术器械检测、3D组织重建、单目深度估计、实例分割、新颖视图合成以及视觉-语言模型（VLM）开发。SurgiSR4K为高分辨率手术成像研究提供了坚实的基础，并促进了旨在提高图像引导机器人手术性能、安全性和易用性的智能成像技术的发展。 

---
# Surgical Neural Radiance Fields from One Image 

**Title (ZH)**: 单张图像的手术神经辐射场 

**Authors**: Alberto Neri, Maximilan Fehrentz, Veronica Penza, Leonardo S. Mattos, Nazim Haouchine  

**Link**: [PDF](https://arxiv.org/pdf/2507.00969)  

**Abstract**: Purpose: Neural Radiance Fields (NeRF) offer exceptional capabilities for 3D reconstruction and view synthesis, yet their reliance on extensive multi-view data limits their application in surgical intraoperative settings where only limited data is available. In particular, collecting such extensive data intraoperatively is impractical due to time constraints. This work addresses this challenge by leveraging a single intraoperative image and preoperative data to train NeRF efficiently for surgical scenarios.
Methods: We leverage preoperative MRI data to define the set of camera viewpoints and images needed for robust and unobstructed training. Intraoperatively, the appearance of the surgical image is transferred to the pre-constructed training set through neural style transfer, specifically combining WTC2 and STROTSS to prevent over-stylization. This process enables the creation of a dataset for instant and fast single-image NeRF training.
Results: The method is evaluated with four clinical neurosurgical cases. Quantitative comparisons to NeRF models trained on real surgical microscope images demonstrate strong synthesis agreement, with similarity metrics indicating high reconstruction fidelity and stylistic alignment. When compared with ground truth, our method demonstrates high structural similarity, confirming good reconstruction quality and texture preservation.
Conclusion: Our approach demonstrates the feasibility of single-image NeRF training in surgical settings, overcoming the limitations of traditional multi-view methods. 

**Abstract (ZH)**: 目的：神经辐射场（NeRF）在三维重建和视图合成方面提供了卓越的能力，但由于其依赖于大量多视角数据，限制了其在仅能获得有限数据的外科手术内镜环境下应用。特别是，由于时间限制，在手术过程中收集大量数据是不现实的。本研究通过利用术前图像和预手术数据，在手术场景中高效训练NeRF，从而应对这一挑战。 

---
# Deep learning-based segmentation of T1 and T2 cardiac MRI maps for automated disease detection 

**Title (ZH)**: 基于深度学习的T1和T2心脏MRI分割方法及其在自动化疾病检测中的应用 

**Authors**: Andreea Bianca Popescu, Andreas Seitz, Heiko Mahrholdt, Jens Wetzl, Athira Jacob, Lucian Mihai Itu, Constantin Suciu, Teodora Chitiboi  

**Link**: [PDF](https://arxiv.org/pdf/2507.00903)  

**Abstract**: Objectives Parametric tissue mapping enables quantitative cardiac tissue characterization but is limited by inter-observer variability during manual delineation. Traditional approaches relying on average relaxation values and single cutoffs may oversimplify myocardial complexity. This study evaluates whether deep learning (DL) can achieve segmentation accuracy comparable to inter-observer variability, explores the utility of statistical features beyond mean T1/T2 values, and assesses whether machine learning (ML) combining multiple features enhances disease detection. Materials & Methods T1 and T2 maps were manually segmented. The test subset was independently annotated by two observers, and inter-observer variability was assessed. A DL model was trained to segment left ventricle blood pool and myocardium. Average (A), lower quartile (LQ), median (M), and upper quartile (UQ) were computed for the myocardial pixels and employed in classification by applying cutoffs or in ML. Dice similarity coefficient (DICE) and mean absolute percentage error evaluated segmentation performance. Bland-Altman plots assessed inter-user and model-observer agreement. Receiver operating characteristic analysis determined optimal cutoffs. Pearson correlation compared features from model and manual segmentations. F1-score, precision, and recall evaluated classification performance. Wilcoxon test assessed differences between classification methods, with p < 0.05 considered statistically significant. Results 144 subjects were split into training (100), validation (15) and evaluation (29) subsets. Segmentation model achieved a DICE of 85.4%, surpassing inter-observer agreement. Random forest applied to all features increased F1-score (92.7%, p < 0.001). Conclusion DL facilitates segmentation of T1/ T2 maps. Combining multiple features with ML improves disease detection. 

**Abstract (ZH)**: 基于深度学习的T1/T2图分割及其多特征组合对疾病检测的改善 

---
# LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling 

**Title (ZH)**: LD-RPS: 零样本统一图像恢复 via 潜在扩散递归后验采样 

**Authors**: Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00790)  

**Abstract**: Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data will be available at this https URL. 

**Abstract (ZH)**: 统一图像恢复是低级视觉中的一个显著挑战任务。现有的方法要么为特定任务设计专门的方案，限制了其在各种退化类型上的泛化能力，要么依赖配对数据集训练，因而受到闭集约束。为解决这些问题，我们提出了一种新的、无需数据集、统一的方法，通过预先训练的潜在扩散模型进行循环后验采样。该方法结合多模态理解模型，在无任务约束条件下为生成模型提供语义先验。此外，利用轻量级模块将退化输入与扩散模型生成的偏好对齐，并采用循环细化进行后验采样。广泛的实验表明，我们的方法优于现有先进方法，验证了其有效性和鲁棒性。相关代码和数据将在该网页获取。 

---
# Audio-3DVG: Unified Audio - Point Cloud Fusion for 3D Visual Grounding 

**Title (ZH)**: 音频-3DVG：统一的音频-点云融合用于三维视觉定位 

**Authors**: Duc Cao-Dinh, Khai Le-Duc, Anh Dao, Bach Phan Tat, Chris Ngo, Duy M. H. Nguyen, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00669)  

**Abstract**: 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an Audio-Guided Attention module that captures interactions between candidate objects and relational speech cues, improving target discrimination in cluttered scenes. To support benchmarking, we synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods-highlighting the promise of integrating spoken language into 3D vision tasks. 

**Abstract (ZH)**: 基于音频的3D视觉定位（Audio-3DVG）：结合音频和空间信息的简单有效框架 

---
# AI-Generated Video Detection via Perceptual Straightening 

**Title (ZH)**: 基于感知纠正的AI生成视频检测 

**Authors**: Christian Internò, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, David Klindt  

**Link**: [PDF](https://arxiv.org/pdf/2507.00583)  

**Abstract**: The rapid advancement of generative AI enables highly realistic synthetic videos, posing significant challenges for content authentication and raising urgent concerns about misuse. Existing detection methods often struggle with generalization and capturing subtle temporal inconsistencies. We propose ReStraV(Representation Straightening Video), a novel approach to distinguish natural from AI-generated videos. Inspired by the "perceptual straightening" hypothesis -- which suggests real-world video trajectories become more straight in neural representation domain -- we analyze deviations from this expected geometric property. Using a pre-trained self-supervised vision transformer (DINOv2), we quantify the temporal curvature and stepwise distance in the model's representation domain. We aggregate statistics of these measures for each video and train a classifier. Our analysis shows that AI-generated videos exhibit significantly different curvature and distance patterns compared to real videos. A lightweight classifier achieves state-of-the-art detection performance (e.g., 97.17% accuracy and 98.63% AUROC on the VidProM benchmark), substantially outperforming existing image- and video-based methods. ReStraV is computationally efficient, it is offering a low-cost and effective detection solution. This work provides new insights into using neural representation geometry for AI-generated video detection. 

**Abstract (ZH)**: 基于生成AI的合成视频真实性验证：ReStraV方法的研究 

---
# Visual Anagrams Reveal Hidden Differences in Holistic Shape Processing Across Vision Models 

**Title (ZH)**: 视觉异序揭示视觉模型在整体形状处理方面隐含的差异 

**Authors**: Fenil R. Doshi, Thomas Fel, Talia Konkle, George Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2507.00493)  

**Abstract**: Humans are able to recognize objects based on both local texture cues and the configuration of object parts, yet contemporary vision models primarily harvest local texture cues, yielding brittle, non-compositional features. Work on shape-vs-texture bias has pitted shape and texture representations in opposition, measuring shape relative to texture, ignoring the possibility that models (and humans) can simultaneously rely on both types of cues, and obscuring the absolute quality of both types of representation. We therefore recast shape evaluation as a matter of absolute configural competence, operationalized by the Configural Shape Score (CSS), which (i) measures the ability to recognize both images in Object-Anagram pairs that preserve local texture while permuting global part arrangement to depict different object categories. Across 86 convolutional, transformer, and hybrid models, CSS (ii) uncovers a broad spectrum of configural sensitivity with fully self-supervised and language-aligned transformers -- exemplified by DINOv2, SigLIP2 and EVA-CLIP -- occupying the top end of the CSS spectrum. Mechanistic probes reveal that (iii) high-CSS networks depend on long-range interactions: radius-controlled attention masks abolish performance showing a distinctive U-shaped integration profile, and representational-similarity analyses expose a mid-depth transition from local to global coding. A BagNet control remains at chance (iv), ruling out "border-hacking" strategies. Finally, (v) we show that configural shape score also predicts other shape-dependent evals. Overall, we propose that the path toward truly robust, generalizable, and human-like vision systems may not lie in forcing an artificial choice between shape and texture, but rather in architectural and learning frameworks that seamlessly integrate both local-texture and global configural shape. 

**Abstract (ZH)**: 基于配置知觉能力的形状评分：构建真正 robust、可泛化且类人的视觉系统 

---
# Physics-Aware Style Transfer for Adaptive Holographic Reconstruction 

**Title (ZH)**: 物理感知样式迁移以实现自适应全息重建 

**Authors**: Chanseok Lee, Fakhriyya Mammadova, Jiseong Barg, Mooseok Jang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00482)  

**Abstract**: Inline holographic imaging presents an ill-posed inverse problem of reconstructing objects' complex amplitude from recorded diffraction patterns. Although recent deep learning approaches have shown promise over classical phase retrieval algorithms, they often require high-quality ground truth datasets of complex amplitude maps to achieve a statistical inverse mapping operation between the two domains. Here, we present a physics-aware style transfer approach that interprets the object-to-sensor distance as an implicit style within diffraction patterns. Using the style domain as the intermediate domain to construct cyclic image translation, we show that the inverse mapping operation can be learned in an adaptive manner only with datasets composed of intensity measurements. We further demonstrate its biomedical applicability by reconstructing the morphology of dynamically flowing red blood cells, highlighting its potential for real-time, label-free imaging. As a framework that leverages physical cues inherently embedded in measurements, the presented method offers a practical learning strategy for imaging applications where ground truth is difficult or impossible to obtain. 

**Abstract (ZH)**: 基于inline全息成像的物镜成像 Presents an不适定逆问题，即从记录的衍射图案重构物体的复振幅。尽管最近的深度学习方法在经典相位恢复算法上显示出前景，但它们通常需要高质量的复振幅地图 ground truth 数据集，以实现两个域之间的统计逆映射操作。在这里，我们提出了一种物理感知的风格迁移方法，将物镜到传感器的距离解释为衍射图案中的隐式风格。利用风格域作为中间域构建循环图像转换，我们展示了仅使用由强度测量组成的数据集即可学习逆映射操作。我们还通过重构动态流动的红细胞的形态，展示了其在生物医学成像中的应用，突显了其实时和无标记成像的潜力。作为一种利用测量中内在物理线索的框架，所提出的方法为难以或不可能获得 ground truth 的成像应用提供了实用的学习策略。 

---
# Geological Everything Model 3D: A Promptable Foundation Model for Unified and Zero-Shot Subsurface Understanding 

**Title (ZH)**: 地质万物模型3D：一种可提示的基础模型，用于统一和零样本地下理解 

**Authors**: Yimin Dou, Xinming Wu, Nathan L Bangs, Harpreet Singh Sethi, Jintao Li, Hang Gao, Zhixiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00419)  

**Abstract**: Understanding Earth's subsurface is critical for energy transition, natural hazard mitigation, and planetary science. Yet subsurface analysis remains fragmented, with separate models required for structural interpretation, stratigraphic analysis, geobody segmentation, and property modeling-each tightly coupled to specific data distributions and task formulations. We introduce the Geological Everything Model 3D (GEM), a unified generative architecture that reformulates all these tasks as prompt-conditioned inference along latent structural frameworks derived from subsurface imaging. This formulation moves beyond task-specific models by enabling a shared inference mechanism, where GEM propagates human-provided prompts-such as well logs, masks, or structural sketches-along inferred structural frameworks to produce geologically coherent outputs. Through this mechanism, GEM achieves zero-shot generalization across tasks with heterogeneous prompt types, without retraining for new tasks or data sources. This capability emerges from a two-stage training process that combines self-supervised representation learning on large-scale field seismic data with adversarial fine-tuning using mixed prompts and labels across diverse subsurface tasks. GEM demonstrates broad applicability across surveys and tasks, including Martian radar stratigraphy analysis, structural interpretation in subduction zones, full seismic stratigraphic interpretation, geobody delineation, and property modeling. By bridging expert knowledge with generative reasoning in a structurally aware manner, GEM lays the foundation for scalable, human-in-the-loop geophysical AI-transitioning from fragmented pipelines to a vertically integrated, promptable reasoning system. Project page: this https URL 

**Abstract (ZH)**: 理解和阐释地球的地下结构对于能源转型、自然灾害mitigation以及行星科学至关重要。然而，地下分析仍然碎片化，每个任务（如结构解释、沉积层析分析、地质体分割和属性建模）都需要独立的模型，并且这些模型紧密依赖于特定的数据分布和任务形式。我们提出了一种统一的生成架构地质一切3D模型（GEM），它将所有这些任务重新表述为沿地下成像推断的潜在结构框架进行提示条件下的推断。这种表述通过使GEM能够沿推断出的结构框架传播来自人类的提示（如测井数据、掩码或结构草图），从而生成地质上连贯的输出，从而超越了特定任务模型的限制。GEM能够在不同提示类型之间实现零样本泛化，无需为新任务或数据源重新训练。这一能力源自于一种两阶段的训练过程，该过程结合了大规模现场地震数据的自监督表示学习和使用混合提示和标签进行的竞争性微调，以适用于多种地下任务。GEM在各种调查和任务中具有广泛的适用性，包括火星雷达沉积层析分析、俯冲带结构解释、完整地震沉积层析解释、地质体划分和属性建模。通过以结构感知的方式结合专家知识和生成推理，GEM为可扩展、人在环的地质物理AI奠定了基础，从分离的管道转变为垂直集成、可提示推理系统。 

---
# CGEarthEye:A High-Resolution Remote Sensing Vision Foundation Model Based on the Jilin-1 Satellite Constellation 

**Title (ZH)**: CGEarthEye：基于吉林一号卫星星座的高分辨率遥感视觉基础模型 

**Authors**: Zhiwei Yi, Xin Cheng, Jingyu Ma, Ruifei Zhu, Junwei Tian, Yuanxiu Zhou, Xinge Zhao, Hongzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00356)  

**Abstract**: Deep learning methods have significantly advanced the development of intelligent rinterpretation in remote sensing (RS), with foundational model research based on large-scale pre-training paradigms rapidly reshaping various domains of Earth Observation (EO). However, compared to the open accessibility and high spatiotemporal coverage of medium-resolution data, the limited acquisition channels for ultra-high-resolution optical RS imagery have constrained the progress of high-resolution remote sensing vision foundation models (RSVFM). As the world's largest sub-meter-level commercial RS satellite constellation, the Jilin-1 constellation possesses abundant sub-meter-level image resources. This study proposes CGEarthEye, a RSVFM framework specifically designed for Jilin-1 satellite characteristics, comprising five backbones with different parameter scales with totaling 2.1 billion parameters. To enhance the representational capacity of the foundation model, we developed JLSSD, the first 15-million-scale multi-temporal self-supervised learning (SSL) dataset featuring global coverage with quarterly temporal sampling within a single year, constructed through multi-level representation clustering and sampling strategies. The framework integrates seasonal contrast, augmentation-based contrast, and masked patch token contrastive strategies for pre-training. Comprehensive evaluations across 10 benchmark datasets covering four typical RS tasks demonstrate that the CGEarthEye consistently achieves state-of-the-art (SOTA) performance. Further analysis reveals CGEarthEye's superior characteristics in feature visualization, model convergence, parameter efficiency, and practical mapping applications. This study anticipates that the exceptional representation capabilities of CGEarthEye will facilitate broader and more efficient applications of Jilin-1 data in traditional EO application. 

**Abstract (ZH)**: 基于吉林一号星座的高分辨率遥感视觉基础模型框架CGEarthEye 

---
# Training for X-Ray Vision: Amodal Segmentation, Amodal Content Completion, and View-Invariant Object Representation from Multi-Camera Video 

**Title (ZH)**: X射线视觉训练：多摄像头视频中的无界分割、无界内容完成及视点不变对象表示 

**Authors**: Alexander Moore, Amar Saini, Kylie Cancilla, Doug Poland, Carmen Carrano  

**Link**: [PDF](https://arxiv.org/pdf/2507.00339)  

**Abstract**: Amodal segmentation and amodal content completion require using object priors to estimate occluded masks and features of objects in complex scenes. Until now, no data has provided an additional dimension for object context: the possibility of multiple cameras sharing a view of a scene. We introduce MOVi-MC-AC: Multiple Object Video with Multi-Cameras and Amodal Content, the largest amodal segmentation and first amodal content dataset to date. Cluttered scenes of generic household objects are simulated in multi-camera video. MOVi-MC-AC contributes to the growing literature of object detection, tracking, and segmentation by including two new contributions to the deep learning for computer vision world. Multiple Camera (MC) settings where objects can be identified and tracked between various unique camera perspectives are rare in both synthetic and real-world video. We introduce a new complexity to synthetic video by providing consistent object ids for detections and segmentations between both frames and multiple cameras each with unique features and motion patterns on a single scene. Amodal Content (AC) is a reconstructive task in which models predict the appearance of target objects through occlusions. In the amodal segmentation literature, some datasets have been released with amodal detection, tracking, and segmentation labels. While other methods rely on slow cut-and-paste schemes to generate amodal content pseudo-labels, they do not account for natural occlusions present in the modal masks. MOVi-MC-AC provides labels for ~5.8 million object instances, setting a new maximum in the amodal dataset literature, along with being the first to provide ground-truth amodal content. The full dataset is available at this https URL , 

**Abstract (ZH)**: 无姿态遮挡分割和无姿态内容完成需要利用对象先验来估计复杂场景中被遮挡的 MASK 和特征。MOVi-MC-AC: 多对象视频与多摄像头及无姿态内容，迄今为止最大的无姿态分割数据集及首个无姿态内容数据集。 

---
# Self-Supervised Multiview Xray Matching 

**Title (ZH)**: 自监督多视图X射线匹配 

**Authors**: Mohamad Dabboussi, Malo Huard, Yann Gousseau, Pietro Gori  

**Link**: [PDF](https://arxiv.org/pdf/2507.00287)  

**Abstract**: Accurate interpretation of multi-view radiographs is crucial for diagnosing fractures, muscular injuries, and other anomalies. While significant advances have been made in AI-based analysis of single images, current methods often struggle to establish robust correspondences between different X-ray views, an essential capability for precise clinical evaluations. In this work, we present a novel self-supervised pipeline that eliminates the need for manual annotation by automatically generating a many-to-many correspondence matrix between synthetic X-ray views. This is achieved using digitally reconstructed radiographs (DRR), which are automatically derived from unannotated CT volumes. Our approach incorporates a transformer-based training phase to accurately predict correspondences across two or more X-ray views. Furthermore, we demonstrate that learning correspondences among synthetic X-ray views can be leveraged as a pretraining strategy to enhance automatic multi-view fracture detection on real data. Extensive evaluations on both synthetic and real X-ray datasets show that incorporating correspondences improves performance in multi-view fracture classification. 

**Abstract (ZH)**: 多视角X射线准确解释对于骨折、肌肉损伤和其他异常的诊断至关重要。尽管基于AI的单张图像分析取得了显著进展，但当前方法在建立不同X射线视图之间的稳健对应关系方面仍面临挑战，这对精确临床评估至关重要。本文介绍了一种新颖的自监督流水线，该流水线通过自动生成合成X射线视图之间的多对多对应矩阵，消除了手动标注的需要。这一成果是通过从未标注的CT体积自动推导出的数字化重建射线摄影（DRR）实现的。我们的方法采用基于变换器的训练阶段，以准确预测两张或多张X射线视图之间的对应关系。此外，我们展示了在合成X射线视图之间学习对应关系可以作为一种预训练策略，以增强对实际数据的自动多视角骨折检测性能。对合成和真实X射线数据集的广泛评估表明，整合对应关系可以提高多视角骨折分类的性能。 

---
# Developing Lightweight DNN Models With Limited Data For Real-Time Sign Language Recognition 

**Title (ZH)**: 基于有限数据的轻量级DNN模型在实时手语识别中的开发 

**Authors**: Nikita Nikitin, Eugene Fomin  

**Link**: [PDF](https://arxiv.org/pdf/2507.00248)  

**Abstract**: We present a novel framework for real-time sign language recognition using lightweight DNNs trained on limited data. Our system addresses key challenges in sign language recognition, including data scarcity, high computational costs, and discrepancies in frame rates between training and inference environments. By encoding sign language specific parameters, such as handshape, palm orientation, movement, and location into vectorized inputs, and leveraging MediaPipe for landmark extraction, we achieve highly separable input data representations. Our DNN architecture, optimized for sub 10MB deployment, enables accurate classification of 343 signs with less than 10ms latency on edge devices. The data annotation platform 'slait data' facilitates structured labeling and vector extraction. Our model achieved 92% accuracy in isolated sign recognition and has been integrated into the 'slait ai' web application, where it demonstrates stable inference. 

**Abstract (ZH)**: 基于轻量级DNN的有限数据实时手语识别新型框架 

---
# Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros 

**Title (ZH)**: 基于各种文本编码器的文本到层次扩散模型：应用于超级马里奥 Bros 

**Authors**: Jacob Schrum, Olivia Kilday, Emilio Salas, Bess Hagan, Reid Williams  

**Link**: [PDF](https://arxiv.org/pdf/2507.00184)  

**Abstract**: Recent research shows how diffusion models can unconditionally generate tile-based game levels, but use of diffusion models for text-to-level generation is underexplored. There are practical considerations for creating a usable model: caption/level pairs are needed, as is a text embedding model, and a way of generating entire playable levels, rather than individual scenes. We present strategies to automatically assign descriptive captions to an existing level dataset, and train diffusion models using both pretrained text encoders and simple transformer models trained from scratch. Captions are automatically assigned to generated levels so that the degree of overlap between input and output captions can be compared. We also assess the diversity and playability of the resulting levels. Results are compared with an unconditional diffusion model and a generative adversarial network, as well as the text-to-level approaches Five-Dollar Model and MarioGPT. Notably, the best diffusion model uses a simple transformer model for text embedding, and takes less time to train than diffusion models employing more complex text encoders, indicating that reliance on larger language models is not necessary. We also present a GUI allowing designers to construct long levels from model-generated scenes. 

**Abstract (ZH)**: 最近的研究显示扩散模型可以生成基于瓷砖的游戏关卡，但使用扩散模型进行文本-to-关卡生成的研究尚不充分。在创建可使用模型时存在实际考虑：需要.Caption/关卡对，需要一个文本嵌入模型，以及一种生成整个可玩关卡的方法，而不仅仅是单独的场景。我们提出了自动为现有关卡数据集分配描述性标题的策略，并使用预训练文本编码器和从头训练的简单transformer模型训练扩散模型。为生成的关卡自动分配标题，以便比较输入和输出标题之间的重叠程度。我们还评估了生成关卡的多样性和可玩性。将结果与无条件扩散模型、生成式对抗网络以及Five-Dollar Model和MarioGPT等文本-to-关卡方法进行了比较。值得注意的是，效果最佳的扩散模型使用简单的transformer模型进行文本嵌入，并且比使用更复杂文本编码器的扩散模型训练速度更快，表明依赖更大的语言模型不是必需的。我们还提供了一个GUI，允许设计师从模型生成的场景构建长关卡。 

---
# An efficient plant disease detection using transfer learning approach 

**Title (ZH)**: 使用迁移学习方法的高效植物病害检测 

**Authors**: Bosubabu Sambana, Hillary Sunday Nnadi, Mohd Anas Wajid, Nwosu Ogochukwu Fidelia, Claudia Camacho-Zuñiga, Henry Dozie Ajuzie, Edeh Michael Onyema  

**Link**: [PDF](https://arxiv.org/pdf/2507.00070)  

**Abstract**: Plant diseases pose significant challenges to farmers and the agricultural sector at large. However, early detection of plant diseases is crucial to mitigating their effects and preventing widespread damage, as outbreaks can severely impact the productivity and quality of crops. With advancements in technology, there are increasing opportunities for automating the monitoring and detection of disease outbreaks in plants. This study proposed a system designed to identify and monitor plant diseases using a transfer learning approach. Specifically, the study utilizes YOLOv7 and YOLOv8, two state-ofthe-art models in the field of object detection. By fine-tuning these models on a dataset of plant leaf images, the system is able to accurately detect the presence of Bacteria, Fungi and Viral diseases such as Powdery Mildew, Angular Leaf Spot, Early blight and Tomato mosaic virus. The model's performance was evaluated using several metrics, including mean Average Precision (mAP), F1-score, Precision, and Recall, yielding values of 91.05, 89.40, 91.22, and 87.66, respectively. The result demonstrates the superior effectiveness and efficiency of YOLOv8 compared to other object detection methods, highlighting its potential for use in modern agricultural practices. The approach provides a scalable, automated solution for early any plant disease detection, contributing to enhanced crop yield, reduced reliance on manual monitoring, and supporting sustainable agricultural practices. 

**Abstract (ZH)**: 植物疾病对农民和整个农业部门构成了重大挑战。然而，早期检测植物疾病对于减轻其影响和防止广泛损害至关重要，因为暴发可能严重影响作物的产量和质量。随着技术的进步，自动化监测和检测植物疾病暴发的机会不断增加。本研究提出了一种系统，旨在通过迁移学习方法识别和监测植物疾病。具体而言，该研究利用了YOLOv7和YOLOv8两种当前最先进的目标检测模型。通过在植物叶片图像数据集上对这些模型进行微调，该系统能够准确检测细菌、 fungi 和病毒性疾病，如白粉病、角度叶斑、早疫病和番茄花叶病毒。模型的性能通过平均精确召回率（mAP）、F1分数、精确率和召回率等多项指标进行了评估，分别为91.05、89.40、91.22和87.66。结果表明，YOLOv8在目标检测方法中表现出更优异的效果和效率，突显了其在现代农业实践中的应用潜力。该方法提供了一种可扩大的自动解决方案，用于早期植物疾病检测，有助于提高作物产量、减少手动监控的依赖，并支持可持续的农业实践。 

---
# Vision Transformer with Adversarial Indicator Token against Adversarial Attacks in Radio Signal Classifications 

**Title (ZH)**: 带有对抗指示标记的视觉变换器在无线电信号分类中的对抗攻击防御 

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Xuekang Liu, Fabio Roli, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2507.00015)  

**Abstract**: The remarkable success of transformers across various fields such as natural language processing and computer vision has paved the way for their applications in automatic modulation classification, a critical component in the communication systems of Internet of Things (IoT) devices. However, it has been observed that transformer-based classification of radio signals is susceptible to subtle yet sophisticated adversarial attacks. To address this issue, we have developed a defensive strategy for transformer-based modulation classification systems to counter such adversarial attacks. In this paper, we propose a novel vision transformer (ViT) architecture by introducing a new concept known as adversarial indicator (AdvI) token to detect adversarial attacks. To the best of our knowledge, this is the first work to propose an AdvI token in ViT to defend against adversarial attacks. Integrating an adversarial training method with a detection mechanism using AdvI token, we combine a training time defense and running time defense in a unified neural network model, which reduces architectural complexity of the system compared to detecting adversarial perturbations using separate models. We investigate into the operational principles of our method by examining the attention mechanism. We show the proposed AdvI token acts as a crucial element within the ViT, influencing attention weights and thereby highlighting regions or features in the input data that are potentially suspicious or anomalous. Through experimental results, we demonstrate that our approach surpasses several competitive methods in handling white-box attack scenarios, including those utilizing the fast gradient method, projected gradient descent attacks and basic iterative method. 

**Abstract (ZH)**: 变压器在自然语言处理和计算机视觉等领域取得的显著成功为其在物联网设备通信系统中的自动调制分类应用铺平了道路。然而，观察到基于变压器的调制分类对微妙且复杂的对抗性攻击易感。为解决这一问题，我们开发了一种针对基于变压器的调制分类系统的防御策略以对抗此类对抗性攻击。在本文中，我们通过引入新的概念——对抗指标（AdvI）标记，提出了一种新颖的视觉变压器（ViT）架构以检测对抗性攻击。据我们所知，这是首次在ViT中提出AdvI标记以防御对抗性攻击的工作。通过结合对抗训练方法和使用AdvI标记的检测机制，我们在统一的神经网络模型中实现了训练时防御和运行时防御，与使用单独模型检测对抗性扰动相比，简化了系统的架构复杂度。通过研究方法的运作原理并考察注意力机制，我们证明了所提出的AdvI标记是ViT中的关键元素，影响注意力权重，从而突出输入数据中可能存在可疑或异常的区域或特征。实验结果表明，在白盒攻击场景下，包括使用快速梯度方法、投影梯度下降攻击和基本迭代方法的攻击中，我们的方法超越了多种竞争性方法。 

---
