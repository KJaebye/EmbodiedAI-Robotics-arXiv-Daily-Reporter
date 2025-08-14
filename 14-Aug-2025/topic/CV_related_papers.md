# RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians 

**Title (ZH)**: RayletDF: Raylet 距离场在点云或高斯分布的一般化3D表面重建中的应用 

**Authors**: Shenxing Wei, Jinxi Li, Yafei Yang, Siyuan Zhou, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09830)  

**Abstract**: In this paper, we present a generalizable method for 3D surface reconstruction from raw point clouds or pre-estimated 3D Gaussians by 3DGS from RGB images. Unlike existing coordinate-based methods which are often computationally intensive when rendering explicit surfaces, our proposed method, named RayletDF, introduces a new technique called raylet distance field, which aims to directly predict surface points from query rays. Our pipeline consists of three key modules: a raylet feature extractor, a raylet distance field predictor, and a multi-raylet blender. These components work together to extract fine-grained local geometric features, predict raylet distances, and aggregate multiple predictions to reconstruct precise surface points. We extensively evaluate our method on multiple public real-world datasets, demonstrating superior performance in surface reconstruction from point clouds or 3D Gaussians. Most notably, our method achieves exceptional generalization ability, successfully recovering 3D surfaces in a single-forward pass across unseen datasets in testing. 

**Abstract (ZH)**: 本文提出了一种用于从原始点云或预先估计的3D高斯分布恢复3D表面的通用方法，该方法基于RGB图像的3DGS。不同于现有基于坐标的方法在绘制显式表面时往往计算密集，我们提出的方法RayletDF引入了一种新的技术——射线距离场，旨在直接从查询射线预测表面点。我们的pipeline由三个关键模块组成：射线特征提取器、射线距离场预测器和多射线融合器。这些组件协同工作以提取细粒度的局部几何特征、预测射线距离，并通过聚合多个预测来重建精确的表面点。我们在多个公开的真实世界数据集上进行了广泛评估，证明了该方法在从点云或3D高斯分布恢复表面方面的优越性能。尤为 noteworthy的是，该方法表现出色的泛化能力，在测试中成功在同一前向传递中恢复未见数据集的3D表面。 

---
# TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos 

**Title (ZH)**: TRACE: 从多视角视频中学习三维高斯物理动力学 

**Authors**: Jinxi Li, Ziyang Song, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09811)  

**Abstract**: In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural nets, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. We propose a new framework named TRACE to model the motion physics of complex dynamic 3D scenes. The key novelty of our method is that, by formulating each 3D point as a rigid particle with size and orientation in space, we directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and one newly created challenging synthetic datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters. 

**Abstract (ZH)**: 本文旨在仅从动态多视角视频中建模3D场景的几何、外观和物理信息，而不使用任何human标签。通过利用物理约束或在神经网络中集成简单的物理模型，现有工作往往难以学习复杂的运动物理，或者学习这些物理需要额外的标签如物体类型或掩码。我们提出了一种新的框架TRACE来建模复杂动态3D场景的运动物理。我们方法的主要创新之处在于，将每个3D点公式化为具有大小和空间方向的刚体粒子，直接学习每个粒子的平移旋转动力学系统，明确估计一组完整的物理参数来控制粒子随时间的运动。在三个现有动态数据集和一个新创建的具有挑战性的合成数据集上进行的广泛实验表明，与基线方法相比，我们的方法在未来帧外推任务中具有出色的性能。我们框架的一个良好特性是，只需通过聚类学习到的物理参数即可轻松地分割多个物体或部分。 

---
# Predictive Uncertainty for Runtime Assurance of a Real-Time Computer Vision-Based Landing System 

**Title (ZH)**: 基于实时计算机视觉的着陆系统运行时保证的预测不确定性 

**Authors**: Romeo Valentin, Sydney M. Katz, Artur B. Carneiro, Don Walker, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2508.09732)  

**Abstract**: Recent advances in data-driven computer vision have enabled robust autonomous navigation capabilities for civil aviation, including automated landing and runway detection. However, ensuring that these systems meet the robustness and safety requirements for aviation applications remains a major challenge. In this work, we present a practical vision-based pipeline for aircraft pose estimation from runway images that represents a step toward the ability to certify these systems for use in safety-critical aviation applications. Our approach features three key innovations: (i) an efficient, flexible neural architecture based on a spatial Soft Argmax operator for probabilistic keypoint regression, supporting diverse vision backbones with real-time inference; (ii) a principled loss function producing calibrated predictive uncertainties, which are evaluated via sharpness and calibration metrics; and (iii) an adaptation of Residual-based Receiver Autonomous Integrity Monitoring (RAIM), enabling runtime detection and rejection of faulty model outputs. We implement and evaluate our pose estimation pipeline on a dataset of runway images. We show that our model outperforms baseline architectures in terms of accuracy while also producing well-calibrated uncertainty estimates with sub-pixel precision that can be used downstream for fault detection. 

**Abstract (ZH)**: Recent Advances in Data-Driven Computer Vision for Robust Aircraft Pose Estimation from Runway Images: A Step Toward Certification for Safety-Critical Aviation Applications 

---
# Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision 

**Title (ZH)**: Surg-InvNeRF: 可逆NeRF在手术视觉中的3D跟踪与重建 

**Authors**: Gerardo Loza, Junlei Hu, Dominic Jones, Sharib Ali, Pietro Valdastri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09681)  

**Abstract**: We proposed a novel test-time optimisation (TTO) approach framed by a NeRF-based architecture for long-term 3D point tracking. Most current methods in point tracking struggle to obtain consistent motion or are limited to 2D motion. TTO approaches frame the solution for long-term tracking as optimising a function that aggregates correspondences from other specialised state-of-the-art methods. Unlike the state-of-the-art on TTO, we propose parametrising such a function with our new invertible Neural Radiance Field (InvNeRF) architecture to perform both 2D and 3D tracking in surgical scenarios. Our approach allows us to exploit the advantages of a rendering-based approach by supervising the reprojection of pixel correspondences. It adapts strategies from recent rendering-based methods to obtain a bidirectional deformable-canonical mapping, to efficiently handle a defined workspace, and to guide the rays' density. It also presents our multi-scale HexPlanes for fast inference and a new algorithm for efficient pixel sampling and convergence criteria. We present results in the STIR and SCARE datasets, for evaluating point tracking and testing the integration of kinematic data in our pipeline, respectively. In 2D point tracking, our approach surpasses the precision and accuracy of the TTO state-of-the-art methods by nearly 50% on average precision, while competing with other approaches. In 3D point tracking, this is the first TTO approach, surpassing feed-forward methods while incorporating the benefits of a deformable NeRF-based reconstruction. 

**Abstract (ZH)**: 基于神经辐射场逆架构的
的时间优化长期33人体外显手术跟踪方法 

---
# Plane Detection and Ranking via Model Information Optimization 

**Title (ZH)**: 基于模型信息优化的平面检测与排序 

**Authors**: Daoxin Zhong, Jun Li, Meng Yee Michael Chuah  

**Link**: [PDF](https://arxiv.org/pdf/2508.09625)  

**Abstract**: Plane detection from depth images is a crucial subtask with broad robotic applications, often accomplished by iterative methods such as Random Sample Consensus (RANSAC). While RANSAC is a robust strategy with strong probabilistic guarantees, the ambiguity of its inlier threshold criterion makes it susceptible to false positive plane detections. This issue is particularly prevalent in complex real-world scenes, where the true number of planes is unknown and multiple planes coexist. In this paper, we aim to address this limitation by proposing a generalised framework for plane detection based on model information optimization. Building on previous works, we treat the observed depth readings as discrete random variables, with their probability distributions constrained by the ground truth planes. Various models containing different candidate plane constraints are then generated through repeated random sub-sampling to explain our observations. By incorporating the physics and noise model of the depth sensor, we can calculate the information for each model, and the model with the least information is accepted as the most likely ground truth. This information optimization process serves as an objective mechanism for determining the true number of planes and preventing false positive detections. Additionally, the quality of each detected plane can be ranked by summing the information reduction of inlier points for each plane. We validate these properties through experiments with synthetic data and find that our algorithm estimates plane parameters more accurately compared to the default Open3D RANSAC plane segmentation. Furthermore, we accelerate our algorithm by partitioning the depth map using neural network segmentation, which enhances its ability to generate more realistic plane parameters in real-world data. 

**Abstract (ZH)**: 基于模型信息优化的平面检测框架 

---
# SegDAC: Segmentation-Driven Actor-Critic for Visual Reinforcement Learning 

**Title (ZH)**: SegDAC：基于分割的演员-评论家视觉强化学习 

**Authors**: Alexandre Brown, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.09325)  

**Abstract**: Visual reinforcement learning (RL) is challenging due to the need to learn both perception and actions from high-dimensional inputs and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains unclear. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground segments semantically via text prompts. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks. 

**Abstract (ZH)**: Segmentation-驱动的Actor- Critic方法（SegDAC）：基于视觉分割的强化学习方法 

---
# Echo-4o: Harnessing the Power of GPT-4o Synthetic Images for Improved Image Generation 

**Title (ZH)**: Echo-4o: 利用GPT-4o合成图像提升图像生成能力 

**Authors**: Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09987)  

**Abstract**: Recently, GPT-4o has garnered significant attention for its strong performance in image generation, yet open-source models still lag behind. Several studies have explored distilling image data from GPT-4o to enhance open-source models, achieving notable progress. However, a key question remains: given that real-world image datasets already constitute a natural source of high-quality data, why should we use GPT-4o-generated synthetic data? In this work, we identify two key advantages of synthetic images. First, they can complement rare scenarios in real-world datasets, such as surreal fantasy or multi-reference image generation, which frequently occur in user queries. Second, they provide clean and controllable supervision. Real-world data often contains complex background noise and inherent misalignment between text descriptions and image content, whereas synthetic images offer pure backgrounds and long-tailed supervision signals, facilitating more accurate text-to-image alignment. Building on these insights, we introduce Echo-4o-Image, a 180K-scale synthetic dataset generated by GPT-4o, harnessing the power of synthetic image data to address blind spots in real-world coverage. Using this dataset, we fine-tune the unified multimodal generation baseline Bagel to obtain Echo-4o. In addition, we propose two new evaluation benchmarks for a more accurate and challenging assessment of image generation capabilities: GenEval++, which increases instruction complexity to mitigate score saturation, and Imagine-Bench, which focuses on evaluating both the understanding and generation of imaginative content. Echo-4o demonstrates strong performance across standard benchmarks. Moreover, applying Echo-4o-Image to other foundation models (e.g., OmniGen2, BLIP3-o) yields consistent performance gains across multiple metrics, highlighting the datasets strong transferability. 

**Abstract (ZH)**: Recently, GPT-4o 在图像生成任务中的强大表现引起了广泛关注，但开源模型仍 lagging behind。一些研究探索将图像数据从 GPT-4o 中提取以提升开源模型，取得了一定进展。然而，一个关键问题仍然存在：既然现实世界图像数据本身就是高质量数据的自然来源，为何要使用 GPT-4o 生成的合成数据？在本文中，我们识别了合成图像的两个主要优势。首先，它们可以补充现实世界数据集中罕见的场景，如超现实的幻想场景或多参考图像生成，这些场景在用户查询中常见。其次，它们提供了清洁和可控的监督。现实世界数据通常包含复杂的背景噪声和文本描述与图像内容之间的固有不对齐，而合成图像则提供了干净的背景和长尾监督信号，有助于更准确的文字到图像的对齐。基于这些见解，我们引入了 Echo-4o-Image，这是一个由 GPT-4o 生成的 180K 规模的合成数据集，利用合成图像数据来弥补现实世界覆盖的盲点。使用此数据集，我们对统一的多模态生成基线 Bagel 进行微调，得到 Echo-4o。此外，我们提出了两个新的评估基准：GenEval++，增加指令复杂性以减轻得分饱和；Imagine-Bench，专注于评估对想象力内容的理解和生成能力。Echo-4o 在标准基准测试中的表现强劲。此外，将 Echo-4o-Image 应用于其他基础模型（例如 OmniGen2 和 BLIP3-o）在多个指标上取得了一致的性能提升，突显了数据集的强转移能力。

Title:
合成图像的两个关键优势：Echo-4o-Image 的研究与应用 

---
# T-CACE: A Time-Conditioned Autoregressive Contrast Enhancement Multi-Task Framework for Contrast-Free Liver MRI Synthesis, Segmentation, and Diagnosis 

**Title (ZH)**: T-CACE：一种基于时间条件的自回归对比增强多任务框架，用于无对比剂肝MRI合成、分割和诊断 

**Authors**: Xiaojiao Xiao, Jianfeng Zhao, Qinmin Vivian Hu, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09919)  

**Abstract**: Magnetic resonance imaging (MRI) is a leading modality for the diagnosis of liver cancer, significantly improving the classification of the lesion and patient outcomes. However, traditional MRI faces challenges including risks from contrast agent (CA) administration, time-consuming manual assessment, and limited annotated datasets. To address these limitations, we propose a Time-Conditioned Autoregressive Contrast Enhancement (T-CACE) framework for synthesizing multi-phase contrast-enhanced MRI (CEMRI) directly from non-contrast MRI (NCMRI). T-CACE introduces three core innovations: a conditional token encoding (CTE) mechanism that unifies anatomical priors and temporal phase information into latent representations; and a dynamic time-aware attention mask (DTAM) that adaptively modulates inter-phase information flow using a Gaussian-decayed attention mechanism, ensuring smooth and physiologically plausible transitions across phases. Furthermore, a constraint for temporal classification consistency (TCC) aligns the lesion classification output with the evolution of the physiological signal, further enhancing diagnostic reliability. Extensive experiments on two independent liver MRI datasets demonstrate that T-CACE outperforms state-of-the-art methods in image synthesis, segmentation, and lesion classification. This framework offers a clinically relevant and efficient alternative to traditional contrast-enhanced imaging, improving safety, diagnostic efficiency, and reliability for the assessment of liver lesion. The implementation of T-CACE is publicly available at: this https URL. 

**Abstract (ZH)**: 磁
user
基于时间的自回归对比增强框架（T-CACE）：直接合成非对比增强
user
基于时间条件的自回归对比增强框架（T-CACE）：直接从非对比MRI合成多期对比增强MRI（CEMRI） 

---
# Automated Segmentation of Coronal Brain Tissue Slabs for 3D Neuropathology 

**Title (ZH)**: 冠状脑组织切片的自动化分割以进行三维神经病理学研究 

**Authors**: Jonathan Williams Ramirez, Dina Zemlyanker, Lucas Deden-Binder, Rogeny Herisse, Erendira Garcia Pallares, Karthik Gopinath, Harshvardhan Gazula, Christopher Mount, Liana N. Kozanno, Michael S. Marshall, Theresa R. Connors, Matthew P. Frosch, Mark Montine, Derek H. Oakley, Christine L. Mac Donald, C. Dirk Keene, Bradley T. Hyman, Juan Eugenio Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2508.09805)  

**Abstract**: Advances in image registration and machine learning have recently enabled volumetric analysis of \emph{postmortem} brain tissue from conventional photographs of coronal slabs, which are routinely collected in brain banks and neuropathology laboratories worldwide. One caveat of this methodology is the requirement of segmentation of the tissue from photographs, which currently requires costly manual intervention. In this article, we present a deep learning model to automate this process. The automatic segmentation tool relies on a U-Net architecture that was trained with a combination of \textit{(i)}1,414 manually segmented images of both fixed and fresh tissue, from specimens with varying diagnoses, photographed at two different sites; and \textit{(ii)}~2,000 synthetic images with randomized contrast and corresponding masks generated from MRI scans for improved generalizability to unseen photographic setups. Automated model predictions on a subset of photographs not seen in training were analyzed to estimate performance compared to manual labels -- including both inter- and intra-rater variability. Our model achieved a median Dice score over 0.98, mean surface distance under 0.4~mm, and 95\% Hausdorff distance under 1.60~mm, which approaches inter-/intra-rater levels. Our tool is publicly available at this http URL. 

**Abstract (ZH)**: 图像注册和机器学习的进步最近使得可以从常规收集的冠状切片的常规照片中分析死后的脑组织体素，这在全球各地的脑银行和 neuropathology 实验室中是很常见的。这一方法的一个局限性是需要对照片进行组织分割，目前这要求昂贵的手动干预。本文介绍了使用深度学习模型来自动化这一过程。该自动分割工具基于一种采用了两种不同位置拍摄的固定和新鲜组织的 1,414 张手动分割图像的 U-Net 架构，以及 2,000 张来自 MRI 扫描并具有随机对比度的合成图像及其对应的掩码，以提高其对未见过的照片设置的泛化能力。在训练数据之外的一小部分照片上进行自动模型预测分析，以估计其性能并与手动标签进行比较，包括不同评估者之间的变异性。我们的模型达到了中位Dice得分为0.98以上，平均表面距离小于0.4毫米，95% Hausdorff距离小于1.60毫米，接近不同评估者之间的水平。我们的工具可在以下网址公开获取。 

---
# Combinative Matching for Geometric Shape Assembly 

**Title (ZH)**: 几何形状装配的组合匹配 

**Authors**: Nahyuk Lee, Juhong Min, Junhong Lee, Chunghyun Park, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.09780)  

**Abstract**: This paper introduces a new shape-matching methodology, combinative matching, to combine interlocking parts for geometric shape assembly. Previous methods for geometric assembly typically rely on aligning parts by finding identical surfaces between the parts as in conventional shape matching and registration. In contrast, we explicitly model two distinct properties of interlocking shapes: 'identical surface shape' and 'opposite volume occupancy.' Our method thus learns to establish correspondences across regions where their surface shapes appear identical but their volumes occupy the inverted space to each other. To facilitate this process, we also learn to align regions in rotation by estimating their shape orientations via equivariant neural networks. The proposed approach significantly reduces local ambiguities in matching and allows a robust combination of parts in assembly. Experimental results on geometric assembly benchmarks demonstrate the efficacy of our method, consistently outperforming the state of the art. Project page: this https URL. 

**Abstract (ZH)**: 这篇论文介绍了一种新的形状匹配方法——组合匹配，用于几何形状装配。以往的几何装配方法通常依赖于通过找到部件之间的相同表面来进行对齐，类似于传统的形状匹配和注册。相比之下，我们明确地建模了嵌锁形状的两种不同属性：“相同表面形状”和“相反体积占用”。因此，我们的方法学会在表面形状看似相同但体积却占据彼此相反空间的区域建立对应关系。为了促进这一过程，我们还学习通过不变神经网络估计形状方向来对齐旋转区域。所提出的方法显著减少了匹配中的局部歧义，并允许装配过程中部件的稳健组合。在几何装配基准上的实验结果表明，该方法的有效性一致超越了现有最佳方法。项目页面：这个 https URL。 

---
# Region-to-Region: Enhancing Generative Image Harmonization with Adaptive Regional Injection 

**Title (ZH)**: 区域到区域：通过自适应区域注射增强生成图像的一致性 kukai
 vidé 

**Authors**: Zhiqiu Zhang, Dongqi Fan, Mingjie Wang, Qiang Tang, Jian Yang, Zili Yi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09746)  

**Abstract**: The goal of image harmonization is to adjust the foreground in a composite image to achieve visual consistency with the background. Recently, latent diffusion model (LDM) are applied for harmonization, achieving remarkable results. However, LDM-based harmonization faces challenges in detail preservation and limited harmonization ability. Additionally, current synthetic datasets rely on color transfer, which lacks local variations and fails to capture complex real-world lighting conditions. To enhance harmonization capabilities, we propose the Region-to-Region transformation. By injecting information from appropriate regions into the foreground, this approach preserves original details while achieving image harmonization or, conversely, generating new composite data. From this perspective, We propose a novel model R2R. Specifically, we design Clear-VAE to preserve high-frequency details in the foreground using Adaptive Filter while eliminating disharmonious elements. To further enhance harmonization, we introduce the Harmony Controller with Mask-aware Adaptive Channel Attention (MACA), which dynamically adjusts the foreground based on the channel importance of both foreground and background regions. To address the limitation of existing datasets, we propose Random Poisson Blending, which transfers color and lighting information from a suitable region to the foreground, thereby generating more diverse and challenging synthetic images. Using this method, we construct a new synthetic dataset, RPHarmony. Experiments demonstrate the superiority of our method over other methods in both quantitative metrics and visual harmony. Moreover, our dataset helps the model generate more realistic images in real examples. Our code, dataset, and model weights have all been released for open access. 

**Abstract (ZH)**: 区域到区域的转换：增强图像谐调能力 

---
# Preacher: Paper-to-Video Agentic System 

**Title (ZH)**: Preacher: 从论文到视频的代理系统 

**Authors**: Jingwei Liu, Ling Yang, Hao Luo, Fan Wang Hongyan Li, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09632)  

**Abstract**: The paper-to-video task converts a research paper into a structured video abstract, distilling key concepts, methods, and conclusions into an accessible, well-organized format. While state-of-the-art video generation models demonstrate potential, they are constrained by limited context windows, rigid video duration constraints, limited stylistic diversity, and an inability to represent domain-specific knowledge. To address these limitations, we introduce Preacher, the first paper-to-video agentic system. Preacher employs a top-down approach to decompose, summarize, and reformulate the paper, followed by bottom-up video generation, synthesizing diverse video segments into a coherent abstract. To align cross-modal representations, we define key scenes and introduce a Progressive Chain of Thought (P-CoT) for granular, iterative planning. Preacher successfully generates high-quality video abstracts across five research fields, demonstrating expertise beyond current video generation models. Code will be released at: this https URL 

**Abstract (ZH)**: 论文视频化任务将研究论文转化为结构化的视频摘要，提炼出关键概念、方法和结论，以一种易于理解且组织良好的格式呈现。尽管最先进的视频生成模型显示出潜力，但它们受到有限上下文窗口、固定的视频时长约束、有限的风格多样性以及无法表示领域特定知识的限制。为了解决这些问题，我们引入了Preacher，这是第一个论文视频化的代理系统。Preacher采用自上而下的方法分解、总结和重述论文，并通过自下而上的视频生成，将多样化的视频片段合成出一个连贯的摘要。为了对齐跨模态表示，我们定义了关键场景并引入了渐进式的因果思维链（P-CoT）进行细粒度和迭代的规划。Preacher成功地在五个研究领域生成了高质量的视频摘要，展示了超越当前视频生成模型的专业知识。代码将在以下链接发布：this https URL。 

---
# COXNet: Cross-Layer Fusion with Adaptive Alignment and Scale Integration for RGBT Tiny Object Detection 

**Title (ZH)**: COXNet：跨层融合带有自适应对齐和尺度集成的RGBT微小目标检测 

**Authors**: Peiran Peng, Tingfa Xu, Liqiang Song, Mengqi Zhu, Yuqiang Fang, Jianan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09533)  

**Abstract**: Detecting tiny objects in multimodal Red-Green-Blue-Thermal (RGBT) imagery is a critical challenge in computer vision, particularly in surveillance, search and rescue, and autonomous navigation. Drone-based scenarios exacerbate these challenges due to spatial misalignment, low-light conditions, occlusion, and cluttered backgrounds. Current methods struggle to leverage the complementary information between visible and thermal modalities effectively. We propose COXNet, a novel framework for RGBT tiny object detection, addressing these issues through three core innovations: i) the Cross-Layer Fusion Module, fusing high-level visible and low-level thermal features for enhanced semantic and spatial accuracy; ii) the Dynamic Alignment and Scale Refinement module, correcting cross-modal spatial misalignments and preserving multi-scale features; and iii) an optimized label assignment strategy using the GeoShape Similarity Measure for better localization. COXNet achieves a 3.32\% mAP$_{50}$ improvement on the RGBTDronePerson dataset over state-of-the-art methods, demonstrating its effectiveness for robust detection in complex environments. 

**Abstract (ZH)**: RGBT多模态图像中微小目标检测是计算机视觉中的一个关键挑战，特别是在监控、搜索与救援以及自主导航领域。基于无人机的场景加剧了这些挑战，由于空间错位、低光照条件、遮挡和复杂背景。当前方法难以有效地利用可见光和热成像之间的互补信息。我们提出COXNet，一种针对RGBT微小目标检测的新型框架，通过三大创新解决这些问题：i) 多层融合模块，融合高层可见光和低层热成像特征，增强语义和空间准确性；ii) 动态对齐与尺度细化模块，纠正跨模态的空间错位并保留多尺度特征；iii) 使用地理形状相似度度量的优化标签分配策略，以实现更好的定位。COXNet在RGBTDronePerson数据集上实现了3.32%的mAP$_{50}$改进，展示了其在复杂环境中的鲁棒检测能力。 

---
# Generation of Indian Sign Language Letters, Numbers, and Words 

**Title (ZH)**: 生成印度手语字母、数字和单词 

**Authors**: Ajeet Kumar Yadav, Nishant Kumar, Rathna G N  

**Link**: [PDF](https://arxiv.org/pdf/2508.09522)  

**Abstract**: Sign language, which contains hand movements, facial expressions and bodily gestures, is a significant medium for communicating with hard-of-hearing people. A well-trained sign language community communicates easily, but those who don't know sign language face significant challenges. Recognition and generation are basic communication methods between hearing and hard-of-hearing individuals. Despite progress in recognition, sign language generation still needs to be explored. The Progressive Growing of Generative Adversarial Network (ProGAN) excels at producing high-quality images, while the Self-Attention Generative Adversarial Network (SAGAN) generates feature-rich images at medium resolutions. Balancing resolution and detail is crucial for sign language image generation. We are developing a Generative Adversarial Network (GAN) variant that combines both models to generate feature-rich, high-resolution, and class-conditional sign language images. Our modified Attention-based model generates high-quality images of Indian Sign Language letters, numbers, and words, outperforming the traditional ProGAN in Inception Score (IS) and Fréchet Inception Distance (FID), with improvements of 3.2 and 30.12, respectively. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words. 

**Abstract (ZH)**: 基于注意力机制的渐进生成对抗网络变体在印度手语图像生成中的应用 

---
# RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization 

**Title (ZH)**: RelayFormer：一种统一的局部-全局注意力框架，用于可扩展的图像和视频操纵定位 

**Authors**: Wen Huang, Jiarui Yang, Tao Dai, Jiawei Li, Shaoxiong Zhan, Bin Wang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.09459)  

**Abstract**: Visual manipulation localization (VML) -- across both images and videos -- is a crucial task in digital forensics that involves identifying tampered regions in visual content. However, existing methods often lack cross-modal generalization and struggle to handle high-resolution or long-duration inputs efficiently.
We propose RelayFormer, a unified and modular architecture for visual manipulation localization across images and videos. By leveraging flexible local units and a Global-Local Relay Attention (GLoRA) mechanism, it enables scalable, resolution-agnostic processing with strong generalization. Our framework integrates seamlessly with existing Transformer-based backbones, such as ViT and SegFormer, via lightweight adaptation modules that require only minimal architectural changes, ensuring compatibility without disrupting pretrained representations.
Furthermore, we design a lightweight, query-based mask decoder that supports one-shot inference across video sequences with linear complexity. Extensive experiments across multiple benchmarks demonstrate that our approach achieves state-of-the-art localization performance, setting a new baseline for scalable and modality-agnostic VML. Code is available at: this https URL. 

**Abstract (ZH)**: 视觉操控定位（VML）——适用于图像和视频——是数字鉴证中的一个关键任务，涉及识别视觉内容中的篡改区域。然而，现有方法往往缺乏跨模态的泛化能力，并且难以高效处理高分辨率或长时间的输入。

我们提出RelayFormer，一种用于图像和视频的统一和模块化视觉操控定位架构。通过利用灵活的局部单元和全局-局部传递注意力（GLoRA）机制，它实现了可扩展、分辨率无关的处理能力，并且具有强大的泛化能力。我们的框架可以通过轻量级适应模块无缝集成现有的基于Transformer的骨干网络，如ViT和SegFormer，仅需进行少量的架构改动，确保兼容性而不破坏预训练表示。

此外，我们设计了一种轻量级、基于查询的掩码解码器，支持在视频序列上进行一-shot推理，具有一线性复杂度。在多个基准上的广泛实验表明，我们的方法实现了最先进的定位性能，为可扩展和模态无关的VML设定了新的基准。代码可在以下链接获取：this https URL。 

---
# RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata 

**Title (ZH)**: RampNet：街道景观图像中开放政府元数据辅助坡道检测的两阶段框架 

**Authors**: John S. O'Meara, Jared Hwang, Zeyu Wang, Michael Saugstad, Jon E. Froehlich  

**Link**: [PDF](https://arxiv.org/pdf/2508.09415)  

**Abstract**: Curb ramps are critical for urban accessibility, but robustly detecting them in images remains an open problem due to the lack of large-scale, high-quality datasets. While prior work has attempted to improve data availability with crowdsourced or manually labeled data, these efforts often fall short in either quality or scale. In this paper, we introduce and evaluate a two-stage pipeline called RampNet to scale curb ramp detection datasets and improve model performance. In Stage 1, we generate a dataset of more than 210,000 annotated Google Street View (GSV) panoramas by auto-translating government-provided curb ramp location data to pixel coordinates in panoramic images. In Stage 2, we train a curb ramp detection model (modified ConvNeXt V2) from the generated dataset, achieving state-of-the-art performance. To evaluate both stages of our pipeline, we compare to manually labeled panoramas. Our generated dataset achieves 94.0% precision and 92.5% recall, and our detection model reaches 0.9236 AP -- far exceeding prior work. Our work contributes the first large-scale, high-quality curb ramp detection dataset, benchmark, and model. 

**Abstract (ZH)**: 基于图像的无障碍坡道检测：RampNet双阶段管道方法 

---
# X-UniMotion: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents 

**Title (ZH)**: X-UniMotion: 动态人类图像的表情化、统一且身份无关的动力学潜在表示 

**Authors**: Guoxian Song, Hongyi Xu, Xiaochen Zhao, You Xie, Tianpei Gu, Zenan Li, Chenxu Zhang, Linjie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.09383)  

**Abstract**: We present X-UniMotion, a unified and expressive implicit latent representation for whole-body human motion, encompassing facial expressions, body poses, and hand gestures. Unlike prior motion transfer methods that rely on explicit skeletal poses and heuristic cross-identity adjustments, our approach encodes multi-granular motion directly from a single image into a compact set of four disentangled latent tokens -- one for facial expression, one for body pose, and one for each hand. These motion latents are both highly expressive and identity-agnostic, enabling high-fidelity, detailed cross-identity motion transfer across subjects with diverse identities, poses, and spatial configurations.
To achieve this, we introduce a self-supervised, end-to-end framework that jointly learns the motion encoder and latent representation alongside a DiT-based video generative model, trained on large-scale, diverse human motion datasets. Motion--identity disentanglement is enforced via 2D spatial and color augmentations, as well as synthetic 3D renderings of cross-identity subject pairs under shared poses. Furthermore, we guide motion token learning with auxiliary decoders that promote fine-grained, semantically aligned, and depth-aware motion embeddings.
Extensive experiments show that X-UniMotion outperforms state-of-the-art methods, producing highly expressive animations with superior motion fidelity and identity preservation. 

**Abstract (ZH)**: X-UniMotion：统一且表达丰富的整体人体运动隐式潜在表示 

---
# A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition 

**Title (ZH)**: 基于标识者不变的收敛器和多尺度融合变压器的连续手语识别 

**Authors**: Md Rezwanul Haque, Md. Milon Islam, S M Taslim Uddin Raju, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2508.09372)  

**Abstract**: Continuous Sign Language Recognition (CSLR) faces multiple challenges, including significant inter-signer variability and poor generalization to novel sentence structures. Traditional solutions frequently fail to handle these issues efficiently. For overcoming these constraints, we propose a dual-architecture framework. For the Signer-Independent (SI) challenge, we propose a Signer-Invariant Conformer that combines convolutions with multi-head self-attention to learn robust, signer-agnostic representations from pose-based skeletal keypoints. For the Unseen-Sentences (US) task, we designed a Multi-Scale Fusion Transformer with a novel dual-path temporal encoder that captures both fine-grained posture dynamics, enabling the model's ability to comprehend novel grammatical compositions. Experiments on the challenging Isharah-1000 dataset establish a new standard for both CSLR benchmarks. The proposed conformer architecture achieves a Word Error Rate (WER) of 13.07% on the SI challenge, a reduction of 13.53% from the state-of-the-art. On the US task, the transformer model scores a WER of 47.78%, surpassing previous work. In the SignEval 2025 CSLR challenge, our team placed 2nd in the US task and 4th in the SI task, demonstrating the performance of these models. The findings validate our key hypothesis: that developing task-specific networks designed for the particular challenges of CSLR leads to considerable performance improvements and establishes a new baseline for further research. The source code is available at: this https URL. 

**Abstract (ZH)**: 连续手语识别（CSLR）面临的挑战包括显著的发际间变异性以及对新型句子结构的 poor 通用性。传统解决方案往往难以高效应对这些难题。为克服这些限制，我们提出了一种双架构框架。对于发际间不变性（SI）挑战，我们提出了一个招式不变式 Conformer，结合卷积与多头自我注意，从基于姿态的骨骼关键点中学习 robust、招式无偏的表示。对于未见句子（US）任务，我们设计了一种多尺度融合变换器，带有新颖的双路径时间编码器，能够捕捉细微的姿态动力学，使模型能够理解新型语法组成。在具有挑战性的 Isharah-1000 数据集上的实验确立了 CSLR 基准的新标准。提出的 Conformer 架构在 SI 挑战中实现了 13.07% 的词错误率（WER），比最先进的方法降低了 13.53%。在 US 任务中，变换器模型的 WER 为 47.78%，超过了之前的工作。在 SignEval 2025 CSLR 挑战中，我们的团队在 US 任务中排名第 2，在 SI 任务中排名第 4，展示了这些模型的性能。研究结果验证了我们的核心假设：开发针对 CSLR 特定挑战的任务特定网络可以带来显著的性能提升，并为未来的进一步研究建立了新的基线。源代码可在以下链接获取：this https URL。 

---
# Gradient-Direction-Aware Density Control for 3D Gaussian Splatting 

**Title (ZH)**: 面向梯度方向的密度控制用于3D Gaussian散列 

**Authors**: Zheng Zhou, Yu-Jie Xiong, Chun-Ming Xia, Jia-Chen Zhang, Hong-Jian Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09239)  

**Abstract**: The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS), a gradient-direction-aware adaptive density control framework to address these challenges. Our key innovations: the gradient coherence ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations with 50\% reduced memory consumption through optimized Gaussians utilization. 

**Abstract (ZH)**: 基于梯度方向的自适应密度控制高斯溅射（GDAGS）：解决三维场景合成中的过度重建与过度密集问题 

---
# Towards Scalable Training for Handwritten Mathematical Expression Recognition 

**Title (ZH)**: 面向手写数学表达式识别的可扩展训练方法 

**Authors**: Haoyang Li, Jiaqing Li, Jialun Cao, Zongyuan Yang, Yongping Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.09220)  

**Abstract**: Large foundation models have achieved significant performance gains through scalable training on massive datasets. However, the field of \textbf{H}andwritten \textbf{M}athematical \textbf{E}xpression \textbf{R}ecognition (HMER) has been impeded by the scarcity of data, primarily due to the arduous and costly process of manual annotation. To bridge this gap, we propose a novel method integrating limited handwritten formulas with large-scale LaTeX-rendered formulas by developing a scalable data engine to generate complex and consistent LaTeX sequences. With this engine, we built the largest formula dataset to date, termed \texttt{Tex80M}, comprising over 80 million high-quality training instances. Then we propose \texttt{TexTeller}, the first HMER model trained at scale, by mix-training \texttt{Tex80M} with a relatively small HME dataset. The expansive training dataset and our refined pipeline have equipped \texttt{TexTeller} with state-of-the-art (SOTA) performance across nearly all benchmarks. To advance the field, we will openly release our complete model, entire dataset, and full codebase, enabling further research building upon our contributions. 

**Abstract (ZH)**: 大规模基础模型通过在大规模数据集上进行可扩展训练实现了显著的性能提升。然而，手写数学表达式识别（HMER）领域因数据稀缺而受到阻碍，主要原因是人工标注过程既繁复又昂贵。为解决这一问题，我们提出了一种新颖的方法，将有限的手写公式与大规模的LaTeX渲染公式相结合，通过开发一个可扩展的数据引擎生成复杂且一致的LaTeX序列。利用这一引擎，我们构建了迄今为止最大的公式数据集，称为Tex80M，包含超过8000万个高质量的训练实例。然后，我们提出了第一款在大规模数据上训练的手写数学表达式识别模型TexTeller，该模型通过混合训练Tex80M与相对较小的手写数学表达式（HME）数据集实现。扩展的训练数据集和我们精炼的流水线使得TexTeller在几乎所有基准测试中都达到了最先进的（SOTA）性能。为推动该领域的发展，我们将公开发布我们的完整模型、整个数据集和全部代码，以促进进一步的研究。 

---
# Real-time deep learning phase imaging flow cytometer reveals blood cell aggregate biomarkers for haematology diagnostics 

**Title (ZH)**: 实时深度学习相位成像细胞分析仪揭示血液细胞聚集体生物标志物用于血液学诊断 

**Authors**: Kerem Delikoyun, Qianyu Chen, Liu Wei, Si Ko Myo, Johannes Krell, Martin Schlegel, Win Sen Kuan, John Tshon Yit Soong, Gerhard Schneider, Clarissa Prazeres da Costa, Percy A. Knolle, Laurent Renia, Matthew Edward Cove, Hwee Kuan Lee, Klaus Diepold, Oliver Hayden  

**Link**: [PDF](https://arxiv.org/pdf/2508.09215)  

**Abstract**: While analysing rare blood cell aggregates remains challenging in automated haematology, they could markedly advance label-free functional diagnostics. Conventional flow cytometers efficiently perform cell counting with leukocyte differentials but fail to identify aggregates with flagged results, requiring manual reviews. Quantitative phase imaging flow cytometry captures detailed aggregate morphologies, but clinical use is hampered by massive data storage and offline processing. Incorporating hidden biomarkers into routine haematology panels would significantly improve diagnostics without flagged results. We present RT-HAD, an end-to-end deep learning-based image and data processing framework for off-axis digital holographic microscopy (DHM), which combines physics-consistent holographic reconstruction and detection, representing each blood cell in a graph to recognize aggregates. RT-HAD processes >30 GB of image data on-the-fly with turnaround time of <1.5 min and error rate of 8.9% in platelet aggregate detection, which matches acceptable laboratory error rates of haematology biomarkers and solves the big data challenge for point-of-care diagnostics. 

**Abstract (ZH)**: 基于端到端深度学习的离轴数字全息显微镜图像和数据处理框架 RT-HAD：用于识别血小板聚集体的物理一致全息重构与检测 

---
# Personalized Feature Translation for Expression Recognition: An Efficient Source-Free Domain Adaptation Method 

**Title (ZH)**: 基于个性特征翻译的表情识别：一种高效的源无监督域适应方法 

**Authors**: Masoumeh Sharafi, Soufiane Belharbi, Houssem Ben Salem, Ali Etemad, Alessandro Lameiras Koerich, Marco Pedersoli, Simon Bacon, Eric Granger  

**Link**: [PDF](https://arxiv.org/pdf/2508.09202)  

**Abstract**: Facial expression recognition (FER) models are employed in many video-based affective computing applications, such as human-computer interaction and healthcare monitoring. However, deep FER models often struggle with subtle expressions and high inter-subject variability, limiting their performance in real-world applications. To improve their performance, source-free domain adaptation (SFDA) methods have been proposed to personalize a pretrained source model using only unlabeled target domain data, thereby avoiding data privacy, storage, and transmission constraints. This paper addresses a challenging scenario where source data is unavailable for adaptation, and only unlabeled target data consisting solely of neutral expressions is available. SFDA methods are not typically designed to adapt using target data from only a single class. Further, using models to generate facial images with non-neutral expressions can be unstable and computationally intensive. In this paper, personalized feature translation (PFT) is proposed for SFDA. Unlike current image translation methods for SFDA, our lightweight method operates in the latent space. We first pre-train the translator on the source domain data to transform the subject-specific style features from one source subject into another. Expression information is preserved by optimizing a combination of expression consistency and style-aware objectives. Then, the translator is adapted on neutral target data, without using source data or image synthesis. By translating in the latent space, PFT avoids the complexity and noise of face expression generation, producing discriminative embeddings optimized for classification. Using PFT eliminates the need for image synthesis, reduces computational overhead (using a lightweight translator), and only adapts part of the model, making the method efficient compared to image-based translation. 

**Abstract (ZH)**: 无源域适配中个性化特征翻译方法：面部表情识别模型的改进 

---
# Hybrid(Transformer+CNN)-based Polyp Segmentation 

**Title (ZH)**: 基于Transformer和CNN结合的息肉分割 

**Authors**: Madan Baduwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.09189)  

**Abstract**: Colonoscopy is still the main method of detection and segmentation of colonic polyps, and recent advancements in deep learning networks such as U-Net, ResUNet, Swin-UNet, and PraNet have made outstanding performance in polyp segmentation. Yet, the problem is extremely challenging due to high variation in size, shape, endoscopy types, lighting, imaging protocols, and ill-defined boundaries (fluid, folds) of the polyps, rendering accurate segmentation a challenging and problematic task. To address these critical challenges in polyp segmentation, we introduce a hybrid (Transformer + CNN) model that is crafted to enhance robustness against evolving polyp characteristics. Our hybrid architecture demonstrates superior performance over existing solutions, particularly in addressing two critical challenges: (1) accurate segmentation of polyps with ill-defined margins through boundary-aware attention mechanisms, and (2) robust feature extraction in the presence of common endoscopic artifacts, including specular highlights, motion blur, and fluid occlusions. Quantitative evaluations reveal significant improvements in segmentation accuracy (Recall improved by 1.76%, i.e., 0.9555, accuracy improved by 0.07%, i.e., 0.9849) and artifact resilience compared to state-of-the-art polyp segmentation methods. 

**Abstract (ZH)**: 基于Transformer和CNN的混合模型在结肠息肉分割中的应用：提高对 evolving 特征的鲁棒性 

---
# Generative Artificial Intelligence in Medical Imaging: Foundations, Progress, and Clinical Translation 

**Title (ZH)**: 医学成像中的生成型人工智能：基础、进展与临床转化 

**Authors**: Xuanru Zhou, Cheng Li, Shuqiang Wang, Ye Li, Tao Tan, Hairong Zheng, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09177)  

**Abstract**: Generative artificial intelligence (AI) is rapidly transforming medical imaging by enabling capabilities such as data synthesis, image enhancement, modality translation, and spatiotemporal modeling. This review presents a comprehensive and forward-looking synthesis of recent advances in generative modeling including generative adversarial networks (GANs), variational autoencoders (VAEs), diffusion models, and emerging multimodal foundation architectures and evaluates their expanding roles across the clinical imaging continuum. We systematically examine how generative AI contributes to key stages of the imaging workflow, from acquisition and reconstruction to cross-modality synthesis, diagnostic support, and treatment planning. Emphasis is placed on both retrospective and prospective clinical scenarios, where generative models help address longstanding challenges such as data scarcity, standardization, and integration across modalities. To promote rigorous benchmarking and translational readiness, we propose a three-tiered evaluation framework encompassing pixel-level fidelity, feature-level realism, and task-level clinical relevance. We also identify critical obstacles to real-world deployment, including generalization under domain shift, hallucination risk, data privacy concerns, and regulatory hurdles. Finally, we explore the convergence of generative AI with large-scale foundation models, highlighting how this synergy may enable the next generation of scalable, reliable, and clinically integrated imaging systems. By charting technical progress and translational pathways, this review aims to guide future research and foster interdisciplinary collaboration at the intersection of AI, medicine, and biomedical engineering. 

**Abstract (ZH)**: 生成型人工智能（AI）正在通过使能数据合成、图像增强、时序建模等功能迅速变革医学影像领域。本文综述了生成型建模领域的近期进展，包括生成型对抗网络（GANs）、变分自动编码器（VAEs）、扩散模型等等，并新兴的多模态基础架构，并评估了生成型 AI 在临床影像连续统中的不断拓展角色。我们系统性地探讨了生成型 AI 如何贯穿于影像工作流中的关键步骤，包括采集与重建、多模态合成、辅助诊断、治疗规划等。特别强调了回顾性和前瞻性临床场景中生成型模型如何应对长久以来的挑战，如如 scarcity of standardization data across modalities and integration 的稀缺性标准标记数据及模态集成方面的困难与前景。我们提倡制定了一个包含像素级保真度、特征层面的真实性和任务级临床相关性的三层评估框架，并还 identified key challenges to real-world deployment 包括普遍性性的域下转移、错觉生成、隐私关切与监管障碍。最后，，我们提出了生成型 AI 与大基预模型的融合可能会促成可下一代可可、可靠且临床集成的影像系统。通过突显技术进步与转化路径，本文旨在推动未来研究和促进人工智能与生物医药工程的跨学科合作。 

---
# Physics-Constrained Fine-Tuning of Flow-Matching Models for Generation and Inverse Problems 

**Title (ZH)**: 基于物理约束的流匹配模型微调方法及其在生成和逆问题中的应用 

**Authors**: Jan Tauberschmidt, Sophie Fellenz, Sebastian J. Vollmer, Andrew B. Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09156)  

**Abstract**: We present a framework for fine-tuning flow-matching generative models to enforce physical constraints and solve inverse problems in scientific systems. Starting from a model trained on low-fidelity or observational data, we apply a differentiable post-training procedure that minimizes weak-form residuals of governing partial differential equations (PDEs), promoting physical consistency and adherence to boundary conditions without distorting the underlying learned distribution. To infer unknown physical inputs, such as source terms, material parameters, or boundary data, we augment the generative process with a learnable latent parameter predictor and propose a joint optimization strategy. The resulting model produces physically valid field solutions alongside plausible estimates of hidden parameters, effectively addressing ill-posed inverse problems in a data-driven yet physicsaware manner. We validate our method on canonical PDE benchmarks, demonstrating improved satisfaction of PDE constraints and accurate recovery of latent coefficients. Our approach bridges generative modelling and scientific inference, opening new avenues for simulation-augmented discovery and data-efficient modelling of physical systems. 

**Abstract (ZH)**: 我们提出了一种 fine-tuning 流匹配生成模型的框架，以施加物理约束并解决科学系统中的逆问题。从低保真度或观测数据训练的模型出发，我们应用了一个可微的后训练程序，最小化控制方程（PDEs）的弱形式残差，促进物理一致性并遵守边界条件而不扭曲底层学习的分布。为推断未知的物理输入，如源项、材料参数或边界数据，我们通过可学习的潜在参数预测器扩展生成过程，并提出了一种联合优化策略。由此产生的模型不仅生成了物理有效的场解，还给出了潜在参数的合理估计，以数据驱动且物理感知的方式有效解决了病态逆问题。我们在经典的 PDE 标准测试上验证了该方法，展示了对 PDE 约束的更好满足以及对潜在系数的准确恢复。我们的方法将生成建模与科学推断相结合，为基于模拟的发现和物理系统的数据高效建模开辟了新途径。 

---
