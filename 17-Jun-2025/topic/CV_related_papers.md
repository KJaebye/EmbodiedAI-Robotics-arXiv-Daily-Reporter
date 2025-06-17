# Strategic Vantage Selection for Learning Viewpoint-Agnostic Manipulation Policies 

**Title (ZH)**: 基于视角无关操作策略的学习选择性优势分析 

**Authors**: Sreevishakh Vasudevan, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2506.12261)  

**Abstract**: Vision-based manipulation has shown remarkable success, achieving promising performance across a range of tasks. However, these manipulation policies often fail to generalize beyond their training viewpoints, which is a persistent challenge in achieving perspective-agnostic manipulation, especially in settings where the camera is expected to move at runtime. Although collecting data from many angles seems a natural solution, such a naive approach is both resource-intensive and degrades manipulation policy performance due to excessive and unstructured visual diversity. This paper proposes Vantage, a framework that systematically identifies and integrates data from optimal perspectives to train robust, viewpoint-agnostic policies. By formulating viewpoint selection as a continuous optimization problem, we iteratively fine-tune policies on a few vantage points. Since we leverage Bayesian optimization to efficiently navigate the infinite space of potential camera configurations, we are able to balance exploration of novel views and exploitation of high-performing ones, thereby ensuring data collection from a minimal number of effective viewpoints. We empirically evaluate this framework on diverse standard manipulation tasks using multiple policy learning methods, demonstrating that fine-tuning with data from strategic camera placements yields substantial performance gains, achieving average improvements of up to 46.19% when compared to fixed, random, or heuristic-based strategies. 

**Abstract (ZH)**: 基于视觉的操控已经在多种任务中取得了显著的成果，但这些操控策略往往难以在训练视角之外进行泛化，这是实现视角无关操控的一个持续性挑战，尤其是在相机在运行时需要移动的场景中。虽然从多个角度收集数据似乎是自然的解决方案，但这种简单的做法既资源密集又会因为过多且无序的视觉多样性而降低操控策略的性能。本文提出了一种名为Vantage的框架，该框架系统地识别并整合来自最优视角的数据，以训练鲁棒的、视角无关的策略。通过将视角选择形式化为一个连续优化问题，我们迭代地在少数关键视角上微调策略。由于我们利用贝叶斯优化高效地探索潜在相机配置的空间，从而能够平衡新视角的探索和高性能视角的利用，从而确保从少量有效的视角收集数据。我们在多种标准操控任务上使用多种策略学习方法进行实证评估，证明了从战略性相机位置收集的数据能够带来显著的性能提升，在与固定、随机或启发式策略相比时，平均提升了46.19%。 

---
# Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning 

**Title (ZH)**: 开集LiDAR全景分割：基于不确定性感知学习 

**Authors**: Rohit Mohan, Julia Hindel, Florian Drews, Claudius Gläser, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2506.13265)  

**Abstract**: Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods. 

**Abstract (ZH)**: 自主导航于开放环境的车辆可能会遇到未见的对象类别。然而，现有的大多数LiDAR全景分割模型依赖于封闭集合假设，无法检测到未知对象实例。在本工作中，我们提出了一种名为ULOPS的不确定性引导的开放集合全景分割框架，该框架利用Dirichlet基的证据学习来建模预测不确定性。我们的架构包括用于语义分割的不确定性估计解码器、嵌入与原型关联以及实例中心预测的独立解码器。在推断过程中，利用不确定性估计识别和分割未知实例。为了增强模型区分已知和未知对象的能力，我们引入了三种基于不确定性的损失函数。均匀证据损失以鼓励未知区域的高不确定性。自适应不确定性分离损失确保在全局尺度上已知和未知对象的不确定性估计之间的一致差异。对比不确定性损失在细粒度级别细化这种分离。为了评估开放集合性能，我们扩展了KITTI-360基准设置，并为nuScenes引入了一种新的开放集合评估。广泛的实验表明，ULOPS在开放集合LiDAR全景分割方面始终优于现有方法。 

---
# SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure 

**Title (ZH)**: SuperPoint-SLAM3: 结合深度特征、自适应非极大值抑制和基于学习的环形闭回路的ORB-SLAM3增强版 

**Authors**: Shahram Najam Syed, Ishir Roongta, Kavin Ravie, Gangadhar Nageswar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13089)  

**Abstract**: Visual simultaneous localization and mapping (SLAM) must remain accurate under extreme viewpoint, scale and illumination variations. The widely adopted ORB-SLAM3 falters in these regimes because it relies on hand-crafted ORB keypoints. We introduce SuperPoint-SLAM3, a drop-in upgrade that (i) replaces ORB with the self-supervised SuperPoint detector--descriptor, (ii) enforces spatially uniform keypoints via adaptive non-maximal suppression (ANMS), and (iii) integrates a lightweight NetVLAD place-recognition head for learning-based loop closure.
On the KITTI Odometry benchmark SuperPoint-SLAM3 reduces mean translational error from 4.15% to 0.34% and mean rotational error from 0.0027 deg/m to 0.0010 deg/m. On the EuRoC MAV dataset it roughly halves both errors across every sequence (e.g., V2\_03: 1.58% -> 0.79%). These gains confirm that fusing modern deep features with a learned loop-closure module markedly improves ORB-SLAM3 accuracy while preserving its real-time operation.
Implementation, pretrained weights and reproducibility scripts are available at this https URL. 

**Abstract (ZH)**: SuperPoint-SLAM3：一种在极端视角、尺度和光照变化下保持准确的同时定位与建图方法 

---
# Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving 

**Title (ZH)**: 端到端驾驶中的高效三平面多摄像头分词 

**Authors**: Boris Ivanovic, Cristiano Saltori, Yurong You, Yan Wang, Wenjie Luo, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.12251)  

**Abstract**: Autoregressive Transformers are increasingly being deployed as end-to-end robot and autonomous vehicle (AV) policy architectures, owing to their scalability and potential to leverage internet-scale pretraining for generalization. Accordingly, tokenizing sensor data efficiently is paramount to ensuring the real-time feasibility of such architectures on embedded hardware. To this end, we present an efficient triplane-based multi-camera tokenization strategy that leverages recent advances in 3D neural reconstruction and rendering to produce sensor tokens that are agnostic to the number of input cameras and their resolution, while explicitly accounting for their geometry around an AV. Experiments on a large-scale AV dataset and state-of-the-art neural simulator demonstrate that our approach yields significant savings over current image patch-based tokenization strategies, producing up to 72% fewer tokens, resulting in up to 50% faster policy inference while achieving the same open-loop motion planning accuracy and improved offroad rates in closed-loop driving simulations. 

**Abstract (ZH)**: 自回归变压器越来越多地被用作机器人和自动驾驶车辆（AV）策略架构的端到端模型，得益于其可扩展性和通过互联网规模预训练进行泛化的潜力。因此，高效地 tokenize 传感器数据对于确保此类架构在嵌入式硬件上的实时可行性至关重要。为此，我们提出了一种基于三平面的多相机 tokenize 策略，利用最近在三维神经重建和渲染方面的进展，生成与输入相机数量和分辨率无关的传感器 tokenize，同时明确考虑其在自动驾驶车辆周围的几何关系。在大规模自动驾驶车辆数据集和最先进的神经模拟器上的实验表明，与当前基于图像块的 tokenize 策略相比，我们的方法可实现显著节约，生成多达 72% 的较少 tokenize，从而在保持开环运动规划精度相同的同时，闭环驾驶模拟表现出更快的策略推理速度和更高的离路率。 

---
# VideoPDE: Unified Generative PDE Solving via Video Inpainting Diffusion Models 

**Title (ZH)**: 视频PDE统一生成型偏微分方程求解方法：基于视频_inpainting_扩散模型 

**Authors**: Edward Li, Zichen Wang, Jiahe Huang, Jeong Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.13754)  

**Abstract**: We present a unified framework for solving partial differential equations (PDEs) using video-inpainting diffusion transformer models. Unlike existing methods that devise specialized strategies for either forward or inverse problems under full or partial observation, our approach unifies these tasks under a single, flexible generative framework. Specifically, we recast PDE-solving as a generalized inpainting problem, e.g., treating forward prediction as inferring missing spatiotemporal information of future states from initial conditions. To this end, we design a transformer-based architecture that conditions on arbitrary patterns of known data to infer missing values across time and space. Our method proposes pixel-space video diffusion models for fine-grained, high-fidelity inpainting and conditioning, while enhancing computational efficiency through hierarchical modeling. Extensive experiments show that our video inpainting-based diffusion model offers an accurate and versatile solution across a wide range of PDEs and problem setups, outperforming state-of-the-art baselines. 

**Abstract (ZH)**: 我们提出了一种统一框架，利用视频修复变换器模型求解偏微分方程（PDEs）。不同于现有方法针对完全或不完全观测下的前向或反向问题设计专门策略，我们的方法将这些任务统一在一个灵活的生成框架下。具体而言，我们将PDE求解重新阐述为一个泛化的修复问题，例如，将前向预测视为从初始条件推断未来状态的缺失时空信息。为此，我们设计了一种基于变换器的架构，可根据任意已知数据模式来推断时空中的缺失值。我们的方法提出了一种像素空间视频扩散模型进行细粒度、高保真修复和条件推断，并通过分层建模提高计算效率。 extensive实验表明，基于视频修复的扩散模型在广泛类型的PDE和问题设置中提供了准确且通用的解决方案，优于最先进的基线方法。 

---
# Contrastive Self-Supervised Learning As Neural Manifold Packing 

**Title (ZH)**: 对比自监督学习作为神经流形打包 

**Authors**: Guanming Zhang, David J. Heeger, Stefano Martiniani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13717)  

**Abstract**: Contrastive self-supervised learning based on point-wise comparisons has been widely studied for vision tasks. In the visual cortex of the brain, neuronal responses to distinct stimulus classes are organized into geometric structures known as neural manifolds. Accurate classification of stimuli can be achieved by effectively separating these manifolds, akin to solving a packing problem. We introduce Contrastive Learning As Manifold Packing (CLAMP), a self-supervised framework that recasts representation learning as a manifold packing problem. CLAMP introduces a loss function inspired by the potential energy of short-range repulsive particle systems, such as those encountered in the physics of simple liquids and jammed packings. In this framework, each class consists of sub-manifolds embedding multiple augmented views of a single image. The sizes and positions of the sub-manifolds are dynamically optimized by following the gradient of a packing loss. This approach yields interpretable dynamics in the embedding space that parallel jamming physics, and introduces geometrically meaningful hyperparameters within the loss function. Under the standard linear evaluation protocol, which freezes the backbone and trains only a linear classifier, CLAMP achieves competitive performance with state-of-the-art self-supervised models. Furthermore, our analysis reveals that neural manifolds corresponding to different categories emerge naturally and are effectively separated in the learned representation space, highlighting the potential of CLAMP to bridge insights from physics, neural science, and machine learning. 

**Abstract (ZH)**: 基于点WISE比较的对比自监督学习在视觉任务中已有广泛研究。我们提出了一种新的自监督框架Contrastive Learning As Manifold Packing (CLAMP)，将表示学习重新定义为流形填充问题。CLAMP 引入了一种损失函数，该损失函数受到短程排斥粒子系统的潜在能量的启发，类似于简单液体和挤实堆积在物理学中的情况。在该框架中，每个类别由嵌入单张图像多种增强视图的子流形组成。子流形的大小和位置通过跟随堆积损失的梯度动态优化。该方法在嵌入空间中产生了与阻塞物理学相平行的可解释动力学，并在损失函数中引入了几何上有意义的超参数。在标准的线性评估协议下，即冻结骨干网络并仅训练线性分类器，CLAMP 达到了与最先进的自监督模型相当的性能。进一步的分析表明，不同的类别对应的神经流形在学习表示空间中自然涌现并得到有效分离，突显了CLAMP 能够在物理学、神经科学和机器学习之间架起桥梁的潜力。 

---
# DualEdit: Dual Editing for Knowledge Updating in Vision-Language Models 

**Title (ZH)**: DualEdit: 双重编辑用于视觉-语言模型的知识更新 

**Authors**: Zhiyi Shi, Binjie Wang, Chongjie Si, Yichen Wu, Junsik Kim, Hanspeter Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2506.13638)  

**Abstract**: Model editing aims to efficiently update a pre-trained model's knowledge without the need for time-consuming full retraining. While existing pioneering editing methods achieve promising results, they primarily focus on editing single-modal language models (LLMs). However, for vision-language models (VLMs), which involve multiple modalities, the role and impact of each modality on editing performance remain largely unexplored. To address this gap, we explore the impact of textual and visual modalities on model editing and find that: (1) textual and visual representations reach peak sensitivity at different layers, reflecting their varying importance; and (2) editing both modalities can efficiently update knowledge, but this comes at the cost of compromising the model's original capabilities. Based on our findings, we propose DualEdit, an editor that modifies both textual and visual modalities at their respective key layers. Additionally, we introduce a gating module within the more sensitive textual modality, allowing DualEdit to efficiently update new knowledge while preserving the model's original information. We evaluate DualEdit across multiple VLM backbones and benchmark datasets, demonstrating its superiority over state-of-the-art VLM editing baselines as well as adapted LLM editing methods on different evaluation metrics. 

**Abstract (ZH)**: 模型编辑旨在高效更新预训练模型的知识，而无需进行耗时的完全重新训练。尽管现有的先驱编辑方法取得了令人鼓舞的结果，它们主要集中在编辑单模态语言模型（LLMs）上。然而，对于涉及多模态的视觉语言模型（VLMs），每个模态在编辑性能中的作用和影响尚未得到充分探索。为解决这一问题，我们研究了文本和视觉模态对模型编辑的影响，并发现：（1）文本和视觉表示在不同的层达到最大敏感性，反映出它们的不同重要性；（2）同时编辑这两个模态可以高效率地更新知识，但会牺牲模型的原始能力。基于我们的发现，我们提出DualEdit，这是一种在各自的关键层修改文本和视觉模态的编辑器。此外，我们还在更敏感的文本模态中引入了一个门控模块，使DualEdit能够在高效更新新知识的同时保留模型的原始信息。我们在多个VLM骨干网络和基准数据集上评估DualEdit，展示了其在不同评估指标上优于最先进的VLM编辑基线以及适应的LLM编辑方法的优越性。 

---
# UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data 

**Title (ZH)**: 基于自定义地理参考数据的采矿工业元宇宙中无人机目标检测与定位 

**Authors**: Vasiliki Balaska, Ioannis Tsampikos Papapetros, Katerina Maria Oikonomou, Loukas Bampis, Antonios Gasteratos  

**Link**: [PDF](https://arxiv.org/pdf/2506.13505)  

**Abstract**: The mining sector increasingly adopts digital tools to improve operational efficiency, safety, and data-driven decision-making. One of the key challenges remains the reliable acquisition of high-resolution, geo-referenced spatial information to support core activities such as extraction planning and on-site monitoring. This work presents an integrated system architecture that combines UAV-based sensing, LiDAR terrain modeling, and deep learning-based object detection to generate spatially accurate information for open-pit mining environments. The proposed pipeline includes geo-referencing, 3D reconstruction, and object localization, enabling structured spatial outputs to be integrated into an industrial digital twin platform. Unlike traditional static surveying methods, the system offers higher coverage and automation potential, with modular components suitable for deployment in real-world industrial contexts. While the current implementation operates in post-flight batch mode, it lays the foundation for real-time extensions. The system contributes to the development of AI-enhanced remote sensing in mining by demonstrating a scalable and field-validated geospatial data workflow that supports situational awareness and infrastructure safety. 

**Abstract (ZH)**: 矿业领域 increasingly采用数字工具以提高运营效率、安全性和数据驱动的决策能力。其中一个关键挑战是可靠地获取高分辨率、地理参考的空间信息，以支持诸如开采规划和现场监控等核心活动。本文提出了一种集成系统架构，结合了基于无人机的传感、LiDAR地形建模以及基于深度学习的对象检测，以生成适用于露天矿业环境的空间精确信息。提出的流程包括地理参考、三维重建和对象定位，使结构化空间输出能够集成到工业数字孪生平台中。与传统的静态测量方法相比，该系统提供了更高的覆盖范围和自动化潜力，并具有模块化组件，适用于实际工业环境的部署。虽然目前的实现方式在飞行后以批处理模式运行，但它为实时扩展奠定了基础。该系统通过展示一种可扩展且已在现场验证过的地理空间数据工作流，支持态势感知和基础设施安全，从而促进了采矿领域的AI增强遥感技术的发展。 

---
# ESRPCB: an Edge guided Super-Resolution model and Ensemble learning for tiny Printed Circuit Board Defect detection 

**Title (ZH)**: ESRPCB：边缘引导的超分辨率模型与集成学习在微小印制电路板缺陷检测中的应用 

**Authors**: Xiem HoangVan, Dang Bui Dinh, Thanh Nguyen Canh, Van-Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13476)  

**Abstract**: Printed Circuit Boards (PCBs) are critical components in modern electronics, which require stringent quality control to ensure proper functionality. However, the detection of defects in small-scale PCBs images poses significant challenges as a result of the low resolution of the captured images, leading to potential confusion between defects and noise. To overcome these challenges, this paper proposes a novel framework, named ESRPCB (edgeguided super-resolution for PCBs defect detection), which combines edgeguided super-resolution with ensemble learning to enhance PCBs defect detection. The framework leverages the edge information to guide the EDSR (Enhanced Deep Super-Resolution) model with a novel ResCat (Residual Concatenation) structure, enabling it to reconstruct high-resolution images from small PCBs inputs. By incorporating edge features, the super-resolution process preserves critical structural details, ensuring that tiny defects remain distinguishable in the enhanced image. Following this, a multi-modal defect detection model employs ensemble learning to analyze the super-resolved 

**Abstract (ZH)**: 基于边缘引导超分辨率的PCB缺陷检测框架（ESRPCB） 

---
# Simple is what you need for efficient and accurate medical image segmentation 

**Title (ZH)**: 简单即为高效准确医疗图像分割所需 

**Authors**: Xiang Yu, Yayan Chen, Guannan He, Qing Zeng, Yue Qin, Meiling Liang, Dandan Luo, Yimei Liao, Zeyu Ren, Cheng Kang, Delong Yang, Bocheng Liang, Bin Pu, Ying Yuan, Shengli Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13415)  

**Abstract**: While modern segmentation models often prioritize performance over practicality, we advocate a design philosophy prioritizing simplicity and efficiency, and attempted high performance segmentation model design. This paper presents SimpleUNet, a scalable ultra-lightweight medical image segmentation model with three key innovations: (1) A partial feature selection mechanism in skip connections for redundancy reduction while enhancing segmentation performance; (2) A fixed-width architecture that prevents exponential parameter growth across network stages; (3) An adaptive feature fusion module achieving enhanced representation with minimal computational overhead. With a record-breaking 16 KB parameter configuration, SimpleUNet outperforms LBUNet and other lightweight benchmarks across multiple public datasets. The 0.67 MB variant achieves superior efficiency (8.60 GFLOPs) and accuracy, attaining a mean DSC/IoU of 85.76%/75.60% on multi-center breast lesion datasets, surpassing both U-Net and TransUNet. Evaluations on skin lesion datasets (ISIC 2017/2018: mDice 84.86%/88.77%) and endoscopic polyp segmentation (KVASIR-SEG: 86.46%/76.48% mDice/mIoU) confirm consistent dominance over state-of-the-art models. This work demonstrates that extreme model compression need not compromise performance, providing new insights for efficient and accurate medical image segmentation. Codes can be found at this https URL. 

**Abstract (ZH)**: 现代分割模型往往重视性能而忽视实用性，我们提倡一种以简洁和高效为优先的设计哲学，并尝试设计高性能的分割模型。本文提出了SimpleUNet，一种具有三大创新的可扩展极轻量级医学图像分割模型：(1) 跳链接中的部分特征选择机制以减少冗余并增强分割性能；(2) 固定宽度架构以防止网络各层参数数量指数级增长；(3) 可适应特征融合模块以实现增强表示并最小化计算开销。通过创纪录的16 KB参数配置，SimpleUNet在多个公开数据集上超越LBUNet和其他轻量级基准模型，展现出优于U-Net和TransUNet的效率和准确性。在皮肤病变数据集（ISIC 2017/2018：mDice 84.86%/88.77%）和内镜息肉分割数据集（KVASIR-SEG：86.46%/76.48% mDice/mIoU）上的评估进一步证实了其在最先进的模型中的持续领先地位。本工作表明，极端模型压缩不必牺牲性能，为高效准确的医学图像分割提供了新的见解。代码可在以下链接找到。 

---
# ViT-NeBLa: A Hybrid Vision Transformer and Neural Beer-Lambert Framework for Single-View 3D Reconstruction of Oral Anatomy from Panoramic Radiographs 

**Title (ZH)**: 基于混合视觉变换器和 Beer-Lambert 神经网络框架的全景放射影像单视角口腔解剖三维重建 

**Authors**: Bikram Keshari Parida, Anusree P. Sunilkumar, Abhijit Sen, Wonsang You  

**Link**: [PDF](https://arxiv.org/pdf/2506.13195)  

**Abstract**: Dental diagnosis relies on two primary imaging modalities: panoramic radiographs (PX) providing 2D oral cavity representations, and Cone-Beam Computed Tomography (CBCT) offering detailed 3D anatomical information. While PX images are cost-effective and accessible, their lack of depth information limits diagnostic accuracy. CBCT addresses this but presents drawbacks including higher costs, increased radiation exposure, and limited accessibility. Existing reconstruction models further complicate the process by requiring CBCT flattening or prior dental arch information, often unavailable clinically. We introduce ViT-NeBLa, a vision transformer-based Neural Beer-Lambert model enabling accurate 3D reconstruction directly from single PX. Our key innovations include: (1) enhancing the NeBLa framework with Vision Transformers for improved reconstruction capabilities without requiring CBCT flattening or prior dental arch information, (2) implementing a novel horseshoe-shaped point sampling strategy with non-intersecting rays that eliminates intermediate density aggregation required by existing models due to intersecting rays, reducing sampling point computations by $52 \%$, (3) replacing CNN-based U-Net with a hybrid ViT-CNN architecture for superior global and local feature extraction, and (4) implementing learnable hash positional encoding for better higher-dimensional representation of 3D sample points compared to existing Fourier-based dense positional encoding. Experiments demonstrate that ViT-NeBLa significantly outperforms prior state-of-the-art methods both quantitatively and qualitatively, offering a cost-effective, radiation-efficient alternative for enhanced dental diagnostics. 

**Abstract (ZH)**: 基于视变压器的NeBLa模型：直接从单张全景牙片实现准确三维重建 

---
# DualFast: Dual-Speedup Framework for Fast Sampling of Diffusion Models 

**Title (ZH)**: DualFast：双重加速框架用于扩散模型的快速采样 

**Authors**: Hu Yu, Hao Luo, Fan Wang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13058)  

**Abstract**: Diffusion probabilistic models (DPMs) have achieved impressive success in visual generation. While, they suffer from slow inference speed due to iterative sampling. Employing fewer sampling steps is an intuitive solution, but this will also introduces discretization error. Existing fast samplers make inspiring efforts to reduce discretization error through the adoption of high-order solvers, potentially reaching a plateau in terms of optimization. This raises the question: can the sampling process be accelerated further? In this paper, we re-examine the nature of sampling errors, discerning that they comprise two distinct elements: the widely recognized discretization error and the less explored approximation error. Our research elucidates the dynamics between these errors and the step by implementing a dual-error disentanglement strategy. Building on these foundations, we introduce an unified and training-free acceleration framework, DualFast, designed to enhance the speed of DPM sampling by concurrently accounting for both error types, thereby minimizing the total sampling error. DualFast is seamlessly compatible with existing samplers and significantly boost their sampling quality and speed, particularly in extremely few sampling steps. We substantiate the effectiveness of our framework through comprehensive experiments, spanning both unconditional and conditional sampling domains, across both pixel-space and latent-space DPMs. 

**Abstract (ZH)**: 扩散概率模型(DPMs)在视觉生成任务中取得了显著的成功，但由于迭代采样过程导致推断速度缓慢。减少采样步骤是一种直观的解决方案，但这也引入了离散化误差。现有的快速采样器通过采用高阶求解器来减少离散化误差，但可能在优化方面达到瓶颈。这引发了进一步的问题：采样过程是否可以进一步加速？在本文中，我们重新审视了采样误差的本质，发现它们由两部分组成：广为人知的离散化误差和较少研究的逼近误差。我们的研究通过实施双误差分离策略阐明了这些误差之间的动态关系。在此基础上，我们提出了一种无需训练且统一的加速框架——DualFast，旨在通过同时考虑两种类型的误差来加速DPM采样的速度，从而最小化总的采样误差。DualFast 无缝兼容现有的采样器，并在极少数采样步骤中显著提升其采样质量和速度。我们通过覆盖无条件和有条件采样领域、像素空间和潜在空间DPM的全面实验验证了该框架的有效性。 

---
# Scene-aware SAR ship detection guided by unsupervised sea-land segmentation 

**Title (ZH)**: 场景 aware SAR 船舶检测，基于无监督海陆分割 

**Authors**: Han Ke, Xiao Ke, Ye Yan, Rui Liu, Jinpeng Yang, Tianwen Zhang, Xu Zhan, Xiaowo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12775)  

**Abstract**: DL based Synthetic Aperture Radar (SAR) ship detection has tremendous advantages in numerous areas. However, it still faces some problems, such as the lack of prior knowledge, which seriously affects detection accuracy. In order to solve this problem, we propose a scene-aware SAR ship detection method based on unsupervised sea-land segmentation. This method follows a classical two-stage framework and is enhanced by two models: the unsupervised land and sea segmentation module (ULSM) and the land attention suppression module (LASM). ULSM and LASM can adaptively guide the network to reduce attention on land according to the type of scenes (inshore scene and offshore scene) and add prior knowledge (sea land segmentation information) to the network, thereby reducing the network's attention to land directly and enhancing offshore detection performance relatively. This increases the accuracy of ship detection and enhances the interpretability of the model. Specifically, in consideration of the lack of land sea segmentation labels in existing deep learning-based SAR ship detection datasets, ULSM uses an unsupervised approach to classify the input data scene into inshore and offshore types and performs sea-land segmentation for inshore scenes. LASM uses the sea-land segmentation information as prior knowledge to reduce the network's attention to land. We conducted our experiments using the publicly available SSDD dataset, which demonstrated the effectiveness of our network. 

**Abstract (ZH)**: 基于DL的合成孔径雷达（SAR）船舶检测在许多领域具有巨大的优势。然而，它仍然面临一些问题，如缺乏先验知识，严重影响了检测精度。为了解决这一问题，我们提出了一种基于无监督海-陆分割的场景感知SAR船舶检测方法。该方法遵循经典的两阶段框架，并通过两个模型进行增强：无监督海-陆分割模块（ULSM）和陆地注意力抑制模块（LASM）。ULSM和LASM可以根据场景类型（近岸场景和远海场景）和先验知识（海-陆分割信息）自适应地引导网络减少对陆地的注意力，从而降低网络对陆地的关注，相对增强远海检测性能，提高船舶检测的准确性并增强模型的可解释性。具体而言，考虑到现有基于深度学习的SAR船舶检测数据集中缺乏海-陆分割标签，ULSM采用无监督方法将输入数据场景分类为近岸和远海类型，并对近岸场景进行海-陆分割。LASM利用海-陆分割信息作为先验知识，减少网络对陆地的注意力。我们使用公开的SSDD数据集进行了实验，验证了我们网络的有效性。 

---
# Unleashing Diffusion and State Space Models for Medical Image Segmentation 

**Title (ZH)**: 释放扩散模型和状态空间模型在医疗影像分割中的潜力 

**Authors**: Rong Wu, Ziqi Chen, Liming Zhong, Heng Li, Hai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12747)  

**Abstract**: Existing segmentation models trained on a single medical imaging dataset often lack robustness when encountering unseen organs or tumors. Developing a robust model capable of identifying rare or novel tumor categories not present during training is crucial for advancing medical imaging applications. We propose DSM, a novel framework that leverages diffusion and state space models to segment unseen tumor categories beyond the training data. DSM utilizes two sets of object queries trained within modified attention decoders to enhance classification accuracy. Initially, the model learns organ queries using an object-aware feature grouping strategy to capture organ-level visual features. It then refines tumor queries by focusing on diffusion-based visual prompts, enabling precise segmentation of previously unseen tumors. Furthermore, we incorporate diffusion-guided feature fusion to improve semantic segmentation performance. By integrating CLIP text embeddings, DSM captures category-sensitive classes to improve linguistic transfer knowledge, thereby enhancing the model's robustness across diverse scenarios and multi-label tasks. Extensive experiments demonstrate the superior performance of DSM in various tumor segmentation tasks. Code is available at this https URL. 

**Abstract (ZH)**: 现有的医学影像数据集训练的分割模型在遇到未见过的器官或肿瘤时往往缺乏鲁棒性。开发一种能够在未见过的罕见或新型肿瘤类别上进行精确识别的鲁棒模型对于医学影像应用的推进至关重要。我们提出了一种新颖的DSM框架，该框架利用扩散模型和状态空间模型来分割超出训练数据的未见过的肿瘤类别。DSM通过在修改后的注意力解码器中训练两组对象查询来增强分类准确性。首先，模型采用对象感知特征分组策略学习器官查询，以捕获器官级的视觉特征。然后，通过聚焦于基于扩散的视觉提示来细化肿瘤查询，从而实现对未见过的肿瘤的精确分割。此外，我们还引入了基于扩散的特征融合以提高语义分割性能。通过集成CLIP文本嵌入，DSM捕获类别敏感的类以增强语言迁移知识，从而提高模型在多种场景和多标签任务中的鲁棒性。广泛的经验表明，DSM在各种肿瘤分割任务中表现出优越的性能。相关代码可在以下链接获取：this https URL。 

---
# SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration 

**Title (ZH)**: SP-VLA：一种联合模型调度和token剪枝的VLA模型加速方法 

**Authors**: Ye Li, Yuan Meng, Zewen Sun, Kangye Ji, Chen Tang, Jiajun Fan, Xinzhu Ma, Shutao Xia, Zhi Wang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12723)  

**Abstract**: Vision-Language-Action (VLA) models have attracted increasing attention for their strong control capabilities. However, their high computational cost and low execution frequency hinder their suitability for real-time tasks such as robotic manipulation and autonomous navigation. Existing VLA acceleration methods primarily focus on structural optimization, overlooking the fact that these models operate in sequential decision-making environments. As a result, temporal redundancy in sequential action generation and spatial redundancy in visual input remain unaddressed. To this end, we propose SP-VLA, a unified framework that accelerates VLA models by jointly scheduling models and pruning tokens. Specifically, we design an action-aware model scheduling mechanism that reduces temporal redundancy by dynamically switching between VLA model and a lightweight generator. Inspired by the human motion pattern of focusing on key decision points while relying on intuition for other actions, we categorize VLA actions into deliberative and intuitive, assigning the former to the VLA model and the latter to the lightweight generator, enabling frequency-adaptive execution through collaborative model scheduling. To address spatial redundancy, we further develop a spatio-semantic dual-aware token pruning method. Tokens are classified into spatial and semantic types and pruned based on their dual-aware importance to accelerate VLA inference. These two mechanisms work jointly to guide the VLA in focusing on critical actions and salient visual information, achieving effective acceleration while maintaining high accuracy. Experimental results demonstrate that our method achieves up to 1.5$\times$ acceleration with less than 3% drop in accuracy, outperforming existing approaches in multiple tasks. 

**Abstract (ZH)**: SP-VLA: 一种联合调度与剪枝的视觉-语言-行动模型加速框架 

---
# MGDFIS: Multi-scale Global-detail Feature Integration Strategy for Small Object Detection 

**Title (ZH)**: 多尺度全局细节特征整合策略用于小目标检测 

**Authors**: Yuxiang Wang, Xuecheng Bai, Boyu Hu, Chuanzhi Xu, Haodong Chen, Vera Chung, Tingxue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12697)  

**Abstract**: Small object detection in UAV imagery is crucial for applications such as search-and-rescue, traffic monitoring, and environmental surveillance, but it is hampered by tiny object size, low signal-to-noise ratios, and limited feature extraction. Existing multi-scale fusion methods help, but add computational burden and blur fine details, making small object detection in cluttered scenes difficult. To overcome these challenges, we propose the Multi-scale Global-detail Feature Integration Strategy (MGDFIS), a unified fusion framework that tightly couples global context with local detail to boost detection performance while maintaining efficiency. MGDFIS comprises three synergistic modules: the FusionLock-TSS Attention Module, which marries token-statistics self-attention with DynamicTanh normalization to highlight spectral and spatial cues at minimal cost; the Global-detail Integration Module, which fuses multi-scale context via directional convolution and parallel attention while preserving subtle shape and texture variations; and the Dynamic Pixel Attention Module, which generates pixel-wise weighting maps to rebalance uneven foreground and background distributions and sharpen responses to true object regions. Extensive experiments on the VisDrone benchmark demonstrate that MGDFIS consistently outperforms state-of-the-art methods across diverse backbone architectures and detection frameworks, achieving superior precision and recall with low inference time. By striking an optimal balance between accuracy and resource usage, MGDFIS provides a practical solution for small-object detection on resource-constrained UAV platforms. 

**Abstract (ZH)**: 多尺度全局细节特征整合策略（MGDFIS）：兼顾效率与性能的统一融合框架 

---
# Comparative Analysis of Deep Learning Strategies for Hypertensive Retinopathy Detection from Fundus Images: From Scratch and Pre-trained Models 

**Title (ZH)**: 从零构建与预训练模型在黄斑糖尿病视网膜病变从眼底图像检测中的深度学习策略比较分析 

**Authors**: Yanqiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12492)  

**Abstract**: This paper presents a comparative analysis of deep learning strategies for detecting hypertensive retinopathy from fundus images, a central task in the HRDC challenge~\cite{qian2025hrdc}. We investigate three distinct approaches: a custom CNN, a suite of pre-trained transformer-based models, and an AutoML solution. Our findings reveal a stark, architecture-dependent response to data augmentation. Augmentation significantly boosts the performance of pure Vision Transformers (ViTs), which we hypothesize is due to their weaker inductive biases, forcing them to learn robust spatial and structural features. Conversely, the same augmentation strategy degrades the performance of hybrid ViT-CNN models, whose stronger, pre-existing biases from the CNN component may be "confused" by the transformations. We show that smaller patch sizes (ViT-B/8) excel on augmented data, enhancing fine-grained detail capture. Furthermore, we demonstrate that a powerful self-supervised model like DINOv2 fails on the original, limited dataset but is "rescued" by augmentation, highlighting the critical need for data diversity to unlock its potential. Preliminary tests with a ViT-Large model show poor performance, underscoring the risk of using overly-capacitive models on specialized, smaller datasets. This work provides critical insights into the interplay between model architecture, data augmentation, and dataset size for medical image classification. 

**Abstract (ZH)**: 本研究探讨了检测高血压视网膜病变的基金照片中深度学习策略的比较分析，这是HRDC挑战~\cite{qian2025hrdc}中的核心任务。我们调查了三种不同的方法：自定义CNN、一系列预训练的变压器模型以及AutoML解决方案。我们的研究发现，数据增强对模型架构高度依赖。增强显著提升了纯视力变换器(ViTs)的性能，我们认为这是由于它们较强的归纳偏置较弱，迫使它们学习稳健的空间和结构特征。相反，相同的增强策略降低了混合ViT-CNN模型的性能，这些模型的较强预存偏置可能因变换被“迷惑”。我们展示了较小的 patch 大小（ViT-B/8）在增强数据上表现出色，增强了细粒度细节的捕获。此外，我们证明了强大的自监督模型DINOv2在原始受限数据集上表现不佳，但在增强后得以“拯救”，突显了数据多样性在解锁其潜力方面的关键作用。初步测试中，ViT-Large模型表现不佳，强调了在专门的小型数据集上使用能力过强模型的风险。本研究提供了关于模型架构、数据增强和数据集大小在医学图像分类中相互作用的关键见解。 

---
# Generalizable Trajectory Prediction via Inverse Reinforcement Learning with Mamba-Graph Architecture 

**Title (ZH)**: 基于Mamba-Graph架构的逆强化学习可迁移轨迹预测 

**Authors**: Wenyun Li, Wenjie Huang, Zejian Deng, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.12474)  

**Abstract**: Accurate driving behavior modeling is fundamental to safe and efficient trajectory prediction, yet remains challenging in complex traffic scenarios. This paper presents a novel Inverse Reinforcement Learning (IRL) framework that captures human-like decision-making by inferring diverse reward functions, enabling robust cross-scenario adaptability. The learned reward function is utilized to maximize the likelihood of output by the encoder-decoder architecture that combines Mamba blocks for efficient long-sequence dependency modeling with graph attention networks to encode spatial interactions among traffic agents. Comprehensive evaluations on urban intersections and roundabouts demonstrate that the proposed method not only outperforms various popular approaches in prediction accuracy but also achieves 2 times higher generalization performance to unseen scenarios compared to other IRL-based method. 

**Abstract (ZH)**: 准确的驾驶行为建模对于复杂交通场景下的安全高效的轨迹预测至关重要，但仍具有挑战性。本文提出了一种新颖的逆强化学习（IRL）框架，通过推断多样的奖励函数来捕捉类似人类的决策过程，从而实现跨场景的鲁棒适应性。所学习的奖励函数被用于最大化结合Mamba块的编码-解码架构的输出概率，该架构利用图注意力网络编码交通代理之间的空间交互，以高效建模长序列依赖关系。在城市交叉口和环岛的全面评估中表明，所提出的方法不仅在预测准确性上超越了各种流行的预测方法，而且在未见过的场景上的泛化性能比其他基于IRL的方法高出两倍。 

---
# MS-UMamba: An Improved Vision Mamba Unet for Fetal Abdominal Medical Image Segmentation 

**Title (ZH)**: MS-UMamba: 一种改进的Vision Mamba Unet胎儿腹部医疗图像分割方法 

**Authors**: Caixu Xu, Junming Wei, Huizhen Chen, Pengchen Liang, Bocheng Liang, Ying Tan, Xintong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.12441)  

**Abstract**: Recently, Mamba-based methods have become popular in medical image segmentation due to their lightweight design and long-range dependency modeling capabilities. However, current segmentation methods frequently encounter challenges in fetal ultrasound images, such as enclosed anatomical structures, blurred boundaries, and small anatomical structures. To address the need for balancing local feature extraction and global context modeling, we propose MS-UMamba, a novel hybrid convolutional-mamba model for fetal ultrasound image segmentation. Specifically, we design a visual state space block integrated with a CNN branch (SS-MCAT-SSM), which leverages Mamba's global modeling strengths and convolutional layers' local representation advantages to enhance feature learning. In addition, we also propose an efficient multi-scale feature fusion module that integrates spatial attention mechanisms, which Integrating feature information from different layers enhances the feature representation ability of the model. Finally, we conduct extensive experiments on a non-public dataset, experimental results demonstrate that MS-UMamba model has excellent performance in segmentation performance. 

**Abstract (ZH)**: 基于Mamba的方法近年来在医学图像分割中变得流行，由于其轻量级设计和长程依赖建模能力。然而，当前的分割方法在胎儿超声图像中经常遇到挑战，如封闭的解剖结构、模糊的边界和小的解剖结构。为了解决局部特征提取和全局上下文建模之间的平衡需求，我们提出了一种新颖的混合卷积-Mamba模型MS-UMamba，适用于胎儿超声图像分割。具体而言，我们设计了一个整合CNN分支的视觉状态空间块（SS-MCAT-SSM），该块利用Mamba的全局建模优势和卷积层的局部表示优势，增强特征学习。此外，我们还提出了一种高效的多尺度特征融合模块，该模块集成了空间注意力机制，以整合不同层的特征信息，增强模型的特征表示能力。最后，我们在一个非公开数据集上进行了广泛的实验，实验结果表明，MS-UMamba模型在分割性能方面表现出色。 

---
# ViSAGe: Video-to-Spatial Audio Generation 

**Title (ZH)**: ViSAGe: 视频到空间音频生成 

**Authors**: Jaeyeon Kim, Heeseung Yun, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12199)  

**Abstract**: Spatial audio is essential for enhancing the immersiveness of audio-visual experiences, yet its production typically demands complex recording systems and specialized expertise. In this work, we address a novel problem of generating first-order ambisonics, a widely used spatial audio format, directly from silent videos. To support this task, we introduce YT-Ambigen, a dataset comprising 102K 5-second YouTube video clips paired with corresponding first-order ambisonics. We also propose new evaluation metrics to assess the spatial aspect of generated audio based on audio energy maps and saliency metrics. Furthermore, we present Video-to-Spatial Audio Generation (ViSAGe), an end-to-end framework that generates first-order ambisonics from silent video frames by leveraging CLIP visual features, autoregressive neural audio codec modeling with both directional and visual guidance. Experimental results demonstrate that ViSAGe produces plausible and coherent first-order ambisonics, outperforming two-stage approaches consisting of video-to-audio generation and audio spatialization. Qualitative examples further illustrate that ViSAGe generates temporally aligned high-quality spatial audio that adapts to viewpoint changes. 

**Abstract (ZH)**: 基于视频生成一阶 ambisonics 的端到端框架：ViSAGe 

---
# MARché: Fast Masked Autoregressive Image Generation with Cache-Aware Attention 

**Title (ZH)**: MARché: 快速掩码自回归图像生成与缓存意识注意 

**Authors**: Chaoyi Jiang, Sungwoo Kim, Lei Gao, Hossein Entezari Zarch, Won Woo Ro, Murali Annavaram  

**Link**: [PDF](https://arxiv.org/pdf/2506.12035)  

**Abstract**: Masked autoregressive (MAR) models unify the strengths of masked and autoregressive generation by predicting tokens in a fixed order using bidirectional attention for image generation. While effective, MAR models suffer from significant computational overhead, as they recompute attention and feed-forward representations for all tokens at every decoding step, despite most tokens remaining semantically stable across steps. We propose a training-free generation framework MARché to address this inefficiency through two key components: cache-aware attention and selective KV refresh. Cache-aware attention partitions tokens into active and cached sets, enabling separate computation paths that allow efficient reuse of previously computed key/value projections without compromising full-context modeling. But a cached token cannot be used indefinitely without recomputation due to the changing contextual information over multiple steps. MARché recognizes this challenge and applies a technique called selective KV refresh. Selective KV refresh identifies contextually relevant tokens based on attention scores from newly generated tokens and updates only those tokens that require recomputation, while preserving image generation quality. MARché significantly reduces redundant computation in MAR without modifying the underlying architecture. Empirically, MARché achieves up to 1.7x speedup with negligible impact on image quality, offering a scalable and broadly applicable solution for efficient masked transformer generation. 

**Abstract (ZH)**: Masked autoregressive (MAR)模型通过使用双向注意力以固定顺序预测 tokens 来统一掩蔽生成和自回归生成的优势，适用于图像生成。尽管有效，MAR模型在每个解码步骤中都会为所有tokens重新计算注意力和 feed-forward 表示，尽管大多数tokens在步骤间保持语义稳定，导致显著的计算开销。我们提出了一种无需训练的生成框架MARché，通过两个关键组件解决这一低效问题：aware 缓存注意力和选择性 KV 刷新。aware 缓存注意力将 tokens 分为活动集和缓存集，启用独立的计算路径，允许高效重用先前计算的 key/value 投影，同时保持全面上下文建模能力。但缓存的 token 由于多步骤中的上下文信息变化，无法无限期使用而无需重新计算。MARché 认识到这一挑战，并应用一种称为选择性 KV 刷新的技术。选择性 KV 刷新根据新生成 tokens 的注意力分数识别上下文相关 tokens，并仅更新需要重新计算的 tokens，同时保持图像生成质量。MARché 在不修改底层架构的情况下显著减少了 MAR 中的冗余计算。实证结果表明，MARché 在图像质量影响可以忽略不计的情况下可实现高达 1.7 倍的速度提升，提供了一种可扩展且广泛适用的高效掩蔽变换器生成解决方案。 

---
# Physics-Informed Neural Networks for Vessel Trajectory Prediction: Learning Time-Discretized Kinematic Dynamics via Finite Differences 

**Title (ZH)**: 基于物理的信息神经网络的血管轨迹预测：通过有限差分学习时间离散化的运动学动力学 

**Authors**: Md Mahbub Alam, Amilcar Soares, José F. Rodrigues-Jr, Gabriel Spadon  

**Link**: [PDF](https://arxiv.org/pdf/2506.12029)  

**Abstract**: Accurate vessel trajectory prediction is crucial for navigational safety, route optimization, traffic management, search and rescue operations, and autonomous navigation. Traditional data-driven models lack real-world physical constraints, leading to forecasts that disobey vessel motion dynamics, such as in scenarios with limited or noisy data where sudden course changes or speed variations occur due to external factors. To address this limitation, we propose a Physics-Informed Neural Network (PINN) approach for trajectory prediction that integrates a streamlined kinematic model for vessel motion into the neural network training process via a first- and second-order, finite difference physics-based loss function. This loss function, discretized using the first-order forward Euler method, Heun's second-order approximation, and refined with a midpoint approximation based on Taylor series expansion, enforces fidelity to fundamental physical principles by penalizing deviations from expected kinematic behavior. We evaluated PINN using real-world AIS datasets that cover diverse maritime conditions and compared it with state-of-the-art models. Our results demonstrate that the proposed method reduces average displacement errors by up to 32% across models and datasets while maintaining physical consistency. These results enhance model reliability and adherence to mission-critical maritime activities, where precision translates into better situational awareness in the oceans. 

**Abstract (ZH)**: 准确的船舶轨迹预测对于导航安全、航线优化、交通管理、搜救行动和自主导航至关重要。传统的数据驱动模型缺乏现实世界物理约束，导致预测结果违背船舶运动动力学，尤其是在数据有限或噪声较大时，由于外部因素导致的航向突变或速度变化场景中表现不佳。为解决这一问题，我们提出了一种物理信息神经网络（PINN）方法，通过引入简化动力学模型将物理约束整合到神经网络训练过程中，使用基于有限差分的零阶和二阶物理损失函数。该损失函数通过一阶向前欧拉方法、Heun二阶逼近方法，并结合泰勒级数展开的中点逼近方法进行离散化，从而通过惩罚与预期动力学行为的偏差，确保模型符合基本的物理原理。我们使用覆盖各种海上条件的真实世界AIS数据集评估了PINN，并将其与最先进的模型进行了比较。结果显示，所提出的方法在模型和数据集上将平均位移误差降低了高达32%，同时保持了物理一致性。这些结果增强了模型的可靠性和对关键海事业务的适应性，精准性在海洋中转化为更好的态势感知能力。 

---
