# GS-SDF: LiDAR-Augmented Gaussian Splatting and Neural SDF for Geometrically Consistent Rendering and Reconstruction 

**Title (ZH)**: GS-SDF: 基于LiDAR增强的高斯点云和神经SDF几何一致性渲染与重建 

**Authors**: Jianheng Liu, Yunfei Wan, Bowen Wang, Chunran Zheng, Jiarong Lin, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10170)  

**Abstract**: Digital twins are fundamental to the development of autonomous driving and embodied artificial intelligence. However, achieving high-granularity surface reconstruction and high-fidelity rendering remains a challenge. Gaussian splatting offers efficient photorealistic rendering but struggles with geometric inconsistencies due to fragmented primitives and sparse observational data in robotics applications. Existing regularization methods, which rely on render-derived constraints, often fail in complex environments. Moreover, effectively integrating sparse LiDAR data with Gaussian splatting remains challenging. We propose a unified LiDAR-visual system that synergizes Gaussian splatting with a neural signed distance field. The accurate LiDAR point clouds enable a trained neural signed distance field to offer a manifold geometry field, This motivates us to offer an SDF-based Gaussian initialization for physically grounded primitive placement and a comprehensive geometric regularization for geometrically consistent rendering and reconstruction. Experiments demonstrate superior reconstruction accuracy and rendering quality across diverse trajectories. To benefit the community, the codes will be released at this https URL. 

**Abstract (ZH)**: 数字孪生是自动驾驶和嵌入式人工智能发展的基石。然而，实现高粒度表面重建和高保真渲染仍然是一个挑战。高斯体绘制提供高效的逼真渲染，但在机器人应用中由于碎片化的基础元素和稀疏的观测数据，难以避免几何不一致。现有依赖渲染衍生约束的正则化方法往往在复杂环境中失效。此外，有效地将稀疏激光雷达数据与高斯体绘制结合仍然是一个挑战。我们提出了一种统一的激光雷达-视觉系统，将高斯体绘制与神经签量距离场相结合。准确的激光雷达点云使得训练好的神经签量距离场能够提供流形几何场，从而促使我们基于签量距离场提供物理基础的高斯初始化，并进行全面的几何正则化，以实现几何一致的渲染和重建。实验表明，该方法在多样化轨迹上的重建精度和渲染质量均表现更优。为了惠及社区，代码将在该地址发布：this https URL。 

---
# Adaptive Anomaly Recovery for Telemanipulation: A Diffusion Model Approach to Vision-Based Tracking 

**Title (ZH)**: 基于视觉跟踪的遥操作异常恢复的自适应方法：扩散模型 Approach 

**Authors**: Haoyang Wang, Haoran Guo, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.09632)  

**Abstract**: Dexterous telemanipulation critically relies on the continuous and stable tracking of the human operator's commands to ensure robust operation. Vison-based tracking methods are widely used but have low stability due to anomalies such as occlusions, inadequate lighting, and loss of sight. Traditional filtering, regression, and interpolation methods are commonly used to compensate for explicit information such as angles and positions. These approaches are restricted to low-dimensional data and often result in information loss compared to the original high-dimensional image and video data. Recent advances in diffusion-based approaches, which can operate on high-dimensional data, have achieved remarkable success in video reconstruction and generation. However, these methods have not been fully explored in continuous control tasks in robotics. This work introduces the Diffusion-Enhanced Telemanipulation (DET) framework, which incorporates the Frame-Difference Detection (FDD) technique to identify and segment anomalies in video streams. These anomalous clips are replaced after reconstruction using diffusion models, ensuring robust telemanipulation performance under challenging visual conditions. We validated this approach in various anomaly scenarios and compared it with the baseline methods. Experiments show that DET achieves an average RMSE reduction of 17.2% compared to the cubic spline and 51.1% compared to FFT-based interpolation for different occlusion durations. 

**Abstract (ZH)**: 基于扩散增强的 Dexterous 电信操作 

---
# OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions 

**Title (ZH)**: OSMa-Bench：在不同光照条件下评估开放语义映射 

**Authors**: Maxim Popov, Regina Kurkova, Mikhail Iumanov, Jaafar Mahmoud, Sergey Kolyubin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10331)  

**Abstract**: Open Semantic Mapping (OSM) is a key technology in robotic perception, combining semantic segmentation and SLAM techniques. This paper introduces a dynamically configurable and highly automated LLM/LVLM-powered pipeline for evaluating OSM solutions called OSMa-Bench (Open Semantic Mapping Benchmark). The study focuses on evaluating state-of-the-art semantic mapping algorithms under varying indoor lighting conditions, a critical challenge in indoor environments. We introduce a novel dataset with simulated RGB-D sequences and ground truth 3D reconstructions, facilitating the rigorous analysis of mapping performance across different lighting conditions. Through experiments on leading models such as ConceptGraphs, BBQ and OpenScene, we evaluate the semantic fidelity of object recognition and segmentation. Additionally, we introduce a Scene Graph evaluation method to analyze the ability of models to interpret semantic structure. The results provide insights into the robustness of these models, forming future research directions for developing resilient and adaptable robotic systems. Our code is available at this https URL. 

**Abstract (ZH)**: Open Semantic Mapping (OSM)是机器人感知中的关键技术，结合了语义分割和SLAM技术。本文介绍了一种基于LLM/LVLM的动态可配置且高度自动化的管线OSMa-Bench（Open Semantic MappingBenchmark），用于评估OSM解决方案。该研究重点评估了在不同室内光照条件下的前沿语义映射算法，这是室内环境中的一项关键挑战。我们引入了一个新的包含模拟RGB-D序列及其地面 truth 3D重建的数据集，便于对不同光照条件下的映射性能进行严格的分析。通过在ConceptGraphs、BBQ和OpenScene等领先模型上进行实验，我们评估了物体识别和分割的语义精度。此外，我们引入了一种场景图评估方法，以分析模型理解语义结构的能力。结果提供了这些模型鲁棒性的见解，为开发稳健且适应性强的机器人系统的未来研究指明了方向。相关代码可在此处获取。 

---
# Post-disaster building indoor damage and survivor detection using autonomous path planning and deep learning with unmanned aerial vehicles 

**Title (ZH)**: 灾后建筑物室内损坏与幸存者检测利用自主路径规划和无人机深度学习 

**Authors**: Xiao Pan, Sina Tavasoli, T. Y. Yang, Sina Poorghasem  

**Link**: [PDF](https://arxiv.org/pdf/2503.10027)  

**Abstract**: Rapid response to natural disasters such as earthquakes is a crucial element in ensuring the safety of civil infrastructures and minimizing casualties. Traditional manual inspection is labour-intensive, time-consuming, and can be dangerous for inspectors and rescue workers. This paper proposed an autonomous inspection approach for structural damage inspection and survivor detection in the post-disaster building indoor scenario, which incorporates an autonomous navigation method, deep learning-based damage and survivor detection method, and a customized low-cost micro aerial vehicle (MAV) with onboard sensors. Experimental studies in a pseudo-post-disaster office building have shown the proposed methodology can achieve high accuracy in structural damage inspection and survivor detection. Overall, the proposed inspection approach shows great potential to improve the efficiency of existing manual post-disaster building inspection. 

**Abstract (ZH)**: 自然灾害如地震的快速响应是确保民用基础设施安全和减少人员伤亡的关键要素。传统的手动检查劳动密集、耗时且对检查员和救援人员可能存在危险。本文提出了一种在灾后建筑室内场景中用于结构损伤检查和幸存者检测的自主巡检方法，该方法结合了自主导航方法、基于深度学习的损伤和幸存者检测方法以及搭载传感器的定制低成本微型飞行器（MAV）。在伪灾后办公室建筑的实验研究中展示了所提出的 methodology 在结构损伤检查和幸存者检测方面可以实现高精度。总体而言，所提出的巡检方法显示出显著提高现有灾后建筑巡检效率的潜力。 

---
# PanoGen++: Domain-Adapted Text-Guided Panoramic Environment Generation for Vision-and-Language Navigation 

**Title (ZH)**: PanoGen++：领域适配的文本引导全景环境生成在视觉语言导航中的应用 

**Authors**: Sen Wang, Dongliang Zhou, Liang Xie, Chao Xu, Ye Yan, Erwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.09938)  

**Abstract**: Vision-and-language navigation (VLN) tasks require agents to navigate three-dimensional environments guided by natural language instructions, offering substantial potential for diverse applications. However, the scarcity of training data impedes progress in this field. This paper introduces PanoGen++, a novel framework that addresses this limitation by generating varied and pertinent panoramic environments for VLN tasks. PanoGen++ incorporates pre-trained diffusion models with domain-specific fine-tuning, employing parameter-efficient techniques such as low-rank adaptation to minimize computational costs. We investigate two settings for environment generation: masked image inpainting and recursive image outpainting. The former maximizes novel environment creation by inpainting masked regions based on textual descriptions, while the latter facilitates agents' learning of spatial relationships within panoramas. Empirical evaluations on room-to-room (R2R), room-for-room (R4R), and cooperative vision-and-dialog navigation (CVDN) datasets reveal significant performance enhancements: a 2.44% increase in success rate on the R2R test leaderboard, a 0.63% improvement on the R4R validation unseen set, and a 0.75-meter enhancement in goal progress on the CVDN validation unseen set. PanoGen++ augments the diversity and relevance of training environments, resulting in improved generalization and efficacy in VLN tasks. 

**Abstract (ZH)**: 全景生成增强框架：面向多模态导航任务的多样化和相关性全景环境生成 

---
# CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation 

**Title (ZH)**: CleverDistiller: 简洁且空间一致的跨模态蒸馏 

**Authors**: Hariprasath Govindarajan, Maciej K. Wozniak, Marvin Klingner, Camille Maurice, B Ravi Kiran, Senthil Yogamani  

**Link**: [PDF](https://arxiv.org/pdf/2503.09878)  

**Abstract**: Vision foundation models (VFMs) such as DINO have led to a paradigm shift in 2D camera-based perception towards extracting generalized features to support many downstream tasks. Recent works introduce self-supervised cross-modal knowledge distillation (KD) as a way to transfer these powerful generalization capabilities into 3D LiDAR-based models. However, they either rely on highly complex distillation losses, pseudo-semantic maps, or limit KD to features useful for semantic segmentation only. In this work, we propose CleverDistiller, a self-supervised, cross-modal 2D-to-3D KD framework introducing a set of simple yet effective design choices: Unlike contrastive approaches relying on complex loss design choices, our method employs a direct feature similarity loss in combination with a multi layer perceptron (MLP) projection head to allow the 3D network to learn complex semantic dependencies throughout the projection. Crucially, our approach does not depend on pseudo-semantic maps, allowing for direct knowledge transfer from a VFM without explicit semantic supervision. Additionally, we introduce the auxiliary self-supervised spatial task of occupancy prediction to enhance the semantic knowledge, obtained from a VFM through KD, with 3D spatial reasoning capabilities. Experiments on standard autonomous driving benchmarks for 2D-to-3D KD demonstrate that CleverDistiller achieves state-of-the-art performance in both semantic segmentation and 3D object detection (3DOD) by up to 10% mIoU, especially when fine tuning on really low data amounts, showing the effectiveness of our simple yet powerful KD strategy 

**Abstract (ZH)**: 基于视觉的自监督跨模态知识蒸馏：CleverDistiller在2D到3D知识蒸馏中的应用 

---
# Studying Classifier(-Free) Guidance From a Classifier-Centric Perspective 

**Title (ZH)**: 从分类器为中心的角度研究无分类器引导 

**Authors**: Xiaoming Zhao, Alexander G. Schwing  

**Link**: [PDF](https://arxiv.org/pdf/2503.10638)  

**Abstract**: Classifier-free guidance has become a staple for conditional generation with denoising diffusion models. However, a comprehensive understanding of classifier-free guidance is still missing. In this work, we carry out an empirical study to provide a fresh perspective on classifier-free guidance. Concretely, instead of solely focusing on classifier-free guidance, we trace back to the root, i.e., classifier guidance, pinpoint the key assumption for the derivation, and conduct a systematic study to understand the role of the classifier. We find that both classifier guidance and classifier-free guidance achieve conditional generation by pushing the denoising diffusion trajectories away from decision boundaries, i.e., areas where conditional information is usually entangled and is hard to learn. Based on this classifier-centric understanding, we propose a generic postprocessing step built upon flow-matching to shrink the gap between the learned distribution for a pre-trained denoising diffusion model and the real data distribution, majorly around the decision boundaries. Experiments on various datasets verify the effectiveness of the proposed approach. 

**Abstract (ZH)**: 无分类指导已成为去噪扩散模型条件生成的标准方法，然而对其全面理解仍不足。本文通过实证研究，从无分类指导追溯到分类指导，剖析核心假设，系统研究分类器的作用。我们发现，无论是在分类指导还是无分类指导中，条件生成都是通过将去噪扩散轨迹远离决策边界（即条件信息通常交织且难以学习的区域）来实现的。基于这种以分类器为中心的理解，我们提出了一种基于流匹配的通用后处理步骤，以缩小预训练去噪扩散模型学习的分布与真实数据分布之间的差距，尤其是在决策边界附近。在各种数据集上的实验验证了所提出方法的有效性。 

---
# LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds 

**Title (ZH)**: 秒级从单张图像构建大容量可动画的人体重建模型 

**Authors**: Lingteng Qiu, Xiaodong Gu, Peihao Li, Qi Zuo, Weichao Shen, Junfei Zhang, Kejie Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, Liefeng Bo  

**Link**: [PDF](https://arxiv.org/pdf/2503.10625)  

**Abstract**: Animatable 3D human reconstruction from a single image is a challenging problem due to the ambiguity in decoupling geometry, appearance, and deformation. Recent advances in 3D human reconstruction mainly focus on static human modeling, and the reliance of using synthetic 3D scans for training limits their generalization ability. Conversely, optimization-based video methods achieve higher fidelity but demand controlled capture conditions and computationally intensive refinement processes. Motivated by the emergence of large reconstruction models for efficient static reconstruction, we propose LHM (Large Animatable Human Reconstruction Model) to infer high-fidelity avatars represented as 3D Gaussian splatting in a feed-forward pass. Our model leverages a multimodal transformer architecture to effectively encode the human body positional features and image features with attention mechanism, enabling detailed preservation of clothing geometry and texture. To further boost the face identity preservation and fine detail recovery, we propose a head feature pyramid encoding scheme to aggregate multi-scale features of the head regions. Extensive experiments demonstrate that our LHM generates plausible animatable human in seconds without post-processing for face and hands, outperforming existing methods in both reconstruction accuracy and generalization ability. 

**Abstract (ZH)**: 单张图像中可动画化三维人体重建是一个具有挑战性的问题，因为难以区分几何结构、外观和变形的不确定性。近期在三维人体重建方面的进展主要集中在静态人体建模上，使用合成三维扫描进行训练限制了其泛化能力。相反，基于优化的视频方法实现了更高的保真度，但需要受控的捕捉条件和计算密集型的细化过程。受高效静态重建的大规模重建模型的启发，我们提出了LHM（大规模可动画化人体重建模型），在前向传递过程中推断高保真度以3D高斯点表示的avatar。我们的模型利用多模态变压器架构，通过注意机制有效地编码人体位置特征和图像特征，从而实现详细的服装几何结构和纹理保留。为了进一步提升面部身份保留和细部恢复，我们提出了头部特征金字塔编码方案，以聚合头部区域的多尺度特征。广泛实验表明，我们的LHM在无需面部和手部后处理的情况下，以秒为单位生成合乎情理的可动画化人体，在重建精度和泛化能力方面优于现有方法。 

---
# ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness 

**Title (ZH)**: ETCH: 将体形拟合推广到穿着衣服的人 via 协变紧致度 

**Authors**: Boqian Li, Haiwen Feng, Zeyu Cai, Michael J. Black, Yuliang Xiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10624)  

**Abstract**: Fitting a body to a 3D clothed human point cloud is a common yet challenging task. Traditional optimization-based approaches use multi-stage pipelines that are sensitive to pose initialization, while recent learning-based methods often struggle with generalization across diverse poses and garment types. We propose Equivariant Tightness Fitting for Clothed Humans, or ETCH, a novel pipeline that estimates cloth-to-body surface mapping through locally approximate SE(3) equivariance, encoding tightness as displacement vectors from the cloth surface to the underlying body. Following this mapping, pose-invariant body features regress sparse body markers, simplifying clothed human fitting into an inner-body marker fitting task. Extensive experiments on CAPE and 4D-Dress show that ETCH significantly outperforms state-of-the-art methods -- both tightness-agnostic and tightness-aware -- in body fitting accuracy on loose clothing (16.7% ~ 69.5%) and shape accuracy (average 49.9%). Our equivariant tightness design can even reduce directional errors by (67.2% ~ 89.8%) in one-shot (or out-of-distribution) settings. Qualitative results demonstrate strong generalization of ETCH, regardless of challenging poses, unseen shapes, loose clothing, and non-rigid dynamics. We will release the code and models soon for research purposes at this https URL. 

**Abstract (ZH)**: 基于局部近似SE(3)等变性的Clothed Human紧致 fitting方法：ETCH 

---
# Lightweight Models for Emotional Analysis in Video 

**Title (ZH)**: 轻量级模型在视频情绪分析中的应用 

**Authors**: Quoc-Tien Nguyen, Hong-Hai Nguyen, Van-Thong Huynh  

**Link**: [PDF](https://arxiv.org/pdf/2503.10530)  

**Abstract**: In this study, we present an approach for efficient spatiotemporal feature extraction using MobileNetV4 and a multi-scale 3D MLP-Mixer-based temporal aggregation module. MobileNetV4, with its Universal Inverted Bottleneck (UIB) blocks, serves as the backbone for extracting hierarchical feature representations from input image sequences, ensuring both computational efficiency and rich semantic encoding. To capture temporal dependencies, we introduce a three-level MLP-Mixer module, which processes spatial features at multiple resolutions while maintaining structural integrity. Experimental results on the ABAW 8th competition demonstrate the effectiveness of our approach, showing promising performance in affective behavior analysis. By integrating an efficient vision backbone with a structured temporal modeling mechanism, the proposed framework achieves a balance between computational efficiency and predictive accuracy, making it well-suited for real-time applications in mobile and embedded computing environments. 

**Abstract (ZH)**: 本研究提出了一种使用MobileNetV4和多尺度3D MLP-Mixer基Temporal Aggregation模块的有效时空特征提取方法。通过Universal Inverted Bottleneck (UIB)块，MobileNetV4确保了计算效率和丰富的语义编码。为了捕获时间依赖性，我们引入了一个三级MLP-Mixer模块，在保持结构完整性的同时处理多分辨率的空间特征。实验结果表明，该方法在情绪行为分析方面具有良好的性能。通过结合高效的视觉主干网络和结构化的时间建模机制，所提出的框架在保持计算效率和预测准确性方面达到平衡，使其适用于移动和嵌入式计算环境中的实时应用。 

---
# RealGeneral: Unifying Visual Generation via Temporal In-Context Learning with Video Models 

**Title (ZH)**: RealGeneral: 通过视频模型的时序内上下文学习统一视觉生成 

**Authors**: Yijing Lin, Mengqi Huang, Shuhan Zhuang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10406)  

**Abstract**: Unifying diverse image generation tasks within a single framework remains a fundamental challenge in visual generation. While large language models (LLMs) achieve unification through task-agnostic data and generation, existing visual generation models fail to meet these principles. Current approaches either rely on per-task datasets and large-scale training or adapt pre-trained image models with task-specific modifications, limiting their generalizability. In this work, we explore video models as a foundation for unified image generation, leveraging their inherent ability to model temporal correlations. We introduce RealGeneral, a novel framework that reformulates image generation as a conditional frame prediction task, analogous to in-context learning in LLMs. To bridge the gap between video models and condition-image pairs, we propose (1) a Unified Conditional Embedding module for multi-modal alignment and (2) a Unified Stream DiT Block with decoupled adaptive LayerNorm and attention mask to mitigate cross-modal interference. RealGeneral demonstrates effectiveness in multiple important visual generation tasks, e.g., it achieves a 14.5% improvement in subject similarity for customized generation and a 10% enhancement in image quality for canny-to-image task. Project page: this https URL 

**Abstract (ZH)**: 将多样的图像生成任务统一在一个框架内仍然是视觉生成中的一个基础挑战。虽然大型语言模型（LLMs）通过任务无关的数据和生成实现统一，但现有的视觉生成模型未能遵循这些原则。当前的方法要么依赖于每个任务的数据集和大规模训练，要么通过特定任务的修改预训练图像模型，这限制了它们的泛化能力。在本工作中，我们探索使用视频模型作为统一图像生成的基础，利用其固有的建模时间相关性的能力。我们引入了RealGeneral，一种新颖的框架，将图像生成重新表述为条件帧预测任务，类似于LLMs中的上下文无关学习。为了弥合视频模型与条件图像配对之间的差距，我们提出了一种统一的条件嵌入模块进行多模态对齐，以及一种统一的Stream DiT块，具有解耦的自适应层规范和注意掩码，以减轻跨模态干扰。RealGeneral在重要的视觉生成任务中显示出有效性，例如，在定制生成中实现了14.5%的主题相似性提升，在Canny-to-image任务中实现了10%的图像质量增强。项目页面: [this](this https URL) 

---
# Object detection characteristics in a learning factory environment using YOLOv8 

**Title (ZH)**: 使用YOLOv8在学习工厂环境中进行目标检测的特点 

**Authors**: Toni Schneidereit, Stefan Gohrenz, Michael Breuß  

**Link**: [PDF](https://arxiv.org/pdf/2503.10356)  

**Abstract**: AI-based object detection, and efforts to explain and investigate their characteristics, is a topic of high interest. The impact of, e.g., complex background structures with similar appearances as the objects of interest, on the detection accuracy and, beforehand, the necessary dataset composition are topics of ongoing research. In this paper, we present a systematic investigation of background influences and different features of the object to be detected. The latter includes various materials and surfaces, partially transparent and with shiny reflections in the context of an Industry 4.0 learning factory. Different YOLOv8 models have been trained for each of the materials on different sized datasets, where the appearance was the only changing parameter. In the end, similar characteristics tend to show different behaviours and sometimes unexpected results. While some background components tend to be detected, others with the same features are not part of the detection. Additionally, some more precise conclusions can be drawn from the results. Therefore, we contribute a challenging dataset with detailed investigations on 92 trained YOLO models, addressing some issues on the detection accuracy and possible overfitting. 

**Abstract (ZH)**: 基于AI的物体检测及其背景影响和特征研究是一个高度关注的课题。复杂背景结构与目标物体相似的外观对其检测准确性和事先所需的数据库构成的影响是持续的研究主题。本文系统地研究了背景影响和要检测物体的不同特征，包括工业4.0学习工厂背景下各种材料和表面，部分透明且带有光泽反射。针对每种材料分别训练了不同规模数据集的YOLOv8模型，其中仅改变外观参数。最终，相似特征表现出不同的行为，有时甚至是出人意料的结果。虽然某些背景成分会被检测到，但具有相同特征的其他成分则不会被包含在内。此外，从结果中还可以得出一些更精确的结论。因此，本文贡献了一个具有详尽研究的挑战性数据集，涉及92个训练好的YOLO模型，解决了检测准确性和可能出现的过拟合问题。 

---
# Deep Learning-Based Direct Leaf Area Estimation using Two RGBD Datasets for Model Development 

**Title (ZH)**: 基于深度学习的直接叶片面积估计：使用两种RGBD数据集构建模型 

**Authors**: Namal Jayasuriya, Yi Guo, Wen Hu, Oula Ghannoum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10129)  

**Abstract**: Estimation of a single leaf area can be a measure of crop growth and a phenotypic trait to breed new varieties. It has also been used to measure leaf area index and total leaf area. Some studies have used hand-held cameras, image processing 3D reconstruction and unsupervised learning-based methods to estimate the leaf area in plant images. Deep learning works well for object detection and segmentation tasks; however, direct area estimation of objects has not been explored. This work investigates deep learning-based leaf area estimation, for RGBD images taken using a mobile camera setup in real-world scenarios. A dataset for attached leaves captured with a top angle view and a dataset for detached single leaves were collected for model development and testing. First, image processing-based area estimation was tested on manually segmented leaves. Then a Mask R-CNN-based model was investigated, and modified to accept RGBD images and to estimate the leaf area. The detached-leaf data set was then mixed with the attached-leaf plant data set to estimate the single leaf area for plant images, and another network design with two backbones was proposed: one for segmentation and the other for area estimation. Instead of trying all possibilities or random values, an agile approach was used in hyperparameter tuning. The final model was cross-validated with 5-folds and tested with two unseen datasets: detached and attached leaves. The F1 score with 90% IoA for segmentation result on unseen detached-leaf data was 1.0, while R-squared of area estimation was 0.81. For unseen plant data segmentation, the F1 score with 90% IoA was 0.59, while the R-squared score was 0.57. The research suggests using attached leaves with ground truth area to improve the results. 

**Abstract (ZH)**: 基于深度学习的RGBD图像中单片叶面积 estimation 

---
# MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation 

**Title (ZH)**: MoFlow: 通过隐含最大似然估计蒸馏的一步流匹配人类轨迹预测 

**Authors**: Yuxiang Fu, Qi Yan, Lele Wang, Ke Li, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09950)  

**Abstract**: In this paper, we address the problem of human trajectory forecasting, which aims to predict the inherently multi-modal future movements of humans based on their past trajectories and other contextual cues. We propose a novel motion prediction conditional flow matching model, termed MoFlow, to predict K-shot future trajectories for all agents in a given scene. We design a novel flow matching loss function that not only ensures at least one of the $K$ sets of future trajectories is accurate but also encourages all $K$ sets of future trajectories to be diverse and plausible. Furthermore, by leveraging the implicit maximum likelihood estimation (IMLE), we propose a novel distillation method for flow models that only requires samples from the teacher model. Extensive experiments on the real-world datasets, including SportVU NBA games, ETH-UCY, and SDD, demonstrate that both our teacher flow model and the IMLE-distilled student model achieve state-of-the-art performance. These models can generate diverse trajectories that are physically and socially plausible. Moreover, our one-step student model is $\textbf{100}$ times faster than the teacher flow model during sampling. The code, model, and data are available at our project page: this https URL 

**Abstract (ZH)**: 本文探讨了基于人类过往轨迹和其他上下文线索预测人类固有的多模态未来运动的问题。提出了一种新颖的运动预测条件流匹配模型MoFlow，用于预测给定场景中所有代理的K-shot未来轨迹。设计了一种新颖的流匹配损失函数，不仅确保至少有一组未来的轨迹准确，还鼓励所有K组未来的轨迹具有多样性和合理性。此外，通过利用隐式最大似然估计(IMLE)，提出了一种仅需教师模型样本的新型流模型蒸馏方法。在包含SportVU NBA比赛、ETH-UCY和SDD的真实世界数据集的大量实验中，证明了我们的教师流模型和IMLE蒸馏的学生模型均达到了最新性能。这些模型可以生成物理上和社会上合理的多样化轨迹。此外，我们的单步学生模型在抽样时比教师流模型快100倍。代码、模型和数据可在我们的项目页面获得：this https URL 

---
# TGP: Two-modal occupancy prediction with 3D Gaussian and sparse points for 3D Environment Awareness 

**Title (ZH)**: TGP: 基于3D高斯分布和稀疏点的两模态占用预测方法以提升三维环境意识 

**Authors**: Mu Chen, Wenyu Chen, Mingchuan Yang, Yuan Zhang, Tao Han, Xinchi Li, Yunlong Li, Huaici Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09941)  

**Abstract**: 3D semantic occupancy has rapidly become a research focus in the fields of robotics and autonomous driving environment perception due to its ability to provide more realistic geometric perception and its closer integration with downstream tasks. By performing occupancy prediction of the 3D space in the environment, the ability and robustness of scene understanding can be effectively improved. However, existing occupancy prediction tasks are primarily modeled using voxel or point cloud-based approaches: voxel-based network structures often suffer from the loss of spatial information due to the voxelization process, while point cloud-based methods, although better at retaining spatial location information, face limitations in representing volumetric structural details. To address this issue, we propose a dual-modal prediction method based on 3D Gaussian sets and sparse points, which balances both spatial location and volumetric structural information, achieving higher accuracy in semantic occupancy prediction. Specifically, our method adopts a Transformer-based architecture, taking 3D Gaussian sets, sparse points, and queries as inputs. Through the multi-layer structure of the Transformer, the enhanced queries and 3D Gaussian sets jointly contribute to the semantic occupancy prediction, and an adaptive fusion mechanism integrates the semantic outputs of both modalities to generate the final prediction results. Additionally, to further improve accuracy, we dynamically refine the point cloud at each layer, allowing for more precise location information during occupancy prediction. We conducted experiments on the Occ3DnuScenes dataset, and the experimental results demonstrate superior performance of the proposed method on IoU based metrics. 

**Abstract (ZH)**: 3D语义 occupancy 双模态预测：基于3D高斯集合和稀疏点的方法 

---
# SeqSAM: Autoregressive Multiple Hypothesis Prediction for Medical Image Segmentation using SAM 

**Title (ZH)**: SeqSAM：基于SAM的自回归多假设医学图像分割预测 

**Authors**: Benjamin Towle, Xin Chen, Ke Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.09797)  

**Abstract**: Pre-trained segmentation models are a powerful and flexible tool for segmenting images. Recently, this trend has extended to medical imaging. Yet, often these methods only produce a single prediction for a given image, neglecting inherent uncertainty in medical images, due to unclear object boundaries and errors caused by the annotation tool. Multiple Choice Learning is a technique for generating multiple masks, through multiple learned prediction heads. However, this cannot readily be extended to producing more outputs than its initial pre-training hyperparameters, as the sparse, winner-takes-all loss function makes it easy for one prediction head to become overly dominant, thus not guaranteeing the clinical relevancy of each mask produced. We introduce SeqSAM, a sequential, RNN-inspired approach to generating multiple masks, which uses a bipartite matching loss for ensuring the clinical relevancy of each mask, and can produce an arbitrary number of masks. We show notable improvements in quality of each mask produced across two publicly available datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 预训练分割模型是图像分割的强大且灵活的工具。近年来，这一趋势已扩展到医学成像。然而，这些方法通常只为给定图像生成一个预测，忽略了医学图像中固有的不确定性，由于物体边界不清楚和标注工具引起的误差。多项选择学习是一种通过多个学习预测头生成多个掩码的技术。然而，这不能轻易扩展为生成超过初始预训练超参数的更多输出，因为稀疏的、赢家通吃的损失函数会使一个预测头变得过于主导，从而不能保证每个生成的掩码的临床相关性。我们引入了SeqSAM，这是一种受循环神经网络启发的生成多个掩码的顺序方法，使用双部分匹配损失来确保每个掩码的临床相关性，并可以生成任意数量的掩码。我们在两个公开的数据集上展示了每个生成的掩码的质量上有显著改进。我们的代码可在以下链接获取。 

---
# Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models 

**Title (ZH)**: 无声品牌攻击：无需触发的数据中毒攻击针对文本到图像扩散模型 

**Authors**: Sangwon Jang, June Suk Choi, Jaehyeong Jo, Kimin Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09669)  

**Abstract**: Text-to-image diffusion models have achieved remarkable success in generating high-quality contents from text prompts. However, their reliance on publicly available data and the growing trend of data sharing for fine-tuning make these models particularly vulnerable to data poisoning attacks. In this work, we introduce the Silent Branding Attack, a novel data poisoning method that manipulates text-to-image diffusion models to generate images containing specific brand logos or symbols without any text triggers. We find that when certain visual patterns are repeatedly in the training data, the model learns to reproduce them naturally in its outputs, even without prompt mentions. Leveraging this, we develop an automated data poisoning algorithm that unobtrusively injects logos into original images, ensuring they blend naturally and remain undetected. Models trained on this poisoned dataset generate images containing logos without degrading image quality or text alignment. We experimentally validate our silent branding attack across two realistic settings on large-scale high-quality image datasets and style personalization datasets, achieving high success rates even without a specific text trigger. Human evaluation and quantitative metrics including logo detection show that our method can stealthily embed logos. 

**Abstract (ZH)**: 基于文本的图像扩散模型已在从文本提示生成高质量内容方面取得了显著成功。然而，它们对公开可用数据的依赖以及数据共享以进行微调的趋势使其特别容易受到数据中毒攻击。本文中，我们介绍了无声品牌攻击，这是一种新型数据中毒方法，可以操纵基于文本的图像扩散模型生成包含特定品牌标志或符号的图像，而不使用任何文本触发器。我们发现，当某些视觉模式在训练数据中反复出现时，模型会自然地在输出中复制这些模式，即使没有提示提及。利用这一点，我们开发了一种自动数据中毒算法，能够不显眼地将标志注入原始图像，确保它们自然融合并保持不被察觉。基于此中毒数据集训练的模型能够生成包含标志的图像，而不降低图像质量或文本对齐。我们通过在大规模高质量图像数据集和个人风格定制数据集上的两个现实场景中实验验证了我们的无声品牌攻击，即使没有特定的文本触发器，也取得了 high success rates。人类评估和包括标志检测在内的定量指标表明，我们的方法能够隐秘地嵌入标志。 

---
# Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k 

**Title (ZH)**: Open-Sora 2.0: 在200万美元预算内训练商业级视频生成模型 

**Authors**: Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2503.09642)  

**Abstract**: Video generation models have achieved remarkable progress in the past year. The quality of AI video continues to improve, but at the cost of larger model size, increased data quantity, and greater demand for training compute. In this report, we present Open-Sora 2.0, a commercial-level video generation model trained for only $200k. With this model, we demonstrate that the cost of training a top-performing video generation model is highly controllable. We detail all techniques that contribute to this efficiency breakthrough, including data curation, model architecture, training strategy, and system optimization. According to human evaluation results and VBench scores, Open-Sora 2.0 is comparable to global leading video generation models including the open-source HunyuanVideo and the closed-source Runway Gen-3 Alpha. By making Open-Sora 2.0 fully open-source, we aim to democratize access to advanced video generation technology, fostering broader innovation and creativity in content creation. All resources are publicly available at: this https URL. 

**Abstract (ZH)**: 商业级视频生成模型Open-Sora 2.0：仅20万美元训练的高性能视频生成模型 

---
# FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting 

**Title (ZH)**: FPGS: 前馈语义aware的高保真样式转移方法研究（基于大规模高斯散步） 

**Authors**: GeonU Kim, Kim Youwang, Lee Hyoseok, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.09635)  

**Abstract**: We present FPGS, a feed-forward photorealistic style transfer method of large-scale radiance fields represented by Gaussian Splatting. FPGS, stylizes large-scale 3D scenes with arbitrary, multiple style reference images without additional optimization while preserving multi-view consistency and real-time rendering speed of 3D Gaussians. Prior arts required tedious per-style optimization or time-consuming per-scene training stage and were limited to small-scale 3D scenes. FPGS efficiently stylizes large-scale 3D scenes by introducing a style-decomposed 3D feature field, which inherits AdaIN's feed-forward stylization machinery, supporting arbitrary style reference images. Furthermore, FPGS supports multi-reference stylization with the semantic correspondence matching and local AdaIN, which adds diverse user control for 3D scene styles. FPGS also preserves multi-view consistency by applying semantic matching and style transfer processes directly onto queried features in 3D space. In experiments, we demonstrate that FPGS achieves favorable photorealistic quality scene stylization for large-scale static and dynamic 3D scenes with diverse reference images. Project page: this https URL 

**Abstract (ZH)**: 我们提出FPGS，这是一种基于高斯 splatting 表示的大规模辐射场的端到端 photorealistic 风格转移方法。FPGS能够在不进行附加优化的情况下，使用任意多个风格参考图像对大规模 3D 场景进行风格化处理，同时保持多视角一致性和 3D 高斯体的实时渲染速度。先前的方法需要针对每种风格进行繁琐的优化或针对每个场景进行耗时的训练阶段，且仅限于小规模 3D 场景。FPGS通过引入风格分解的 3D 特征场，继承了 AdaIN 的端到端风格化机制，支持任意风格参考图像。此外，FPGS还支持通过语义对应匹配和局部 AdaIN 进行多参考图像风格化，为 3D 场景风格提供了多样化的用户控制。FPGS通过直接在 3D 空间中的查询特征上应用语义匹配和风格转移过程，保持多视角一致性。在实验中，我们展示了FPGS能够使用各种参考图像实现大规模静态和动态 3D 场景的优质 photorealistic 风格化效果。项目页面：这个链接 

---
