# BEV-LIO(LC): BEV Image Assisted LiDAR-Inertial Odometry with Loop Closure 

**Title (ZH)**: BEV-LIO(LC): 基于BEV图像的 LiDAR-惯性里程计环路闭合辅助定位 

**Authors**: Haoxin Cai, Shenghai Yuan, Xinyi Li, Junfeng Guo, Jianqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19242)  

**Abstract**: This work introduces BEV-LIO(LC), a novel LiDAR-Inertial Odometry (LIO) framework that combines Bird's Eye View (BEV) image representations of LiDAR data with geometry-based point cloud registration and incorporates loop closure (LC) through BEV image features. By normalizing point density, we project LiDAR point clouds into BEV images, thereby enabling efficient feature extraction and matching. A lightweight convolutional neural network (CNN) based feature extractor is employed to extract distinctive local and global descriptors from the BEV images. Local descriptors are used to match BEV images with FAST keypoints for reprojection error construction, while global descriptors facilitate loop closure detection. Reprojection error minimization is then integrated with point-to-plane registration within an iterated Extended Kalman Filter (iEKF). In the back-end, global descriptors are used to create a KD-tree-indexed keyframe database for accurate loop closure detection. When a loop closure is detected, Random Sample Consensus (RANSAC) computes a coarse transform from BEV image matching, which serves as the initial estimate for Iterative Closest Point (ICP). The refined transform is subsequently incorporated into a factor graph along with odometry factors, improving the global consistency of localization. Extensive experiments conducted in various scenarios with different LiDAR types demonstrate that BEV-LIO(LC) outperforms state-of-the-art methods, achieving competitive localization accuracy. Our code, video and supplementary materials can be found at this https URL. 

**Abstract (ZH)**: BEV-LIO(LC)：一种结合鸟瞰图表示的环视闭合LiDAR-惯性里程计框架 

---
# Efficient and Distributed Large-Scale Point Cloud Bundle Adjustment via Majorization-Minimization 

**Title (ZH)**: 大规模点云_bundle_调整的高效分布式majorization-minimization方法 

**Authors**: Rundong Li, Zheng Liu, Hairuo Wei, Yixi Cai, Haotian Li, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18801)  

**Abstract**: Point cloud bundle adjustment is critical in large-scale point cloud mapping. However, it is both computationally and memory intensive, with its complexity growing cubically as the number of scan poses increases. This paper presents BALM3.0, an efficient and distributed large-scale point cloud bundle adjustment method. The proposed method employs the majorization-minimization algorithm to decouple the scan poses in the bundle adjustment process, thus performing the point cloud bundle adjustment on large-scale data with improved computational efficiency. The key difficulty of applying majorization-minimization on bundle adjustment is to identify the proper surrogate cost function. In this paper, the proposed surrogate cost function is based on the point-to-plane distance. The primary advantages of decoupling the scan poses via a majorization-minimization algorithm stem from two key aspects. First, the decoupling of scan poses reduces the optimization time complexity from cubic to linear, significantly enhancing the computational efficiency of the bundle adjustment process in large-scale environments. Second, it lays the theoretical foundation for distributed bundle adjustment. By distributing both data and computation across multiple devices, this approach helps overcome the limitations posed by large memory and computational requirements, which may be difficult for a single device to handle. The proposed method is extensively evaluated in both simulated and real-world environments. The results demonstrate that the proposed method achieves the same optimal residual with comparable accuracy while offering up to 704 times faster optimization speed and reducing memory usage to 1/8. Furthermore, this paper also presented and implemented a distributed bundle adjustment framework and successfully optimized large-scale data (21,436 poses with 70 GB point clouds) with four consumer-level laptops. 

**Abstract (ZH)**: 大规模点云Bundle调整的BALM3.0高效分布式方法 

---
# MaskPlanner: Learning-Based Object-Centric Motion Generation from 3D Point Clouds 

**Title (ZH)**: MaskPlanner: 基于学习的物体中心运动生成从3D点云 

**Authors**: Gabriele Tiboni, Raffaello Camoriano, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18745)  

**Abstract**: Object-Centric Motion Generation (OCMG) plays a key role in a variety of industrial applications$\unicode{x2014}$such as robotic spray painting and welding$\unicode{x2014}$requiring efficient, scalable, and generalizable algorithms to plan multiple long-horizon trajectories over free-form 3D objects. However, existing solutions rely on specialized heuristics, expensive optimization routines, or restrictive geometry assumptions that limit their adaptability to real-world scenarios. In this work, we introduce a novel, fully data-driven framework that tackles OCMG directly from 3D point clouds, learning to generalize expert path patterns across free-form surfaces. We propose MaskPlanner, a deep learning method that predicts local path segments for a given object while simultaneously inferring "path masks" to group these segments into distinct paths. This design induces the network to capture both local geometric patterns and global task requirements in a single forward pass. Extensive experimentation on a realistic robotic spray painting scenario shows that our approach attains near-complete coverage (above 99%) for unseen objects, while it remains task-agnostic and does not explicitly optimize for paint deposition. Moreover, our real-world validation on a 6-DoF specialized painting robot demonstrates that the generated trajectories are directly executable and yield expert-level painting quality. Our findings crucially highlight the potential of the proposed learning method for OCMG to reduce engineering overhead and seamlessly adapt to several industrial use cases. 

**Abstract (ZH)**: 对象中心的运动生成（OCMG）在各种工业应用中发挥着关键作用，如机器人喷涂和焊接，需要高效的、可扩展的和通用的算法来规划自由形式3D物体上的长时_horizon_轨迹。然而，现有解决方案依赖于专门的启发式方法、昂贵的优化过程或限制性的几何假设，这限制了它们在实际场景中的适应性。在本文中，我们提出了一种全新的、完全数据驱动的框架，直接从3D点云中处理OCMG问题，学习在自由形态表面上泛化专家路径模式。我们提出了一种名为MaskPlanner的深度学习方法，该方法在给定对象的情况下预测局部路径片段，同时推断“路径掩码”将这些片段聚合成不同的路径。这种设计促使网络在一个前向传递过程中同时捕捉局部几何模式和全局任务需求。在现实的机器人喷涂场景中的广泛实验表明，我们的方法对于未见过的对象可以获得接近完全的覆盖（超过99%），并且它是任务无关的，未明确针对涂料沉积进行优化。此外，在一个六自由度专业喷涂机器人上的实际验证表明，生成的轨迹可以直接执行并达到专家级的喷涂质量。我们的研究结果关键性地突显了所提出的学习方法在OCMG中的潜力，可以减少工程开销并无缝适应多个工业应用场景。 

---
# QueryAdapter: Rapid Adaptation of Vision-Language Models in Response to Natural Language Queries 

**Title (ZH)**: QueryAdapter: 面向自然语言查询的视觉语言模型快速适配 

**Authors**: Nicolas Harvey Chapman, Feras Dayoub, Will Browne, Christopher Lehnert  

**Link**: [PDF](https://arxiv.org/pdf/2502.18735)  

**Abstract**: A domain shift exists between the large-scale, internet data used to train a Vision-Language Model (VLM) and the raw image streams collected by a robot. Existing adaptation strategies require the definition of a closed-set of classes, which is impractical for a robot that must respond to diverse natural language queries. In response, we present QueryAdapter; a novel framework for rapidly adapting a pre-trained VLM in response to a natural language query. QueryAdapter leverages unlabelled data collected during previous deployments to align VLM features with semantic classes related to the query. By optimising learnable prompt tokens and actively selecting objects for training, an adapted model can be produced in a matter of minutes. We also explore how objects unrelated to the query should be dealt with when using real-world data for adaptation. In turn, we propose the use of object captions as negative class labels, helping to produce better calibrated confidence scores during adaptation. Extensive experiments on ScanNet++ demonstrate that QueryAdapter significantly enhances object retrieval performance compared to state-of-the-art unsupervised VLM adapters and 3D scene graph methods. Furthermore, the approach exhibits robust generalization to abstract affordance queries and other datasets, such as Ego4D. 

**Abstract (ZH)**: 一个大型互联网数据训练的视觉-语言模型与机器人收集的原始图像流之间存在领域转换问题。现有的适应策略需要定义一个封闭类集，这对于必须响应 diverse 自然语言查询的机器人来说是不切实际的。为此，我们提出 QueryAdapter；一种针对自然语言查询快速适应预训练视觉-语言模型的新型框架。QueryAdapter 利用先前部署中收集的未标注数据，将 VLM 特征对齐到与查询相关的语义类。通过优化可学习的提示标记并主动选择用于训练的对象，可以在几分钟内生成适应模型。我们还探讨了在使用真实世界数据进行适应时，如何处理与查询无关的对象。为此，我们建议使用对象描述作为负面类标签，有助于在适应过程中产生更好的校准置信分数。广泛的 ScanNet++ 实验表明，QueryAdapter 在对象检索性能上显著优于最先进的无监督 VLM 调适方法和 3D 场景图方法。此外，该方法在抽象用法查询和其他数据集（如 Ego4D）上表现出稳健的泛化能力。 

---
# Autonomous Vision-Guided Resection of Central Airway Obstruction 

**Title (ZH)**: 自主视觉引导中央气道阻塞切除术 

**Authors**: M. E. Smith, N. Yilmaz, T. Watts, P. M. Scheikl, J. Ge, A. Deguet, A. Kuntz, A. Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18586)  

**Abstract**: Existing tracheal tumor resection methods often lack the precision required for effective airway clearance, and robotic advancements offer new potential for autonomous resection. We present a vision-guided, autonomous approach for palliative resection of tracheal tumors. This system models the tracheal surface with a fifth-degree polynomial to plan tool trajectories, while a custom Faster R-CNN segmentation pipeline identifies the trachea and tumor boundaries. The electrocautery tool angle is optimized using handheld surgical demonstrations, and trajectories are planned to maintain a 1 mm safety clearance from the tracheal surface. We validated the workflow successfully in five consecutive experiments on ex-vivo animal tissue models, successfully clearing the airway obstruction without trachea perforation in all cases (with more than 90% volumetric tumor removal). These results support the feasibility of an autonomous resection platform, paving the way for future developments in minimally-invasive autonomous resection. 

**Abstract (ZH)**: 基于视觉引导的自动气管肿瘤消融方法 

---
# Deep Learning-Based Transfer Learning for Classification of Cassava Disease 

**Title (ZH)**: 基于深度学习的迁移学习在甘薯疾病分类中的应用 

**Authors**: Ademir G. Costa Junior, Fábio S. da Silva, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2502.19351)  

**Abstract**: This paper presents a performance comparison among four Convolutional Neural Network architectures (EfficientNet-B3, InceptionV3, ResNet50, and VGG16) for classifying cassava disease images. The images were sourced from an imbalanced dataset from a competition. Appropriate metrics were employed to address class imbalance. The results indicate that EfficientNet-B3 achieved on this task accuracy of 87.7%, precision of 87.8%, revocation of 87.8% and F1-Score of 87.7%. These findings suggest that EfficientNet-B3 could be a valuable tool to support Digital Agriculture. 

**Abstract (ZH)**: 本文比较了四种卷积神经网络架构（EfficientNet-B3、InceptionV3、ResNet50和VGG16）在分类甘蔗病害图像中的性能。所用图像来自于一个竞赛的不平衡数据集。使用适当的指标解决类别不平衡问题。结果表明，EfficientNet-B3在该项任务中的准确率为87.7%，精确率为87.8%，召回率为87.8%，F1分数为87.7%。这些发现表明，EfficientNet-B3可能是支持数字农业的一个有价值工具。 

---
# EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region 

**Title (ZH)**: 阿拉伯湾地区自动驾驶的视觉多任务基准数据集：EMT 

**Authors**: Nadya Abdel Madjid, Murad Mebrahtu, Abdelmoamen Nasser, Bilal Hassan, Naoufel Werghi, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19260)  

**Abstract**: This paper introduces the Emirates Multi-Task (EMT) dataset - the first publicly available dataset for autonomous driving collected in the Arab Gulf region. The EMT dataset captures the unique road topology, high traffic congestion, and distinctive characteristics of the Gulf region, including variations in pedestrian clothing and weather conditions. It contains over 30,000 frames from a dash-camera perspective, along with 570,000 annotated bounding boxes, covering approximately 150 kilometers of driving routes. The EMT dataset supports three primary tasks: tracking, trajectory forecasting and intention prediction. Each benchmark dataset is complemented with corresponding evaluations: (1) multi-agent tracking experiments, focusing on multi-class scenarios and occlusion handling; (2) trajectory forecasting evaluation using deep sequential and interaction-aware models; and (3) intention benchmark experiments conducted for predicting agents intentions from observed trajectories. The dataset is publicly available at this https URL, and pre-processing scripts along with evaluation models can be accessed at this https URL. 

**Abstract (ZH)**: 这篇论文介绍了阿联酋多任务（EMT）数据集——阿拉伯海湾地区首个公开的自主驾驶数据集。EMT数据集捕捉到了独特的道路拓扑结构、高交通拥堵情况以及海湾地区的特色，包括行人服饰和天气条件的差异。数据集包含超过30,000帧前视摄像头视角的数据，以及570,000个标注的边界框，覆盖约150公里的驾驶路线。EMT数据集支持三项主要任务：跟踪、轨迹预测和意图预测。每个基准数据集都配备了相应的评估方法：（1）多代理跟踪实验，专注于多类别场景和遮挡处理；（2）使用深度序列和交互感知模型的轨迹预测评估；（3）意图基准实验，用于从观察轨迹预测代理的意图。数据集可在以下网址公开获取，预处理脚本及评估模型可在以下网址访问。 

---
# A Lightweight and Extensible Cell Segmentation and Classification Model for Whole Slide Images 

**Title (ZH)**: 一种轻量级且可扩展的Whole Slide Images细胞分割与分类模型 

**Authors**: Nikita Shvetsov, Thomas K. Kilvaer, Masoud Tafavvoghi, Anders Sildnes, Kajsa Møllersen, Lill-Tove Rasmussen Busund, Lars Ailo Bongo  

**Link**: [PDF](https://arxiv.org/pdf/2502.19217)  

**Abstract**: Developing clinically useful cell-level analysis tools in digital pathology remains challenging due to limitations in dataset granularity, inconsistent annotations, high computational demands, and difficulties integrating new technologies into workflows. To address these issues, we propose a solution that enhances data quality, model performance, and usability by creating a lightweight, extensible cell segmentation and classification model. First, we update data labels through cross-relabeling to refine annotations of PanNuke and MoNuSAC, producing a unified dataset with seven distinct cell types. Second, we leverage the H-Optimus foundation model as a fixed encoder to improve feature representation for simultaneous segmentation and classification tasks. Third, to address foundation models' computational demands, we distill knowledge to reduce model size and complexity while maintaining comparable performance. Finally, we integrate the distilled model into QuPath, a widely used open-source digital pathology platform. Results demonstrate improved segmentation and classification performance using the H-Optimus-based model compared to a CNN-based model. Specifically, average $R^2$ improved from 0.575 to 0.871, and average $PQ$ score improved from 0.450 to 0.492, indicating better alignment with actual cell counts and enhanced segmentation quality. The distilled model maintains comparable performance while reducing parameter count by a factor of 48. By reducing computational complexity and integrating into workflows, this approach may significantly impact diagnostics, reduce pathologist workload, and improve outcomes. Although the method shows promise, extensive validation is necessary prior to clinical deployment. 

**Abstract (ZH)**: 开发用于数字病理学的临床有用细胞水平分析工具仍具有挑战性，由于数据集粒度不足、标注不一致、高计算需求以及将新技术集成到工作流程中困难等原因。为解决这些问题，我们提出了一种解决方案，通过创建一个轻量级且可扩展的细胞分割和分类模型来提升数据质量、模型性能和易用性。首先，通过交叉重新标注来更新数据标签，从而细化PANUKE和MoNuSAC的标注，生成一个包含七种不同细胞类型的统一数据集。其次，利用H-Optimus基础模型作为固定编码器，以提高同时进行分割和分类任务的特征表示。第三，为解决基础模型的高计算需求，通过知识蒸馏来减小模型大小和复杂性，同时保持相似的性能。最后，将蒸馏后的模型集成到QuPath这一广泛使用的开源数字病理学平台中。结果表明，基于H-Optimus模型的分割和分类性能优于基于CNN的模型，特别是平均$R^2$从0.575提升到0.871，平均$PQ$分数从0.450提升到0.492，显示出更好的细胞计数对齐性和增强的分割质量。蒸馏后的模型保持相似性能的同时，参数量减少了48倍。通过降低计算复杂度并集成到工作流程中，这种方法可能对诊断产生重大影响，减少病理学家的工作负担，并改善结果。尽管该方法显示出潜力，但在临床部署之前仍需进行广泛的验证。 

---
# From Traditional to Deep Learning Approaches in Whole Slide Image Registration: A Methodological Review 

**Title (ZH)**: 从传统方法到深度学习在全视野组织图像配准中的应用：一种方法学综述 

**Authors**: Behnaz Elhaminia, Abdullah Alsalemi, Esha Nasir, Mostafa Jahanifar, Ruqayya Awan, Lawrence S. Young, Nasir M. Rajpoot, Fayyaz Minhas, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2502.19123)  

**Abstract**: Whole slide image (WSI) registration is an essential task for analysing the tumour microenvironment (TME) in histopathology. It involves the alignment of spatial information between WSIs of the same section or serial sections of a tissue sample. The tissue sections are usually stained with single or multiple biomarkers before imaging, and the goal is to identify neighbouring nuclei along the Z-axis for creating a 3D image or identifying subclasses of cells in the TME. This task is considerably more challenging compared to radiology image registration, such as magnetic resonance imaging or computed tomography, due to various factors. These include gigapixel size of images, variations in appearance between differently stained tissues, changes in structure and morphology between non-consecutive sections, and the presence of artefacts, tears, and deformations. Currently, there is a noticeable gap in the literature regarding a review of the current approaches and their limitations, as well as the challenges and opportunities they present. We aim to provide a comprehensive understanding of the available approaches and their application for various purposes. Furthermore, we investigate current deep learning methods used for WSI registration, emphasising their diverse methodologies. We examine the available datasets and explore tools and software employed in the field. Finally, we identify open challenges and potential future trends in this area of research. 

**Abstract (ZH)**: 全视野图像（WSI）对齐在组织病理学分析肿瘤微环境（TME）中的重要性及其挑战 

---
# InternVQA: Advancing Compressed Video QualityAssessment with Distilling Large Foundation Model 

**Title (ZH)**: InternVQA： advancing compressed video quality assessment with distilling large foundation models 

**Authors**: Fengbin Guan, Zihao Yu, Yiting Lu, Xin Li, Zhibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19026)  

**Abstract**: Video quality assessment tasks rely heavily on the rich features required for video understanding, such as semantic information, texture, and temporal motion. The existing video foundational model, InternVideo2, has demonstrated strong potential in video understanding tasks due to its large parameter size and large-scale multimodal data pertaining. Building on this, we explored the transferability of InternVideo2 to video quality assessment under compression scenarios. To design a lightweight model suitable for this task, we proposed a distillation method to equip the smaller model with rich compression quality priors. Additionally, we examined the performance of different backbones during the distillation process. The results showed that, compared to other methods, our lightweight model distilled from InternVideo2 achieved excellent performance in compression video quality assessment. 

**Abstract (ZH)**: 基于压缩场景的视频质量评估中InternVideo2轻量化模型的研究与性能分析 

---
# Inscanner: Dual-Phase Detection and Classification of Auxiliary Insulation Using YOLOv8 Models 

**Title (ZH)**: Inscanner: 辅助绝缘的两阶段检测与分类方法基于YOLOv8模型 

**Authors**: Youngtae Kim, Soonju Jeong, Sardar Arslan, Dhananjay Agnihotri, Yahya Ahmed, Ali Nawaz, Jinhee Song, Hyewon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18871)  

**Abstract**: This study proposes a two-phase methodology for detecting and classifying auxiliary insulation in structural components. In the detection phase, a YOLOv8x model is trained on a dataset of complete structural blueprints, each annotated with bounding boxes indicating areas that should contain insulation. In the classification phase, these detected insulation patches are cropped and categorized into two classes: present or missing. These are then used to train a YOLOv8x-CLS model that determines the presence or absence of auxiliary insulation. Preprocessing steps for both datasets included annotation, augmentation, and appropriate cropping of the insulation regions. The detection model achieved a mean average precision (mAP) score of 82%, while the classification model attained an accuracy of 98%. These findings demonstrate the effectiveness of the proposed approach in automating insulation detection and classification, providing a foundation for further advancements in this domain. 

**Abstract (ZH)**: 本研究提出了一种两阶段方法，用于检测和分类结构组件中的辅助绝缘。在检测阶段，使用每个标注有表示应包含绝缘区域边界的边界框的完整结构蓝图数据集对YOLOv8x模型进行训练。在分类阶段，检测到的绝缘补丁被裁剪并归类为存在或缺失两类，然后使用这些补丁来训练YOLOv8x-CLS模型以确定辅助绝缘的存在或缺失。两个数据集的预处理步骤包括标注、增强和适当裁剪绝缘区域。检测模型达到了82%的平均精度（mAP），分类模型的准确率为98%。这些发现表明，所提出的方法在自动化绝缘检测和分类方面是有效的，为该领域的进一步发展奠定了基础。 

---
# Cross-Modality Investigation on WESAD Stress Classification 

**Title (ZH)**: 跨模态研究在WESAD压力分类中的应用 

**Authors**: Eric Oliver, Sagnik Dakshit  

**Link**: [PDF](https://arxiv.org/pdf/2502.18733)  

**Abstract**: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for stress detection. Furthermore, this study explores cross-modal performance and also explains the same using 2D visualization of the learned embedding space and quantitative analysis based on data variance. Despite the large body of work on stress detection and monitoring, the robustness and generalization of these models across different modalities has not been explored. This research represents one of the initial efforts to interpret embedding spaces for stress detection, providing valuable information on cross-modal performance. 

**Abstract (ZH)**: 深度学习的广泛应用促进了其在医疗健康领域的广泛应用，其中人工智能和传感器的进步提升了诊断、治疗和监测的效果。在移动医疗中，基于人工智能的工具能够实现压力等疾病的早期诊断和持续监测。可穿戴技术与多模态生理数据使得压力检测愈发可行，但模型的有效性依赖于数据的质量、数量和模态。本研究使用WESAD数据集开发了针对压力检测的变换器模型，并通过对心电图（ECG）、电导率活动（EDA）、肌电图（EMG）、呼吸率（RESP）、温度（TEMP）以及3轴加速度计（ACC）信号的训练，展示了单模态变换器在分析生理信号方面的有效性，在压力检测中取得了高达99.73%至99.95%的准确率、精确率和召回率，同时探讨了跨模态性能，并通过学习嵌入空间的2D可视化及基于数据方差的定量分析进行了解释。尽管已有大量关于压力检测和监测的研究，但这些模型在不同模态下的鲁棒性和泛化性尚未被充分探索。本研究是首次尝试解释嵌入空间在压力检测中的跨模态性能，提供了有价值的信息。 

---
# Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision 

**Title (ZH)**: 基于注意力机制、双方向长短期记忆网络（BiLSTM）和CNN的人体冲突检测在计算机视觉中的应用 

**Authors**: Erick da Silva Farias, Eduardo Palhares Junior  

**Link**: [PDF](https://arxiv.org/pdf/2502.18555)  

**Abstract**: The automatic detection of human conflicts through videos is a crucial area in computer vision, with significant applications in monitoring and public safety policies. However, the scarcity of public datasets and the complexity of human interactions make this task challenging. This study investigates the integration of advanced deep learning techniques, including Attention Mechanism, Convolutional Neural Networks (CNNs), and Bidirectional Long ShortTerm Memory (BiLSTM), to improve the detection of violent behaviors in videos. The research explores how the use of the attention mechanism can help focus on the most relevant parts of the video, enhancing the accuracy and robustness of the model. The experiments indicate that the combination of CNNs with BiLSTM and the attention mechanism provides a promising solution for conflict monitoring, offering insights into the effectiveness of different strategies. This work opens new possibilities for the development of automated surveillance systems that can operate more efficiently in real-time detection of violent events. 

**Abstract (ZH)**: 通过视频自动检测人类冲突是计算机视觉中的一个重要领域，具有在监控和公共安全政策中的广泛应用。然而，公共数据集的稀缺性和人类互动的复杂性使得这一任务具有挑战性。本研究探讨了结合注意力机制、卷积神经网络（CNNs）和双向长短期记忆（BiLSTM）等高级深度学习技术的方法，以提高视频中暴力行为检测的性能。研究探讨了注意力机制在帮助聚焦视频中最相关部分方面的应用，从而提高模型的准确性和健壮性。实验表明，将CNNs与BiLSTM结合使用并加入注意力机制提供了一种有前景的冲突监测解决方案，揭示了不同策略的有效性。本研究为开发更有效地进行实时暴力事件检测的自动化监控系统开辟了新的可能性。 

---
# FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition 

**Title (ZH)**: FreeTumor: 在计算机断层扫描图像中大规模生成肿瘤合成以提高肿瘤识别 

**Authors**: Linshan Wu, Jiaxin Zhuang, Yanning Zhou, Sunan He, Jiabo Ma, Luyang Luo, Xi Wang, Xuefeng Ni, Xiaoling Zhong, Mingxiang Wu, Yinghua Zhao, Xiaohui Duan, Varut Vardhanabhuti, Pranav Rajpurkar, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18519)  

**Abstract**: Tumor is a leading cause of death worldwide, with an estimated 10 million deaths attributed to tumor-related diseases every year. AI-driven tumor recognition unlocks new possibilities for more precise and intelligent tumor screening and diagnosis. However, the progress is heavily hampered by the scarcity of annotated datasets, which demands extensive annotation efforts by radiologists. To tackle this challenge, we introduce FreeTumor, an innovative Generative AI (GAI) framework to enable large-scale tumor synthesis for mitigating data scarcity. Specifically, FreeTumor effectively leverages a combination of limited labeled data and large-scale unlabeled data for tumor synthesis training. Unleashing the power of large-scale data, FreeTumor is capable of synthesizing a large number of realistic tumors on images for augmenting training datasets. To this end, we create the largest training dataset for tumor synthesis and recognition by curating 161,310 publicly available Computed Tomography (CT) volumes from 33 sources, with only 2.3% containing annotated tumors. To validate the fidelity of synthetic tumors, we engaged 13 board-certified radiologists in a Visual Turing Test to discern between synthetic and real tumors. Rigorous clinician evaluation validates the high quality of our synthetic tumors, as they achieved only 51.1% sensitivity and 60.8% accuracy in distinguishing our synthetic tumors from real ones. Through high-quality tumor synthesis, FreeTumor scales up the recognition training datasets by over 40 times, showcasing a notable superiority over state-of-the-art AI methods including various synthesis methods and foundation models. These findings indicate promising prospects of FreeTumor in clinical applications, potentially advancing tumor treatments and improving the survival rates of patients. 

**Abstract (ZH)**: 肿瘤是全球主要的死亡原因，每年约有1000万人死于肿瘤相关疾病。基于AI的肿瘤识别技术开启了更为精准和智能的肿瘤筛查与诊断的新可能。然而，进展受到标注数据稀缺性的严重制约，这要求放射学家付出大量的标注努力。为应对这一挑战，我们引入了FreeTumor，一种创新的生成AI（GAI）框架，以缓解数据稀缺性问题。具体而言，FreeTumor有效地利用有限的标注数据和大量的未标注数据进行肿瘤合成训练。通过大规模数据的强大功能，FreeTumor能够在影像中合成大量逼真的肿瘤，以扩充训练数据集。为此，我们创建了最大的肿瘤合成与识别训练数据集，从33个来源收集了161,310个公开的计算机断层扫描（CT）体积，其中仅有2.3%包含标注的肿瘤。为了验证合成肿瘤的真实性，我们邀请了13名经过认证的放射学家进行视觉图灵测试，以区分合成肿瘤和真实肿瘤。严格的临床评估验证了我们合成肿瘤的高质量，它们仅在区分合成肿瘤和真实肿瘤方面达到了51.1%的敏感性和60.8%的准确性。凭借高质量的肿瘤合成，FreeTumor将识别训练数据集扩展了40多倍，展示了在其与各种合成方法和基础模型相比的显著优势。这些发现表明，FreeTumor在临床应用中拥有广阔的前景，可能促进肿瘤治疗并提高患者的生存率。 

---
# A Comprehensive Survey on Composed Image Retrieval 

**Title (ZH)**: 综述性研究：合成图像检索 

**Authors**: Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18495)  

**Abstract**: Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration. 

**Abstract (ZH)**: 多重模态的图像检索（Composed Image Retrieval, CIR）是一项新兴且具有挑战性的任务，允许用户使用包含参考图像和修改文本的多模态查询来搜索目标图像。随着深度学习的进步，CIR 成为了计算机视觉和机器学习领域的一个快速发展的研究热点。据我们所知，目前尚缺乏对该领域的全面综述。因此，我们综合了超过120篇发表在顶级会议和期刊上的论文，如ACM TOIS、SIGIR和CVPR，对其进行系统分类，并通过跨多个数据集的实验结果对现有监督和零样本CIR方法进行分析。此外，我们还简要讨论了与CIR紧密相关的任务，如基于属性的CIR和对话驱动的CIR，并介绍了用于评估的基准数据集，提出了该领域有前景的研究方向，为感兴趣的科研人员提供实用建议。 

---
