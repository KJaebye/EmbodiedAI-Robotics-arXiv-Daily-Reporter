# OmniMap: A General Mapping Framework Integrating Optics, Geometry, and Semantics 

**Title (ZH)**: OmniMap：一种综合光学、几何与语义的一般映射框架 

**Authors**: Yinan Deng, Yufeng Yue, Jianyu Dou, Jingyu Zhao, Jiahui Wang, Yujie Tang, Yi Yang, Mengyin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07500)  

**Abstract**: Robotic systems demand accurate and comprehensive 3D environment perception, requiring simultaneous capture of photo-realistic appearance (optical), precise layout shape (geometric), and open-vocabulary scene understanding (semantic). Existing methods typically achieve only partial fulfillment of these requirements while exhibiting optical blurring, geometric irregularities, and semantic ambiguities. To address these challenges, we propose OmniMap. Overall, OmniMap represents the first online mapping framework that simultaneously captures optical, geometric, and semantic scene attributes while maintaining real-time performance and model compactness. At the architectural level, OmniMap employs a tightly coupled 3DGS-Voxel hybrid representation that combines fine-grained modeling with structural stability. At the implementation level, OmniMap identifies key challenges across different modalities and introduces several innovations: adaptive camera modeling for motion blur and exposure compensation, hybrid incremental representation with normal constraints, and probabilistic fusion for robust instance-level understanding. Extensive experiments show OmniMap's superior performance in rendering fidelity, geometric accuracy, and zero-shot semantic segmentation compared to state-of-the-art methods across diverse scenes. The framework's versatility is further evidenced through a variety of downstream applications, including multi-domain scene Q&A, interactive editing, perception-guided manipulation, and map-assisted navigation. 

**Abstract (ZH)**: 全方位映射：同时实现光学、几何和语义场景属性的实时紧凑映射 

---
# Aerial-ground Cross-modal Localization: Dataset, Ground-truth, and Benchmark 

**Title (ZH)**: 空中-地面跨模态定位：数据集、 ground-truth 和基准 

**Authors**: Yandi Yang, Jianping Li, Youqi Liao, Yuhao Li, Yizhe Zhang, Zhen Dong, Bisheng Yang, Naser El-Sheimy  

**Link**: [PDF](https://arxiv.org/pdf/2509.07362)  

**Abstract**: Accurate visual localization in dense urban environments poses a fundamental task in photogrammetry, geospatial information science, and robotics. While imagery is a low-cost and widely accessible sensing modality, its effectiveness on visual odometry is often limited by textureless surfaces, severe viewpoint changes, and long-term drift. The growing public availability of airborne laser scanning (ALS) data opens new avenues for scalable and precise visual localization by leveraging ALS as a prior map. However, the potential of ALS-based localization remains underexplored due to three key limitations: (1) the lack of platform-diverse datasets, (2) the absence of reliable ground-truth generation methods applicable to large-scale urban environments, and (3) limited validation of existing Image-to-Point Cloud (I2P) algorithms under aerial-ground cross-platform settings. To overcome these challenges, we introduce a new large-scale dataset that integrates ground-level imagery from mobile mapping systems with ALS point clouds collected in Wuhan, Hong Kong, and San Francisco. 

**Abstract (ZH)**: 准确的城市密集环境中视觉定位是摄影测量、地理空间信息科学和机器人技术中的基本任务。虽然图像是一种低成本且广泛可访问的传感模态，但其在视觉里程计中的有效性往往受限于纹理缺乏的表面、严重的视角变化和长期漂移。随着空中激光扫描（ALS）数据的日益公开，利用ALS作为先验地图来实现可扩展和精确的视觉定位开启了新的途径。然而，由于三个关键限制，ALS基于的定位潜力尚未得到充分探索：（1）缺乏平台多样化的数据集，（2）缺乏适用于大规模城市环境的可靠的地面真实生成方法，以及（3）现有图像到点云（I2P）算法在航空与地面跨平台设置下的验证不足。为克服这些挑战，我们引入了一个新的大规模数据集，该数据集将武汉、香港和旧金山等地的移动测绘系统获取的地面级图像与ALS点云相结合。 

---
# Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search 

**Title (ZH)**: Mini-o3: 扩大规模以增强视觉搜索中的推理模式和交互轮次 

**Authors**: Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07969)  

**Abstract**: Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning -- spanning tens of steps -- and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3-style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems. 

**Abstract (ZH)**: 近期大规模多模态模型的发展已经利用基于图像的工具与强化学习相结合来解决视觉问题。然而，现有的开源方法往往表现出单调的推理模式，并且只允许有限的交互回合次数，这使得它们对于需要试错探索的任务显得不足。在本工作中，我们通过扩展基于工具的交互来解决这一限制，并引入了Mini-o3系统，该系统执行深度、多回合推理——跨越数十个步骤——并在具有挑战性的视觉搜索任务上实现了最先进的性能。我们重现OpenAI o3风格行为的配方包括三个关键组件。首先，我们构建了Visual Probe Dataset，这是一个包含数千个旨在用于探索性推理的具有挑战性的视觉搜索问题集合。其次，我们开发了迭代数据收集管道，以获取冷启动轨迹，其中包括深度优先搜索、试错和目标维持等多样化的推理模式。第三，我们提出了超轮次遮掩策略，在强化学习过程中防止对达到最大轮次数的响应进行惩罚，从而在训练效率与测试可扩展性之间保持平衡。尽管仅使用上限为六次交互轮次进行训练，但我们的模型在推理时能够自然扩展到数十次轮次，并且随着轮次数的增加，准确性也会提高。深入的实验表明，Mini-o3生成了丰富的推理模式和深层次的思考路径，有效地解决了具有挑战性的视觉搜索问题。 

---
# Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s 

**Title (ZH)**: 面向消费者GPU的YOLOv10s本地AI加速：一种硬件感知动态策略 

**Authors**: Mahmudul Islam Masum, Miad Islam, Arif I. Sarwat  

**Link**: [PDF](https://arxiv.org/pdf/2509.07928)  

**Abstract**: As local AI grows in popularity, there is a critical gap between the benchmark performance of object detectors and their practical viability on consumer-grade hardware. While models like YOLOv10s promise real-time speeds, these metrics are typically achieved on high-power, desktop-class GPUs. This paper reveals that on resource-constrained systems, such as laptops with RTX 4060 GPUs, performance is not compute-bound but is instead dominated by system-level bottlenecks, as illustrated by a simple bottleneck test. To overcome this hardware-level constraint, we introduce a Two-Pass Adaptive Inference algorithm, a model-independent approach that requires no architectural changes. This study mainly focuses on adaptive inference strategies and undertakes a comparative analysis of architectural early-exit and resolution-adaptive routing, highlighting their respective trade-offs within a unified evaluation framework. The system uses a fast, low-resolution pass and only escalates to a high-resolution model pass when detection confidence is low. On a 5000-image COCO dataset, our method achieves a 1.85x speedup over a PyTorch Early-Exit baseline, with a modest mAP loss of 5.51%. This work provides a practical and reproducible blueprint for deploying high-performance, real-time AI on consumer-grade devices by shifting the focus from pure model optimization to hardware-aware inference strategies that maximize throughput. 

**Abstract (ZH)**: 随着本地AI的流行，目标检测器在基准性能和消费级硬件上的实际可行性之间存在关键差距。尽管像YOLOv10s这样的模型承诺实现实时速度，但这些指标通常是在高性能的台式机级GPU上达成的。本文揭示，在资源受限的系统上，如配备RTX 4060 GPU的笔记本电脑，性能不是由计算能力决定的，而是受系统级瓶颈主导，这一点通过一个简单的瓶颈测试得到说明。为克服这一硬件限制，我们提出了一种两阶段自适应推理算法，这是一种模型无关的方法，不需要架构修改。本研究主要集中于自适应推理策略，并在统一的评估框架下对架构早期退出和分辨率自适应路由进行了比较分析，突显了它们各自的权衡。系统使用快速低分辨率通道，在检测置信度低时才升级到高分辨率模型通道。在5000张图片的COCO数据集上，我们的方法相对于PyTorch早期退出基线实现了1.85倍的速度提升，同时保持了轻微的mAP损失（5.51%）。这项工作提供了一种实用且可重现的蓝图，通过将注意力从纯粹的模型优化转向最大化吞吐量的硬件感知推理策略，使高性能实时AI能够在消费级设备上部署。 

---
# Enhanced SegNet with Integrated Grad-CAM for Interpretable Retinal Layer Segmentation in OCT Images 

**Title (ZH)**: 集成Grad-CAM的增强SegNet在OCT图像中实现可解释的视网膜层分割 

**Authors**: S M Asiful Islam Saky, Ugyen Tshering  

**Link**: [PDF](https://arxiv.org/pdf/2509.07795)  

**Abstract**: Optical Coherence Tomography (OCT) is essential for diagnosing conditions such as glaucoma, diabetic retinopathy, and age-related macular degeneration. Accurate retinal layer segmentation enables quantitative biomarkers critical for clinical decision-making, but manual segmentation is time-consuming and variable, while conventional deep learning models often lack interpretability. This work proposes an improved SegNet-based deep learning framework for automated and interpretable retinal layer segmentation. Architectural innovations, including modified pooling strategies, enhance feature extraction from noisy OCT images, while a hybrid loss function combining categorical cross-entropy and Dice loss improves performance for thin and imbalanced retinal layers. Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide visual explanations, allowing clinical validation of model decisions. Trained and validated on the Duke OCT dataset, the framework achieved 95.77% validation accuracy, a Dice coefficient of 0.9446, and a Jaccard Index (IoU) of 0.8951. Class-wise results confirmed robust performance across most layers, with challenges remaining for thinner boundaries. Grad-CAM visualizations highlighted anatomically relevant regions, aligning segmentation with clinical biomarkers and improving transparency. By combining architectural improvements, a customized hybrid loss, and explainable AI, this study delivers a high-performing SegNet-based framework that bridges the gap between accuracy and interpretability. The approach offers strong potential for standardizing OCT analysis, enhancing diagnostic efficiency, and fostering clinical trust in AI-driven ophthalmic tools. 

**Abstract (ZH)**: 基于改进SegNet的全自动可解释视网膜层分割深度学习框架 

---
# Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks 

**Title (ZH)**: 基于深卷积神经网络的音类别和子类别水平音谱与节奏特征性能评估 

**Authors**: Friedrich Wolf-Monheim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07756)  

**Abstract**: Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs. 

**Abstract (ZH)**: 除了决策树和k最近邻算法之外，卷积神经网络（CNN）在音乐、语音或环境声音等领域广泛用于音频数据分类。通过使用熔频谱图、熔频率倒谱系数（MFCC）、循环节拍图、短时傅里叶变换（STFT）色谱图、常数Q变换（CQT）色谱图和色谱能归一化统计（CENS）色谱图等频谱和节拍特征作为神经网络的数字图像输入数据来训练特定的CNN。本研究使用端到端的深度学习管道，基于包含2000个标记的环境音频录制的ESC-50数据集，详细探讨了这些频谱和节拍特征在音频类别级别和音频类级别分类中的性能。评估的多类别分类准确率、精确率、召回率和F1分数明确显示，熔频谱图和熔频率倒谱系数（MFCC）在这项研究中用于深度CNN的音频分类任务中显著优于其他研究中研究的频谱和节拍特征。 

---
# Transformer-Based Approach to Optimal Sensor Placement for Structural Health Monitoring of Probe Cards 

**Title (ZH)**: 基于变换器的方法在探针卡结构健康监测中的最优传感器布置 

**Authors**: Mehdi Bejani, Marco Mauri, Daniele Acconcia, Simone Todaro, Stefano Mariani  

**Link**: [PDF](https://arxiv.org/pdf/2509.07603)  

**Abstract**: This paper presents an innovative Transformer-based deep learning strategy for optimizing the placement of sensors aiming at structural health monitoring of semiconductor probe cards. Failures in probe cards, including substrate cracks and loosened screws, would critically affect semiconductor manufacturing yield and reliability. Some failure modes could be detected by equipping a probe card with adequate sensors. Frequency response functions from simulated failure scenarios are adopted within a finite element model of a probe card. A comprehensive dataset, enriched by physics-informed scenario expansion and physics-aware statistical data augmentation, is exploited to train a hybrid Convolutional Neural Network and Transformer model. The model achieves high accuracy (99.83%) in classifying the probe card health states (baseline, loose screw, crack) and an excellent crack detection recall (99.73%). Model robustness is confirmed through a rigorous framework of 3 repetitions of 10-fold stratified cross-validation. The attention mechanism also pinpoints critical sensor locations: an analysis of the attention weights offers actionable insights for designing efficient, cost-effective monitoring systems by optimizing sensor configurations. This research highlights the capability of attention-based deep learning to advance proactive maintenance, enhancing operational reliability and yield in semiconductor manufacturing. 

**Abstract (ZH)**: 基于变压器的创新深度学习策略在半导体探针卡结构健康监测中优化传感器布局的研究 

---
# Attention Maps in 3D Shape Classification for Dental Stage Estimation with Class Node Graph Attention Networks 

**Title (ZH)**: 三维形状分类中的注意力图在牙科发展阶段估计中的应用：类节点图形注意力网络 

**Authors**: Barkin Buyukcakir, Rocharles Cavalcante Fontenele, Reinhilde Jacobs, Jannick De Tobel, Patrick Thevissen, Dirk Vandermeulen, Peter Claes  

**Link**: [PDF](https://arxiv.org/pdf/2509.07581)  

**Abstract**: Deep learning offers a promising avenue for automating many recognition tasks in fields such as medicine and forensics. However, the black-box nature of these models hinders their adoption in high-stakes applications where trust and accountability are required. For 3D shape recognition tasks in particular, this paper introduces the Class Node Graph Attention Network (CGAT) architecture to address this need. Applied to 3D meshes of third molars derived from CBCT images, for Demirjian stage allocation, CGAT utilizes graph attention convolutions and an inherent attention mechanism, visualized via attention rollout, to explain its decision-making process. We evaluated the local mean curvature and distance to centroid node features, both individually and in combination, as well as model depth, finding that models incorporating directed edges to a global CLS node produced more intuitive attention maps, while also yielding desirable classification performance. We analyzed the attention-based explanations of the models, and their predictive performances to propose optimal settings for the CGAT. The combination of local mean curvature and distance to centroid as node features yielded a slight performance increase with 0.76 weighted F1 score, and more comprehensive attention visualizations. The CGAT architecture's ability to generate human-understandable attention maps can enhance trust and facilitate expert validation of model decisions. While demonstrated on dental data, CGAT is broadly applicable to graph-based classification and regression tasks, promoting wider adoption of transparent and competitive deep learning models in high-stakes environments. 

**Abstract (ZH)**: 深度学习为医学和取证等领域中的许多识别任务提供了有前景的自动化途径。然而，这些模型的黑箱性质阻碍了其在需要信任和问责的应用中的采用。特别是在3D形状识别任务中，本文引入了Class Node Graph Attention Network (CGAT) 架构以应对这一需求。将CGAT应用于从CBCT图像衍生的第三磨牙3D网格，用于Demirjian阶段分配，CGAT利用图注意力卷积和固有的注意力机制，通过注意力展开可视化其决策过程。我们评估了局部均曲率和质心节点距离这两种特征及其组合的效果，以及模型深度，发现将有向边连接到全局CLS节点的模型生成了更具直观性的注意力图，并且还提供了理想的分类性能。我们分析了模型的基于注意力的解释及其预测性能，提出了CGAT的最佳设置。局部均曲率和质心距离的节点特征组合在加权F1分数为0.76的情况下产生了轻微的性能提升，并提供了更加全面的注意力可视化。CGAT架构生成可人类理解的注意力图的能力可以增强信任，并促进专家对模型决策的验证。虽然CGAT在牙科数据上进行了展示，但它广泛适用于基于图的分类和回归任务，促进了在高风险环境中更广泛采用透明和竞争力强的深度学习模型。 

---
# HU-based Foreground Masking for 3D Medical Masked Image Modeling 

**Title (ZH)**: 基于HU值的医学图像掩码前景屏蔽方法 

**Authors**: Jin Lee, Vu Dang, Gwang-Hyun Yu, Anh Le, Zahid Rahman, Jin-Ho Jang, Heonzoo Lee, Kun-Yung Kim, Jin-Sul Kim, Jin-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.07534)  

**Abstract**: While Masked Image Modeling (MIM) has revolutionized fields of computer vision, its adoption in 3D medical image computing has been limited by the use of random masking, which overlooks the density of anatomical objects. To address this limitation, we enhance the pretext task with a simple yet effective masking strategy. Leveraging Hounsfield Unit (HU) measurements, we implement an HU-based Foreground Masking, which focuses on the intensity distribution of visceral organs and excludes non-tissue regions, such as air and fluid, that lack diagnostically meaningful features. Extensive experiments on five public 3D medical imaging datasets demonstrate that our masking consistently improves performance, both in quality of segmentation and Dice score (BTCV:~84.64\%, Flare22:~92.43\%, MM-WHS:~90.67\%, Amos22:~88.64\%, BraTS:~78.55\%). These results underscore the importance of domain-centric MIM and suggest a promising direction for representation learning in medical image segmentation. Implementation is available at this http URL. 

**Abstract (ZH)**: 基于HU值的前景掩模：在3D医学图像计算中中心化掩模图像建模对于器官分割性能的提升研究 

---
# Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition 

**Title (ZH)**: 通过局部混合和 logits 优化生成可移植的对抗样本以用于遥感目标识别 

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07495)  

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate. 

**Abstract (ZH)**: 深度神经网络（DNNs）容易受到 adversarial 攻击，在遥感应用中的部署面临重大安全威胁。针对 adversarial 攻击的研究不仅揭示了模型的脆弱性，还提供了增强鲁棒性的关键见解。尽管已经提出了基于混合的策略来提高 adversarial 示例的可移植性，但这些策略要么进行全局混合，要么直接交换图像中的区域，这可能会破坏全局语义特征并误导 adversarial 示例的优化。此外，它们依赖于交叉熵损失来进行扰动优化，这会导致迭代更新过程中梯度衰减，降低 adversarial 示例的质量。为了解决这些问题，我们专注于非针对性攻击，并提出了一种基于局部混合和 logits 优化的新型框架。首先，我们提出了一种局部混合策略，以生成多样且语义一致的输入。与全局混合两个图像的 MixUp 不同，与将图像拼接在一起的 MixCut 不同，我们的方法仅混合局部区域，以保留全局语义信息。其次，我们将针对性攻击中的 logits 损失适应非针对性场景，缓解交叉熵损失的梯度消失问题。第三，应用扰动平滑损失以抑制高频噪声并提高可移植性。在 FGSCR-42 和 MTARSI 数据集上的广泛实验表明，我们的方法在 6 种替代模型上的性能优于 12 种当前最先进的方法。特别地，使用 ResNet 作为替代模型在 MTARSI 上，我们的方法在黑盒攻击成功率上平均提高了 17.28%。 

---
# Breast Cancer Detection in Thermographic Images via Diffusion-Based Augmentation and Nonlinear Feature Fusion 

**Title (ZH)**: 基于扩散增强和非线性特征融合的热图乳腺癌检测 

**Authors**: Sepehr Salem, M. Moein Esfahani, Jingyu Liu, Vince Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2509.07277)  

**Abstract**: Data scarcity hinders deep learning for medical imaging. We propose a framework for breast cancer classification in thermograms that addresses this using a Diffusion Probabilistic Model (DPM) for data augmentation. Our DPM-based augmentation is shown to be superior to both traditional methods and a ProGAN baseline. The framework fuses deep features from a pre-trained ResNet-50 with handcrafted nonlinear features (e.g., Fractal Dimension) derived from U-Net segmented tumors. An XGBoost classifier trained on these fused features achieves 98.0\% accuracy and 98.1\% sensitivity. Ablation studies and statistical tests confirm that both the DPM augmentation and the nonlinear feature fusion are critical, statistically significant components of this success. This work validates the synergy between advanced generative models and interpretable features for creating highly accurate medical diagnostic tools. 

**Abstract (ZH)**: 数据稀缺性阻碍了医疗影像领域的深度学习应用。我们提出了一种基于扩散概率模型（DPM）的数据增强框架，用于热像中的乳腺癌分类。该DPM增强方法在数据增强方面优于传统方法和ProGAN基线。该框架将预训练的ResNet-50提取的深度特征与来自U-Net分割肿瘤的手工构建非线性特征（例如，分形维数）融合在一起。基于这些融合特征训练的XGBoost分类器的准确率为98.0%，敏感性为98.1%。消融研究和统计测试证实，DPM增强和非线性特征融合都是该成功的关键、统计显著的组成部分。本工作验证了高级生成模型与可解释特征之间的协同作用对于创建高度准确的医疗诊断工具的重要性。 

---
# Evaluation of Machine Learning Reconstruction Techniques for Accelerated Brain MRI Scans 

**Title (ZH)**: 加速脑部MRI扫描的机器学习重建技术评估 

**Authors**: Jonathan I. Mandel, Shivaprakash Hiremath, Hedyeh Keshtgar, Timothy Scholl, Sadegh Raeisi  

**Link**: [PDF](https://arxiv.org/pdf/2509.07193)  

**Abstract**: This retrospective-prospective study evaluated whether a deep learning-based MRI reconstruction algorithm can preserve diagnostic quality in brain MRI scans accelerated up to fourfold, using both public and prospective clinical data. The study included 18 healthy volunteers (scans acquired at 3T, January 2024-March 2025), as well as selected fastMRI public datasets with diverse pathologies. Phase-encoding-undersampled 2D/3D T1, T2, and FLAIR sequences were reconstructed with DeepFoqus-Accelerate and compared with standard-of-care (SOC). Three board-certified neuroradiologists and two MRI technologists independently reviewed 36 paired SOC/AI reconstructions from both datasets using a 5-point Likert scale, while quantitative similarity was assessed for 408 scans and 1224 datasets using Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Haar wavelet-based Perceptual Similarity Index (HaarPSI). No AI-reconstructed scan scored below 3 (minimally acceptable), and 95% scored $\geq 4$. Mean SSIM was 0.95 $\pm$ 0.03 (90% cases >0.90), PSNR >41.0 dB, and HaarPSI >0.94. Inter-rater agreement was slight to moderate. Rare artifacts did not affect diagnostic interpretation. These findings demonstrate that DeepFoqus-Accelerate enables robust fourfold brain MRI acceleration with 75% reduced scan time, while preserving diagnostic image quality and supporting improved workflow efficiency. 

**Abstract (ZH)**: 基于深度学习的MRI重建算法能否在加速四倍的脑部MRI扫描中保持诊断质量：一项回顾前瞻研究 

---
# Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models 

**Title (ZH)**: 基于矩和功率谱的高斯性正则化方法用于文本到图像模型 

**Authors**: Jisung Hwang, Jaihoon Kim, Minhyuk Sung  

**Link**: [PDF](https://arxiv.org/pdf/2509.07027)  

**Abstract**: We propose a novel regularization loss that enforces standard Gaussianity, encouraging samples to align with a standard Gaussian distribution. This facilitates a range of downstream tasks involving optimization in the latent space of text-to-image models. We treat elements of a high-dimensional sample as one-dimensional standard Gaussian variables and define a composite loss that combines moment-based regularization in the spatial domain with power spectrum-based regularization in the spectral domain. Since the expected values of moments and power spectrum distributions are analytically known, the loss promotes conformity to these properties. To ensure permutation invariance, the losses are applied to randomly permuted inputs. Notably, existing Gaussianity-based regularizations fall within our unified framework: some correspond to moment losses of specific orders, while the previous covariance-matching loss is equivalent to our spectral loss but incurs higher time complexity due to its spatial-domain computation. We showcase the application of our regularization in generative modeling for test-time reward alignment with a text-to-image model, specifically to enhance aesthetics and text alignment. Our regularization outperforms previous Gaussianity regularization, effectively prevents reward hacking and accelerates convergence. 

**Abstract (ZH)**: 我们提出了一种新颖的正则化损失，强制标准高斯性，鼓励样本与标准高斯分布对齐。这促进了涉及文本到图像模型的潜在空间优化的一系列下游任务。我们将高维样本的元素视为一维标准高斯变量，并定义一个结合空间域moment为基础的正则化与频域基础的功率谱正则化的复合损失。由于矩和功率谱分布的期望值是解析已知的，损失促进了这些属性的符合性。为了确保置换不变性，损失应用于随机置换的输入。值得注意的是，现有的高斯性正则化都包含在我们统一的框架中：一些对应于特定阶数的moment损失，而之前的协方差匹配损失等价于我们的频域损失，但由于其在空间域的计算，导致更高的时间复杂度。我们在测试时的奖励对齐生成建模中展示了该正则化应用，特别是用于增强美学和文本对齐。该正则化优于先前的高斯性正则化，有效防止了奖励作弊并加速了收敛。 

---
# Estimating forest carbon stocks from high-resolution remote sensing imagery by reducing domain shift with style transfer 

**Title (ZH)**: 基于样式迁移减少领域偏移的高分辨率遥感影像森林碳储量估算 

**Authors**: Zhenyu Yu, Jinnian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.00784)  

**Abstract**: Forests function as crucial carbon reservoirs on land, and their carbon sinks can efficiently reduce atmospheric CO2 concentrations and mitigate climate change. Currently, the overall trend for monitoring and assessing forest carbon stocks is to integrate ground monitoring sample data with satellite remote sensing imagery. This style of analysis facilitates large-scale observation. However, these techniques require improvement in accuracy. We used GF-1 WFV and Landsat TM images to analyze Huize County, Qujing City, Yunnan Province in China. Using the style transfer method, we introduced Swin Transformer to extract global features through attention mechanisms, converting the carbon stock estimation into an image translation. 

**Abstract (ZH)**: 森林作为陆地上的关键碳汇，其碳汇功能能有效降低大气中的CO2浓度并缓解气候变化。目前，监测和评估森林碳储量的整体趋势是将地面监测样本数据与卫星遥感图像相结合。这种分析方式有助于大规模观测，但这些技术在准确性上仍需要改进。我们使用GF-1 WFV和Landsat TM影像对中国云南省曲靖市Huize县进行分析，采用风格迁移方法，引入Swin Transformer通过注意力机制提取全局特征，将碳储量估算转化为图像翻译。 

---
