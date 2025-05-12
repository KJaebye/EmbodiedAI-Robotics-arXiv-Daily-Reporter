# 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks 

**Title (ZH)**: 3D CAVLA: 利用深度和三维上下文泛化视觉语言动作模型以应对未知任务 

**Authors**: Vineet Bhat, Yu-Hsiang Lan, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2505.05800)  

**Abstract**: Robotic manipulation in 3D requires learning an $N$ degree-of-freedom joint space trajectory of a robot manipulator. Robots must possess semantic and visual perception abilities to transform real-world mappings of their workspace into the low-level control necessary for object manipulation. Recent work has demonstrated the capabilities of fine-tuning large Vision-Language Models (VLMs) to learn the mapping between RGB images, language instructions, and joint space control. These models typically take as input RGB images of the workspace and language instructions, and are trained on large datasets of teleoperated robot demonstrations. In this work, we explore methods to improve the scene context awareness of a popular recent Vision-Language-Action model by integrating chain-of-thought reasoning, depth perception, and task-oriented region of interest detection. Our experiments in the LIBERO simulation environment show that our proposed model, 3D-CAVLA, improves the success rate across various LIBERO task suites, achieving an average success rate of 98.1$\%$. We also evaluate the zero-shot capabilities of our method, demonstrating that 3D scene awareness leads to robust learning and adaptation for completely unseen tasks. 3D-CAVLA achieves an absolute improvement of 8.8$\%$ on unseen tasks. We will open-source our code and the unseen tasks dataset to promote community-driven research here: this https URL 

**Abstract (ZH)**: 三维空间中的机器人操作需要学习一个具有N个自由度的机器人操作臂关节空间轨迹。机器人必须具备语义和视觉感知能力，将工作空间的实际映射转换为用于物体操作的低级控制。近期的研究表明，可以通过微调大型视觉-语言模型（VLMs）来学习RGB图像、语言指令与关节空间控制之间的映射关系。这些模型通常以工作空间的RGB图像和语言指令作为输入，并在大量的遥操作机器人演示数据集上进行训练。在这项工作中，我们通过集成链式思维推理、深度感知和任务导向的兴趣区域检测方法，探讨了提高一种流行的视觉-语言-行动模型的场景上下文意识的方法。我们在LIBERO仿真环境中进行的实验显示，我们提出的3D-CAVLA模型在各种LIBERO任务套件中提高了成功率，平均成功率为98.1%。我们也评估了该方法的零样本能力，证明了三维场景感知能够使模型在完全未见过的任务中表现出鲁棒的学习和适应能力。3D-CAVLA在未见过的任务中实现了8.8%的绝对改进。我们将开源我们的代码和未见过的任务数据集，以促进社区驱动的研究：this https URL。 

---
# VIN-NBV: A View Introspection Network for Next-Best-View Selection for Resource-Efficient 3D Reconstruction 

**Title (ZH)**: VIN-NBV：一种视角 introspection 网络，用于资源高效 3D 重建的下一次最佳视角选择 

**Authors**: Noah Frahm, Dongxu Zhao, Andrea Dunn Beltran, Ron Alterovitz, Jan-Michael Frahm, Junier Oliva, Roni Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06219)  

**Abstract**: Next Best View (NBV) algorithms aim to acquire an optimal set of images using minimal resources, time, or number of captures to enable efficient 3D reconstruction of a scene. Existing approaches often rely on prior scene knowledge or additional image captures and often develop policies that maximize coverage. Yet, for many real scenes with complex geometry and self-occlusions, coverage maximization does not lead to better reconstruction quality directly. In this paper, we propose the View Introspection Network (VIN), which is trained to predict the reconstruction quality improvement of views directly, and the VIN-NBV policy. A greedy sequential sampling-based policy, where at each acquisition step, we sample multiple query views and choose the one with the highest VIN predicted improvement score. We design the VIN to perform 3D-aware featurization of the reconstruction built from prior acquisitions, and for each query view create a feature that can be decoded into an improvement score. We then train the VIN using imitation learning to predict the reconstruction improvement score. We show that VIN-NBV improves reconstruction quality by ~30% over a coverage maximization baseline when operating with constraints on the number of acquisitions or the time in motion. 

**Abstract (ZH)**: View Introspection Network (VIN) and VIN-NBV Policy for Efficient 3D Reconstruction 

---
# Automating Infrastructure Surveying: A Framework for Geometric Measurements and Compliance Assessment Using Point Cloud Data 

**Title (ZH)**: 基于点云数据的基础设施测量与合规性评估自动化框架 

**Authors**: Amin Ghafourian, Andrew Lee, Dechen Gao, Tyler Beer, Kin Yen, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2505.05752)  

**Abstract**: Automation can play a prominent role in improving efficiency, accuracy, and scalability in infrastructure surveying and assessing construction and compliance standards. This paper presents a framework for automation of geometric measurements and compliance assessment using point cloud data. The proposed approach integrates deep learning-based detection and segmentation, in conjunction with geometric and signal processing techniques, to automate surveying tasks. As a proof of concept, we apply this framework to automatically evaluate the compliance of curb ramps with the Americans with Disabilities Act (ADA), demonstrating the utility of point cloud data in survey automation. The method leverages a newly collected, large annotated dataset of curb ramps, made publicly available as part of this work, to facilitate robust model training and evaluation. Experimental results, including comparison with manual field measurements of several ramps, validate the accuracy and reliability of the proposed method, highlighting its potential to significantly reduce manual effort and improve consistency in infrastructure assessment. Beyond ADA compliance, the proposed framework lays the groundwork for broader applications in infrastructure surveying and automated construction evaluation, promoting wider adoption of point cloud data in these domains. The annotated database, manual ramp survey data, and developed algorithms are publicly available on the project's GitHub page: this https URL. 

**Abstract (ZH)**: 自动化在提高基础设施测量和施工合规性评估效率、准确性和可扩展性方面可以发挥重要作用。本文提出了一种基于点云数据的自动化几何测量和合规评估框架。所提出的方法结合了基于深度学习的检测和分割技术，以及几何和信号处理技术，以自动化测绘任务。作为概念验证，我们将此框架应用于自动评估缘石坡道的Americans with Disabilities Act (ADA) 合规性，展示了点云数据在测绘自动化中的应用价值。该方法利用了一个新收集的大型标注数据集——缘石坡道数据集，该数据集作为本工作的成果之一已公开发布，以促进稳健模型的训练和评估。实验结果，包括与几个斜坡的手动现场测量结果的比较，验证了所提出方法的准确性和可靠性，突显了其显著减少人工努力并提高基础设施评估一致性方面的潜力。除了ADA合规性，所提出框架为更广泛的基础设施测绘和自动化施工评估应用奠定了基础，促进了点云数据在这些领域的更广泛应用。标注数据库、手动斜坡测绘数据和开发的算法可在项目的GitHub页面上公开获取：this https URL。 

---
# Web2Grasp: Learning Functional Grasps from Web Images of Hand-Object Interactions 

**Title (ZH)**: Web2Grasp: 从手物交互网页图像中学习功能抓取 

**Authors**: Hongyi Chen, Yunchao Yao, Yufei Ye, Zhixuan Xu, Homanga Bharadhwaj, Jiashun Wang, Shubham Tulsiani, Zackory Erickson, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.05517)  

**Abstract**: Functional grasp is essential for enabling dexterous multi-finger robot hands to manipulate objects effectively. However, most prior work either focuses on power grasping, which simply involves holding an object still, or relies on costly teleoperated robot demonstrations to teach robots how to grasp each object functionally. Instead, we propose extracting human grasp information from web images since they depict natural and functional object interactions, thereby bypassing the need for curated demonstrations. We reconstruct human hand-object interaction (HOI) 3D meshes from RGB images, retarget the human hand to multi-finger robot hands, and align the noisy object mesh with its accurate 3D shape. We show that these relatively low-quality HOI data from inexpensive web sources can effectively train a functional grasping model. To further expand the grasp dataset for seen and unseen objects, we use the initially-trained grasping policy with web data in the IsaacGym simulator to generate physically feasible grasps while preserving functionality. We train the grasping model on 10 object categories and evaluate it on 9 unseen objects, including challenging items such as syringes, pens, spray bottles, and tongs, which are underrepresented in existing datasets. The model trained on the web HOI dataset, achieving a 75.8% success rate on seen objects and 61.8% across all objects in simulation, with a 6.7% improvement in success rate and a 1.8x increase in functionality ratings over baselines. Simulator-augmented data further boosts performance from 61.8% to 83.4%. The sim-to-real transfer to the LEAP Hand achieves a 85% success rate. Project website is at: this https URL. 

**Abstract (ZH)**: 基于互联网图像的功能性抓取在多指机器人手上的应用研究 

---
# The Application of Deep Learning for Lymph Node Segmentation: A Systematic Review 

**Title (ZH)**: 深度学习在淋巴结分割中的应用：一项系统性回顾 

**Authors**: Jingguo Qu, Xinyang Han, Man-Lik Chui, Yao Pu, Simon Takadiyi Gunda, Ziman Chen, Jing Qin, Ann Dorothy King, Winnie Chiu-Wing Chu, Jing Cai, Michael Tin-Cheung Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.06118)  

**Abstract**: Automatic lymph node segmentation is the cornerstone for advances in computer vision tasks for early detection and staging of cancer. Traditional segmentation methods are constrained by manual delineation and variability in operator proficiency, limiting their ability to achieve high accuracy. The introduction of deep learning technologies offers new possibilities for improving the accuracy of lymph node image analysis. This study evaluates the application of deep learning in lymph node segmentation and discusses the methodologies of various deep learning architectures such as convolutional neural networks, encoder-decoder networks, and transformers in analyzing medical imaging data across different modalities. Despite the advancements, it still confronts challenges like the shape diversity of lymph nodes, the scarcity of accurately labeled datasets, and the inadequate development of methods that are robust and generalizable across different imaging modalities. To the best of our knowledge, this is the first study that provides a comprehensive overview of the application of deep learning techniques in lymph node segmentation task. Furthermore, this study also explores potential future research directions, including multimodal fusion techniques, transfer learning, and the use of large-scale pre-trained models to overcome current limitations while enhancing cancer diagnosis and treatment planning strategies. 

**Abstract (ZH)**: 自动淋巴结分割是推进早期癌症检测和分期的计算机视觉任务中的基石。传统的分割方法受限于手动勾画和操作者熟练程度的差异，限制了其达到高准确性的能力。深度学习技术的引入为提高淋巴结图像分析的准确性提供了新的可能性。本研究评估了深度学习在淋巴结分割中的应用，并讨论了各种深度学习架构（如卷积神经网络、编码器-解码器网络和变压器）在不同医学成像模态数据分析中的方法学。尽管取得了进展，但仍然面临淋巴结形状多样性、准确标注数据集稀缺以及方法难以在不同成像模态之间稳健且普适的问题。据我们所知，这是第一篇全面概述深度学习技术在淋巴结分割任务中应用的研究。此外，本研究还探讨了潜在的未来研究方向，包括多模态融合技术、迁移学习以及使用大规模预训练模型来克服当前局限性，从而增强癌症诊断和治疗规划策略。 

---
# Achieving 3D Attention via Triplet Squeeze and Excitation Block 

**Title (ZH)**: 通过三重挤压与激励模块实现3D注意力 

**Authors**: Maan Alhazmi, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05943)  

**Abstract**: The emergence of ConvNeXt and its variants has reaffirmed the conceptual and structural suitability of CNN-based models for vision tasks, re-establishing them as key players in image classification in general, and in facial expression recognition (FER) in particular. In this paper, we propose a new set of models that build on these advancements by incorporating a new set of attention mechanisms that combines Triplet attention with Squeeze-and-Excitation (TripSE) in four different variants. We demonstrate the effectiveness of these variants by applying them to the ResNet18, DenseNet and ConvNext architectures to validate their versatility and impact. Our study shows that incorporating a TripSE block in these CNN models boosts their performances, particularly for the ConvNeXt architecture, indicating its utility. We evaluate the proposed mechanisms and associated models across four datasets, namely CIFAR100, ImageNet, FER2013 and AffectNet datasets, where ConvNext with TripSE achieves state-of-the-art results with an accuracy of \textbf{78.27\%} on the popular FER2013 dataset, a new feat for this dataset. 

**Abstract (ZH)**: ConvNeXt及其变种的出现再次证实了基于CNN的模型在视觉任务中的概念和结构适宜性，重新确立了它们在图像分类以及面部表情识别（FER）中的关键地位。本文提出了一种新的模型集合，通过将Triple Attention与Squeeze-and-Excitation (TripSE)结合，在四种不同的变体中引入新的注意力机制。我们通过将这些变体应用于ResNet18、DenseNet和ConvNeXt架构，验证了它们的通用性和影响。研究表明，在这些CNN模型中加入TripSE模块可以提升其性能，特别是在ConvNeXt架构中效果显著，显示了其实用性。我们在CIFAR100、ImageNet、FER2013和AffectNet四个数据集中评估了所提出机制及其相关模型，其中使用ConvNeXt和TripSE的模型在流行的数据集FER2013上取得了78.27%的准确率，是该数据集的一个新成就。 

---
# Towards Facial Image Compression with Consistency Preserving Diffusion Prior 

**Title (ZH)**: 面向一致性保留扩散先验的面部图像压缩 

**Authors**: Yimin Zhou, Yichong Xia, Bin Chen, Baoyi An, Haoqian Wang, Zhi Wang, Yaowei Wang, Zikun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.05870)  

**Abstract**: With the widespread application of facial image data across various domains, the efficient storage and transmission of facial images has garnered significant attention. However, the existing learned face image compression methods often produce unsatisfactory reconstructed image quality at low bit rates. Simply adapting diffusion-based compression methods to facial compression tasks results in reconstructed images that perform poorly in downstream applications due to insufficient preservation of high-frequency information. To further explore the diffusion prior in facial image compression, we propose Facial Image Compression with a Stable Diffusion Prior (FaSDiff), a method that preserves consistency through frequency enhancement. FaSDiff employs a high-frequency-sensitive compressor in an end-to-end framework to capture fine image details and produce robust visual prompts. Additionally, we introduce a hybrid low-frequency enhancement module that disentangles low-frequency facial semantics and stably modulates the diffusion prior alongside visual prompts. The proposed modules allow FaSDiff to leverage diffusion priors for superior human visual perception while minimizing performance loss in machine vision due to semantic inconsistency. Extensive experiments show that FaSDiff outperforms state-of-the-art methods in balancing human visual quality and machine vision accuracy. The code will be released after the paper is accepted. 

**Abstract (ZH)**: 面部图像压缩中稳定扩散先验的方法（FaSDiff） 

---
# Enhancing Satellite Object Localization with Dilated Convolutions and Attention-aided Spatial Pooling 

**Title (ZH)**: 使用膨胀卷积和注意力辅助空间池化增强卫星目标定位 

**Authors**: Seraj Al Mahmud Mostafa, Chenxi Wang, Jia Yue, Yuta Hozumi, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05599)  

**Abstract**: Object localization in satellite imagery is particularly challenging due to the high variability of objects, low spatial resolution, and interference from noise and dominant features such as clouds and city lights. In this research, we focus on three satellite datasets: upper atmospheric Gravity Waves (GW), mesospheric Bores (Bore), and Ocean Eddies (OE), each presenting its own unique challenges. These challenges include the variability in the scale and appearance of the main object patterns, where the size, shape, and feature extent of objects of interest can differ significantly. To address these challenges, we introduce YOLO-DCAP, a novel enhanced version of YOLOv5 designed to improve object localization in these complex scenarios. YOLO-DCAP incorporates a Multi-scale Dilated Residual Convolution (MDRC) block to capture multi-scale features at scale with varying dilation rates, and an Attention-aided Spatial Pooling (AaSP) module to focus on the global relevant spatial regions, enhancing feature selection. These structural improvements help to better localize objects in satellite imagery. Experimental results demonstrate that YOLO-DCAP significantly outperforms both the YOLO base model and state-of-the-art approaches, achieving an average improvement of 20.95% in mAP50 and 32.23% in IoU over the base model, and 7.35% and 9.84% respectively over state-of-the-art alternatives, consistently across all three satellite datasets. These consistent gains across all three satellite datasets highlight the robustness and generalizability of the proposed approach. Our code is open sourced at this https URL. 

**Abstract (ZH)**: 卫星影像中的目标定位特别具有挑战性，由于对象的高度变异性、低空间分辨率，以及来自噪声、云彩和城市灯光等主要特征的干扰。本研究聚焦于三种卫星数据集：高层大气重力波（GW）、中层大气波（Bore）和海洋涡旋（OE），每种数据集都具有其独特的挑战。这些挑战包括主要对象模式在尺度和外观上的变异性，导致感兴趣对象的大小、形状和特征扩展范围存在显著差异。为应对这些挑战，我们介绍了一种名为YOLO-DCAP的新型增强版YOLOv5，旨在改善这些复杂场景中的目标定位能力。YOLO-DCAP结合了多尺度空洞残差卷积（MDRC）模块来捕捉不同膨胀率下的多尺度特征，并结合了注意力辅助空间聚类（AaSP）模块以聚焦于全局相关空间区域，从而增强特征选择。结构上的改进有助于在卫星影像中更好地定位物体。实验结果显示，YOLO-DCAP在mAP50和IoU方面显著优于YOLO基模型和最先进的方法，分别提高了20.95%和32.23%，相对于最先进的替代方法，分别提高了7.35%和9.84%，在所有三个卫星数据集中表现出一致的改善。这些一致的收益突显了所提出方法的稳健性和泛化能力。我们的代码已开源，可通过此链接获取：https://github.com/Alibaba-Qwen/YOLO-DCAP 

---
# ReactDance: Progressive-Granular Representation for Long-Term Coherent Reactive Dance Generation 

**Title (ZH)**: ReactDance：渐进细粒度表示生成长期连贯反应舞蹈 

**Authors**: Jingzhong Lin, Yuanyuan Qi, Xinru Li, Wenxuan Huang, Xiangfeng Xu, Bangyan Li, Xuejiao Wang, Gaoqi He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05589)  

**Abstract**: Reactive dance generation (RDG) produces follower movements conditioned on guiding dancer and music while ensuring spatial coordination and temporal coherence. However, existing methods overemphasize global constraints and optimization, overlooking local information, such as fine-grained spatial interactions and localized temporal context. Therefore, we present ReactDance, a novel diffusion-based framework for high-fidelity RDG with long-term coherence and multi-scale controllability. Unlike existing methods that struggle with interaction fidelity, synchronization, and temporal consistency in duet synthesis, our approach introduces two key innovations: 1)Group Residual Finite Scalar Quantization (GRFSQ), a multi-scale disentangled motion representation that captures interaction semantics from coarse body rhythms to fine-grained joint dynamics, and 2)Blockwise Local Context (BLC), a sampling strategy eliminating error accumulation in long sequence generation via local block causal masking and periodic positional encoding. Built on the decoupled multi-scale GRFSQ representation, we implement a diffusion model withLayer-Decoupled Classifier-free Guidance (LDCFG), allowing granular control over motion semantics across scales. Extensive experiments on standard benchmarks demonstrate that ReactDance surpasses existing methods, achieving state-of-the-art performance. 

**Abstract (ZH)**: 基于扩散的高保真反应舞动生成框架ReactDance 

---
# Prompt to Polyp: Clinically-Aware Medical Image Synthesis with Diffusion Models 

**Title (ZH)**: 从息肉到息肉：基于临床意识的医疗图像合成 

**Authors**: Mikhail Chaichuk, Sushant Gautam, Steven Hicks, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2505.05573)  

**Abstract**: The generation of realistic medical images from text descriptions has significant potential to address data scarcity challenges in healthcare AI while preserving patient privacy. This paper presents a comprehensive study of text-to-image synthesis in the medical domain, comparing two distinct approaches: (1) fine-tuning large pre-trained latent diffusion models and (2) training small, domain-specific models. We introduce a novel model named MSDM, an optimized architecture based on Stable Diffusion that integrates a clinical text encoder, variational autoencoder, and cross-attention mechanisms to better align medical text prompts with generated images. Our study compares two approaches: fine-tuning large pre-trained models (FLUX, Kandinsky) versus training compact domain-specific models (MSDM). Evaluation across colonoscopy (MedVQA-GI) and radiology (ROCOv2) datasets reveals that while large models achieve higher fidelity, our optimized MSDM delivers comparable quality with lower computational costs. Quantitative metrics and qualitative evaluations by medical experts reveal strengths and limitations of each approach. 

**Abstract (ZH)**: 从文本描述生成真实医疗图像在医疗保健AI中具有解决数据稀缺挑战的潜在价值，同时保护患者隐私。本文对医疗领域的文本到图像合成进行了全面研究，比较了两种不同的方法：（1）微调大型预训练潜扩散模型和（2）训练小型领域特定模型。我们介绍了一种名为MSDM的新模型，这是一种基于Stable Diffusion的优化架构，集成了临床文本编码器、变分自编码器和交叉注意力机制，以更好地使医学文本提示与生成的图像对齐。我们的研究比较了两种方法：微调大型预训练模型（FLUX，Kandinsky）与训练紧凑的领域特定模型（MSDM）。在结肠镜检查（MedVQA-GI）和放射学（ROCOv2）数据集上的评估表明，虽然大型模型具有更高的保真度，但我们的优化MSDM模型以较低的计算成本提供了可比较的质量。定量指标和医学专家的定性评估揭示了每种方法的优缺点。 

---
# GaMNet: A Hybrid Network with Gabor Fusion and NMamba for Efficient 3D Glioma Segmentation 

**Title (ZH)**: GaMNet: 结合Gabor融合和NMamba的高效脑胶质瘤三维分割混合网络 

**Authors**: Chengwei Ye, Huanzhen Zhang, Yufei Lin, Kangsheng Wang, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05520)  

**Abstract**: Gliomas are aggressive brain tumors that pose serious health risks. Deep learning aids in lesion segmentation, but CNN and Transformer-based models often lack context modeling or demand heavy computation, limiting real-time use on mobile medical devices. We propose GaMNet, integrating the NMamba module for global modeling and a multi-scale CNN for efficient local feature extraction. To improve interpretability and mimic the human visual system, we apply Gabor filters at multiple scales. Our method achieves high segmentation accuracy with fewer parameters and faster computation. Extensive experiments show GaMNet outperforms existing methods, notably reducing false positives and negatives, which enhances the reliability of clinical diagnosis. 

**Abstract (ZH)**: 基于NMamba模块和多尺度CNN的GaMNet在胶质瘤分割中的应用 

---
# DetoxAI: a Python Toolkit for Debiasing Deep Learning Models in Computer Vision 

**Title (ZH)**: DetoxAI：计算机视觉中深度学习模型去偏见的Python工具包 

**Authors**: Ignacy Stępka, Lukasz Sztukiewicz, Michał Wiliński, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.05492)  

**Abstract**: While machine learning fairness has made significant progress in recent years, most existing solutions focus on tabular data and are poorly suited for vision-based classification tasks, which rely heavily on deep learning. To bridge this gap, we introduce DetoxAI, an open-source Python library for improving fairness in deep learning vision classifiers through post-hoc debiasing. DetoxAI implements state-of-the-art debiasing algorithms, fairness metrics, and visualization tools. It supports debiasing via interventions in internal representations and includes attribution-based visualization tools and quantitative algorithmic fairness metrics to show how bias is mitigated. This paper presents the motivation, design, and use cases of DetoxAI, demonstrating its tangible value to engineers and researchers. 

**Abstract (ZH)**: 尽管机器学习公平性在近年来取得了显著进展，但现有的大多数解决方案主要针对表格式数据，而不适合依赖深度学习的视觉分类任务。为解决这一问题，我们引入了DetoxAI，这是一个用于通过后处理去偏见来提高深度学习视觉分类器公平性的开源Python库。DetoxAI实现了最先进的去偏见算法、公平性指标和可视化工具。它支持通过内部表示的干预进行去偏见，并包括基于 Attribution 的可视化工具和定量的算法公平性指标，以展示如何减轻偏见。本文介绍了DetoxAI的动机、设计和应用场景，展示了其对工程师和研究人员的实际价值。 

---
# MDDFNet: Mamba-based Dynamic Dual Fusion Network for Traffic Sign Detection 

**Title (ZH)**: MDDFNet：基于Mamba的动态双模融合网络用于交通标志检测 

**Authors**: TianYi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05491)  

**Abstract**: The Detection of small objects, especially traffic signs, is a critical sub-task in object detection and autonomous driving. Despite signficant progress in previous research, two main challenges remain. First, the issue of feature extraction being too singular. Second, the detection process struggles to efectively handle objects of varying sizes or scales. These problems are also prevalent in general object detection tasks. To address these challenges, we propose a novel object detection network, Mamba-based Dynamic Dual Fusion Network (MDDFNet), for traffic sign detection. The network integrates a dynamic dual fusion module and a Mamba-based backbone to simultaneously tackle the aforementioned issues. Specifically, the dynamic dual fusion module utilizes multiple branches to consolidate various spatial and semantic information, thus enhancing feature diversity. The Mamba-based backbone leverages global feature fusion and local feature interaction, combining features in an adaptive manner to generate unique classification characteristics. Extensive experiments conducted on the TT100K (Tsinghua-Tencent 100K) datasets demonstrate that MDDFNet outperforms other state-of-the-art detectors, maintaining real-time processing capabilities of single-stage models while achieving superior performance. This confirms the efectiveness of MDDFNet in detecting small traffic signs. 

**Abstract (ZH)**: 基于Mamba的动态双分支融合网络在交通标志检测中的应用 

---
