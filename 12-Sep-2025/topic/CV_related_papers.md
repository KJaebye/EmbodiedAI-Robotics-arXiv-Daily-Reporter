# ObjectReact: Learning Object-Relative Control for Visual Navigation 

**Title (ZH)**: ObjectReact: 学习对象相对控制 Woche视觉导航 

**Authors**: Sourav Garg, Dustin Craggs, Vineeth Bhat, Lachlan Mares, Stefan Podgorski, Madhava Krishna, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.09594)  

**Abstract**: Visual navigation using only a single camera and a topological map has recently become an appealing alternative to methods that require additional sensors and 3D maps. This is typically achieved through an "image-relative" approach to estimating control from a given pair of current observation and subgoal image. However, image-level representations of the world have limitations because images are strictly tied to the agent's pose and embodiment. In contrast, objects, being a property of the map, offer an embodiment- and trajectory-invariant world representation. In this work, we present a new paradigm of learning "object-relative" control that exhibits several desirable characteristics: a) new routes can be traversed without strictly requiring to imitate prior experience, b) the control prediction problem can be decoupled from solving the image matching problem, and c) high invariance can be achieved in cross-embodiment deployment for variations across both training-testing and mapping-execution settings. We propose a topometric map representation in the form of a "relative" 3D scene graph, which is used to obtain more informative object-level global path planning costs. We train a local controller, dubbed "ObjectReact", conditioned directly on a high-level "WayObject Costmap" representation that eliminates the need for an explicit RGB input. We demonstrate the advantages of learning object-relative control over its image-relative counterpart across sensor height variations and multiple navigation tasks that challenge the underlying spatial understanding capability, e.g., navigating a map trajectory in the reverse direction. We further show that our sim-only policy is able to generalize well to real-world indoor environments. Code and supplementary material are accessible via project page: this https URL 

**Abstract (ZH)**: 仅使用单个摄像头和拓扑地图的视觉导航recently成为了一种有吸引力的替代方法，这种方法无需额外传感器和3D地图。这一目标通常通过估计给定当前观察和子目标图像 pair 的“图像相对”控制来实现。然而，基于图像的世界表示存在局限性，因为图像严格依赖于代理的姿态和体现。相比之下，物体作为地图的属性，提供了体现和轨迹不变的世界表示。在本文中，我们提出了一种新的“物体相对”控制的学习范式，具有若干 desirable 特性：a) 新的路径可以被遍历而无需严格模仿先前的经验，b) 控制预测问题可以从解决图像匹配问题中解耦，c) 在训练-测试和建图-执行设置中实现高不变性。我们提出了一种拓扑地图表示，形式为“相对”的3D 场景图，用于获得更具信息量的物体级全局路径规划代价。我们训练了一个本地控制器，称为“ObjectReact”，直接基于消除显式 RGB 输入的“WayObject 成本图”高层表示。我们展示了学习物体相对控制在多种导航任务中的优势，这些任务对底层空间理解能力提出了挑战，例如逆向导航地图轨迹。我们进一步证明，仅使用模拟策略能够很好地泛化到真实世界的室内环境。代码和补充材料可通过项目页面访问：this https URL。 

---
# Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles 

**Title (ZH)**: 基于外部观察技术的驾驶员行为分类方法研究 

**Authors**: Ian Nell, Shane Gilroy  

**Link**: [PDF](https://arxiv.org/pdf/2509.09349)  

**Abstract**: Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions. 

**Abstract (ZH)**: 基于外部观察的新型驾驶行为分类系统：检测分心和受酒精及其他物质影响的指标 

---
# Model-Agnostic Open-Set Air-to-Air Visual Object Detection for Reliable UAV Perception 

**Title (ZH)**: 模型无关的开放集空对空视觉目标检测以实现可靠无人机感知 

**Authors**: Spyridon Loukovitis, Anastasios Arsenos, Vasileios Karampinis, Athanasios Voulodimos  

**Link**: [PDF](https://arxiv.org/pdf/2509.09297)  

**Abstract**: Open-set detection is crucial for robust UAV autonomy in air-to-air object detection under real-world conditions. Traditional closed-set detectors degrade significantly under domain shifts and flight data corruption, posing risks to safety-critical applications. We propose a novel, model-agnostic open-set detection framework designed specifically for embedding-based detectors. The method explicitly handles unknown object rejection while maintaining robustness against corrupted flight data. It estimates semantic uncertainty via entropy modeling in the embedding space and incorporates spectral normalization and temperature scaling to enhance open-set discrimination. We validate our approach on the challenging AOT aerial benchmark and through extensive real-world flight tests. Comprehensive ablation studies demonstrate consistent improvements over baseline methods, achieving up to a 10\% relative AUROC gain compared to standard YOLO-based detectors. Additionally, we show that background rejection further strengthens robustness without compromising detection accuracy, making our solution particularly well-suited for reliable UAV perception in dynamic air-to-air environments. 

**Abstract (ZH)**: 开放集检测是真实环境中空中目标检测实现无人机 robust 自主性的关键。传统的封闭集检测器在领域偏移和飞行数据污染下表现显著下降，这对安全关键应用构成了风险。我们提出了一种适用于嵌入式检测器的新型、模型无关的开放集检测框架。该方法明确处理未知目标的拒绝，同时保持对污染飞行数据的鲁棒性。通过嵌入空间中的熵建模估计语义不确定性，并结合频谱规范化和温度缩放以增强开放集辨别能力。我们在具有挑战性的 AOT 航空基准测试和广泛的实地飞行试验中验证了该方法。全面的消融研究证明，相对于基准方法实现了持续改进，与标准 YOLO 基础检测器相比，相对 AUROC 提升高达 10%。此外，我们展示了背景拒绝进一步增强了鲁棒性而不牺牲检测准确性，使我们的解决方案特别适合动态的空战环境中的可靠无人机感知。 

---
# Improving Video Diffusion Transformer Training by Multi-Feature Fusion and Alignment from Self-Supervised Vision Encoders 

**Title (ZH)**: 基于自我监督视觉编码器的多特征融合与对齐改进视频扩散变换器训练 

**Authors**: Dohun Lee, Hyeonho Jeong, Jiwook Kim, Duygu Ceylan, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.09547)  

**Abstract**: Video diffusion models have advanced rapidly in the recent years as a result of series of architectural innovations (e.g., diffusion transformers) and use of novel training objectives (e.g., flow matching). In contrast, less attention has been paid to improving the feature representation power of such models. In this work, we show that training video diffusion models can benefit from aligning the intermediate features of the video generator with feature representations of pre-trained vision encoders. We propose a new metric and conduct an in-depth analysis of various vision encoders to evaluate their discriminability and temporal consistency, thereby assessing their suitability for video feature alignment. Based on the analysis, we present Align4Gen which provides a novel multi-feature fusion and alignment method integrated into video diffusion model training. We evaluate Align4Gen both for unconditional and class-conditional video generation tasks and show that it results in improved video generation as quantified by various metrics. Full video results are available on our project page: this https URL 

**Abstract (ZH)**: 近年来，由于一系列架构创新（如扩散变压器）和使用新颖的训练目标（如流匹配），视频扩散模型取得了快速进展。相比之下，较少关注如何提升这些模型的特征表示能力。在本工作中，我们展示了将视频生成器的中间特征与预训练的视觉编码器的特征表示进行对齐，可以受益于视频扩散模型的训练。我们提出了一种新的评估指标，并对多种视觉编码器进行了深入分析，评估其可区分性和时序一致性，从而评估其在视频特征对齐中的适用性。基于分析，我们提出了Align4Gen，这是一种集成到视频扩散模型训练中的新型多特征融合与对齐方法。我们分别在无条件和类别条件的视频生成任务中评估了Align4Gen，并通过多种指标证实其可以提高视频生成效果。完整视频结果可在我们的项目页面查看：this https URL 

---
# Resource-Efficient Glioma Segmentation on Sub-Saharan MRI 

**Title (ZH)**: 资源高效的小glioma分割在撒哈拉以南地区的MRI图像中 

**Authors**: Freedmore Sidume, Oumayma Soula, Joseph Muthui Wacira, YunFei Zhu, Abbas Rabiu Muhammad, Abderrazek Zeraii, Oluwaseun Kalejaye, Hajer Ibrahim, Olfa Gaddour, Brain Halubanza, Dong Zhang, Udunna C Anazodo, Confidence Raymond  

**Link**: [PDF](https://arxiv.org/pdf/2509.09469)  

**Abstract**: Gliomas are the most prevalent type of primary brain tumors, and their accurate segmentation from MRI is critical for diagnosis, treatment planning, and longitudinal monitoring. However, the scarcity of high-quality annotated imaging data in Sub-Saharan Africa (SSA) poses a significant challenge for deploying advanced segmentation models in clinical workflows. This study introduces a robust and computationally efficient deep learning framework tailored for resource-constrained settings. We leveraged a 3D Attention UNet architecture augmented with residual blocks and enhanced through transfer learning from pre-trained weights on the BraTS 2021 dataset. Our model was evaluated on 95 MRI cases from the BraTS-Africa dataset, a benchmark for glioma segmentation in SSA MRI data. Despite the limited data quality and quantity, our approach achieved Dice scores of 0.76 for the Enhancing Tumor (ET), 0.80 for Necrotic and Non-Enhancing Tumor Core (NETC), and 0.85 for Surrounding Non-Functional Hemisphere (SNFH). These results demonstrate the generalizability of the proposed model and its potential to support clinical decision making in low-resource settings. The compact architecture, approximately 90 MB, and sub-minute per-volume inference time on consumer-grade hardware further underscore its practicality for deployment in SSA health systems. This work contributes toward closing the gap in equitable AI for global health by empowering underserved regions with high-performing and accessible medical imaging solutions. 

**Abstract (ZH)**: Gliomas在撒哈拉以南非洲地区的主要脑肿瘤类型，其从MRI图像中的准确分割对于诊断、治疗规划和纵向监测至关重要。然而，撒哈拉以南非洲地区高质量标注影像数据的稀缺性给高级分割模型在临床工作流程中的部署带来了重大挑战。本研究介绍了一种针对资源限制性设置优化的稳健且计算高效的深度学习框架。我们利用带有残差块的3D注意力UNet架构，并通过在BraTS 2021数据集上预训练权重进行迁移学习加以增强。该模型在BraTS-Africa数据集的95例MRI病例上进行了评估，这是一个针对撒哈拉以南非洲脑胶质瘤分割的基准数据集。尽管数据质量和服务量有限，但我们的方法在增强肿瘤(ET)、坏死和非增强肿瘤核心(NETC)以及周围非功能半球(SNFH)上的Dice分数分别为0.76、0.80和0.85。这些结果证明了所提出模型的泛化能力和在资源有限环境中支持临床决策的潜力。紧凑的架构，约90 MB，以及消费级硬件上每卷秒级的推理时间进一步突显了其在撒哈拉以南非洲卫生系统中的实用性。这项工作朝着实现全球卫生中公平的人工智能差距做出了贡献，为服务不足的地区提供了高性能且易于访问的医学影像解决方案。 

---
# Modality-Agnostic Input Channels Enable Segmentation of Brain lesions in Multimodal MRI with Sequences Unavailable During Training 

**Title (ZH)**: 模态无关的输入通道 enables 多模态MRI中不可用训练序列情况下脑病变的分割 

**Authors**: Anthony P. Addison, Felix Wagner, Wentian Xu, Natalie Voets, Konstantinos Kamnitsas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09290)  

**Abstract**: Segmentation models are important tools for the detection and analysis of lesions in brain MRI. Depending on the type of brain pathology that is imaged, MRI scanners can acquire multiple, different image modalities (contrasts). Most segmentation models for multimodal brain MRI are restricted to fixed modalities and cannot effectively process new ones at inference. Some models generalize to unseen modalities but may lose discriminative modality-specific information. This work aims to develop a model that can perform inference on data that contain image modalities unseen during training, previously seen modalities, and heterogeneous combinations of both, thus allowing a user to utilize any available imaging modalities. We demonstrate this is possible with a simple, thus practical alteration to the U-net architecture, by integrating a modality-agnostic input channel or pathway, alongside modality-specific input channels. To train this modality-agnostic component, we develop an image augmentation scheme that synthesizes artificial MRI modalities. Augmentations differentially alter the appearance of pathological and healthy brain tissue to create artificial contrasts between them while maintaining realistic anatomical integrity. We evaluate the method using 8 MRI databases that include 5 types of pathologies (stroke, tumours, traumatic brain injury, multiple sclerosis and white matter hyperintensities) and 8 modalities (T1, T1+contrast, T2, PD, SWI, DWI, ADC and FLAIR). The results demonstrate that the approach preserves the ability to effectively process MRI modalities encountered during training, while being able to process new, unseen modalities to improve its segmentation. Project code: this https URL 

**Abstract (ZH)**: 多模态脑MRI中未见模态的分割模型研究：结合模态无关输入通道的简单U-net架构改进 

---
# CoAtNeXt:An Attention-Enhanced ConvNeXtV2-Transformer Hybrid Model for Gastric Tissue Classification 

**Title (ZH)**: CoAtNeXt：一种增强注意力的ConvNeXtV2-Transformer混合模型用于胃组织分类 

**Authors**: Mustafa Yurdakul, Sakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09242)  

**Abstract**: Background and objective Early diagnosis of gastric diseases is crucial to prevent fatal outcomes. Although histopathologic examination remains the diagnostic gold standard, it is performed entirely manually, making evaluations labor-intensive and prone to variability among pathologists. Critical findings may be missed, and lack of standard procedures reduces consistency. These limitations highlight the need for automated, reliable, and efficient methods for gastric tissue analysis. Methods In this study, a novel hybrid model named CoAtNeXt was proposed for the classification of gastric tissue images. The model is built upon the CoAtNet architecture by replacing its MBConv layers with enhanced ConvNeXtV2 blocks. Additionally, the Convolutional Block Attention Module (CBAM) is integrated to improve local feature extraction through channel and spatial attention mechanisms. The architecture was scaled to achieve a balance between computational efficiency and classification performance. CoAtNeXt was evaluated on two publicly available datasets, HMU-GC-HE-30K for eight-class classification and GasHisSDB for binary classification, and was compared against 10 Convolutional Neural Networks (CNNs) and ten Vision Transformer (ViT) models. Results CoAtNeXt achieved 96.47% accuracy, 96.60% precision, 96.47% recall, 96.45% F1 score, and 99.89% AUC on HMU-GC-HE-30K. On GasHisSDB, it reached 98.29% accuracy, 98.07% precision, 98.41% recall, 98.23% F1 score, and 99.90% AUC. It outperformed all CNN and ViT models tested and surpassed previous studies in the literature. Conclusion Experimental results show that CoAtNeXt is a robust architecture for histopathological classification of gastric tissue images, providing performance on binary and multiclass. Its highlights its potential to assist pathologists by enhancing diagnostic accuracy and reducing workload. 

**Abstract (ZH)**: 背景与目的 早期诊断胃病对于预防致命后果至关重要。尽管组织病理学检查仍然是诊断的金标准，但其完全靠手动操作，使评估劳动密集型且易于病理学家之间出现变异。关键发现可能会被遗漏，缺乏标准化流程降低了一致性。这些限制突出了需要自动、可靠且高效的胃组织分析方法。方法 本文提出了一种新的混合模型CoAtNeXt，用于胃组织图像分类。该模型基于CoAtNet架构，用增强的ConvNeXtV2块替换其MBConv层，并集成了卷积块注意力模块（CBAM）以通过通道和空间注意力机制提高局部特征提取效果。架构通过权衡计算效率和分类性能进行了扩展。CoAtNeXt在两个公开可用的数据集HMU-GC-HE-30K（用于八类分类）和GasHisSDB（用于二分类）上进行了评估，并与十个卷积神经网络（CNN）和十个视觉变换器（ViT）模型进行了比较。结果 在HMU-GC-HE-30K数据集上，CoAtNeXt实现了96.47%的准确率、96.60%的精确率、96.47%的召回率、96.45%的F1分数和99.89%的AUC。在GasHisSDB数据集上，其准确率为98.29%、精确率为98.07%、召回率为98.41%、F1分数为98.23%和AUC为99.90%。CoAtNeXt在所有测试的CNN和ViT模型中表现最佳，并超越了文献中的先前研究。结论 实验结果表明，CoAtNeXt是一种稳健的架构，适用于胃组织图像的组织病理学分类，提供了二分类和多分类的性能。其潜在能力在于通过提高诊断准确性和减轻工作负担来辅助病理学家。 

---
# Virtual staining for 3D X-ray histology of bone implants 

**Title (ZH)**: 骨植入物的3D X射线组织学虚拟染色 

**Authors**: Sarah C. Irvine, Christian Lucas, Diana Krüger, Bianca Guedert, Julian Moosmann, Berit Zeller-Plumhoff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09235)  

**Abstract**: Three-dimensional X-ray histology techniques offer a non-invasive alternative to conventional 2D histology, enabling volumetric imaging of biological tissues without the need for physical sectioning or chemical staining. However, the inherent greyscale image contrast of X-ray tomography limits its biochemical specificity compared to traditional histological stains. Within digital pathology, deep learning-based virtual staining has demonstrated utility in simulating stained appearances from label-free optical images. In this study, we extend virtual staining to the X-ray domain by applying cross-modality image translation to generate artificially stained slices from synchrotron-radiation-based micro-CT scans. Using over 50 co-registered image pairs of micro-CT and toluidine blue-stained histology from bone-implant samples, we trained a modified CycleGAN network tailored for limited paired data. Whole slide histology images were downsampled to match the voxel size of the CT data, with on-the-fly data augmentation for patch-based training. The model incorporates pixelwise supervision and greyscale consistency terms, producing histologically realistic colour outputs while preserving high-resolution structural detail. Our method outperformed Pix2Pix and standard CycleGAN baselines across SSIM, PSNR, and LPIPS metrics. Once trained, the model can be applied to full CT volumes to generate virtually stained 3D datasets, enhancing interpretability without additional sample preparation. While features such as new bone formation were able to be reproduced, some variability in the depiction of implant degradation layers highlights the need for further training data and refinement. This work introduces virtual staining to 3D X-ray imaging and offers a scalable route for chemically informative, label-free tissue characterisation in biomedical research. 

**Abstract (ZH)**: 三维X射线显微断层扫描技术提供了一种与传统2D组织学相比无创的替代方案，无需物理切片或化学染色即可实现生物组织的体视成像。然而，X射线断层成像固有的灰度图像对比度使其在生物化学特异性方面逊于传统组织学染色。在数字病理学中，基于深度学习的虚拟染色已被证明能够从无标记光学图像中模拟染色外观。在本研究中，我们通过应用跨模态图像翻译将虚拟染色扩展到X射线领域，从同步辐射微CT扫描中生成人工染色切片。利用来自骨植入物样本的50多对共注册微CT和阿拉伯糖蓝染色组织学图像对，我们训练了一个针对有限配对数据修改后的CycleGAN网络。整张切片组织学图像被下采样以匹配CT数据的体素大小，并在基于补丁的训练中进行实时数据增强。该模型包含逐像素监督和平面灰度一致性项，产生具有组织学真实感的彩色输出，同时保留高分辨率的结构细节。我们的方法在SSIM、PSNR和LPIPS指标上优于Pix2Pix和标准CycleGAN基线。一旦训练完成，该模型可以应用于整个CT体积，生成虚拟染色的3D数据集，从而提高可解释性而无需额外的样品准备。虽然能够再现新的骨形成特征，但植入物降解层的某些差异显示了需要进一步训练数据和细化的必要性。本研究将虚拟染色引入三维X射线成像，并提供了一条在生物医药研究中实现化学信息性无标记组织表征的可扩展途径。 

---
# Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems 

**Title (ZH)**: 真诚的跨测试揭示了音频深度假音检测系统的薄弱环节 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Zhen Qiu, Chi Hung Chi, Kwok Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09204)  

**Abstract**: Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at this https URL. 

**Abstract (ZH)**: Authentic语音跨测试：一种新的评估框架 

---
# Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection 

**Title (ZH)**: Dark-ISP: 提升低光照条件下RAW图像处理性能以增强物体检测 

**Authors**: Jiasheng Guo, Xin Gao, Yuxiang Yan, Guanghao Li, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09183)  

**Abstract**: Low-light Object detection is crucial for many real-world applications but remains challenging due to degraded image quality. While recent studies have shown that RAW images offer superior potential over RGB images, existing approaches either use RAW-RGB images with information loss or employ complex frameworks. To address these, we propose a lightweight and self-adaptive Image Signal Processing (ISP) plugin, Dark-ISP, which directly processes Bayer RAW images in dark environments, enabling seamless end-to-end training for object detection. Our key innovations are: (1) We deconstruct conventional ISP pipelines into sequential linear (sensor calibration) and nonlinear (tone mapping) sub-modules, recasting them as differentiable components optimized through task-driven losses. Each module is equipped with content-aware adaptability and physics-informed priors, enabling automatic RAW-to-RGB conversion aligned with detection objectives. (2) By exploiting the ISP pipeline's intrinsic cascade structure, we devise a Self-Boost mechanism that facilitates cooperation between sub-modules. Through extensive experiments on three RAW image datasets, we demonstrate that our method outperforms state-of-the-art RGB- and RAW-based detection approaches, achieving superior results with minimal parameters in challenging low-light environments. 

**Abstract (ZH)**: 低光照环境下物体检测对于许多实际应用至关重要，但由于图像质量退化而仍然具有挑战性。虽然最近的研究表明，RAW图像相较于RGB图像具有更优的潜力，但现有方法要么使用包含信息丢失的RAW-RGB图像，要么采用复杂的框架。为了解决这些问题，我们提出了一种轻量级且自适应的图像信号处理（ISP）插件——Dark-ISP，该插件可以直接在暗环境处理拜耶（Bayer）RAW图像，从而实现物体检测的端到端无缝训练。我们的关键创新在于：（1）我们将传统的ISP流水线分解为顺序的线性（传感器校准）和非线性（色调映射）子模块，并将它们重新定义为通过任务驱动的损失优化的不同可微分组件。每个模块配备了内容感知的自适应能力和物理先验，能够自动将RAW图像转换为RGB图像，与检测目标对齐。（2）通过利用ISP流水线固有的级联结构，我们设计了一种自我增强机制，促进了子模块之间的协作。在三个RAW图像数据集上的广泛实验表明，我们的方法在挑战性的低光照环境下优于最先进的RGB和RAW基检测方法，在最少参数的情况下取得了优越的结果。 

---
# A Knowledge Noise Mitigation Framework for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于知识的视觉问答中知识噪声 mitigation 的框架 

**Authors**: Zhiyue Liu, Sihang Liu, Jinyuan Liu, Xinru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09159)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) requires a model to understand images and utilize external knowledge to provide accurate answers. Existing approaches often directly augment models with retrieved information from knowledge sources while ignoring substantial knowledge redundancy, which introduces noise into the answering process. To address this, we propose a training-free framework with knowledge focusing for KB-VQA, that mitigates the impact of noise by enhancing knowledge relevance and reducing redundancy. First, for knowledge retrieval, our framework concludes essential parts from the image-question pairs, creating low-noise queries that enhance the retrieval of highly relevant knowledge. Considering that redundancy still persists in the retrieved knowledge, we then prompt large models to identify and extract answer-beneficial segments from knowledge. In addition, we introduce a selective knowledge integration strategy, allowing the model to incorporate knowledge only when it lacks confidence in answering the question, thereby mitigating the influence of redundant information. Our framework enables the acquisition of accurate and critical knowledge, and extensive experiments demonstrate that it outperforms state-of-the-art methods. 

**Abstract (ZH)**: 基于知识的视觉问答（KB-VQA）要求模型理解图像并利用外部知识提供准确的答案。现有的方法往往直接通过从知识源中检索信息来增强模型，但忽视了知识冗余，引入了噪声。为了解决这个问题，我们提出了一种无需训练的知识聚焦框架，通过增强知识的相关性和减少冗余来减轻噪声的影响。首先，在知识检索方面，我们的框架从图像-问题对中得出关键部分，创建低噪声查询以增强高度相关知识的检索。考虑到检索的知识中仍然存在冗余，我们随后促使大模型识别并提取有益于答案的片段。此外，我们引入了一种选择性知识整合策略，允许模型仅在缺乏回答问题的信心时才整合知识，从而减轻冗余信息的影响。该框架能够获取准确和关键的知识，并且大量的实验表明，它优于现有最佳方法。 

---
# OCELOT 2023: Cell Detection from Cell-Tissue Interaction Challenge 

**Title (ZH)**: OCELOT 2023: 细胞检测从细胞-组织相互作用挑战赛 

**Authors**: JaeWoong Shin, Jeongun Ryu, Aaron Valero Puche, Jinhee Lee, Biagio Brattoli, Wonkyung Jung, Soo Ick Cho, Kyunghyun Paeng, Chan-Young Ock, Donggeun Yoo, Zhaoyang Li, Wangkai Li, Huayu Mai, Joshua Millward, Zhen He, Aiden Nibali, Lydia Anette Schoenpflug, Viktor Hendrik Koelzer, Xu Shuoyu, Ji Zheng, Hu Bin, Yu-Wen Lo, Ching-Hui Yang, Sérgio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2509.09153)  

**Abstract**: Pathologists routinely alternate between different magnifications when examining Whole-Slide Images, allowing them to evaluate both broad tissue morphology and intricate cellular details to form comprehensive diagnoses. However, existing deep learning-based cell detection models struggle to replicate these behaviors and learn the interdependent semantics between structures at different magnifications. A key barrier in the field is the lack of datasets with multi-scale overlapping cell and tissue annotations. The OCELOT 2023 challenge was initiated to gather insights from the community to validate the hypothesis that understanding cell and tissue (cell-tissue) interactions is crucial for achieving human-level performance, and to accelerate the research in this field. The challenge dataset includes overlapping cell detection and tissue segmentation annotations from six organs, comprising 673 pairs sourced from 306 The Cancer Genome Atlas (TCGA) Whole-Slide Images with hematoxylin and eosin staining, divided into training, validation, and test subsets. Participants presented models that significantly enhanced the understanding of cell-tissue relationships. Top entries achieved up to a 7.99 increase in F1-score on the test set compared to the baseline cell-only model that did not incorporate cell-tissue relationships. This is a substantial improvement in performance over traditional cell-only detection methods, demonstrating the need for incorporating multi-scale semantics into the models. This paper provides a comparative analysis of the methods used by participants, highlighting innovative strategies implemented in the OCELOT 2023 challenge. 

**Abstract (ZH)**: 病理学家在检查全视野图像时会交替使用不同的放大倍数，以便评估组织的宏观结构和细胞的细微细节，从而形成全面的诊断。然而，现有的基于深度学习的细胞检测模型难以复制这些行为，并学习不同放大倍数下结构之间的相互依赖语义。领域内的主要障碍是没有多尺度重叠细胞和组织注释的数据集。2023年OCELOT挑战赛旨在从社区中获得见解，验证理解细胞和组织（细胞-组织）相互作用对实现人类水平性能是至关重要的假说，并加速该领域的研究。挑战数据集包括来自306张苏木精和伊红染色的癌症基因组图谱（TCGA）全视野图像中六种器官的重叠细胞检测和组织分割注释，共计673对，分为训练、验证和测试子集。参与者展示了显著增强细胞-组织关系理解的模型。顶级参赛作品在测试集上的F1分数比不考虑细胞-组织关系的基本细胞检测模型提高了7.99%，这一性能提升显著优于传统的细胞检测方法，证明了将多尺度语义纳入模型的必要性。本文对OCELOT 2023挑战赛中参赛者使用的各种方法进行了比较分析，突出了该挑战赛中实施的创新策略。 

---
# Video Understanding by Design: How Datasets Shape Architectures and Insights 

**Title (ZH)**: 设计中的视频理解：数据集如何塑造架构与洞察 

**Authors**: Lei Wang, Piotr Koniusz, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09151)  

**Abstract**: Video understanding has advanced rapidly, fueled by increasingly complex datasets and powerful architectures. Yet existing surveys largely classify models by task or family, overlooking the structural pressures through which datasets guide architectural evolution. This survey is the first to adopt a dataset-driven perspective, showing how motion complexity, temporal span, hierarchical composition, and multimodal richness impose inductive biases that models should encode. We reinterpret milestones, from two-stream and 3D CNNs to sequential, transformer, and multimodal foundation models, as concrete responses to these dataset-driven pressures. Building on this synthesis, we offer practical guidance for aligning model design with dataset invariances while balancing scalability and task demands. By unifying datasets, inductive biases, and architectures into a coherent framework, this survey provides both a comprehensive retrospective and a prescriptive roadmap for advancing general-purpose video understanding. 

**Abstract (ZH)**: 基于数据集视角的视频理解模型演化综述：结构压力下的引致偏置与模型设计指南 

---
# Objectness Similarity: Capturing Object-Level Fidelity in 3D Scene Evaluation 

**Title (ZH)**: 对象相似性：3D 场景评估中的对象级保真度捕获 

**Authors**: Yuiko Uchida, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2509.09143)  

**Abstract**: This paper presents Objectness SIMilarity (OSIM), a novel evaluation metric for 3D scenes that explicitly focuses on "objects," which are fundamental units of human visual perception. Existing metrics assess overall image quality, leading to discrepancies with human perception. Inspired by neuropsychological insights, we hypothesize that human recognition of 3D scenes fundamentally involves attention to individual objects. OSIM enables object-centric evaluations by leveraging an object detection model and its feature representations to quantify the "objectness" of each object in the scene. Our user study demonstrates that OSIM aligns more closely with human perception compared to existing metrics. We also analyze the characteristics of OSIM using various approaches. Moreover, we re-evaluate recent 3D reconstruction and generation models under a standardized experimental setup to clarify advancements in this field. The code is available at this https URL. 

**Abstract (ZH)**: Objectness SIMilarity: 一种针对3D场景的新颖评估指标 

---
# Can Vision-Language Models Solve Visual Math Equations? 

**Title (ZH)**: 视觉-语言模型能否解决视觉数学方程？ 

**Authors**: Monjoy Narayan Choudhury, Junling Wang, Yifan Hou, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09013)  

**Abstract**: Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning. 

**Abstract (ZH)**: 尽管视觉语言模型在视觉理解和语言推理方面表现出色，但在需要整合感知与符号计算的任务中表现不佳。我们通过可视化方程求解任务来研究这一限制，在该任务中，数学方程嵌入图像中，变量由对象图示表示，系数必须通过计数推断。尽管视觉语言模型在文本方程方面表现良好，但在基于视觉的任务中却失败了。为了理解这一差距，我们将任务拆解为系数计数和变量识别，并发现计数是主要瓶颈，即使识别准确也是如此。我们还观察到，组合识别与推理会引入额外错误，突显了多步骤视觉推理的挑战。最后，随着方程复杂性的增加，符号推理本身也成为了限制因素。这些发现揭示了当前视觉语言模型的关键薄弱环节，并指出了未来提高基于视觉的数学推理能力的方向。 

---
