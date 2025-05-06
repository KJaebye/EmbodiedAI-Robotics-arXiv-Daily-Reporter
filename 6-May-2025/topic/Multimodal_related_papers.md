# HapticVLM: VLM-Driven Texture Recognition Aimed at Intelligent Haptic Interaction 

**Title (ZH)**: HapticVLM: 面向智能触觉交互的VLM驱动的纹理识别 

**Authors**: Muhammad Haris Khan, Miguel Altamirano Cabrera, Dmitrii Iarchuk, Yara Mahmoud, Daria Trinitatova, Issatay Tokmurziyev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.02569)  

**Abstract**: This paper introduces HapticVLM, a novel multimodal system that integrates vision-language reasoning with deep convolutional networks to enable real-time haptic feedback. HapticVLM leverages a ConvNeXt-based material recognition module to generate robust visual embeddings for accurate identification of object materials, while a state-of-the-art Vision-Language Model (Qwen2-VL-2B-Instruct) infers ambient temperature from environmental cues. The system synthesizes tactile sensations by delivering vibrotactile feedback through speakers and thermal cues via a Peltier module, thereby bridging the gap between visual perception and tactile experience. Experimental evaluations demonstrate an average recognition accuracy of 84.67% across five distinct auditory-tactile patterns and a temperature estimation accuracy of 86.7% based on a tolerance-based evaluation method with an 8°C margin of error across 15 scenarios. Although promising, the current study is limited by the use of a small set of prominent patterns and a modest participant pool. Future work will focus on expanding the range of tactile patterns and increasing user studies to further refine and validate the system's performance. Overall, HapticVLM presents a significant step toward context-aware, multimodal haptic interaction with potential applications in virtual reality, and assistive technologies. 

**Abstract (ZH)**: HapticVLM：一种将视觉-语言推理与深度卷积网络结合的新型多模态系统 

---
# Multimodal Graph Representation Learning for Robust Surgical Workflow Recognition with Adversarial Feature Disentanglement 

**Title (ZH)**: 多模态图表示学习在对抗特征解耦下的手术工作流程robust识别 

**Authors**: Long Bai, Boyi Ma, Ruohan Wang, Guankun Wang, Beilei Cui, Zhongliang Jiang, Mobarakol Islam, Zhe Min, Jiewen Lai, Nassir Navab, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.01766)  

**Abstract**: Surgical workflow recognition is vital for automating tasks, supporting decision-making, and training novice surgeons, ultimately improving patient safety and standardizing procedures. However, data corruption can lead to performance degradation due to issues like occlusion from bleeding or smoke in surgical scenes and problems with data storage and transmission. In this case, we explore a robust graph-based multimodal approach to integrating vision and kinematic data to enhance accuracy and reliability. Vision data captures dynamic surgical scenes, while kinematic data provides precise movement information, overcoming limitations of visual recognition under adverse conditions. We propose a multimodal Graph Representation network with Adversarial feature Disentanglement (GRAD) for robust surgical workflow recognition in challenging scenarios with domain shifts or corrupted data. Specifically, we introduce a Multimodal Disentanglement Graph Network that captures fine-grained visual information while explicitly modeling the complex relationships between vision and kinematic embeddings through graph-based message modeling. To align feature spaces across modalities, we propose a Vision-Kinematic Adversarial framework that leverages adversarial training to reduce modality gaps and improve feature consistency. Furthermore, we design a Contextual Calibrated Decoder, incorporating temporal and contextual priors to enhance robustness against domain shifts and corrupted data. Extensive comparative and ablation experiments demonstrate the effectiveness of our model and proposed modules. Moreover, our robustness experiments show that our method effectively handles data corruption during storage and transmission, exhibiting excellent stability and robustness. Our approach aims to advance automated surgical workflow recognition, addressing the complexities and dynamism inherent in surgical procedures. 

**Abstract (ZH)**: 手术流程识别对于自动化任务、支持决策制定和培训新手外科医生至关重要，最终提高患者安全性和标准化手术程序。然而，数据损坏可能会由于出血或手术场景中的烟雾导致的遮挡等问题以及数据存储和传输问题而导致性能下降。在这种情况下，我们探索了一种鲁棒的基于图的多模态方法，结合视觉和运动数据以提高准确性和可靠性。视觉数据捕捉动态的手术场景，而运动数据提供精确的运动信息，克服了不利条件下视觉识别的局限性。我们提出了一种具有对抗特征解耦的多模态图表示网络（GRAD），以在迁移或数据损坏的挑战性场景中实现鲁棒的手术流程识别。具体而言，我们引入了一种多模态分解图网络，以捕获细粒度的视觉信息，并通过基于图的消息建模显式地建模视觉和运动嵌入之间的复杂关系。为了跨模态对齐特征空间，我们提出了一种视觉-运动对抗框架，利用对抗训练减少模态差异并提高特征一致性。此外，我们设计了一种上下文校准解码器，结合时间上下文先验以增强对迁移和数据损坏的鲁棒性。广泛的对比实验和消融实验展示了我们模型及其模块的有效性。此外，我们的鲁棒性实验表明，我们的方法能够有效处理存储和传输过程中的数据损坏，表现出出色的稳定性和鲁棒性。我们的方法旨在推动自动手术流程识别的发展，以应对手术过程中的复杂性和动态性。 

---
# MSFNet-CPD: Multi-Scale Cross-Modal Fusion Network for Crop Pest Detection 

**Title (ZH)**: MSFNet-CPD：多尺度跨模态融合网络在农作物害虫检测中的应用 

**Authors**: Jiaqi Zhang, Zhuodong Liu, Kejian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02441)  

**Abstract**: Accurate identification of agricultural pests is essential for crop protection but remains challenging due to the large intra-class variance and fine-grained differences among pest species. While deep learning has advanced pest detection, most existing approaches rely solely on low-level visual features and lack effective multi-modal integration, leading to limited accuracy and poor interpretability. Moreover, the scarcity of high-quality multi-modal agricultural datasets further restricts progress in this field. To address these issues, we construct two novel multi-modal benchmarks-CTIP102 and STIP102-based on the widely-used IP102 dataset, and introduce a Multi-scale Cross-Modal Fusion Network (MSFNet-CPD) for robust pest detection. Our approach enhances visual quality via a super-resolution reconstruction module, and feeds both the original and reconstructed images into the network to improve clarity and detection performance. To better exploit semantic cues, we propose an Image-Text Fusion (ITF) module for joint modeling of visual and textual features, and an Image-Text Converter (ITC) that reconstructs fine-grained details across multiple scales to handle challenging backgrounds. Furthermore, we introduce an Arbitrary Combination Image Enhancement (ACIE) strategy to generate a more complex and diverse pest detection dataset, MTIP102, improving the model's generalization to real-world scenarios. Extensive experiments demonstrate that MSFNet-CPD consistently outperforms state-of-the-art methods on multiple pest detection benchmarks. All code and datasets will be made publicly available at: this https URL. 

**Abstract (ZH)**: 准确识别农业害虫对于作物保护至关重要，但由于害虫种类内部变异大和细粒度差异，这一任务仍然具有挑战性。尽管深度学习推动了害虫检测的发展，但大多数现有方法仅依赖低级视觉特征，缺乏有效的多模态整合，导致准确率有限且可解释性差。此外，高质量多模态农业数据集的稀缺性进一步限制了该领域的进展。为解决这些问题，我们基于广泛使用的IP102数据集构建了两个新的多模态基准CTIP102和STIP102，并提出了一种多尺度跨模态融合网络（MSFNet-CPD）以实现稳健的害虫检测。该方法通过超分辨率重建模块提升视觉质量，并将原始图像和重建图像同时输入网络，以提高清晰度和检测性能。为更好地利用语义线索，我们提出了一种图像-文本融合（ITF）模块联合建模视觉和文本特征，并提出了一种图像-文本转换器（ITC），能够多尺度重建细粒度细节以处理复杂背景。此外，我们引入了一种任意组合图像增强（ACIE）策略生成更复杂多样的害虫检测数据集MTIP102，提高了模型对真实场景的泛化能力。广泛实验表明，MSFNet-CPD在多个害虫检测基准上均优于现有方法。所有代码和数据集将在以下网址公开：this https URL。 

---
# SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness 

**Title (ZH)**: SafeMate: 基于模型上下文协议的多模态应急准备代理 

**Authors**: Junfeng Jiao, Jihyung Park, Yiming Xu, Lucy Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.02306)  

**Abstract**: Despite the abundance of public safety documents and emergency protocols, most individuals remain ill-equipped to interpret and act on such information during crises. Traditional emergency decision support systems (EDSS) are designed for professionals and rely heavily on static documents like PDFs or SOPs, which are difficult for non-experts to navigate under stress. This gap between institutional knowledge and public accessibility poses a critical barrier to effective emergency preparedness and response.
We introduce SafeMate, a retrieval-augmented AI assistant that delivers accurate, context-aware guidance to general users in both preparedness and active emergency scenarios. Built on the Model Context Protocol (MCP), SafeMate dynamically routes user queries to tools for document retrieval, checklist generation, and structured summarization. It uses FAISS with cosine similarity to identify relevant content from trusted sources. 

**Abstract (ZH)**: 尽管存在大量的公共安全文件和应急规程，大多数人在危机期间仍无法解读和采取相应行动。传统的应急决策支持系统（EDSS）面向专业人士，高度依赖静态文档如PDF或SOP，这些文档在压力下难以供非专家导航。这种机构知识与公众可访问性之间的差距是有效应急准备和响应的重要障碍。我们介绍了一种检索增强的人工智能助手SafeMate，它为普通用户提供准确、情境相关的指导，适用于准备阶段和活跃的应急场景。SafeMate基于Model Context Protocol (MCP) 架构，动态将用户查询路由至文档检索、检查清单生成和结构化总结工具。它使用FAISS和余弦相似度识别可信来源中的相关内容。 

---
# Improving Physical Object State Representation in Text-to-Image Generative Systems 

**Title (ZH)**: 改进物理对象状态表示的文本到图像生成系统 

**Authors**: Tianle Chen, Chaitanya Chakka, Deepti Ghadiyaram  

**Link**: [PDF](https://arxiv.org/pdf/2505.02236)  

**Abstract**: Current text-to-image generative models struggle to accurately represent object states (e.g., "a table without a bottle," "an empty tumbler"). In this work, we first design a fully-automatic pipeline to generate high-quality synthetic data that accurately captures objects in varied states. Next, we fine-tune several open-source text-to-image models on this synthetic data. We evaluate the performance of the fine-tuned models by quantifying the alignment of the generated images to their prompts using GPT4o-mini, and achieve an average absolute improvement of 8+% across four models on the public GenAI-Bench dataset. We also curate a collection of 200 prompts with a specific focus on common objects in various physical states. We demonstrate a significant improvement of an average of 24+% over the baseline on this dataset. We release all evaluation prompts and code. 

**Abstract (ZH)**: 当前的文本到图像生成模型在准确表示物体状态（如“没有瓶子的桌子”、“空的酒杯”）方面存在挑战。本文首先设计了一个全自动的工作流程，生成高质量的合成数据，准确捕捉物体在不同状态下的表现。随后，我们在这些合成数据上微调了几种开源的文本到图像模型。通过使用GPT4o-mini量化生成图像与提示之间的对齐程度，我们在公共GenAI-Bench数据集上实现了四个模型平均绝对改进8+％。我们还收集了一组200个具有特定常见物体在不同物理状态焦点的提示集。在该数据集上，相对于 baseline，我们实现了平均24+％的重要改进。我们发布了所有评估提示和代码。 

---
# PhytoSynth: Leveraging Multi-modal Generative Models for Crop Disease Data Generation with Novel Benchmarking and Prompt Engineering Approach 

**Title (ZH)**: PhytoSynth：利用新型基准测试和提示工程方法结合多模态生成模型生成作物疾病数据 

**Authors**: Nitin Rai, Arnold W. Schumann, Nathan Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2505.01823)  

**Abstract**: Collecting large-scale crop disease images in the field is labor-intensive and time-consuming. Generative models (GMs) offer an alternative by creating synthetic samples that resemble real-world images. However, existing research primarily relies on Generative Adversarial Networks (GANs)-based image-to-image translation and lack a comprehensive analysis of computational requirements in agriculture. Therefore, this research explores a multi-modal text-to-image approach for generating synthetic crop disease images and is the first to provide computational benchmarking in this context. We trained three Stable Diffusion (SD) variants-SDXL, SD3.5M (medium), and SD3.5L (large)-and fine-tuned them using Dreambooth and Low-Rank Adaptation (LoRA) fine-tuning techniques to enhance generalization. SD3.5M outperformed the others, with an average memory usage of 18 GB, power consumption of 180 W, and total energy use of 1.02 kWh/500 images (0.002 kWh per image) during inference task. Our results demonstrate SD3.5M's ability to generate 500 synthetic images from just 36 in-field samples in 1.5 hours. We recommend SD3.5M for efficient crop disease data generation. 

**Abstract (ZH)**: 在田间收集大规模作物病害图像劳动密集且耗时。生成模型（GMs）通过生成与真实图像相似的合成样本提供了一种替代方案。然而，现有的研究主要依赖基于生成对抗网络（GANs）的图像到图像转换，缺乏在农业领域对计算需求的全面分析。因此，本研究探索了一种多模态文本到图像的方法来生成合成作物病害图像，并首次在此背景下提供了计算基准测试。我们训练了三种Stable Diffusion（SD）变体——SDXL、SD3.5M（中型）和SD3.5L（大型），并通过Dreambooth和低秩适应（LoRA）微调技术对它们进行微调以增强泛化能力。SD3.5M在推理任务中表现出色，平均内存使用为18GB，功率消耗为180W，并且每生成500张图像的总能耗为1.02千瓦时（每张图像0.002千瓦时）。我们的结果表明，SD3.5M能够在1.5小时内从36张田间样本中生成500张合成图像。我们推荐使用SD3.5M进行高效的作物病害数据生成。 

---
# A Multimodal Framework for Explainable Evaluation of Soft Skills in Educational Environments 

**Title (ZH)**: 一种多模态框架，用于教育资源环境中可解释的软技能评估 

**Authors**: Jared D.T. Guerrero-Sosa, Francisco P. Romero, Víctor Hugo Menéndez-Domínguez, Jesus Serrano-Guerrero, Andres Montoro-Montarroso, Jose A. Olivas  

**Link**: [PDF](https://arxiv.org/pdf/2505.01794)  

**Abstract**: In the rapidly evolving educational landscape, the unbiased assessment of soft skills is a significant challenge, particularly in higher education. This paper presents a fuzzy logic approach that employs a Granular Linguistic Model of Phenomena integrated with multimodal analysis to evaluate soft skills in undergraduate students. By leveraging computational perceptions, this approach enables a structured breakdown of complex soft skill expressions, capturing nuanced behaviours with high granularity and addressing their inherent uncertainties, thereby enhancing interpretability and reliability. Experiments were conducted with undergraduate students using a developed tool that assesses soft skills such as decision-making, communication, and creativity. This tool identifies and quantifies subtle aspects of human interaction, such as facial expressions and gesture recognition. The findings reveal that the framework effectively consolidates multiple data inputs to produce meaningful and consistent assessments of soft skills, showing that integrating multiple modalities into the evaluation process significantly improves the quality of soft skills scores, making the assessment work transparent and understandable to educational stakeholders. 

**Abstract (ZH)**: 在快速演进的教育环境中，软技能的公正评估是一个重大挑战，特别是在高等教育中。本文提出了一种基于模糊逻辑的方法，该方法结合了颗粒语言模型和多模态分析，用于评估本科生的软技能。通过利用计算感知，该方法实现了复杂软技能表达的结构化分解，以高粒度捕捉细微行为并解决它们的固有不确定性，从而提高解释性和可靠性。实验使用一个开发的工具对本科生进行了软技能（如决策制定、沟通和创造力）的评估，该工具识别并量化了人类互动的微妙方面，如面部表情和手势识别。研究发现，该框架有效整合了多种数据输入，产生了具有意义且一致的软技能评估结果，显示了将多种模态集成到评估过程中显著提高了软技能评分的质量，使评估工作对教育利益相关者透明且易于理解。 

---
# Automated ARAT Scoring Using Multimodal Video Analysis, Multi-View Fusion, and Hierarchical Bayesian Models: A Clinician Study 

**Title (ZH)**: 基于多模态视频分析、多视角融合和分层贝叶斯模型的 Automated ARAT 评分自动化研究：一项临床医师研究 

**Authors**: Tamim Ahmed, Thanassis Rikakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.01680)  

**Abstract**: Manual scoring of the Action Research Arm Test (ARAT) for upper extremity assessment in stroke rehabilitation is time-intensive and variable. We propose an automated ARAT scoring system integrating multimodal video analysis with SlowFast, I3D, and Transformer-based models using OpenPose keypoints and object locations. Our approach employs multi-view data (ipsilateral, contralateral, and top perspectives), applying early and late fusion to combine features across views and models. Hierarchical Bayesian Models (HBMs) infer movement quality components, enhancing interpretability. A clinician dashboard displays task scores, execution times, and quality assessments. We conducted a study with five clinicians who reviewed 500 video ratings generated by our system, providing feedback on its accuracy and usability. Evaluated on a stroke rehabilitation dataset, our framework achieves 89.0% validation accuracy with late fusion, with HBMs aligning closely with manual assessments. This work advances automated rehabilitation by offering a scalable, interpretable solution with clinical validation. 

**Abstract (ZH)**: 基于多模态视频分析的手动评分自动化系统：Action Research Arm Test (ARAT) 在中风康复中的上肢评估时间密集且变异性强。我们提出了一种结合多模态视频分析和 SlowFast、I3D 和基于 Transformer 的模型的自动化 ARAT 评分系统，使用 OpenPose 关键点和物体位置。该方法采用多视图数据（同侧、对侧和顶视角），运用早期和晚期融合以结合视图和模型的特征。通过分层贝叶斯模型 (HBM) 推断运动质量组件，增强可解释性。临床医生仪表盘显示任务评分、执行时间和质量评估。我们在五位临床医生对系统生成的 500 个视频评分进行审查的基础上，提供了关于其准确性和易用性的反馈。在中风康复数据集上评估时，我们的框架通过晚期融合达到 89.0% 的验证准确性，HBM 与手动评估高度一致。这项工作通过提供一种可扩展且可解释的自动化康复方案，实现了临床验证，推动了自动化康复的发展。 

---
# Seeing Heat with Color -- RGB-Only Wildfire Temperature Inference from SAM-Guided Multimodal Distillation using Radiometric Ground Truth 

**Title (ZH)**: 仅使用RGB图像通过SAM引导的多模态蒸馏进行基于辐射测温的 wildfire 温度推断——以颜色观火 

**Authors**: Michael Marinaccio, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2505.01638)  

**Abstract**: High-fidelity wildfire monitoring using Unmanned Aerial Vehicles (UAVs) typically requires multimodal sensing - especially RGB and thermal imagery - which increases hardware cost and power consumption. This paper introduces SAM-TIFF, a novel teacher-student distillation framework for pixel-level wildfire temperature prediction and segmentation using RGB input only. A multimodal teacher network trained on paired RGB-Thermal imagery and radiometric TIFF ground truth distills knowledge to a unimodal RGB student network, enabling thermal-sensor-free inference. Segmentation supervision is generated using a hybrid approach of segment anything (SAM)-guided mask generation, and selection via TOPSIS, along with Canny edge detection and Otsu's thresholding pipeline for automatic point prompt selection. Our method is the first to perform per-pixel temperature regression from RGB UAV data, demonstrating strong generalization on the recent FLAME 3 dataset. This work lays the foundation for lightweight, cost-effective UAV-based wildfire monitoring systems without thermal sensors. 

**Abstract (ZH)**: 基于RGB输入的像素级 wildfire温度预测与分割的新型教师-学生蒸馏框架：无需热传感器的轻量级UAV野火监测系统 

---
# Multimodal and Multiview Deep Fusion for Autonomous Marine Navigation 

**Title (ZH)**: 多模态多视角深层融合自主海洋导航 

**Authors**: Dimitrios Dagdilelis, Panagiotis Grigoriadis, Roberto Galeazzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01615)  

**Abstract**: We propose a cross attention transformer based method for multimodal sensor fusion to build a birds eye view of a vessels surroundings supporting safer autonomous marine navigation. The model deeply fuses multiview RGB and long wave infrared images with sparse LiDAR point clouds. Training also integrates X band radar and electronic chart data to inform predictions. The resulting view provides a detailed reliable scene representation improving navigational accuracy and robustness. Real world sea trials confirm the methods effectiveness even in adverse weather and complex maritime settings. 

**Abstract (ZH)**: 基于跨注意力变换器的多模态传感器融合方法以构建船舶周围环境的鸟瞰图，支持更安全的自主海洋导航 

---
