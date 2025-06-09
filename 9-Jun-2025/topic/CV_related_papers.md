# TD-TOG Dataset: Benchmarking Zero-Shot and One-Shot Task-Oriented Grasping for Object Generalization 

**Title (ZH)**: TD-TOG数据集：面向对象泛化的零-shot和少-shot任务导向抓取基准测试 

**Authors**: Valerija Holomjova, Jamie Grech, Dewei Yi, Bruno Yun, Andrew Starkey, Pascal Meißner  

**Link**: [PDF](https://arxiv.org/pdf/2506.05576)  

**Abstract**: Task-oriented grasping (TOG) is an essential preliminary step for robotic task execution, which involves predicting grasps on regions of target objects that facilitate intended tasks. Existing literature reveals there is a limited availability of TOG datasets for training and benchmarking despite large demand, which are often synthetic or have artifacts in mask annotations that hinder model performance. Moreover, TOG solutions often require affordance masks, grasps, and object masks for training, however, existing datasets typically provide only a subset of these annotations. To address these limitations, we introduce the Top-down Task-oriented Grasping (TD-TOG) dataset, designed to train and evaluate TOG solutions. TD-TOG comprises 1,449 real-world RGB-D scenes including 30 object categories and 120 subcategories, with hand-annotated object masks, affordances, and planar rectangular grasps. It also features a test set for a novel challenge that assesses a TOG solution's ability to distinguish between object subcategories. To contribute to the demand for TOG solutions that can adapt and manipulate previously unseen objects without re-training, we propose a novel TOG framework, Binary-TOG. Binary-TOG uses zero-shot for object recognition, and one-shot learning for affordance recognition. Zero-shot learning enables Binary-TOG to identify objects in multi-object scenes through textual prompts, eliminating the need for visual references. In multi-object settings, Binary-TOG achieves an average task-oriented grasp accuracy of 68.9%. Lastly, this paper contributes a comparative analysis between one-shot and zero-shot learning for object generalization in TOG to be used in the development of future TOG solutions. 

**Abstract (ZH)**: 面向任务的抓取（TOG）数据集：Top-down Task-oriented Grasping (TD-TOG) 

---
# A Compendium of Autonomous Navigation using Object Detection and Tracking in Unmanned Aerial Vehicles 

**Title (ZH)**: 基于对象检测与跟踪的自主导航综合研究（应用于无人驾驶航空车辆） 

**Authors**: Mohit Arora, Pratyush Shukla, Shivali Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2506.05378)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are one of the most revolutionary inventions of 21st century. At the core of a UAV lies the central processing system that uses wireless signals to control their movement. The most popular UAVs are quadcopters that use a set of four motors, arranged as two on either side with opposite spin. An autonomous UAV is called a drone. Drones have been in service in the US army since the 90's for covert missions critical to national security. It would not be wrong to claim that drones make up an integral part of the national security and provide the most valuable service during surveillance operations. While UAVs are controlled using wireless signals, there reside some challenges that disrupt the operation of such vehicles such as signal quality and range, real time processing, human expertise, robust hardware and data security. These challenges can be solved by programming UAVs to be autonomous, using object detection and tracking, through Computer Vision algorithms. Computer Vision is an interdisciplinary field that seeks the use of deep learning to gain a high-level understanding of digital images and videos for the purpose of automating the task of human visual system. Using computer vision, algorithms for detecting and tracking various objects can be developed suitable to the hardware so as to allow real time processing for immediate judgement. This paper attempts to review the various approaches several authors have proposed for the purpose of autonomous navigation of UAVs by through various algorithms of object detection and tracking in real time, for the purpose of applications in various fields such as disaster management, dense area exploration, traffic vehicle surveillance etc. 

**Abstract (ZH)**: 无人驾驶航空车辆（UAVs）是21世纪最革命性的发明之一。UAV的核心是中央处理系统，通过无线信号控制其运动。最流行的UAV是四旋翼无人机，它配备了一组四个电机，每边两个，旋转方向相反。自主无人机称为无人机。无人机自20世纪90年代以来一直在美国军队中服役，用于关键性的隐蔽任务，对国家安全至关重要。可以说，无人机是国家安全不可或缺的一部分，并在 surveillance 操作中提供了最宝贵的服务。虽然UAV是通过无线信号控制的，但存在一些挑战，如信号质量与范围、实时处理、人力专业知识、鲁棒硬件和数据安全等问题。通过编程使无人机自主，使用计算机视觉算法进行目标检测与跟踪，可以解决这些问题。计算机视觉是一个跨学科领域，利用深度学习来理解数字图像和视频，旨在自动化人类视觉系统完成的任务。利用计算机视觉，可以开发适合硬件的检测与跟踪算法，以实现实时处理并立即做出判断。本文旨在回顾若干作者提出的多种方法，通过实时目标检测与跟踪算法实现无人机的自主导航，以应用于灾害管理、密集区域探索、交通车辆监控等各个领域。 

---
# GenIR: Generative Visual Feedback for Mental Image Retrieval 

**Title (ZH)**: GenIR: 生成式视觉反馈的心理图像检索 

**Authors**: Diji Yang, Minghao Liu, Chung-Hsiang Lo, Yi Zhang, James Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.06220)  

**Abstract**: Vision-language models (VLMs) have shown strong performance on text-to-image retrieval benchmarks. However, bridging this success to real-world applications remains a challenge. In practice, human search behavior is rarely a one-shot action. Instead, it is often a multi-round process guided by clues in mind, that is, a mental image ranging from vague recollections to vivid mental representations of the target image. Motivated by this gap, we study the task of Mental Image Retrieval (MIR), which targets the realistic yet underexplored setting where users refine their search for a mentally envisioned image through multi-round interactions with an image search engine. Central to successful interactive retrieval is the capability of machines to provide users with clear, actionable feedback; however, existing methods rely on indirect or abstract verbal feedback, which can be ambiguous, misleading, or ineffective for users to refine the query. To overcome this, we propose GenIR, a generative multi-round retrieval paradigm leveraging diffusion-based image generation to explicitly reify the AI system's understanding at each round. These synthetic visual representations provide clear, interpretable feedback, enabling users to refine their queries intuitively and effectively. We further introduce a fully automated pipeline to generate a high-quality multi-round MIR dataset. Experimental results demonstrate that GenIR significantly outperforms existing interactive methods in the MIR scenario. This work establishes a new task with a dataset and an effective generative retrieval method, providing a foundation for future research in this direction. 

**Abstract (ZH)**: 基于视觉-语言模型的思维图像检索（Mental Image Retrieval, MIR）：一种生成式的多轮检索范式 

---
# HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion 

**Title (ZH)**: HAVIR: 分层视觉引导的CLIP指导可变扩散图像重建 

**Authors**: Shiyi Zhang, Dong Liang, Hairong Zheng, Yihang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06035)  

**Abstract**: Reconstructing visual information from brain activity bridges the gap between neuroscience and computer vision. Even though progress has been made in decoding images from fMRI using generative models, a challenge remains in accurately recovering highly complex visual stimuli. This difficulty stems from their elemental density and diversity, sophisticated spatial structures, and multifaceted semantic information.
To address these challenges, we propose HAVIR that contains two adapters: (1) The AutoKL Adapter transforms fMRI voxels into a latent diffusion prior, capturing topological structures; (2) The CLIP Adapter converts the voxels to CLIP text and image embeddings, containing semantic information. These complementary representations are fused by Versatile Diffusion to generate the final reconstructed image. To extract the most essential semantic information from complex scenarios, the CLIP Adapter is trained with text captions describing the visual stimuli and their corresponding semantic images synthesized from these captions. The experimental results demonstrate that HAVIR effectively reconstructs both structural features and semantic information of visual stimuli even in complex scenarios, outperforming existing models. 

**Abstract (ZH)**: 从大脑活动重建视觉信息跨越了神经科学与计算机视觉的鸿沟。尽管已经使用生成模型从fMRI解码图像取得了进展，但在准确恢复高度复杂的视觉刺激方面仍面临挑战。这种困难源于其基本密度和多样性、复杂的空间结构以及多方面的语义信息。

为应对这些挑战，我们提出了HAVIR，其包含两个适配器：(1) AutoKL适配器将fMRI体素转换为潜在扩散先验，捕捉拓扑结构；(2) CLIP适配器将体素转换为CLIP文本和图像嵌入，包含语义信息。这些互补的表示由通用扩散融合生成最终重建图像。为了从复杂场景中提取最核心的语义信息，CLIP适配器基于描述视觉刺激及其对应语义图像的文本说明进行训练。实验结果表明，HAVIR在复杂场景中有效地重建了视觉刺激的结构特征和语义信息，优于现有模型。 

---
# Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models 

**Title (ZH)**: 使用混合机器学习与深度学习模型增强正痘病毒图像分类 

**Authors**: Alejandro Puente-Castro, Enrique Fernandez-Blanco, Daniel Rivero, Andres Molares-Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06007)  

**Abstract**: Orthopoxvirus infections must be accurately classified from medical pictures for an easy and early diagnosis and epidemic prevention. The necessity for automated and scalable solutions is highlighted by the fact that traditional diagnostic techniques can be time-consuming and require expert interpretation and there are few and biased data sets of the different types of Orthopox. In order to improve classification performance and lower computational costs, a hybrid strategy is put forth in this paper that uses Machine Learning models combined with pretrained Deep Learning models to extract deep feature representations without the need for augmented data. The findings show that this feature extraction method, when paired with other methods in the state-of-the-art, produces excellent classification outcomes while preserving training and inference efficiency. The proposed approach demonstrates strong generalization and robustness across multiple evaluation settings, offering a scalable and interpretable solution for real-world clinical deployment. 

**Abstract (ZH)**: 正痘病毒感染需要通过医学影像准确分类以便实现早期诊断和疫情预防。本研究强调了自动化和可扩展解决方案的必要性，因为传统诊断技术耗时且需要专家解释，而正痘病毒的不同类型数据集较少且带有偏见。为了提高分类性能并降低计算成本，本文提出了一种结合机器学习模型和预训练深度学习模型的混合策略，以提取深层特征表示，无需增加数据。研究结果表明，这种方法与其他先进的方法结合使用时，能够产生出色的分类效果，同时保持训练和推理效率。所提出的建模方法在多种评估设置中展现出良好的泛化能力和鲁棒性，提供了一种可扩展且可解释的临床部署解决方案。 

---
# MOGO: Residual Quantized Hierarchical Causal Transformer for High-Quality and Real-Time 3D Human Motion Generation 

**Title (ZH)**: MOGO：残差量化分层因果变换器，用于高保真实时三维人体运动生成 

**Authors**: Dongjie Fu, Tengjiao Sun, Pengcheng Fang, Xiaohao Cai, Hansung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05952)  

**Abstract**: Recent advances in transformer-based text-to-motion generation have led to impressive progress in synthesizing high-quality human motion. Nevertheless, jointly achieving high fidelity, streaming capability, real-time responsiveness, and scalability remains a fundamental challenge. In this paper, we propose MOGO (Motion Generation with One-pass), a novel autoregressive framework tailored for efficient and real-time 3D motion generation. MOGO comprises two key components: (1) MoSA-VQ, a motion scale-adaptive residual vector quantization module that hierarchically discretizes motion sequences with learnable scaling to produce compact yet expressive representations; and (2) RQHC-Transformer, a residual quantized hierarchical causal transformer that generates multi-layer motion tokens in a single forward pass, significantly reducing inference latency. To enhance semantic fidelity, we further introduce a text condition alignment mechanism that improves motion decoding under textual control. Extensive experiments on benchmark datasets including HumanML3D, KIT-ML, and CMP demonstrate that MOGO achieves competitive or superior generation quality compared to state-of-the-art transformer-based methods, while offering substantial improvements in real-time performance, streaming generation, and generalization under zero-shot settings. 

**Abstract (ZH)**: 近期基于变压器的文字到运动生成技术取得了显著进展，极大地促进了高质量人类运动合成。然而，同时实现高保真度、流式传输能力、实时响应性和可扩展性仍是一项基本挑战。本文提出了一种新型自回归框架MOGO（运动生成一站式），旨在高效地进行实时三维运动生成。MOGO包含两个关键组成部分：（1）MoSA-VQ，一种运动尺度自适应残差矢量量化模块，该模块通过可学习的缩放对运动序列进行分层离散化，生成紧凑而富有表现力的表示；（2）RQHC-Transformer，一种残差量化层次因原因子变压器，在单向前传播中生成多层运动标记，显著减少了推理延迟。为进一步提升语义保真度，我们引入了一种文本条件对齐机制，该机制在文本控制下改善了运动解码。在包括HumanML3D、KIT-ML和CMP基准数据集上的广泛实验表明，MOGO在实时性能、流式生成和零样本设置下的泛化方面取得了显著改进，同时生成质量与最先进的基于变压器的方法相当或更优。 

---
# FADE: Frequency-Aware Diffusion Model Factorization for Video Editing 

**Title (ZH)**: 频率 Awareness 下的扩散模型因子分解方法及其在视频编辑中的应用 

**Authors**: Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yansong Tang, Lei Chen, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05934)  

**Abstract**: Recent advancements in diffusion frameworks have significantly enhanced video editing, achieving high fidelity and strong alignment with textual prompts. However, conventional approaches using image diffusion models fall short in handling video dynamics, particularly for challenging temporal edits like motion adjustments. While current video diffusion models produce high-quality results, adapting them for efficient editing remains difficult due to the heavy computational demands that prevent the direct application of previous image editing techniques. To overcome these limitations, we introduce FADE, a training-free yet highly effective video editing approach that fully leverages the inherent priors from pre-trained video diffusion models via frequency-aware factorization. Rather than simply using these models, we first analyze the attention patterns within the video model to reveal how video priors are distributed across different components. Building on these insights, we propose a factorization strategy to optimize each component's specialized role. Furthermore, we devise spectrum-guided modulation to refine the sampling trajectory with frequency domain cues, preventing information leakage and supporting efficient, versatile edits while preserving the basic spatial and temporal structure. Extensive experiments on real-world videos demonstrate that our method consistently delivers high-quality, realistic and temporally coherent editing results both qualitatively and quantitatively. Code is available at this https URL . 

**Abstract (ZH)**: 近期在扩散框架方面的进展显著提升了视频编辑效果，实现了高度逼真和与文本提示的强烈对齐。然而，传统的使用图像扩散模型的方法在处理视频动态方面仍然不足，特别是在处理如运动调整等具有挑战性的时间编辑时。尽管当前的视频扩散模型能够生成高质量的结果，但由于繁重的计算需求使得难以直接应用先前的图像编辑技术，进而将其用于高效的编辑操作。为了解决这些局限性，我们提出了FADE，这是一种无需训练即可高效实现视频编辑的方法，通过频率感知的因子分解充分利用预训练的视频扩散模型固有的先验知识。我们不仅使用这些模型，而是首先分析视频模型中的注意力模式，揭示视频先验在不同组件中的分布情况。基于这些洞见，我们提出了一种因子分解策略来优化每个组件的专业角色。此外，我们设计了频谱导向的调制，通过频域线索改进采样轨迹，防止信息泄露，同时支持高效和多样的编辑操作，并保持基本的空间和时间结构。在实际视频上的广泛实验表明，我们的方法在定性和定量上都能够持续产出高质量、逼真且具有时序一致性的编辑结果。代码已发布于此 https URL 。 

---
# Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness 

**Title (ZH)**: 超越准确性：重新思考半监督分割的可靠性和鲁棒性 

**Authors**: Steven Landgraf, Markus Hillemann, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.05917)  

**Abstract**: Semantic segmentation is critical for scene understanding but demands costly pixel-wise annotations, attracting increasing attention to semi-supervised approaches to leverage abundant unlabeled data. While semi-supervised segmentation is often promoted as a path toward scalable, real-world deployment, it is astonishing that current evaluation protocols exclusively focus on segmentation accuracy, entirely overlooking reliability and robustness. These qualities, which ensure consistent performance under diverse conditions (robustness) and well-calibrated model confidences as well as meaningful uncertainties (reliability), are essential for safety-critical applications like autonomous driving, where models must handle unpredictable environments and avoid sudden failures at all costs. To address this gap, we introduce the Reliable Segmentation Score (RSS), a novel metric that combines predictive accuracy, calibration, and uncertainty quality measures via a harmonic mean. RSS penalizes deficiencies in any of its components, providing an easy and intuitive way of holistically judging segmentation models. Comprehensive evaluations of UniMatchV2 against its predecessor and a supervised baseline show that semi-supervised methods often trade reliability for accuracy. While out-of-domain evaluations demonstrate UniMatchV2's robustness, they further expose persistent reliability shortcomings. We advocate for a shift in evaluation protocols toward more holistic metrics like RSS to better align semi-supervised learning research with real-world deployment needs. 

**Abstract (ZH)**: 可靠的分割评分：一种综合预测准确性、校准和不确定性质量的新评价指标 

---
# FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks 

**Title (ZH)**: FuseUNet：U形网络的多尺度特征融合方法 

**Authors**: Quansong He, Xiangde Min, Kaishen Wang, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05821)  

**Abstract**: Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of the encoder and decoder architectures, making it adaptable to various U-Net-like networks. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance. The code is available at this https URL. 

**Abstract (ZH)**: 医学图像分割是计算机视觉中的一个关键任务，UNet作为一种里程碑式的架构起到了重要作用。UNet家族的典型组件是跳跃连接，然而，这些跳跃连接面临两大显著限制：（1）它们在不同尺度特征之间的有效交互不足；（2）它们依赖于简单的连接或加法操作，限制了高效信息整合。尽管最近对UNet的改进主要集中在增强编码器和解码器的能力上，但这些限制仍被忽视。为克服这些挑战，我们提出了一种新颖的多尺度特征融合方法，将UNet的解码过程重新构想为求解初值问题（IVP），并将跳跃连接视为离散节点。通过利用线性多步法原理，我们提出了一个自适应常微分方程方法，以实现有效的多尺度特征融合。该方法独立于编码器和解码器架构，使其适用于各种U-Net类型的网络。在ACDC、KiTS2023、MSD脑肿瘤和ISIC2017/2018皮肤病变分割数据集上的实验表明，该方法能提高特征利用效率、减少网络参数数量，同时保持高性能。代码可在以下链接获取。 

---
# Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from DataSeeds' Annotated Imagery 

**Title (ZH)**: 基于同伴排序精度：创建一个用于从DataSeeds标注图像fine-tune视觉模型的基础数据集 

**Authors**: Sajjad Abdoli, Freeman Lewin, Gediminas Vasiliauskas, Fabian Schonholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.05673)  

**Abstract**: The development of modern Artificial Intelligence (AI) models, particularly diffusion-based models employed in computer vision and image generation tasks, is undergoing a paradigmatic shift in development methodologies. Traditionally dominated by a "Model Centric" approach, in which performance gains were primarily pursued through increasingly complex model architectures and hyperparameter optimization, the field is now recognizing a more nuanced "Data-Centric" approach. This emergent framework foregrounds the quality, structure, and relevance of training data as the principal driver of model performance. To operationalize this paradigm shift, we introduce the this http URL sample dataset (the "DSD"), initially comprised of approximately 10,610 high-quality human peer-ranked photography images accompanied by extensive multi-tier annotations. The DSD is a foundational computer vision dataset designed to usher in a new standard for commercial image datasets. Representing a small fraction of this http URL's 100 million-plus image catalog, the DSD provides a scalable foundation necessary for robust commercial and multimodal AI development. Through this in-depth exploratory analysis, we document the quantitative improvements generated by the DSD on specific models against known benchmarks and make the code and the trained models used in our evaluation publicly available. 

**Abstract (ZH)**: 现代人工 Intelligence (AI) 模型的发展，特别是用于计算机视觉和图像生成任务的扩散基于模型的发展，正在经历一种开发方法范式的转变。传统上，该领域主要受到以“模型为中心”方法的影响，通过构建越来越复杂的数据架构和超参数优化来追求性能提升，但现正开始认识到一种更精细的“数据为中心”方法。这种新兴框架强调训练数据的质量、结构和相关性是模型性能的主要驱动因素。为实现这一范式转变，我们介绍了这个 http://thisurl.com 样本数据集（“DSD”），初始包含约 10,610 张高质量的人类同行评分的摄影作品及其详尽的多级注释。DSD 是一个基础的计算机视觉数据集，旨在推动新的商用图像数据集标准。作为这个 http://thisurl.com 百万以上图像目录的小部分，DSD 提供了一个可扩展的基础，对于稳健的商用和多模态 AI 开发至关重要。通过这项深入的探索性分析，我们记录了 DSD 对特定模型在已知基准上的量化改进，并向公众提供了我们在评估中使用的代码和训练模型。 

---
# DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models 

**Title (ZH)**: DriveAction: 一种探索VLA模型中人类-like 驾驶决策的标准基准 

**Authors**: Yuhan Hao, Zhengning Li, Lei Sun, Weilong Wang, Naixin Yi, Sheng Song, Caihong Qin, Mofan Zhou, Yifei Zhan, Peng Jia, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05667)  

**Abstract**: Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by users of production-level autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from users' actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving. 

**Abstract (ZH)**: 基于视觉-语言-行动的DriveAction基准：面向自主驾驶的行动驱动评测 

---
# TissUnet: Improved Extracranial Tissue and Cranium Segmentation for Children through Adulthood 

**Title (ZH)**: TissUnet: 用于从童年至成年的 Extracranial 组织和颅骨分割改进方法 

**Authors**: Markian Mandzak, Elvira Yang, Anna Zapaishchykova, Yu-Hui Chen, Lucas Heilbroner, John Zielke, Divyanshu Tak, Reza Mojahed-Yazdi, Francesca Romana Mussa, Zezhong Ye, Sridhar Vajapeyam, Viviana Benitez, Ralph Salloum, Susan N. Chi, Houman Sotoudeh, Jakob Seidlitz, Sabine Mueller, Hugo J.W.L. Aerts, Tina Y. Poussaint, Benjamin H. Kann  

**Link**: [PDF](https://arxiv.org/pdf/2506.05660)  

**Abstract**: Extracranial tissues visible on brain magnetic resonance imaging (MRI) may hold significant value for characterizing health conditions and clinical decision-making, yet they are rarely quantified. Current tools have not been widely validated, particularly in settings of developing brains or underlying pathology. We present TissUnet, a deep learning model that segments skull bone, subcutaneous fat, and muscle from routine three-dimensional T1-weighted MRI, with or without contrast enhancement. The model was trained on 155 paired MRI-computed tomography (CT) scans and validated across nine datasets covering a wide age range and including individuals with brain tumors. In comparison to AI-CT-derived labels from 37 MRI-CT pairs, TissUnet achieved a median Dice coefficient of 0.79 [IQR: 0.77-0.81] in a healthy adult cohort. In a second validation using expert manual annotations, median Dice was 0.83 [IQR: 0.83-0.84] in healthy individuals and 0.81 [IQR: 0.78-0.83] in tumor cases, outperforming previous state-of-the-art method. Acceptability testing resulted in an 89% acceptance rate after adjudication by a tie-breaker(N=108 MRIs), and TissUnet demonstrated excellent performance in the blinded comparative review (N=45 MRIs), including both healthy and tumor cases in pediatric populations. TissUnet enables fast, accurate, and reproducible segmentation of extracranial tissues, supporting large-scale studies on craniofacial morphology, treatment effects, and cardiometabolic risk using standard brain T1w MRI. 

**Abstract (ZH)**: Extracranial 组织在脑磁共振成像(MRI)中的可视化：对于表征健康状况和临床决策具有重要意义，但鲜有量化。目前的工具在发育脑或潜在病理情况下未广泛验证。我们提出了一种深度学习模型 TissUnet，可以从常规三维 T1 加权 MRI（有或无对比增强）中分割颅骨骨、皮下脂肪和肌肉。该模型使用 155 对 MRI-CT 扫描进行训练，并在九个涵盖广泛年龄范围且包括脑肿瘤患者的数据库中进行验证。与来自 37 对 MRI-CT 的 AI-CT 提取标签相比，在健康成人组中，TissUnet 的中位 Dice 系数为 0.79 [IQR: 0.77-0.81]。在使用专家手动注释进行的第二次验证中，在健康个体中，中位 Dice 为 0.83 [IQR: 0.83-0.84]，在肿瘤病例中为 0.81 [IQR: 0.78-0.83]，优于之前的最先进方法。接受性测试后，在决选裁定者的评估下，接受率为 89%（N=108），而在盲法比较审查中，TissUnet 在包括儿童患者在内的健康和肿瘤病例中表现出色。TissUnet 使 Extracranial 组织的快速、准确和可重复分割成为可能，支持使用标准脑 T1 加权 MRI 对颅面形态、治疗效果和心血管代谢风险进行大规模研究。 

---
# LFA applied to CNNs: Efficient Singular Value Decomposition of Convolutional Mappings by Local Fourier Analysis 

**Title (ZH)**: LFA应用于CNNs：局部傅里叶分析下的卷积映射高效奇异值分解 

**Authors**: Antonia van Betteray, Matthias Rottmann, Karsten Kahl  

**Link**: [PDF](https://arxiv.org/pdf/2506.05617)  

**Abstract**: The singular values of convolutional mappings encode interesting spectral properties, which can be used, e.g., to improve generalization and robustness of convolutional neural networks as well as to facilitate model compression. However, the computation of singular values is typically very resource-intensive. The naive approach involves unrolling the convolutional mapping along the input and channel dimensions into a large and sparse two-dimensional matrix, making the exact calculation of all singular values infeasible due to hardware limitations. In particular, this is true for matrices that represent convolutional mappings with large inputs and a high number of channels. Existing efficient methods leverage the Fast Fourier transformation (FFT) to transform convolutional mappings into the frequency domain, enabling the computation of singular values for matrices representing convolutions with larger input and channel dimensions. For a constant number of channels in a given convolution, an FFT can compute N singular values in O(N log N) complexity. In this work, we propose an approach of complexity O(N) based on local Fourier analysis, which additionally exploits the shift invariance of convolutional operators. We provide a theoretical analysis of our algorithm's runtime and validate its efficiency through numerical experiments. Our results demonstrate that our proposed method is scalable and offers a practical solution to calculate the entire set of singular values - along with the corresponding singular vectors if needed - for high-dimensional convolutional mappings. 

**Abstract (ZH)**: 卷积映射的奇异值编码了有趣的谱性质，这些性质可以用于改进卷积神经网络的泛化能力和鲁棒性，以及促进模型压缩。然而，奇异值的计算通常非常耗费资源。朴素的方法是将卷积映射沿输入和通道维度展开成一个大的稀疏二维矩阵，由于硬件限制，这使得所有奇异值的精确计算变得不可行，尤其是对于具有大量输入和高通道数的卷积映射矩阵而言。现有的高效方法利用快速傅里叶变换（FFT）将卷积映射转换到频域中，从而能够计算表示大输入和通道尺寸卷积的矩阵的奇异值。对于给定卷积中通道数不变的情况，FFT可以在O(N log N)复杂度下计算N个奇异值。本文我们提出了一种基于局部傅里叶分析的复杂度为O(N)的方法，并且该方法还利用了卷积算子的移不变性。我们对算法的运行时间进行了理论分析，并通过数值实验验证了其效率。我们的结果表明，所提出的方法具有可扩展性，并提供了一种计算高维卷积映射全部奇异值（如果需要，还包括相应的奇异向量）的实用解决方案。 

---
# Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving 

**Title (ZH)**: 结构化标签使端到端自主驾驶中的视觉语言模型加速训练成为可能 

**Authors**: Hao Jiang, Chuan Hu, Yukang Shi, Yuan He, Ke Wang, Xi Zhang, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05442)  

**Abstract**: Vision-Language Models (VLMs) offer a promising approach to end-to-end autonomous driving due to their human-like reasoning capabilities. However, troublesome gaps remains between current VLMs and real-world autonomous driving applications. One major limitation is that existing datasets with loosely formatted language descriptions are not machine-friendly and may introduce redundancy. Additionally, high computational cost and massive scale of VLMs hinder the inference speed and real-world deployment. To bridge the gap, this paper introduces a structured and concise benchmark dataset, NuScenes-S, which is derived from the NuScenes dataset and contains machine-friendly structured representations. Moreover, we present FastDrive, a compact VLM baseline with 0.9B parameters. In contrast to existing VLMs with over 7B parameters and unstructured language processing(e.g., LLaVA-1.5), FastDrive understands structured and concise descriptions and generates machine-friendly driving decisions with high efficiency. Extensive experiments show that FastDrive achieves competitive performance on structured dataset, with approximately 20% accuracy improvement on decision-making tasks, while surpassing massive parameter baseline in inference speed with over 10x speedup. Additionally, ablation studies further focus on the impact of scene annotations (e.g., weather, time of day) on decision-making tasks, demonstrating their importance on decision-making tasks in autonomous driving. 

**Abstract (ZH)**: Vision-Language模型（VLMs）提供了端到端自动驾驶的一种有前景的方法，由于其类似人类的推理能力。然而，当前的VLMs与实际自动驾驶应用之间仍存在显著差距。一个主要限制是现有的、松散格式的语言描述数据集不便于机器处理，并可能引入冗余。此外，高计算成本和大规模的VLMs也阻碍了推理速度和实际部署。为解决这些问题，本文介绍了一个结构化且简洁的基准数据集NuScenes-S，该数据集源自NuScenes数据集，并包含便于机器处理的结构化表示。同时，我们呈现了一个紧凑的VLM基线FastDrive，其参数量仅为0.9B。与现有的超过7B参数和非结构化语言处理的VLMs（如LLaVA-1.5）相比，FastDrive能够理解结构化和简洁的描述，并以高效率生成便于机器处理的驾驶决策。 extensive实验表明，FastDrive在结构化数据集上实现了竞争力的表现，决策任务准确率提高了约20%，同时在推理速度上超过大规模参数基线，快了超过10倍。此外，消融研究进一步关注场景标注（如天气、时间段）对决策任务的影响，证明了它们在自动驾驶决策任务中的重要性。 

---
# BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models 

**Title (ZH)**: BYO-Eval: 自建数据集以实现多模态语言模型细粒度视觉评估 

**Authors**: Ludovic Arnould, Salim Khazem, Hugues Ali Mehenni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05440)  

**Abstract**: Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at this https URL. 

**Abstract (ZH)**: 视觉语言模型（VLMs）现在足够先进，可以支持一系列应用，包括回答复杂的视觉问题，并且越来越被期待以各种方式与图像互动。为了评估它们，当前的基准测试通常集中在特定领域（例如，阅读图表），通过构建带有预定义多项选择题（MCQs）的标注真实图像数据集来报告总体准确率分数。然而，这种基准测试涉及高昂的标注成本，存在信息泄漏的风险，并不能明确区分失败是源于视觉感知、推理还是通用知识的局限性。我们提出了一种新的评估方法，受眼科诊断的启发，利用程序生成合成图像以控制视觉属性并精确揭示VLMs的感知失败。具体而言，我们构建了具有不同挑战性内容变化的图集（例如，计数任务中的对象数量），同时保持其他视觉参数不变。这种诊断方法可进行系统性的压力测试和精细的失败分析，将重点从粗略的基准测试转向针对和可解释性强的评估VLM能力。我们的代码可在以下链接获取：this https URL。 

---
# Robustness Evaluation for Video Models with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的视频模型稳健性评估 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05431)  

**Abstract**: Evaluating the robustness of Video classification models is very challenging, specifically when compared to image-based models. With their increased temporal dimension, there is a significant increase in complexity and computational cost. One of the key challenges is to keep the perturbations to a minimum to induce misclassification. In this work, we propose a multi-agent reinforcement learning approach (spatial and temporal) that cooperatively learns to identify the given video's sensitive spatial and temporal regions. The agents consider temporal coherence in generating fine perturbations, leading to a more effective and visually imperceptible attack. Our method outperforms the state-of-the-art solutions on the Lp metric and the average queries. Our method enables custom distortion types, making the robustness evaluation more relevant to the use case. We extensively evaluate 4 popular models for video action recognition on two popular datasets, HMDB-51 and UCF-101. 

**Abstract (ZH)**: 评估视频分类模型的鲁棒性非常具有挑战性，特别是在与基于图像的模型相比时。随着其增加的时间维度，复杂性和计算成本显著增加。其中一个关键挑战是将扰动保持在最低限度以诱导误分类。在本工作中，我们提出了一种时空多智能体强化学习方法，该方法协同学习以识别给定视频的敏感时空区域。智能体在生成细微扰动时考虑时间一致性，从而产生更有效且视觉上不可感知的攻击。我们的方法在Lp度量和平均查询上优于现有最佳解决方案。我们的方法支持自定义失真类型，使鲁棒性评估更符合实际应用。我们对HMDB-51和UCF-101两个流行数据集上的4种流行视频动作识别模型进行了广泛评估。 

---
# Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning 

**Title (ZH)**: 基于视觉的强化学习的自我预测动力学泛化方法 

**Authors**: Kyungsoo Kim, Jeongsoo Ha, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05418)  

**Abstract**: Vision-based reinforcement learning requires efficient and robust representations of image-based observations, especially when the images contain distracting (task-irrelevant) elements such as shadows, clouds, and light. It becomes more important if those distractions are not exposed during training. We design a Self-Predictive Dynamics (SPD) method to extract task-relevant features efficiently, even in unseen observations after training. SPD uses weak and strong augmentations in parallel, and learns representations by predicting inverse and forward transitions across the two-way augmented versions. In a set of MuJoCo visual control tasks and an autonomous driving task (CARLA), SPD outperforms previous studies in complex observations, and significantly improves the generalization performance for unseen observations. Our code is available at this https URL. 

**Abstract (ZH)**: 基于视觉的强化学习需要在图像包含阴影、云朵和光线等干扰元素时，尤其是在这些干扰元素在训练中未被暴露的情况下，有效地提取相关的特征表示。我们设计了一种自我预测动力学（SPD）方法，在训练后的未见 observation 中也能高效提取任务相关特征。SPD 并行使用弱增强和强增强，并通过预测双向增强版本的逆向和正向转换来学习表示。在一系列 MuJoCo 视觉控制任务和自主驾驶任务（CARLA）中，SPD 在复杂 observation 中优于先前研究，并显著提高了未见 observation 的泛化性能。相关代码已发布在以下链接：此 https URL。 

---
# AD-EE: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving 

**Title (ZH)**: AD-EE: 自动驾驶中快速可靠视觉语言模型的早期退出方法 

**Authors**: Lianming Huang, Haibo Hu, Yufei Cui, Jiacheng Zuo, Shangyu Wu, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.05404)  

**Abstract**: With the rapid advancement of autonomous driving, deploying Vision-Language Models (VLMs) to enhance perception and decision-making has become increasingly common. However, the real-time application of VLMs is hindered by high latency and computational overhead, limiting their effectiveness in time-critical driving scenarios. This challenge is particularly evident when VLMs exhibit over-inference, continuing to process unnecessary layers even after confident predictions have been reached. To address this inefficiency, we propose AD-EE, an Early Exit framework that incorporates domain characteristics of autonomous driving and leverages causal inference to identify optimal exit layers. We evaluate our method on large-scale real-world autonomous driving datasets, including Waymo and the corner-case-focused CODA, as well as on a real vehicle running the Autoware Universe platform. Extensive experiments across multiple VLMs show that our method significantly reduces latency, with maximum improvements reaching up to 57.58%, and enhances object detection accuracy, with maximum gains of up to 44%. 

**Abstract (ZH)**: 随着自主驾驶技术的rapid advancement, 将视觉-语言模型(Vision-Language Models, VLMs)部署以增强感知和决策的应用越来越普遍。然而,VLMs的实时应用受到高延迟和计算开销的限制, 在时间敏感的驾驶场景中限制了其有效性。特别是在VLMs表现出过度推断的情况下, 即使在达成自信预测后仍持续处理不必要的层, 这一挑战尤为明显。为解决这一低效率问题, 我们提出AD-EE, 一种Early Exit框架, 结合自主驾驶领域的特性并利用因果推理来识别最优退出层。我们在大规模的实际自主驾驶数据集上评估了我们的方法, 包括Waymo和以边缘情况为重点的CODA, 以及在使用Autoware Universe平台的真实车辆上进行了评估。多个视觉-语言模型的广泛实验结果显示, 我们的方法显著减少了延迟, 最大改善幅度达到57.58%, 并提升了对象检测准确性, 最大增益达到44%。 

---
# Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation 

**Title (ZH)**: 基于注意力的变换器模型在跨语言图像 Captioning 中的应用：一种深入的综述与评估 

**Authors**: Israa A. Albadarneh, Bassam H. Hammo, Omar S. Al-Kadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05399)  

**Abstract**: Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning. 

**Abstract (ZH)**: 基于注意力的图像字幕模型综述 

---
# Gen4D: Synthesizing Humans and Scenes in the Wild 

**Title (ZH)**: Gen4D：在自然场景中合成人类和场景 

**Authors**: Jerrin Bright, Zhibo Wang, Yuhao Chen, Sirisha Rambhatla, John Zelek, David Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05397)  

**Abstract**: Lack of input data for in-the-wild activities often results in low performance across various computer vision tasks. This challenge is particularly pronounced in uncommon human-centric domains like sports, where real-world data collection is complex and impractical. While synthetic datasets offer a promising alternative, existing approaches typically suffer from limited diversity in human appearance, motion, and scene composition due to their reliance on rigid asset libraries and hand-crafted rendering pipelines. To address this, we introduce Gen4D, a fully automated pipeline for generating diverse and photorealistic 4D human animations. Gen4D integrates expert-driven motion encoding, prompt-guided avatar generation using diffusion-based Gaussian splatting, and human-aware background synthesis to produce highly varied and lifelike human sequences. Based on Gen4D, we present SportPAL, a large-scale synthetic dataset spanning three sports: baseball, icehockey, and soccer. Together, Gen4D and SportPAL provide a scalable foundation for constructing synthetic datasets tailored to in-the-wild human-centric vision tasks, with no need for manual 3D modeling or scene design. 

**Abstract (ZH)**: 野生活动输入数据的缺乏常常导致各种计算机视觉任务性能低下。这一挑战在复杂的非常见人类中心领域（如体育）表现尤为明显，因为在这些领域中，现实世界数据的收集既复杂又不现实。虽然合成数据集提供了有前景的替代方案，但现有方法通常由于依赖于刚体资产库和手工编写的渲染管线，而在人类外观、动作和场景构成的多样性方面存在局限。为了解决这一问题，我们介绍了Gen4D，这是一个全自动的生成多样化和逼真4D人体动画的pipeline。Gen4D结合了专家驱动的动作编码、提示引导的基于扩散的Gaussian溅射avatar生成以及人体意识背景合成，以产生高度多样且逼真的人类序列。基于Gen4D，我们介绍了SportPAL，这是一个涵盖三类运动（棒球、冰球和足球）的大规模合成数据集。结合Gen4D和SportPAL，我们提供了一个可扩展的基础，用于构建针对野生人类中心视觉任务的定制合成数据集，无需手动3D建模或场景设计。 

---
# How stealthy is stealthy? Studying the Efficacy of Black-Box Adversarial Attacks in the Real World 

**Title (ZH)**: 隐形的有多隐形？探究黑盒对抗攻击在实际环境中的有效性 

**Authors**: Francesco Panebianco, Mario D'Onghia, Stefano Zanero aand Michele Carminati  

**Link**: [PDF](https://arxiv.org/pdf/2506.05382)  

**Abstract**: Deep learning systems, critical in domains like autonomous vehicles, are vulnerable to adversarial examples (crafted inputs designed to mislead classifiers). This study investigates black-box adversarial attacks in computer vision. This is a realistic scenario, where attackers have query-only access to the target model. Three properties are introduced to evaluate attack feasibility: robustness to compression, stealthiness to automatic detection, and stealthiness to human inspection. State-of-the-Art methods tend to prioritize one criterion at the expense of others. We propose ECLIPSE, a novel attack method employing Gaussian blurring on sampled gradients and a local surrogate model. Comprehensive experiments on a public dataset highlight ECLIPSE's advantages, demonstrating its contribution to the trade-off between the three properties. 

**Abstract (ZH)**: 深度学习系统在自动驾驶等领域的应用极易受到对抗样本的攻击（特制的输入旨在误导分类器）。本文研究了计算机视觉中的黑盒对抗攻击。攻击者仅对目标模型具有查询访问权限，是一种现实场景。本文引入了评估攻击可行性的三个属性：对压缩的鲁棒性、自动检测中的隐形性和人工检查中的隐形性。现有先进方法往往在这些标准之间权衡取舍。我们提出了一种新颖的攻击方法ECLIPSE，该方法使用高斯模糊处理采样梯度并采用局部代理模型。在公共数据集上的全面实验突显了ECLIPSE的优势，展示了其在三个属性之间的权衡中的贡献。 

---
# Category Query Learning for Human-Object Interaction Classification 

**Title (ZH)**: 人类对象交互分类的类别查询学习 

**Authors**: Chi Xie, Fangao Zeng, Yue Hu, Shuang Liang, Yichen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2303.14005)  

**Abstract**: Unlike most previous HOI methods that focus on learning better human-object features, we propose a novel and complementary approach called category query learning. Such queries are explicitly associated to interaction categories, converted to image specific category representation via a transformer decoder, and learnt via an auxiliary image-level classification task. This idea is motivated by an earlier multi-label image classification method, but is for the first time applied for the challenging human-object interaction classification task. Our method is simple, general and effective. It is validated on three representative HOI baselines and achieves new state-of-the-art results on two benchmarks. 

**Abstract (ZH)**: 不同于大多数以往的人机对象方法侧重于学习更好的人-物特征，我们提出了一种新颖且互补的方法，称为类别查询学习。此类别查询明确与交互类别相关联，通过变压器解码器转换为图像特定的类别表示，并通过辅助的图像级分类任务进行学习。这一想法受到早期的多标签图像分类方法的启发，但首次应用于具有挑战性的交互分类任务。我们的方法简单、通用且有效，并在三个代表性的HOI基准上进行了验证，在两个基准上取得了新的最佳结果。 

---
