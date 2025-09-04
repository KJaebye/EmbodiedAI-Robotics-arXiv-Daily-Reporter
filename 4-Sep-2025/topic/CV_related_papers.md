# Real-Time Instrument Planning and Perception for Novel Measurements of Dynamic Phenomena 

**Title (ZH)**: 实时仪器规划与感知以实现动态现象的新测量 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien  

**Link**: [PDF](https://arxiv.org/pdf/2509.03500)  

**Abstract**: Advancements in onboard computing mean remote sensing agents can employ state-of-the-art computer vision and machine learning at the edge. These capabilities can be leveraged to unlock new rare, transient, and pinpoint measurements of dynamic science phenomena. In this paper, we present an automated workflow that synthesizes the detection of these dynamic events in look-ahead satellite imagery with autonomous trajectory planning for a follow-up high-resolution sensor to obtain pinpoint measurements. We apply this workflow to the use case of observing volcanic plumes. We analyze classification approaches including traditional machine learning algorithms and convolutional neural networks. We present several trajectory planning algorithms that track the morphological features of a plume and integrate these algorithms with the classifiers. We show through simulation an order of magnitude increase in the utility return of the high-resolution instrument compared to baselines while maintaining efficient runtimes. 

**Abstract (ZH)**: 随着机载计算技术的进步，遥感代理可以利用边缘的先进计算机视觉和机器学习技术。这些能力可以被利用以解锁新的稀有、短暂和精确的动态科学现象测量。在本文中，我们提出了一种自动工作流，该工作流结合了前瞻性卫星图像中的动态事件检测与自主轨迹规划，以获取高分辨率传感器的精确测量。我们将此工作流应用于观察火山喷气流的应用场景。我们分析了包括传统机器学习算法和卷积神经网络在内的分类方法。我们介绍了几种用于跟踪喷气流形态特征的轨迹规划算法，并将这些算法与分类器集成。通过模拟，我们展示了与基准相比，高分辨率仪器的效用提高了数量级，同时保持了高效的运行时间。 

---
# Continuous Saudi Sign Language Recognition: A Vision Transformer Approach 

**Title (ZH)**: 连续Saudi手语识别：一种视觉变换器方法 

**Authors**: Soukeina Elhassen, Lama Al Khuzayem, Areej Alhothali, Ohoud Alzamzami, Nahed Alowaidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.03467)  

**Abstract**: Sign language (SL) is an essential communication form for hearing-impaired and deaf people, enabling engagement within the broader society. Despite its significance, limited public awareness of SL often leads to inequitable access to educational and professional opportunities, thereby contributing to social exclusion, particularly in Saudi Arabia, where over 84,000 individuals depend on Saudi Sign Language (SSL) as their primary form of communication. Although certain technological approaches have helped to improve communication for individuals with hearing impairments, there continues to be an urgent requirement for more precise and dependable translation techniques, especially for Arabic sign language variants like SSL. Most state-of-the-art solutions have primarily focused on non-Arabic sign languages, resulting in a considerable absence of resources dedicated to Arabic sign language, specifically SSL. The complexity of the Arabic language and the prevalence of isolated sign language datasets that concentrate on individual words instead of continuous speech contribute to this issue. To address this gap, our research represents an important step in developing SSL resources. To address this, we introduce the first continuous Saudi Sign Language dataset called KAU-CSSL, focusing on complete sentences to facilitate further research and enable sophisticated recognition systems for SSL recognition and translation. Additionally, we propose a transformer-based model, utilizing a pretrained ResNet-18 for spatial feature extraction and a Transformer Encoder with Bidirectional LSTM for temporal dependencies, achieving 99.02\% accuracy at signer dependent mode and 77.71\% accuracy at signer independent mode. This development leads the way to not only improving communication tools for the SSL community but also making a substantial contribution to the wider field of sign language. 

**Abstract (ZH)**: 手语（SL）是听力障碍和 deaf 人士的重要沟通方式，有助于他们在更广泛的社会中进行交流。尽管手语非常重要，但由于公众对手语认识有限，这往往导致教育和职业机会获取不平等，进而加剧社会排斥，特别是在沙特阿拉伯，超过 84,000 人依赖沙特手语（SSL）作为他们的主要沟通方式。尽管某些技术手段有助于改善听力障碍人士的沟通，但仍然迫切需要更准确可靠的翻译技术，特别是针对阿拉伯手语变体如 SSL。现有的大多数最先进解决方案主要关注非阿拉伯手语，导致阿拉伯手语资源尤其是 SSL 的资源相对匮乏。阿拉伯语言的复杂性和孤立的手语数据集主要关注单个手语词汇而不是连贯的手语交流，进一步加剧了这一问题。为解决这一差距，我们的研究代表了开发 SSL 资源的重要一步。为此，我们介绍了第一个连续的沙特手语数据集 KAU-CSSL，专注于完整的句子以促进进一步研究，并实现针对 SSL 识别和翻译的复杂识别系统。此外，我们提出了一种基于变压器的模型，利用预训练的 ResNet-18 进行空间特征提取，并使用双向 LSTM 与变压器编码器处理时间依赖性，分别在书写者依赖模式和书写者独立模式下达到 99.02% 和 77.71% 的准确率。这一发展不仅为 SSL 社区提供更好的沟通工具，也为更广泛的手语领域做出了重要贡献。 

---
# TinyDrop: Tiny Model Guided Token Dropping for Vision Transformers 

**Title (ZH)**: TinyDrop: 由Tiny模型引导的Token丢弃方法在视觉变换器中的应用 

**Authors**: Guoxin Wang, Qingyuan Wang, Binhua Huang, Shaowu Chen, Deepu John  

**Link**: [PDF](https://arxiv.org/pdf/2509.03379)  

**Abstract**: Vision Transformers (ViTs) achieve strong performance in image classification but incur high computational costs from processing all image tokens. To reduce inference costs in large ViTs without compromising accuracy, we propose TinyDrop, a training-free token dropping framework guided by a lightweight vision model. The guidance model estimates the importance of tokens while performing inference, thereby selectively discarding low-importance tokens if large vit models need to perform attention calculations. The framework operates plug-and-play, requires no architectural modifications, and is compatible with diverse ViT architectures. Evaluations on standard image classification benchmarks demonstrate that our framework reduces FLOPs by up to 80% for ViTs with minimal accuracy degradation, highlighting its generalization capability and practical utility for efficient ViT-based classification. 

**Abstract (ZH)**: Vision Transformers (ViTs)通过图像分类表现强劲，但处理所有图像标记会带来高昂的计算成本。为在不牺牲准确性的前提下减少大ViTs的推理成本，我们提出TinyDrop，这是一种基于轻量级视觉模型的无训练-token丢弃框架。指导模型在推理过程中估计标记的重要性，从而在大型vit模型需要执行注意力计算时选择性地丢弃低重要性标记。该框架插即用，无需修改架构，并且兼容多种ViT架构。在标准图像分类基准上的评估结果显示，本框架可将ViTs的FLOPs最多降低80%，同时准确率下降可忽略不计，突显了其泛化能力和高效ViT分类的实际应用价值。 

---
# Heatmap Guided Query Transformers for Robust Astrocyte Detection across Immunostains and Resolutions 

**Title (ZH)**: heatmap引导的查询变换器在跨免疫染色和分辨率的星形胶质细胞检测中的稳健检测 

**Authors**: Xizhe Zhang, Jiayang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03323)  

**Abstract**: Astrocytes are critical glial cells whose altered morphology and density are hallmarks of many neurological disorders. However, their intricate branching and stain dependent variability make automated detection of histological images a highly challenging task. To address these challenges, we propose a hybrid CNN Transformer detector that combines local feature extraction with global contextual reasoning. A heatmap guided query mechanism generates spatially grounded anchors for small and faint astrocytes, while a lightweight Transformer module improves discrimination in dense clusters. Evaluated on ALDH1L1 and GFAP stained astrocyte datasets, the model consistently outperformed Faster R-CNN, YOLOv11 and DETR, achieving higher sensitivity with fewer false positives, as confirmed by FROC analysis. These results highlight the potential of hybrid CNN Transformer architectures for robust astrocyte detection and provide a foundation for advanced computational pathology tools. 

**Abstract (ZH)**: 星形胶质细胞是关键的胶质细胞，其异常形态和密度是许多神经系统疾病的特点。然而，它们复杂的分支结构和染色依赖的变异使自动化检测组织学图像成为一个极具挑战的任务。为应对这些挑战，我们提出了一种结合局部特征提取与全局上下文推理的混合CNN变压器检测器。热图引导的查询机制生成空间定位的锚点，以识别小而弱的星形胶质细胞，而轻量级的变压器模块则在密集簇中提高区分能力。该模型在ALDH1L1和GFAP染色的星形胶质细胞数据集上测试，一致优于Faster R-CNN、YOLOv11和DETR，展现出更高的敏感性并减少假阳性，FROC分析证实了这一点。这些结果突显了混合CNN transformer架构在稳健星形胶质细胞检测中的潜力，并为先进的计算病理学工具提供了基础。 

---
# LGBP-OrgaNet: Learnable Gaussian Band Pass Fusion of CNN and Transformer Features for Robust Organoid Segmentation and Tracking 

**Title (ZH)**: LGBP-OrgaNet: 可学习的高斯带通融合网络用于稳健的类器官分割与追踪 

**Authors**: Jing Zhang, Siying Tao, Jiao Li, Tianhe Wang, Junchen Wu, Ruqian Hao, Xiaohui Du, Ruirong Tan, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.03221)  

**Abstract**: Organoids replicate organ structure and function, playing a crucial role in fields such as tumor treatment and drug screening. Their shape and size can indicate their developmental status, but traditional fluorescence labeling methods risk compromising their structure. Therefore, this paper proposes an automated, non-destructive approach to organoid segmentation and tracking. We introduced the LGBP-OrgaNet, a deep learning-based system proficient in accurately segmenting, tracking, and quantifying organoids. The model leverages complementary information extracted from CNN and Transformer modules and introduces the innovative feature fusion module, Learnable Gaussian Band Pass Fusion, to merge data from two branches. Additionally, in the decoder, the model proposes a Bidirectional Cross Fusion Block to fuse multi-scale features, and finally completes the decoding through progressive concatenation and upsampling. SROrga demonstrates satisfactory segmentation accuracy and robustness on organoids segmentation datasets, providing a potent tool for organoid research. 

**Abstract (ZH)**: 类器官再现器官结构和功能，在肿瘤治疗和药物筛选等领域发挥着重要作用。它们的形状和大小可以反映其发育状态，但传统的荧光标记方法可能损害其结构。因此，本文提出了一种自动化的非破坏性类器官分割与跟踪方法。我们引入了基于深度学习的LGBP-OrgaNet系统，能够准确地分割、跟踪和定量类器官。该模型利用从CNN和Transformer模块中提取的互补信息，并引入了可学习高斯带通融合模块，将两个分支的数据进行融合。此外，在解码器中，模型提出了双向交叉融合块来融合多尺度特征，并最终通过逐步连接和上采样完成解码。SROrga在类器官分割数据集上展示了满意的分割精度和鲁棒性，为类器官研究提供了有力的工具。 

---
# AutoDetect: Designing an Autoencoder-based Detection Method for Poisoning Attacks on Object Detection Applications in the Military Domain 

**Title (ZH)**: AutoDetect：针对军事领域目标检测应用中的投毒攻击的自动编码器基于检测方法设计 

**Authors**: Alma M. Liezenga, Stefan Wijnja, Puck de Haan, Niels W. T. Brink, Jip J. van Stijn, Yori Kamphuis, Klamer Schutte  

**Link**: [PDF](https://arxiv.org/pdf/2509.03179)  

**Abstract**: Poisoning attacks pose an increasing threat to the security and robustness of Artificial Intelligence systems in the military domain. The widespread use of open-source datasets and pretrained models exacerbates this risk. Despite the severity of this threat, there is limited research on the application and detection of poisoning attacks on object detection systems. This is especially problematic in the military domain, where attacks can have grave consequences. In this work, we both investigate the effect of poisoning attacks on military object detectors in practice, and the best approach to detect these attacks. To support this research, we create a small, custom dataset featuring military vehicles: MilCivVeh. We explore the vulnerability of military object detectors for poisoning attacks by implementing a modified version of the BadDet attack: a patch-based poisoning attack. We then assess its impact, finding that while a positive attack success rate is achievable, it requires a substantial portion of the data to be poisoned -- raising questions about its practical applicability. To address the detection challenge, we test both specialized poisoning detection methods and anomaly detection methods from the visual industrial inspection domain. Since our research shows that both classes of methods are lacking, we introduce our own patch detection method: AutoDetect, a simple, fast, and lightweight autoencoder-based method. Our method shows promising results in separating clean from poisoned samples using the reconstruction error of image slices, outperforming existing methods, while being less time- and memory-intensive. We urge that the availability of large, representative datasets in the military domain is a prerequisite to further evaluate risks of poisoning attacks and opportunities patch detection. 

**Abstract (ZH)**: 中毒攻击日益威胁军事领域人工智能系统的安全性和鲁棒性。开源数据集和预训练模型的广泛应用进一步加剧了这一风险。尽管该威胁极为严重，但针对目标检测系统中毒攻击的应用与检测研究仍有限。特别是在军事领域，攻击可能导致严重后果。在本研究中，我们不仅探讨了中毒攻击对军事目标检测器的实际影响，还研究了检测这些攻击的最佳方法。为支持该研究，我们创建了一个小型定制数据集——MilCivVeh，其中包括军事车辆。我们通过实现基于补丁的BadDet攻击变体来探索军事目标检测器对中毒攻击的脆弱性。然后评估其影响，发现虽然可以实现一定的攻击成功率，但需要大量数据被中毒——这引发了其实际应用性的质疑。为应对检测挑战，我们测试了专门的中毒检测方法以及来自视觉工业检查领域的异常检测方法。鉴于我们的研究发现这两种方法都有局限性，我们引入了我们自己的补丁检测方法——AutoDetect，这是一种基于简单、快速且轻量级自动编码器的方法。该方法通过图像切片的重构误差分离干净样本和中毒样本，表现出优于现有方法的性能，同时占用时间及内存较少。我们呼吁在军事领域拥有大量并具有代表性的数据集是评估中毒攻击风险和补丁检测机会的先决条件。 

---
# S2M2ECG: Spatio-temporal bi-directional State Space Model Enabled Multi-branch Mamba for ECG 

**Title (ZH)**: S2M2ECG：空间时间双向状态空间模型驱动的多分支Mamba心电图分析方法 

**Authors**: Huaicheng Zhang, Ruoxin Wang, Chenlian Zhou, Jiguang Shi, Yue Ge, Zhoutong Li, Sheng Chang, Hao Wang, Jin He, Qijun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03066)  

**Abstract**: As one of the most effective methods for cardiovascular disease (CVD) diagnosis, multi-lead Electrocardiogram (ECG) signals present a characteristic multi-sensor information fusion challenge that has been continuously researched in deep learning domains. Despite the numerous algorithms proposed with different DL architectures, maintaining a balance among performance, computational complexity, and multi-source ECG feature fusion remains challenging. Recently, state space models (SSMs), particularly Mamba, have demonstrated remarkable effectiveness across various fields. Their inherent design for high-efficiency computation and linear complexity makes them particularly suitable for low-dimensional data like ECGs. This work proposes S2M2ECG, an SSM architecture featuring three-level fusion mechanisms: (1) Spatio-temporal bi-directional SSMs with segment tokenization for low-level signal fusion, (2) Intra-lead temporal information fusion with bi-directional scanning to enhance recognition accuracy in both forward and backward directions, (3) Cross-lead feature interaction modules for spatial information fusion. To fully leverage the ECG-specific multi-lead mechanisms inherent in ECG signals, a multi-branch design and lead fusion modules are incorporated, enabling individual analysis of each lead while ensuring seamless integration with others. Experimental results reveal that S2M2ECG achieves superior performance in the rhythmic, morphological, and clinical scenarios. Moreover, its lightweight architecture ensures it has nearly the fewest parameters among existing models, making it highly suitable for efficient inference and convenient deployment. Collectively, S2M2ECG offers a promising alternative that strikes an excellent balance among performance, computational complexity, and ECG-specific characteristics, paving the way for high-performance, lightweight computations in CVD diagnosis. 

**Abstract (ZH)**: 基于时空双向状态空间模型的多导联心电图特征融合方法（S2M2ECG） 

---
# MedLiteNet: Lightweight Hybrid Medical Image Segmentation Model 

**Title (ZH)**: MedLiteNet: 轻量级混合医学图像分割模型 

**Authors**: Pengyang Yu, Haoquan Wang, Gerard Marks, Tahar Kechadi, Laurence T. Yang, Sahraoui Dhelim, Nyothiri Aung  

**Link**: [PDF](https://arxiv.org/pdf/2509.03041)  

**Abstract**: Accurate skin-lesion segmentation remains a key technical challenge for computer-aided diagnosis of skin cancer. Convolutional neural networks, while effective, are constrained by limited receptive fields and thus struggle to model long-range dependencies. Vision Transformers capture global context, yet their quadratic complexity and large parameter budgets hinder use on the small-sample medical datasets common in dermatology. We introduce the MedLiteNet, a lightweight CNN Transformer hybrid tailored for dermoscopic segmentation that achieves high precision through hierarchical feature extraction and multi-scale context aggregation. The encoder stacks depth-wise Mobile Inverted Bottleneck blocks to curb computation, inserts a bottleneck-level cross-scale token-mixing unit to exchange information between resolutions, and embeds a boundary-aware self-attention module to sharpen lesion contours. 

**Abstract (ZH)**: 准确的皮肤病灶分割仍然是皮肤癌计算机辅助诊断中的关键技术挑战。MedLiteNet，一种针对皮肤镜分割的轻量化CNNTransformer混合模型，通过分层特征提取和多尺度上下文聚合实现高精度。 

---
# Lesion-Aware Visual-Language Fusion for Automated Image Captioning of Ulcerative Colitis Endoscopic Examinations 

**Title (ZH)**: 基于病灶aware的视觉-语言融合方法：全自动溃疡性结肠炎内镜检查图像captioning 

**Authors**: Alexis Ivan Lopez Escamilla, Gilberto Ochoa, Sharib Al  

**Link**: [PDF](https://arxiv.org/pdf/2509.03011)  

**Abstract**: We present a lesion-aware image captioning framework for ulcerative colitis (UC). The model integrates ResNet embeddings, Grad-CAM heatmaps, and CBAM-enhanced attention with a T5 decoder. Clinical metadata (MES score 0-3, vascular pattern, bleeding, erythema, friability, ulceration) is injected as natural-language prompts to guide caption generation. The system produces structured, interpretable descriptions aligned with clinical practice and provides MES classification and lesion tags. Compared with baselines, our approach improves caption quality and MES classification accuracy, supporting reliable endoscopic reporting. 

**Abstract (ZH)**: 一种溃疡性结肠炎病变感知的图像描述框架：结合ResNet嵌入、Grad-CAM热图和CBAM增强注意力的T5解码器应用于溃疡性结肠炎的临床图像描述 

---
# KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models 

**Title (ZH)**: KEPT：知识增强的连续驾驶帧轨迹预测方法基于视觉语言模型 

**Authors**: Yujin Wang, Tianyi Wang, Quanfeng Liu, Wenxian Fan, Junfeng Jiao, Christian Claudel, Yunbing Yan, Bingzhao Gao, Jianqiang Wang, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.02966)  

**Abstract**: Accurate short-horizon trajectory prediction is pivotal for safe and reliable autonomous driving, yet existing vision-language models (VLMs) often fail to effectively ground their reasoning in scene dynamics and domain knowledge. To address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM framework that predicts ego trajectories directly from consecutive front-view driving frames. KEPT couples a temporal frequency-spatial fusion (TFSF) video encoder, trained via self-supervised learning with hard-negative mining, with a scalable k-means + HNSW retrieval stack that supplies scene-aligned exemplars. Retrieved priors are embedded into chain-of-thought (CoT) prompts with explicit planning constraints, while a triple-stage fine-tuning schedule incrementally aligns the language head to metric spatial cues, physically feasible motion, and temporally conditioned front-view planning. Evaluated on nuScenes dataset, KEPT achieves state-of-the-art performance across open-loop protocols: under NoAvg, it achieves 0.70m average L2 with a 0.21\% collision rate; under TemAvg with lightweight ego status, it attains 0.31m average L2 and a 0.07\% collision rate. Ablation studies show that all three fine-tuning stages contribute complementary benefits, and that using Top-2 retrieved exemplars yields the best accuracy-safety trade-off. The k-means-clustered HNSW index delivers sub-millisecond retrieval latency, supporting practical deployment. These results indicate that retrieval-augmented, CoT-guided VLMs offer a promising, data-efficient pathway toward interpretable and trustworthy autonomous driving. 

**Abstract (ZH)**: 知识增强的短时轨迹预测框架 

---
# A Single Detect Focused YOLO Framework for Robust Mitotic Figure Detection 

**Title (ZH)**: 基于单次检测的聚焦YOLO框架用于稳健的有丝分裂 figure 检测 

**Authors**: Yasemin Topuz, M. Taha Gökcan, Serdar Yıldız, Songül Varlı  

**Link**: [PDF](https://arxiv.org/pdf/2509.02637)  

**Abstract**: Mitotic figure detection is a crucial task in computational pathology, as mitotic activity serves as a strong prognostic marker for tumor aggressiveness. However, domain variability that arises from differences in scanners, tissue types, and staining protocols poses a major challenge to the robustness of automated detection methods. In this study, we introduce SDF-YOLO (Single Detect Focused YOLO), a lightweight yet domain-robust detection framework designed specifically for small, rare targets such as mitotic figures. The model builds on YOLOv11 with task-specific modifications, including a single detection head aligned with mitotic figure scale, coordinate attention to enhance positional sensitivity, and improved cross-channel feature mixing. Experiments were conducted on three datasets that span human and canine tumors: MIDOG ++, canine cutaneous mast cell tumor (CCMCT), and canine mammary carcinoma (CMC). When submitted to the preliminary test set for the MIDOG2025 challenge, SDF-YOLO achieved an average precision (AP) of 0.799, with a precision of 0.758, a recall of 0.775, an F1 score of 0.766, and an FROC-AUC of 5.793, demonstrating both competitive accuracy and computational efficiency. These results indicate that SDF-YOLO provides a reliable and efficient framework for robust mitotic figure detection across diverse domains. 

**Abstract (ZH)**: SDF-YOLO：一种针对mitotic figures的轻量级且 domain-robust 的检测框架 

---
# A Two-Stage Strategy for Mitosis Detection Using Improved YOLO11x Proposals and ConvNeXt Classification 

**Title (ZH)**: 基于改进YOLO11x提案和ConvNeXt分类的两阶段.mitosis检测策略 

**Authors**: Jie Xiao, Mengye Lyu, Shaojun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02627)  

**Abstract**: MIDOG 2025 Track 1 requires mitosis detection in whole-slide images (WSIs) containing non-tumor, inflamed, and necrotic regions. Due to the complicated and heterogeneous context, as well as possible artifacts, there are often false positives and false negatives, thus degrading the detection F1-score. To address this problem, we propose a two-stage framework. Firstly, an improved YOLO11x, integrated with EMA attention and LSConv, is employed to generate mitosis candidates. We use a low confidence threshold to generate as many proposals as possible, ensuring the detection recall. Then, a ConvNeXt-Tiny classifier is employed to filter out the false positives, ensuring the detection precision. Consequently, the proposed two-stage framework can generate a high detection F1-score. Evaluated on a fused dataset comprising MIDOG++, MITOS_WSI_CCMCT, and MITOS_WSI_CMC, our framework achieves an F1-score of 0.882, which is 0.035 higher than the single-stage YOLO11x baseline. This performance gain is produced by a significant precision improvement, from 0.762 to 0.839, and a comparable recall. The code is available at this https URL. 

**Abstract (ZH)**: MIDOG 2025 轨道 1 要求在包含非肿瘤、炎症和坏死区域的全滑片图像（WSI）中检测有丝分裂。由于复杂的复杂数字化背景以及潜在的伪影，常常会出现假阳性和假阴性，从而降低检测的 F1 分数。为了解决这个问题，我们提出了一种两阶段框架。首先，结合 EMA 注意力和 LSConv 的改进版 YOLO11x 被用于生成有丝分裂候选区域。我们采用较低的置信度阈值生成尽可能多的建议，确保检测召回率。然后，使用 ConvNeXt-Tiny 分类器进一步筛选出假阳性结果，确保检测的精度。因此，提出的两阶段框架可以生成高 F1 分数。在融合了 MIDOG++、MITOS_WSI_CCMCT 和 MITOS_WSI_CMC 的数据集上，我们的框架实现了 0.882 的 F1 分数，比单阶段 YOLO11x 基线提高了 0.035。此性能提升来自于显著的精度提升，从 0.762 提高到 0.839，召回率相当。代码发布于该网址。 

---
# Is Synthetic Image Augmentation Useful for Imbalanced Classification Problems? Case-Study on the MIDOG2025 Atypical Cell Detection Competition 

**Title (ZH)**: 合成图像增强对不平衡分类问题有用吗？以MIDOG2025异常细胞检测竞赛为例 

**Authors**: Leire Benito-Del-Valle, Pedro A. Moreno-Sánchez, Itziar Egusquiza, Itsaso Vitoria, Artzai Picón, Cristina López-Saratxaga, Adrian Galdran  

**Link**: [PDF](https://arxiv.org/pdf/2509.02612)  

**Abstract**: The MIDOG 2025 challenge extends prior work on mitotic figure detection by introducing a new Track 2 on atypical mitosis classification. This task aims to distinguish normal from atypical mitotic figures in histopathology images, a clinically relevant but highly imbalanced and cross-domain problem. We investigated two complementary backbones: (i) ConvNeXt-Small, pretrained on ImageNet, and (ii) a histopathology-specific ViT from Lunit trained via self-supervision. To address the strong prevalence imbalance (9408 normal vs. 1741 atypical), we synthesized additional atypical examples to approximate class balance and compared models trained with real-only vs. real+synthetic data. Using five-fold cross-validation, both backbones reached strong performance (mean AUROC approximately 95 percent), with ConvNeXt achieving slightly higher peaks while Lunit exhibited greater fold-to-fold stability. Synthetic balancing, however, did not lead to consistent improvements. On the organizers' preliminary hidden test set, explicitly designed as an out-of-distribution debug subset, ConvNeXt attained the highest AUROC (95.4 percent), whereas Lunit remained competitive on balanced accuracy. These findings suggest that both ImageNet and domain-pretrained backbones are viable for atypical mitosis classification, with domain-pretraining conferring robustness and ImageNet pretraining reaching higher peaks, while naive synthetic balancing has limited benefit. Full hidden test set results will be reported upon challenge completion. 

**Abstract (ZH)**: MIDOG 2025 挑战赛扩展了前期关于有丝分裂图检测的工作，通过引入一个非典型有丝分裂分类的新赛道 Track 2。该任务旨在区分病理图像中的正常与非典型有丝分裂图，这是一个临床相关但高度不平衡且跨领域的难题。我们研究了两个互补的骨干网络：(i) ImageNet 上预训练的 ConvNeXt-Small，(ii) 由 Lunit 提供并在自监督下训练的病理专用 ViT。为解决强先验不平衡问题（正常有丝分裂图 9408 例 vs. 非典型有丝分裂图 1741 例），我们合成额外的非典型样本以逼近类平衡，并比较了使用真实数据 vs. 真实数据加合成数据训练的模型。使用五折交叉验证，两个骨干网络都取得了强劲的表现（平均 AUROC 约 95%），ConvNeXt 达到略高峰值，而 Lunit 展现了更好的折间稳定性。合成平衡并未带来一致的改善。在组织者初步隐藏测试集中，该集明确设计为一个离群值调试子集，ConvNeXt 达到了最高的 AUROC（95.4%），而 Lunit 在平衡准确率上仍然具有竞争力。这些发现表明，对于非典型有丝分裂分类，ImageNet 和领域预训练的骨干网络都是可行的，领域预训练提供鲁棒性，而 ImageNet 预训练达到更高峰值，未经修改的合成平衡具有有限的效果。在挑战赛完成后，将报告完整隐藏测试集的结果。 

---
# Robust Pan-Cancer Mitotic Figure Detection with YOLOv12 

**Title (ZH)**: 基于YOLOv12的稳健全景癌变有丝分裂图检测 

**Authors**: Raphaël Bourgade, Guillaume Balezo, Thomas Walter  

**Link**: [PDF](https://arxiv.org/pdf/2509.02593)  

**Abstract**: Mitotic figures represent a key histoprognostic feature in tumor pathology, providing crucial insights into tumor aggressiveness and proliferation. However, their identification remains challenging, subject to significant inter-observer variability, even among experienced pathologists. To address this issue, the MItosis DOmain Generalization (MIDOG) 2025 challenge marks the third edition of an international competition aiming to develop robust mitosis detection algorithms. In this paper, we present a mitotic figures detection approach based on the YOLOv12 object detection architecture, achieving a $F_1$-score of 0.801 on the preliminary test set of the MIDOG 2025 challenge, without relying on external data. 

**Abstract (ZH)**: 分裂相是肿瘤病理中一个关键的组织预后特征，提供了关于肿瘤侵袭性和增殖的重要见解。然而，其识别仍然具有挑战性，甚至在经验丰富的病理学家之间也存在显著的主观差异。为了应对这一问题，MItosis DOmain Generalization (MIDOG) 2025 挑战赛是旨在开发稳健的分裂相检测算法的第三届国际竞赛。在本文中，我们提出了一种基于 YOLOv12 对象检测架构的分裂相检测方法，在 MIDOG 2025 挑战赛初步测试集上达到了 0.801 的 $F_1$-分数，未依赖外部数据。 

---
# Normal and Atypical Mitosis Image Classifier using Efficient Vision Transformer 

**Title (ZH)**: 使用高效视觉变换器的正常与异常有丝分裂图像分类器 

**Authors**: Xuan Qi, Dominic Labella, Thomas Sanford, Maxwell Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.02589)  

**Abstract**: We tackle atypical versus normal mitosis classification in the MIDOG 2025 challenge using EfficientViT-L2, a hybrid CNN--ViT architecture optimized for accuracy and efficiency. A unified dataset of 13,938 nuclei from seven cancer types (MIDOG++ and AMi-Br) was used, with atypical mitoses comprising ~15. To assess domain generalization, we applied leave-one-cancer-type-out cross-validation with 5-fold ensembles, using stain-deconvolution for image augmentation. For challenge submissions, we trained an ensemble with the same 5-fold split but on all cancer types. In the preliminary evaluation phase, this model achieved balanced accuracy of 0.859, ROC AUC of 0.942, and raw accuracy of 0.85, demonstrating competitive and well-balanced performance across metrics. 

**Abstract (ZH)**: 我们使用EfficientViT-L2架构在MIDOG 2025挑战中解决异常有丝分裂与正常有丝分裂的分类问题，EfficientViT-L2是一种优化准确性和效率的混合CNN-ViT架构。我们使用来自七种癌症类型（MIDOG++和AMi-Br）的统一数据集（13,938个核），其中异常有丝分裂约占15%。为了评估域泛化能力，我们采用了留一癌种交叉验证方法，并使用去染色技术进行图像增强。对于挑战提交，我们在相同的5折分割上训练了一个包含所有癌种的数据集。在初步评估阶段，该模型的平衡准确率为0.859，ROC AUC为0.942，原始准确率为0.85，显示出跨指标上具有竞争力和均衡的表现。 

---
# MitoDetect++: A Domain-Robust Pipeline for Mitosis Detection and Atypical Subtyping 

**Title (ZH)**: MitoDetect++: 一种领域稳健的纺锤体检测和非典型亚型分类管道 

**Authors**: Esha Sadia Nasir, Jiaqi Lv, Mostafa Jahanifer, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2509.02586)  

**Abstract**: Automated detection and classification of mitotic figures especially distinguishing atypical from normal remain critical challenges in computational pathology. We present MitoDetect++, a unified deep learning pipeline designed for the MIDOG 2025 challenge, addressing both mitosis detection and atypical mitosis classification. For detection (Track 1), we employ a U-Net-based encoder-decoder architecture with EfficientNetV2-L as the backbone, enhanced with attention modules, and trained via combined segmentation losses. For classification (Track 2), we leverage the Virchow2 vision transformer, fine-tuned efficiently using Low-Rank Adaptation (LoRA) to minimize resource consumption. To improve generalization and mitigate domain shifts, we integrate strong augmentations, focal loss, and group-aware stratified 5-fold cross-validation. At inference, we deploy test-time augmentation (TTA) to boost robustness. Our method achieves a balanced accuracy of 0.892 across validation domains, highlighting its clinical applicability and scalability across tasks. 

**Abstract (ZH)**: 自动检测和分类有丝分裂图谱，尤其是区分异常与正常有丝分裂仍然是在计算病理学中面临的关键挑战。我们提出MitoDetect++，一个用于MIDOG 2025挑战的统一深度学习管道，旨在解决有丝分裂检测和异常有丝分裂分类问题。对于检测（赛道1），我们采用基于U-Net的编码解码架构，以EfficientNetV2-L作为骨干网络，并通过注意力模块增强，结合分割损失进行训练。对于分类（赛道2），我们利用Virchow2视觉变换器，并通过低秩适应（LoRA）高效微调，以减少资源消耗。为了提高泛化能力和减轻领域偏移，我们整合了强增强、焦点损失和组意识分层5折交叉验证。在推断阶段，我们部署测试时增强（TTA）以提高鲁棒性。我们的方法在验证域上实现了0.892的平衡准确率，突显其在不同任务中的临床适用性和可扩展性。 

---
