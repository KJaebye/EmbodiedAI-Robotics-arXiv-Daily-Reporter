# PianoVAM: A Multimodal Piano Performance Dataset 

**Title (ZH)**: PianoVAM：一种多模态钢琴表演数据集 

**Authors**: Yonghyun Kim, Junhyung Park, Joonhyung Bae, Kirak Kim, Taegyun Kwon, Alexander Lerch, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2509.08800)  

**Abstract**: The multimodal nature of music performance has driven increasing interest in data beyond the audio domain within the music information retrieval (MIR) community. This paper introduces PianoVAM, a comprehensive piano performance dataset that includes videos, audio, MIDI, hand landmarks, fingering labels, and rich metadata. The dataset was recorded using a Disklavier piano, capturing audio and MIDI from amateur pianists during their daily practice sessions, alongside synchronized top-view videos in realistic and varied performance conditions. Hand landmarks and fingering labels were extracted using a pretrained hand pose estimation model and a semi-automated fingering annotation algorithm. We discuss the challenges encountered during data collection and the alignment process across different modalities. Additionally, we describe our fingering annotation method based on hand landmarks extracted from videos. Finally, we present benchmarking results for both audio-only and audio-visual piano transcription using the PianoVAM dataset and discuss additional potential applications. 

**Abstract (ZH)**: 多模态音乐表演的性质推动了音乐信息检索领域对音频以外数据的兴趣增加。本文介绍了PianoVAM，这是一个全面的钢琴表演数据集，包括视频、音频、MIDI、手部关键点、指法标签和丰富的元数据。数据集使用Disklavier钢琴录制，捕捉了业余钢琴演奏者在日常练习期间的音频和MIDI，并伴有同步的顶部视角视频，所有这些都处于实际多样化的表演条件下。手部关键点和指法标签通过预训练的手部姿态估计模型和半自动的指法标注算法提取。我们讨论了数据收集和不同模态之间对齐过程中遇到的挑战，并描述了我们基于视频中提取的手部关键点的指法标注方法，最后，我们使用PianoVAM数据集展示了仅音频和音频-视觉钢琴转录的基准测试结果，并讨论了其他潜在应用。 

---
# UOPSL: Unpaired OCT Predilection Sites Learning for Fundus Image Diagnosis Augmentation 

**Title (ZH)**: UOPSL: 无配对OCT倾向性病灶学习以增强眼底图像诊断 

**Authors**: Zhihao Zhao, Yinzheng Zhao, Junjie Yang, Xiangtong Yao, Quanmin Liang, Daniel Zapp, Kai Huang, Nassir Navab, M.Ali Nasseri  

**Link**: [PDF](https://arxiv.org/pdf/2509.08624)  

**Abstract**: Significant advancements in AI-driven multimodal medical image diagnosis have led to substantial improvements in ophthalmic disease identification in recent years. However, acquiring paired multimodal ophthalmic images remains prohibitively expensive. While fundus photography is simple and cost-effective, the limited availability of OCT data and inherent modality imbalance hinder further progress. Conventional approaches that rely solely on fundus or textual features often fail to capture fine-grained spatial information, as each imaging modality provides distinct cues about lesion predilection sites. In this study, we propose a novel unpaired multimodal framework \UOPSL that utilizes extensive OCT-derived spatial priors to dynamically identify predilection sites, enhancing fundus image-based disease recognition. Our approach bridges unpaired fundus and OCTs via extended disease text descriptions. Initially, we employ contrastive learning on a large corpus of unpaired OCT and fundus images while simultaneously learning the predilection sites matrix in the OCT latent space. Through extensive optimization, this matrix captures lesion localization patterns within the OCT feature space. During the fine-tuning or inference phase of the downstream classification task based solely on fundus images, where paired OCT data is unavailable, we eliminate OCT input and utilize the predilection sites matrix to assist in fundus image classification learning. Extensive experiments conducted on 9 diverse datasets across 28 critical categories demonstrate that our framework outperforms existing benchmarks. 

**Abstract (ZH)**: 基于广泛 OCT 提取的空间先验的无配对多模态框架 \UOPSL 在眼底图像病灶识别中的应用 

---
# Prompt-Driven Image Analysis with Multimodal Generative AI: Detection, Segmentation, Inpainting, and Interpretation 

**Title (ZH)**: 基于提示驱动的多模态生成AI图像分析：检测、分割、修复与解释 

**Authors**: Kaleem Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2509.08489)  

**Abstract**: Prompt-driven image analysis converts a single natural-language instruction into multiple steps: locate, segment, edit, and describe. We present a practical case study of a unified pipeline that combines open-vocabulary detection, promptable segmentation, text-conditioned inpainting, and vision-language description into a single workflow. The system works end to end from a single prompt, retains intermediate artifacts for transparent debugging (such as detections, masks, overlays, edited images, and before and after composites), and provides the same functionality through an interactive UI and a scriptable CLI for consistent, repeatable runs. We highlight integration choices that reduce brittleness, including threshold adjustments, mask inspection with light morphology, and resource-aware defaults. In a small, single-word prompt segment, detection and segmentation produced usable masks in over 90% of cases with an accuracy above 85% based on our criteria. On a high-end GPU, inpainting makes up 60 to 75% of total runtime under typical guidance and sampling settings, which highlights the need for careful tuning. The study offers implementation-guided advice on thresholds, mask tightness, and diffusion parameters, and details version pinning, artifact logging, and seed control to support replay. Our contribution is a transparent, reliable pattern for assembling modern vision and multimodal models behind a single prompt, with clear guardrails and operational practices that improve reliability in object replacement, scene augmentation, and removal. 

**Abstract (ZH)**: 基于提示的图像分析将单个自然语言指令转换为多个步骤：定位、分割、编辑和描述。我们提出了一种统一的工作流案例研究，该工作流结合了开放式词汇检测、可提示分割、文本条件插画填充和视觉语言描述。系统从单个提示端到端工作，保留中间结果以透明地调试（如检测结果、掩码、叠加、编辑的图像以及前后组合），并通过交互式UI和可脚本化的CLI界面提供一致且可重复的功能。我们强调了减少脆弱性的集成选择，包括阈值调整、轻度形态学掩码检查和资源感知默认值。在一个小型单词提示片段中，在我们的标准下，检测和分割在90%以上的案例中产生了可用的掩码，准确率超过85%。在高性能GPU上，插画填充通常占总运行时间的60%到75%，突显了仔细调参的必要性。该研究提供了阈值、掩码紧致性和扩散参数的实现指导建议，并详细说明了版本锁定、中间结果日志记录和种子控制，以支持重新运行。我们的贡献是提供了一种透明且可靠的工作流模式，用于在单个提示后组装现代视觉和多模态模型，并通过清晰的护栏和操作实践提高物体替换、场景增删和修改的可靠性。 

---
# Retrieval-Augmented VLMs for Multimodal Melanoma Diagnosis 

**Title (ZH)**: 多模态黑色素瘤诊断的检索增强大模型 

**Authors**: Jihyun Moon, Charmgil Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.08338)  

**Abstract**: Accurate and early diagnosis of malignant melanoma is critical for improving patient outcomes. While convolutional neural networks (CNNs) have shown promise in dermoscopic image analysis, they often neglect clinical metadata and require extensive preprocessing. Vision-language models (VLMs) offer a multimodal alternative but struggle to capture clinical specificity when trained on general-domain data. To address this, we propose a retrieval-augmented VLM framework that incorporates semantically similar patient cases into the diagnostic prompt. Our method enables informed predictions without fine-tuning and significantly improves classification accuracy and error correction over conventional baselines. These results demonstrate that retrieval-augmented prompting provides a robust strategy for clinical decision support. 

**Abstract (ZH)**: 准确且早期诊断恶性黑色素瘤对于改善患者预后至关重要。虽然卷积神经网络在皮肤镜图像分析中显示出潜力，但它们往往忽视临床元数据并需要大量的预处理。视觉语言模型提供了多模态的替代方案，但在使用通用领域数据训练时难以捕捉临床特异性。为解决这一问题，我们提出了一种检索增强的视觉语言模型框架，该框架将语义相似的患者病例融入诊断提示中。我们的方法在无需微调的情况下实现了知情预测，并且在分类准确性和错误修正方面显著优于传统的基准方法。这些结果表明，检索增强的提示策略是一种稳健的临床决策支持策略。 

---
# A New Dataset and Benchmark for Grounding Multimodal Misinformation 

**Title (ZH)**: 一个新的数据集和基准用于多模态 misinformation 定位 

**Authors**: Bingjian Yang, Danni Xu, Kaipeng Niu, Wenxuan Liu, Zheng Wang, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2509.08008)  

**Abstract**: The proliferation of online misinformation videos poses serious societal risks. Current datasets and detection methods primarily target binary classification or single-modality localization based on post-processed data, lacking the interpretability needed to counter persuasive misinformation. In this paper, we introduce the task of Grounding Multimodal Misinformation (GroundMM), which verifies multimodal content and localizes misleading segments across modalities. We present the first real-world dataset for this task, GroundLie360, featuring a taxonomy of misinformation types, fine-grained annotations across text, speech, and visuals, and validation with Snopes evidence and annotator reasoning. We also propose a VLM-based, QA-driven baseline, FakeMark, using single- and cross-modal cues for effective detection and grounding. Our experiments highlight the challenges of this task and lay a foundation for explainable multimodal misinformation detection. 

**Abstract (ZH)**: 在线错误信息视频的蔓延对社会构成严重风险。当前的数据集和检测方法主要针对二元分类或基于后处理数据的单模态定位，缺乏对抗有说服力的错误信息所需的可解释性。本文介绍了多模态错误信息定位任务（GroundMM），该任务验证多模态内容并在不同模态中定位误导性片段。我们首次提出了这一任务的第一个现实世界数据集GroundLie360，其中包括错误信息类型的分类体系、细粒度的跨文本、语音和视觉的注释，并通过Snopes证据和注释者推理进行了验证。我们还提出了一种基于VLM和QA的基线方法FakeMark，使用单模态和跨模态线索进行有效的检测和定位。我们的实验突显了该任务的挑战，并为进一步可解释的多模态错误信息检测奠定了基础。 

---
