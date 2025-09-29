# See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation 

**Title (ZH)**: 看、指、飞：一种无需学习的VLM框架用于通用无人驾驶航空导航 

**Authors**: Chih Yao Hu, Yang-Sen Lin, Yuna Lee, Chih-Hai Su, Jie-Ying Lee, Shr-Ruei Tsai, Chin-Yang Lin, Kuan-Wen Chen, Tsung-Wei Ke, Yu-Lun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.22653)  

**Abstract**: We present See, Point, Fly (SPF), a training-free aerial vision-and-language navigation (AVLN) framework built atop vision-language models (VLMs). SPF is capable of navigating to any goal based on any type of free-form instructions in any kind of environment. In contrast to existing VLM-based approaches that treat action prediction as a text generation task, our key insight is to consider action prediction for AVLN as a 2D spatial grounding task. SPF harnesses VLMs to decompose vague language instructions into iterative annotation of 2D waypoints on the input image. Along with the predicted traveling distance, SPF transforms predicted 2D waypoints into 3D displacement vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the traveling distance to facilitate more efficient navigation. Notably, SPF performs navigation in a closed-loop control manner, enabling UAVs to follow dynamic targets in dynamic environments. SPF sets a new state of the art in DRL simulation benchmark, outperforming the previous best method by an absolute margin of 63%. In extensive real-world evaluations, SPF outperforms strong baselines by a large margin. We also conduct comprehensive ablation studies to highlight the effectiveness of our design choice. Lastly, SPF shows remarkable generalization to different VLMs. Project page: this https URL 

**Abstract (ZH)**: 我们提出了See, Point, Fly (SPF)：一种基于视觉语言模型的无需训练的空中视觉与语言导航（AVLN）框架。SPF 能够根据任意类型的自由格式指令在任意环境条件下导航至任意目标。与现有基于视觉语言模型的方法将动作预测视为文本生成任务不同，我们的关键洞察是将AVLN中的动作预测视为二维空间定位任务。SPF 利用视觉语言模型将模糊的语言指令分解为对输入图像上二维航点的迭代标注。除了预测的行进距离外，SPF 还将预测的二维航点转换为用于无人机的三维位移向量作为动作命令。此外，SPF 还会自适应调整行进距离以促进更高效的导航。值得注意的是，SPF 以闭环控制的方式进行导航，能够使无人机在动态环境中追踪动态目标。在DRL仿真基准测试中，SPF 达到了新最优水平，绝对优势领先于之前的最佳方法63%。在广泛的现实世界评估中，SPF 显著优于强大的基线方法。我们还进行了全面的消融研究以突出我们设计选择的有效性。最后，SPF 在不同的视觉语言模型中展示出了卓越的泛化能力。项目页面：这个 https URL。 

---
# DHAGrasp: Synthesizing Affordance-Aware Dual-Hand Grasps with Text Instructions 

**Title (ZH)**: DHAGrasp: 基于文本指令合成知觉aware双臂抓取 

**Authors**: Quanzhou Li, Zhonghua Wu, Jingbo Wang, Chen Change Loy, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22175)  

**Abstract**: Learning to generate dual-hand grasps that respect object semantics is essential for robust hand-object interaction but remains largely underexplored due to dataset scarcity. Existing grasp datasets predominantly focus on single-hand interactions and contain only limited semantic part annotations. To address these challenges, we introduce a pipeline, SymOpt, that constructs a large-scale dual-hand grasp dataset by leveraging existing single-hand datasets and exploiting object and hand symmetries. Building on this, we propose a text-guided dual-hand grasp generator, DHAGrasp, that synthesizes Dual-Hand Affordance-aware Grasps for unseen objects. Our approach incorporates a novel dual-hand affordance representation and follows a two-stage design, which enables effective learning from a small set of segmented training objects while scaling to a much larger pool of unsegmented data. Extensive experiments demonstrate that our method produces diverse and semantically consistent grasps, outperforming strong baselines in both grasp quality and generalization to unseen objects. The project page is at this https URL. 

**Abstract (ZH)**: 学习生成尊重物体语义的双臂抓取对于稳健的手物交互至关重要，但由于数据集稀缺，这一领域仍处于很大程度上的未开发状态。现有的抓取数据集主要侧重于单手交互，并仅包含有限的语义部分标注。为了解决这些挑战，我们引入了一种名为SymOpt的管道，该管道通过利用现有单手数据集并利用物体和手的对称性构造了一个大规模的双臂抓取数据集。在此基础上，我们提出了一种文本引导的双臂抓取生成器DHAGrasp，用于生成未见物体的双臂可利用抓取。我们的方法结合了一种新颖的双臂可利用性表示，并采用两阶段设计，能够有效从少量分割的训练对象中学习，同时扩展到更大规模的非分割数据集。广泛的实验表明，我们的方法生成了多样且语义一致的抓取，在抓取质量和对未见物体的泛化方面均优于强基线方法。项目页面参见此链接：https://your-project-url.com 

---
# CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning 

**Title (ZH)**: CapRL: 通过强化学习激发密集图像描述能力 

**Authors**: Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.22647)  

**Abstract**: Image captioning is a fundamental task that bridges the visual and linguistic domains, playing a critical role in pre-training Large Vision-Language Models (LVLMs). Current state-of-the-art captioning models are typically trained with Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable data annotated by humans or proprietary models. This approach often leads to models that memorize specific ground-truth answers, limiting their generality and ability to generate diverse, creative descriptions. To overcome the limitation of SFT, we propose applying the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning. A primary challenge, however, is designing an objective reward function for the inherently subjective nature of what constitutes a "good" caption. We introduce Captioning Reinforcement Learning (CapRL), a novel training framework that redefines caption quality through its utility: a high-quality caption should enable a non-visual language model to accurately answer questions about the corresponding image. CapRL employs a decoupled two-stage pipeline where an LVLM generates a caption, and the objective reward is derived from the accuracy of a separate, vision-free LLM answering Multiple-Choice Questions based solely on that caption. As the first study to apply RLVR to the subjective image captioning task, we demonstrate that CapRL significantly enhances multiple settings. Pretraining on the CapRL-5M caption dataset annotated by CapRL-3B results in substantial gains across 12 benchmarks. Moreover, within the Prism Framework for caption quality evaluation, CapRL achieves performance comparable to Qwen2.5-VL-72B, while exceeding the baseline by an average margin of 8.4%. Code is available here: this https URL. 

**Abstract (ZH)**: 基于验证性奖励的图像字幕 reinforcement learning with verifiable rewards for image captioning 

---
# Chimera: Diagnosing Shortcut Learning in Visual-Language Understanding 

**Title (ZH)**: chimera: 视觉-语言理解中捷径学习的诊断 

**Authors**: Ziheng Chi, Yifan Hou, Chenxi Pang, Shaobo Cui, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22437)  

**Abstract**: Diagrams convey symbolic information in a visual format rather than a linear stream of words, making them especially challenging for AI models to process. While recent evaluations suggest that vision-language models (VLMs) perform well on diagram-related benchmarks, their reliance on knowledge, reasoning, or modality shortcuts raises concerns about whether they genuinely understand and reason over diagrams. To address this gap, we introduce Chimera, a comprehensive test suite comprising 7,500 high-quality diagrams sourced from Wikipedia; each diagram is annotated with its symbolic content represented by semantic triples along with multi-level questions designed to assess four fundamental aspects of diagram comprehension: entity recognition, relation understanding, knowledge grounding, and visual reasoning. We use Chimera to measure the presence of three types of shortcuts in visual question answering: (1) the visual-memorization shortcut, where VLMs rely on memorized visual patterns; (2) the knowledge-recall shortcut, where models leverage memorized factual knowledge instead of interpreting the diagram; and (3) the Clever-Hans shortcut, where models exploit superficial language patterns or priors without true comprehension. We evaluate 15 open-source VLMs from 7 model families on Chimera and find that their seemingly strong performance largely stems from shortcut behaviors: visual-memorization shortcuts have slight impact, knowledge-recall shortcuts play a moderate role, and Clever-Hans shortcuts contribute significantly. These findings expose critical limitations in current VLMs and underscore the need for more robust evaluation protocols that benchmark genuine comprehension of complex visual inputs (e.g., diagrams) rather than question-answering shortcuts. 

**Abstract (ZH)**: 图示以视觉格式而非线性字流传递符号信息，这使它们特别难以供AI模型处理。虽然最近的评估表明，视觉-语言模型(VLMs)在图示相关基准测试中表现良好，但它们依赖知识、推理或模态捷径的方式引发了对其是否真正理解并推理图示的担忧。为解决这一差距，我们引入了 Chimera，一个包含7,500个高质量图示的综合测试套件，这些图示来源于维基百科；每个图示都用语义三元组标注其符号内容，并配有多层次问题以评估图示理解的四个基本方面：实体识别、关系理解、知识接地和视觉推理。我们使用Chimera来测量视觉问答中三种捷径的存在：（1）视觉记忆捷径，其中VLMs依赖于记忆中的视觉模式；（2）知识回忆捷径，其中模型利用记忆中的事实知识而非解释图示；（3）Clever-Hans捷径，其中模型利用表面的语言模式或先验知识而没有真正的理解。我们将15个开源VLMs从7个模型家族进行Chimera上的评估，并发现它们看似强大的表现主要源于捷径行为：视觉记忆捷径影响轻微，知识回忆捷径起中等作用，而Clever-Hans捷径贡献显著。这些发现揭示了当前VLMs的关键局限性，并突显了需要使用更 robust 的评估协议来基准测试对复杂视觉输入（例如，图示）的真正理解，而非问答捷径。 

---
# RAU: Reference-based Anatomical Understanding with Vision Language Models 

**Title (ZH)**: RAU：基于参考的解剖理解与视觉语言模型 

**Authors**: Yiwei Li, Yikang Liu, Jiaqi Guo, Lin Zhao, Zheyuan Zhang, Xiao Chen, Boris Mailhe, Ankush Mukherjee, Terrence Chen, Shanhui Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22404)  

**Abstract**: Anatomical understanding through deep learning is critical for automatic report generation, intra-operative navigation, and organ localization in medical imaging; however, its progress is constrained by the scarcity of expert-labeled data. A promising remedy is to leverage an annotated reference image to guide the interpretation of an unlabeled target. Although recent vision-language models (VLMs) exhibit non-trivial visual reasoning, their reference-based understanding and fine-grained localization remain limited. We introduce RAU, a framework for reference-based anatomical understanding with VLMs. We first show that a VLM learns to identify anatomical regions through relative spatial reasoning between reference and target images, trained on a moderately sized dataset. We validate this capability through visual question answering (VQA) and bounding box prediction. Next, we demonstrate that the VLM-derived spatial cues can be seamlessly integrated with the fine-grained segmentation capability of SAM2, enabling localization and pixel-level segmentation of small anatomical regions, such as vessel segments. Across two in-distribution and two out-of-distribution datasets, RAU consistently outperforms a SAM2 fine-tuning baseline using the same memory setup, yielding more accurate segmentations and more reliable localization. More importantly, its strong generalization ability makes it scalable to out-of-distribution datasets, a property crucial for medical image applications. To the best of our knowledge, RAU is the first to explore the capability of VLMs for reference-based identification, localization, and segmentation of anatomical structures in medical images. Its promising performance highlights the potential of VLM-driven approaches for anatomical understanding in automated clinical workflows. 

**Abstract (ZH)**: 通过深度学习进行解剖理解对于医学影像的自动报告生成、术中导航和器官定位至关重要，但其进展受限于专家标注数据的稀缺性。一种有前景的解决方案是利用注释参考图像来指导未标注目标的解释。尽管近期的视觉-语言模型（VLMs）显示出了非平凡的视觉推理能力，但它们的参考驱动理解和细粒度定位能力仍有限。我们提出RAU，一种基于参考的解剖理解框架，使用VLMs。我们首先展示了VLM能够在适度大小的数据集上通过参考图像和目标图像之间的相对空间推理学习识别解剖区域。通过视觉问答（VQA）和边界框预测，验证了这一能力。随后，我们展示了VLM提取的空间线索可以无缝集成到SAM2的细粒度分割能力中，从而实现小解剖区域（如血管段）的精确定位和像素级分割。在两个在分布数据集和两个跨分布数据集中，RAU在相同的内存配置下始终优于SAM2微调基线，提供了更准确的分割和更可靠的定位。更重要的是，其强大的泛化能力使其可以扩展到跨分布数据集，这是医学影像应用中关键的属性。据我们所知，RAU是首次探索VLMs在医疗影像中进行基于参考的识别、定位和分割结构的能力。其有前景的性能突显了VLM驱动方法在自动化临床工作流程中进行解剖理解的潜力。 

---
# Deep Learning-Based Cross-Anatomy CT Synthesis Using Adapted nnResU-Net with Anatomical Feature Prioritized Loss 

**Title (ZH)**: 基于深度学习的adapted nnResU-Net介导的解剖特征优先损失跨解剖CT合成 

**Authors**: Javier Sequeiro González, Arthur Longuefosse, Miguel Díaz Benito, Álvaro García Martín, Fabien Baldacci  

**Link**: [PDF](https://arxiv.org/pdf/2509.22394)  

**Abstract**: We present a patch-based 3D nnUNet adaptation for MR to CT and CBCT to CT image translation using the multicenter SynthRAD2025 dataset, covering head and neck (HN), thorax (TH), and abdomen (AB) regions. Our approach leverages two main network configurations: a standard UNet and a residual UNet, both adapted from nnUNet for image synthesis. The Anatomical Feature-Prioritized (AFP) loss was introduced, which compares multilayer features extracted from a compact segmentation network trained on TotalSegmentator labels, enhancing reconstruction of clinically relevant structures. Input volumes were normalized per-case using zscore normalization for MRIs, and clipping plus dataset level zscore normalization for CBCT and CT. Training used 3D patches tailored to each anatomical region without additional data augmentation. Models were trained for 1000 and 1500 epochs, with AFP fine-tuning performed for 500 epochs using a combined L1+AFP objective. During inference, overlapping patches were aggregated via mean averaging with step size of 0.3, and postprocessing included reverse zscore normalization. Both network configurations were applied across all regions, allowing consistent model design while capturing local adaptations through residual learning and AFP loss. Qualitative and quantitative evaluation revealed that residual networks combined with AFP yielded sharper reconstructions and improved anatomical fidelity, particularly for bone structures in MR to CT and lesions in CBCT to CT, while L1only networks achieved slightly better intensity-based metrics. This methodology provides a stable solution for cross modality medical image synthesis, demonstrating the effectiveness of combining the automatic nnUNet pipeline with residual learning and anatomically guided feature losses. 

**Abstract (ZH)**: 基于补丁的3D nnUNet适应性研究：使用SynthRAD2025多中心数据集将MR转换为CT及CBCT转换为CT图像翻译，涵盖头部和颈部、胸腔和腹部区域 

---
# HiGS: History-Guided Sampling for Plug-and-Play Enhancement of Diffusion Models 

**Title (ZH)**: HiGS: 历史指导采样以提高扩散模型的即插即用增强功能 

**Authors**: Seyedmorteza Sadat, Farnood Salehi, Romann M. Weber  

**Link**: [PDF](https://arxiv.org/pdf/2509.22300)  

**Abstract**: While diffusion models have made remarkable progress in image generation, their outputs can still appear unrealistic and lack fine details, especially when using fewer number of neural function evaluations (NFEs) or lower guidance scales. To address this issue, we propose a novel momentum-based sampling technique, termed history-guided sampling (HiGS), which enhances quality and efficiency of diffusion sampling by integrating recent model predictions into each inference step. Specifically, HiGS leverages the difference between the current prediction and a weighted average of past predictions to steer the sampling process toward more realistic outputs with better details and structure. Our approach introduces practically no additional computation and integrates seamlessly into existing diffusion frameworks, requiring neither extra training nor fine-tuning. Extensive experiments show that HiGS consistently improves image quality across diverse models and architectures and under varying sampling budgets and guidance scales. Moreover, using a pretrained SiT model, HiGS achieves a new state-of-the-art FID of 1.61 for unguided ImageNet generation at 256$\times$256 with only 30 sampling steps (instead of the standard 250). We thus present HiGS as a plug-and-play enhancement to standard diffusion sampling that enables faster generation with higher fidelity. 

**Abstract (ZH)**: 基于历史引导的采样技术（HiGS）：一种提升扩散模型图像生成质量与效率的方法 

---
# Rigidity-Aware 3D Gaussian Deformation from a Single Image 

**Title (ZH)**: 基于刚性意识的单图三维高斯变形 

**Authors**: Jinhyeok Kim, Jaehun Bang, Seunghyun Seo, Kyungdon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2509.22222)  

**Abstract**: Reconstructing object deformation from a single image remains a significant challenge in computer vision and graphics. Existing methods typically rely on multi-view video to recover deformation, limiting their applicability under constrained scenarios. To address this, we propose DeformSplat, a novel framework that effectively guides 3D Gaussian deformation from only a single image. Our method introduces two main technical contributions. First, we present Gaussian-to-Pixel Matching which bridges the domain gap between 3D Gaussian representations and 2D pixel observations. This enables robust deformation guidance from sparse visual cues. Second, we propose Rigid Part Segmentation consisting of initialization and refinement. This segmentation explicitly identifies rigid regions, crucial for maintaining geometric coherence during deformation. By combining these two techniques, our approach can reconstruct consistent deformations from a single image. Extensive experiments demonstrate that our approach significantly outperforms existing methods and naturally extends to various applications,such as frame interpolation and interactive object manipulation. 

**Abstract (ZH)**: 仅从单张图像重构物体变形仍然是计算机视觉和图形学中的一个重大挑战。现有方法通常依赖多视角视频来恢复变形，这限制了其在受限场景中的应用。为解决这一问题，我们提出了一种名为DeformSplat的新框架，该框架能够仅通过单张图像有效引导3D高斯变形。我们的方法引入了两项主要的技术贡献。首先，我们提出了高斯到像素匹配技术，以弥合3D高斯表示与2D像素观测之间的域差距，从而能够从稀疏的视觉线索中实现稳健的变形引导。其次，我们提出了刚性部分分割，包括初始化和细化两个步骤，该分割明确地识别刚性区域，这对于在变形过程中保持几何一致性至关重要。通过结合这两种技术，我们的方法可以从单张图像中重建一致的变形。广泛的实验表明，我们的方法在性能上显著优于现有方法，并且自然地扩展到各种应用中，如帧内插和交互式对象操作。 

---
# REFINE-CONTROL: A Semi-supervised Distillation Method For Conditional Image Generation 

**Title (ZH)**: REFINE-CONTROL：一种条件图像生成的半监督精炼蒸馏方法 

**Authors**: Yicheng Jiang, Jin Yuan, Hua Yuan, Yao Zhang, Yong Rui  

**Link**: [PDF](https://arxiv.org/pdf/2509.22139)  

**Abstract**: Conditional image generation models have achieved remarkable results by leveraging text-based control to generate customized images. However, the high resource demands of these models and the scarcity of well-annotated data have hindered their deployment on edge devices, leading to enormous costs and privacy concerns, especially when user data is sent to a third party. To overcome these challenges, we propose Refine-Control, a semi-supervised distillation framework. Specifically, we improve the performance of the student model by introducing a tri-level knowledge fusion loss to transfer different levels of knowledge. To enhance generalization and alleviate dataset scarcity, we introduce a semi-supervised distillation method utilizing both labeled and unlabeled data. Our experiments reveal that Refine-Control achieves significant reductions in computational cost and latency, while maintaining high-fidelity generation capabilities and controllability, as quantified by comparative metrics. 

**Abstract (ZH)**: 基于文本控制的条件图像生成模型已在生成定制化图像方面取得了显著成果。然而，这些模型对资源的高需求以及标注数据的稀缺性限制了其在边缘设备上的部署，导致巨大的成本和隐私问题，尤其是在用户数据被发送到第三方时。为克服这些挑战，我们提出了一种半监督蒸馏框架Refine-Control。具体来说，我们通过引入多层次知识融合损失来提高学生模型的性能，以转移不同层次的知识。为增强泛化能力和缓解数据集稀缺性，我们引入了一种利用标记和未标记数据的半监督蒸馏方法。实验结果显示，Refine-Control在降低计算成本和延迟方面取得了显著成效，同时保持了高保真生成能力和可控性，如通过比较性指标所证明的那样。 

---
# Latent Diffusion : Multi-Dimension Stable Diffusion Latent Space Explorer 

**Title (ZH)**: 潜在扩散：多维稳定扩散潜在空间探索者 

**Authors**: Zhihua Zhong, Xuanyang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22038)  

**Abstract**: Latent space is one of the key concepts in generative AI, offering powerful means for creative exploration through vector manipulation. However, diffusion models like Stable Diffusion lack the intuitive latent vector control found in GANs, limiting their flexibility for artistic expression. This paper introduces \workname, a framework for integrating customizable latent space operations into the diffusion process. By enabling direct manipulation of conceptual and spatial representations, this approach expands creative possibilities in generative art. We demonstrate the potential of this framework through two artworks, \textit{Infinitepedia} and \textit{Latent Motion}, highlighting its use in conceptual blending and dynamic motion generation. Our findings reveal latent space structures with semantic and meaningless regions, offering insights into the geometry of diffusion models and paving the way for further explorations of latent space. 

**Abstract (ZH)**: 一种将可定制的潜在空间操作集成到扩散过程中的框架：探索概念与空间表示的直接操纵在生成艺术中的应用 

---
# No-Reference Image Contrast Assessment with Customized EfficientNet-B0 

**Title (ZH)**: 基于定制化EfficientNet-B0的无参考图像对比度评估 

**Authors**: Javad Hassannataj Joloudari, Bita Mesbahzadeh, Omid Zare, Emrah Arslan, Roohallah Alizadehsani, Hossein Moosaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.21967)  

**Abstract**: Image contrast was a fundamental factor in visual perception and played a vital role in overall image quality. However, most no reference image quality assessment NR IQA models struggled to accurately evaluate contrast distortions under diverse real world conditions. In this study, we proposed a deep learning based framework for blind contrast quality assessment by customizing and fine-tuning three pre trained architectures, EfficientNet B0, ResNet18, and MobileNetV2, for perceptual Mean Opinion Score, along with an additional model built on a Siamese network, which indicated a limited ability to capture perceptual contrast distortions. Each model is modified with a contrast-aware regression head and trained end to end using targeted data augmentations on two benchmark datasets, CID2013 and CCID2014, containing synthetic and authentic contrast distortions. Performance is evaluated using Pearson Linear Correlation Coefficient and Spearman Rank Order Correlation Coefficient, which assess the alignment between predicted and human rated scores. Among these three models, our customized EfficientNet B0 model achieved state-of-the-art performance with PLCC = 0.9286 and SRCC = 0.9178 on CCID2014 and PLCC = 0.9581 and SRCC = 0.9369 on CID2013, surpassing traditional methods and outperforming other deep baselines. These results highlighted the models robustness and effectiveness in capturing perceptual contrast distortion. Overall, the proposed method demonstrated that contrast aware adaptation of lightweight pre trained networks can yield a high performing, scalable solution for no reference contrast quality assessment suitable for real time and resource constrained applications. 

**Abstract (ZH)**: 基于深度学习的盲对比度质量评估框架：定制与微调 EfficientNet B0、ResNet18 和 MobileNetV2 及 Siamese 网络在视觉对比度感知评分中的应用 

---
# SemanticControl: A Training-Free Approach for Handling Loosely Aligned Visual Conditions in ControlNet 

**Title (ZH)**: 语义控制：一种无需训练的方法，用于处理ControlNet中的松散对齐的视觉条件 

**Authors**: Woosung Joung, Daewon Chae, Jinkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21938)  

**Abstract**: ControlNet has enabled detailed spatial control in text-to-image diffusion models by incorporating additional visual conditions such as depth or edge maps. However, its effectiveness heavily depends on the availability of visual conditions that are precisely aligned with the generation goal specified by text prompt-a requirement that often fails in practice, especially for uncommon or imaginative scenes. For example, generating an image of a cat cooking in a specific pose may be infeasible due to the lack of suitable visual conditions. In contrast, structurally similar cues can often be found in more common settings-for instance, poses of humans cooking are widely available and can serve as rough visual guides. Unfortunately, existing ControlNet models struggle to use such loosely aligned visual conditions, often resulting in low text fidelity or visual artifacts. To address this limitation, we propose SemanticControl, a training-free method for effectively leveraging misaligned but semantically relevant visual conditions. Our approach adaptively suppresses the influence of the visual condition where it conflicts with the prompt, while strengthening guidance from the text. The key idea is to first run an auxiliary denoising process using a surrogate prompt aligned with the visual condition (e.g., "a human playing guitar" for a human pose condition) to extract informative attention masks, and then utilize these masks during the denoising of the actual target prompt (e.g., cat playing guitar). Experimental results demonstrate that our method improves performance under loosely aligned conditions across various conditions, including depth maps, edge maps, and human skeletons, outperforming existing baselines. Our code is available at this https URL. 

**Abstract (ZH)**: SemanticControl：一种无需训练的利用语义相关但不精确对齐的视觉条件的方法 

---
# EqDiff-CT: Equivariant Conditional Diffusion model for CT Image Synthesis from CBCT 

**Title (ZH)**: EqDiff-CT: 具有不变性的条件扩散模型在CBCT图像合成中的应用 

**Authors**: Alzahra Altalib, Chunhui Li, Alessandro Perelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.21913)  

**Abstract**: Cone-beam computed tomography (CBCT) is widely used for image-guided radiotherapy (IGRT). It provides real time visualization at low cost and dose. However, photon scattering and beam hindrance cause artifacts in CBCT. These include inaccurate Hounsfield Units (HU), reducing reliability for dose calculation, and adaptive planning. By contrast, computed tomography (CT) offers better image quality and accurate HU calibration but is usually acquired offline and fails to capture intra-treatment anatomical changes. Thus, accurate CBCT-to-CT synthesis is needed to close the imaging-quality gap in adaptive radiotherapy workflows.
To cater to this, we propose a novel diffusion-based conditional generative model, coined EqDiff-CT, to synthesize high-quality CT images from CBCT. EqDiff-CT employs a denoising diffusion probabilistic model (DDPM) to iteratively inject noise and learn latent representations that enable reconstruction of anatomically consistent CT images. A group-equivariant conditional U-Net backbone, implemented with e2cnn steerable layers, enforces rotational equivariance (cyclic C4 symmetry), helping preserve fine structural details while minimizing noise and artifacts.
The system was trained and validated on the SynthRAD2025 dataset, comprising CBCT-CT scans across multiple head-and-neck anatomical sites, and we compared it with advanced methods such as CycleGAN and DDPM. EqDiff-CT provided substantial gains in structural fidelity, HU accuracy and quantitative metrics. Visual findings further confirm the improved recovery, sharper soft tissue boundaries, and realistic bone reconstructions. The findings suggest that the diffusion model has offered a robust and generalizable framework for CBCT improvements. The proposed solution helps in improving the image quality as well as the clinical confidence in the CBCT-guided treatment planning and dose calculations. 

**Abstract (ZH)**: 基于扩散的条件生成模型 EqDiff-CT 从 CBCT 合成高质 CT 图像 

---
# Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization 

**Title (ZH)**: 解锁美的本质：基于相对-绝对策略优化的高级美学推理 

**Authors**: Boyang Liu, Yifan Hu, Senjie Jin, Shihan Dou, Gonglei Shi, Jie Shao, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21871)  

**Abstract**: Multimodal large language models (MLLMs) are well suited to image aesthetic assessment, as they can capture high-level aesthetic features leveraging their cross-modal understanding capacity. However, the scarcity of multimodal aesthetic reasoning data and the inherently subjective nature of aesthetic judgment make it difficult for MLLMs to generate accurate aesthetic judgments with interpretable rationales. To this end, we propose Aes-R1, a comprehensive aesthetic reasoning framework with reinforcement learning (RL). Concretely, Aes-R1 integrates a pipeline, AesCoT, to construct and filter high-quality chain-of-thought aesthetic reasoning data used for cold-start. After teaching the model to generate structured explanations prior to scoring, we then employ the Relative-Absolute Policy Optimization (RAPO), a novel RL algorithm that jointly optimizes absolute score regression and relative ranking order, improving both per-image accuracy and cross-image preference judgments. Aes-R1 enables MLLMs to generate grounded explanations alongside faithful scores, thereby enhancing aesthetic scoring and reasoning in a unified framework. Extensive experiments demonstrate that Aes-R1 improves the backbone's average PLCC/SRCC by 47.9%/34.8%, surpassing state-of-the-art baselines of similar size. More ablation studies validate Aes-R1's robust generalization under limited supervision and in out-of-distribution scenarios. 

**Abstract (ZH)**: 多模态大型语言模型(MLLMs)在图像美学评估中表现出色，因为它们能够通过跨模态理解能力捕捉到高层次的美学特征。然而，多模态美学推理数据的稀缺性和美学判断的主观性使得MLLMs难以生成具有可解释理由的准确美学判断。为此，我们提出了一种基于强化学习(RL)的综合性美学推理框架Aes-R1。具体而言，Aes-R1集成了一个管道AesCoT，用于构建和筛选冷启动阶段所需的高度高质量的链式思维美学推理数据。在教学模型在评分前生成结构化解释之后，我们采用了联合优化绝对评分回归和相对排名顺序的新型RL算法Relative-Absolute Policy Optimization (RAPO)，从而提高单张图像准确性和跨图像偏好判断。Aes-R1使MLLMs能够在统一框架中生成基于事实的解释和忠实评分，从而提高美学评分和推理。广泛实验表明，Aes-R1将骨干模型的平均PLCC/SRCC提高了47.9%/34.8%，超过相似规模的先进基线。更多消融研究验证了Aes-R1在有限监督和分布外场景下的稳健泛化能力。 

---
# LFA-Net: A Lightweight Network with LiteFusion Attention for Retinal Vessel Segmentation 

**Title (ZH)**: LFA-Net：一种用于视网膜血管分割的轻量网络与LiteFusion注意力机制 

**Authors**: Mehwish Mehmood, Ivor Spence, Muhammad Fahim  

**Link**: [PDF](https://arxiv.org/pdf/2509.21738)  

**Abstract**: Lightweight retinal vessel segmentation is important for the early diagnosis of vision-threatening and systemic diseases, especially in a real-world clinical environment with limited computational resources. Although segmentation methods based on deep learning are improving, existing models are still facing challenges of small vessel segmentation and high computational costs. To address these challenges, we proposed a new vascular segmentation network, LFA-Net, which incorporates a newly designed attention module, LiteFusion-Attention. This attention module incorporates residual learning connections, Vision Mamba-inspired dynamics, and modulation-based attention, enabling the model to capture local and global context efficiently and in a lightweight manner. LFA-Net offers high performance with 0.11 million parameters, 0.42 MB memory size, and 4.46 GFLOPs, which make it ideal for resource-constrained environments. We validated our proposed model on DRIVE, STARE, and CHASE_DB with outstanding performance in terms of dice scores of 83.28, 87.44, and 84.50% and Jaccard indices of 72.85, 79.31, and 74.70%, respectively. The code of LFA-Net is available online this https URL. 

**Abstract (ZH)**: 轻量级视网膜血管分割对于早期诊断致盲性和全身性疾病至关重要，尤其是在资源受限的临床环境中。虽然基于深度学习的分割方法正在改进，但现有模型仍在小血管分割和高计算成本方面面临挑战。为应对这些挑战，我们提出了一种新的血管分割网络LFA-Net，该网络结合了一种新设计的注意力模块LiteFusion-Attention。该注意力模块采用了残差学习连接、Vision Mamba启发的动力学以及基于调制的注意力机制，使模型能够高效且轻量地捕捉局部和全局上下文。LFA-Net具有11万参数、42 KB内存大小和4.46 GFLOPs，使其适合资源受限的环境。我们在DRIVE、STARE和CHASE_DB上验证了我们提出的模型，性能指标分别为Dice分数83.28%、87.44%和84.50%及Jaccard指数72.85%、79.31%和74.70%。LFA-Net的代码已在线发布于此[https://]。 

---
# MORPH: Shape-agnostic PDE Foundation Models 

**Title (ZH)**: MORPH: 形状无关的偏微分方程基础模型 

**Authors**: Mahindra Singh Rautela, Alexander Most, Siddharth Mansingh, Bradley C. Love, Ayan Biswas, Diane Oyen, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2509.21670)  

**Abstract**: We introduce MORPH, a shape-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data dimensionality (1D--3D) at different resolutions, multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorizes full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters (LoRA), MORPH outperforms models trained from scratch in both zero-shot and full-shot generalization. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning. 

**Abstract (ZH)**: MORPH：一种通用自回归基础模型，用于偏微分方程 

---
# A Data-driven Typology of Vision Models from Integrated Representational Metrics 

**Title (ZH)**: 基于综合表征度量的数据驱动视觉模型类型划分 

**Authors**: Jialin Wu, Shreya Saha, Yiqing Bo, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.21628)  

**Abstract**: Large vision models differ widely in architecture and training paradigm, yet we lack principled methods to determine which aspects of their representations are shared across families and which reflect distinctive computational strategies. We leverage a suite of representational similarity metrics, each capturing a different facet-geometry, unit tuning, or linear decodability-and assess family separability using multiple complementary measures. Metrics preserving geometry or tuning (e.g., RSA, Soft Matching) yield strong family discrimination, whereas flexible mappings such as Linear Predictivity show weaker separation. These findings indicate that geometry and tuning carry family-specific signatures, while linearly decodable information is more broadly shared. To integrate these complementary facets, we adapt Similarity Network Fusion (SNF), a method inspired by multi-omics integration. SNF achieves substantially sharper family separation than any individual metric and produces robust composite signatures. Clustering of the fused similarity matrix recovers both expected and surprising patterns: supervised ResNets and ViTs form distinct clusters, yet all self-supervised models group together across architectural boundaries. Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders, suggesting convergence between architectural modernization and reconstruction-based training. This biology-inspired framework provides a principled typology of vision models, showing that emergent computational strategies-shaped jointly by architecture and training objective-define representational structure beyond surface design categories. 

**Abstract (ZH)**: 大型视觉模型在架构和训练范式上存在显著差异，但我们缺乏明确的方法来确定它们表示中哪些方面是跨家族共享的，哪些反映了独特的计算策略。我们利用一系列表示相似性度量，每种度量捕捉不同的方面-几何结构、单个单元的调谐或线性可解码性-并通过多种互补的度量评估家族可分性。保留几何结构或调谐的度量（例如，RSA、Soft Matching）能够产生强烈的家庭区分效果，而灵活的映射如线性预测性则显示出较弱的分离。这些发现表明，几何结构和调谐携带家族特有的签名，而线性可解码的信息则更广泛共享。为了整合这些互补的方面，我们改进了启发于多组学整合的相似性网络融合（SNF）方法。SNF在家庭分离的锐度上显著优于任何单一的度量，并生成稳健的综合签名。融合相似性矩阵的聚类既恢复了预期模式，也发现了一些意想不到的模式：监督ResNets和ViTs形成不同的簇，但所有自监督模型跨越架构边界聚集在一起。混合架构（ConvNeXt、Swin）与掩码自编码器聚集在一起，表明架构现代化与基于重建的训练之间存在趋同。这种生物学启发的框架提供了一种系统的视觉模型分类，表明由架构和训练目标共同塑造的新兴计算策略决定表示结构，超越了表面设计分类。 

---
# Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis 

**Title (ZH)**: 时间维度 vs. 空间维度：DINOv3和V-JEPA2特征表示在视频动作分析中的比较 

**Authors**: Sai Varun Kodathala, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2509.21595)  

**Abstract**: This study presents a comprehensive comparative analysis of two prominent self-supervised learning architectures for video action recognition: DINOv3, which processes frames independently through spatial feature extraction, and V-JEPA2, which employs joint temporal modeling across video sequences. We evaluate both approaches on the UCF Sports dataset, examining feature quality through multiple dimensions including classification accuracy, clustering performance, intra-class consistency, and inter-class discrimination. Our analysis reveals fundamental architectural trade-offs: DINOv3 achieves superior clustering performance (Silhouette score: 0.31 vs 0.21) and demonstrates exceptional discrimination capability (6.16x separation ratio) particularly for pose-identifiable actions, while V-JEPA2 exhibits consistent reliability across all action types with significantly lower performance variance (0.094 vs 0.288). Through action-specific evaluation, we identify that DINOv3's spatial processing architecture excels at static pose recognition but shows degraded performance on motion-dependent actions, whereas V-JEPA2's temporal modeling provides balanced representation quality across diverse action categories. These findings contribute to the understanding of architectural design choices in video analysis systems and provide empirical guidance for selecting appropriate feature extraction methods based on task requirements and reliability constraints. 

**Abstract (ZH)**: 本研究对两种 prominant 自监督学习架构在视频动作识别中的表现进行了全面比较分析：DINOv3 通过空间特征提取独立处理每一帧，而 V-JEPA2 则在视频序列中采用联合时间建模。我们使用 UCF Sports 数据集评估这两种方法，在分类准确性、聚类表现、类别内一致性以及类别间区分能力等多个维度上分析特征质量。我们的分析揭示了架构上的根本权衡：DINOv3 在聚类性能方面表现更优（轮廓得分：0.31 对比 0.21），特别是在姿态可辨识的动作中展现出卓越的区分能力（6.16 倍分离比），而 V-JEPA2 在所有动作类型中表现出一致的可靠性，并且具有显著更低的性能变异（0.094 对比 0.288）。通过动作特异性评估，我们发现 DINOv3 的空间处理架构在静态姿态识别方面表现出色，但在依赖运动的动作上表现较差，而 V-JEPA2 的时间建模能够在多样化的动作类别中提供均衡的表示质量。这些发现有助于理解视频分析系统中架构设计选择，并为基于任务需求和可靠性约束选择合适的特征提取方法提供实证指导。 

---
# What Happens Next? Anticipating Future Motion by Generating Point Trajectories 

**Title (ZH)**: 接下来会发生什么？通过生成点轨迹来预测未来运动 

**Authors**: Gabrijel Boduljak, Laurynas Karazija, Iro Laina, Christian Rupprecht, Andrea Vedaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21592)  

**Abstract**: We consider the problem of forecasting motion from a single image, i.e., predicting how objects in the world are likely to move, without the ability to observe other parameters such as the object velocities or the forces applied to them. We formulate this task as conditional generation of dense trajectory grids with a model that closely follows the architecture of modern video generators but outputs motion trajectories instead of pixels. This approach captures scene-wide dynamics and uncertainty, yielding more accurate and diverse predictions than prior regressors and generators. We extensively evaluate our method on simulated data, demonstrate its effectiveness on downstream applications such as robotics, and show promising accuracy on real-world intuitive physics datasets. Although recent state-of-the-art video generators are often regarded as world models, we show that they struggle with forecasting motion from a single image, even in simple physical scenarios such as falling blocks or mechanical object interactions, despite fine-tuning on such data. We show that this limitation arises from the overhead of generating pixels rather than directly modeling motion. 

**Abstract (ZH)**: 从单张图像预测运动：一种基于条件生成密集轨迹网格的方法 

---
# Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms 

**Title (ZH)**: 通过在半变异图中发现困难负样本以增强对比学习在地学定位中的效果 

**Authors**: Boyi Chen, Zhangyu Wang, Fabian Deuser, Johann Maximilian Zollner, Martin Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.21573)  

**Abstract**: Accurate and robust image-based geo-localization at a global scale is challenging due to diverse environments, visually ambiguous scenes, and the lack of distinctive landmarks in many regions. While contrastive learning methods show promising performance by aligning features between street-view images and corresponding locations, they neglect the underlying spatial dependency in the geographic space. As a result, they fail to address the issue of false negatives -- image pairs that are both visually and geographically similar but labeled as negatives, and struggle to effectively distinguish hard negatives, which are visually similar but geographically distant. To address this issue, we propose a novel spatially regularized contrastive learning strategy that integrates a semivariogram, which is a geostatistical tool for modeling how spatial correlation changes with distance. We fit the semivariogram by relating the distance of images in feature space to their geographical distance, capturing the expected visual content in a spatial correlation. With the fitted semivariogram, we define the expected visual dissimilarity at a given spatial distance as reference to identify hard negatives and false negatives. We integrate this strategy into GeoCLIP and evaluate it on the OSV5M dataset, demonstrating that explicitly modeling spatial priors improves image-based geo-localization performance, particularly at finer granularity. 

**Abstract (ZH)**: 在全球范围内实现具有多样环境、视觉歧义场景和缺乏明显地标区域的准确且 robust 的基于图像的地理解析具有挑战性。尽管对比学习方法通过在街景图像和对应位置之间对齐特征显示出有前途的性能，但它们忽略了地理空间中的潜在空间依赖性。因此，它们无法解决视觉和地理上相似但被标记为负例的假阴性问题，并且难以有效区分视觉上相似但地理上相距较远的负例。为解决这一问题，我们提出了一种新颖的空间正则化对比学习策略，该策略结合了半变异函数，这是一种用于建模空间相关性随距离变化的地理统计工具。我们通过将特征空间中图像的距离与地理距离联系起来拟合半变异函数，捕捉空间相关性中预期的视觉内容。使用拟合好的半变异函数，我们定义给定空间距离下的预期视觉差异作为参考，以识别难以区分的负例和假阴性。我们将该策略集成到GeoCLIP中，并在OSV5M数据集上进行评估，证明明确建模空间先验有助于提高基于图像的地理解析性能，尤其是在精细粒度上。 

---
# DistillKac: Few-Step Image Generation via Damped Wave Equations 

**Title (ZH)**: DistillKac: 通过阻尼波方程实现多步图像生成 

**Authors**: Weiqiao Han, Chenlin Meng, Christopher D. Manning, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.21513)  

**Abstract**: We present DistillKac, a fast image generator that uses the damped wave equation and its stochastic Kac representation to move probability mass at finite speed. In contrast to diffusion models whose reverse time velocities can become stiff and implicitly allow unbounded propagation speed, Kac dynamics enforce finite speed transport and yield globally bounded kinetic energy. Building on this structure, we introduce classifier-free guidance in velocity space that preserves square integrability under mild conditions. We then propose endpoint only distillation that trains a student to match a frozen teacher over long intervals. We prove a stability result that promotes supervision at the endpoints to closeness along the entire path. Experiments demonstrate DistillKac delivers high quality samples with very few function evaluations while retaining the numerical stability benefits of finite speed probability flows. 

**Abstract (ZH)**: DistillKac：一种快速图像生成器及其在有限速度概率流中的应用 

---
# Score-based Idempotent Distillation of Diffusion Models 

**Title (ZH)**: 基于评分的幂等蒸馏扩散模型 

**Authors**: Shehtab Zaman, Chengyan Liu, Kenneth Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21470)  

**Abstract**: Idempotent generative networks (IGNs) are a new line of generative models based on idempotent mapping to a target manifold. IGNs support both single-and multi-step generation, allowing for a flexible trade-off between computational cost and sample quality. But similar to Generative Adversarial Networks (GANs), conventional IGNs require adversarial training and are prone to training instabilities and mode collapse. Diffusion and score-based models are popular approaches to generative modeling that iteratively transport samples from one distribution, usually a Gaussian, to a target data distribution. These models have gained popularity due to their stable training dynamics and high-fidelity generation quality. However, this stability and quality come at the cost of high computational cost, as the data must be transported incrementally along the entire trajectory. New sampling methods, model distillation, and consistency models have been developed to reduce the sampling cost and even perform one-shot sampling from diffusion models. In this work, we unite diffusion and IGNs by distilling idempotent models from diffusion model scores, called SIGN. Our proposed method is highly stable and does not require adversarial losses. We provide a theoretical analysis of our proposed score-based training methods and empirically show that IGNs can be effectively distilled from a pre-trained diffusion model, enabling faster inference than iterative score-based models. SIGNs can perform multi-step sampling, allowing users to trade off quality for efficiency. These models operate directly on the source domain; they can project corrupted or alternate distributions back onto the target manifold, enabling zero-shot editing of inputs. We validate our models on multiple image datasets, achieving state-of-the-art results for idempotent models on the CIFAR and CelebA datasets. 

**Abstract (ZH)**: 同态生成网络（IGNs）是一类基于同态映射到目标流形的生成模型。IGNs 支持单步和多步生成，允许在计算成本和样本质量之间灵活权衡。但与生成对抗网络（GANs）类似，传统的IGNs 需要对抗训练，并且容易出现训练不稳定性和模式枯竭问题。扩散模型和基于分数的模型是生成建模的流行方法，通过迭代将样本从一个分布，通常是高斯分布，转移到目标数据分布。这些模型由于其稳定的训练动态和高保真生成质量而受到欢迎。然而，这种稳定性和高质量是以高计算成本为代价的，因为数据必须在整个轨迹中逐步传输。为了降低采样成本，甚至可以从扩散模型实现一次采样，开发了新的采样方法、模型蒸馏和一致性模型。在本文中，我们通过从扩散模型分数蒸馏同态模型，提出了一种称为SIGN的方法。我们提出的方法非常稳定，并不需要对抗损失。我们提供了我们提出的基于分数的训练方法的理论分析，并通过实验证明预先训练的扩散模型可以有效地蒸馏出IGNs，从而使得IGNs在比迭代基于分数的模型更快的推理中表现出色。SIGNs 支持多步采样，允许用户在质量和效率之间进行权衡。这些模型可以直接在源域上操作；它们可以将受损或替代分布投影回目标流形，从而实现零样本输入编辑。我们在多个图像数据集上验证了我们的模型，实现了CIFAR和CelebA数据集上同态模型的最佳结果。 

---
# Large AI Model-Enabled Generative Semantic Communications for Image Transmission 

**Title (ZH)**: 基于大型AI模型的生成语义通信在图像传输中的应用 

**Authors**: Qiyu Ma, Wanli Ni, Zhijin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.21394)  

**Abstract**: The rapid development of generative artificial intelligence (AI) has introduced significant opportunities for enhancing the efficiency and accuracy of image transmission within semantic communication systems. Despite these advancements, existing methodologies often neglect the difference in importance of different regions of the image, potentially compromising the reconstruction quality of visually critical content. To address this issue, we introduce an innovative generative semantic communication system that refines semantic granularity by segmenting images into key and non-key regions. Key regions, which contain essential visual information, are processed using an image oriented semantic encoder, while non-key regions are efficiently compressed through an image-to-text modeling approach. Additionally, to mitigate the substantial storage and computational demands posed by large AI models, the proposed system employs a lightweight deployment strategy incorporating model quantization and low-rank adaptation fine-tuning techniques, significantly boosting resource utilization without sacrificing performance. Simulation results demonstrate that the proposed system outperforms traditional methods in terms of both semantic fidelity and visual quality, thereby affirming its effectiveness for image transmission tasks. 

**Abstract (ZH)**: 生成式人工智能的迅猛发展为语义通信系统中的图像传输效率和精度提升了重要机会。尽管取得了这些进展，现有的方法往往忽视了图像不同区域重要性的差异，这可能会影响关键视觉内容的重建质量。为了解决这个问题，我们提出了一种创新的生成式语义通信系统，通过将图像分割为关键和非关键区域来细化语义粒度。关键区域包含重要的视觉信息，使用面向图像的语义编码器进行处理，而非关键区域则通过图像到文本建模方法进行高效压缩。此外，为了减轻大型人工智能模型带来的显著存储和计算需求，提出的系统采用了轻量级部署策略，结合模型量化和低秩适应微调技术，显著提升了资源利用率，同时不牺牲性能。仿真结果显示，与传统方法相比，所提出系统在语义保真度和视觉质量方面均表现出更优的效果，从而证实了其在图像传输任务中的有效性。 

---
# In silico Deep Learning Protocols for Label-Free Super-Resolution Microscopy: A Comparative Study of Network Architectures and SNR Dependence 

**Title (ZH)**: 基于计算的深度学习协议在无标记超分辨率显微镜中的比较研究：网络架构和信噪比依赖性分析 

**Authors**: Shiraz S Kaderuppan, Jonathan Mar, Andrew Irvine, Anurag Sharma, Muhammad Ramadan Saifuddin, Wai Leong Eugene Wong, Wai Lok Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21376)  

**Abstract**: The field of optical microscopy spans across numerous industries and research domains, ranging from education to healthcare, quality inspection and analysis. Nonetheless, a key limitation often cited by optical microscopists refers to the limit of its lateral resolution (typically defined as ~200nm), with potential circumventions involving either costly external modules (e.g. confocal scan heads, etc) and/or specialized techniques [e.g. super-resolution (SR) fluorescent microscopy]. Addressing these challenges in a normal (non-specialist) context thus remains an aspect outside the scope of most microscope users & facilities. This study thus seeks to evaluate an alternative & economical approach to achieving SR optical microscopy, involving non-fluorescent phase-modulated microscopical modalities such as Zernike phase contrast (PCM) and differential interference contrast (DIC) microscopy. Two in silico deep neural network (DNN) architectures which we developed previously (termed O-Net and Theta-Net) are assessed on their abilities to resolve a custom-fabricated test target containing nanoscale features calibrated via atomic force microscopy (AFM). The results of our study demonstrate that although both O-Net and Theta-Net seemingly performed well when super-resolving these images, they were complementary (rather than competing) approaches to be considered for image SR, particularly under different image signal-to-noise ratios (SNRs). High image SNRs favoured the application of O-Net models, while low SNRs inclined preferentially towards Theta-Net models. These findings demonstrate the importance of model architectures (in conjunction with the source image SNR) on model performance and the SR quality of the generated images where DNN models are utilized for non-fluorescent optical nanoscopy, even where the same training dataset & number of epochs are being used. 

**Abstract (ZH)**: 光学显微镜领域涵盖了教育、医疗、质量检验与分析等多个行业和研究领域。然而，光显微镜用户时常提及的一个关键限制是其横向分辨率的限制（通常定义为~200nm），这可能导致通过昂贵的外部模块（如共聚焦扫描头等）和/或专门的技术（如超分辨率荧光显微镜）来解决。在非专业人士的常规背景下，解决这些挑战仍然超出了大多数显微镜用户和设施的范围。本研究旨在评估一种替代的经济方法，以实现超分辨率光学显微镜，涉及非荧光相位调制显微成像模式，如Zernike相位对比（PCM）显微镜和相差干涉显微镜（DIC）。我们先前开发了两种深度神经网络（DNN）架构（分别称为O-Net和Theta-Net），并评估了它们在通过原子力显微镜（AFM）校准纳米尺度特征的自定义制造测试目标上的超分辨能力。研究结果表明，尽管O-Net和Theta-Net在这些图像的超分辨方面表现良好，但它们是互补（而非竞争）的超分辨方法，在不同图像信噪比（SNR）下应被考虑。高图像SNR有助于O-Net模型的应用，而低SNR则更偏好Theta-Net模型。这些发现强调了模型架构（与源图像SNR相结合）对模型性能和DNN模型在非荧光光学纳米显微镜中生成的超分辨率图像质量的重要性，即使使用相同的训练数据集和训练周期数也是如此。 

---
# MDF-MLLM: Deep Fusion Through Cross-Modal Feature Alignment for Contextually Aware Fundoscopic Image Classification 

**Title (ZH)**: MDF-MLLM：跨模态特征对齐的深度融合方法用于上下文感知眼底图像分类 

**Authors**: Jason Jordan, Mohammadreza Akbari Lor, Peter Koulen, Mei-Ling Shyu, Shu-Ching Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21358)  

**Abstract**: This study aimed to enhance disease classification accuracy from retinal fundus images by integrating fine-grained image features and global textual context using a novel multimodal deep learning architecture. Existing multimodal large language models (MLLMs) often struggle to capture low-level spatial details critical for diagnosing retinal diseases such as glaucoma, diabetic retinopathy, and retinitis pigmentosa. This model development and validation study was conducted on 1,305 fundus image-text pairs compiled from three public datasets (FIVES, HRF, and StoneRounds), covering acquired and inherited retinal diseases, and evaluated using classification accuracy and F1-score. The MDF-MLLM integrates skip features from four U-Net encoder layers into cross-attention blocks within a LLaMA 3.2 11B MLLM. Vision features are patch-wise projected and fused using scaled cross-attention and FiLM-based U-Net modulation. Baseline MLLM achieved 60% accuracy on the dual-type disease classification task. MDF-MLLM, with both U-Net and MLLM components fully fine-tuned during training, achieved a significantly higher accuracy of 94%, representing a 56% improvement. Recall and F1-scores improved by as much as 67% and 35% over baseline, respectively. Ablation studies confirmed that the multi-depth fusion approach contributed to substantial gains in spatial reasoning and classification, particularly for inherited diseases with rich clinical text. MDF-MLLM presents a generalizable, interpretable, and modular framework for fundus image classification, outperforming traditional MLLM baselines through multi-scale feature fusion. The architecture holds promise for real-world deployment in clinical decision support systems. Future work will explore synchronized training techniques, a larger pool of diseases for more generalizability, and extending the model for segmentation tasks. 

**Abstract (ZH)**: 本研究旨在通过结合纤细的图像特征和全局文本上下文，利用新颖的多模态深度学习架构提高从视网膜底片图像中进行疾病分类的准确性。现有的多模态大型语言模型（MLLM）在捕捉用于诊断青光眼、糖尿病视网膜病变和色素性视网膜炎等视网膜疾病的低级空间细节方面常常力不从心。该模型开发与验证研究在来自三个公开数据集（FIVES、HRF和StoneRounds）的1,305张视网膜底片图像-文本对上进行，涵盖获得性和遗传性视网膜疾病，并使用分类准确率和F1得分进行评估。MDF-MLLM 将四个U-Net编码器层的跳连特征整合到LLaMA 3.2 11B MLLM的交叉注意力模块中。视觉特征通过缩放交叉注意力和基于FiLM的U-Net调制逐块投影和融合。基于的MLLM在双类型疾病分类任务中实现了60%的准确率。MDF-MLLM，在训练过程中将U-Net和MLLM组件完全微调后，实现了显著更高的准确率94%，代表了56%的提升。召回率和F1得分分别提高了67%和35%。消融研究证实了多深度融合方法在空间推理和分类中的重要贡献，尤其是在临床文本丰富的遗传性疾病中。MDF-MLLM 提供了一种通用、可解释且模块化的视网膜底片图像分类框架，通过多尺度特征融合超越了传统的MLLM基线。该架构在临床决策支持系统中的实际部署中具有前景。未来的工作将探索同步训练技术、更广泛的疾病池以提高通用性，并扩展模型以进行分割任务。 

---
# Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports 

**Title (ZH)**: 基于短语的地基的胸部X光报告事实核查 

**Authors**: Razi Mahmood, Diego Machado-Reyes, Joy Wu, Parisa Kaviani, Ken C.L. Wong, Niharika D'Souza, Mannudeep Kalra, Ge Wang, Pingkun Yan, Tanveer Syeda-Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2509.21356)  

**Abstract**: With the emergence of large-scale vision language models (VLM), it is now possible to produce realistic-looking radiology reports for chest X-ray images. However, their clinical translation has been hampered by the factual errors and hallucinations in the produced descriptions during inference. In this paper, we present a novel phrase-grounded fact-checking model (FC model) that detects errors in findings and their indicated locations in automatically generated chest radiology reports.
Specifically, we simulate the errors in reports through a large synthetic dataset derived by perturbing findings and their locations in ground truth reports to form real and fake findings-location pairs with images. A new multi-label cross-modal contrastive regression network is then trained on this dataset. We present results demonstrating the robustness of our method in terms of accuracy of finding veracity prediction and localization on multiple X-ray datasets. We also show its effectiveness for error detection in reports of SOTA report generators on multiple datasets achieving a concordance correlation coefficient of 0.997 with ground truth-based verification, thus pointing to its utility during clinical inference in radiology workflows. 

**Abstract (ZH)**: 大规模视觉语言模型的出现使得生成胸部X光片的现实主义医疗报告成为可能。然而，在推断过程中生成的描述中事实错误和幻觉限制了其临床转化。本文提出了一种新颖的短语本地区检查模型（FC模型），用于检测自动生成的胸部放射学报告中发现结果及其指示位置的错误。具体来说，我们通过一个大规模的合成数据集来模拟报告中的错误，该数据集是通过对真实报告中的发现和位置进行扰动形成的，形成了包含图像的真实和虚假发现-位置配对。然后，在该数据集上训练了一个新的多标签跨模态对比回归网络。我们展示了该方法在多个X光数据集上的准确性预测和定位的稳健性结果。我们还展示了其在多个数据集上对SOTA报告生成器报告中的错误检测的有效性，通过基于真实性的验证达到了0.997的 Kendall一致性相关系数，从而表明其在放射学工作流程中的临床推断过程中的实用性。 

---
