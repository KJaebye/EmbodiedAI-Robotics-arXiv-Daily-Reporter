# IRAF-SLAM: An Illumination-Robust and Adaptive Feature-Culling Front-End for Visual SLAM in Challenging Environments 

**Title (ZH)**: IRAF-SLAM：具有光照鲁棒性和自适应特征精简前端的视觉SLAM技术 

**Authors**: Thanh Nguyen Canh, Bao Nguyen Quoc, Haolan Zhang, Bupesh Rethinam Veeraiah, Xiem HoangVan, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07752)  

**Abstract**: Robust Visual SLAM (vSLAM) is essential for autonomous systems operating in real-world environments, where challenges such as dynamic objects, low texture, and critically, varying illumination conditions often degrade performance. Existing feature-based SLAM systems rely on fixed front-end parameters, making them vulnerable to sudden lighting changes and unstable feature tracking. To address these challenges, we propose ``IRAF-SLAM'', an Illumination-Robust and Adaptive Feature-Culling front-end designed to enhance vSLAM resilience in complex and challenging environments. Our approach introduces: (1) an image enhancement scheme to preprocess and adjust image quality under varying lighting conditions; (2) an adaptive feature extraction mechanism that dynamically adjusts detection sensitivity based on image entropy, pixel intensity, and gradient analysis; and (3) a feature culling strategy that filters out unreliable feature points using density distribution analysis and a lighting impact factor. Comprehensive evaluations on the TUM-VI and European Robotics Challenge (EuRoC) datasets demonstrate that IRAF-SLAM significantly reduces tracking failures and achieves superior trajectory accuracy compared to state-of-the-art vSLAM methods under adverse illumination conditions. These results highlight the effectiveness of adaptive front-end strategies in improving vSLAM robustness without incurring significant computational overhead. The implementation of IRAF-SLAM is publicly available at https://thanhnguyencanh. this http URL. 

**Abstract (ZH)**: 鲁棒视觉SLAM（vSLAM）对于在实际环境中共自主系统至关重要，但动态物体、低纹理和关键的光照条件变化常常会降低其性能。现有的基于特征的SLAM系统依赖固定前端参数，使其对突发性光照变化和不稳定特征跟踪非常脆弱。为解决这些问题，我们提出了“IRAF-SLAM”，一种鲁棒性和自适应特征剔除的前端设计，旨在增强vSLAM在复杂和具有挑战性环境中的鲁棒性。我们的方法引入了：（1）一种图像增强方案，用于在不同光照条件下预处理和调整图像质量；（2）一种自适应特征提取机制，根据图像熵、像素强度和梯度分析动态调整检测灵敏度；以及（3）一种特征剔除策略，使用密度分布分析和光照影响因子筛选不可靠的特征点。在TUM-VI和欧洲机器人挑战赛（EuRoC）数据集上的综合评估表明，IRAF-SLAM在不良光照条件下显著减少了跟踪失败，并实现了优于现有最先进的vSLAM方法的轨迹准确性。这些结果突显了自适应前端策略在提高vSLAM鲁棒性方面的有效性，而不会造成显著的计算开销。IRAF-SLAM的实现已公开发布于<https://thanhnguyencanh. this http URL>。 

---
# Aerial Maritime Vessel Detection and Identification 

**Title (ZH)**: 空中海上舰船检测与识别 

**Authors**: Antonella Barisic Kulas, Frano Petric, Stjepan Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07153)  

**Abstract**: Autonomous maritime surveillance and target vessel identification in environments where Global Navigation Satellite Systems (GNSS) are not available is critical for a number of applications such as search and rescue and threat detection. When the target vessel is only described by visual cues and its last known position is not available, unmanned aerial vehicles (UAVs) must rely solely on on-board vision to scan a large search area under strict computational constraints. To address this challenge, we leverage the YOLOv8 object detection model to detect all vessels in the field of view. We then apply feature matching and hue histogram distance analysis to determine whether any detected vessel corresponds to the target. When found, we localize the target using simple geometric principles. We demonstrate the proposed method in real-world experiments during the MBZIRC2023 competition, integrated into a fully autonomous system with GNSS-denied navigation. We also evaluate the impact of perspective on detection accuracy and localization precision and compare it with the oracle approach. 

**Abstract (ZH)**: 在全球导航卫星系统不可用的环境下实现自主 maritime 监视和目标船只识别对于搜索救援和威胁检测等应用至关重要。当目标船只仅通过视觉特征描述且其最后已知位置不可用时，无人驾驶航空车辆（UAV）必须在严格的计算约束下依赖机载视觉扫描大面积搜索区域。为应对这一挑战，我们利用YOLOv8目标检测模型检测视野内的所有船只。然后，我们应用特征匹配和色调直方图距离分析来确定是否有任何检测到的船只对应于目标。一旦发现目标，我们使用简单的几何原理进行定位。我们在2023年 MBZIRC 竞赛的真实世界实验中演示了所提出的方法，并将其集成到具有全球导航卫星系统受限导航的完全自主系统中。我们还评估了视角对检测准确性和定位精度的影响，并将其与基准方法进行了比较。 

---
# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology 

**Title (ZH)**: 可追溯证据增强的视觉 grounded 理论推理：评估与方法 

**Authors**: Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07999)  

**Abstract**: Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at this https URL. 

**Abstract (ZH)**: 基于可追溯证据的视觉推理基准（TreeBench）：准确认知与解释性推理路径的训练范式（TreeVGR） 

---
# Single-pass Adaptive Image Tokenization for Minimum Program Search 

**Title (ZH)**: 单遍自适应图像令牌化以实现最小程序搜索 

**Authors**: Shivam Duggal, Sanghyun Byun, William T. Freeman, Antonio Torralba, Phillip Isola  

**Link**: [PDF](https://arxiv.org/pdf/2507.07995)  

**Abstract**: According to Algorithmic Information Theory (AIT) -- Intelligent representations compress data into the shortest possible program that can reconstruct its content, exhibiting low Kolmogorov Complexity (KC). In contrast, most visual representation learning systems use fixed-length representations for all inputs, ignoring variations in complexity or familiarity. Recent adaptive tokenization methods address this by allocating variable-length representations but typically require test-time search over multiple encodings to find the most predictive one. Inspired by Kolmogorov Complexity principles, we propose a single-pass adaptive tokenizer, KARL, which predicts the appropriate number of tokens for an image in a single forward pass, halting once its approximate KC is reached. The token count serves as a proxy for the minimum description length. KARL's training procedure closely resembles the Upside-Down Reinforcement Learning paradigm, as it learns to conditionally predict token halting based on a desired reconstruction quality. KARL matches the performance of recent adaptive tokenizers while operating in a single pass. We present scaling laws for KARL, analyzing the role of encoder/decoder size, continuous vs. discrete tokenization and more. Additionally, we offer a conceptual study drawing an analogy between Adaptive Image Tokenization and Algorithmic Information Theory, examining the predicted image complexity (KC) across axes such as structure vs. noise and in- vs. out-of-distribution familiarity -- revealing alignment with human intuition. 

**Abstract (ZH)**: 基于算法信息理论的单步自适应分词器KARL：预测图像的适当分词数量并通过最小描述长度进行优化 

---
# Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling 

**Title (ZH)**: 几何强制：将视频扩散与三维表示结合以实现一致的世界建模 

**Authors**: Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2507.07982)  

**Abstract**: Videos inherently represent 2D projections of a dynamic 3D world. However, our analysis suggests that video diffusion models trained solely on raw video data often fail to capture meaningful geometric-aware structure in their learned representations. To bridge this gap between video diffusion models and the underlying 3D nature of the physical world, we propose Geometry Forcing, a simple yet effective method that encourages video diffusion models to internalize latent 3D representations. Our key insight is to guide the model's intermediate representations toward geometry-aware structure by aligning them with features from a pretrained geometric foundation model. To this end, we introduce two complementary alignment objectives: Angular Alignment, which enforces directional consistency via cosine similarity, and Scale Alignment, which preserves scale-related information by regressing unnormalized geometric features from normalized diffusion representation. We evaluate Geometry Forcing on both camera view-conditioned and action-conditioned video generation tasks. Experimental results demonstrate that our method substantially improves visual quality and 3D consistency over the baseline methods. Project page: this https URL. 

**Abstract (ZH)**: 视频本质上代表动态三维世界的空间投影。然而，我们的分析表明，仅基于原始视频数据训练的视频扩散模型往往无法在其学习表示中捕捉到有意义的几何意识结构。为了弥合视频扩散模型与物理世界内在三维性质之间的差距，我们提出了一种简单而有效的方法——几何强制，该方法促使视频扩散模型内部化潜在的三维表示。我们的关键是通过将模型的中间表示与预训练的几何基础模型的特征对齐，引导其向几何意识结构靠拢。为此，我们引入了两种互补的对齐目标：角度对齐，通过余弦相似性确保方向一致性；尺度对齐，通过从归一化的扩散表示中回归未归一化的几何特征来保留与尺度相关的信息。我们在相机视角条件下的视频生成和动作条件下的视频生成任务上评估了几何强制方法。实验结果表明，与基线方法相比，我们的方法显著提高了视觉质量和三维一致性。项目页面：这个 https URL。 

---
# Scaling RL to Long Videos 

**Title (ZH)**: 将RL扩展到长视频 

**Authors**: Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu, Hongxu Yin, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.07966)  

**Abstract**: We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 52K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In experiments, LongVILA-R1-7B achieves strong performance on long video QA benchmarks such as VideoMME. It also outperforms Video-R1-7B and even matches Gemini-1.5-Pro across temporal reasoning, goal and purpose reasoning, spatial reasoning, and plot reasoning on our LongVideo-Reason-eval benchmark. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. LongVILA-R1 demonstrates consistent performance gains as the number of input video frames scales. LongVILA-R1 marks a firm step towards long video reasoning in VLMs. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens). 

**Abstract (ZH)**: 一种面向长视频的视觉-语言模型全流程框架：基于强化学习的长视频推理扩展（LongVILA-R1-7B: A Full-Stack Framework for Long Video Reasoning in Vision-Language Models via Reinforcement Learning） 

---
# Where are we with calibration under dataset shift in image classification? 

**Title (ZH)**: 图像分类中数据集迁移下的校准状态如何？ 

**Authors**: Mélanie Roschewitz, Raghav Mehta, Fabio de Sousa Ribeiro, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2507.07780)  

**Abstract**: We conduct an extensive study on the state of calibration under real-world dataset shift for image classification. Our work provides important insights on the choice of post-hoc and in-training calibration techniques, and yields practical guidelines for all practitioners interested in robust calibration under shift. We compare various post-hoc calibration methods, and their interactions with common in-training calibration strategies (e.g., label smoothing), across a wide range of natural shifts, on eight different classification tasks across several imaging domains. We find that: (i) simultaneously applying entropy regularisation and label smoothing yield the best calibrated raw probabilities under dataset shift, (ii) post-hoc calibrators exposed to a small amount of semantic out-of-distribution data (unrelated to the task) are most robust under shift, (iii) recent calibration methods specifically aimed at increasing calibration under shifts do not necessarily offer significant improvements over simpler post-hoc calibration methods, (iv) improving calibration under shifts often comes at the cost of worsening in-distribution calibration. Importantly, these findings hold for randomly initialised classifiers, as well as for those finetuned from foundation models, the latter being consistently better calibrated compared to models trained from scratch. Finally, we conduct an in-depth analysis of ensembling effects, finding that (i) applying calibration prior to ensembling (instead of after) is more effective for calibration under shifts, (ii) for ensembles, OOD exposure deteriorates the ID-shifted calibration trade-off, (iii) ensembling remains one of the most effective methods to improve calibration robustness and, combined with finetuning from foundation models, yields best calibration results overall. 

**Abstract (ZH)**: 我们在真实世界数据集变化下的图像分类校准状态进行了广泛研究。我们的工作提供了关于后验和训练中校准技术选择的重要见解，并为所有希望在变化条件下实现稳健校准的实践者提供了实用指南。我们比较了各种后验校准方法及其与常用训练中校准策略（如标签平滑）的交互作用，涵盖了广泛的自然变化，涉及八个不同分类任务的多个成像领域。我们发现：（i）同时应用熵正则化和标签平滑在数据集变化下能获得最好的校准原始概率；（ii）暴露于少量语义无关分布外数据（与任务无关）的后验校准器在变化下最稳健；（iii）专门针对提高变化下校准效果的最新校准方法不一定比简单的后验校准方法提供显著改进；（iv）提高变化下校准效果通常会以牺牲域内校准为代价。重要的是，这些发现不仅适用于随机初始化的分类器，也适用于从基础模型微调的分类器，后者相比从头训练的模型具有更稳健的校准效果。最后，我们深入分析了集成效应，发现：（i）在集成之前而非之后进行校准对变化下的校准更为有效；（ii）对于集成而言，分布外暴露会恶化ID-变化下的校准权衡；（iii）集成仍然是提高校准稳健性最有效的方法之一，并与来自基础模型的微调相结合，总体上能获得最佳校准结果。 

---
# Bayesian Discrete Diffusion Beats Autoregressive Perplexity 

**Title (ZH)**: 贝叶斯离散扩散优于自回归困惑度 

**Authors**: Cooper Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2507.07586)  

**Abstract**: We reveal a hidden Bayesian core of discrete-diffusion language models by showing that the expected denoiser output under the forward masking distribution recovers the exact posterior over clean tokens. Under minimal assumptions, Monte Carlo marginalization over K independent corruptions converges to this posterior at rate O(1/sqrt(K)), yielding a simple proof of consistency and finite-sample error bounds. Building on this insight, we introduce a lightweight inference-time ensemble that averages K mask-and-denoise passes to obtain posterior-aware token probabilities and uncertainty estimates at no extra training cost. On WikiText-2, our method achieves test perplexity 8.8 with K=8, versus 20.3 for GPT-2 Small, despite using a model of comparable size. Code is available at this https URL. 

**Abstract (ZH)**: 我们通过证明前向遮掩分布下预期去噪器输出恢复了干净词件的精确后验分布，揭示了离散扩散语言模型中的隐式贝叶斯核心。在最小的假设条件下，K个独立污染的蒙特卡洛边缘化收敛于此后验分布，以线性率O(1/sqrt(K))收敛，从而给出了一致性和有限样本误差界的一个简单证明。在此基础上，我们提出了一种轻量级的推理时集成方法，通过K次掩蔽与去噪处理的平均值获得后验感知词件概率和不确定性估计，而不增加额外的训练成本。在WikiText-2上，我们的方法使用K=8时的测试困惑度为8.8，而GPT-2 Small为20.3，尽管所用模型规模相当。代码可在以下网址获取。 

---
# NexViTAD: Few-shot Unsupervised Cross-Domain Defect Detection via Vision Foundation Models and Multi-Task Learning 

**Title (ZH)**: NexViTAD: 通过视觉基础模型和多任务学习的少样本无监督跨域缺陷检测 

**Authors**: Tianwei Mu, Feiyu Duan, Bo Zhou, Dan Xue, Manhong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07579)  

**Abstract**: This paper presents a novel few-shot cross-domain anomaly detection framework, Nexus Vision Transformer for Anomaly Detection (NexViTAD), based on vision foundation models, which effectively addresses domain-shift challenges in industrial anomaly detection through innovative shared subspace projection mechanisms and multi-task learning (MTL) module. The main innovations include: (1) a hierarchical adapter module that adaptively fuses complementary features from Hiera and DINO-v2 pre-trained models, constructing more robust feature representations; (2) a shared subspace projection strategy that enables effective cross-domain knowledge transfer through bottleneck dimension constraints and skip connection mechanisms; (3) a MTL Decoder architecture supports simultaneous processing of multiple source domains, significantly enhancing model generalization capabilities; (4) an anomaly score inference method based on Sinkhorn-K-means clustering, combined with Gaussian filtering and adaptive threshold processing for precise pixel level. Valuated on the MVTec AD dataset, NexViTAD delivers state-of-the-art performance with an AUC of 97.5%, AP of 70.4%, and PRO of 95.2% in the target domains, surpassing other recent models, marking a transformative advance in cross-domain defect detection. 

**Abstract (ZH)**: 基于视觉基础模型的新型少样本跨域异常检测框架：NexViTAD 

---
# Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision 

**Title (ZH)**: Objectomaly: 具有结构一致性和边界精度的物体意识分割方法异类物质区分 

**Authors**: Jeonghoon Song, Sunghun Kim, Jaegyun Im, Byeongjoon Noh  

**Link**: [PDF](https://arxiv.org/pdf/2507.07460)  

**Abstract**: Out-of-Distribution (OoD) segmentation is critical for safety-sensitive applications like autonomous driving. However, existing mask-based methods often suffer from boundary imprecision, inconsistent anomaly scores within objects, and false positives from background noise. We propose \textbf{\textit{Objectomaly}}, an objectness-aware refinement framework that incorporates object-level priors. Objectomaly consists of three stages: (1) Coarse Anomaly Scoring (CAS) using an existing OoD backbone, (2) Objectness-Aware Score Calibration (OASC) leveraging SAM-generated instance masks for object-level score normalization, and (3) Meticulous Boundary Precision (MBP) applying Laplacian filtering and Gaussian smoothing for contour refinement. Objectomaly achieves state-of-the-art performance on key OoD segmentation benchmarks, including SMIYC AnomalyTrack/ObstacleTrack and RoadAnomaly, improving both pixel-level (AuPRC up to 96.99, FPR$_{95}$ down to 0.07) and component-level (F1$-$score up to 83.44) metrics. Ablation studies and qualitative results on real-world driving videos further validate the robustness and generalizability of our method. Code will be released upon publication. 

**Abstract (ZH)**: Out-of-Distribution (OoD) 分段对于自动驾驶等安全敏感应用至关重要。现有的基于掩码的方法通常存在边界不清、对象内异常得分不一致以及背景噪声引起的假阳性等问题。我们提出了一种名为 \textbf{\textit{Objectomaly}} 的对象意识精炼框架，该框架结合了对象级别的先验知识。Objectomaly 包含三个阶段：（1）粗略异常评分（CAS）使用现有的 OoD 主干网络，（2）对象意识得分校准（OASC）利用 SAM 生成的实例掩码进行对象级别得分规范化，以及（3）细致边界精度（MBP）应用拉普拉斯滤波和高斯平滑进行轮廓精炼。Objectomaly 在关键的 OoD 分段基准数据集 SMIYC AnomalyTrack/ObstacleTrack 和 RoadAnomaly 上取得了最先进的性能，提升了像素级（AuPRC 最高 96.99，FPR$_{95}$ 最低至 0.07）和组件级（F1-score 最高 83.44）的指标。在真实驾驶视频上的消融研究和定性结果进一步验证了该方法的鲁棒性和泛化能力。代码将在发表后公开。 

---
# Bluish Veil Detection and Lesion Classification using Custom Deep Learnable Layers with Explainable Artificial Intelligence (XAI) 

**Title (ZH)**: 使用具有可解释人工智能(XAI)的自定义可学习层进行蓝膜检测与病变分类 

**Authors**: M. A. Rasel, Sameem Abdul Kareem, Zhenli Kwan, Shin Shen Yong, Unaizah Obaidellah  

**Link**: [PDF](https://arxiv.org/pdf/2507.07453)  

**Abstract**: Melanoma, one of the deadliest types of skin cancer, accounts for thousands of fatalities globally. The bluish, blue-whitish, or blue-white veil (BWV) is a critical feature for diagnosing melanoma, yet research into detecting BWV in dermatological images is limited. This study utilizes a non-annotated skin lesion dataset, which is converted into an annotated dataset using a proposed imaging algorithm based on color threshold techniques on lesion patches and color palettes. A Deep Convolutional Neural Network (DCNN) is designed and trained separately on three individual and combined dermoscopic datasets, using custom layers instead of standard activation function layers. The model is developed to categorize skin lesions based on the presence of BWV. The proposed DCNN demonstrates superior performance compared to conventional BWV detection models across different datasets. The model achieves a testing accuracy of 85.71% on the augmented PH2 dataset, 95.00% on the augmented ISIC archive dataset, 95.05% on the combined augmented (PH2+ISIC archive) dataset, and 90.00% on the Derm7pt dataset. An explainable artificial intelligence (XAI) algorithm is subsequently applied to interpret the DCNN's decision-making process regarding BWV detection. The proposed approach, coupled with XAI, significantly improves the detection of BWV in skin lesions, outperforming existing models and providing a robust tool for early melanoma diagnosis. 

**Abstract (ZH)**: 黑色素瘤，一种 deadliest 的皮肤癌类型，全球范围内导致数千人死亡。蓝白色或蓝白 veil（BWV）是诊断黑色素瘤的关键特征，但由于对在皮肤影像中检测 BWV 的研究有限，本研究利用了一个未标注的皮肤病变数据集，并通过基于颜色阈值技术提出的一种成像算法将其转换为标注数据集。设计并分别在三个个体和组合的皮肤镜影像数据集上训练了一个深度卷积神经网络（DCNN），使用自定义层而非标准激活函数层。该模型旨在根据 BWV 的存在对皮肤病变进行分类。提出的 DCNN 在不同数据集上的性能优于传统 BWV 检测模型。该模型在增强的 PH2 数据集上的测试准确率为 85.71%，在增强的 ISIC 存档数据集上的测试准确率为 95.00%，在联合增强（PH2+ISIC 存档）数据集上的测试准确率为 95.05%，在 Derm7pt 数据集上的测试准确率为 90.00%。随后应用了解释型人工智能（XAI）算法来解释 DCNN 在 BWV 检测中的决策过程。结合 XAI 的提出方法显著提高了在皮肤病变中检测 BWV 的性能，优于现有的模型，并提供了一个早期诊断黑色素瘤的稳健工具。 

---
# KeyRe-ID: Keypoint-Guided Person Re-Identification using Part-Aware Representation in Videos 

**Title (ZH)**: 基于关键点引导的分部位aware表示的人员重识别 

**Authors**: Jinseong Kim, Junghoon Song, Gyeongseon Baek, Byeongjoon Noh  

**Link**: [PDF](https://arxiv.org/pdf/2507.07393)  

**Abstract**: We propose \textbf{KeyRe-ID}, a keypoint-guided video-based person re-identification framework consisting of global and local branches that leverage human keypoints for enhanced spatiotemporal representation learning. The global branch captures holistic identity semantics through Transformer-based temporal aggregation, while the local branch dynamically segments body regions based on keypoints to generate fine-grained, part-aware features. Extensive experiments on MARS and iLIDS-VID benchmarks demonstrate state-of-the-art performance, achieving 91.73\% mAP and 97.32\% Rank-1 accuracy on MARS, and 96.00\% Rank-1 and 100.0\% Rank-5 accuracy on iLIDS-VID. The code for this work will be publicly available on GitHub upon publication. 

**Abstract (ZH)**: KeyRe-ID：一种基于关键点引导的视频人员再识别框架 

---
# SonicMotion: Dynamic Spatial Audio Soundscapes with Latent Diffusion Models 

**Title (ZH)**: SonicMotion：基于潜在扩散模型的动态空间音频音景 

**Authors**: Christian Templin, Yanda Zhu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07318)  

**Abstract**: Spatial audio is an integral part of immersive entertainment, such as VR/AR, and has seen increasing popularity in cinema and music as well. The most common format of spatial audio is described as first-order Ambisonics (FOA). We seek to extend recent advancements in FOA generative AI models to enable the generation of 3D scenes with dynamic sound sources. Our proposed end-to-end model, SonicMotion, comes in two variations which vary in their user input and level of precision in sound source localization. In addition to our model, we also present a new dataset of simulated spatial audio-caption pairs. Evaluation of our models demonstrate that they are capable of matching the semantic alignment and audio quality of state of the art models while capturing the desired spatial attributes. 

**Abstract (ZH)**: 空间音频是沉浸式娱乐，如VR/AR的一个重要组成部分，并且在电影和音乐领域也越来越受欢迎。空间音频最常见的格式是初级球面声学（FOA）。我们致力于将最新的生成型AI模型技术扩展到能够生成具有动态声源的三维场景。我们提出的端到端模型SonicMotion有两种变体，它们在用户输入和声源定位的精确度上有所不同。除了我们的模型，我们还提供了一组新的模拟空间音频-图例对数据集。我们的模型评估表明，它们能够匹配最先进的模型的语义对齐和音质，同时捕捉到所需的空间属性。 

---
# Generative Panoramic Image Stitching 

**Title (ZH)**: 生成全景图像拼接 

**Authors**: Mathieu Tuli, Kaveh Kamali, David B. Lindell  

**Link**: [PDF](https://arxiv.org/pdf/2507.07133)  

**Abstract**: We introduce the task of generative panoramic image stitching, which aims to synthesize seamless panoramas that are faithful to the content of multiple reference images containing parallax effects and strong variations in lighting, camera capture settings, or style. In this challenging setting, traditional image stitching pipelines fail, producing outputs with ghosting and other artifacts. While recent generative models are capable of outpainting content consistent with multiple reference images, they fail when tasked with synthesizing large, coherent regions of a panorama. To address these limitations, we propose a method that fine-tunes a diffusion-based inpainting model to preserve a scene's content and layout based on multiple reference images. Once fine-tuned, the model outpaints a full panorama from a single reference image, producing a seamless and visually coherent result that faithfully integrates content from all reference images. Our approach significantly outperforms baselines for this task in terms of image quality and the consistency of image structure and scene layout when evaluated on captured datasets. 

**Abstract (ZH)**: 生成全景图像缝合任务：基于多个参考图像合成忠实于内容且无拼接痕迹的无缝全景图像 

---
# DpDNet: An Dual-Prompt-Driven Network for Universal PET-CT Segmentation 

**Title (ZH)**: DpDNet：一种双提示驱动的通用PET-CT分割网络 

**Authors**: Xinglong Liang, Jiaju Huang, Luyi Han, Tianyu Zhang, Xin Wang, Yuan Gao, Chunyao Lu, Lishan Cai, Tao Tan, Ritse Mann  

**Link**: [PDF](https://arxiv.org/pdf/2507.07126)  

**Abstract**: PET-CT lesion segmentation is challenging due to noise sensitivity, small and variable lesion morphology, and interference from physiological high-metabolic signals. Current mainstream approaches follow the practice of one network solving the segmentation of multiple cancer lesions by treating all cancers as a single task. However, this overlooks the unique characteristics of different cancer types. Considering the specificity and similarity of different cancers in terms of metastatic patterns, organ preferences, and FDG uptake intensity, we propose DpDNet, a Dual-Prompt-Driven network that incorporates specific prompts to capture cancer-specific features and common prompts to retain shared knowledge. Additionally, to mitigate information forgetting caused by the early introduction of prompts, prompt-aware heads are employed after the decoder to adaptively handle multiple segmentation tasks. Experiments on a PET-CT dataset with four cancer types show that DpDNet outperforms state-of-the-art models. Finally, based on the segmentation results, we calculated MTV, TLG, and SUVmax for breast cancer survival analysis. The results suggest that DpDNet has the potential to serve as a valuable tool for personalized risk stratification, supporting clinicians in optimizing treatment strategies and improving outcomes. Code is available at this https URL. 

**Abstract (ZH)**: PET-CT病灶分割由于噪声敏感性、小且变化多端的病灶形态以及生理高代谢信号的干扰而具有挑战性。当前主流方法将所有癌症视为单一任务来解决多个癌症病灶的分割问题，但忽略了不同癌症类型的独特特征。考虑到不同癌症在转移模式、器官偏好和氟脱氧葡萄糖摄取强度上的特异性和相似性，我们提出DpDNet，这是一种双重提示驱动网络，结合了特定提示以捕捉癌症特异性特征，并结合了通用提示以保留共享知识。此外，为了减轻过早引入提示造成的知识遗忘，我们在解码器之后使用提示感知头部以适应性地处理多个分割任务。实验结果表明，DpDNet在包含四种癌症类型的PET-CT数据集上优于现有最先进的模型。最后，基于分割结果，我们计算了乳腺癌的MTV、TLG和SUVmax，用于生存分析。结果表明，DpDNet有潜力作为个性化风险分层的重要工具，支持临床医生优化治疗策略并改善预后。代码可在以下链接获取。 

---
