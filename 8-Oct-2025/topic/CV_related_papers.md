# Dropping the D: RGB-D SLAM Without the Depth Sensor 

**Title (ZH)**: RGB-D SLAM Without the Depth Sensor 

**Authors**: Mert Kiray, Alican Karaomer, Benjamin Busam  

**Link**: [PDF](https://arxiv.org/pdf/2510.06216)  

**Abstract**: We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors. The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network. Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features. These are processed by an unmodified RGB-D SLAM back end for tracking and mapping. On the TUM RGB-D benchmark, DropD-SLAM attains 7.4 cm mean ATE on static sequences and 1.8 cm on dynamic sequences, matching or surpassing state-of-the-art RGB-D methods while operating at 22 FPS on a single GPU. These results suggest that modern pretrained vision models can replace active depth sensors as reliable, real-time sources of metric scale, marking a step toward simpler and more cost-effective SLAM systems. 

**Abstract (ZH)**: DropD-SLAM：一种无需深度传感器即可实现RGB-D级准确性的实时单目SLAM系统 

---
# Smartphone-based iris recognition through high-quality visible-spectrum iris image capture.V2 

**Title (ZH)**: 基于智能手机的高质量可见光谱虹膜图像捕获虹膜识别V2 

**Authors**: Naveenkumar G Venkataswamy, Yu Liu, Soumyabrata Dey, Stephanie Schuckers, Masudul H Imtiaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.06170)  

**Abstract**: Smartphone-based iris recognition in the visible spectrum (VIS) remains difficult due to illumination variability, pigmentation differences, and the absence of standardized capture controls. This work presents a compact end-to-end pipeline that enforces ISO/IEC 29794-6 quality compliance at acquisition and demonstrates that accurate VIS iris recognition is feasible on commodity devices. Using a custom Android application performing real-time framing, sharpness evaluation, and feedback, we introduce the CUVIRIS dataset of 752 compliant images from 47 subjects. A lightweight MobileNetV3-based multi-task segmentation network (LightIrisNet) is developed for efficient on-device processing, and a transformer matcher (IrisFormer) is adapted to the VIS domain. Under a standardized protocol and comparative benchmarking against prior CNN baselines, OSIRIS attains a TAR of 97.9% at FAR=0.01 (EER=0.76%), while IrisFormer, trained only on UBIRIS.v2, achieves an EER of 0.057% on CUVIRIS. The acquisition app, trained models, and a public subset of the dataset are released to support reproducibility. These results confirm that standardized capture and VIS-adapted lightweight models enable accurate and practical iris recognition on smartphones. 

**Abstract (ZH)**: 基于可见光谱的智能手机虹膜识别仍因照明变化、色素差异和缺乏标准化捕获控制而具有挑战性。本工作提出了一种紧凑的端到端管道，确保捕获过程符合ISO/IEC 29794-6质量标准，并证明在消费级设备上实现准确的可见光谱虹膜识别是可行的。 

---
# Bimanual 3D Hand Motion and Articulation Forecasting in Everyday Images 

**Title (ZH)**: 双手三维手部运动与articulation预测在日常图像中 

**Authors**: Aditya Prakash, David Forsyth, Saurabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.06145)  

**Abstract**: We tackle the problem of forecasting bimanual 3D hand motion & articulation from a single image in everyday settings. To address the lack of 3D hand annotations in diverse settings, we design an annotation pipeline consisting of a diffusion model to lift 2D hand keypoint sequences to 4D hand motion. For the forecasting model, we adopt a diffusion loss to account for the multimodality in hand motion distribution. Extensive experiments across 6 datasets show the benefits of training on diverse data with imputed labels (14% improvement) and effectiveness of our lifting (42% better) & forecasting (16.4% gain) models, over the best baselines, especially in zero-shot generalization to everyday images. 

**Abstract (ZH)**: 我们解决了一种在日常场景中从单张图像预测双手三维手部运动与articulation的问题。为了解决各种场景下缺少3D手部注释的问题，我们设计了一种注释管道，其中包括一个扩散模型将2D手部关键点序列提升到4D手部运动。在预测模型中，我们采用了扩散损失来考虑手部运动分布的多模态性。在6个数据集上的广泛实验表明，使用具有填充标签的多样数据训练（性能提升14%）以及我们提出的提升（性能提升42%）和预测（性能提升16.4%）模型的有效性，尤其是在零样本泛化到日常图像方面的表现尤为突出。 

---
# When Thinking Drifts: Evidential Grounding for Robust Video Reasoning 

**Title (ZH)**: 思维漂移：稳健视频推理的证据接地 

**Authors**: Mi Luo, Zihui Xue, Alex Dimakis, Kristen Grauman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06077)  

**Abstract**: Video reasoning, the task of enabling machines to infer from dynamic visual content through multi-step logic, is crucial for advanced AI. While the Chain-of-Thought (CoT) mechanism has enhanced reasoning in text-based tasks, its application to video understanding remains underexplored. This paper presents a systematic analysis revealing that CoT often degrades performance in video reasoning, generating verbose but misleading internal monologues, and leading to hallucinated visual details and overridden correct intuitions - a phenomenon we term "visual thinking drift". We explain this drift through a Bayesian lens, positing that CoT traces often diverge from actual visual evidence, instead amplifying internal biases or language priors, causing models to storytell rather than engage in grounded reasoning. To counteract this, we introduce Visual Evidence Reward (VER), a novel reinforcement learning framework that explicitly rewards the generation of reasoning traces that are verifiably grounded in visual evidence. Comprehensive evaluation across 10 diverse video understanding benchmarks demonstrates that our Video-VER consistently achieves top performance. Our work sheds light on the distinct challenges of video-centric reasoning and encourages the development of AI that robustly grounds its inferences in visual evidence - for large multimodal models that not only "think before answering", but also "see while thinking". 

**Abstract (ZH)**: 基于视觉的联想推理：一种新的强化学习框架VER及其应用 

---
# GLVD: Guided Learned Vertex Descent 

**Title (ZH)**: GLVD: 引导式学习顶点下降 

**Authors**: Pol Caselles Rico, Francesc Moreno Noguer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06046)  

**Abstract**: Existing 3D face modeling methods usually depend on 3D Morphable Models, which inherently constrain the representation capacity to fixed shape priors. Optimization-based approaches offer high-quality reconstructions but tend to be computationally expensive. In this work, we introduce GLVD, a hybrid method for 3D face reconstruction from few-shot images that extends Learned Vertex Descent (LVD) by integrating per-vertex neural field optimization with global structural guidance from dynamically predicted 3D keypoints. By incorporating relative spatial encoding, GLVD iteratively refines mesh vertices without requiring dense 3D supervision. This enables expressive and adaptable geometry reconstruction while maintaining computational efficiency. GLVD achieves state-of-the-art performance in single-view settings and remains highly competitive in multi-view scenarios, all while substantially reducing inference time. 

**Abstract (ZH)**: 基于少量图像的3D人脸重建：GLVD方法及其应用 

---
# VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization 

**Title (ZH)**: VideoMiner: 通过树基于组相对策略优化迭代定位小时长视频的关键帧 

**Authors**: Xinye Cao, Hongcan Guo, Jiawen Qian, Guoshun Nan, Chao Wang, Yuqi Pan, Tianhao Hou, Xiaojuan Wang, Yutong Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06040)  

**Abstract**: Understanding hour-long videos with multi-modal large language models (MM-LLMs) enriches the landscape of human-centered AI applications. However, for end-to-end video understanding with LLMs, uniformly sampling video frames results in LLMs being overwhelmed by a vast amount of irrelevant information as video length increases. Existing hierarchical key frame extraction methods improve the accuracy of video understanding but still face two critical challenges. 1) How can the interference of extensive redundant information in long videos be mitigated? 2) How can a model dynamically adapt to complex hierarchical structures while accurately identifying key frames? To address these issues, we propose VideoMiner, which iteratively segments, captions, and clusters long videos, forming a hierarchical tree structure. The proposed VideoMiner progresses from long videos to events to frames while preserving temporal coherence, effectively addressing the first challenge. To precisely locate key frames, we introduce T-GRPO, a tree-based group relative policy optimization in reinforcement learning method that guides the exploration of the VideoMiner. The proposed T-GRPO is specifically designed for tree structures, integrating spatiotemporal information at the event level while being guided by the question, thus solving the second challenge. We achieve superior performance in all long-video understanding tasks and uncover several interesting insights. Our proposed T-GRPO surprisingly incentivizes the model to spontaneously generate a reasoning chain. Additionally, the designed tree growth auxin dynamically adjusts the expansion depth, obtaining accuracy and efficiency gains. The code is publicly available at this https URL. 

**Abstract (ZH)**: 利用多模态大规模语言模型（MM-LLMs）理解一小时长度的视频丰富了以人类为中心的AI应用图谱。然而，对于基于LLMs的端到端视频理解，均匀采样视频帧会导致随着视频长度增加，LLMs受到大量无关信息的困扰。现有分层关键帧提取方法提高了视频理解的准确性，但仍面临两个关键挑战：1）长视频中的大量冗余信息如何被减轻？2）模型如何动态适应复杂的分层结构并准确识别关键帧？为解决这些问题，我们提出了VideoMiner，该方法迭代地对齐、加字幕和聚类长视频，形成分层树结构。提出的VideoMiner从长视频到事件再到帧，同时保持时间连贯性，有效解决了第一个挑战。为了精确定位关键帧，我们引入了基于树结构的分组相对策略优化方法T-GRPO，指导VideoMiner的探索。提出的T-GRPO专门针对树结构，整合事件级别的时空信息，并由问题引导，从而解决了第二个挑战。我们在所有长视频理解任务中实现了优越的性能，并揭示了几项有趣的研究洞察。我们提出的T-GRPO意外地激励模型自发生成推理链。此外，所设计的树生长植物动态调整扩展深度，获得准确性和效率的提升。代码已公开，可通过以下链接获取。 

---
# Diffusion Models for Low-Light Image Enhancement: A Multi-Perspective Taxonomy and Performance Analysis 

**Title (ZH)**: 低光照图像增强的扩散模型：多视角分类与性能分析 

**Authors**: Eashan Adhikarla, Yixin Liu, Brian D. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2510.05976)  

**Abstract**: Low-light image enhancement (LLIE) is vital for safety-critical applications such as surveillance, autonomous navigation, and medical imaging, where visibility degradation can impair downstream task performance. Recently, diffusion models have emerged as a promising generative paradigm for LLIE due to their capacity to model complex image distributions via iterative denoising. This survey provides an up-to-date critical analysis of diffusion models for LLIE, distinctively featuring an in-depth comparative performance evaluation against Generative Adversarial Network and Transformer-based state-of-the-art methods, a thorough examination of practical deployment challenges, and a forward-looking perspective on the role of emerging paradigms like foundation models. We propose a multi-perspective taxonomy encompassing six categories: Intrinsic Decomposition, Spectral & Latent, Accelerated, Guided, Multimodal, and Autonomous; that map enhancement methods across physical priors, conditioning schemes, and computational efficiency. Our taxonomy is grounded in a hybrid view of both the model mechanism and the conditioning signals. We evaluate qualitative failure modes, benchmark inconsistencies, and trade-offs between interpretability, generalization, and inference efficiency. We also discuss real-world deployment constraints (e.g., memory, energy use) and ethical considerations. This survey aims to guide the next generation of diffusion-based LLIE research by highlighting trends and surfacing open research questions, including novel conditioning, real-time adaptation, and the potential of foundation models. 

**Abstract (ZH)**: 低光照图像增强中的扩散模型综述：与生成对抗网络和基于Transformer的先进方法的深入性能比较、实际部署挑战的全面 examination 以及新兴范式（如基础模型）的作用展望 

---
# $\bf{D^3}$QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection 

**Title (ZH)**: D³QE: 学习自回归生成图像检测中的离散分布差异感知量化误差 

**Authors**: Yanran Zhang, Bingyao Yu, Yu Zheng, Wenzhao Zheng, Yueqi Duan, Lei Chen, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05891)  

**Abstract**: The emergence of visual autoregressive (AR) models has revolutionized image generation while presenting new challenges for synthetic image detection. Unlike previous GAN or diffusion-based methods, AR models generate images through discrete token prediction, exhibiting both marked improvements in image synthesis quality and unique characteristics in their vector-quantized representations. In this paper, we propose to leverage Discrete Distribution Discrepancy-aware Quantization Error (D$^3$QE) for autoregressive-generated image detection that exploits the distinctive patterns and the frequency distribution bias of the codebook existing in real and fake images. We introduce a discrete distribution discrepancy-aware transformer that integrates dynamic codebook frequency statistics into its attention mechanism, fusing semantic features and quantization error latent. To evaluate our method, we construct a comprehensive dataset termed ARForensics covering 7 mainstream visual AR models. Experiments demonstrate superior detection accuracy and strong generalization of D$^3$QE across different AR models, with robustness to real-world perturbations. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 视觉自回归（AR）模型的出现革新了图像生成，但也带来了合成图像检测的新挑战。不同于以往的GAN或基于扩散的方法，AR模型通过离散令牌预测生成图像，不仅在图像合成质量上取得了显著提升，还在其向量量化表示中展现出独特的特性。本文提出利用离散分布差异感知量化误差（D$^3$QE）进行自回归生成图像检测，该方法利用真实和虚假图像中存在的代码书的独特模式和频率分布偏差。我们引入了一种离散分布差异感知变换器，将动态代码书频率统计集成到其注意机制中，融合了语义特征和量化误差潜在表示。为了评估该方法，我们构建了一个名为ARForensics的综合数据集，涵盖了7种主流视觉AR模型。实验结果证明，D$^3$QE在不同AR模型上具有优越的检测准确性和强健的泛化能力，能够抵抗现实世界干扰的鲁棒性。代码可在\href{this https URL}{这个链接}获取。 

---
# Deformable Image Registration for Self-supervised Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images 

**Title (ZH)**: 自监督心脏相位检测的可变形图像配准在多视角多疾病心脏磁共振图像中 

**Authors**: Sven Koehler, Sarah Kaye Mueller, Jonathan Kiekenap, Gerald Greil, Tarique Hussain, Samir Sarikouch, Florian André, Norbert Frey, Sandy Engelhardt  

**Link**: [PDF](https://arxiv.org/pdf/2510.05819)  

**Abstract**: Cardiovascular magnetic resonance (CMR) is the gold standard for assessing cardiac function, but individual cardiac cycles complicate automatic temporal comparison or sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion. We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber long-axis (4CH) cine CMR. Initially, dense deformable registration fields are derived from the images and used to compute a 1D motion descriptor, which provides valuable insights into global cardiac contraction and relaxation patterns. From these characteristic curves, keyframes are determined using a simple set of rules. The method was independently evaluated for both views using three public, multicentre, multidisease datasets. M&Ms-2 (n=360) dataset was used for training and evaluation, and M&Ms (n=345) and ACDC (n=100) datasets for repeatability control. Furthermore, generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) dataset. Our self-supervised approach achieved improved detection accuracy by 30% - 51% for SAX and 11% - 47% for 4CH in ED and ES, as measured by cyclic frame difference (cFD), compared with the volume-based approach. We can detect ED and ES, as well as three additional keyframes throughout the cardiac cycle with a mean cFD below 1.31 frames for SAX and 1.73 for LAX. Our approach enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths. GitHub repository: this https URL 

**Abstract (ZH)**: 基于自监督深度学习的心脏磁共振关键帧检测方法：短轴位和四 chamber长轴位心肌运动分析 

---
# Redefining Generalization in Visual Domains: A Two-Axis Framework for Fake Image Detection with FusionDetect 

**Title (ZH)**: 视觉领域中泛化能力的重定义：FusionDetect融合检测的双轴框架 

**Authors**: Amirtaha Amanzadi, Zahra Dehghanian, Hamid Beigy, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2510.05740)  

**Abstract**: The rapid development of generative models has made it increasingly crucial to develop detectors that can reliably detect synthetic images. Although most of the work has now focused on cross-generator generalization, we argue that this viewpoint is too limited. Detecting synthetic images involves another equally important challenge: generalization across visual domains. To bridge this gap,we present the OmniGen Benchmark. This comprehensive evaluation dataset incorporates 12 state-of-the-art generators, providing a more realistic way of evaluating detector performance under realistic conditions. In addition, we introduce a new method, FusionDetect, aimed at addressing both vectors of generalization. FusionDetect draws on the benefits of two frozen foundation models: CLIP & Dinov2. By deriving features from both complementary models,we develop a cohesive feature space that naturally adapts to changes in both thecontent and design of the generator. Our extensive experiments demonstrate that FusionDetect delivers not only a new state-of-the-art, which is 3.87% more accurate than its closest competitor and 6.13% more precise on average on established benchmarks, but also achieves a 4.48% increase in accuracy on OmniGen,along with exceptional robustness to common image perturbations. We introduce not only a top-performing detector, but also a new benchmark and framework for furthering universal AI image detection. The code and dataset are available at this http URL 

**Abstract (ZH)**: 生成模型的快速发展使得可靠检测合成图像的检测器变得日益重要。尽管目前大多数工作重心已经转向跨生成器泛化，我们认为这一视角过于局限。检测合成图像还涉及另一个同样重要的挑战：跨视觉域的泛化。为了弥补这一差距，我们提出了OmniGen基准。该全面评估数据集包含12个当前最先进的生成器，提供了一种在现实条件下更真实地评估检测器性能的方法。此外，我们还引入了一种新的方法FusionDetect，旨在解决泛化问题的两个方面。FusionDetect结合了两个冻结基础模型CLIP与Dinov2的优点，通过从两个互补模型中提取特征，我们建立了一个综合特征空间，自然适应生成器内容和设计的变化。我们的大量实验证明，FusionDetect不仅在现有的基准测试中达到了新的最佳性能，比最接近的竞争对手准确率高出3.87%，平均精确度高出6.13%，还在OmniGen基准测试中实现了4.48%的准确率提升，并且具有出色的抗常见图像扰动能力。我们不仅介绍了性能最优的检测器，还提供了一个新的基准和框架，以进一步推动通用AI图像检测的发展。代码和数据集可在以下网址获得。 

---
# Verifier-free Test-Time Sampling for Vision Language Action Models 

**Title (ZH)**: 无验证者的时间采样方法：面向视觉语言动作模型的测试时采样 

**Authors**: Suhyeok Jang, Dongyoung Kim, Changyeon Kim, Youngsuk Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.05681)  

**Abstract**: Vision-Language-Action models (VLAs) have demonstrated remarkable performance in robot control. However, they remain fundamentally limited in tasks that require high precision due to their single-inference paradigm. While test-time scaling approaches using external verifiers have shown promise, they require additional training and fail to generalize to unseen conditions. We propose Masking Distribution Guided Selection (MG-Select), a novel test-time scaling framework for VLAs that leverages the model's internal properties without requiring additional training or external modules. Our approach utilizes KL divergence from a reference action token distribution as a confidence metric for selecting the optimal action from multiple candidates. We introduce a reference distribution generated by the same VLA but with randomly masked states and language conditions as inputs, ensuring maximum uncertainty while remaining aligned with the target task distribution. Additionally, we propose a joint training strategy that enables the model to learn both conditional and unconditional distributions by applying dropout to state and language conditions, thereby further improving the quality of the reference distribution. Our experiments demonstrate that MG-Select achieves significant performance improvements, including a 28%/35% improvement in real-world in-distribution/out-of-distribution tasks, along with a 168% relative gain on RoboCasa pick-and-place tasks trained with 30 demonstrations. 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）在机器人控制中展现了出色的表现，但由于其单一推理范式，它们在需要高精度的任务中仍然存在根本性的限制。虽然使用外部验证者的测试时缩放方法显示出潜力，但它们需要额外的训练并在未见条件下缺乏泛化能力。我们提出了Masking Distribution Guided Selection（MG-Select），这是一种新颖的适用于VLAs的测试时缩放框架，该框架利用模型的内部属性，无需额外训练或外部模块。我们的方法使用与参考动作令牌分布的KL散度作为从多个候选动作中选择最优动作的信心指标。我们引入了一个由相同的VLA生成的参考分布，该分布使用随机遮掩的状态和语言条件作为输入，从而确保最大的不确定性并保持与目标任务分布的一致性。此外，我们提出了一种联合训练策略，使得模型能够在应用状态和语言条件的dropout后学习条件和无条件分布，从而进一步改进参考分布的质量。实验结果显示，MG-Select实现了显著的性能提升，包括在真实的分布内/分布外任务中分别提高了28%/35%，以及在使用30个示范训练下的RoboCasa抓取和放置任务中相对增益达到168%。 

---
# Beyond Spectral Peaks: Interpreting the Cues Behind Synthetic Image Detection 

**Title (ZH)**: 超出光谱峰值：合成图像检测背后线索的解释 

**Authors**: Sara Mandelli, Diego Vila-Portela, David Vázquez-Padín, Paolo Bestagini, Fernando Pérez-González  

**Link**: [PDF](https://arxiv.org/pdf/2510.05633)  

**Abstract**: Over the years, the forensics community has proposed several deep learning-based detectors to mitigate the risks of generative AI. Recently, frequency-domain artifacts (particularly periodic peaks in the magnitude spectrum), have received significant attention, as they have been often considered a strong indicator of synthetic image generation. However, state-of-the-art detectors are typically used as black-boxes, and it still remains unclear whether they truly rely on these peaks. This limits their interpretability and trust. In this work, we conduct a systematic study to address this question. We propose a strategy to remove spectral peaks from images and analyze the impact of this operation on several detectors. In addition, we introduce a simple linear detector that relies exclusively on frequency peaks, providing a fully interpretable baseline free from the confounding influence of deep learning. Our findings reveal that most detectors are not fundamentally dependent on spectral peaks, challenging a widespread assumption in the field and paving the way for more transparent and reliable forensic tools. 

**Abstract (ZH)**: 近年来，法证社区提出了多种基于深度学习的检测器以减轻生成式AI带来的风险。最近，频域特征（特别是幅度谱中的周期峰值）受到了广泛关注，因为它们经常被认为是有强大指示性的合成图像生成标志。然而，最先进的检测器通常被视为黑盒模型，仍不清楚它们是否真正依赖这些峰值。这限制了它们的可解释性和可信度。在本文中，我们进行了一项系统性的研究来解决这一问题。我们提出了一种从图像中移除频域峰值的策略，并分析了这一操作对该检测器的影响。此外，我们引入了一个仅依赖于频域峰值的简单线性检测器，提供了一个无深度学习混淆影响的完全可解释基准。我们的研究发现表明，大多数检测器本质上并不依赖频域峰值，这挑战了该领域的普遍假设，并为更透明和可靠的法证工具铺平了道路。 

---
# PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction 

**Title (ZH)**: PointNSP: 自回归3D点云生成结合下一尺度细节预测 

**Authors**: Ziqiao Meng, Qichao Wang, Zhiyang Dou, Zixing Song, Zhipeng Zhou, Irwin King, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05613)  

**Abstract**: Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential. 

**Abstract (ZH)**: 自回归点云生成在质量上长期落后于基于扩散的方法。性能差距源于自回归模型对本就无序的点集施加的人工顺序，强制形状生成按局部预测序列进行。这种顺序偏差强调了短程连续性，但削弱了模型捕捉远程依赖的能力，妨碍了其对全局结构属性（如对称性、一致拓扑和大尺度几何规律）的约束。受形状建模中细节层次（LOD）原则的启发，我们提出了一种粗细粒度生成框架PointNSP，该框架在低分辨率下保留全局形状结构，并通过下一个尺度预测范式逐步细化高尺度下的细粒度几何。这种多尺度分解使自回归目标与点集的置换不变性质对齐，促进了丰富的局域交互，同时避免了僵化的固定顺序。实验表明，PointNSP在自回归范式下首次达到最先进的生成质量。此外，它在参数量、训练效率和推理效率上优于强大的基于扩散的方法。最后，在稠密生成8,192点的情况下，PointNSP的优势更加明显，突显了其可扩展性潜力。 

---
# Improving Chain-of-Thought Efficiency for Autoregressive Image Generation 

**Title (ZH)**: 提高自回归图像生成的链条思维效率 

**Authors**: Zeqi Gu, Markos Georgopoulos, Xiaoliang Dai, Marjan Ghazvininejad, Chu Wang, Felix Juefei-Xu, Kunpeng Li, Yujun Shi, Zecheng He, Zijian He, Jiawei Zhou, Abe Davis, Jialiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05593)  

**Abstract**: Autoregressive multimodal large language models have recently gained popularity for image generation, driven by advances in foundation models. To enhance alignment and detail, newer approaches employ chain-of-thought (CoT) reasoning, expanding user inputs into elaborated prompts prior to image synthesis. However, this strategy can introduce unnecessary redundancy -- a phenomenon we call visual overthinking -- which increases computational costs and can introduce details that contradict the original prompt. In this work, we explore how to generate more concise CoT sequences for more efficient image generation. We introduce ShortCoTI, a lightweight optimization framework that encourages more concise CoT while preserving output image quality. ShortCoTI rewards more concise prompts with an adaptive function that scales according to an estimated difficulty for each task. Incorporating this reward into a reinforcement learning paradigm reduces prompt reasoning length by 54% while maintaining or slightly improving quality metrics across multiple benchmarks (T2I-CompBench, GenEval). Qualitative analysis shows that our method eliminates verbose explanations and repetitive refinements, producing reasoning prompts that are both concise and semantically rich. As a result, ShortCoTI improves computational efficiency without compromising the fidelity or visual appeal of generated images. 

**Abstract (ZH)**: 自回归多模态大型语言模型在图像生成中的应用：一种减少视觉过度思考的轻量级优化框架 

---
# See the past: Time-Reversed Scene Reconstruction from Thermal Traces Using Visual Language Models 

**Title (ZH)**: 时光倒流：基于视觉语言模型的热迹时光反向场景重构 

**Authors**: Kebin Contreras, Luis Toscano-Palomino, Mauro Dalla Mura, Jorge Bacca  

**Link**: [PDF](https://arxiv.org/pdf/2510.05408)  

**Abstract**: Recovering the past from present observations is an intriguing challenge with potential applications in forensics and scene analysis. Thermal imaging, operating in the infrared range, provides access to otherwise invisible information. Since humans are typically warmer (37 C -98.6 F) than their surroundings, interactions such as sitting, touching, or leaning leave residual heat traces. These fading imprints serve as passive temporal codes, allowing for the inference of recent events that exceed the capabilities of RGB cameras. This work proposes a time-reversed reconstruction framework that uses paired RGB and thermal images to recover scene states from a few seconds earlier. The proposed approach couples Visual-Language Models (VLMs) with a constrained diffusion process, where one VLM generates scene descriptions and another guides image reconstruction, ensuring semantic and structural consistency. The method is evaluated in three controlled scenarios, demonstrating the feasibility of reconstructing plausible past frames up to 120 seconds earlier, providing a first step toward time-reversed imaging from thermal traces. 

**Abstract (ZH)**: 从当前观察恢复过去：一种潜在应用于法医和场景分析的挑战性问题及其红外成像解决方案 

---
# Generative Inverse Design: From Single Point Optimization to a Diverse Design Portfolio via Conditional Variational Autoencoders 

**Title (ZH)**: 生成逆向设计：通过条件变分自动编码器从单一点优化到多样设计组合 

**Authors**: Muhammad Arif Hakimi Zamrai  

**Link**: [PDF](https://arxiv.org/pdf/2510.05160)  

**Abstract**: Inverse design, which seeks to find optimal parameters for a target output, is a central challenge in engineering. Surrogate-based optimization (SBO) has become a standard approach, yet it is fundamentally structured to converge to a single-point solution, thereby limiting design space exploration and ignoring potentially valuable alternative topologies. This paper presents a paradigm shift from single-point optimization to generative inverse design. We introduce a framework based on a Conditional Variational Autoencoder (CVAE) that learns a probabilistic mapping between a system's design parameters and its performance, enabling the generation of a diverse portfolio of high-performing candidates conditioned on a specific performance objective. We apply this methodology to the complex, non-linear problem of minimizing airfoil self-noise, using a high-performing SBO method from a prior benchmark study as a rigorous baseline. The CVAE framework successfully generated 256 novel designs with a 94.1\% validity rate. A subsequent surrogate-based evaluation revealed that 77.2\% of these valid designs achieved superior performance compared to the single optimal design found by the SBO baseline. This work demonstrates that the generative approach not only discovers higher-quality solutions but also provides a rich portfolio of diverse candidates, fundamentally enhancing the engineering design process by enabling multi-criteria decision-making. 

**Abstract (ZH)**: 基于生成的逆向设计：从单点优化到生成式逆向设计 

---
