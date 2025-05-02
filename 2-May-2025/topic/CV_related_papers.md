# iMacSR: Intermediate Multi-Access Supervision and Regularization in Training Autonomous Driving Models 

**Title (ZH)**: iMacSR：训练自动驾驶模型的中间多访问监督与正则化 

**Authors**: Wei-Bin Kou, Guangxu Zhu, Yichen Jin, Shuai Wang, Ming Tang, Yik-Chung Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00404)  

**Abstract**: Deep Learning (DL)-based street scene semantic understanding has become a cornerstone of autonomous driving (AD). DL model performance heavily relies on network depth. Specifically, deeper DL architectures yield better segmentation performance. However, as models grow deeper, traditional one-point supervision at the final layer struggles to optimize intermediate feature representations, leading to subpar training outcomes. To address this, we propose an intermediate Multi-access Supervision and Regularization (iMacSR) strategy. The proposed iMacSR introduces two novel components: (I) mutual information between latent features and ground truth as intermediate supervision loss ensures robust feature alignment at multiple network depths; and (II) negative entropy regularization on hidden features discourages overconfident predictions and mitigates overfitting. These intermediate terms are combined into the original final-layer training loss to form a unified optimization objective, enabling comprehensive optimization across the network hierarchy. The proposed iMacSR provides a robust framework for training deep AD architectures, advancing the performance of perception systems in real-world driving scenarios. In addition, we conduct theoretical convergence analysis for the proposed iMacSR. Extensive experiments on AD benchmarks (i.e., Cityscapes, CamVid, and SynthiaSF datasets) demonstrate that iMacSR outperforms conventional final-layer single-point supervision method up to 9.19% in mean Intersection over Union (mIoU). 

**Abstract (ZH)**: 基于深度学习的街道场景语义理解已成为自动驾驶的核心基石。提出的iMacSR策略通过引入中间多访问监督和正则化，提升了网络性能。 

---
# T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT 

**Title (ZH)**: T2I-R1：通过协作的语义级和tokens级共推理增强图像生成 

**Authors**: Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00703)  

**Abstract**: Recent advancements in large language models have demonstrated how chain-of-thought (CoT) and reinforcement learning (RL) can improve performance. However, applying such reasoning strategies to the visual generation domain remains largely unexplored. In this paper, we present T2I-R1, a novel reasoning-enhanced text-to-image generation model, powered by RL with a bi-level CoT reasoning process. Specifically, we identify two levels of CoT that can be utilized to enhance different stages of generation: (1) the semantic-level CoT for high-level planning of the prompt and (2) the token-level CoT for low-level pixel processing during patch-by-patch generation. To better coordinate these two levels of CoT, we introduce BiCoT-GRPO with an ensemble of generation rewards, which seamlessly optimizes both generation CoTs within the same training step. By applying our reasoning strategies to the baseline model, Janus-Pro, we achieve superior performance with 13% improvement on T2I-CompBench and 19% improvement on the WISE benchmark, even surpassing the state-of-the-art model FLUX.1. Code is available at: this https URL 

**Abstract (ZH)**: 近期大规模语言模型的发展展示了链式思考(CoT)和强化学习(RL)如何提升性能。然而，将这些推理策略应用于视觉生成领域仍 largely unexplored。在本文中，我们提出了一种名为 T2I-R1 的新型增强推理文本到图像生成模型，该模型基于具有两层链式思考推理过程的强化学习。具体而言，我们识别了两种可以用于提高生成不同阶段的 CoT：(1) 语义层次的链式思考用于高阶提示规划；(2) 令牌层次的链式思考用于生成过程中的像素级处理。为了更好地协调这两种层次的 CoT，我们引入了结合生成奖励的 BiCoT-GRPO，该方法在同一训练步骤中无缝优化了两种生成 CoT。通过将我们的推理策略应用到基线模型 Janus-Pro 中，我们在 T2I-CompBench 上实现了 13% 的性能提升，在 WISE 基准上实现了 19% 的性能提升，甚至超越了最先进的模型 FLUX。1. 代码可在以下链接获取：this https URL 

---
# Visual Test-time Scaling for GUI Agent Grounding 

**Title (ZH)**: GUI代理定位的视觉测试时缩放 

**Authors**: Tiange Luo, Lajanugen Logeswaran, Justin Johnson, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.00684)  

**Abstract**: We introduce RegionFocus, a visual test-time scaling approach for Vision Language Model Agents. Understanding webpages is challenging due to the visual complexity of GUI images and the large number of interface elements, making accurate action selection difficult. Our approach dynamically zooms in on relevant regions, reducing background clutter and improving grounding accuracy. To support this process, we propose an image-as-map mechanism that visualizes key landmarks at each step, providing a transparent action record and enables the agent to effectively choose among action candidates. Even with a simple region selection strategy, we observe significant performance gains of 28+\% on Screenspot-pro and 24+\% on WebVoyager benchmarks on top of two state-of-the-art open vision language model agents, UI-TARS and Qwen2.5-VL, highlighting the effectiveness of visual test-time scaling in interactive settings. We achieve a new state-of-the-art grounding performance of 61.6\% on the ScreenSpot-Pro benchmark by applying RegionFocus to a Qwen2.5-VL-72B model. Our code will be released publicly at this https URL. 

**Abstract (ZH)**: 我们引入了RegionFocus，一种视觉测试时缩放方法，用于视觉语言模型代理。由于网页中的GUI图像具有视觉复杂性且界面元素众多，理解网页颇具挑战性，准确的动作选择变得困难。我们的方法动态放大相关区域，减少背景杂乱，提高语义匹配准确性。为支持这一过程，我们提出了一种图像即地图机制，在每一步可视化关键地标，提供透明的动作记录，并使代理能够有效地在动作候选中进行选择。即使采用简单的区域选择策略，我们也在Screenspot-pro和WebVoyager基准上分别观察到UI-TARS和Qwen2.5-VL两个最先进的开放视觉语言模型代理28%+和24%+的显著性能提升，突显了在交互环境中视觉测试时缩放的有效性。通过将RegionFocus应用于Qwen2.5-VL-72B模型，我们在ScreenSpot-Pro基准上实现了新的最佳语义匹配性能61.6%。我们的代码将在以下链接公开发布：https://this-url。 

---
# Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction 

**Title (ZH)**: Pixel3DMM: 通用的屏幕空间先验进行单张图像三维人脸重建 

**Authors**: Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito, Matthias Nießner  

**Link**: [PDF](https://arxiv.org/pdf/2505.00615)  

**Abstract**: We address the 3D reconstruction of human faces from a single RGB image. To this end, we propose Pixel3DMM, a set of highly-generalized vision transformers which predict per-pixel geometric cues in order to constrain the optimization of a 3D morphable face model (3DMM). We exploit the latent features of the DINO foundation model, and introduce a tailored surface normal and uv-coordinate prediction head. We train our model by registering three high-quality 3D face datasets against the FLAME mesh topology, which results in a total of over 1,000 identities and 976K images. For 3D face reconstruction, we propose a FLAME fitting opitmization that solves for the 3DMM parameters from the uv-coordinate and normal estimates. To evaluate our method, we introduce a new benchmark for single-image face reconstruction, which features high diversity facial expressions, viewing angles, and ethnicities. Crucially, our benchmark is the first to evaluate both posed and neutral facial geometry. Ultimately, our method outperforms the most competitive baselines by over 15% in terms of geometric accuracy for posed facial expressions. 

**Abstract (ZH)**: 我们提出了一种从单张RGB图像重建人类面部的3D重建方法。为此，我们提出了Pixel3DMM，这是一种高度通用的视觉变换器集合，用于预测像素级几何线索，从而约束3D可变形面部模型（3DMM）的优化。我们利用DINO基础模型的潜在特征，并引入了定制的表面法线和uv坐标预测头。我们通过将三个高质量的3D面部数据集注册到FLAME网格拓扑上来训练我们的模型，从而总共获得了超过1000个身份和976K张图像。对于3D面部重建，我们提出了一种FLAME拟合优化方法，从uv坐标和法线估计中求解3DMM参数。为了评估我们的方法，我们引入了一个新的单一图像面部重建基准，该基准具备高度多样化的面部表情、视角和种族特征。最关键的是，我们的基准首次同时评估了表情和中性的面部几何结构。最终，我们的方法在表情面部几何精度方面比最具有竞争力的基线方法高出超过15%。 

---
# Multimodal Masked Autoencoder Pre-training for 3D MRI-Based Brain Tumor Analysis with Missing Modalities 

**Title (ZH)**: 基于MRI的脑肿瘤分析中多模态掩蔽自动编码器预训练方法研究（缺失模态情况） 

**Authors**: Lucas Robinet, Ahmad Berjaoui, Elizabeth Cohen-Jonathan Moyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.00568)  

**Abstract**: Multimodal magnetic resonance imaging (MRI) constitutes the first line of investigation for clinicians in the care of brain tumors, providing crucial insights for surgery planning, treatment monitoring, and biomarker identification. Pre-training on large datasets have been shown to help models learn transferable representations and adapt with minimal labeled data. This behavior is especially valuable in medical imaging, where annotations are often scarce. However, applying this paradigm to multimodal medical data introduces a challenge: most existing approaches assume that all imaging modalities are available during both pre-training and fine-tuning. In practice, missing modalities often occur due to acquisition issues, specialist unavailability, or specific experimental designs on small in-house datasets. Consequently, a common approach involves training a separate model for each desired modality combination, making the process both resource-intensive and impractical for clinical use. Therefore, we introduce BM-MAE, a masked image modeling pre-training strategy tailored for multimodal MRI data. The same pre-trained model seamlessly adapts to any combination of available modalities, extracting rich representations that capture both intra- and inter-modal information. This allows fine-tuning on any subset of modalities without requiring architectural changes, while still benefiting from a model pre-trained on the full set of modalities. Extensive experiments show that the proposed pre-training strategy outperforms or remains competitive with baselines that require separate pre-training for each modality subset, while substantially surpassing training from scratch on several downstream tasks. Additionally, it can quickly and efficiently reconstruct missing modalities, highlighting its practical value. Code and trained models are available at: this https URL 

**Abstract (ZH)**: 多模态磁共振成像（MRI）是临床医生在脑肿瘤护理中进行初步调查的第一手段，为手术规划、治疗监控和生物标志物识别提供了关键见解。在大规模数据集上的预训练已被证明有助于模型学习可迁移的表示，并在最少标注数据的情况下进行适应。这种行为在医学影像领域尤为重要，因为标注数据往往稀缺。然而，将这种范例应用于多模态医学数据引入了挑战：大多数现有方法假设所有成像模态在预训练和微调过程中均可用。实际上，模态缺失往往由于获取问题、专家不可用或小型院内数据集的具体实验设计等原因出现。因此，一种常见做法是为每种所需的模态组合训练一个单独的模型，这不仅资源密集，而且不适用于临床应用。因此，我们引入了BM-MAE，一种专为多模态MRI数据设计的掩码图像建模预训练策略。预训练模型能够无缝适应任何可用模态组合，提取能够捕捉跨模态和层内信息的丰富表示。这使得可以在任何子集模态上进行微调而无需改变架构，同时仍能从预训练了所有模态的模型中受益。广泛的经验表明，提出的预训练策略在多个下游任务中的表现优于或与需要为每个模态子集单独预训练的基线保持竞争力，并在某些任务上显著超越从零开始训练。此外，它还可以快速高效地重建缺失的模态，突显其实用价值。代码和训练模型可在以下链接获取：this https URL。 

---
# TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching 

**Title (ZH)**: TeLoGraF: 基于图编码流匹配的时间逻辑规划 

**Authors**: Yue Meng, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00562)  

**Abstract**: Learning to solve complex tasks with signal temporal logic (STL) specifications is crucial to many real-world applications. However, most previous works only consider fixed or parametrized STL specifications due to the lack of a diverse STL dataset and encoders to effectively extract temporal logic information for downstream tasks. In this paper, we propose TeLoGraF, Temporal Logic Graph-encoded Flow, which utilizes Graph Neural Networks (GNN) encoder and flow-matching to learn solutions for general STL specifications. We identify four commonly used STL templates and collect a total of 200K specifications with paired demonstrations. We conduct extensive experiments in five simulation environments ranging from simple dynamical models in the 2D space to high-dimensional 7DoF Franka Panda robot arm and Ant quadruped navigation. Results show that our method outperforms other baselines in the STL satisfaction rate. Compared to classical STL planning algorithms, our approach is 10-100X faster in inference and can work on any system dynamics. Besides, we show our graph-encoding method's capability to solve complex STLs and robustness to out-distribution STL specifications. Code is available at this https URL 

**Abstract (ZH)**: 使用信号时序逻辑（STL）规范学习解决复杂任务：TeLoGraF，基于图编码流的方法 

---
# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers 

**Title (ZH)**: JointDiT: 提升RGB-深度联合建模的扩散变换器方法 

**Authors**: Kwon Byung-Ki, Qi Dai, Lee Hyoseok, Chong Luo, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00482)  

**Abstract**: We present JointDiT, a diffusion transformer that models the joint distribution of RGB and depth. By leveraging the architectural benefit and outstanding image prior of the state-of-the-art diffusion transformer, JointDiT not only generates high-fidelity images but also produces geometrically plausible and accurate depth maps. This solid joint distribution modeling is achieved through two simple yet effective techniques that we propose, i.e., adaptive scheduling weights, which depend on the noise levels of each modality, and the unbalanced timestep sampling strategy. With these techniques, we train our model across all noise levels for each modality, enabling JointDiT to naturally handle various combinatorial generation tasks, including joint generation, depth estimation, and depth-conditioned image generation by simply controlling the timestep of each branch. JointDiT demonstrates outstanding joint generation performance. Furthermore, it achieves comparable results in depth estimation and depth-conditioned image generation, suggesting that joint distribution modeling can serve as a replaceable alternative to conditional generation. The project page is available at this https URL. 

**Abstract (ZH)**: 我们 presents JointDiT，一种建模RGB和深度联合分布的扩散变换器。通过利用最新扩散变换器的架构优势和出色的图像先验知识，JointDiT 不仅生成高保真图像，还能产生几何上合理且准确的深度图。通过我们提出的一种简单而有效的技术，即适应性调度权重（取决于每种模态的噪声水平）和不平衡时间步采样策略，实现了这种坚实的联合分布建模。借助这些技术，我们在每个模态的所有噪声水平下训练模型，使得JointDiT能够轻松处理各种组合生成任务，包括联合生成、深度估计和深度条件下的图像生成，只需控制每个分支的时间步即可。JointDiT展示了出色的联合生成性能。此外，在深度估计和深度条件下的图像生成方面，它达到了可比的结果，表明联合分布建模可以作为条件生成的一种可替代方案。项目页面在此处 accessible at this https URL。 

---
# T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation 

**Title (ZH)**: T2VPhysBench: 首个文本到视频生成物理一致性基准 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00337)  

**Abstract**: Text-to-video generative models have made significant strides in recent years, producing high-quality videos that excel in both aesthetic appeal and accurate instruction following, and have become central to digital art creation and user engagement online. Yet, despite these advancements, their ability to respect fundamental physical laws remains largely untested: many outputs still violate basic constraints such as rigid-body collisions, energy conservation, and gravitational dynamics, resulting in unrealistic or even misleading content. Existing physical-evaluation benchmarks typically rely on automatic, pixel-level metrics applied to simplistic, life-scenario prompts, and thus overlook both human judgment and first-principles physics. To fill this gap, we introduce \textbf{T2VPhysBench}, a first-principled benchmark that systematically evaluates whether state-of-the-art text-to-video systems, both open-source and commercial, obey twelve core physical laws including Newtonian mechanics, conservation principles, and phenomenological effects. Our benchmark employs a rigorous human evaluation protocol and includes three targeted studies: (1) an overall compliance assessment showing that all models score below 0.60 on average in each law category; (2) a prompt-hint ablation revealing that even detailed, law-specific hints fail to remedy physics violations; and (3) a counterfactual robustness test demonstrating that models often generate videos that explicitly break physical rules when so instructed. The results expose persistent limitations in current architectures and offer concrete insights for guiding future research toward truly physics-aware video generation. 

**Abstract (ZH)**: Text-to-video生成模型在近年来取得了显著进展，不仅在美学和准确的指令遵循方面表现出色，而且成为了数字艺术创作和在线用户参与的核心。然而，尽管取得了这些进步，它们是否遵守基本物理定律方面仍然鲜有测试：许多输出仍然违背了基本约束，如刚体碰撞、能量守恒和重力动力学，导致了不现实甚至误导的内容。现有的物理评估基准通常依赖于自动的、像素级别的指标，应用于简单的生活场景提示，因此未能考虑人类判断和第一性原理物理。为弥补这一缺口，我们引入了**T2VPhysBench**，这是一个基于第一性原理的基准，系统地评估最先进的文本到视频系统，无论是开源的还是商业的，是否遵循包括牛顿力学、守恒原则和表征效应在内的十二项核心物理定律。我们的基准采用了严格的评估协议，并包括三个专门的研究：（1）总体合规性评估，显示所有模型在每个定律类别中的平均得分低于0.60；（2）提示-线索消融实验，揭示即使详细的、针对特定定律的线索也无法纠正物理违规；（3）反事实鲁棒性测试，证明当模型被明确指示时，它们往往生成违反物理法则的视频。结果揭示了当前架构存在的持续局限性，并为指导未来研究走向真正物理感知的视频生成提供了具体的见解。 

---
# Efficient Neural Video Representation with Temporally Coherent Modulation 

**Title (ZH)**: 具有时间连贯调制的高效神经视频表示 

**Authors**: Seungjun Shin, Suji Kim, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00335)  

**Abstract**: Implicit neural representations (INR) has found successful applications across diverse domains. To employ INR in real-life, it is important to speed up training. In the field of INR for video applications, the state-of-the-art approach employs grid-type parametric encoding and successfully achieves a faster encoding speed in comparison to its predecessors. However, the grid usage, which does not consider the video's dynamic nature, leads to redundant use of trainable parameters. As a result, it has significantly lower parameter efficiency and higher bitrate compared to NeRV-style methods that do not use a parametric encoding. To address the problem, we propose Neural Video representation with Temporally coherent Modulation (NVTM), a novel framework that can capture dynamic characteristics of video. By decomposing the spatio-temporal 3D video data into a set of 2D grids with flow information, NVTM enables learning video representation rapidly and uses parameter efficiently. Our framework enables to process temporally corresponding pixels at once, resulting in the fastest encoding speed for a reasonable video quality, especially when compared to the NeRV-style method, with a speed increase of over 3 times. Also, it remarks an average of 1.54dB/0.019 improvements in PSNR/LPIPS on UVG (Dynamic) (even with 10% fewer parameters) and an average of 1.84dB/0.013 improvements in PSNR/LPIPS on MCL-JCV (Dynamic), compared to previous grid-type works. By expanding this to compression tasks, we demonstrate comparable performance to video compression standards (H.264, HEVC) and recent INR approaches for video compression. Additionally, we perform extensive experiments demonstrating the superior performance of our algorithm across diverse tasks, encompassing super resolution, frame interpolation and video inpainting. Project page is this https URL. 

**Abstract (ZH)**: 基于时间一致性调制的神经视频表示（NVTM）：快速高效的学习动态视频表示 

---
# Fine-grained spatial-temporal perception for gas leak segmentation 

**Title (ZH)**: 细粒度时空感知的气体泄漏分割 

**Authors**: Xinlong Zhao, Shan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.00295)  

**Abstract**: Gas leaks pose significant risks to human health and the environment. Despite long-standing concerns, there are limited methods that can efficiently and accurately detect and segment leaks due to their concealed appearance and random shapes. In this paper, we propose a Fine-grained Spatial-Temporal Perception (FGSTP) algorithm for gas leak segmentation. FGSTP captures critical motion clues across frames and integrates them with refined object features in an end-to-end network. Specifically, we first construct a correlation volume to capture motion information between consecutive frames. Then, the fine-grained perception progressively refines the object-level features using previous outputs. Finally, a decoder is employed to optimize boundary segmentation. Because there is no highly precise labeled dataset for gas leak segmentation, we manually label a gas leak video dataset, GasVid. Experimental results on GasVid demonstrate that our model excels in segmenting non-rigid objects such as gas leaks, generating the most accurate mask compared to other state-of-the-art (SOTA) models. 

**Abstract (ZH)**: 细粒度时空知觉算法（FGSTP）在天然气泄露分割中的应用 

---
# Empowering Agentic Video Analytics Systems with Video Language Models 

**Title (ZH)**: 赋能于视频语言模型的代理视频分析系统 

**Authors**: Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, Yuqing Yang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00254)  

**Abstract**: AI-driven video analytics has become increasingly pivotal across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Video-Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVA, a VLM-powered system designed for open-ended, advanced video analytics. AVA incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively, significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVA-100, AVA achieves top-tier performance with an accuracy of 75.8%. 

**Abstract (ZH)**: 基于AI驱动的视频分析在多个领域中变得越来越关键。然而，现有系统往往受限于特定的预定义任务，限制了其在开放性分析场景中的适应性。最近出现的视频-语言模型（VLMs）作为一种变革性技术，提供了使开放性的视频理解、推理和分析成为可能的巨大潜力。但是，它们有限的上下文窗口在处理广泛存在的超长视频内容时带来了挑战。为解决这个问题，我们引入了AVA，这是一种基于VLM的系统，旨在实现开放性的高级视频分析。AVA结合了两项关键创新：（1）近实时构建事件知识图（EKGs）以高效索引长或连续的视频流，（2）一种代理检索-生成机制，利用EKGs处理复杂的多样查询。在公共基准LVBench和VideoMME-Long上的全面评估表明，AVA达到了最先进的性能，分别取得了62.3%和64.1%的准确率，显著优于现有的VLM和视频检索增强生成（RAG）系统。此外，为评估超长和开放世界视频场景下的视频分析，我们引入了一个新的基准AVA-100。该基准包含8个各超过10小时的视频，以及120个手动标注的多样化和复杂的问答对。在AVA-100上，AVA实现了顶级性能，准确率为75.8%。 

---
