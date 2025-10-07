# CLEAR-IR: Clarity-Enhanced Active Reconstruction of Infrared Imagery 

**Title (ZH)**: CLEAR-IR: 光学清晰度增强的主动红外成像重构 

**Authors**: Nathan Shankar, Pawel Ladosz, Hujun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04883)  

**Abstract**: This paper presents a novel approach for enabling robust robotic perception in dark environments using infrared (IR) stream. IR stream is less susceptible to noise than RGB in low-light conditions. However, it is dominated by active emitter patterns that hinder high-level tasks such as object detection, tracking and localisation. To address this, a U-Net-based architecture is proposed that reconstructs clean IR images from emitter-populated input, improving both image quality and downstream robotic performance. This approach outperforms existing enhancement techniques and enables reliable operation of vision-driven robotic systems across illumination conditions from well-lit to extreme low-light scenes. 

**Abstract (ZH)**: 利用红外流在暗环境下实现稳健的机器人感知的新方法 

---
# OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS 

**Title (ZH)**: OKVIS2-X：基于开放关键帧的视觉-惯性SLAM，支持密集深度或LiDAR及GNSS配置 

**Authors**: Simon Boche, Jaehyung Jung, Sebastián Barbas Laina, Stefan Leutenegger  

**Link**: [PDF](https://arxiv.org/pdf/2510.04612)  

**Abstract**: To empower mobile robots with usable maps as well as highest state estimation accuracy and robustness, we present OKVIS2-X: a state-of-the-art multi-sensor Simultaneous Localization and Mapping (SLAM) system building dense volumetric occupancy maps, while scalable to large environments and operating in realtime. Our unified SLAM framework seamlessly integrates different sensor modalities: visual, inertial, measured or learned depth, LiDAR and Global Navigation Satellite System (GNSS) measurements. Unlike most state-of-the-art SLAM systems, we advocate using dense volumetric map representations when leveraging depth or range-sensing capabilities. We employ an efficient submapping strategy that allows our system to scale to large environments, showcased in sequences of up to 9 kilometers. OKVIS2-X enhances its accuracy and robustness by tightly-coupling the estimator and submaps through map alignment factors. Our system provides globally consistent maps, directly usable for autonomous navigation. To further improve the accuracy of OKVIS2-X, we also incorporate the option of performing online calibration of camera extrinsics. Our system achieves the highest trajectory accuracy in EuRoC against state-of-the-art alternatives, outperforms all competitors in the Hilti22 VI-only benchmark, while also proving competitive in the LiDAR version, and showcases state of the art accuracy in the diverse and large-scale sequences from the VBR dataset. 

**Abstract (ZH)**: OKVIS2-X：一种构建稠密体占图的高性能多传感器SLAM系统 

---
# TCB-VIO: Tightly-Coupled Focal-Plane Binary-Enhanced Visual Inertial Odometry 

**Title (ZH)**: TCB-VIO: 紧密耦合的焦平面二进制增强视觉惯性odometry 

**Authors**: Matthew Lisondra, Junseo Kim, Glenn Takashi Shimoda, Kourosh Zareinia, Sajad Saeedi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03919)  

**Abstract**: Vision algorithms can be executed directly on the image sensor when implemented on the next-generation sensors known as focal-plane sensor-processor arrays (FPSP)s, where every pixel has a processor. FPSPs greatly improve latency, reducing the problems associated with the bottleneck of data transfer from a vision sensor to a processor. FPSPs accelerate vision-based algorithms such as visual-inertial odometry (VIO). However, VIO frameworks suffer from spatial drift due to the vision-based pose estimation, whilst temporal drift arises from the inertial measurements. FPSPs circumvent the spatial drift by operating at a high frame rate to match the high-frequency output of the inertial measurements. In this paper, we present TCB-VIO, a tightly-coupled 6 degrees-of-freedom VIO by a Multi-State Constraint Kalman Filter (MSCKF), operating at a high frame-rate of 250 FPS and from IMU measurements obtained at 400 Hz. TCB-VIO outperforms state-of-the-art methods: ROVIO, VINS-Mono, and ORB-SLAM3. 

**Abstract (ZH)**: 基于多状态约束卡尔曼滤波器的高帧率紧密耦合6自由度视觉惯性里程计TCB-VIO 

---
# Seeing the Bigger Picture: 3D Latent Mapping for Mobile Manipulation Policy Learning 

**Title (ZH)**: 从全局视角出发：面向移动 Manipulation 策略学习的 3D 潜在空间映射 

**Authors**: Sunghwan Kim, Woojeh Chung, Zhirui Dai, Dwait Bhatt, Arth Shukla, Hao Su, Yulun Tian, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03885)  

**Abstract**: In this paper, we demonstrate that mobile manipulation policies utilizing a 3D latent map achieve stronger spatial and temporal reasoning than policies relying solely on images. We introduce Seeing the Bigger Picture (SBP), an end-to-end policy learning approach that operates directly on a 3D map of latent features. In SBP, the map extends perception beyond the robot's current field of view and aggregates observations over long horizons. Our mapping approach incrementally fuses multiview observations into a grid of scene-specific latent features. A pre-trained, scene-agnostic decoder reconstructs target embeddings from these features and enables online optimization of the map features during task execution. A policy, trainable with behavior cloning or reinforcement learning, treats the latent map as a state variable and uses global context from the map obtained via a 3D feature aggregator. We evaluate SBP on scene-level mobile manipulation and sequential tabletop manipulation tasks. Our experiments demonstrate that SBP (i) reasons globally over the scene, (ii) leverages the map as long-horizon memory, and (iii) outperforms image-based policies in both in-distribution and novel scenes, e.g., improving the success rate by 25% for the sequential manipulation task. 

**Abstract (ZH)**: 在本文中，我们展示了利用3D潜空间映射的移动操作策略在空间和时间推理方面比仅依赖图像的策略更强大。我们引入了一种端到端的策略学习方法Seeing the Bigger Picture (SBP)，该方法直接操作3D特征映射。在SBP中，映射扩展了机器人的感知范围，超出了当前视野，并在长时间段内聚合观测结果。我们的建图方法逐步将多视图观测结果融合到场景特定的特征格网中。一个预先训练好的、场景无关的解码器从这些特征中重建目标嵌入，并在任务执行期间使映射特征的优化成为可能。一个使用行为克隆或强化学习训练的策略将潜空间映射视为状态变量，并利用3D特征聚合器从映射中获得的全局上下文。我们在场景级移动操作和序列式桌面操作任务上评估了SBP。我们的实验表明，SBP能够在全球范围内推理场景，利用映射作为长时记忆，并在分布内和新颖场景中优于基于图像的策略，例如，在序列操作任务中的成功率提高了25%。 

---
# More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models 

**Title (ZH)**: 不仅仅是表面所见？揭示视觉-语言驾驶模型中推理-规划的缺口 

**Authors**: Xurui Song, Shuo Huai, JingJing Jiang, Jiayi Kong, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04532)  

**Abstract**: Vision-Language Model (VLM) driving agents promise explainable end-to-end autonomy by first producing natural-language reasoning and then predicting trajectory planning. However, whether planning is causally driven by this reasoning remains a critical but unverified assumption. To investigate this, we build DriveMind, a large-scale driving Visual Question Answering (VQA) corpus with plan-aligned Chain-of-Thought (CoT), automatically generated from nuPlan. Our data generation process converts sensors and annotations into structured inputs and, crucially, separates priors from to-be-reasoned signals, enabling clean information ablations. Using DriveMind, we train representative VLM agents with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) and evaluate them with nuPlan's metrics. Our results, unfortunately, indicate a consistent causal disconnect in reasoning-planning: removing ego/navigation priors causes large drops in planning scores, whereas removing CoT produces only minor changes. Attention analysis further shows that planning primarily focuses on priors rather than the CoT. Based on this evidence, we propose the Reasoning-Planning Decoupling Hypothesis, positing that the training-yielded reasoning is an ancillary byproduct rather than a causal mediator. To enable efficient diagnosis, we also introduce a novel, training-free probe that measures an agent's reliance on priors by evaluating its planning robustness against minor input perturbations. In summary, we provide the community with a new dataset and a diagnostic tool to evaluate the causal fidelity of future models. 

**Abstract (ZH)**: 基于视觉-语言模型的驾驶代理承诺实现可解释的端到端自主驾驶，首先产生自然语言推理，然后预测轨迹规划。然而，这种规划是否由这种推理因果驱动仍是一个关键但未验证的假设。为调查这一问题，我们构建了DriveMind，这是一个与计划对齐的链式思维大规模驾驶视觉问答语料库，自动生成自nuPlan。我们的数据生成过程将传感器和注释转换为结构化输入，并 crucial地将先验知识与待推理的信号分离，从而实现清洁的信息消融。使用DriveMind，我们使用监督微调（SFT）和组相对策略优化（GRPO）训练代表性视觉-语言模型代理，并使用nuPlan的指标进行评估。不幸的是，我们的结果显示了推理-规划中的一致因果脱节：移除自/导航先验会导致规划得分的大幅下降，而移除链式思维仅产生轻微变化。注意力分析进一步显示，规划主要关注先验而非链式思维。基于这些证据，我们提出了推理-规划脱耦假设，认为训练产生的推理只是一个辅助副产品，而非因果中介。为了实现高效的诊断，我们还引入了一种新的、无需训练的探针，通过评估代理在轻微输入扰动下的规划鲁棒性来衡量其对先验的依赖程度。总之，我们为社区提供了新的数据集和诊断工具，以评估未来模型的因果忠实度。 

---
# Paper2Video: Automatic Video Generation from Scientific Papers 

**Title (ZH)**: Paper2Video：从科学论文自动生成视频 

**Authors**: Zeyu Zhu, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05096)  

**Abstract**: Academic presentation videos have become an essential medium for research communication, yet producing them remains highly labor-intensive, often requiring hours of slide design, recording, and editing for a short 2 to 10 minutes video. Unlike natural video, presentation video generation involves distinctive challenges: inputs from research papers, dense multi-modal information (text, figures, tables), and the need to coordinate multiple aligned channels such as slides, subtitles, speech, and human talker. To address these challenges, we introduce PaperTalker, the first benchmark of 101 research papers paired with author-created presentation videos, slides, and speaker metadata. We further design four tailored evaluation metrics--Meta Similarity, PresentArena, PresentQuiz, and IP Memory--to measure how videos convey the paper's information to the audience. Building on this foundation, we propose PaperTalker, the first multi-agent framework for academic presentation video generation. It integrates slide generation with effective layout refinement by a novel effective tree search visual choice, cursor grounding, subtitling, speech synthesis, and talking-head rendering, while parallelizing slide-wise generation for efficiency. Experiments on Paper2Video demonstrate that the presentation videos produced by our approach are more faithful and informative than existing baselines, establishing a practical step toward automated and ready-to-use academic video generation. Our dataset, agent, and code are available at this https URL. 

**Abstract (ZH)**: 学术演示视频已成为研究交流的重要媒介，但制作它们仍然高度劳动密集型，常常需要数小时的设计、录制和编辑时间，才能制作出2至10分钟的视频。与自然视频生成不同，演示视频生成涉及独特的挑战：来自研究论文的输入、密集的多模态信息（文本、图表、表格）以及需要协调多个对齐的通道，如幻灯片、字幕、语音和人类发言人。为应对这些挑战，我们介绍了PaperTalker，这是一个包含101篇研究论文及其作者创建的演示视频、幻灯片和发言人元数据的第一套基准数据集。在此基础上，我们提出了PaperTalker，这是首个用于学术演示视频生成的多代理框架。该框架通过一种新颖有效的树搜索视觉选择、光标定位、字幕、语音合成和头部演讲渲染，将幻灯片生成与有效的布局精炼结合起来，同时按幻灯片并行生成以提高效率。实验结果表明，我们方法生成的演示视频比现有基线更为忠实地传达了论文信息，为自动化和即用型学术视频生成奠定了实用步骤。我们的数据集、代理和代码可在以下链接获取。 

---
# SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder 

**Title (ZH)**: SAEdit: 通过稀疏自编码器实现的TokenType级连续图像编辑 

**Authors**: Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2510.05081)  

**Abstract**: Large-scale text-to-image diffusion models have become the backbone of modern image editing, yet text prompts alone do not offer adequate control over the editing process. Two properties are especially desirable: disentanglement, where changing one attribute does not unintentionally alter others, and continuous control, where the strength of an edit can be smoothly adjusted. We introduce a method for disentangled and continuous editing through token-level manipulation of text embeddings. The edits are applied by manipulating the embeddings along carefully chosen directions, which control the strength of the target attribute. To identify such directions, we employ a Sparse Autoencoder (SAE), whose sparse latent space exposes semantically isolated dimensions. Our method operates directly on text embeddings without modifying the diffusion process, making it model agnostic and broadly applicable to various image synthesis backbones. Experiments show that it enables intuitive and efficient manipulations with continuous control across diverse attributes and domains. 

**Abstract (ZH)**: 一种通过 token 级别操纵文本嵌入实现的解耦和连续图像编辑方法 

---
# Bridging Text and Video Generation: A Survey 

**Title (ZH)**: 文本生成与视频生成的桥梁：一个综述 

**Authors**: Nilay Kumar, Priyansh Bhandari, G. Maragatham  

**Link**: [PDF](https://arxiv.org/pdf/2510.04999)  

**Abstract**: Text-to-video (T2V) generation technology holds potential to transform multiple domains such as education, marketing, entertainment, and assistive technologies for individuals with visual or reading comprehension challenges, by creating coherent visual content from natural language prompts. From its inception, the field has advanced from adversarial models to diffusion-based models, yielding higher-fidelity, temporally consistent outputs. Yet challenges persist, such as alignment, long-range coherence, and computational efficiency. Addressing this evolving landscape, we present a comprehensive survey of text-to-video generative models, tracing their development from early GANs and VAEs to hybrid Diffusion-Transformer (DiT) architectures, detailing how these models work, what limitations they addressed in their predecessors, and why shifts toward new architectural paradigms were necessary to overcome challenges in quality, coherence, and control. We provide a systematic account of the datasets, which the surveyed text-to-video models were trained and evaluated on, and, to support reproducibility and assess the accessibility of training such models, we detail their training configurations, including their hardware specifications, GPU counts, batch sizes, learning rates, optimizers, epochs, and other key hyperparameters. Further, we outline the evaluation metrics commonly used for evaluating such models and present their performance across standard benchmarks, while also discussing the limitations of these metrics and the emerging shift toward more holistic, perception-aligned evaluation strategies. Finally, drawing from our analysis, we outline the current open challenges and propose a few promising future directions, laying out a perspective for future researchers to explore and build upon in advancing T2V research and applications. 

**Abstract (ZH)**: 文本到视频生成技术（T2V）有可能通过从自然语言提示生成连贯的视觉内容来改变教育、营销、娱乐和视觉或阅读理解障碍个体辅助技术等多个领域。从其起步至今，该领域已从对抗模型发展到基于扩散的模型，产生了更高保真度、时序一致的输出。然而，仍然存在对齐、长距离连贯性和计算效率等方面的挑战。为了应对这一不断发展的情景，我们提供了一篇全面的文本到视频生成模型综述，追溯了从早期的GANs和VAEs到混合扩散-变换器（DiT）架构的发展过程，详细说明了这些模型的工作原理、它们在前辈模型中解决的限制，以及为何转向新的架构范式是必要的，以克服质量、连贯性和控制方面的挑战。我们系统地概述了所研究的文本到视频模型的训练和评估数据集，并详细描述了它们的训练配置，包括硬件规格、GPU数量、批量大小、学习率、优化器、 epoch 和其他关键超参数，以支持再现性和评估训练此类模型的便利性。进一步地，我们概述了常用于评估此类模型的评价指标，并展示了它们在标准基准上的表现，同时也讨论了这些指标的局限性以及向更加整体、感知对齐的评价策略的新兴转变。最后，基于我们的分析，我们指出现有的开放挑战，并提出了一些建设性的未来方向，为未来的研究人员提供了探索和推动文本到视频研究和应用的视角。 

---
# ActiveMark: on watermarking of visual foundation models via massive activations 

**Title (ZH)**: ActiveMark：视觉基础模型的激活 watermarking 方法 

**Authors**: Anna Chistyakova, Mikhail Pautov  

**Link**: [PDF](https://arxiv.org/pdf/2510.04966)  

**Abstract**: Being trained on large and vast datasets, visual foundation models (VFMs) can be fine-tuned for diverse downstream tasks, achieving remarkable performance and efficiency in various computer vision applications. The high computation cost of data collection and training motivates the owners of some VFMs to distribute them alongside the license to protect their intellectual property rights. However, a dishonest user of the protected model's copy may illegally redistribute it, for example, to make a profit. As a consequence, the development of reliable ownership verification tools is of great importance today, since such methods can be used to differentiate between a redistributed copy of the protected model and an independent model. In this paper, we propose an approach to ownership verification of visual foundation models by fine-tuning a small set of expressive layers of a VFM along with a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. Importantly, the watermarks embedded remain detectable in the functional copies of the protected model, obtained, for example, by fine-tuning the VFM for a particular downstream task. Theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection of a non-watermarked model and a low probability of false misdetection of a watermarked model. 

**Abstract (ZH)**: 基于大规模数据集训练的视觉基础模型的所有权验证方法：通过微调一小组表达层和小型编码-解码网络将数字水印嵌入保留集输入图像的内部表示以实现模型所有权验证 

---
# Bidirectional Mammogram View Translation with Column-Aware and Implicit 3D Conditional Diffusion 

**Title (ZH)**: 基于列意识和隐式3D条件扩散的双向乳腺X光片视图翻译 

**Authors**: Xin Li, Kaixiang Yang, Qiang Li, Zhiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04947)  

**Abstract**: Dual-view mammography, including craniocaudal (CC) and mediolateral oblique (MLO) projections, offers complementary anatomical views crucial for breast cancer diagnosis. However, in real-world clinical workflows, one view may be missing, corrupted, or degraded due to acquisition errors or compression artifacts, limiting the effectiveness of downstream analysis. View-to-view translation can help recover missing views and improve lesion alignment. Unlike natural images, this task in mammography is highly challenging due to large non-rigid deformations and severe tissue overlap in X-ray projections, which obscure pixel-level correspondences. In this paper, we propose Column-Aware and Implicit 3D Diffusion (CA3D-Diff), a novel bidirectional mammogram view translation framework based on conditional diffusion model. To address cross-view structural misalignment, we first design a column-aware cross-attention mechanism that leverages the geometric property that anatomically corresponding regions tend to lie in similar column positions across views. A Gaussian-decayed bias is applied to emphasize local column-wise correlations while suppressing distant mismatches. Furthermore, we introduce an implicit 3D structure reconstruction module that back-projects noisy 2D latents into a coarse 3D feature volume based on breast-view projection geometry. The reconstructed 3D structure is refined and injected into the denoising UNet to guide cross-view generation with enhanced anatomical awareness. Extensive experiments demonstrate that CA3D-Diff achieves superior performance in bidirectional tasks, outperforming state-of-the-art methods in visual fidelity and structural consistency. Furthermore, the synthesized views effectively improve single-view malignancy classification in screening settings, demonstrating the practical value of our method in real-world diagnostics. 

**Abstract (ZH)**: 双视角乳腺X线摄影，包括头脚位（CC）和腋中位（MLO）投照，提供了乳腺癌诊断中互补的解剖视图。然而，在实际临床工作流程中，一个视图可能缺失、损坏或由于获取错误或压缩伪影而退化，限制了下游分析的有效性。视角间翻译可以帮助恢复缺失的视图并改善病灶对齐。与自然图像不同，乳腺X线摄影中的这一任务由于射线投影中巨大的非刚性变形和严重的组织重叠而极具挑战性，这模糊了像素级对应关系。在本文中，我们提出了一种基于条件扩散模型的新型双向乳腺X线摄影视图翻译框架，名为Column-Aware和Implicit 3D Diffusion（CA3D-Diff）。为了解决视角间结构对齐问题，我们首先设计了一种柱体意识跨注意力机制，利用解剖对应区域在不同视角中倾向于位于相似柱体位置的几何特性。应用高斯衰减偏置以强调局部柱体间相关性的同时抑制远距离不匹配。此外，我们引入了一种隐式3D结构重建模块，根据乳腺视图投影几何将嘈杂的2D潜在特征后投影到粗糙的3D特征体中。重建的3D结构经过细化并注入去噪UNet，以增强解剖意识指导视图间生成。大量实验表明，CA3D-Diff在双向任务中表现出优越性能，其视觉保真度和结构一致性均优于现有方法。此外，合成的视图有效提高了筛查场景下单视图恶性程度分类，展示了我们方法在实际临床诊断中的实用价值。 

---
# Did you just see that? Arbitrary view synthesis for egocentric replay of operating room workflows from ambient sensors 

**Title (ZH)**: 你刚刚看到的？基于环境传感器的主观回放合成手术室工作流程的任意视角 

**Authors**: Han Zhang, Lalithkumar Seenivasan, Jose L. Porras, Roger D. Soberanis-Mukul, Hao Ding, Hongchao Shu, Benjamin D. Killeen, Ankita Ghosh, Lonny Yarmus, Masaru Ishii, Angela Christine Argento, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2510.04802)  

**Abstract**: Observing surgical practice has historically relied on fixed vantage points or recollections, leaving the egocentric visual perspectives that guide clinical decisions undocumented. Fixed-camera video can capture surgical workflows at the room-scale, but cannot reconstruct what each team member actually saw. Thus, these videos only provide limited insights into how decisions that affect surgical safety, training, and workflow optimization are made. Here we introduce EgoSurg, the first framework to reconstruct the dynamic, egocentric replays for any operating room (OR) staff directly from wall-mounted fixed-camera video, and thus, without intervention to clinical workflow. EgoSurg couples geometry-driven neural rendering with diffusion-based view enhancement, enabling high-visual fidelity synthesis of arbitrary and egocentric viewpoints at any moment. In evaluation across multi-site surgical cases and controlled studies, EgoSurg reconstructs person-specific visual fields and arbitrary viewpoints with high visual quality and fidelity. By transforming existing OR camera infrastructure into a navigable dynamic 3D record, EgoSurg establishes a new foundation for immersive surgical data science, enabling surgical practice to be visualized, experienced, and analyzed from every angle. 

**Abstract (ZH)**: 手术实践观察历来依赖于固定视角或回忆，未能记录下指导临床决策的自我中心视觉视角。固定摄像头视频可以捕捉整个手术室的手术工作流程，但无法重建每位团队成员实际看到的内容。因此，这些视频只能提供有限的关于如何做出影响手术安全、培训和工作流程优化的决策的见解。我们介绍了EgoSurg，这是首个可以直接从壁挂固定摄像头视频中重构任何手术室（OR）人员的动态自我中心回放的框架，无需干预临床工作流程。EgoSurg 结合几何驱动的神经渲染与基于扩散的视点增强，能够在任何时刻合成立方视和自我中心视角的高质量视觉合成。在多中心手术案例和控制研究中的评估表明，EgoSurg 以高视觉质量和真实性重构了个体化的视觉视野和任意视角。通过将现有的OR摄像头基础设施转化为可导航的动态3D记录，EgoSurg 为沉浸式手术数据科学奠定了新的基础，使手术实践可以从各个角度进行可视化、体验和分析。 

---
# DiT-VTON: Diffusion Transformer Framework for Unified Multi-Category Virtual Try-On and Virtual Try-All with Integrated Image Editing 

**Title (ZH)**: DiffT-VTON: 扩散变压器框架下的统一多类别虚拟试穿与虚拟全试穿整合图像编辑 

**Authors**: Qi Li, Shuwen Qiu, Julien Han, Xingzi Xu, Mehmet Saygin Seyfioglu, Kee Kiat Koo, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2510.04797)  

**Abstract**: The rapid growth of e-commerce has intensified the demand for Virtual Try-On (VTO) technologies, enabling customers to realistically visualize products overlaid on their own images. Despite recent advances, existing VTO models face challenges with fine-grained detail preservation, robustness to real-world imagery, efficient sampling, image editing capabilities, and generalization across diverse product categories. In this paper, we present DiT-VTON, a novel VTO framework that leverages a Diffusion Transformer (DiT), renowned for its performance on text-conditioned image generation, adapted here for the image-conditioned VTO task. We systematically explore multiple DiT configurations, including in-context token concatenation, channel concatenation, and ControlNet integration, to determine the best setup for VTO image conditioning.
To enhance robustness, we train the model on an expanded dataset encompassing varied backgrounds, unstructured references, and non-garment categories, demonstrating the benefits of data scaling for VTO adaptability. DiT-VTON also redefines the VTO task beyond garment try-on, offering a versatile Virtual Try-All (VTA) solution capable of handling a wide range of product categories and supporting advanced image editing functionalities such as pose preservation, localized editing, texture transfer, and object-level customization. Experimental results show that our model surpasses state-of-the-art methods on VITON-HD, achieving superior detail preservation and robustness without reliance on additional condition encoders. It also outperforms models with VTA and image editing capabilities on a diverse dataset spanning thousands of product categories. 

**Abstract (ZH)**: 快速发展的电子商务加剧了对虚拟试穿（VTO）技术的需求，使顾客能够真实地将产品叠加在自己的图像上进行可视化。尽管取得了进展，现有的VTO模型在细节保留、对真实世界图像的鲁棒性、高效采样、图像编辑能力和跨不同产品类别的泛化能力方面仍面临挑战。本文提出了一种新的VTO框架DiT-VTON，该框架利用了性能出色的扩散变压器（DiT），并通过适应性的修改用于图像条件下的VTO任务。我们系统地探索了多种DiT配置，包括上下文标记拼接、通道拼接和ControlNet集成，以确定最适合VTO图像条件的最佳设置。

为了增强鲁棒性，我们在包含多种背景、非结构化参考和非服饰类别的扩展数据集上对模型进行训练，展示了数据规模扩展对VTO适应性的益处。DiT-VTON还超越了传统的VTO任务，提供了一个多功能的虚拟全试穿（VTA）解决方案，能够处理广泛的品类，并支持多种高级图像编辑功能，如姿态保留、局部编辑、纹理转移和对象级别自定义。实验结果表明，我们的模型在VITON-HD上超越了最先进的方法，在细节保留和鲁棒性方面表现出色，无需依赖额外的条件编码器。它还在包含数千个品类的多样化数据集上，优于具有VTA和图像编辑功能的其他模型。 

---
# Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction 

**Title (ZH)**: 具有各向异性意识采样的分阶高斯变换器在开放词汇占用预测中的应用 

**Authors**: Chi Yan, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04759)  

**Abstract**: The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: this https URL 

**Abstract (ZH)**: 基于3D占据预测任务在近年来取得了显著进展，在基于视觉的自动驾驶系统中发挥着重要作用。虽然传统方法局限于固定语义类别，近期的方法转向预测与文本对齐的特征，以实现开放词汇的文本查询。然而，文本对齐场景建模存在权衡：稀疏高斯表示难以捕捉场景中的小对象，而密集表示则会带来显著的计算开销。为了解决这些局限性，我们提出了PG-Occ，一种创新的渐进高斯变换框架，以实现开放词汇的3D占据预测。该框架采用渐进在线稠密化策略，逐步增强3D高斯表示以捕捉细粒度的场景细节。通过逐步增强表示，框架实现了逐步精确和详细的场景理解。另一个重要贡献是引入了一种具有时空融合的各向异性感知采样策略，该策略在不同尺度和阶段自适应地分配接收场给高斯，从而实现更有效的特征聚合和更丰富的场景信息捕获。通过广泛的评估，我们证明PG-Occ在相对mIoU上比之前最佳方法提高了14.3%，并在项目页面上发布代码和预训练模型：this https URL。 

---
# Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba 

**Title (ZH)**: 言说、编辑、重复：具有跨注意机制的高保真语音编辑与零样本TTS 

**Authors**: Baher Mohammad, Magauiya Zhussip, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2510.04738)  

**Abstract**: We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE - edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE - demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with VoiceCraft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ~6x less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention. 

**Abstract (ZH)**: MAVE：基于交叉注意力的Mamba声源编辑与高保真文本到语音合成新颖自回归架构 

---
# SFANet: Spatial-Frequency Attention Network for Deepfake Detection 

**Title (ZH)**: SFANet：空间-频率注意力网络在虚假视频检测中的应用 

**Authors**: Vrushank Ahire, Aniruddh Muley, Shivam Zample, Siddharth Verma, Pranav Menon, Surbhi Madan, Abhinav Dhall  

**Link**: [PDF](https://arxiv.org/pdf/2510.04630)  

**Abstract**: Detecting manipulated media has now become a pressing issue with the recent rise of deepfakes. Most existing approaches fail to generalize across diverse datasets and generation techniques. We thus propose a novel ensemble framework, combining the strengths of transformer-based architectures, such as Swin Transformers and ViTs, and texture-based methods, to achieve better detection accuracy and robustness. Our method introduces innovative data-splitting, sequential training, frequency splitting, patch-based attention, and face segmentation techniques to handle dataset imbalances, enhance high-impact regions (e.g., eyes and mouth), and improve generalization. Our model achieves state-of-the-art performance when tested on the DFWild-Cup dataset, a diverse subset of eight deepfake datasets. The ensemble benefits from the complementarity of these approaches, with transformers excelling in global feature extraction and texturebased methods providing interpretability. This work demonstrates that hybrid models can effectively address the evolving challenges of deepfake detection, offering a robust solution for real-world applications. 

**Abstract (ZH)**: 检测操纵媒体已成为一个紧迫的问题，尤其是在深度假信息的兴起之后。现有大多数方法无法在多种数据集和生成技术之间进行泛化。因此，我们提出了一种新的集成框架，结合了基于变压器的架构（如Swin Transformer和ViT）和纹理基方法的优势，以实现更好的检测准确性和鲁棒性。该方法引入了创新的数据分割、序列训练、频率分割、基于补丁的关注以及面部分割技术，以处理数据集不平衡、增强高影响区域（如眼睛和嘴巴）并提高泛化能力。当在DFWild-Cup数据集中测试时，我们的模型达到了最先进的性能，DFWild-Cup是一个来自八种深度假信息数据集的多样子集。该集成框架得益于这些方法之间的互补性，其中变压器在全局特征提取方面表现出色，而纹理基方法提供了可解释性。本工作证明，混合模型可以有效应对深度假信息检测的不断演变的挑战，并为实际应用提供了稳健的解决方案。 

---
# SONA: Learning Conditional, Unconditional, and Mismatching-Aware Discriminator 

**Title (ZH)**: SONA：学习条件依赖、无条件以及 mismating 意识辨别器 

**Authors**: Yuhta Takida, Satoshi Hayakawa, Takashi Shibuya, Masaaki Imaizumi, Naoki Murata, Bac Nguyen, Toshimitsu Uesaka, Chieh-Hsin Lai, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.04576)  

**Abstract**: Deep generative models have made significant advances in generating complex content, yet conditional generation remains a fundamental challenge. Existing conditional generative adversarial networks often struggle to balance the dual objectives of assessing authenticity and conditional alignment of input samples within their conditional discriminators. To address this, we propose a novel discriminator design that integrates three key capabilities: unconditional discrimination, matching-aware supervision to enhance alignment sensitivity, and adaptive weighting to dynamically balance all objectives. Specifically, we introduce Sum of Naturalness and Alignment (SONA), which employs separate projections for naturalness (authenticity) and alignment in the final layer with an inductive bias, supported by dedicated objective functions and an adaptive weighting mechanism. Extensive experiments on class-conditional generation tasks show that \ours achieves superior sample quality and conditional alignment compared to state-of-the-art methods. Furthermore, we demonstrate its effectiveness in text-to-image generation, confirming the versatility and robustness of our approach. 

**Abstract (ZH)**: 深度生成模型在生成复杂内容方面取得了显著进展，但条件生成仍然是一个基本挑战。现有的条件生成对抗网络往往难以在判别真实性和条件对齐之间找到平衡。为解决这一问题，我们提出了一种新颖的判别器设计，该设计整合了三项关键能力：无条件判别、增强对齐敏感性的匹配感知监督以及自适应加权以动态平衡所有目标。具体而言，我们引入了自然性和对齐之和（SONA），这是一种在最终层使用独立投影来区分自然性和对齐的方法，并带有专用的目标函数和自适应加权机制。在各类条件生成任务的广泛实验中，我们的方法在样本质量和条件对齐方面优于现有最先进的方法。此外，我们展示了其在文本转图像生成任务中的有效性，证实了该方法的 versatility 和 robustness。 

---
# MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models 

**Title (ZH)**: MedCLM: 在医疗视觉语言模型中通过CoT课程学习定位与推理 

**Authors**: Soo Yong Kim, Suin Cho, Vincent-Daniel Yun, Gyeongyeon Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04477)  

**Abstract**: Bridging clinical diagnostic reasoning with AI remains a central challenge in medical imaging. We introduce MedCLM, an automated pipeline that converts detection datasets into large-scale medical visual question answering (VQA) data with Chain-of-Thought (CoT) reasoning by linking lesion boxes to organ segmentation and structured rationales. These contextual signals enable medical vision-language models to generate question-answer pairs with step-by-step reasoning. To utilize this data effectively, we propose an Integrated CoT-Curriculum Strategy composed of an Easy stage with explicit lesion boxes for visual grounding, a Medium stage that encourages implicit localization, and a Hard stage for weakly supervised reasoning. Experimental results demonstrate that MedCLM attains state-of-the-art performance on several medical VQA benchmarks, providing a scalable framework for developing clinically aligned medical vision-language models. 

**Abstract (ZH)**: 将临床诊断推理与AI相结合仍然是医学影像领域的一个核心挑战。我们介绍了MedCLM，这是一种自动化流水线，通过将病灶框与器官分割和结构化理由关联，将检测数据集转换为大规模的医学视觉问答(VQA)数据，并通过Chain-of-Thought (CoT)推理。这些上下文信号使医学视觉-语言模型能够生成带有逐步推理的问题-答案对。为了有效利用这些数据，我们提出了一种综合的CoT-curriculum策略，包括一个简单阶段，用于明确定义病灶框以进行视觉定位，一个鼓励隐式定位的中等阶段，以及一个弱监督推理的困难阶段。实验结果表明，MedCLM在多个医学VQA基准测试中取得了最先进的性能，提供了一个可扩展的框架，用于开发与临床标准对齐的医学视觉-语言模型。 

---
# SPEGNet: Synergistic Perception-Guided Network for Camouflaged Object Detection 

**Title (ZH)**: SPEGNet: 协同感知引导网络在伪装目标检测中的应用 

**Authors**: Baber Jan, Saeed Anwar, Aiman H. El-Maleh, Abdul Jabbar Siddiqui, Abdul Bais  

**Link**: [PDF](https://arxiv.org/pdf/2510.04472)  

**Abstract**: Camouflaged object detection segments objects with intrinsic similarity and edge disruption. Current detection methods rely on accumulated complex components. Each approach adds components such as boundary modules, attention mechanisms, and multi-scale processors independently. This accumulation creates a computational burden without proportional gains. To manage this complexity, they process at reduced resolutions, eliminating fine details essential for camouflage. We present SPEGNet, addressing fragmentation through a unified design. The architecture integrates multi-scale features via channel calibration and spatial enhancement. Boundaries emerge directly from context-rich representations, maintaining semantic-spatial alignment. Progressive refinement implements scale-adaptive edge modulation with peak influence at intermediate resolutions. This design strikes a balance between boundary precision and regional consistency. SPEGNet achieves 0.887 $S_\alpha$ on CAMO, 0.890 on COD10K, and 0.895 on NC4K, with real-time inference speed. Our approach excels across scales, from tiny, intricate objects to large, pattern-similar ones, while handling occlusion and ambiguous boundaries. Code, model weights, and results are available on \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 伪装目标检测通过内在相似性和边缘干扰分割目标。当前的检测方法依赖于累积复杂组件。每种方法独立添加边界模块、注意力机制和多尺度处理器等组件。这种累积带来了计算负担，但没有相应的成效提升。为应对这种复杂性，它们在降低分辨率下进行处理，从而消除对伪装至关重要的细节数。我们提出SPEGNet，通过统一设计解决碎片化问题。该架构通过通道校准和空间增强整合多尺度特征。边缘直接源自丰富语境的表示，保持语义-空间对齐。渐进精炼实施尺度自适应边缘调制，在中间分辨率处达到峰值影响。此设计在边界精度和区域一致性之间达到平衡。SPEGNet在CAMO上取得0.887的$S_\alpha$，在COD10K上取得0.890，在NC4K上取得0.895，同时支持实时推理速度。我们的方法在不同尺度上表现出色，从精细的小目标到相似的大目标，同时处理遮挡和含糊的边界。代码、模型权重和结果可在<此链接>获得。 

---
# Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting 

**Title (ZH)**: 你的视觉语言模型连20都数不到：揭示VLMs在组合计数中的失败 

**Authors**: Xuyang Guo, Zekai Huang, Zhenmei Shi, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04401)  

**Abstract**: Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research. 

**Abstract (ZH)**: Vision-Language模型(VLMs)的能力源于大规模网络数据的训练，在当今的人工智能社区中已成为核心研究焦点。尽管这些模型在图像理解、视频理解、复杂视觉推理和具身人工智能等多样任务中表现出色，但一个基础问题仍然存在：VLMs能否准确计数物体？在本文中，我们提出了一个简单有效的基准VLMCountBench，在仅包含基本几何形状及其组合的简单设置下专注于计数任务，不受到其他因素的干扰。我们采用严格的独立变量控制，并在受控消融实验中系统研究了颜色、大小和提示优化等简单属性的影响。实验结果表明，当只有一种形状类型时，VLMs能够可靠地计数，但在多种形状类型组合（即组合计数）的情况下表现出显著的失败，这揭示了当前VLMs的一个基本经验限制，并为未来的研究指明了重要方向。 

---
# Pitch-Conditioned Instrument Sound Synthesis From an Interactive Timbre Latent Space 

**Title (ZH)**: 基于互动音色潜在空间的条件化音调乐器声音合成 

**Authors**: Christian Limberg, Fares Schulz, Zhe Zhang, Stefan Weinzierl  

**Link**: [PDF](https://arxiv.org/pdf/2510.04339)  

**Abstract**: This paper presents a novel approach to neural instrument sound synthesis using a two-stage semi-supervised learning framework capable of generating pitch-accurate, high-quality music samples from an expressive timbre latent space. Existing approaches that achieve sufficient quality for music production often rely on high-dimensional latent representations that are difficult to navigate and provide unintuitive user experiences. We address this limitation through a two-stage training paradigm: first, we train a pitch-timbre disentangled 2D representation of audio samples using a Variational Autoencoder; second, we use this representation as conditioning input for a Transformer-based generative model. The learned 2D latent space serves as an intuitive interface for navigating and exploring the sound landscape. We demonstrate that the proposed method effectively learns a disentangled timbre space, enabling expressive and controllable audio generation with reliable pitch conditioning. Experimental results show the model's ability to capture subtle variations in timbre while maintaining a high degree of pitch accuracy. The usability of our method is demonstrated in an interactive web application, highlighting its potential as a step towards future music production environments that are both intuitive and creatively empowering: this https URL 

**Abstract (ZH)**: 本文提出了一种新的神经乐器声音合成方法，该方法采用两阶段半监督学习框架，能够从表达性音色 Latent 空间生成准确音高和高质量的音乐样本。现有能够达到足够音乐生产质量的方法往往依赖于难以导航的高维 Latent 表示，提供不直观的用户体验。我们通过两阶段训练范式解决这一限制：首先，使用变分自编码器训练音高-音色分离的 2D 表示；其次，将此表示用于 Transformer 基础生成模型的条件输入。学习到的 2D Latent 空间作为直观的界面，用于导航和探索声音景观。实验结果表明，所提出的方法有效地学习了分离的音色空间，能够实现具有可靠音高条件的表达性和可控性音频生成。用户界面演示了该方法在交互式 Web 应用中的实用性，突显了其作为未来直观且创意上赋能的音乐制作环境的潜力：请点击此处。 

---
# Physics-Inspired All-Pair Interaction Learning for 3D Dynamics Modeling 

**Title (ZH)**: 基于物理启发的全对交互学习三维动力学建模 

**Authors**: Kai Yang, Yuqi Huang, Junheng Tao, Wanyu Wang, Qitian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04233)  

**Abstract**: Modeling 3D dynamics is a fundamental problem in multi-body systems across scientific and engineering domains and has important practical implications in trajectory prediction and simulation. While recent GNN-based approaches have achieved strong performance by enforcing geometric symmetries, encoding high-order features or incorporating neural-ODE mechanics, they typically depend on explicitly observed structures and inherently fail to capture the unobserved interactions that are crucial to complex physical behaviors and dynamics mechanism. In this paper, we propose PAINET, a principled SE(3)-equivariant neural architecture for learning all-pair interactions in multi-body systems. The model comprises: (1) a novel physics-inspired attention network derived from the minimization trajectory of an energy function, and (2) a parallel decoder that preserves equivariance while enabling efficient inference. Empirical results on diverse real-world benchmarks, including human motion capture, molecular dynamics, and large-scale protein simulations, show that PAINET consistently outperforms recently proposed models, yielding 4.7% to 41.5% error reductions in 3D dynamics prediction with comparable computation costs in terms of time and memory. 

**Abstract (ZH)**: 基于SE(3)守恒的物理启发式注意力网络：用于多体系统全对关系学习的原理性架构 

---
# Zoom-In to Sort AI-Generated Images Out 

**Title (ZH)**: 聚焦筛选AI生成的图像 

**Authors**: Yikun Ji, Yan Hong, Bowen Deng, jun lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04225)  

**Abstract**: The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising critical concerns for digital integrity. Vision-language models (VLMs) offer interpretability through explanations but often fail to detect subtle artifacts in high-quality synthetic images. We propose ZoomIn, a two-stage forensic framework that improves both accuracy and interpretability. Mimicking human visual inspection, ZoomIn first scans an image to locate suspicious regions and then performs a focused analysis on these zoomed-in areas to deliver a grounded verdict. To support training, we introduce MagniFake, a dataset of 20,000 real and high-quality synthetic images annotated with bounding boxes and forensic explanations, generated through an automated VLM-based pipeline. Our method achieves 96.39% accuracy with robust generalization, while providing human-understandable explanations grounded in visual evidence. 

**Abstract (ZH)**: AI生成图像的迅猛增长模糊了现实与合成内容的边界，对数字完整性提出了关键性 concern。Vision-Language 模型 (VLMs) 通过解释提供了可解释性，但往往难以检测高质量合成图像中的细微伪影。我们提出 ZoomIn，一种两阶段法医框架，以提高准确性和可解释性。模仿人类视觉检查，ZoomIn 首先扫描图像以定位可疑区域，然后在这些放大区域上进行聚焦分析以提供合规性的裁决。为支持训练，我们引入 MagniFake 数据集，包含 20,000 张标记有边界框和法医解释的真实和高质量合成图像，生成通过自动 VLM 基础管线完成。我们的方法实现了 96.39% 的准确率，并具备稳健的一般化能力，同时提供基于视觉证据的人类可理解解释。 

---
# MASC: Boosting Autoregressive Image Generation with a Manifold-Aligned Semantic Clustering 

**Title (ZH)**: MASC: 基于流形对齐语义聚类的自回归图像生成增强 

**Authors**: Lixuan He, Shikang Zheng, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04220)  

**Abstract**: Autoregressive (AR) models have shown great promise in image generation, yet they face a fundamental inefficiency stemming from their core component: a vast, unstructured vocabulary of visual tokens. This conventional approach treats tokens as a flat vocabulary, disregarding the intrinsic structure of the token embedding space where proximity often correlates with semantic similarity. This oversight results in a highly complex prediction task, which hinders training efficiency and limits final generation quality. To resolve this, we propose Manifold-Aligned Semantic Clustering (MASC), a principled framework that constructs a hierarchical semantic tree directly from the codebook's intrinsic structure. MASC employs a novel geometry-aware distance metric and a density-driven agglomerative construction to model the underlying manifold of the token embeddings. By transforming the flat, high-dimensional prediction task into a structured, hierarchical one, MASC introduces a beneficial inductive bias that significantly simplifies the learning problem for the AR model. MASC is designed as a plug-and-play module, and our extensive experiments validate its effectiveness: it accelerates training by up to 57% and significantly improves generation quality, reducing the FID of LlamaGen-XL from 2.87 to 2.58. MASC elevates existing AR frameworks to be highly competitive with state-of-the-art methods, establishing that structuring the prediction space is as crucial as architectural innovation for scalable generative modeling. 

**Abstract (ZH)**: 基于流形对齐的语义聚类（MASC）在图像生成中的应用 

---
# Prompt-to-Prompt: Text-Based Image Editing Via Cross-Attention Mechanisms -- The Research of Hyperparameters and Novel Mechanisms to Enhance Existing Frameworks 

**Title (ZH)**: Prompt-to-Prompt：基于文本的图像编辑通过跨注意力机制——关于超参数研究及新型机制以增强现有框架的探索 

**Authors**: Linn Bieske, Carla Lorente  

**Link**: [PDF](https://arxiv.org/pdf/2510.04034)  

**Abstract**: Recent advances in image editing have shifted from manual pixel manipulation to employing deep learning methods like stable diffusion models, which now leverage cross-attention mechanisms for text-driven control. This transition has simplified the editing process but also introduced variability in results, such as inconsistent hair color changes. Our research aims to enhance the precision and reliability of prompt-to-prompt image editing frameworks by exploring and optimizing hyperparameters. We present a comprehensive study of the "word swap" method, develop an "attention re-weight method" for better adaptability, and propose the "CL P2P" framework to address existing limitations like cycle inconsistency. This work contributes to understanding and improving the interaction between hyperparameter settings and the architectural choices of neural network models, specifically their attention mechanisms, which significantly influence the composition and quality of the generated images. 

**Abstract (ZH)**: 近期图像编辑的进步已从手动像素操作转向使用如稳定扩散模型等深度学习方法，这些方法现在利用交叉注意力机制实现文本驱动的控制。这一转变简化了编辑过程，但也引入了结果的一致性问题，如不一致的头发颜色变化。我们的研究旨在通过探索和优化超参数来提升提示到提示的图像编辑框架的精确性和可靠性。我们对“词交换”方法进行了全面研究，开发了“注意力重新加权方法”以提高适应性，并提出了“CL P2P”框架以解决现有局限性，如循环一致性问题。这项工作有助于理解并改进超参数设置与神经网络模型架构选择之间的互动，特别是它们的注意力机制，这些机制显著影响生成图像的组成和质量。 

---
# AI-Assisted Pleural Effusion Volume Estimation from Contrast-Enhanced CT Images 

**Title (ZH)**: 基于对比增强CT图像的AI辅助胸腔积液体积估算 

**Authors**: Sanhita Basu, Tomas Fröding, Ali Teymur Kahraman, Dimitris Toumpanakis, Tobias Sjöblom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03856)  

**Abstract**: Background: Pleural Effusions (PE) is a common finding in many different clinical conditions, but accurately measuring their volume from CT scans is challenging. Purpose: To improve PE segmentation and quantification for enhanced clinical management, we have developed and trained a semi-supervised deep learning framework on contrast-enhanced CT volumes. Materials and Methods: This retrospective study collected CT Pulmonary Angiogram (CTPA) data from internal and external datasets. A subset of 100 cases was manually annotated for model training, while the remaining cases were used for testing and validation. A novel semi-supervised deep learning framework, Teacher-Teaching Assistant-Student (TTAS), was developed and used to enable efficient training in non-segmented examinations. Segmentation performance was compared to that of state-of-the-art models. Results: 100 patients (mean age, 72 years, 28 [standard deviation]; 55 men) were included in the study. The TTAS model demonstrated superior segmentation performance compared to state-of-the-art models, achieving a mean Dice score of 0.82 (95% CI, 0.79 - 0.84) versus 0.73 for nnU-Net (p < 0.0001, Student's T test). Additionally, TTAS exhibited a four-fold lower mean Absolute Volume Difference (AbVD) of 6.49 mL (95% CI, 4.80 - 8.20) compared to nnU-Net's AbVD of 23.16 mL (p < 0.0001). Conclusion: The developed TTAS framework offered superior PE segmentation, aiding accurate volume determination from CT scans. 

**Abstract (ZH)**: 背景：胸腔积液（PE）是多种临床条件下的一种常见发现，但从CT扫描中准确测量其体积颇具挑战性。目的：为改善PE的分割和量化，以增强临床管理，我们开发并训练了一个半监督深度学习框架，应用于对比增强CT体积数据。材料与方法：本回顾性研究从内部和外部数据集中收集了CT肺血管造影（CTPA）数据。100例病例被手动标注用于模型训练，其余病例用于测试和验证。开发并使用了一种新颖的半监督深度学习框架——教师-助教-学生（TTAS）模型，以实现非分割检查中的高效训练。与最先进的模型相比，分割性能进行了比较。结果：研究共包括100名患者（平均年龄72岁，标准差28岁；55名男性）。TTAS模型在分割性能方面优于最先进的模型，平均Dice分数为0.82（95％CI，0.79-0.84），而nnU-Net的平均Dice分数为0.73（t检验，p < 0.0001）。此外，TTAS模型的平均绝对体积差异（AbVD）为6.49 mL（95％CI，4.80-8.20），比nnU-Net的AbVD（23.16 mL）低四倍（p < 0.0001）。结论：开发的TTAS框架提供了更优的PE分割，有助于从CT扫描中准确确定体积。 

---
# Diverse Text-to-Image Generation via Contrastive Noise Optimization 

**Title (ZH)**: 通过对比噪声优化实现多样的文本到图像生成 

**Authors**: Byungjun Kim, Soobin Um, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.03813)  

**Abstract**: Text-to-image (T2I) diffusion models have demonstrated impressive performance in generating high-fidelity images, largely enabled by text-guided inference. However, this advantage often comes with a critical drawback: limited diversity, as outputs tend to collapse into similar modes under strong text guidance. Existing approaches typically optimize intermediate latents or text conditions during inference, but these methods deliver only modest gains or remain sensitive to hyperparameter tuning. In this work, we introduce Contrastive Noise Optimization, a simple yet effective method that addresses the diversity issue from a distinct perspective. Unlike prior techniques that adapt intermediate latents, our approach shapes the initial noise to promote diverse outputs. Specifically, we develop a contrastive loss defined in the Tweedie data space and optimize a batch of noise latents. Our contrastive optimization repels instances within the batch to maximize diversity while keeping them anchored to a reference sample to preserve fidelity. We further provide theoretical insights into the mechanism of this preprocessing to substantiate its effectiveness. Extensive experiments across multiple T2I backbones demonstrate that our approach achieves a superior quality-diversity Pareto frontier while remaining robust to hyperparameter choices. 

**Abstract (ZH)**: 基于文本到图像的发散模型：通过对比噪声优化提升多样性和保真度 

---
# ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs 

**Title (ZH)**: ReTiDe: 适用于FPGA的高效实时噪声消除运动图像处理 

**Authors**: Changhong Li, Clément Bled, Rosa Fernandez, Shreejith Shanker  

**Link**: [PDF](https://arxiv.org/pdf/2510.03812)  

**Abstract**: Denoising is a core operation in modern video pipelines. In codecs, in-loop filters suppress sensor noise and quantisation artefacts to improve rate-distortion performance; in cinema post-production, denoisers are used for restoration, grain management, and plate clean-up. However, state-of-the-art deep denoisers are computationally intensive and, at scale, are typically deployed on GPUs, incurring high power and cost for real-time, high-resolution streams. This paper presents Real-Time Denoise (ReTiDe), a hardware-accelerated denoising system that serves inference on data-centre Field Programmable Gate Arrays (FPGAs). A compact convolutional model is quantised (post-training quantisation plus quantisation-aware fine-tuning) to INT8 and compiled for AMD Deep Learning Processor Unit (DPU)-based FPGAs. A client-server integration offloads computation from the host CPU/GPU to a networked FPGA service, while remaining callable from existing workflows, e.g., NUKE, without disrupting artist tooling. On representative benchmarks, ReTiDe delivers 37.71$\times$ Giga Operations Per Second (GOPS) throughput and 5.29$\times$ higher energy efficiency than prior FPGA denoising accelerators, with negligible degradation in Peak Signal-to-Noise Ratio (PSNR)/Structural Similarity Index (SSIM). These results indicate that specialised accelerators can provide practical, scalable denoising for both encoding pipelines and post-production, reducing energy per frame without sacrificing quality or workflow compatibility. Code is available at this https URL. 

**Abstract (ZH)**: 实时去噪（ReTiDe）：一种加速的数据中心可编程门阵列去噪系统 

---
# Artery-Vein Segmentation from Fundus Images using Deep Learning 

**Title (ZH)**: 基金us图像中的动脉-静脉分割方法研究 

**Authors**: Sharan SK, Subin Sahayam, Umarani Jayaraman, Lakshmi Priya A  

**Link**: [PDF](https://arxiv.org/pdf/2510.03717)  

**Abstract**: Segmenting of clinically important retinal blood vessels into arteries and veins is a prerequisite for retinal vessel analysis. Such analysis can provide potential insights and bio-markers for identifying and diagnosing various retinal eye diseases. Alteration in the regularity and width of the retinal blood vessels can act as an indicator of the health of the vasculature system all over the body. It can help identify patients at high risk of developing vasculature diseases like stroke and myocardial infarction. Over the years, various Deep Learning architectures have been proposed to perform retinal vessel segmentation. Recently, attention mechanisms have been increasingly used in image segmentation tasks. The work proposes a new Deep Learning approach for artery-vein segmentation. The new approach is based on the Attention mechanism that is incorporated into the WNet Deep Learning model, and we call the model as Attention-WNet. The proposed approach has been tested on publicly available datasets such as HRF and DRIVE datasets. The proposed approach has outperformed other state-of-art models available in the literature. 

**Abstract (ZH)**: 临床重要视网膜血管的分割是视网膜血管分析的前提。such分析可以提供识别和诊断各种视网膜眼病的潜在洞察和生物标志物。视网膜血管的规律性和宽度的变化可以作为整体血管系统健康状况的指标，有助于识别高血压和心肌梗死等血管疾病高风险患者。多年来，各种深度学习架构被提出用于执行视网膜血管分割。最近，在图像分割任务中越来越多地使用注意力机制。本文提出了一种新的基于注意力机制的深度学习方法用于动脉-静脉分割。该新方法将注意力机制整合到了WNet深度学习模型中，我们称之为Attention-WNet。所提出的观点已经在HRF和DRIVE等公开可用的数据集上进行了测试，并在文献中的其他先进模型中表现出色。 

---
# Referring Expression Comprehension for Small Objects 

**Title (ZH)**: 小目标的引用表达理解 

**Authors**: Kanoko Goto, Takumi Hirose, Mahiro Ukai, Shuhei Kurita, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2510.03701)  

**Abstract**: Referring expression comprehension (REC) aims to localize the target object described by a natural language expression. Recent advances in vision-language learning have led to significant performance improvements in REC tasks. However, localizing extremely small objects remains a considerable challenge despite its importance in real-world applications such as autonomous driving. To address this issue, we introduce a novel dataset and method for REC targeting small objects. First, we present the small object REC (SOREC) dataset, which consists of 100,000 pairs of referring expressions and corresponding bounding boxes for small objects in driving scenarios. Second, we propose the progressive-iterative zooming adapter (PIZA), an adapter module for parameter-efficient fine-tuning that enables models to progressively zoom in and localize small objects. In a series of experiments, we apply PIZA to GroundingDINO and demonstrate a significant improvement in accuracy on the SOREC dataset. Our dataset, codes and pre-trained models are publicly available on the project page. 

**Abstract (ZH)**: 小目标参照表达理解（SOREC）旨在定位自然语言表达描述的小目标物体。尽管视觉-语言学习的进步已经在参照表达理解（REC）任务上取得了显著的性能提升，但在自动驾驶等实际应用场景中，定位极小目标依然是一项重大挑战。为解决这一问题，我们提出了一个针对小目标的新型数据集和方法。首先，我们展示了包含100,000个-driver场景中小目标的参照表达和对应边界框的SOREC数据集。其次，我们提出了渐进迭代放大型（PIZA），这是一种用于参数高效微调的适配器模块，使模型能够逐步放大并定位小目标。在一系列实验中，我们将PIZA应用于GroundingDINO，并在SOREC数据集上证明了显著的准确性提高。我们的数据集、代码和预训练模型在项目页面上公开可用。 

---
# Neon: Negative Extrapolation From Self-Training Improves Image Generation 

**Title (ZH)**: Neon: 自训练的负外推 improves 图像生成 

**Authors**: Sina Alemohammad, Zhangyang Wang, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2510.03597)  

**Abstract**: Scaling generative AI models is bottlenecked by the scarcity of high-quality training data. The ease of synthesizing from a generative model suggests using (unverified) synthetic data to augment a limited corpus of real data for the purpose of fine-tuning in the hope of improving performance. Unfortunately, however, the resulting positive feedback loop leads to model autophagy disorder (MAD, aka model collapse) that results in a rapid degradation in sample quality and/or diversity. In this paper, we introduce Neon (for Negative Extrapolation frOm self-traiNing), a new learning method that turns the degradation from self-training into a powerful signal for self-improvement. Given a base model, Neon first fine-tunes it on its own self-synthesized data but then, counterintuitively, reverses its gradient updates to extrapolate away from the degraded weights. We prove that Neon works because typical inference samplers that favor high-probability regions create a predictable anti-alignment between the synthetic and real data population gradients, which negative extrapolation corrects to better align the model with the true data distribution. Neon is remarkably easy to implement via a simple post-hoc merge that requires no new real data, works effectively with as few as 1k synthetic samples, and typically uses less than 1% additional training compute. We demonstrate Neon's universality across a range of architectures (diffusion, flow matching, autoregressive, and inductive moment matching models) and datasets (ImageNet, CIFAR-10, and FFHQ). In particular, on ImageNet 256x256, Neon elevates the xAR-L model to a new state-of-the-art FID of 1.02 with only 0.36% additional training compute. Code is available at this https URL 

**Abstract (ZH)**: 负外推从自训练中提升生成AI模型：Neon的新学习方法 

---
# A Hybrid Co-Finetuning Approach for Visual Bug Detection in Video Games 

**Title (ZH)**: 视频游戏中的视觉错误检测的混合共微调方法 

**Authors**: Faliu Yi, Sherif Abdelfattah, Wei Huang, Adrian Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.03591)  

**Abstract**: Manual identification of visual bugs in video games is a resource-intensive and costly process, often demanding specialized domain knowledge. While supervised visual bug detection models offer a promising solution, their reliance on extensive labeled datasets presents a significant challenge due to the infrequent occurrence of such bugs. To overcome this limitation, we propose a hybrid Co-FineTuning (CFT) method that effectively integrates both labeled and unlabeled data. Our approach leverages labeled samples from the target game and diverse co-domain games, additionally incorporating unlabeled data to enhance feature representation learning. This strategy maximizes the utility of all available data, substantially reducing the dependency on labeled examples from the specific target game. The developed framework demonstrates enhanced scalability and adaptability, facilitating efficient visual bug detection across various game titles. Our experimental results show the robustness of the proposed method for game visual bug detection, exhibiting superior performance compared to conventional baselines across multiple gaming environments. Furthermore, CFT maintains competitive performance even when trained with only 50% of the labeled data from the target game. 

**Abstract (ZH)**: 手动识别视频游戏中视觉错误是一个资源密集型和成本高昂的过程，通常需要专门的领域知识。虽然监督视觉错误检测模型提供了有前途的解决方案，但由于此类错误出现频率低，其依赖于海量标注数据集构成了一个重要的挑战。为克服这一限制，我们提出了一种有效的混合Co-FineTuning（CFT）方法，结合了标记和未标记的数据。我们的方法利用目标游戏及其不同领域游戏的标记样本，并进一步整合未标记数据以增强特征表示学习。该策略最大限度地利用了所有可用数据，显著减少了对目标游戏特定标记示例的依赖。所开发的框架展示了增强的可扩展性和适应性，促进了各种游戏标题中的高效视觉错误检测。实验结果表明，所提出的方法在游戏视觉错误检测方面具有鲁棒性，并在多个游戏环境中展现出优于传统基线方法的性能。此外，即使仅使用目标游戏标记数据的50%进行训练，CFT也能保持竞争力。 

---
# Platonic Transformers: A Solid Choice For Equivariance 

**Title (ZH)**: 柏拉图变换器：对于等变性而言是一个稳健的选择 

**Authors**: Mohammad Mohaiminul Islam, Rishabh Anand, David R. Wessels, Friso de Kruiff, Thijs P. Kuipers, Rex Ying, Clara I. Sánchez, Sharvaree Vadgama, Georg Bökman, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2510.03511)  

**Abstract**: While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost. 

**Abstract (ZH)**: 尽管变压器在广泛应用，但缺乏对科学和计算机视觉中常见的几何对称性的归纳偏置。现有的一些协变方法往往通过复杂且计算密集的设计来牺牲变压器的高效性和灵活性。我们引入了 platonic transformer 来解决这一权衡。通过将注意力定义为参考框架，基于platonic固体对称群，我们的方法诱导出一种原理上正确的权重共享方案。这使得模型能够在保持标准变压器的精确架构和计算成本的同时，联合具有连续平移和platonic对称的协变性。此外，我们证明这种注意力与动态群卷积形式上等价，揭示了模型学习自适应几何滤波器的能力，并允许实现一种高度可扩展且线性时间的卷积变体。在计算机视觉（CIFAR-10）、3D 点云（ScanObjectNN）和分子性质预测（QM9, OMol25）等多个基准测试中，platonic transformer 通过利用这些几何约束实现了竞争力的性能，而并无额外的成本。 

---
# DuPLUS: Dual-Prompt Vision-Language Framework for Universal Medical Image Segmentation and Prognosis 

**Title (ZH)**: DuPLUS: 双提示视觉-语言框架在通用医疗图像分割和预后中的应用 

**Authors**: Numan Saeed, Tausifa Jan Saleem, Fadillah Maani, Muhammad Ridzuan, Hu Wang, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2510.03483)  

**Abstract**: Deep learning for medical imaging is hampered by task-specific models that lack generalizability and prognostic capabilities, while existing 'universal' approaches suffer from simplistic conditioning and poor medical semantic understanding. To address these limitations, we introduce DuPLUS, a deep learning framework for efficient multi-modal medical image analysis. DuPLUS introduces a novel vision-language framework that leverages hierarchical semantic prompts for fine-grained control over the analysis task, a capability absent in prior universal models. To enable extensibility to other medical tasks, it includes a hierarchical, text-controlled architecture driven by a unique dual-prompt mechanism. For segmentation, DuPLUS is able to generalize across three imaging modalities, ten different anatomically various medical datasets, encompassing more than 30 organs and tumor types. It outperforms the state-of-the-art task specific and universal models on 8 out of 10 datasets. We demonstrate extensibility of its text-controlled architecture by seamless integration of electronic health record (EHR) data for prognosis prediction, and on a head and neck cancer dataset, DuPLUS achieved a Concordance Index (CI) of 0.69. Parameter-efficient fine-tuning enables rapid adaptation to new tasks and modalities from varying centers, establishing DuPLUS as a versatile and clinically relevant solution for medical image analysis. The code for this work is made available at: this https URL 

**Abstract (ZH)**: 深度学习在医学影像分析中的应用受限于缺乏泛化能力和预后能力的任务特定模型，而现有的“通用”方法则因简化的条件和较差的医学语义理解而受限。为了解决这些局限性，我们提出了DuPLUS，一种高效的多模态医学影像分析深度学习框架。DuPLUS引入了一种新颖的 vision-language 框架，利用层次语义提示进行细粒度的分析任务控制，这是之前通用模型所缺乏的能力。为了使模型扩展到其他医学任务，它采用了由独特双提示机制驱动的层次化、文本控制架构。对于分割任务，DuPLUS能够在三种成像模态和十个不同解剖学多样化的医学数据集中泛化，覆盖了30多种器官和肿瘤类型。在十个数据集中，DuPLUS在八个数据集上超过了最先进的任务特定和通用模型。我们通过无缝集成电子健康记录（EHR）数据来预测预后，展示了其文本控制架构的扩展性，在头颈部癌症数据集上，DuPLUS的一致性指数（CI）达到了0.69。高效的微调参数使DuPLUS能够快速适应来自不同中心的新任务和模态，确立了其作为医学影像分析中多功能且临床相关解决方案的地位。该工作的代码可在以下链接获取：this https URL。 

---
# Application of a Virtual Imaging Framework for Investigating a Deep Learning-Based Reconstruction Method for 3D Quantitative Photoacoustic Computed Tomography 

**Title (ZH)**: 基于虚拟成像框架的深度学习重建方法用于三维定量光声计算机断层成像的研究 

**Authors**: Refik Mert Cam, Seonyeong Park, Umberto Villa, Mark A. Anastasio  

**Link**: [PDF](https://arxiv.org/pdf/2510.03431)  

**Abstract**: Quantitative photoacoustic computed tomography (qPACT) is a promising imaging modality for estimating physiological parameters such as blood oxygen saturation. However, developing robust qPACT reconstruction methods remains challenging due to computational demands, modeling difficulties, and experimental uncertainties. Learning-based methods have been proposed to address these issues but remain largely unvalidated. Virtual imaging (VI) studies are essential for validating such methods early in development, before proceeding to less-controlled phantom or in vivo studies. Effective VI studies must employ ensembles of stochastically generated numerical phantoms that accurately reflect relevant anatomy and physiology. Yet, most prior VI studies for qPACT relied on overly simplified phantoms. In this work, a realistic VI testbed is employed for the first time to assess a representative 3D learning-based qPACT reconstruction method for breast imaging. The method is evaluated across subject variability and physical factors such as measurement noise and acoustic aberrations, offering insights into its strengths and limitations. 

**Abstract (ZH)**: 基于学习的三维定量光声计算机断层成像方法在乳房成像中的虚拟影像测试与评估 

---
# Inference-Time Search using Side Information for Diffusion-based Image Reconstruction 

**Title (ZH)**: 基于辅信息的推断时搜索图像重建 

**Authors**: Mahdi Farahbakhsh, Vishnu Teja Kunde, Dileep Kalathil, Krishna Narayanan, Jean-Francois Chamberland  

**Link**: [PDF](https://arxiv.org/pdf/2510.03352)  

**Abstract**: Diffusion models have emerged as powerful priors for solving inverse problems. However, existing approaches typically overlook side information that could significantly improve reconstruction quality, especially in severely ill-posed settings. In this work, we propose a novel inference-time search algorithm that guides the sampling process using the side information in a manner that balances exploration and exploitation. This enables more accurate and reliable reconstructions, providing an alternative to the gradient-based guidance that is prone to reward-hacking artifacts. Our approach can be seamlessly integrated into a wide range of existing diffusion-based image reconstruction pipelines. Through extensive experiments on a number of inverse problems, such as box inpainting, super-resolution, and various deblurring tasks including motion, Gaussian, nonlinear, and blind deblurring, we show that our approach consistently improves the qualitative and quantitative performance of diffusion-based image reconstruction algorithms. We also show the superior performance of our approach with respect to other baselines, including reward gradient-based guidance algorithms. The code is available at \href{this https URL}{this repository}. 

**Abstract (ZH)**: 基于侧信息的探索与利用平衡搜索算法在反问题中增强扩散模型的图像重建性能 

---
# Pool Me Wisely: On the Effect of Pooling in Transformer-Based Models 

**Title (ZH)**: 池化用得智慧些：关于基于变压器的模型中池化的效应 

**Authors**: Sofiane Ennadir, Levente Zólyomi, Oleg Smirnov, Tianze Wang, John Pertoft, Filip Cornell, Lele Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03339)  

**Abstract**: Transformer models have become the dominant backbone for sequence modeling, leveraging self-attention to produce contextualized token representations. These are typically aggregated into fixed-size vectors via pooling operations for downstream tasks. While much of the literature has focused on attention mechanisms, the role of pooling remains underexplored despite its critical impact on model behavior. In this paper, we introduce a theoretical framework that rigorously characterizes the expressivity of Transformer-based models equipped with widely used pooling methods by deriving closed-form bounds on their representational capacity and the ability to distinguish similar inputs. Our analysis extends to different variations of attention formulations, demonstrating that these bounds hold across diverse architectural variants. We empirically evaluate pooling strategies across tasks requiring both global and local contextual understanding, spanning three major modalities: computer vision, natural language processing, and time-series analysis. Results reveal consistent trends in how pooling choices affect accuracy, sensitivity, and optimization behavior. Our findings unify theoretical and empirical perspectives, providing practical guidance for selecting or designing pooling mechanisms suited to specific tasks. This work positions pooling as a key architectural component in Transformer models and lays the foundation for more principled model design beyond attention alone. 

**Abstract (ZH)**: 基于Transformer模型的聚合方法在不同任务中的理论分析与实证研究：统一表达能力和任务适应性 

---
# Photorealistic Inpainting for Perturbation-based Explanations in Ecological Monitoring 

**Title (ZH)**: 基于扰动的生态监测光orealistic修复解释 

**Authors**: Günel Aghakishiyeva, Jiayi Zhou, Saagar Arya, James David Poling, Holly R. Houliston, Jamie N. Womble, David W. Johnston, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2510.03317)  

**Abstract**: Ecological monitoring is increasingly automated by vision models, yet opaque predictions limit trust and field adoption. We present an inpainting-guided, perturbation-based explanation technique that produces photorealistic, mask-localized edits that preserve scene context. Unlike masking or blurring, these edits stay in-distribution and reveal which fine-grained morphological cues drive predictions in tasks such as species recognition and trait attribution. We demonstrate the approach on a YOLOv9 detector fine-tuned for harbor seal detection in Glacier Bay drone imagery, using Segment-Anything-Model-refined masks to support two interventions: (i) object removal/replacement (e.g., replacing seals with plausible ice/water or boats) and (ii) background replacement with original animals composited onto new scenes. Explanations are assessed by re-scoring perturbed images (flip rate, confidence drop) and by expert review for ecological plausibility and interpretability. The resulting explanations localize diagnostic structures, avoid deletion artifacts common to traditional perturbations, and yield domain-relevant insights that support expert validation and more trustworthy deployment of AI in ecology. 

**Abstract (ZH)**: 视觉模型驱动的生态监测日益 automation，然而不透明的预测限制了信任和现场应用。我们提出了一种 inpainting 引导的扰动基解释技术，它生成保场景上下文的、照片级真实的、掩码局部化的编辑。与掩码或模糊不同，这些编辑保持在分布内，并揭示了在物种识别和性状归因等任务中驱动预测的细粒度形态学线索。我们通过使用 Segment-Anything-Model 加工的掩码在 Glaciers Bay 彩鹰无人机图像上对 YOLOv9 检测器进行微调来演示该方法，支持两种干预措施：(i) 对象移除/替换（例如，用可能的冰/水或船只替换海豹），以及 (ii) 背景替换，将原始动物重新组合到新场景中。通过重新评分扰动图像（翻转率、置信度下降）和专家评审生态合理性与可解释性来评估解释。生成的解释定位诊断结构，避免了传统扰动常见的删除伪影，并提供了与领域相关的重要见解，支持专家验证和更具可信度的 AI 在生态学中的部署。 

---
# Creative synthesis of kinematic mechanisms 

**Title (ZH)**: 创造性合成运动学机构 

**Authors**: Jiong Lin, Jialong Ning, Judah Goldfeder, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03308)  

**Abstract**: In this paper, we formulate the problem of kinematic synthesis for planar linkages as a cross-domain image generation task. We develop a planar linkages dataset using RGB image representations, covering a range of mechanisms: from simple types such as crank-rocker and crank-slider to more complex eight-bar linkages like Jansen's mechanism. A shared-latent variational autoencoder (VAE) is employed to explore the potential of image generative models for synthesizing unseen motion curves and simulating novel kinematics. By encoding the drawing speed of trajectory points as color gradients, the same architecture also supports kinematic synthesis conditioned on both trajectory shape and velocity profiles. We validate our method on three datasets of increasing complexity: a standard four-bar linkage set, a mixed set of four-bar and crank-slider mechanisms, and a complex set including multi-loop mechanisms. Preliminary results demonstrate the effectiveness of image-based representations for generative mechanical design, showing that mechanisms with revolute and prismatic joints, and potentially cams and gears, can be represented and synthesized within a unified image generation framework. 

**Abstract (ZH)**: 本文将平面连杆机构的运动合成问题形式化为一个跨域图像生成任务。利用RGB图像表示构建平面连杆机构数据集，涵盖了从简单类型如曲柄摇杆和曲柄滑块机构到更复杂的八连杆机构如詹森机制等各类机制。采用共享潜在空间的变分自编码器（VAE），探索图像生成模型在合成未见过的运动曲线和模拟新型运动学方面的潜力。通过将轨迹点的绘制速度编码为颜色梯度，相同的架构还支持基于轨迹形状和速度分布的运动合成。我们在三个逐步复杂的数据集上验证了该方法：标准四连杆机构集、四连杆与曲柄滑块机构混合集，以及包含多环机构的复杂集。初步结果表明，基于图像的表示在生成机械设计中有效，展示了旋转铰链和平移铰链机制，以及可能的凸轮和齿轮机制，可以在统一的图像生成框架中表示和合成。 

---
# Convolutional Neural Nets vs Vision Transformers: A SpaceNet Case Study with Balanced vs Imbalanced Regimes 

**Title (ZH)**: 卷积神经网络vs视觉变换器：关于平衡状态与不平衡状态的SpaceNet案例研究 

**Authors**: Akshar Gothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03297)  

**Abstract**: We present a controlled comparison of a convolutional neural network (EfficientNet-B0) and a Vision Transformer (ViT-Base) on SpaceNet under two label-distribution regimes: a naturally imbalanced five-class split and a balanced-resampled split with 700 images per class (70:20:10 train/val/test). With matched preprocessing (224x224, ImageNet normalization), lightweight augmentations, and a 40-epoch budget on a single NVIDIA P100, we report accuracy, macro-F1, balanced accuracy, per-class recall, and deployment metrics (model size and latency). On the imbalanced split, EfficientNet-B0 reaches 93% test accuracy with strong macro-F1 and lower latency; ViT-Base is competitive at 93% with a larger parameter count and runtime. On the balanced split, both models are strong; EfficientNet-B0 reaches 99% while ViT-Base remains competitive, indicating that balancing narrows architecture gaps while CNNs retain an efficiency edge. We release manifests, logs, and per-image predictions to support reproducibility. 

**Abstract (ZH)**: 我们在SpaceNet上对卷积神经网络（EfficientNet-B0）和视觉变压器（ViT-Base）在两种标签分布制度下的表现进行了受控比较：自然不平衡的五分类划分和重采样的平衡划分（每类700张图像，70:20:10训练/验证/测试）。通过匹配的预处理（224x224，ImageNet归一化）、轻量级的数据增强并在单个NVIDIA P100上运行40个epoch，我们报告了准确率、宏F1值、均衡准确率、每类召回率以及部署指标（模型大小和延迟）。在不平衡划分中，EfficientNet-B0达到93%的测试准确率，宏F1值较强且延迟较低；ViT-Base在93%的准确率上具有竞争力，但参数量和运行时间更大。在平衡划分中，两种模型表现强劲；EfficientNet-B0达到99%，而ViT-Base保持竞争力，这表明平衡化缩小了架构差距而CNN在效率方面仍占优势。我们发布了元数据、日志和单张图像预测以支持可重复性。 

---
# QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks 

**Title (ZH)**: QuadEnhancer: 利用二次变换增强深度神经网络 

**Authors**: Qian Chen, Linxin Yang, Akang Wang, Xiaodong Luo, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03276)  

**Abstract**: The combination of linear transformations and non-linear activation functions forms the foundation of most modern deep neural networks, enabling them to approximate highly complex functions. This paper explores the introduction of quadratic transformations to further increase nonlinearity in neural networks, with the aim of enhancing the performance of existing architectures. To reduce parameter complexity and computational complexity, we propose a lightweight quadratic enhancer that uses low-rankness, weight sharing, and sparsification techniques. For a fixed architecture, the proposed approach introduces quadratic interactions between features at every layer, while only adding negligible amounts of additional model parameters and forward computations. We conduct a set of proof-of-concept experiments for the proposed method across three tasks: image classification, text classification, and fine-tuning large-language models. In all tasks, the proposed approach demonstrates clear and substantial performance gains. 

**Abstract (ZH)**: 线性变换与非线性激活函数的结合构成了大多数现代深度神经网络的基础，使它们能够逼近高度复杂的功能。本文探讨引入二次变换以进一步增加神经网络的非线性，旨在提升现有架构的表现。为减少参数复杂度和计算复杂度，我们提出了一种轻量级的二次增强器，利用低秩性、权重共享和稀疏化技术。对于固定架构，所提出的方法在每一层都引入了特征间的二次相互作用，同时仅增加了可忽略不计的额外模型参数和前向计算量。我们在图像分类、文本分类和大型语言模型微调三个任务中进行了所提出方法的概念验证实验。在所有任务中，所提出的方法均表现出明确且显著的性能提升。 

---
# Textured Gaussians for Enhanced 3D Scene Appearance Modeling 

**Title (ZH)**: 纹理高斯函数用于增强的3D场景外观建模 

**Authors**: Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2411.18625)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently emerged as a state-of-the-art 3D reconstruction and rendering technique due to its high-quality results and fast training and rendering time. However, pixels covered by the same Gaussian are always shaded in the same color up to a Gaussian falloff scaling factor. Furthermore, the finest geometric detail any individual Gaussian can represent is a simple ellipsoid. These properties of 3DGS greatly limit the expressivity of individual Gaussian primitives. To address these issues, we draw inspiration from texture and alpha mapping in traditional graphics and integrate it with 3DGS. Specifically, we propose a new generalized Gaussian appearance representation that augments each Gaussian with alpha~(A), RGB, or RGBA texture maps to model spatially varying color and opacity across the extent of each Gaussian. As such, each Gaussian can represent a richer set of texture patterns and geometric structures, instead of just a single color and ellipsoid as in naive Gaussian Splatting. Surprisingly, we found that the expressivity of Gaussians can be greatly improved by using alpha-only texture maps, and further augmenting Gaussians with RGB texture maps achieves the highest expressivity. We validate our method on a wide variety of standard benchmark datasets and our own custom captures at both the object and scene levels. We demonstrate image quality improvements over existing methods while using a similar or lower number of Gaussians. 

**Abstract (ZH)**: 基于3D高斯体的增强纹理映射技术：一种改进的3D重建与渲染方法 

---
# A Modular Conditional Diffusion Framework for Image Reconstruction 

**Title (ZH)**: 一种模块化的条件扩散框架用于图像重建 

**Authors**: Magauiya Zhussip, Iaroslav Koshelev, Stamatis Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2411.05993)  

**Abstract**: Diffusion Probabilistic Models (DPMs) have been recently utilized to deal with various blind image restoration (IR) tasks, where they have demonstrated outstanding performance in terms of perceptual quality. However, the task-specific nature of existing solutions and the excessive computational costs related to their training, make such models impractical and challenging to use for different IR tasks than those that were initially trained for. This hinders their wider adoption, especially by those who lack access to powerful computational resources and vast amount of training data. In this work we aim to address the above issues and enable the successful adoption of DPMs in practical IR-related applications. Towards this goal, we propose a modular diffusion probabilistic IR framework (DP-IR), which allows us to combine the performance benefits of existing pre-trained state-of-the-art IR networks and generative DPMs, while it requires only the additional training of a relatively small module (0.7M params) related to the particular IR task of interest. Moreover, the architecture of the proposed framework allows for a sampling strategy that leads to at least four times reduction of neural function evaluations without suffering any performance loss, while it can also be combined with existing acceleration techniques such as DDIM. We evaluate our model on four benchmarks for the tasks of burst JDD-SR, dynamic scene deblurring, and super-resolution. Our method outperforms existing approaches in terms of perceptual quality while it retains a competitive performance with respect to fidelity metrics. 

**Abstract (ZH)**: 基于扩散概率模型的盲图像恢复框架：模块化设计与高效应用 

---
